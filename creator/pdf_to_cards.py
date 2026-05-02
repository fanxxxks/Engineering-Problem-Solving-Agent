"""Extract solved-example knowledge cards from PDF problem sets via Mimo.

Pipeline (parallel per PDF):
  1. Render PDF pages as images
  2. Send page batches to Mimo (multimodal) → extract question/answer/reasoning
  3. Checker Mimo call validates format & correctness
  4. Save each card to solved_examples.jsonl immediately

Usage:
    conda activate EPSA
    pip install pymupdf          # one-time
    python creator/pdf_to_cards.py
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

from openai import OpenAI

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PDF_DIR = SCRIPT_DIR / "original"
MIMO_CONFIG = SCRIPT_DIR / "mimo.txt"
OUTPUT_FILE = PROJECT_DIR / "eng_solver_agent" / "retrieval" / "solved_examples.jsonl"

PAGES_PER_BATCH = 4          # pages per Mimo call (multimodal)
MAX_WORKERS = 4              # parallel PDFs
JPEG_QUALITY = 50
MAX_IMAGE_WIDTH = 1024

_EXISTING_IDS: set[str] = set()
write_lock = Lock()
stats_lock = Lock()
_stats = {"extracted": 0, "failed": 0}


# ── Mimo client ───────────────────────────────────────────────────────────

def _load_mimo_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    for line in MIMO_CONFIG.read_text("utf-8").strip().split("\n"):
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("\"'")
        m = re.match(r'os\.environ\.get\(["\'](.+?)["\']\)', v)
        cfg[k] = m.group(1) if m else v
    return cfg


def _build_client() -> OpenAI:
    cfg = _load_mimo_config()
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=300.0, max_retries=2)


# ── PDF → images ──────────────────────────────────────────────────────────

def _render_pages(pdf_path: Path) -> list[bytes]:
    """Render each page of a PDF to a compressed JPEG byte string."""
    import fitz  # pymupdf

    pages: list[bytes] = []
    doc = fitz.open(str(pdf_path))
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at 150 DPI (good enough for reading math)
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = page.get_pixmap(matrix=mat)
        # Convert to PIL for compression
        from PIL import Image

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        w, h = img.size
        if w > MAX_IMAGE_WIDTH:
            img = img.resize((MAX_IMAGE_WIDTH, int(h * MAX_IMAGE_WIDTH / w)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        pages.append(buf.getvalue())
    doc.close()
    return pages


def _pages_to_b64(pages: list[bytes]) -> list[str]:
    """Convert JPEG bytes to base64 data URIs."""
    return [f"data:image/jpeg;base64,{base64.b64encode(p).decode()}" for p in pages]


# ── Mimo extraction ───────────────────────────────────────────────────────

_EXTRACT_SYSTEM = (
    "你是一位精通微积分的数学专家和题目提取器。\n"
    "从习题课PDF中提取每一道题目，转换为标准知识卡片格式。\n"
    "每道题必须包含：question（题目原文）、answer（最终答案）、\n"
    "reasoning_process（解题步骤，从已知条件→公式→代入→结论）。\n"
    "无论原PDF中是否有答案，你都必须为每道题生成完整的解答过程。\n"
    "只返回严格的 JSON 数组，不要任何解释文字。\n"
    '格式：[{"question_id": "...", "subject": "calculus", "topic": "...", '
    '"question": "...", "reasoning_process": "...", "answer": "...", '
    '"tags": ["..."]}]'
)

_EXTRACT_PROMPT = (
    "请从以下习题课页面中提取所有题目，并为每一道题生成完整的知识卡片。\n"
    "要求：\n"
    "1. question_id 用 PDF文件名缩写 + 序号，如 CALC_LIMIT_001\n"
    "2. subject 统一填 calculus\n"
    "3. topic 根据内容推断：derivative / integration / limit / taylor_series / series 等\n"
    "4. question 保留原题文字和公式（用 LaTeX 格式）\n"
    "5. reasoning_process 写完整解题步骤：已知条件→所用公式→代入→结论\n"
    "6. answer 只写最终答案\n"
    "7. tags 提取 2-4 个关键词\n"
    "8. 【重要】无论原页面中是否有解答，你都必须自己为每道题生成完整的 reasoning_process 和 answer。\n"
    "   如果原页面已有答案，可以参考；如果没有，请自己解题并写出完整步骤。\n\n"
    "返回纯 JSON 数组，不要 markdown 代码块。"
)


def _extract_cards(client: OpenAI, model: str, pages_b64: list[str], pdf_name: str) -> list[dict]:
    """Send page images to Mimo, get extracted knowledge cards."""
    # Build multimodal content: text + all page images
    content: list[dict] = [{"type": "text", "text": _EXTRACT_PROMPT}]
    for b64 in pages_b64:
        content.append({"type": "image_url", "image_url": {"url": b64}})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": content},
            ],
            temperature=1.0,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:
        print(f"    Extract API error: {exc}")
        return []

    # Parse JSON from response
    text = raw.strip()
    # Remove markdown fences
    for pat in [r"```json\s*\n?(.*?)\n?```", r"```\s*\n?(.*?)\n?```"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            text = m.group(1).strip()
            break
    # Find JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        cards = json.loads(text)
        if isinstance(cards, list):
            return cards
    except json.JSONDecodeError:
        pass
    return []


# ── Checker ────────────────────────────────────────────────────────────────

_CHECK_SYSTEM = (
    "你是一个知识卡片审核器。检查提取的例题卡片是否符合格式要求，"
    "修正任何问题后返回修正后的 JSON 数组。\n"
    "检查项：question_id 格式、subject 是否 calculus、topic 是否合理、"
    "reasoning_process 是否完整（已知→公式→代入→结论）、answer 是否正确。\n"
    "只返回修正后的 JSON 数组，不要解释。"
)


def _check_cards(client: OpenAI, model: str, cards: list[dict]) -> list[dict]:
    """Validate and fix extracted cards."""
    if not cards:
        return []
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CHECK_SYSTEM},
                {"role": "user", "content": json.dumps(cards, ensure_ascii=False, indent=2)},
            ],
            temperature=1.0,
        )
        raw = resp.choices[0].message.content or ""
    except Exception:
        return cards  # return original if check fails

    text = raw.strip()
    for pat in [r"```json\s*\n?(.*?)\n?```", r"```\s*\n?(.*?)\n?```"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            text = m.group(1).strip()
            break
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        checked = json.loads(text)
        if isinstance(checked, list):
            return checked
    except json.JSONDecodeError:
        pass
    return cards


# ── Save ───────────────────────────────────────────────────────────────────

def _save_cards(cards: list[dict]) -> int:
    """Write valid cards to JSONL. Returns count of newly saved."""
    saved = 0
    with write_lock:
        with OUTPUT_FILE.open("a", encoding="utf-8") as f:
            for card in cards:
                qid = card.get("question_id", "")
                if not qid or qid in _EXISTING_IDS:
                    continue
                required = ["subject", "question", "reasoning_process", "answer"]
                if not all(card.get(k) for k in required):
                    continue
                card.setdefault("topic", "general")
                card.setdefault("tags", [])
                f.write(json.dumps(card, ensure_ascii=False) + "\n")
                f.flush()
                _EXISTING_IDS.add(qid)
                saved += 1
    return saved


# ── Per-PDF pipeline ──────────────────────────────────────────────────────

def _process_pdf(pdf_path: Path, client: OpenAI, model: str) -> tuple[str, int, int]:
    """Full pipeline for one PDF: render → extract → check → save."""
    name = pdf_path.stem
    print(f"  [{name}] Rendering pages...")
    try:
        pages = _render_pages(pdf_path)
    except Exception as exc:
        print(f"  [{name}] Render FAILED: {exc}")
        return name, 0, len(pages) if "pages" in dir() else 0

    total_extracted = 0
    total_saved = 0

    # Process in batches
    for batch_idx in range(0, len(pages), PAGES_PER_BATCH):
        batch = pages[batch_idx : batch_idx + PAGES_PER_BATCH]
        batch_b64 = _pages_to_b64(batch)
        page_range = f"{batch_idx + 1}-{min(batch_idx + PAGES_PER_BATCH, len(pages))}"

        print(f"  [{name}] Pages {page_range}/{len(pages)} → extracting...")
        cards = _extract_cards(client, model, batch_b64, name)
        if not cards:
            continue

        print(f"  [{name}] Pages {page_range} → checking {len(cards)} cards...")
        checked = _check_cards(client, model, cards)

        saved = _save_cards(checked)
        total_extracted += len(checked)
        total_saved += saved
        print(f"  [{name}] Pages {page_range} → {len(checked)} cards, {saved} new")

    return name, total_saved, total_extracted


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in creator/original/")
        return 1

    print(f"Found {len(pdfs)} PDFs\n")

    # Load existing IDs
    if OUTPUT_FILE.exists():
        for line in OUTPUT_FILE.read_text("utf-8").strip().split("\n"):
            if not line.strip():
                continue
            try:
                _EXISTING_IDS.add(json.loads(line).get("question_id", ""))
            except json.JSONDecodeError:
                pass
    print(f"Existing cards: {len(_EXISTING_IDS)}\n")

    cfg = _load_mimo_config()
    model = cfg.get("model", "mimo-v2.5-pro")

    grand_saved = 0
    grand_failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for pdf in pdfs:
            client = _build_client()
            futures[executor.submit(_process_pdf, pdf, client, model)] = pdf

        for future in as_completed(futures):
            pdf = futures[future]
            try:
                name, saved, total = future.result()
                with stats_lock:
                    grand_saved += saved
                    if total == 0:
                        grand_failed += 1
                print(f"  DONE {name}: {saved} new cards saved")
            except Exception as exc:
                print(f"  FAILED {pdf.name}: {exc}")
                grand_failed += 1

    print(f"\n{'='*60}")
    print(f"All done: {grand_saved} new cards, {grand_failed} failed PDFs")
    print(f"Total knowledge base: {len(_EXISTING_IDS) + grand_saved} cards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
