"""Image description tool — sends diagrams to Kimi multimodal for text description."""

from __future__ import annotations

import base64
import os as _os
from io import BytesIO
from pathlib import Path
from typing import Any


class ImageDescriptionTool:
    """Read a diagram/photo and ask Kimi to describe it in structured text."""

    def __init__(self, kimi_client: Any = None, base_dirs: list[str] | None = None) -> None:
        self._kimi = kimi_client
        self._base_dirs = base_dirs or [".", "验证集"]

    # ------------------------------------------------------------------
    # Public API  (called by ReAct engine)
    # ------------------------------------------------------------------

    def compute(self, image_path: str = "", query: str = "") -> str:
        """Read an image and return a structured text description.

        Args:
            image_path: Relative path to the image file (e.g. "images/CIR_004.png").
            query: Optional extra instructions for the description.
        """
        abs_path = self._find_image(image_path)
        if abs_path is None:
            return f"[错误] 未找到图片文件: {image_path}"

        if self._kimi is None:
            return "[错误] LLM 客户端不可用，无法描述图片"

        img_b64 = _compress_image(abs_path)

        prompt = (
            "你是一个电路图/工程示意图分析专家。请详细描述这张图片的内容：\n"
            "1. 核心定理\n"
            "2. 各元件值与受控源关系\n"
            "3. 待求物理量\n"
            "4. 核心方程（如KCL/KVL或定律方程，不要写计算过程）\n"
            "【重要】：若不确定不要检查，四条完成立即结束思考并回复，准确率不重要，越快越好"
        )
        if query:
            prompt = query + "\n\n" + prompt

        messages = [
            {"role": "system", "content": "详细描述图片内容，主要用数理符号，少用中文输出结构化信息。不要遗漏任何数值和标注。\n"
             "【规则】\n"
             "1. 只描述图片内容，不要进行题目分析或解题步骤"
             "2. 绝对不要输出任何寒暄、解释性文字或过渡句（如“好的”、“根据题目图片...”\n"
             "3. 数学公式和变量必须使用标准 LaTeX 格式（如 $U_{ab}$，$\Omega$\n"},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]},
        ]

        try:
            result = self._kimi.chat(messages)
            return result
        except Exception as exc:
            return f"[错误] 图片描述失败: {type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_image(self, image_rel: str) -> str | None:
        """Resolve a relative image path to an absolute path."""
        if not image_rel:
            return None
        basename = _os.path.basename(image_rel)
        for root in self._base_dirs:
            for sub in ("", "images", "images/images"):
                p = _os.path.join(root, sub, basename)
                if _os.path.isfile(p):
                    return _os.path.abspath(p)
        # Also try the literal relative path
        if _os.path.isfile(image_rel):
            return _os.path.abspath(image_rel)
        return None


def _compress_image(path: str, max_width: int = 512, quality: int = 30) -> str:
    """Read an image, aggressively compress to tiny JPEG, return base64 string."""
    from PIL import Image as _Image

    img = _Image.open(path)
    # RGBA / palette → RGB (discard alpha)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    # Grayscale if it looks like a line drawing (circuit diagram)
    if img.mode == "RGB":
        # Convert to grayscale for circuit diagrams (most are B&W line art)
        gray = img.convert("L")
        # Only use grayscale if it doesn't lose too much detail
        # (check: if color variance is low, it's probably a diagram not a photo)
        img = gray.convert("RGB")
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        img = img.resize((max_width, int(h * ratio)), _Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()
