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
            "1. 元器件类型和数量（电阻、电容、电感、电源等）\n"
            "2. 各元器件的参数值\n"
            "3. 连接关系/拓扑结构\n"
            "4. 图中标注的文字和数值\n"
        )
        if query:
            prompt = query + "\n\n" + prompt

        messages = [
            {"role": "system", "content": "请用中文详细描述图片内容，输出结构化信息。不要遗漏任何数值和标注。"},
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


def _compress_image(path: str, max_width: int = 1024, quality: int = 60) -> str:
    """Read an image, resize+compress to JPEG, return base64 string."""
    from PIL import Image as _Image

    img = _Image.open(path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        img = img.resize((max_width, int(h * ratio)), _Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()
