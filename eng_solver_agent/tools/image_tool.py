"""Image description tool - sends diagrams to Kimi multimodal for text description."""

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

    _CIRCUIT_PROMPT = (
        "【角色设定】\n"
        "你是一位拥有顶级专业水平的电子电气工程师（精通电路原理、模拟CMOS集成电路设计）"
        "及图像细节提取专家。你的任务是将我提供的图片，完全转化为零歧义、纯文字的结构化拓扑描述。\n\n"
        "【核心提取原则】\n"
        "严禁空间描述：禁止使用'左边'、'上面'、'并联'、'串联'等词，必须且只能使用'节点（Node）连接法'描述。\n"
        "尊重原图符号：对于相量、复数阻抗、变量下标，必须使用准确的Markdown/LaTeX语法还原，绝不能漏掉打点或正负号。\n"
        "识别全局网络：识别VDD、GND（地）、独立端口，并将它们视为绝对的全局节点。\n\n"
        "【执行步骤】\n"
        "请严格判断图片类型，并按对应逻辑输出报告：\n\n"
        "预检：判定图片类型\n"
        "观察图片，判定它属于以下哪一类，并执行对应的解析逻辑：\n"
        "类型A：波形图/函数图（带有横纵坐标的曲线）。如果是此类，请跳过后续步骤，直接描述横纵坐标代表的物理量、波形的形状及关键特征点。\n"
        "类型B：相量图（只有带有箭头的矢量线段，无电路元件）。如果是此类，请直接描述各矢量的名称、相对长度关系及相位关系。\n"
        "类型C：电路拓扑图（包含电阻、电容、晶体管、电源等）。如果是此类，请严格执行后续步骤1-6。\n\n"
        "【电路图专属提取步骤】（仅针对类型C执行）\n"
        "步骤一：定义全局节点与端口\n"
        "提取图中已明确标出的节点字母/文字。明确列出所有的电源轨和接地端。"
        "明确列出所有开放端口。如果图中有未命名的交叉连接点，请自行用N1,N2...为其命名。\n"
        "步骤二：提取基础无源元件\n"
        "逐一列出电阻(R)、电容(C)、电感(L)或阻抗盒(Z)。"
        "格式：名称+参数/阻抗值+连接的两个节点。\n"
        "步骤三：提取电源与受控源\n"
        "独立电源：电压源/电流源。必须明确正负极连接的节点，或电流流向的节点。"
        "受控源（菱形符号）：明确类型、连接节点及极性/方向、增益系数、控制变量的原始定义位置和参考方向。\n"
        "步骤四：提取有源器件与集成电路\n"
        "MOSFET/晶体管：指出类型(NMOS/PMOS)，栅极(Gate)、漏极(Drain)、源极(Source)各连接哪个节点。"
        "运算放大器：指出同相输入端(+)、反相输入端(-)、输出端各接哪个节点。\n"
        "步骤五：提取参考方向与网孔信息\n"
        "支路变量：提取图中所有参考电流及其流向，参考电压及其正负极位置。"
        "网孔电流：描述网孔包围了哪些元件及环形箭头的方向。\n"
        "步骤六：提取动态动作\n"
        "如图中有开关(S)，请描述其初始状态及箭头指示的动作方向。\n\n"
        "【重要】：只描述图片内容，不要解题。不要输出寒暄或解释。越快越好。"
    )

    _GENERAL_PROMPT = (
        "你是一个工程图表分析专家。请详细描述这张图片的内容：\n"
        "1. 图片类型（函数图像/受力分析图/坐标图/矩阵/公式推导/其他）\n"
        "2. 图中标注的所有数值、坐标、变量名\n"
        "3. 图中曲线/直线的几何特征（交点、极值点、渐近线、斜率变化等，如有）\n"
        "4. 图中隐含的物理/数学关系（如有公式，用 LaTeX 写出）\n"
        "【重要】：只描述图片内容，不要解题。四条完成立即结束。"
    )

    _CIRCUIT_SYSTEM = (
        "详细描述图片内容，主要用数理符号，少用中文。不要遗漏任何数值和标注。\n"
        "【规则】\n"
        "1. 只描述图片内容，不要进行题目分析或解题步骤\n"
        "2. 不要输出寒暄或解释性文字\n"
        "3. 数学公式和变量必须使用标准 LaTeX 格式（如 $U_{ab}$，$\\Omega$）\n"
        "【重要】：只描述图片内容，不要解题。所有步骤完成立即结束，越快越好。"
    )

    _GENERAL_SYSTEM = (
        "详细描述图片内容，主要用数理符号和 LaTeX 格式，少用中文。不要遗漏任何数值和标注。\n"
        "【规则】\n"
        "1. 只描述图片内容，不要进行题目分析或解题步骤\n"
        "2. 不要输出寒暄或解释性文字\n"
        "3. 数学公式和变量必须使用标准 LaTeX 格式"
    )

    def compute(self, image_path: str = "", subject: str = "", query: str = "") -> str:
        """Read an image and return a structured text description.

        Args:
            image_path: Relative path to the image file (e.g. "images/CIR_004.png").
            subject: "circuits" uses a circuit-specific prompt,
                     anything else uses a general-purpose prompt.
            query: Optional extra instructions.
        """
        abs_path = self._find_image(image_path)
        if abs_path is None:
            return f"[错误] 未找到图片文件: {image_path}"

        if self._kimi is None:
            return "[错误] LLM 客户端不可用，无法描述图片"

        img_b64 = _compress_image(abs_path)

        is_circuit = str(subject).strip().lower() == "circuits"
        if query:
            base = self._CIRCUIT_PROMPT if is_circuit else self._GENERAL_PROMPT
            prompt = query + "\n\n" + base
        else:
            prompt = self._CIRCUIT_PROMPT if is_circuit else self._GENERAL_PROMPT
        system = self._CIRCUIT_SYSTEM if is_circuit else self._GENERAL_SYSTEM

        messages = [
            {"role": "system", "content": system},
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
        if _os.path.isfile(image_rel):
            return _os.path.abspath(image_rel)
        return None


def _compress_image(path: str, max_width: int = 512, quality: int = 30) -> str:
    """Read an image, aggressively compress to tiny JPEG, return base64 string."""
    from PIL import Image as _Image

    img = _Image.open(path)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    if img.mode == "RGB":
        gray = img.convert("L")
        img = gray.convert("RGB")
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        img = img.resize((max_width, int(h * ratio)), _Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()
