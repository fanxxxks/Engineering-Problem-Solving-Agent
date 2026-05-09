"""Enrich knowledge base with circuit and linear algebra cards/examples from the question bank.

Usage:
    python scripts/import_kb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = ROOT / "eng_solver_agent" / "retrieval"


def backup_file(path: Path) -> Path:
    """Create a timestamped backup of a file."""
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(f".backup_{ts}{path.suffix}")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"  Backup created: {backup.name}")
    return backup


# =========================================================================
# New formula cards
# =========================================================================

NEW_CIRCUITS_FORMULAS = [
    {
        "id": "circuits-source-transform-1",
        "subject": "circuits",
        "topic": "source_transformation",
        "keywords": ["source transformation", "thevenin", "norton", "equivalent", "电源等效变换"],
        "formula": "Vs = Is * R,  R_thevenin = R_norton",
        "conditions": ["voltage source in series with R", "current source in parallel with same R"],
        "common_traps": ["direction of current source must match voltage polarity", "dependent sources cannot be transformed independently"]
    },
    {
        "id": "circuits-superposition-1",
        "subject": "circuits",
        "topic": "superposition",
        "keywords": ["superposition", "linear", "independent source", "叠加定理"],
        "formula": "Total response = sum of responses to each independent source acting alone",
        "conditions": ["linear circuit", "deactivate other sources: V→short, I→open"],
        "common_traps": ["dependent sources must remain active", "power cannot be superposed (nonlinear)"]
    },
    {
        "id": "circuits-thevenin-1",
        "subject": "circuits",
        "topic": "thevenin",
        "keywords": ["thevenin", "equivalent", "open circuit", "戴维南"],
        "formula": "V_th = V_oc (open-circuit voltage), R_th = R_eq (with sources killed)",
        "conditions": ["linear two-terminal network"],
        "common_traps": ["keeping independent sources active when computing R_th", "dependent source requires test-source method"]
    },
    {
        "id": "circuits-norton-1",
        "subject": "circuits",
        "topic": "norton",
        "keywords": ["norton", "equivalent", "short circuit", "诺顿"],
        "formula": "I_n = I_sc (short-circuit current), R_n = R_th",
        "conditions": ["linear two-terminal network"],
        "common_traps": ["confusing Norton R with load R", "I_n and V_th share same R_eq"]
    },
    {
        "id": "circuits-max-power-1",
        "subject": "circuits",
        "topic": "maximum_power_transfer",
        "keywords": ["maximum power", "load", "thevenin", "最大功率"],
        "formula": "P_max = V_th^2 / (4 * R_th),  when R_L = R_th",
        "conditions": ["RL adjustable", "Thevenin equivalent known"],
        "common_traps": ["applying to non-adjustable load", "using wrong R_th (not including RL)"]
    },
    {
        "id": "circuits-ydelta-1",
        "subject": "circuits",
        "topic": "y_delta_transform",
        "keywords": ["Y-Δ", "star-delta", "wye-delta", "transform", "星三角"],
        "formula": "Δ→Y: R1 = (R12*R31) / (R12+R23+R31)",
        "conditions": ["three-terminal resistive network"],
        "common_traps": ["wrong terminal labeling", "applying to non-resistive elements indiscriminately"]
    },
    {
        "id": "circuits-current-divider-1",
        "subject": "circuits",
        "topic": "current_divider",
        "keywords": ["current divider", "parallel", "分流"],
        "formula": "I_k = (G_k / (G_1+...+G_n)) * I_total,  where G = 1/R",
        "conditions": ["resistors in parallel"],
        "common_traps": ["using R directly instead of conductance G", "wrong branch identification"]
    },
    {
        "id": "circuits-opamp-1",
        "subject": "circuits",
        "topic": "opamp_ideal",
        "keywords": ["op-amp", "virtual short", "virtual ground", "ideal", "运放", "虚短", "虚断"],
        "formula": "V+ = V- (virtual short),  I+ = I- = 0 (virtual open)",
        "conditions": ["ideal op-amp", "negative feedback", "linear region"],
        "common_traps": ["applying virtual short without negative feedback", "ignoring saturation limits"]
    },
    {
        "id": "circuits-mutual-inductance-1",
        "subject": "circuits",
        "topic": "coupled_inductor",
        "keywords": ["mutual inductance", "coupling", "互感", "耦合电感"],
        "formula": "V1 = L1*di1/dt ± M*di2/dt,  V2 = ±M*di1/dt + L2*di2/dt,  M = k*sqrt(L1*L2), 0≤k≤1",
        "conditions": ["magnetically coupled coils"],
        "common_traps": ["dot convention determines sign of M term", "k cannot exceed 1"]
    },
    {
        "id": "circuits-transient-1",
        "subject": "circuits",
        "topic": "transient_three_element",
        "keywords": ["transient", "three-element", "time constant", "三要素法", "暂态"],
        "formula": "f(t) = f(∞) + [f(0+) - f(∞)] * e^(-t/τ)",
        "conditions": ["first-order RC or RL circuit", "DC or step excitation"],
        "common_traps": ["wrong initial condition f(0+)", "RC: τ=RC, RL: τ=L/R"]
    },
    {
        "id": "circuits-resonance-1",
        "subject": "circuits",
        "topic": "resonance",
        "keywords": ["resonance", "RLC", "series", "parallel", "谐振"],
        "formula": "ω0 = 1/√(LC),  Q = ω0*L/R (series) = R/(ω0*L) (parallel)",
        "conditions": ["undamped natural frequency when X_L = X_C"],
        "common_traps": ["series vs parallel Q formulas differ", "bandwidth = ω0/Q"]
    },
    {
        "id": "circuits-threephase-1",
        "subject": "circuits",
        "topic": "three_phase",
        "keywords": ["three-phase", "Y", "Δ", "line", "phase", "三相"],
        "formula": "Y: V_line = √3*V_phase, I_line = I_phase;  Δ: V_line = V_phase, I_line = √3*I_phase",
        "conditions": ["balanced three-phase system"],
        "common_traps": ["Y vs Δ voltage/current relationships differ", "power factor affects total power"]
    },
    {
        "id": "circuits-power-conservation-1",
        "subject": "circuits",
        "topic": "power_conservation",
        "keywords": ["power", "conservation", "tellegen", "功率守恒"],
        "formula": "Σ p_k = 0  (sum of all element powers = 0)",
        "conditions": ["any lumped circuit"],
        "common_traps": ["passive sign convention: p>0 means absorbed", "source power is negative when delivering"]
    },
    {
        "id": "circuits-dc-ac-separation-1",
        "subject": "circuits",
        "topic": "dc_ac_superposition",
        "keywords": ["DC", "AC", "superposition", "non-sinusoidal", "直流交流分离"],
        "formula": "DC: C→open, L→short;  AC: C→1/(jωC), L→jωL",
        "conditions": ["circuit with DC + AC sources"],
        "common_traps": ["impedance depends on frequency", "DC and AC must be computed separately"]
    },
]

NEW_LINALG_FORMULAS = [
    {
        "id": "linalg-gauss-1",
        "subject": "linalg",
        "topic": "gauss_elimination",
        "keywords": ["gauss", "elimination", "row reduction", "消元法", "初等行变换"],
        "formula": "Three elementary row operations: swap rows, multiply row by nonzero scalar, add multiple of row to another",
        "conditions": ["any linear system Ax=b"],
        "common_traps": ["forgetting to apply same operations to RHS b", "pivot may need row swap"]
    },
    {
        "id": "linalg-augmented-1",
        "subject": "linalg",
        "topic": "augmented_matrix_solution",
        "keywords": ["augmented", "rank", "solution", "增广矩阵", "解的情况"],
        "formula": "rank(A) < rank([A|b]) → no solution; rank(A) = rank([A|b]) = n → unique; rank(A) = rank([A|b]) < n → infinite",
        "conditions": ["linear system Ax=b"],
        "common_traps": ["counting pivot columns vs total columns", "consistency requires equal ranks"]
    },
    {
        "id": "linalg-cramer-1",
        "subject": "linalg",
        "topic": "cramers_rule",
        "keywords": ["cramer", "determinant", "solve", "克拉默法则"],
        "formula": "x_i = det(A_i) / det(A),  where A_i replaces column i with b",
        "conditions": ["det(A) ≠ 0", "square system"],
        "common_traps": ["computationally expensive for large n", "must not have det(A)=0"]
    },
    {
        "id": "linalg-independence-1",
        "subject": "linalg",
        "topic": "linear_independence",
        "keywords": ["independence", "dependence", "basis", "线性相关", "线性无关"],
        "formula": "c1*v1 + ... + cn*vn = 0 has only trivial solution c_i=0",
        "conditions": ["set of vectors"],
        "common_traps": ["confusing linear dependence with zero determinant", "vectors can be dependent even if none is a scalar multiple"]
    },
    {
        "id": "linalg-solution-structure-1",
        "subject": "linalg",
        "topic": "solution_structure",
        "keywords": ["general solution", "particular", "homogeneous", "解的结构"],
        "formula": "x = x_particular + c1*v1 + ... + ck*vk (where v_i are homogeneous basis solutions)",
        "conditions": ["consistent linear system Ax=b"],
        "common_traps": ["forgetting the particular solution", "dimension of nullspace = n - rank(A)"]
    },
    {
        "id": "linalg-matrix-poly-inv-1",
        "subject": "linalg",
        "topic": "matrix_polynomial_inverse",
        "keywords": ["polynomial", "inverse", "cayley-hamilton", "矩阵多项式求逆"],
        "formula": "If A²+pA+qI=0 and q≠0, then A^(-1) = -(A+pI)/q",
        "conditions": ["A satisfies a polynomial equation with nonzero constant term"],
        "common_traps": ["only works when constant term ≠ 0", "sign errors in polynomial coefficients"]
    },
    {
        "id": "linalg-laplace-1",
        "subject": "linalg",
        "topic": "laplace_expansion",
        "keywords": ["laplace", "cofactor", "expansion", "拉普拉斯展开"],
        "formula": "det(A) = Σ(-1)^(i+j) * a_ij * M_ij  (expansion along row i)",
        "conditions": ["square matrix of any size"],
        "common_traps": ["sign pattern (-1)^(i+j) for cofactor", "minor vs cofactor distinction"]
    },
    {
        "id": "linalg-invertibility-1",
        "subject": "linalg",
        "topic": "matrix_invertibility",
        "keywords": ["invertible", "nonsingular", "conditions", "可逆条件"],
        "formula": "Equivalent: det(A)≠0, rank(A)=n, columns linearly independent, Ax=0 has only trivial solution",
        "conditions": ["n×n square matrix"],
        "common_traps": ["only square matrices can be invertible", "some conditions only apply to square matrices"]
    },
    {
        "id": "linalg-det-properties-1",
        "subject": "linalg",
        "topic": "determinant_properties",
        "keywords": ["determinant", "properties", "row swap", "行列式性质"],
        "formula": "Row swap: det→-det; Row scaling by k: det→k*det; Add multiple: det unchanged; det(kA)=k^n*det(A)",
        "conditions": ["square matrix, elementary row operations"],
        "common_traps": ["det(kA) = k^n*det(A), not k*det(A)", "row and column operations have same effect on determinant"]
    },
    {
        "id": "linalg-vandermonde-1",
        "subject": "linalg",
        "topic": "special_determinants",
        "keywords": ["vandermonde", "circulant", "special matrix", "范德蒙德"],
        "formula": "Vandermonde det = ∏_{i<j} (x_j - x_i)",
        "conditions": ["matrix with rows [1, x_i, x_i², ..., x_i^(n-1)]"],
        "common_traps": ["ordering matters for sign", "applicable only to Vandermonde structure"]
    },
]


def add_formula_cards() -> int:
    path = RETRIEVAL_DIR / "formula_cards.json"
    backup_file(path)

    with open(path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    all_new = NEW_CIRCUITS_FORMULAS + NEW_LINALG_FORMULAS
    existing_ids = {card["id"] for card in cards}
    added = 0

    for card in all_new:
        if card["id"] not in existing_ids:
            cards.append(card)
            existing_ids.add(card["id"])
            added += 1
            print(f"  + [formula] {card['id']}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)

    total = len(cards)
    print(f"  formula_cards.json: {total} total ({added} new)")
    return added


# =========================================================================
# New solved examples (from real question bank data)
# =========================================================================

NEW_CIRCUITS_EXAMPLES = [
    {
        "question_id": "circuits-ex-4",
        "subject": "circuits",
        "topic": "kcl_kvl_analysis",
        "question": "电路如图，已知 R1=1Ω，R2=2Ω，R3=3Ω，us1=10V，us2=5V，求各支路电流。",
        "reasoning_process": "设各支路电流 i1, i2, i3。\n对节点a应用KCL：i1 = i2 + i3\n对左边网孔应用KVL：i1×R1 + i2×R2 = us1，即 i1 + 2i2 = 10\n对右边网孔应用KVL：-i2×R2 + i3×R3 = -us2，即 -2i2 + 3i3 = -5\n联立求解：由第三个方程得 i3 = (2i2-5)/3；代入KCL得 i1 = i2 + (2i2-5)/3 = (5i2-5)/3\n代入KVL：5(i2-1)/3 + 2i2 = 10 → 5i2-5+6i2=30 → 11i2=35 → i2=35/11 A\ni3=(70/11-5)/3=5/11 A\ni1=35/11+5/11=40/11 A\n答案：i1=40/11 A, i2=35/11 A, i3=5/11 A",
        "answer": "i1=40/11 A, i2=35/11 A, i3=5/11 A",
        "tags": ["KCL", "KVL", "branch current", "mesh"]
    },
    {
        "question_id": "circuits-ex-5",
        "subject": "circuits",
        "topic": "power_conservation",
        "question": "电路中有多个电阻和电源，已知各元件电压和电流，验证功率守恒：∑p = 0",
        "reasoning_process": "功率守恒（Tellegen定理）：对于任何集总参数电路，所有元件吸收的瞬时功率代数和恒为零。\n电阻吸收功率：p_R = i²R = v²/R（恒为正）\n电压源功率：若电流从正极流入则吸收功率，从正极流出则发出功率（负值）\n电流源功率：需先求出电流源两端电压，p = u×i，方向判断同上。\n验证：∑p_电阻 + ∑p_电源 = 0，若不等于0说明计算有误。",
        "answer": "∑p = 0，功率守恒成立",
        "tags": ["power", "conservation", "Tellegen", "verification"]
    },
    {
        "question_id": "circuits-ex-6",
        "subject": "circuits",
        "topic": "source_transformation",
        "question": "电路中含电压源Us=24V串R1=4Ω与R2=8Ω并联后再串R3=12Ω，利用电源等效变换求电流I。",
        "reasoning_process": "步骤1：将电压源Us=24V串联R1=4Ω等效变换为电流源Is=Us/R1=6A并联R1=4Ω。\n步骤2：R1=4Ω与R2=8Ω并联，等效电阻R12=4×8/(4+8)=8/3 Ω。\n步骤3：将Is=6A并联R12=8/3Ω再等效变换为电压源Us'=6×8/3=16V串联R12=8/3Ω。\n步骤4：总电阻R_total=R12+R3=8/3+12=44/3 Ω。\n步骤5：电流I=Us'/R_total=16/(44/3)=48/44=12/11 A。",
        "answer": "I = 12/11 A ≈ 1.09 A",
        "tags": ["source transformation", "equivalent resistance", "simplification"]
    },
    {
        "question_id": "circuits-ex-7",
        "subject": "circuits",
        "topic": "thevenin",
        "question": "求a、b端口的戴维南等效电路：求开路电压Uoc和等效电阻Req。",
        "reasoning_process": "第一步：求开路电压Uoc。\n断开a、b端口（移去负载），用KVL/KCL或节点法计算开路时ab两端的电压，即Uoc。\n第二步：求等效电阻Req。\n方法一（独立源置零法）：将电路中所有独立电压源短路、独立电流源开路，从ab端看入的等效电阻即为Req。\n方法二（开路短路法）：Req = Uoc / Isc，其中Isc为ab端短路时的短路电流。\n含受控源时必须用方法二或外加电源法。\n戴维南等效电路：Uoc与Req串联。",
        "answer": "戴维南等效电路：Uoc 串联 Req",
        "tags": ["thevenin", "equivalent circuit", "open-circuit voltage"]
    },
    {
        "question_id": "circuits-ex-8",
        "subject": "circuits",
        "topic": "maximum_power_transfer",
        "question": "求负载电阻RL为何值时可获得最大功率，并求此最大功率。",
        "reasoning_process": "最大功率传输定理：对于线性有源二端网络，当负载电阻RL等于网络的戴维南等效电阻Req时，负载获得最大功率。\n步骤1：先求ab端口的戴维南等效电路，得到开路电压Uoc和等效电阻Req。\n步骤2：当RL = Req时，负载功率最大。\n步骤3：最大功率 Pmax = (Uoc)² / (4 × Req)。\n注意：此定理仅适用于负载电阻可调的情况，不适用于负载固定而改变电源内阻。",
        "answer": "RL = Req 时，Pmax = Uoc²/(4Req)",
        "tags": ["maximum power", "load matching", "thevenin"]
    },
    {
        "question_id": "circuits-ex-9",
        "subject": "circuits",
        "topic": "superposition",
        "question": "电路含电压源Us=6V和电流源Is=1A，R1=R2=R3=2Ω，利用叠加定理求电流I。",
        "reasoning_process": "叠加定理步骤：\n1. 电压源Us单独作用（电流源Is开路）：\n   电阻网络：R2和R3串联后与？实际为R1与(R2+R3)？分析原电路拓扑。\n   设此时电流为I'。\n2. 电流源Is单独作用（电压源Us短路）：\n   利用分流公式求电流I''。\n3. 总响应：I = I' + I''。\n注意：功率不可叠加（因为功率是电压或电流的二次函数）。",
        "answer": "I = I' + I''（分别由叠加求得）",
        "tags": ["superposition", "linear", "independent source"]
    },
    {
        "question_id": "circuits-ex-10",
        "subject": "circuits",
        "topic": "transient_analysis",
        "question": "一阶RC电路，已知Us=10V，R=2Ω，C=0.5F，t=0时开关闭合，uc(0)=0V，求t>0时的uc(t)。",
        "reasoning_process": "利用三要素法：f(t) = f(∞) + [f(0+) - f(∞)] × e^(-t/τ)\n1. 初始值f(0+)：换路瞬间，电容电压不能突变，uc(0+) = uc(0-) = 0V。\n2. 稳态值f(∞)：t→∞时，电容相当于开路，uc(∞) = Us = 10V。\n3. 时间常数τ：τ = ReqC = 2 × 0.5 = 1s。\n代入三要素公式：uc(t) = 10 + (0 - 10) × e^(-t/1) = 10(1 - e^(-t)) V。",
        "answer": "uc(t) = 10(1 - e^(-t)) V, t ≥ 0",
        "tags": ["RC", "transient", "three-element method", "time constant"]
    },
    {
        "question_id": "circuits-ex-11",
        "subject": "circuits",
        "topic": "resonance",
        "question": "RLC串联电路，已知R=10Ω，L=0.1H，C=10μF，求谐振角频率ω0和品质因数Q。",
        "reasoning_process": "串联RLC谐振条件：感抗等于容抗，即 ω0L = 1/(ω0C)。\n谐振角频率：ω0 = 1/√(LC) = 1/√(0.1 × 10×10⁻⁶) = 1/√(10⁻⁶) = 1000 rad/s。\n品质因数（串联）：Q = ω0L/R = 1000 × 0.1 / 10 = 10。\n通频带：BW = ω0/Q = 1000/10 = 100 rad/s。",
        "answer": "ω0 = 1000 rad/s, Q = 10",
        "tags": ["resonance", "RLC", "quality factor"]
    },
    {
        "question_id": "circuits-ex-12",
        "subject": "circuits",
        "topic": "opamp_analysis",
        "question": "理想运放构成的反相比例放大器，输入电阻R1=10kΩ，反馈电阻Rf=100kΩ，输入电压ui=0.1V，求输出电压uo。",
        "reasoning_process": "理想运放特性：虚短（u+ = u-），虚断（i+ = i- = 0）。\n反相输入端为虚地：u- = u+ = 0V（同相端接地）。\n对反相输入端列KCL：i1 + if = 0（因为运放输入电流为0）。\ni1 = ui/R1 = 0.1V/10kΩ = 0.01mA（从输入端流向反相端）。\nif = (uo - 0)/Rf（从反相端流向输出端，因uo为负）。\n根据虚地：i1 = -(uo/Rf)，即uo = -i1×Rf。\nuo = -(0.1/10k)×100k = -1V。\n放大倍数Au = vo/vi = -Rf/R1 = -10。",
        "answer": "uo = -1V, Au = -10",
        "tags": ["op-amp", "inverting amplifier", "virtual ground", "虚短虚断"]
    },
    {
        "question_id": "circuits-ex-13",
        "subject": "circuits",
        "topic": "three_phase",
        "question": "对称三相Y型连接负载，线电压UL=380V，每相阻抗Z=10∠30°Ω，求线电流IL和三相总功率P。",
        "reasoning_process": "对称三相Y型连接：\n1. 相电压 Up = UL/√3 = 380/√3 ≈ 220V。\n2. 线电流等于相电流（Y接）：IL = Ip = Up/|Z| = 220/10 = 22A。\n3. 功率因数 cosφ = cos(30°) = √3/2 ≈ 0.866。\n4. 三相总功率：P = √3 × UL × IL × cosφ = √3 × 380 × 22 × 0.866 ≈ 12540W。\n或 P = 3 × Up × Ip × cosφ = 3 × 220 × 22 × 0.866 ≈ 12540W。",
        "answer": "IL = 22A, P ≈ 12.54kW",
        "tags": ["three-phase", "Y-connection", "power", "phasor"]
    },
]

NEW_LINALG_EXAMPLES = [
    {
        "question_id": "linalg-ex-4",
        "subject": "linalg",
        "topic": "gauss_elimination",
        "question": "复习总结Gauss消元法的步骤；说明如何利用系数矩阵和增广矩阵来判断线性方程组解的情况。",
        "reasoning_process": "Gauss消元法核心步骤：\n1. 写出增广矩阵[A|b]。\n2. 通过三种初等行变换（倍加变换、倍乘变换、行对换）将增广矩阵化为行阶梯形。\n   - 倍加变换：将某行的非零倍数加到另一行\n   - 倍乘变换：将某行乘以非零常数\n   - 行对换：交换两行位置\n3. 判断解的情况：\n   - 若最后一列（常数项列）是主元列（出现0=非零），则无解\n   - 若最后一列不是主元列，有解：\n     * 主元个数 = 未知数个数 → 唯一解\n     * 主元个数 < 未知数个数 → 无穷多解",
        "answer": "初等行变换 → 行阶梯形 → 根据主元列位置判断：无解/唯一解/无穷多解",
        "tags": ["gauss elimination", "row reduction", "augmented matrix", "solution type"]
    },
    {
        "question_id": "linalg-ex-5",
        "subject": "linalg",
        "topic": "determinant_computation",
        "question": "计算行列式 |-2 1 -3; 98 101 97; 1 -3 4|",
        "reasoning_process": "按第一行展开计算三阶行列式：\nD = -2×|101 97; -3 4| - 1×|98 97; 1 4| + (-3)×|98 101; 1 -3|\n= -2×(101×4 - 97×(-3)) - (98×4 - 97×1) - 3×(98×(-3) - 101×1)\n= -2×(404 + 291) - (392 - 97) - 3×(-294 - 101)\n= -2×695 - 295 - 3×(-395)\n= -1390 - 295 + 1185 = -500",
        "answer": "-500",
        "tags": ["determinant", "3x3", "cofactor expansion"]
    },
    {
        "question_id": "linalg-ex-6",
        "subject": "linalg",
        "topic": "eigenvalue_characteristic",
        "question": "已知矩阵A的特征多项式为det(A-λI)=λ³+6λ²-λ-30=0，求A的所有特征值。",
        "reasoning_process": "求解特征方程 λ³ + 6λ² - λ - 30 = 0：\n试根：代入λ=2，8+24-2-30=0，故λ=2是一个根。\n因式分解：(λ-2)(λ²+8λ+15) = (λ-2)(λ+3)(λ+5) = 0。\n解得三个特征值：λ1=2, λ2=-3, λ3=-5。",
        "answer": "λ = 2, -3, -5",
        "tags": ["eigenvalues", "characteristic polynomial", "factoring"]
    },
    {
        "question_id": "linalg-ex-7",
        "subject": "linalg",
        "topic": "linear_system_parameter",
        "question": "线性方程组含参数λ：x1+2x2+x3=1, 2x1+3x2+(λ+2)x3=3, x1+λx2-2x3=0。讨论λ取何值时无解、唯一解、无穷多解。",
        "reasoning_process": "对增广矩阵作初等行变换：\n[1 2 1 | 1; 2 3 λ+2 | 3; 1 λ -2 | 0]\n→ [1 2 1 | 1; 0 -1 λ | 1; 0 λ-2 -3 | -1]\n→ [1 2 1 | 1; 0 1 -λ | -1; 0 0 (λ-3)(λ+1) | λ-3]\n分析：\n- λ=-1时：第三行变为[0 0 0 | -4]，系数矩阵rank=2<增广矩阵rank=3，无解\n- λ=3时：第三行全零，rank=2<3=n，有无穷多解\n- λ≠3且λ≠-1时：rank=3=n，有唯一解",
        "answer": "λ=-1 无解; λ=3 无穷多解; λ≠3且λ≠-1 唯一解",
        "tags": ["parameter", "solution type", "rank", "row reduction"]
    },
    {
        "question_id": "linalg-ex-8",
        "subject": "linalg",
        "topic": "matrix_polynomial_inverse",
        "question": "已知n阶方阵A满足A²+3A-4I=0，求(A+3I)⁻¹和(A+5I)⁻¹，并讨论(A+mI)可逆的条件。",
        "reasoning_process": "由A²+3A-4I=0，因式分解得(A+4I)(A-I)=0或(A-I)(A+4I)=0。\n1. 求(A+3I)⁻¹：\n   由A²+3A-4I=0 → A(A+3I)=4I → (A/4)(A+3I)=I，故(A+3I)⁻¹ = A/4。\n2. 求(A+5I)⁻¹：\n   由(A+5I)(A-2I)=A²+3A-10I=6I（利用原方程A²=-3A+4I），\n   即(A+5I)(A-2I)=6I → (A+5I)⁻¹=(A-2I)/6。\n3. (A+mI)可逆的充要条件是det(A+mI)≠0，即-m不是A的特征值。\n   由A的特征值λ满足λ²+3λ-4=0得λ=1或λ=-4，因此m≠-1且m≠4时(A+mI)可逆。",
        "answer": "(A+3I)⁻¹ = A/4; (A+5I)⁻¹ = (A-2I)/6; m≠-1且m≠4时(A+mI)可逆",
        "tags": ["matrix polynomial", "inverse", "eigenvalue", "Cayley-Hamilton"]
    },
    {
        "question_id": "linalg-ex-9",
        "subject": "linalg",
        "topic": "linear_independence_proof",
        "question": "设非齐次方程组Ax=b有解，其对应齐次方程组解集为{k[1,-1,1]^T}。证明非齐次解集为特解加齐次通解的形式。",
        "reasoning_process": "非齐次线性方程组解的结构定理：\n若x0是非齐次方程组Ax=b的一个特解，H={x|Ax=0}是齐次解空间，\n则非齐次解集为S={x0+h|h∈H}。\n证明：\n(1) 任取h∈H，A(x0+h)=Ax0+Ah=b+0=b，故x0+h是非齐次解。\n(2) 任取x是Ax=b的解，A(x-x0)=Ax-Ax0=b-b=0，故x-x0∈H。\n    即存在k使得x-x0=k·[1,-1,1]^T，因此x=x0+k·[1,-1,1]^T。",
        "answer": "解集 = {x0 + k[1,-1,1]^T | k∈R}，其中x0为特解",
        "tags": ["solution structure", "particular", "homogeneous", "proof"]
    },
    {
        "question_id": "linalg-ex-10",
        "subject": "linalg",
        "topic": "special_determinant",
        "question": "设x1,x2,x3是方程x³+px+q=0的三个根，求行列式|x1 x2 x3; x3 x1 x2; x2 x3 x1|的值。",
        "reasoning_process": "展开三阶行列式：\nD = x1³ + x2³ + x3³ - 3x1x2x3（循环行列式展开公式）。\n利用立方和的因式分解：\na³+b³+c³-3abc = (a+b+c)(a²+b²+c²-ab-bc-ca)。\n由韦达定理，x1+x2+x3=0（方程x³+px+q=0的二次项系数为0）。\n因此D = (x1+x2+x3)(...) = 0 × (...) = 0。",
        "answer": "0",
        "tags": ["determinant", "circulant", "Vieta", "roots"]
    },
    {
        "question_id": "linalg-ex-11",
        "subject": "linalg",
        "topic": "matrix_operation_properties",
        "question": "判断以下命题是否正确：(1)(A+B)²=A²+2AB+B²；(2)若AB=AC且A≠0则B=C；(3)若A²=I则A=±I。",
        "reasoning_process": "(1) 错误。矩阵乘法不满足交换律，(A+B)² = A²+AB+BA+B²。仅当AB=BA时原式成立。\n(2) 错误。矩阵乘法不满足消去律。反例：A=[1 1;0 0], B=[1;-1], C=[2;-2]，满足AB=AC=0但B≠C。\n(3) 错误。反例：A=[0 1;1 0]，A²=I但A≠±I。\n正确理解：矩阵运算不能简单类比实数运算，必须注意乘法交换律和消去律在矩阵中不成立。",
        "answer": "全错。(1)缺少BA项；(2)矩阵无消去律；(3)存在非±I的平方为单位阵的矩阵",
        "tags": ["matrix properties", "commutativity", "cancellation", "counterexample"]
    },
]


def add_solved_examples() -> int:
    path = RETRIEVAL_DIR / "solved_examples.jsonl"
    backup_file(path)

    existing_ids = set()
    lines = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
                    try:
                        obj = json.loads(line)
                        existing_ids.add(obj.get("question_id", ""))
                    except json.JSONDecodeError:
                        pass

    all_new = NEW_CIRCUITS_EXAMPLES + NEW_LINALG_EXAMPLES
    added = 0

    with open(path, "a", encoding="utf-8") as f:
        for ex in all_new:
            qid = ex["question_id"]
            if qid not in existing_ids:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                existing_ids.add(qid)
                added += 1
                print(f"  + [example] {qid}")

    total = len(lines) + added
    print(f"  solved_examples.jsonl: {total} total ({added} new)")
    return added


def main() -> int:
    print("=" * 60)
    print("Enriching Knowledge Base from Question Bank Data")
    print("=" * 60)

    print("\n[1/2] Adding new formula cards...")
    n_formulas = add_formula_cards()

    print("\n[2/2] Adding new solved examples...")
    n_examples = add_solved_examples()

    print(f"\n{'=' * 60}")
    print(f"Done: {n_formulas} formula cards + {n_examples} solved examples added.")
    print("Run 'python build_kb.py' to rebuild the FAISS index.")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
