# Engineering Problem Solving Agent — 工程问题求解智能体

Competition-grade agent for solving engineering foundation course problems (Calculus, Linear Algebra, Circuits, Physics).

## 一、项目概述

本项目是为 **"未央城" AI Agent 竞赛** 打造的竞赛级工程问题求解系统，专注于解决**工科基础课程**（微积分、线性代数、电路原理、基础物理）的数学与物理问题。

系统采用 **"大语言模型（LLM）推理 + 精确工具计算"** 的双引擎架构，既能通过 LLM 处理需要符号推导、证明和复杂推理的问题，又能通过基于 SymPy 的工具层完成高精度的数值计算和符号运算。整个系统针对中文竞赛环境进行了深度优化，支持中文题干解析、矩阵字面量提取、物理量自动识别等特性。

### 核心设计目标

| 目标 | 说明 |
|------|------|
| **准确性（40%）** | LLM 负责证明与符号推导，工具负责数值验证 |
| **推理质量（25%）** | ReAct 循环生成详细的中文数学推导步骤 |
| **方案设计（20%）** | 创新的多模式架构，工具与 LLM 深度集成 |
| **效率（10%）** | 基于 asyncio.Semaphore 的并行求解，控制并发避免限流 |
| **鲁棒性（5%）** | 工具/LLM 失败时的优雅降级，保证始终有输出 |

### 关键特性

- **统一入口（UnifiedAgent）**：将原先 3 个独立的智能体合并为一个 API，提供 5 种求解模式
- **ReAct 推理引擎**：Think → Act → Observe 循环，最大化推理过程得分
- **并行处理**：`async_solve()` 控制并发数（默认 5），避免 API 速率限制
- **中文文本解析**：支持 `[[a,b],[c,d]]` 矩阵字面量、`|a b; c d|` 行列式标记、电阻值提取
- **零依赖运行**：核心代码仅依赖 Python 标准库，SymPy/Pydantic 为可选增强
- **智能答案评分**：公式匹配、数值容差、关键词 Jaccard 相似度

---

## 二、整体架构

### 2.1 架构全景图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         统一入口层 (Entrypoint)                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  UnifiedAgent (推荐) / CompetitionAgent / EngineeringSolverAgent │ │
│  │  - solve_one()  单题求解                                     │   │
│  │  - solve()      顺序批量求解                                  │   │
│  │  - async_solve() 并行批量求解 (asyncio + Semaphore)            │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
         ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
         │   路由层     │  │  适配层     │  │  验证层     │
         │ QuestionRouter│  │QuestionAdapter│  │  verifier   │
         └─────────────┘  └─────────────┘  └─────────────┘
                    │              │              │
                    └──────────────┼──────────────┘
                                   ▼
         ┌───────────────────────────────────────────────────────────┐
         │                   求解策略层 (Strategy)                     │
         │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │
         │  │  auto   │ │  react  │ │ legacy  │ │llm_only │ │tool_only│ │
         │  │ 自适应  │ │ReAct推理│ │ 经典流水线│ │ 纯LLM  │ │ 纯工具  │ │
         │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────────┘ │
         └───────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
         ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
         │   LLM 层    │  │  工具调度层  │  │  知识检索层  │
         │ KimiClient  │  │ToolDispatcher│  │  Retriever  │
         │PromptBuilder│  │             │  │  KBLoader   │
         └─────────────┘  └─────────────┘  └─────────────┘
                    │              │              │
                    └──────────────┼──────────────┘
                                   ▼
         ┌───────────────────────────────────────────────────────────┐
         │                    工具执行层 (Tools)                       │
         │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
         │  │CalculusTool│ │AlgebraTool│ │CircuitTool│ │PhysicsTool│      │
         │  │ 微积分工具 │ │ 线性代数  │ │ 电路分析  │ │ 物理求解  │      │
         │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
         └───────────────────────────────────────────────────────────┘
```

### 2.2 数据流（以 `auto` 模式为例）

```
输入题目 (dict)
    │
    ▼
QuestionAdapter.normalize() ──→ 标准化字段 (question_id, question, subject)
    │
    ▼
QuestionRouter.route_with_confidence() ──→ 学科分类 (physics/circuits/linalg/calculus) + 置信度
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  模式选择：auto 模式会检测 LLM 可用性                                │
│  - 若 LLM 可用 → 走 ReAct 推理 (_solve_react)                       │
│  - 若 LLM 不可用 → 走经典流水线 (_solve_legacy)                      │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
ReAct 模式：
    ReActEngine.solve()
        ├── Step 1~N: Think → Act → Observe 循环
        │   - LLM 输出 "思考" + "行动"（工具调用或无）
        │   - 若行动为工具调用 → 执行工具 → 观察结果 → 反馈给 LLM
        │   - 若行动为 "最终答案" → 结束循环
        └── 格式化推理过程 → format_submission_item()

Legacy 模式：
    Stage 1: _analyze()
        └── LLM 分析题目 → 提取 subject, topic, knowns, unknowns, equations...
    Stage 2: ToolDispatcher.dispatch()
        └── 根据学科和题目内容调用对应工具
    Stage 3: _draft()
        └── LLM 结合分析结果和工具输出，生成中文推理过程和最终答案
    Stage 4: format_submission_item() + verifier()
        └── 格式化为竞赛标准 JSON 并验证字段完整性
```

---

## 三、模块详解

### 3.1 统一入口层

#### `unified_agent.py` — UnifiedAgent（推荐入口）

这是项目的**核心统一入口**，合并了早期分散在 `agent.py`、`agent_v2.py`、`competition_agent.py` 中的功能。

**设计动机**：
- 原项目存在 3 个功能重叠但接口不一致的智能体类，导致调用方困惑
- UnifiedAgent 提供单一干净的 API，同时保留所有求解策略的灵活性

**核心 API**：

```python
agent = UnifiedAgent()

# 单题求解
result = agent.solve_one(question, mode="auto")

# 顺序批量求解
results = agent.solve(questions, mode="auto")

# 并行批量求解（竞赛推荐）
results = asyncio.run(agent.async_solve(questions, max_concurrent=5, mode="auto"))
```

**构造函数参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `settings` | `Settings` | `None` | 配置对象，默认从环境变量加载 |
| `router` | `QuestionRouter` | `None` | 题目路由器，默认创建 |
| `kimi_client` | `Any` | `None` | LLM 客户端，`None` 表示不启用 LLM |
| `retriever` | `Retriever` | `None` | 知识检索器，默认从本地 JSON 加载 |
| `tool_registry` | `dict` | `None` | 工具注册表，默认创建 4 类工具 |
| `default_mode` | `str` | `"auto"` | 默认求解模式 |

**五种求解模式**：

| 模式 | LLM | 工具 | 速度 | 适用场景 |
|------|-----|------|------|----------|
| `auto` | ✅ 自适应 | ✅ | 中等 | **默认推荐**，自动选择最优策略 |
| `react` | ✅ 多轮 | ✅ | 较慢 | 需要详细逐步推理的题目 |
| `legacy` | ✅ 两阶段 | ✅ | 中等 | 兼容旧版行为的流水线模式 |
| `llm_only` | ✅ 直接 | ❌ | 中等 | 证明题、符号推导、无需精确计算 |
| `tool_only` | ❌ | ✅ | <1ms/题 | 纯数值计算，无 API 调用，最快 |

---

#### `agent.py` — EngineeringSolverAgent（原始智能体）

这是项目的**初代智能体**，定义了经典的 **"分析 → 工具 → 起草"** 三阶段流水线。

**流水线阶段**：

1. **`_normalize_input()`**：通过 `QuestionAdapter` 标准化输入字段
2. **`_analyze_question()`**：调用 LLM 分析题目，提取结构化信息（subject, topic, knowns, unknowns, equations_or_theorems, should_use_tool, target_form, possible_traps）
3. **`_run_tool()`**：根据分析结果的学科，调用对应工具进行精确计算
4. **`_draft_answer()`**：LLM 结合分析结果和工具输出，生成中文推理过程和最终答案

**关键特性**：
- 内置丰富的**中文文本提取器**：表达式提取、矩阵提取、电阻值提取、物理量提取
- 预编译正则表达式（模块级常量），避免重复编译开销
- 完善的 fallback 机制：LLM 不可用时自动降级为基于规则的分析和草稿生成

**文本提取能力**：
- **表达式提取**：支持中英文模式，如 "求 x^2 的导数"、"derivative of x^2"
- **矩阵提取**：支持 `[[1,2],[3,4]]`、`|1 2; 3 4|`、`[1 2; 3 4]` 三种格式
- **电阻提取**：从 "2Ω"、"3 欧姆" 等中文文本中提取数值
- **物理量提取**：自动识别质量 (kg)、速度 (m/s)、加速度 (m/s²)、力 (N)

---

#### `competition_agent.py` — CompetitionAgent（竞赛专用）

针对**竞赛评分规则**优化的智能体，核心洞察是：**即使答案错误，只要公式和步骤正确，仍能拿到 70% 的分数**。

**设计决策**：
- 优先使用 **ReAct 推理引擎** 生成详细的逐步中文数学推导
- 工具仅作为验证手段，而非答案的唯一来源
- 当 ReAct 失败时，优雅降级到 legacy 流水线

---

#### `agent_v2.py` — EnhancedSolverAgent（增强版）

在 `EngineeringSolverAgent` 基础上增加了 **LLM 直接求解** 的优先路径。

**工作流程**：
1. 首先尝试让 LLM 直接完整求解题目（带 few-shot 提示）
2. 若直接求解成功且答案非空，直接返回结果
3. 若失败，回退到父类的经典三阶段流水线

---

### 3.2 推理引擎层

#### `reasoning_engine.py` — ReActEngine

实现 **ReAct（Reasoning + Acting）** 推理循环，这是本项目在竞赛中的**架构创新点**。

**ReAct 循环**：

```
初始化：构建 system prompt + 题目描述 + 可用工具列表

For step = 1..MAX_STEPS(8):
    1. LLM 输出：
       - 思考：当前步骤的推理过程
       - 行动：[工具名] / [无] / [最终答案]
       - 行动输入：工具参数的 JSON
    2. 若行动 == "最终答案" → 结束循环，返回答案
    3. 若行动 == 工具名 → 执行工具 → 获得观察结果 → 反馈给 LLM
    4. 若行动 == "无" → 继续下一步推理

格式化：将所有步骤拼接为竞赛要求的 reasoning_process 字符串
```

**数据结构**：
- `ReasoningStep`：记录每步的 step_number, thought, action, action_input, observation, is_final
- `ReasoningResult`：汇总所有步骤、最终推理过程、答案、工具调用记录、成功标志

**工具调用协议**：
LLM 需按以下格式输出（中文）：
```
思考: [详细的数学推导步骤]
行动: [工具名称] 或 [无] 或 [最终答案]
行动输入: [JSON 格式的工具参数] 或 [空]
```

---

### 3.3 路由层

#### `router.py` — QuestionRouter

基于**规则 + 结构特征**的题目分类器，将题目映射到四个学科之一。

**分类规则**：

| 学科 | 关键词（英文） | 关键词（中文） |
|------|---------------|---------------|
| physics | force, velocity, acceleration, mass, energy, momentum | 力, 速度, 加速度, 质量, 能量, 动量 |
| circuits | circuit, resistor, voltage, current, ohm | 电路, 电阻, 电压, 电流, 串联, 并联, 欧姆 |
| linalg | matrix, vector, eigen, determinant, rank | 矩阵, 向量, 特征值, 行列式, 秩, 逆 |
| calculus | derivative, integral, limit, differentiation | 导数, 积分, 极限, 微分, 泰勒 |

**结构增强（Structural Boosts）**：
- 检测到 `[[1,2],[3,4]]` 矩阵字面量 → linalg 分数 +3
- 检测到 `|1 2; 3 4|` 行列式标记 → linalg 分数 +3
- 检测到方程组多行模式 → linalg 分数 +2
- 检测到 `A^{2025}` 矩阵幂标记 → linalg 分数 +2
- 强信号关键词（如 "行列式"、"求导数"、"串联"）→ 对应学科 +5

**冲突处理**：
当两个学科得分相同（平局）时，按优先级排序：`circuits > physics > calculus > linalg`

**输出**：`RouteDecision(subject, confidence, matched_rules)`

---

#### `smart_router.py`

已弃用（DEPRECATED）。原先计划实现基于 LLM 的智能路由，但发现规则路由已足够高效（0.012ms/题），且分析阶段本身就会调用 LLM，因此 LLM 路由是冗余的。现仅作为 `QuestionRouter` 的别名保留，确保向后兼容。

---

### 3.4 工具调度层

#### `tool_dispatcher.py` — ToolDispatcher

独立的工具调度层，将 `agent.py` 中紧密耦合的工具执行逻辑解耦出来。

**职责**：
- 根据 `AnalyzeResult.subject` 选择对应工具
- 从题目文本中提取结构化参数（表达式、矩阵、电阻值、物理量）
- 调用工具方法并包装为统一的结果格式
- 错误处理：工具不存在、参数提取失败、工具执行异常

**结果格式**：
```python
{
    "tool_name": "工具名",
    "success": True/False,
    "output": "工具输出（字符串）",
    "metadata": {"operation": "具体操作", ...},
    "error_message": "错误信息（若失败）"
}
```

**学科分发逻辑**：
- **calculus**：检测操作类型（derivative/integral/limit/critical/taylor）→ 提取表达式、变量、上下限、极限点 → 调用 CalculusTool
- **linalg**：检测操作类型（determinant/inverse/rank/eigen/power）→ 提取矩阵 → 调用 AlgebraTool
- **circuits**：检测拓扑结构（series/parallel）→ 提取电阻值/netlist/矩阵 → 调用 CircuitTool
- **physics**：推断物理关系（uniform_acceleration/newton_second_law/work_energy/momentum）→ 提取已知量 → 调用 PhysicsTool

---

### 3.5 工具执行层

工具层采用 **"SymPy 优先 + 纯 Python fallback"** 的双层架构。当环境中安装了 SymPy 时，所有符号运算由 SymPy 处理；未安装时，使用项目内置的纯 Python 数学工具完成多项式、矩阵、线性方程组等基础运算。

#### `tools/calculus_tool.py` — CalculusTool（微积分工具）

| 方法 | 功能 | SymPy | Fallback |
|------|------|-------|----------|
| `diff()` | 求导 | `sympy.diff()` | 多项式求导（power rule） |
| `integrate()` | 积分（定/不定） | `sympy.integrate()` | 多项式积分（power rule） |
| `limit()` | 极限 | `sympy.limit()` | 多项式直接代入 |
| `critical_points()` | 临界点 | `sympy.solve(f'(x)=0)` | 二次方程求根公式 |
| `taylor_series()` | 泰勒展开 | `sympy.series()` | 多项式在中心点展开 |
| `series_convergence_radius()` | 幂级数收敛半径 | 比值法 `sympy.limit()` | 不支持，需 SymPy |

**特殊处理**：
- 级数相关问题（收敛域、一致收敛）检测：若题目包含 "级数"、"series"、"收敛域"、"一致收敛" 等关键词，会进入 `_dispatch_series()` 特殊处理
- 一致收敛证明题：工具明确返回失败，提示需使用 LLM 模式
- 不定积分结果格式化：确保输出形如 `1/3*x**3` 的标准格式

---

#### `tools/algebra_tool.py` — AlgebraTool（线性代数工具）

| 方法 | 功能 | SymPy | Fallback |
|------|------|-------|----------|
| `determinant()` | 行列式 | `Matrix.det()` | 高斯消元法（Fraction 精确计算） |
| `matrix_inverse()` | 矩阵求逆 | `Matrix.inv()` | 增广矩阵高斯-约当消元 |
| `rank()` | 矩阵秩 | `Matrix.rank()` | 行阶梯形消元统计主元 |
| `eigenvalues()` | 特征值 | `Matrix.eigenvals()` | 仅支持 2×2 矩阵（特征方程） |
| `eigenvectors()` | 特征向量 | `Matrix.eigenvects()` | 仅支持 2×2 矩阵 |
| `matrix_power()` | 矩阵幂 | `Matrix ** n` | 快速幂 + 手动矩阵乘法 |
| `solve_linear_system()` | 线性方程组 | `sympy.solve()` | 高斯消元法（Fraction） |
| `linear_independence()` | 线性无关性 | `Matrix.rank()` | 比较秩与向量个数 |

**精确计算**：
Fallback 实现全程使用 `fractions.Fraction` 进行有理数运算，避免浮点误差，最后一步才转换为 `float`。

---

#### `tools/circuit_tool.py` — CircuitTool（电路分析工具）

| 方法 | 功能 | 实现方式 |
|------|------|----------|
| `equivalent_resistance()` | 串/并联等效电阻 | 直接公式计算 |
| `node_analysis()` | 节点电压分析 | 构建电导矩阵 + KCL 方程，高斯消元求解 |
| `mesh_analysis()` | 网孔电流分析 | 构建电阻矩阵 + 电压源向量，LU 求解 |
| `first_order_response()` | 一阶 RC/RL 瞬态响应 | 三要素法公式 |

**节点分析实现细节**：
- 解析 netlist（JSON 格式，包含 nodes 和 components）
- components 支持 resistor、current_source、voltage_source
- 电导矩阵组装：`_stamp_conductance()` 标准 stamps
- 电压源处理：接地端直接固定电位，浮空节点使用大电阻近似（1e-9 Ω）

---

#### `tools/physics_tool.py` — PhysicsTool（物理求解工具）

基于**结构化物理关系**的快速求解器，非通用 CAS，而是针对经典力学高频考点的专用实现。

| 关系 | 公式 | 可求目标 |
|------|------|----------|
| `uniform_acceleration` | 匀变速直线运动公式 | v, s, a, t |
| `newton_second_law` | F = ma | F, m, a |
| `work_energy` | W = ΔK = ½m(v²-v₀²) | W, F |
| `momentum` | p = mv, J = FΔt = Δp | p, v, impulse |

**求解器特点**：
- 自动根据已知量和目标量选择合适公式
- 求时间时涉及二次方程，自动求解并取非负最小根
- 功-能关系中支持两种路径：动能变化或直接做功（W = F·s·cosθ）

---

#### `tools/unit_tool.py` — UnitTool（单位检查工具）

轻量级的单位/量纲检查器，目前主要用于推理文本中的单位识别。

- 支持量纲：m, s, kg, A, V, Ω, N, J, W, C, Hz
- 可检查两个单位是否量纲兼容
- 从 reasoning_process 中提取已知/未知单位 token

---

#### `tools/_math_support.py` — 纯 Python 数学工具库

当 **SymPy 未安装**时的完整 fallback 实现，包含：

**多项式运算**：
- `poly_from_ast()`：通过 Python `ast` 模块解析表达式字符串为多项式字典 `{power: Fraction(coeff)}`
- `poly_add()`、`poly_mul()`、`poly_pow()`、`poly_div_const()`
- `poly_diff()`、`poly_integral()`、`poly_eval()`、`poly_to_string()`
- 支持的表达式：加减乘除、整数幂、单变量

**矩阵运算**：
- `solve_linear_system()`：高斯消元（Fraction 精确）
- `determinant()`：高斯消元求行列式
- `inverse()`：增广矩阵求逆
- `rank()`：行阶梯形求秩
- `eigenpairs_2x2()`：2×2 矩阵特征值/特征向量（解析公式）

**其他**：
- `quadratic_critical_points()`：二次函数临界点
- `taylor_series()`：多项式泰勒展开
- `evaluate_expression_at_point()`：求表达式在某点的值

**设计原则**：
- 所有不支持的情况显式抛出 `ToolUnsupportedError`，绝不编造结果
- 使用 `Fraction` 保持精确性，仅在最终输出时转 `float`

---

### 3.6 LLM 集成层

#### `llm/kimi_client.py` — KimiClient

基于 **Python 标准库 `urllib`** 的轻量级 LLM 客户端，**零外部依赖**。

**核心能力**：
- `chat()`：普通对话，返回字符串
- `chat_json()`：强制返回 JSON，支持从 Markdown 代码块中提取 JSON，验证必需字段

**配置方式**（优先级从高到低）：
1. 构造函数参数
2. 环境变量（`KIMI_API_KEY`, `KIMI_BASE_URL`, `KIMI_MODEL`, `KIMI_ENDPOINT_PATH`, `REQUEST_TIMEOUT_SECONDS`, `MAX_RETRY`, `KIMI_TEMPERATURE`）

**默认配置**：
- 模型：`kimi`
- 端点：`/v1/chat/completions`
- 超时：30 秒
- 重试：1 次（共 2 次尝试）
- 温度：1.0

**错误处理**：
- 网络错误、超时、JSON 解析错误均会重试
- 空响应、缺失字段会抛出清晰异常

---

#### `llm/prompt_builder.py` — PromptBuilder

两阶段提示词构建器：分析阶段 + 起草阶段。

**分析提示词（analyze）**：
- 要求 LLM 输出严格 JSON（无 Markdown、无散文）
- 必需字段：subject, topic, knowns, unknowns, equations_or_theorems, should_use_tool, target_form, possible_traps
- 按学科注入不同的 focus、tool_hint、draft_hint

**起草提示词（draft）**：
- 要求 LLM 输出 strict JSON，包含 reasoning_process 和 answer
- 风格规则：先写已知/未知，再写公式/定理，展示代入/推导步骤，最后给出结果和单位
- 可注入 retrieval_context（检索到的公式卡片和例题）

**学科模板（SUBJECT_TEMPLATES）**：
为 physics、circuits、linalg、calculus 分别定制了 focus 和 draft_hint，确保 LLM 按该学科的标准格式输出。

---

### 3.7 知识检索层

#### `retrieval/retriever.py` — Retriever

轻量级**关键词检索**系统，无需外部向量数据库。

**检索原理**：
- 将查询文本分词为 lowercase token 集合
- 与公式卡片/例题的文本字段和关键词字段进行交集匹配
-  scoring = `query_tokens ∩ card_tokens * 2 + query_tokens ∩ keywords * 3`
- 若提供了 topic，额外增加 topic 匹配加分
- 按分数降序排列，取前 `top_k` 个

**预计算优化**：
初始化时预计算所有卡片的 token 集合，避免每次查询重复分词。

**数据来源**：
- `formula_cards.json`：35 张公式卡片（覆盖 4 个学科的核心公式、适用条件、常见陷阱）
- `solved_examples.jsonl`：12 道例题（含 question, reasoning_process, answer, tags）

---

#### `retrieval/kb_loader.py` — KnowledgeBaseLoader

本地知识库加载器，支持 `.json` 和 `.jsonl` 格式。

- JSON：支持根级数组或包含 `items`/`questions` 字段的对象
- JSONL：逐行解析，跳过空行，提供清晰的行号错误报告
- 所有记录强制验证为 `dict` 类型

---

### 3.8 数据模型层

#### `schemas.py`

使用 Pydantic 风格的数据模型定义，通过 `compat/pydantic_compat.py` 兼容无 Pydantic 环境。

| 模型 | 用途 |
|------|------|
| `QuestionInput` | 输入题目结构 |
| `ParsedQuestion` | 解析后的题目结构 |
| `AnalyzeResult` | LLM 分析结果（学科、知识点、已知、未知、公式、陷阱） |
| `DraftResult` | LLM 起草结果（推理过程、答案） |
| `ToolResult` | 工具执行结果 |
| `FinalAnswer` | 最终提交格式 |
| `FormulaCard` | 公式卡片模型 |
| `SolvedExample` | 例题模型 |
| `RetrievalResult` | 检索结果 |

---

### 3.9 评估与测试层

#### `eval/local_eval.py` — 本地评估工具

用于在开发集上评估智能体性能。

**功能**：
- `load_dev_set()`：加载开发集（支持 `.questions`、`.items`、根级数组）
- `run_local_eval()`：运行完整评估循环，生成预测和报告
- `evaluate_dev_set()`：统计各项指标（总题数、答题数、精确匹配数、fallback 比例、分学科准确率）
- `compare_answers()`：智能答案对比（精确匹配 + 数值容差匹配，默认 tolerance=1e-6）

**报告指标**：
- total_count / answered_count / exact_match_count
- exact_match_accuracy
- per_subject accuracy
- failure_count / fallback_like_count
- average_answer_length / average_reasoning_process_length

---

#### `scripts/mini_pytest.py` — 微型测试运行器

零依赖的 pytest 兼容测试运行器。

- 自动发现 `tests/test_*.py` 文件
- 执行以 `test_` 开头的函数
- 输出通过/失败统计
- 支持 `-q` / `--quiet` 安静模式

#### `scripts/smoke_test.py` — 冒烟测试

快速验证智能体基本功能的脚本，执行单题求解和批量求解示例。

---

### 3.10 兼容层

#### `compat/pydantic_compat.py`

当环境中**未安装 Pydantic** 时的最小兼容实现。

**提供的类**：
- `BaseModel`：支持 `__init__`、`model_dump()` / `dict()`、`model_validate()`、类型验证、默认值/工厂、`Optional`/`Literal`/`list`/`dict`
- `Field`：字段默认值和默认工厂
- `ValidationError`：验证失败异常

**类型验证能力**：
- `str`, `int`, `float`, `bool`, `Any`, `Optional[T]`, `Literal[...]`
- `list[T]`, `dict[str, Any]`
- 拒绝 `bool` 作为 `int`/`float` 输入（防止类型混淆）

---

### 3.11 配置与辅助模块

#### `config.py` — Settings

从环境变量加载配置：
- `KIMI_API_KEY`：LLM API 密钥
- `ENG_SOLVER_DEFAULT_ROUTE`：默认路由（默认 general）
- `ENG_SOLVER_RETRIEVAL_ENABLED`：是否启用知识检索（默认 false）

自动加载项目根目录的 `.env` 文件（若安装了 `python-dotenv`）。

#### `adapter.py` — QuestionAdapter

输入适配器，标准化原始竞赛题目：
- 统一 `question_id`（支持 `id` 作为备选）
- 统一 `prompt`（支持 `question` 作为备选）
- 保留原始字典的所有其他字段

#### `formatter.py` — 格式化器

将求解结果格式化为竞赛标准 JSON：
```python
{
    "question_id": "...",
    "reasoning_process": "...",
    "answer": "..."
}
```
- 确保 reasoning_process 和 answer 非空
- 使用 `FinalAnswer` 模型验证

#### `verifier.py` — 验证器

最终提交项验证：
- 检查必需字段：`question_id`, `reasoning_process`, `answer`
- 检查字段非空

---

## 四、高级架构设计

### 4.1 多智能体并行系统（已弃用）

`multi_agent_system.py` 实现了完整的 **多智能体并行处理架构**：

- **WorkerAgent**：工作智能体，负责单题求解
- **CheckerAgent**：检查智能体，验证答案质量（LLM 检查 + 规则检查）
- **MultiAgentOrchestrator**：编排器，管理并行 worker 和检查-重试流程

**处理流程**：
1. Phase 1：ThreadPoolExecutor 并行求解所有题目
2. Phase 2：CheckerAgent 检查每道题的答案有效性
3. Phase 3：对失败/低置信度答案进行重试（最多 max_retries 次）

**弃用原因**：`UnifiedAgent.async_solve()` 基于 asyncio 的实现更简单、更高效，已完全替代此系统。`multi_agent_system_v2.py` 已标记为 DEPRECATED。

---

### 4.2 求解模式深度对比

#### `auto` 模式（默认）

```python
def _solve_auto(self, question, subject):
    if self._has_llm():
        return self._solve_react(question, subject)
    return self._solve_legacy(question, subject)
```

- **逻辑**：检测 LLM 客户端是否可用（配置了 base_url）
- **可用**：走 ReAct 推理（最大化推理质量分）
- **不可用**：走 legacy 流水线（工具计算 + fallback 草稿）
- **适用**：绝大多数场景，特别是竞赛提交

#### `react` 模式

- 强制使用 ReActEngine 进行 Think → Act → Observe 循环
- 最多 8 步，每步调用 LLM
- 工具调用嵌入推理链，可自我纠错
- **适用**：需要展示详细推导过程的题目，最有利于 "reasoning_process" 评分

#### `legacy` 模式

- 经典的两阶段流水线：analyze → tool → draft
- LLM 分析题目结构 → 工具精确计算 → LLM 组织答案
- **适用**：结构化数据充分的数值计算题

#### `llm_only` 模式

- 完全绕过工具层，直接让 LLM 生成 reasoning_process 和 answer
- **适用**：证明题、概念题、无法提取结构化参数的题目

#### `tool_only` 模式

- 完全绕过 LLM，仅使用工具层计算
- **速度**：<1ms/题，零 API 调用成本
- **适用**：纯数值计算、快速验证、LLM 不可用时的兜底

---

### 4.3 错误处理与降级策略

系统在多个层级实现了优雅的降级：

```
UnifiedAgent.solve_one()
    └── 异常捕获 → _error_result() → 返回包含错误信息的合规 JSON

_solve_react()
    └── ReAct 失败 → 自动降级到 _solve_legacy()

_analyze() / _draft()
    └── LLM 调用失败 → fallback_analyze / _build_fallback_draft()

_tool_dispatcher.dispatch()
    └── 工具不存在/执行失败 → 返回结构化错误结果（不影响后续 draft）

KimiClient.chat_json()
    └── 网络/解析/字段缺失错误 → 重试（默认 1 次）→ 最终抛出异常由上层捕获
```

**Fallback 草稿生成策略**：
即使工具完全失败，仍基于已知的 analysis 信息生成结构化推理文本，包含：
- 题目复述
- 已知条件 / 求解目标
- 相关公式
- 工具失败原因
- 结论（承认当前无法精确求解，但展示正确思路）

这让系统在完全失败时仍能获得**部分推理分数**。

---

## 五、性能优化

### 5.1 已实现的优化

| 组件 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| Retriever | 0.88ms/题 | 0.39ms/题 | **55%** |
| Router | 0.013ms/题 | 0.012ms/题 | ~10% |
| Agent batch (10题) | ~750ms | ~700ms | ~7% |

**优化手段**：
- 检索器预计算 token 集合，避免重复正则分词
- 路由层预编译所有正则表达式，模块级复用
- 智能体层减少不必要的字典拷贝和字段查找

### 5.2 并行求解

```python
async def async_solve(self, questions, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def _task(q):
        async with semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.solve_one, q)
    
    tasks = [asyncio.create_task(_task(q)) for q in questions]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

- 使用 `asyncio.Semaphore` 控制并发，避免 API 限流
- `run_in_executor` 将同步的 `solve_one` 包装为异步任务
- `return_exceptions=True` 确保单题异常不影响整体批次

---

## 六、中文环境优化

本项目针对中文竞赛题目进行了大量专门优化：

### 6.1 矩阵字面量解析

支持三种中文/混合格式：
- `[[1,2],[3,4]]` — JSON 风格
- `|1 2; 3 4|` — 行列式标记（使用中文分号 `；` 也可识别）
- `[1 2; 3 4]` — MATLAB 风格

### 6.2 数学表达式提取

中英文混合模式：
- "求 x^2 的导数" → 提取 `x^2`
- "计算 x^3 从 0 到 3 的积分" → 提取 `x^3`, bounds `[0, 3]`
- "求 sin(x) 当 x->0 的极限" → 提取 `sin(x)`, point `0`
- "derivative of x^2" → 提取 `x^2`

### 6.3 物理量提取

从中文题干中自动识别：
- `2kg` / `2 千克` → `{"m": 2.0}`
- `3m/s` / `3 米/秒` → `{"v": 3.0}`
- `10N` / `10 牛顿` → `{"F": 10.0}`

### 6.4 电路参数提取

- `2Ω` / `2 欧姆` / `2ohm` → 电阻值列表 `[2.0]`
- "串联" / "并联" → 拓扑结构 `series` / `parallel`

---

## 七、快速开始

### 7.1 安装

```bash
# 克隆仓库
git clone <repo-url>
cd Engineering-Problem-Solving-Agent

# 安装依赖（核心代码零依赖，以下为可选增强）
pip install sympy          # 启用完整符号计算
pip install python-dotenv  # 支持 .env 文件加载
pip install pydantic       # 使用真实 Pydantic（替代兼容层）
```

### 7.2 配置

创建 `.env` 文件：
```bash
KIMI_BASE_URL=https://api.moonshot.cn/v1
KIMI_API_KEY=your_api_key_here
KIMI_MODEL=kimi
```

或仅设置环境变量：
```bash
export KIMI_BASE_URL=https://api.moonshot.cn/v1
export KIMI_API_KEY=your_api_key_here
```

### 7.3 运行测试

```bash
# 运行所有单元测试
python scripts/mini_pytest.py

# 冒烟测试（验证基本功能）
python scripts/smoke_test.py

# 本地评估（需要 LLM 配置）
python scripts/run_local_eval.py --dev-path data/dev/dev.json
```

### 7.4 使用示例

**Python API**：

```python
from eng_solver_agent import UnifiedAgent
import asyncio

agent = UnifiedAgent()

# 单题求解
question = {
    "question_id": "demo-1",
    "question": "Find the derivative of x^2.",
    "expression": "x^2",
    "subject": "calculus"
}
result = agent.solve_one(question, mode="auto")
print(result)
# 输出: {"question_id": "demo-1", "reasoning_process": "...", "answer": "2*x"}

# 并行批量求解（竞赛推荐）
questions = [question1, question2, question3, ...]
results = asyncio.run(agent.async_solve(questions, max_concurrent=5, mode="auto"))
```

**命令行 - 求解单个数据集**：

```bash
python solve_dataset.py \
    --input data/dev/dev.json \
    --output results.json \
    --mode auto \
    --max-concurrent 5
```

**命令行 - 求解全部四门课程**（竞赛模式）：

```bash
python competition_run.py
```

**Tool-only 模式（最快，无 API 调用）**：

```bash
python solve_dataset.py \
    --input data/dev/dev.json \
    --output results.json \
    --mode tool_only \
    --kimi-client none
```

---

## 八、项目文件结构

```
Engineering-Problem-Solving-Agent/
│
├── eng_solver_agent/              # 核心智能体包
│   ├── __init__.py                # 包入口，导出 UnifiedAgent
│   ├── unified_agent.py           # 统一入口（5种模式 + 并行）⭐
│   ├── agent.py                   # 原始智能体（analyze→tool→draft）
│   ├── agent_v2.py                # 增强版（LLM优先直接求解）
│   ├── competition_agent.py       # 竞赛专用（ReAct推理）
│   ├── multi_agent_system.py      # 多智能体并行（已弃用）
│   ├── multi_agent_system_v2.py   # 多智能体v2（已弃用）
│   │
│   ├── reasoning_engine.py        # ReAct推理引擎（Think→Act→Observe）
│   ├── router.py                  # 规则路由（学科分类）
│   ├── smart_router.py            # 智能路由别名（已合并到router）
│   ├── tool_dispatcher.py         # 工具调度层
│   ├── adapter.py                 # 输入适配器
│   ├── formatter.py               # 输出格式化器
│   ├── verifier.py                # 提交验证器
│   ├── schemas.py                 # Pydantic数据模型
│   ├── config.py                  # 配置管理
│   │
│   ├── llm/                       # LLM集成
│   │   ├── __init__.py
│   │   ├── kimi_client.py         # Kimi API客户端（urllib实现）
│   │   └── prompt_builder.py      # 两阶段提示词构建器
│   │
│   ├── tools/                     # 工具执行层
│   │   ├── __init__.py
│   │   ├── calculus_tool.py       # 微积分工具（diff/integrate/limit/...）
│   │   ├── algebra_tool.py        # 线性代数工具（det/inv/rank/eigen/...）
│   │   ├── circuit_tool.py        # 电路分析工具（等效电阻/节点分析/...）
│   │   ├── physics_tool.py        # 物理求解工具（运动学/牛顿定律/能量/动量）
│   │   ├── unit_tool.py           # 单位检查工具
│   │   └── _math_support.py       # 纯Python数学库（SymPy fallback）
│   │
│   ├── retrieval/                 # 知识检索
│   │   ├── __init__.py
│   │   ├── retriever.py           # 关键词检索器
│   │   ├── kb_loader.py           # 知识库加载器（JSON/JSONL）
│   │   ├── formula_cards.json     # 35张公式卡片
│   │   └── solved_examples.jsonl  # 12道例题
│   │
│   ├── eval/                      # 本地评估
│   │   ├── __init__.py
│   │   └── local_eval.py          # 开发集评估、答案对比、报告生成
│   │
│   └── compat/                    # 兼容层
│       ├── __init__.py
│       └── pydantic_compat.py     # 无Pydantic时的最小兼容实现
│
├── scripts/                       # 脚本工具
│   ├── mini_pytest.py             # 微型测试运行器
│   ├── smoke_test.py              # 冒烟测试
│   └── run_local_eval.py          # 本地评估入口
│
├── tests/                         # 测试目录
│   ├── __init__.py
│   └── _helpers.py                # 测试辅助函数
│
├── data/                          # 数据集
│   ├── dev/
│   │   ├── dev.json               # 开发集（20题，覆盖4学科）
│   │   └── hard_test.json         # 困难测试集（12题，含推导证明）
│   ├── calculus/
│   │   └── teset.json             # 微积分级数专题（10题）
│   └── exports/
│       └── .gitkeep               # 评估结果输出目录
│
├── solve_dataset.py               # 批量求解命令行工具
├── competition_run.py             # 竞赛运行入口（求解4门课程）
├── submission.json                # 竞赛提交配置
│
├── requirements.txt               # 依赖说明（核心零依赖）
├── pytest.bat                     # Windows测试运行脚本
├── .env.example                   # 环境变量示例
├── .gitignore                     # Git忽略规则
└── README.md                      # 本文件
```

---

## 九、竞赛提交配置

`submission.json`：
```json
{
  "module": "eng_solver_agent.unified_agent",
  "class_name": "UnifiedAgent",
  "method_name": "solve"
}
```

竞赛系统会通过此配置动态导入并调用 `UnifiedAgent.solve(questions)` 方法。

---

## 十、依赖说明

### 核心运行（零外部依赖）

项目主体仅依赖 **Python 标准库**：
- `asyncio`, `json`, `re`, `os`, `sys`, `time`, `math`, `cmath`
- `ast`, `fractions`, `dataclasses`, `pathlib`, `typing`, `urllib`
- `collections`, `concurrent.futures`, `importlib`, `inspect`, `traceback`, `types`

### 可选增强

| 包 | 用途 | 无此包时的行为 |
|----|------|---------------|
| `sympy` | 符号计算 | 使用纯 Python `_math_support` fallback |
| `pydantic` | 数据验证 | 使用 `compat/pydantic_compat` shim |
| `python-dotenv` | `.env` 文件加载 | 直接读取系统环境变量 |

### 环境要求

- **Python 3.10+**（使用 `list[dict[str, Any]]` 等新类型语法）
- 网络连接（仅 LLM 模式需要）

---

## 十一、设计决策记录

### 11.1 为什么使用 urllib 而不是 requests？

- 避免引入外部 HTTP 库依赖
- 竞赛环境可能无法 pip install
- urllib 足以处理标准的 OpenAI-compatible API 格式

### 11.2 为什么弃用多智能体系统？

- `asyncio.Semaphore + run_in_executor` 足以实现高效并行
- 多智能体的 Checker + Retry 逻辑增加了复杂度，但边际收益有限
- 竞赛评分更看重单题质量而非多轮验证

### 11.3 为什么保留 agent.py / agent_v2.py / competition_agent.py？

- 向后兼容：早期代码可能直接导入这些类
- 功能细分：不同场景下直接实例化特定智能体更明确
- UnifiedAgent 内部实际上也复用了这些类的核心逻辑

### 11.4 为什么使用 Fraction 而不是 float？

- 矩阵运算、线性方程组求解中，浮点误差会导致错误结果（如行列式接近零时）
- Fraction 保持精确有理数运算，仅在最后一步转换为 float
- 这是数值计算库（如 NumPy/SymPy）的标准做法

### 11.5 为什么公式卡片和例题使用本地 JSON 而不是向量数据库？

- 竞赛环境限制，无法保证有外部服务
- 题目数量为百级，关键词检索（0.39ms）已足够快
- 本地文件零依赖、零配置、可版本控制

---

## 十二、扩展指南

### 12.1 添加新学科

1. 在 `QuestionRouter._RULES` 中添加新学科的关键词
2. 在 `ToolDispatcher._do_dispatch()` 中添加新学科的分发逻辑
3. 在 `tools/` 下新建工具类（如 `thermodynamics_tool.py`）
4. 在 `UnifiedAgent._build_default_tools()` 中注册新工具
5. 在 `schemas.py` 的 `Literal` 类型中添加新学科名

### 12.2 添加新工具方法

以 `CalculusTool` 为例：
1. 在 `CalculusTool` 类中实现新方法（优先 SymPy，fallback 纯 Python）
2. 在 `ToolDispatcher._dispatch_calculus()` 中添加调用逻辑
3. 在 `reasoning_engine.py` 的 `_describe_tools()` 中会自动暴露新方法

### 12.3 接入其他 LLM

`KimiClient` 设计为可替换：
1. 实现相同接口的客户端类：`chat(messages, temperature)` → `str`，`chat_json(...)` → `dict`
2. 实例化 `UnifiedAgent(kimi_client=YourClient())`

或通过环境变量调整 endpoint：
```bash
KIMI_BASE_URL=https://api.openai.com/v1
KIMI_API_KEY=sk-...
KIMI_MODEL=gpt-4
```

---

## 十三、许可证

MIT License
