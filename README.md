# Engineering-Problem-Solving-Agent

竞赛级智能体，用于求解工科基础课程问题（微积分、线性代数、电路、物理）。

本项目基于 **LLM 推理 + 精确工具计算** 的双引擎架构，支持 **ReAct（思考-行动-观察）** 多轮推理循环，针对中文竞赛环境进行了大量专门优化。

---

## 目录

1. [项目概述](#一项目概述)
2. [架构总览](#二架构总览)
3. [模块详解](#三模块详解)
4. [新增功能详解](#四新增功能详解)
5. [求解模式对比](#五求解模式对比)
6. [快速开始](#六快速开始)
7. [项目文件结构](#七项目文件结构)
8. [竞赛提交配置](#八竞赛提交配置)
9. [依赖说明](#九依赖说明)
10. [设计决策记录](#十设计决策记录)
11. [扩展指南](#十一扩展指南)

---

## 一、项目概述

### 1.1 目标

本项目为"未央城"AI Agent 竞赛（工程问题求解赛道）而构建，目标是通过自动化方式求解工科基础课程（微积分、线性代数、电路、物理）的数学和物理问题。

竞赛评分维度：
- **reasoning_process**（推理过程）：展示逐步推导，即使答案错误也能获得公式分
- **answer**（答案）：最终数值或表达式结果

### 1.2 核心能力

| 能力 | 说明 |
|------|------|
| 微积分 | 求导、积分（定/不定）、极限、临界点、泰勒展开、幂级数收敛半径 |
| 线性代数 | 行列式、矩阵求逆、秩、特征值/特征向量、矩阵幂、线性方程组、线性无关性 |
| 电路分析 | 串并联等效电阻、节点电压分析、网孔电流分析、一阶 RC/RL 瞬态响应 |
| 物理 | 匀变速直线运动、牛顿第二定律、功能关系、动量/冲量 |
| 相似题目查找 | 基于语义向量检索 + 关键词匹配，查找知识库中的相似例题和公式 |
| 代码执行沙箱 | 使用 `exec` 执行 Python/SymPy 代码进行任意符号/数值计算 |

### 1.3 双引擎架构

```
┌─────────────────────────────────────────┐
│           用户输入题目                    │
│  (中文/英文, 支持矩阵/表达式/物理量提取)   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         QuestionRouter (题目路由器)       │
│  关键词 + 结构特征 → physics/circuits/    │
│                    linalg/calculus       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      UnifiedAgent (统一求解入口)          │
│  5种模式: auto / react / legacy /         │
│           llm_only / tool_only           │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌──────────────┐   ┌──────────────────┐
│   ReAct引擎   │   │    Legacy流水线   │
│ Think→Act→   │   │ 分析→工具→起草    │
│ Observe循环  │   │                  │
└──────┬───────┘   └────────┬─────────┘
       │                    │
       └────────┬───────────┘
                ▼
┌─────────────────────────────────────────┐
│          工具层 (统一计算工具)            │
│  ┌──────────┐  ┌──────────────────┐    │
│  │ LLM推理  │  │ 精确计算 (SymPy)  │    │
│  │ (Kimi)   │  │ + 纯Python fallback│   │
│  └──────────┘  └──────────────────┘    │
│              ┌──────────────────┐       │
│              │ 语义向量检索 (FAISS)│      │
│              │ + 相似题目匹配      │      │
│              └──────────────────┘       │
└─────────────────────────────────────────┘
```

---

## 二、架构总览

### 2.1 数据流

```
用户题目
    │
    ├──→ QuestionAdapter.normalize() → 标准化字段
    │
    ├──→ QuestionRouter.route() → 确定学科
    │
    ├──→ LangChainRetriever.retrieve() → 检索相似例题/公式
    │
    └──→ UnifiedAgent.solve_one(mode)
            │
            ├── mode="react" → ReActEngine.solve()
            │       ├── 第1步: LLM输出 "思考+行动+行动输入"
            │       ├── 第2步: 执行工具 → 观察结果 → 反馈给LLM
            │       ├── ...
            │       └── 第N步: 得到最终答案
            │
            ├── mode="legacy" → 三阶段流水线
            │       ├── _analyze() → LLM分析题目结构
            │       ├── ToolDispatcher.dispatch() → 精确计算
            │       └── _draft() → LLM组织答案
            │
            ├── mode="llm_only" → LLM直接输出答案
            │
            └── mode="tool_only" → 纯工具计算（<1ms）
                    │
                    └── NumericalComputationTool
                            ├── _CalculusEngine (微积分)
                            ├── _AlgebraEngine (线性代数)
                            ├── _CircuitEngine (电路分析)
                            ├── _PhysicsEngine (物理)
                            └── _ExecEngine (代码沙箱)
```

### 2.2 求解模式深度对比

| 模式 | LLM | 工具 | 检索 | 速度 | 适用场景 |
|------|-----|------|------|------|----------|
| `auto` | 自适应 | 自适应 | 自适应 | 中等 | **默认推荐**，自动选择最优策略 |
| `react` | 多轮 | 嵌入推理链 | 可选 | 较慢 | 需要详细逐步推理，最大化 reasoning_process 分数 |
| `legacy` | 两阶段 | 独立调用 | 可选 | 中等 | 结构化数据充分的数值计算题 |
| `llm_only` | 直接 | 不使用 | 可选 | 中等 | 证明题、概念题、符号推导 |
| `tool_only` | 不使用 | 直接 | 不使用 | <1ms/题 | 纯数值计算、快速验证、零API成本 |

---

## 三、模块详解

### 3.1 统一入口层

#### `unified_agent.py` — UnifiedAgent（推荐入口）

项目的**核心统一入口**，合并了早期分散在 `agent.py`、`agent_v2.py`、`competition_agent.py` 中的功能。

**核心 API**：

```python
from eng_solver_agent import UnifiedAgent
import asyncio

agent = UnifiedAgent()

# 单题求解
result = agent.solve_one(question, mode="auto")

# 顺序批量求解
results = agent.solve(questions, mode="auto")

# 并行批量求解（竞赛推荐）
results = asyncio.run(agent.async_solve(questions, max_concurrent=5, mode="auto"))
```

**五种求解模式**：详见 [2.2 求解模式深度对比](#22-求解模式深度对比)。

**构造函数参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `settings` | `Settings` | `None` | 配置对象，默认从环境变量加载 |
| `router` | `QuestionRouter` | `None` | 题目路由器 |
| `kimi_client` | `Any` | `None` | LLM 客户端，`None` 表示不启用 LLM |
| `retriever` | `Retriever` | `None` | 知识检索器 |
| `tool_registry` | `dict` | `None` | 工具注册表 |
| `default_mode` | `str` | `"auto"` | 默认求解模式 |

---

#### `agent.py` — EngineeringSolverAgent（原始智能体）

经典 **"分析 → 工具 → 起草"** 三阶段流水线。

**流水线阶段**：
1. **`_normalize_input()`**：通过 `QuestionAdapter` 标准化输入字段
2. **`_analyze_question()`**：调用 LLM 分析题目，提取结构化信息
3. **`_run_tool()`**：根据学科调用对应工具进行精确计算
4. **`_draft_answer()`**：LLM 结合分析结果和工具输出，生成中文推理过程

**文本提取能力**：
- **表达式提取**："求 x^2 的导数" → `x^2`
- **矩阵提取**：`[[1,2],[3,4]]`、`|1 2; 3 4|`、`[1 2; 3 4]`
- **电阻提取**："2Ω"、"3 欧姆" → `[2.0]`
- **物理量提取**：`2kg` → `{"m": 2.0}`

---

#### `competition_agent.py` — CompetitionAgent（竞赛专用）

针对**竞赛评分规则**优化：优先使用 ReAct 生成详细中文数学推导，即使答案错误也能拿到公式分。

---

### 3.2 推理引擎层

#### `reasoning_engine.py` — ReActEngine

实现 **ReAct（Reasoning + Acting）** 推理循环。

**ReAct 循环**：
```
初始化：构建 system prompt + 题目描述 + 可用工具列表

For step = 1..MAX_STEPS(8):
    1. LLM 输出：思考 + 行动 + 行动输入
    2. 若行动 == "最终答案" → 结束循环
    3. 若行动 == 工具名 → 执行工具 → 观察结果 → 反馈给 LLM
    4. 若行动 == "无" → 继续下一步
```

**工具调用协议（4 种方式）**：

| 方式 | 示例 | 说明 |
|------|------|------|
| `tool.method` 点语法 | `physics.newton_second_law` | 直接指定工具和方法 |
| 方法名快捷方式 | `diff` | 通过方法注册表自动查找所属工具 |
| 工具名 + method 字段 | `calculus` + `{"method": "integrate"}` | 原有方式 |
| compute/solve 特殊路由 | `compute` / `solve` | 自动路由到统一计算工具 |

LLM 输出格式：
```
思考: [详细的数学推导步骤]
行动: [工具名称] 或 [无] 或 [最终答案]
行动输入: [JSON 格式的工具参数]
```

---

### 3.3 路由层

#### `router.py` — QuestionRouter

基于**规则 + 结构特征**的题目分类器。

**分类规则**：

| 学科 | 关键词（英文） | 关键词（中文） |
|------|---------------|---------------|
| physics | force, velocity, acceleration, mass, energy | 力, 速度, 加速度, 质量, 能量, 动量 |
| circuits | circuit, resistor, voltage, current | 电路, 电阻, 电压, 电流, 串联, 并联 |
| linalg | matrix, vector, eigen, determinant, rank | 矩阵, 向量, 特征值, 行列式, 秩, 逆 |
| calculus | derivative, integral, limit | 导数, 积分, 极限, 微分, 泰勒 |

**结构增强**：矩阵字面量 (+3)、行列式标记 (+3)、强信号关键词 (+5)

**冲突处理**：平局时按 `circuits > physics > calculus > linalg` 优先级排序。

---

### 3.4 工具执行层

#### `tools/numerical_tool.py` — NumericalComputationTool（统一数值计算工具）

**核心设计**：将原先分散的 `CalculusTool`、`AlgebraTool`、`CircuitTool`、`PhysicsTool` 整合为单一工具类，内部通过 5 个引擎实现所有计算功能。

**引擎组成**：

| 引擎 | 来源 | 功能 |
|------|------|------|
| `_CalculusEngine` | 原 CalculusTool | 求导、积分、极限、临界点、泰勒展开、收敛半径 |
| `_AlgebraEngine` | 原 AlgebraTool | 行列式、矩阵求逆、秩、特征值/向量、矩阵幂、线性方程组 |
| `_CircuitEngine` | 原 CircuitTool | 等效电阻、节点分析、网孔分析、一阶 RC/RL 响应 |
| `_PhysicsEngine` | 原 PhysicsTool | 匀变速运动、牛顿定律、功能关系、动量冲量 |
| `_ExecEngine` | 新增 | 代码沙箱执行（exec + sympy + numpy） |

**统一调用接口**：

```python
from eng_solver_agent.tools import NumericalComputationTool

tool = NumericalComputationTool()

# 方式1: 直接调用具体方法
tool.diff("x**2", "x")              # → "2*x"
tool.determinant([[1,2],[3,4]])    # → -2.0

# 方式2: compute(operation, **kwargs)
tool.compute("diff", expression="x**3", var="x")  # → "3*x**2"

# 方式3: solve(query) — 自然语言或代码风格查询
tool.solve("diff(expression='x**2', var='x')")    # → "2*x"
tool.solve("求导数 x**2")                           # → "2*x"

# 方式4: execute_code(code) — 代码沙箱
result = tool.execute_code("""
import sympy as sp
x = sp.Symbol('x')
print(sp.integrate(sp.sin(x)**2, x))
""")
# → "x/2 - sin(x)*cos(x)/2"
```

**双层架构**：SymPy 优先 + `_math_support.py` 纯 Python fallback。

---

#### `tools/similarity_tool.py` — SimilarProblemTool（相似题目查找工具）

基于**语义向量检索 + 关键词匹配**的相似题目查找工具。

**核心功能**：

```python
from eng_solver_agent.tools import SimilarProblemTool

tool = SimilarProblemTool()

# 查找相似题目
result = tool.find_similar(
    query="求导数 x**2",
    subject="calculus",
    top_k=3
)
# 返回: matched_examples, matched_formulas, metadata

# 对比两道题目
tool.compare_questions("求导数 x**2", "计算积分 x**2")
# 返回: jaccard_similarity, structural_similarity, keyword_similarity, overall_similarity

# 计算匹配分数
tool.match_score(query, candidate_example)
# 返回: text_similarity, structural_similarity, keyword_similarity, subject_boost, topic_boost, overall_score
```

**相似度维度**：
- **文本相似度**：Jaccard 系数（token 重叠率）
- **结构相似度**：矩阵模式、方程模式、单位模式、问题类型（证明/计算）
- **关键词相似度**：数学术语匹配

---

#### `tools/_math_support.py` — 纯 Python 数学工具库

当 **SymPy 未安装**时的完整 fallback 实现。

**多项式运算**：`poly_from_ast()`（通过 Python `ast` 模块解析）、`poly_diff()`、`poly_integral()`、`poly_eval()`

**矩阵运算**：`solve_linear_system()`（高斯消元，Fraction 精确）、`determinant()`、`inverse()`、`rank()`、`eigenpairs_2x2()`

---

### 3.5 LLM 集成层

#### `llm/kimi_client.py` — KimiClient

基于 **Python 标准库 `urllib`** 的轻量级 LLM 客户端。

**核心能力**：
- `chat()`：普通对话
- `chat_json()`：强制返回 JSON，支持从 Markdown 代码块提取

**配置方式**（优先级从高到低）：
1. 构造函数参数
2. 环境变量（`KIMI_API_KEY`, `KIMI_BASE_URL`, `KIMI_MODEL`, `REQUEST_TIMEOUT_SECONDS`, `MAX_RETRY`, `KIMI_TEMPERATURE`）

**默认配置**：
- 模型：`kimi-k2.5`
- 端点：`/v1/chat/completions`
- 超时：30 秒
- 重试：1 次
- 温度：1.0

**URL 构建安全**：自动检测并避免路径段重复（如 `BASE_URL=https://api.moonshot.cn/v1` + `endpoint_path=/v1/chat/completions` → 自动去重为 `/v1/chat/completions`）。

---

### 3.6 知识检索层

#### `retrieval/langchain_retriever.py` — LangChainRetriever

基于 **LangChain + HuggingFaceEmbeddings + FAISS** 的语义向量检索系统。

**架构**：
- **嵌入模型**：`sentence-transformers/all-MiniLM-L6-v2`
- **向量存储**：FAISS（支持本地持久化 `save_local()` / `load_local()`）
- **混合打分**：向量相似度（70%）+ 关键词重叠（30%）
- **结构化过滤**：按学科（subject）和主题（topic）过滤

**使用方式**：

```python
from eng_solver_agent.retrieval import LangChainRetriever

retriever = LangChainRetriever(
    formula_cards_path="eng_solver_agent/retrieval/formula_cards.json",
    solved_examples_path="eng_solver_agent/retrieval/solved_examples.jsonl",
)

result = retriever.retrieve("求导数", subject="calculus", top_k=3)
# result.formula_cards, result.solved_examples

# 持久化索引
retriever.save_index("faiss_index/")

# 加载已有索引
retriever.load_index("faiss_index/")
```

**构建脚本**：
```bash
python build_kb.py
```

**降级策略**：当 LangChain 依赖不可用时，自动回退到 `Retriever`（关键词检索）。

---

#### `retrieval/retriever.py` — Retriever（关键词检索）

轻量级关键词检索系统，零外部依赖。

- 将查询文本分词为 lowercase token 集合
- scoring = `query_tokens ∩ card_tokens * 2 + query_tokens ∩ keywords * 3`
- 预计算优化：初始化时预计算所有卡片的 token 集合

---

### 3.7 调试与日志层

#### `debug_logger.py` — 调试日志系统

全链路的调试日志系统，通过环境变量 `AGENT_VERBOSE` 控制输出级别。

**级别**：
- `AGENT_VERBOSE=0`：静默模式
- `AGENT_VERBOSE=1`（默认）：完整输出（彩色终端）
- `AGENT_VERBOSE=2`：完整输出 + 原始 HTTP 负载

**功能**：
- **文件日志（Tee 模式）**：自动将所有 `stdout` 同时输出到终端和时间戳命名的 `.txt` 文件
- **彩色输出**：自动检测 TTY 和 `NO_COLOR` 环境变量
- **专用日志函数**：`section()`、`step()`、`log_llm_request/response/error()`、`log_route()`、`log_react_step/final()`、`log_pipeline_stage()`、`log_error()`

---

### 3.8 数据模型与基础设施层

#### `schemas.py` — 数据模型

Pydantic 风格的数据模型，通过 `compat/pydantic_compat.py` 兼容无 Pydantic 环境。

| 模型 | 用途 |
|------|------|
| `QuestionInput` | 输入题目结构 |
| `AnalyzeResult` | LLM 分析结果 |
| `DraftResult` | LLM 起草结果 |
| `ToolResult` | 工具执行结果 |
| `FinalAnswer` | 最终提交格式 |
| `RetrievalResult` | 检索结果 |
| `SimilarProblemResult` | 相似题目结果 |

---

#### `exceptions.py` — 统一异常层次

```
SolverError（基类）
├── ToolError
│   ├── ToolUnsupportedError
│   ├── ToolInputError
│   └── ToolTimeoutError
├── LLMError
│   ├── LLMResponseError
│   └── LLMJSONError
├── RetrievalError
│   └── IndexNotFoundError
└── RoutingError
```

---

#### `constants.py` — 全局常量

集中管理魔法数字：
- `MAX_REACT_STEPS = 8`
- `DEFAULT_MAX_CONCURRENT = 5`
- `NUMERIC_TOLERANCE = 1e-6`
- `DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`
- `DEFAULT_FAISS_INDEX_DIR = "faiss_index"`

---

### 3.9 评估与测试层

#### `eval/local_eval.py` — 本地评估工具

- `compare_answers()`：精确匹配 + 数值容差匹配（默认 tolerance=1e-6）
- `evaluate_dev_set()`：统计总题数、答题数、精确匹配数、分学科准确率

#### `scripts/mini_pytest.py` — 微型测试运行器

零依赖的 pytest 兼容测试运行器，自动发现 `tests/test_*.py`。

---

## 四、新增功能详解

### 4.1 统一数值计算工具

**背景**：原项目有 4 个独立的工具文件（`calculus_tool.py`、`algebra_tool.py`、`circuit_tool.py`、`physics_tool.py`），每个约 100-200 行，存在大量重复代码和命名不一致。

**改造**：将 4 个工具合并为 `NumericalComputationTool`，内部通过 5 个引擎实现：
- 保留全部原始方法（20+ 个公开方法）
- 新增统一入口 `compute()`、`solve()`、`execute_code()`
- 旧工具类作为内部引擎保留，确保行为完全一致
- 删除 4 个独立文件，减少维护负担

### 4.2 LangChain 向量检索系统

**背景**：原关键词检索器在语义理解上有局限，无法处理同义词和语义相似但字面不同的查询。

**改造**：
- 新增 `LangChainRetriever`，基于 HuggingFace `all-MiniLM-L6-v2` 嵌入模型
- FAISS 本地向量索引，支持持久化和增量加载
- 混合打分：向量相似度 70% + 关键词重叠 30%
- 自动降级：LangChain 不可用时回退到关键词检索
- 新增 `build_kb.py` 一键构建索引脚本

### 4.3 相似题目查找工具

**背景**：竞赛中遇到陌生题目时，查找知识库中的相似例题可以帮助 LLM 生成更准确的推理过程。

**功能**：
- `find_similar()`：语义检索 + 结构化过滤
- `compare_questions()`：多维度相似度对比
- `match_score()`：加权综合匹配分数

### 4.4 代码执行沙箱

**背景**：部分复杂计算（如多元积分、矩阵特征多项式）难以通过预定义方法覆盖。

**实现**：`_ExecEngine` 使用 `exec()` 在受限命名空间中执行 Python/SymPy 代码：
```python
tool.execute_code("""
import sympy as sp
x, y = sp.symbols('x y')
print(sp.integrate(sp.exp(x)*sp.sin(y), (x, 0, 1), (y, 0, sp.pi)))
""")
```

**安全措施**：
- 仅暴露 `math`、`numpy`、`sympy` 等安全模块
- `__builtins__` 白名单（仅允许 `abs`、`float`、`range` 等基础函数）
- 输出通过 `StringIO` 捕获，无文件系统访问

### 4.5 调试日志系统

**背景**：早期排查问题时，缺乏 LLM 请求/响应、工具调用、ReAct 步骤的可见性。

**实现**：
- 模块级日志，通过 `AGENT_VERBOSE` 环境变量控制
- 自动 Tee 到文件（`agent_run_YYYYMMDD_HHMMSS.txt`）
- 彩色终端输出 + 纯文本文件输出

---

## 五、求解模式对比

### `auto` 模式（默认）

```python
def _solve_auto(self, question, subject):
    if self._has_llm():
        return self._solve_react(question, subject)
    return self._solve_legacy(question, subject)
```

- 检测 LLM 客户端是否可用
- 可用：走 ReAct 推理（最大化推理质量）
- 不可用：走 legacy 流水线（工具计算 + fallback 草稿）

### `react` 模式

- 强制使用 ReActEngine 进行 Think → Act → Observe 循环
- 最多 8 步，每步调用 LLM
- 工具调用嵌入推理链，可自我纠错
- **适用**：需要展示详细推导过程的题目

### `legacy` 模式

- 经典两阶段流水线：analyze → tool → draft
- LLM 分析题目结构 → 工具精确计算 → LLM 组织答案

### `llm_only` 模式

- 完全绕过工具层，直接让 LLM 生成答案
- **适用**：证明题、概念题

### `tool_only` 模式

- 完全绕过 LLM，仅使用工具层计算
- **速度**：<1ms/题，零 API 调用成本

---

## 六、快速开始

### 6.1 安装

```bash
# 克隆仓库
git clone <repo-url>
cd Engineering-Problem-Solving-Agent

# 安装所有依赖（推荐）
pip install -r requirements.txt

# 或仅安装核心依赖（项目主体零外部依赖）
pip install sympy python-dotenv

# 构建 FAISS 向量索引（可选，用于语义检索）
python build_kb.py
```

### 6.2 配置

创建 `.env` 文件：
```bash
KIMI_BASE_URL=https://api.moonshot.cn
KIMI_API_KEY=your_api_key_here
KIMI_MODEL=kimi-k2.5
KIMI_TEMPERATURE=1.0
REQUEST_TIMEOUT_SECONDS=120
MAX_RETRY=2
ENG_SOLVER_DEFAULT_ROUTE=general
ENG_SOLVER_RETRIEVAL_ENABLED=true
```

### 6.3 运行测试

```bash
# 运行所有单元测试（57+ 测试）
python scripts/mini_pytest.py

# 冒烟测试
python scripts/smoke_test.py

# 本地评估
python scripts/run_local_eval.py --dev-path data/dev/dev.json
```

### 6.4 使用示例

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

# 并行批量求解
questions = [question1, question2, question3]
results = asyncio.run(agent.async_solve(questions, max_concurrent=5, mode="auto"))
```

**命令行**：

```bash
# 批量求解
python solve_dataset.py \
    --input data/dev/dev.json \
    --output results.json \
    --mode auto \
    --max-concurrent 5

# 竞赛运行（4门课程）
python competition_run.py

# 纯工具模式（最快，零 API 调用）
python solve_dataset.py \
    --input data/dev/dev.json \
    --output results.json \
    --mode tool_only \
    --kimi-client none
```

---

## 七、项目文件结构

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
│   ├── reasoning_engine.py        # ReAct推理引擎
│   ├── router.py                  # 规则路由
│   ├── tool_dispatcher.py         # 工具调度层
│   ├── adapter.py                 # 输入适配器
│   ├── formatter.py               # 输出格式化器
│   ├── verifier.py                # 提交验证器
│   ├── schemas.py                 # Pydantic数据模型
│   ├── config.py                  # 配置管理
│   ├── constants.py               # 全局常量
│   ├── exceptions.py              # 统一异常层次
│   ├── debug_logger.py            # 调试日志系统
│   │
│   ├── llm/                       # LLM集成
│   │   ├── __init__.py
│   │   ├── kimi_client.py         # Kimi API客户端
│   │   └── prompt_builder.py      # 两阶段提示词构建器
│   │
│   ├── tools/                     # 工具执行层
│   │   ├── __init__.py
│   │   ├── numerical_tool.py      # 统一数值计算工具（⭐ 新增）
│   │   ├── similarity_tool.py     # 相似题目查找工具（⭐ 新增）
│   │   ├── unit_tool.py           # 单位检查工具
│   │   └── _math_support.py       # 纯Python数学库
│   │
│   ├── retrieval/                 # 知识检索
│   │   ├── __init__.py
│   │   ├── retriever.py           # 关键词检索器
│   │   ├── langchain_retriever.py # LangChain向量检索器（⭐ 新增）
│   │   ├── kb_loader.py           # 知识库加载器
│   │   ├── formula_cards.json     # 35张公式卡片
│   │   └── solved_examples.jsonl  # 12道例题
│   │
│   ├── eval/                      # 本地评估
│   │   └── local_eval.py
│   │
│   └── compat/                    # 兼容层
│       └── pydantic_compat.py
│
├── scripts/                       # 脚本工具
│   ├── mini_pytest.py             # 微型测试运行器
│   ├── smoke_test.py              # 冒烟测试
│   └── run_local_eval.py          # 本地评估入口
│
├── tests/                         # 测试目录
│   ├── test_calculus_tool.py
│   ├── test_algebra_tool.py
│   ├── test_circuit_tool.py
│   ├── test_physics_tool.py
│   ├── test_langchain_retriever.py（⭐ 新增）
│   ├── test_react_engine.py       （⭐ 新增）
│   └── _helpers.py
│
├── data/                          # 数据集
│   ├── dev/dev.json               # 开发集（20题）
│   ├── dev/hard_test.json         # 困难测试集（12题）
│   └── calculus/teset.json        # 微积分级数专题（10题）
│
├── build_kb.py                    # FAISS索引构建脚本（⭐ 新增）
├── solve_dataset.py               # 批量求解命令行工具
├── competition_run.py             # 竞赛运行入口
├── example.py                     # LangChain演示示例
├── submission.json                # 竞赛提交配置
│
├── requirements.txt               # 依赖说明
├── pytest.bat
├── .env.example
├── .gitignore
└── README.md                      # 本文件
```

---

## 八、竞赛提交配置

`submission.json`：
```json
{
  "module": "eng_solver_agent.unified_agent",
  "class_name": "UnifiedAgent",
  "method_name": "solve"
}
```

---

## 九、依赖说明

### 核心运行（零外部依赖）

项目主体仅依赖 **Python 标准库**：`asyncio`, `json`, `re`, `os`, `sys`, `math`, `ast`, `fractions`, `urllib` 等。

### 可选增强

| 包 | 用途 | 无此包时的行为 |
|----|------|---------------|
| `sympy` | 符号计算 | 使用 `_math_support.py` fallback |
| `pydantic` | 数据验证 | 使用 `pydantic_compat.py` shim |
| `python-dotenv` | `.env` 文件加载 | 读取系统环境变量 |
| `langchain` + `langchain-community` + `langchain-huggingface` | 语义向量检索 | 回退到关键词检索 |
| `faiss-cpu` | 向量索引存储 | 回退到关键词检索 |
| `sentence-transformers` | 文本嵌入 | 回退到关键词检索 |

### 环境要求

- **Python 3.10+**
- 网络连接（仅 LLM 模式和向量索引下载需要）

---

## 十、设计决策记录

### 为什么使用 urllib 而不是 requests？

- 避免引入外部 HTTP 库依赖
- 竞赛环境可能无法 `pip install`

### 为什么将 4 个旧工具合并为统一工具？

- 减少重复代码和维护负担
- 提供统一的 `compute()` / `solve()` / `execute_code()` 入口
- 保持全部原始方法兼容性

### 为什么保留 agent.py / agent_v2.py / competition_agent.py？

- 向后兼容
- 不同场景下直接实例化特定智能体更明确

### 为什么使用 Fraction 而不是 float？

- 矩阵运算中浮点误差会导致错误结果
- Fraction 保持精确有理数运算，仅在最后一步转 float

### 为什么同时保留关键词检索和向量检索？

- 向量检索依赖外部库，可能无法安装
- 关键词检索零依赖，作为可靠的降级方案

---

## 十一、扩展指南

### 添加新学科

1. 在 `router.py` 的 `_RULES` 中添加新学科的关键词
2. 在 `tool_dispatcher.py` 中添加新学科的分发逻辑
3. 在 `tools/` 下新建工具类
4. 在 `UnifiedAgent._build_default_tools()` 中注册

### 添加新工具方法

在 `NumericalComputationTool` 中实现新方法（优先 SymPy，fallback 纯 Python），`_method_registry` 会自动暴露给 ReAct 引擎。

### 接入其他 LLM

实现相同接口：`chat(messages, temperature) → str`，然后：
```python
agent = UnifiedAgent(kimi_client=YourClient())
```

---

## 十二、许可证

MIT License
