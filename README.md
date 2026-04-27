# Engineering-Problem-Solving-Agent

竞赛级智能体，用于求解工科基础课程问题（微积分、线性代数、电路、物理）。

本项目基于 **LLM 推理 + 精确工具计算** 的双引擎架构，支持 **ReAct（思考-行动-观察）** 多轮推理循环，针对中文竞赛环境进行了大量专门优化。

---

## 目录

1. [项目概述](#一项目概述)
2. [架构总览](#二架构总览)
3. [数据格式规范](#三数据格式规范)
4. [模块详解](#四模块详解)
5. [工具层深度文档](#五工具层深度文档)
6. [多智能体与竞赛系统](#六多智能体与竞赛系统)
7. [求解模式对比](#七求解模式对比)
8. [快速开始](#八快速开始)
9. [环境变量完整参考](#九环境变量完整参考)
10. [项目文件结构](#十项目文件结构)
11. [错误处理与降级策略](#十一错误处理与降级策略)
12. [性能优化](#十二性能优化)
13. [依赖说明](#十三依赖说明)
14. [故障排查](#十四故障排查)
15. [设计决策记录](#十五设计决策记录)
16. [扩展指南](#十六扩展指南)
17. [贡献指南](#十七贡献指南)
18. [许可证](#十八许可证)

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

### 2.2 模块层次图

```
┌─────────────────────────────────────────────────────────┐
│                     入口层 (Entry)                        │
│  unified_agent.py  |  solve_dataset.py  |  competition_run.py │
├─────────────────────────────────────────────────────────┤
│                    智能体层 (Agents)                       │
│  agent.py  |  agent_v2.py  |  competition_agent.py       │
│  multi_agent_system.py (Worker/Checker/Orchestrator)     │
├─────────────────────────────────────────────────────────┤
│                   推理与路由层 (Core)                      │
│  reasoning_engine.py (ReAct)  |  router.py (规则路由)      │
│  smart_router.py (兼容别名)  |  tool_dispatcher.py (调度)  │
├─────────────────────────────────────────────────────────┤
│                   工具执行层 (Tools)                       │
│  numerical_tool.py  |  similarity_tool.py  |  unit_tool.py│
│  _math_support.py (纯Python数学库)                        │
├─────────────────────────────────────────────────────────┤
│                   知识检索层 (Retrieval)                   │
│  langchain_retriever.py (向量检索)  |  retriever.py (关键词)│
│  kb_loader.py (知识库加载器)                               │
├─────────────────────────────────────────────────────────┤
│                   LLM 集成层 (LLM)                        │
│  kimi_client.py  |  prompt_builder.py (提示词构建)         │
├─────────────────────────────────────────────────────────┤
│                  基础设施层 (Infra)                        │
│  schemas.py  |  config.py  |  constants.py  |  exceptions.py│
│  formatter.py  |  verifier.py  |  adapter.py               │
│  debug_logger.py  |  compat/pydantic_compat.py             │
├─────────────────────────────────────────────────────────┤
│                  评估与脚本层 (Eval)                       │
│  eval/local_eval.py  |  scripts/mini_pytest.py             │
│  scripts/smoke_test.py  |  scripts/run_local_eval.py       │
└─────────────────────────────────────────────────────────┘
```

---

## 三、数据格式规范

### 3.1 输入题目格式

系统支持多种输入格式，通过 `QuestionAdapter.normalize()` 自动标准化：

**标准格式（推荐）**：
```json
{
  "question_id": "unique-id-001",
  "question": "求函数 f(x) = x^2 的导数",
  "subject": "calculus",
  "expression": "x**2",
  "variable": "x"
}
```

**兼容格式**：
```json
{
  "id": "unique-id-001",
  "prompt": "求函数 f(x) = x^2 的导数"
}
```

**字段说明**：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `question_id` / `id` | `str` | 是 | 题目唯一标识 |
| `question` / `prompt` | `str` | 是 | 题目文本 |
| `subject` | `str` | 否 | 预分类学科 (`calculus`/`linalg`/`circuits`/`physics`) |
| `expression` | `str` | 否 | 数学表达式（微积分） |
| `equation` / `function` | `str` | 否 | 表达式别名 |
| `variable` | `str` | 否 | 自变量（默认 `x`） |
| `matrix` | `list[list]` | 否 | 矩阵数据（线性代数） |
| `rhs` | `list` | 否 | 线性方程组右端项 |
| `resistors` | `list[float]` | 否 | 电阻值列表（电路） |
| `topology` | `str` | 否 | 电路拓扑 (`series`/`parallel`) |
| `knowns` | `dict` | 否 | 已知物理量（物理） |
| `target` | `str` | 否 | 求解目标（物理） |
| `point` | `float` | 否 | 极限点 |
| `lower` / `upper` | `float` | 否 | 定积分上下界 |

### 3.2 输出结果格式

```json
{
  "question_id": "unique-id-001",
  "reasoning_process": "首先，我们识别出这是一个求导问题...\n使用求导法则：...\n最终得到：2*x",
  "answer": "2*x"
}
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `question_id` | `str` | 题目唯一标识（与输入一致） |
| `reasoning_process` | `str` | 中文逐步推导过程 |
| `answer` | `str` | 最终答案（数值或表达式） |

**验证规则**：
- `reasoning_process` 和 `answer` 均不能为空字符串
- 通过 `formatter.py` 和 `verifier.py` 双重校验

---

## 四、模块详解

### 4.1 统一入口层

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

**五种求解模式**：详见 [七、求解模式对比](#七求解模式对比)。

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

#### `agent_v2.py` — EnhancedSolverAgent（增强版）

采用 **LLM 优先** 策略：先尝试让 LLM 直接给出带推理过程的 JSON 答案，失败后再回退到传统的分析-工具-草稿两阶段流水线。

**核心方法**：
- `solve_one(question)`：先走 LLM 直接求解，再走父类传统流程
- `_llm_direct_solve(question)`：构造 few-shot 风格 system prompt，要求 Kimi 返回 `{"reasoning_process": "...", "answer": "..."}`

**适用场景**：显著降低对符号工具可用性的依赖；在工具缺失或 API 异常时，仍能通过纯 LLM 推理输出结构化答案。

---

#### `competition_agent.py` — CompetitionAgent（竞赛专用）

针对**竞赛评分规则**优化：优先使用 ReAct 生成详细中文数学推导，即使答案错误也能拿到公式分。

---

### 4.2 推理引擎层

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

### 4.3 路由层

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

#### `smart_router.py` — 兼容别名

`SmartRouter` 已合并进 `QuestionRouter`，仅为向后兼容保留的别名。

---

### 4.4 工具调度层

#### `tool_dispatcher.py` — ToolDispatcher

整个系统的**工具调度中枢**。根据题目所属学科（`subject`）与文本内容，将问题路由到对应符号计算工具，并负责参数抽取与结果序列化。

**核心能力**：

| 能力 | 说明 |
|------|------|
| 学科分发 | `dispatch_calculus` / `dispatch_linalg` / `dispatch_circuits` / `dispatch_physics` |
| 级数特殊处理 | 一致收敛/证明类题目显式拒绝，提示使用 LLM 模式 |
| 表达式提取 | 支持中英文双语的微积分表达式正则匹配 |
| 矩阵提取 | `[[a,b],[c,d]]`、`|a b; c d|`、`[a b; c d]` 三种格式 |
| 电阻提取 | 匹配 "2Ω"、"3 欧姆" 等中文/英文单位 |
| 物理量提取 | 质量(kg)、速度(m/s)、加速度(m/s²)、力(N) |
| 边界提取 | `from a to b`、`[a, b]` 格式 |
| 极限点提取 | `as x -> 2`、`x 趋向于 2` |

**级数问题降级策略**：
当检测到 "一致收敛"、"uniform convergence"、"证明"、"prove" 等关键词时，直接返回失败并提示使用 LLM 模式，避免工具在无法处理的证明题上浪费时间。

---

### 4.5 LLM 集成层

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

#### `llm/prompt_builder.py` — PromptBuilder

两阶段提示词工厂，为 LLM 的 analyze（分析）与 draft（撰写最终答案）阶段生成严格 JSON 格式的系统提示。

**学科模板**：覆盖 physics、circuits、linalg、calculus 四大学科，每科定义 focus、tool_hint、draft_hint。

**检索增强**：若启用检索，自动将公式卡片与例题拼接到 prompt 中。

**JSON 强制输出**：所有 prompt 均要求 `No markdown. No prose. No code fences.`，便于下游结构化解析。

---

### 4.6 知识检索层

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

#### `retrieval/kb_loader.py` — KnowledgeBaseLoader

知识库加载器，支持 JSON 和 JSONL 格式。

**功能**：
- `load(path)`：自动识别 `.json` / `.jsonl` 后缀
- `load_formula_cards(path)`：加载公式卡片
- `load_solved_examples(path)`：加载例题
- 容错处理：空文件、格式错误均有清晰异常信息

---

### 4.7 调试与日志层

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

### 4.8 数据模型与基础设施层

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

#### `compat/pydantic_compat.py` — Pydantic 兼容层

当环境中**未安装 Pydantic** 时，提供最小兼容实现：
- `BaseModel`：手动实现 `__init__`、`model_dump`、`dict`、`model_validate`
- `Field(...)`：模拟 Pydantic 字段工厂
- `ValidationError`：自定义异常

支持类型注解校验、`Field(default_factory=...)`、`Optional` 识别。

**设计意图**：在无 Pydantic 的受限环境中也能保证 schema 校验不崩溃；若环境允许，建议切回真实 Pydantic。

---

### 4.9 格式化与验证层

#### `formatter.py` — 输出格式化器

- `format_submission_item()`：基于 `FinalAnswer` schema 生成单条提交记录
- `format_submission_batch(items)`：批量格式化
- `_require_non_empty(value, field_name)`：强制校验 `reasoning_process` 与 `answer` 不得为空

#### `verifier.py` — 提交验证器

- `REQUIRED_SUBMISSION_KEYS = ("question_id", "reasoning_process", "answer")`
- `validate_submission_item(item)` / `validate_final_answer(item)`：检查缺失键与空值
- 与 formatter 形成双重保障，在最终输出前再次确认竞赛格式合规

---

### 4.10 评估与脚本层

#### `eval/local_eval.py` — 本地评估工具

- `compare_answers()`：精确匹配 + 数值容差匹配（默认 tolerance=1e-6）
- `evaluate_dev_set()`：统计总题数、答题数、精确匹配数、分学科准确率
- `run_local_eval()`：端到端评估入口，自动加载 dev set 和预测结果
- **fallback 检测**：自动识别 "暂无法可靠给出最终数值" 等提示语

#### `scripts/mini_pytest.py` — 微型测试运行器

零依赖的 pytest 兼容测试运行器，自动发现 `tests/test_*.py`。

#### `scripts/smoke_test.py` — 冒烟测试

极简脚本，快速验证 agent 能否正常求解单题和批量题目。

#### `scripts/run_local_eval.py` — 本地评估入口

命令行封装，默认路径指向 `data/dev/dev.json`。

---

## 五、工具层深度文档

### 5.1 `tools/numerical_tool.py` — NumericalComputationTool（统一数值计算工具）

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

### 5.2 `tools/similarity_tool.py` — SimilarProblemTool（相似题目查找工具）

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

### 5.3 `tools/unit_tool.py` — UnitTool（单位检查工具）

轻量级量纲检查工具，用于验证推理过程中的单位一致性。

**支持的单位**：

| 单位 | 量纲 |
|------|------|
| m | 长度 (L) |
| s | 时间 (T) |
| kg | 质量 (M) |
| A | 电流 (I) |
| V | M·L²·T⁻³·I⁻¹ |
| Ω | M·L²·T⁻³·I⁻² |
| N | M·L·T⁻² |
| J | M·L²·T⁻² |
| W | M·L²·T⁻³ |
| C | T·I |
| Hz | T⁻¹ |

**核心方法**：
- `dimension_of(unit)`：返回单位的量纲向量
- `compatible(unit_a, unit_b)`：判断两个单位是否量纲兼容
- `check_reasoning_process(text)`：从推理文本中提取单位并识别未知 token

---

### 5.4 `tools/_math_support.py` — 纯 Python 数学工具库

当 **SymPy 未安装**时的完整 fallback 实现。使用 `fractions.Fraction` 进行精确有理数运算，避免浮点误差。

**多项式运算**：
- `poly_from_ast()`：通过 Python `ast` 模块解析字符串为多项式
- `poly_diff()` / `poly_integral()`：多项式求导/积分
- `poly_eval()` / `poly_roots()`：多项式求值/求根
- `poly_to_string()` / `poly_from_constant()`：格式转换

**矩阵运算**：
- `solve_linear_system()`：高斯消元（Fraction 精确）
- `determinant()`：行列式计算
- `inverse()`：矩阵求逆
- `rank()`：矩阵秩
- `eigenpairs_2x2()`：2×2 矩阵特征值/特征向量

**其他**：
- `taylor_series()`：泰勒展开（多项式近似）
- `quadratic_critical_points()`：二次函数临界点
- `load_sympy()`：尝试加载 SymPy，失败返回 None

---

## 六、多智能体与竞赛系统

### 6.1 `multi_agent_system.py` — 多智能体并行求解

包含三类角色：**Worker（解题）**、**Checker（质检）**、**Orchestrator（调度与重试）**。

**三阶段流水线**：
1. **并行解题**：`ThreadPoolExecutor` 并行调度多个 `WorkerAgent`
2. **答案质检**：`CheckerAgent` 使用 LLM 或规则对每道题进行质量评估
3. **自动重试**：对低置信度/失败题自动重试（最多 `max_retries` 次）

**WorkerAgent 置信度估计**：
```python
confidence = 0.5
+ 0.2  if answer contains digits
+ 0.15 if reasoning length > 100
- 0.3  if contains uncertainty markers ("无法", "unknown", "error")
```

**CheckerAgent 质检维度**：
- 数学正确性
- 单位一致性
- 数量级合理性
- 推理完整性

> **注意**：`multi_agent_system_v2.py` 已废弃，统一使用 `UnifiedAgent.async_solve()` 进行基于 asyncio 的并行求解。

---

### 6.2 `competition_run.py` — 竞赛运行与智能评分

**功能**：按学科分文件加载、并行求解、智能评分，并输出总报告。

**四级评分策略**（`grade_answer`）：

| 级别 | 策略 | 置信度 | 说明 |
|------|------|--------|------|
| 1 | 直接子串匹配 | 1.0 |  gold 在 pred 中或 vice versa |
| 2 | 公式匹配 | 0.85 |  gold 的关键公式出现在 pred 中 |
| 3 | 数值容差匹配 | 0.8 |  绝对误差 < 1e-3 |
| 4 | 数值近似匹配 | 0.7 |  相对误差 < 5% |
| 5 | 关键词 Jaccard | jaccard |  Jaccard > 0.4 |

**预处理**：
- 去除 LaTeX 包装符
- 中文标点转英文
- 统一小写

**输出**：`output/competition_report.json`，包含学科级与全局级报告。

---

### 6.3 `solve_dataset.py` — 批量求解命令行工具

面向赛题数据集的命令行批量求解入口。

**核心特性**：
- **多文件合并**：支持同时传入多个 JSON 文件，自动合并求解
- **进度条**：终端实时进度 + ETA 估算
- **并发控制**：`asyncio.Semaphore` 控制最大并发 LLM 调用数
- **五种模式**：`auto` / `react` / `legacy` / `llm_only` / `tool_only`

**CLI 参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` / `-i` | `str[]` | 必需 | 输入 JSON 文件路径（可多个） |
| `--output` / `-o` | `str` | 必需 | 输出 JSON 文件路径 |
| `--mode` / `-m` | `str` | `auto` | 求解模式 |
| `--max-concurrent` / `-c` | `int` | 5 | 最大并发数 |
| `--kimi-client` | `str` | `auto` | LLM 客户端配置 (`auto`/`none`) |
| `--submission-info` | `flag` | 否 | 同时生成 submission.json |

---

## 七、求解模式对比

| 模式 | LLM | 工具 | 检索 | 速度 | 适用场景 |
|------|-----|------|------|------|----------|
| `auto` | 自适应 | 自适应 | 自适应 | 中等 | **默认推荐**，自动选择最优策略 |
| `react` | 多轮 | 嵌入推理链 | 可选 | 较慢 | 需要详细逐步推理，最大化 reasoning_process 分数 |
| `legacy` | 两阶段 | 独立调用 | 可选 | 中等 | 结构化数据充分的数值计算题 |
| `llm_only` | 直接 | 不使用 | 可选 | 中等 | 证明题、概念题、符号推导 |
| `tool_only` | 不使用 | 直接 | 不使用 | <1ms/题 | 纯数值计算、快速验证、零API成本 |

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

## 八、快速开始

### 8.1 安装

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

### 8.2 配置

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

### 8.3 运行测试

```bash
# 运行所有单元测试（57+ 测试）
python scripts/mini_pytest.py

# 冒烟测试
python scripts/smoke_test.py

# 本地评估
python scripts/run_local_eval.py --dev-path data/dev/dev.json
```

### 8.4 使用示例

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

## 九、环境变量完整参考

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `KIMI_API_KEY` | `str` | `