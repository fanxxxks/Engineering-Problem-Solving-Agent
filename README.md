# Engineering Problem Solving Agent

Competition-grade agent for solving engineering foundation course problems (Calculus, Linear Algebra, Circuits, Physics). 

## Architecture Overview

```
UnifiedAgent (single entrypoint)
├── Mode: auto      → LLM ReAct when available, tools otherwise
├── Mode: react     → Think → Act → Observe reasoning loop
├── Mode: legacy    → analyze → tool → draft pipeline
├── Mode: llm_only  → Direct LLM solve
└── Mode: tool_only → Tools only, no LLM API calls

Parallel Solving: asyncio.Semaphore(max_concurrent=5)
```

## Key Features

- **Unified Entrypoint**: `UnifiedAgent` replaces 3 conflicting agents with one clean API
- **Parallel Processing**: `async_solve()` controls concurrency to avoid API rate limits
- **ReAct Reasoning**: Think → Act → Observe loop for maximum reasoning_process score
- **Smart Answer Grading**: Formula matching, numeric tolerance, keyword jaccard
- **5 Solving Modes**: `auto`, `react`, `legacy`, `llm_only`, `tool_only`
- **Tool Layer**: SymPy-backed calculus/algebra + circuit physics formulas
- **Chinese Text Parsing**: Matrix literals `[[a,b],[c,d]]`, determinants `|a b; c d|`, resistor values

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment (create .env file)
KIMI_BASE_URL=https://api.moonshot.cn/v1
KIMI_API_KEY=your_api_key_here

# Run tests
python scripts/mini_pytest.py

# Solve a single dataset
python competition_run.py --input data/dev/dev.json --output results.json --mode auto

# Solve all 4 subjects
python competition_run.py
```

## Competition Submission

```json
{
  "module": "eng_solver_agent.unified_agent",
  "class_name": "UnifiedAgent",
  "method_name": "solve"
}
```

## Usage Examples

### Python API

```python
from eng_solver_agent import UnifiedAgent
import asyncio

agent = UnifiedAgent()

# Sequential
results = agent.solve(questions, mode="auto")

# Parallel (recommended for competitions)
results = asyncio.run(agent.async_solve(questions, max_concurrent=5))

# Tool-only (fast, no API calls)
results = agent.solve(questions, mode="tool_only")
```

### Command Line

```bash
# Auto mode with parallel processing
python competition_run.py \
    --input data/基础物理学.json data/微积分.json \
    --output submission.json \
    --mode auto \
    --max-concurrent 5

# Tool-only mode (no LLM)
python competition_run.py \
    --input data/dev/dev.json \
    --output results.json \
    --mode tool_only \
    --kimi-client none
```

## Solving Modes

| Mode | LLM | Tools | Speed | Best For |
|------|-----|-------|-------|----------|
| `auto` | ✅ adaptive | ✅ | Medium | **Default recommendation** |
| `react` | ✅ multi-turn | ✅ | Slow | Detailed reasoning required |
| `legacy` | ✅ 2-stage | ✅ | Medium | Compatible with old behavior |
| `llm_only` | ✅ direct | ❌ | Medium | Proofs, symbolic derivation |
| `tool_only` | ❌ | ✅ | <1ms/q | Numerical calculations only |

## Project Structure

```
eng_solver_agent/
├── unified_agent.py        # Main entrypoint (5 modes + parallel)
├── competition_agent.py    # ReAct wrapper for competitions
├── agent.py               # Original entrypoint (backward compatible)
├── tool_dispatcher.py     # Standalone tool routing
├── reasoning_engine.py    # ReAct Think→Act→Observe loop
├── router.py              # Rule-based subject classification
├── llm/                   # KimiClient + prompt builder
├── tools/                 # Calculus, Algebra, Circuits, Physics
├── retrieval/             # Keyword-based knowledge retrieval
├── schemas.py             # Pydantic data models
└── eval/                  # Local evaluation tools

data/                      # Validation datasets
tests/                     # 44 unit tests
scripts/                   # Smoke tests and helpers
```

## Performance Benchmarks

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Retriever | 0.88ms/q | 0.39ms/q | **55% faster** |
| Router | 0.013ms/q | 0.012ms/q | ~10% faster |
| Agent batch (10q) | ~750ms | ~700ms | ~7% faster |

## Scoring Strategy

Based on competition scoring rules:

- **Accuracy (40%)**: LLM handles proofs/symbolic, tools verify numerical
- **Reasoning Quality (25%)**: ReAct loop produces detailed Chinese math steps
- **Solution Design (20%)**: Innovative architecture with tool integration
- **Efficiency (10%)**: Parallel solving with semaphore control
- **Robustness (5%)**: Graceful fallback on tool/LLM failures

## Testing

```bash
# Run all tests
python scripts/mini_pytest.py

# Expected: 44 passed, 0 failed
```

## Requirements

- Python 3.10+
- `sympy` (optional but recommended for full calculus support)
- `pydantic` v1 or v2 (compat layer handles both)
- Kimi K2.5 API key (for LLM modes)

## License

MIT
