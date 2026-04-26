# Plan: Refactor Retriever and Tools Following example.py Approach

## Context

The user wants to thoroughly refactor the `retriever` and `tools` modules to follow the approach demonstrated in `example.py`:
1. **Knowledge base**: FAISS + HuggingFace embeddings for semantic search (vs. current keyword-only)
2. **Computation**: SymPy exec sandbox for flexible math solving (vs. current hardcoded method dispatch)

## Key Design Principle

**Additive layering** — each new capability sits on top of existing code without replacing it. All existing APIs remain backward-compatible, all 16 test files pass unchanged.

## Changes Summary

### 1. `eng_solver_agent/retrieval/kb_loader.py` — Add FAISS index builder

Add 3 guarded import functions and 2 new module-level functions:

- **`_load_sentence_transformers()`** — guarded import for `sentence-transformers`
- **`_load_faiss()`** — guarded import for `faiss`
- **`_load_pypdf()`** — guarded import for `pypdf` (for future PDF support)
- **`build_faiss_index(formula_cards, solved_examples, output_path)`** — converts KB data → embeddings → FAISS index, saves to disk
- **`load_faiss_index(index_path)`** — loads saved index from disk

Existing `KnowledgeBaseLoader` class: **unchanged**.

### 2. `eng_solver_agent/retrieval/retriever.py` — Add semantic search

Add to `Retriever` class:

- **`__init__` new param**: `faiss_index_path: str | Path | None = None`
- **`_load_semantic_index()`** — called from `__init__`, loads FAISS index if path provided
- **`_semantic_search(query, top_k)`** — embeds query, searches FAISS, returns ranked results
- **`_merge_results(keyword_hits, semantic_hits, top_k)`** — Reciprocal Rank Fusion to blend keyword + semantic rankings
- **`retrieve()` update** — calls `_semantic_search()` + `_merge_results()` when index is available

When `_semantic_index` is None: `retrieve()` produces **exactly the same results as before**.

### 3. `eng_solver_agent/retrieval/__init__.py` — Export new functions

Add `build_faiss_index` and `load_faiss_index` to exports.

### 4. `eng_solver_agent/tools/numerical_tool.py` — Add SymPy sandbox

Add to `NumericalComputationTool` class:

- **`_numpy_available()`** — module-level guarded import
- **`sympy_solve(query: str) -> str`** — safe exec sandbox with sympy/math/numpy in globals, captures stdout
- **`compute_from_query()` update** — route `"sympy:"` prefixed queries to `sympy_solve()`

All existing methods (`diff`, `integrate`, `determinant`, etc.) and internal engines: **unchanged**.

### 5. No changes needed

- `eng_solver_agent/tools/__init__.py` — class names unchanged
- `eng_solver_agent/tools/similarity_tool.py` — auto-benefits from Retriever upgrades
- `eng_solver_agent/tools/unit_tool.py` — unrelated
- `eng_solver_agent/tools/_math_support.py` — still used as fallback
- `eng_solver_agent/schemas.py` — `RetrievalResult` schema unchanged
- All 16 test files — backward-compatible APIs

## Graceful Degradation

| Missing dep | Behavior |
|---|---|
| sentence-transformers | Retriever stays keyword-only |
| faiss | Retriever stays keyword-only |
| sympy | sympy_solve() returns error; existing engines use poly fallback |
| numpy | sympy_solve() omits np from globals |
| No FAISS index file | Retriever stays keyword-only |

All new imports are guarded (follows existing `load_sympy()` pattern in `_math_support.py`).

## Verification

1. `python -m pytest tests/` — all 16 test files pass
2. Test `sympy_solve("print(diff(x**2, x))")` → `"2*x"`
3. Test `compute_from_query("sympy: print(integrate(x**2, x))")` → dispatches to sandbox
4. Test Retriever without FAISS index produces same keyword results as before
