"""Quick verification that the new knowledge base entries are searchable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eng_solver_agent.retrieval.langchain_retriever import LangChainRetriever

rd = ROOT / "eng_solver_agent" / "retrieval"
r = LangChainRetriever(
    formula_cards_path=rd / "formula_cards.json",
    solved_examples_path=rd / "solved_examples.jsonl",
    index_dir=str(ROOT / "faiss_index"),
)
r.load_index()

tests = [
    ("电路: 戴维南等效", "求戴维南等效电路", "circuits"),
    ("线性代数: 高斯消元", "高斯消元法判断线性方程组解的情况", "linalg"),
    ("电路: 三要素法暂态", "一阶RC电路三要素法求暂态响应", "circuits"),
    ("线性代数: 矩阵多项式求逆", "矩阵多项式求逆", "linalg"),
]

for label, query, subject in tests:
    print(f"\n=== {label} ===")
    res = r.retrieve(query, subject=subject, top_k=3)
    for c in res.formula_cards:
        print(f"  [F] {c['id']}: {str(c.get('formula',''))[:60]}")
    for e in res.solved_examples:
        print(f"  [E] {e['question_id']}: {str(e.get('question',''))[:60]}")

print("\n=== 知识库统计 ===")
print(f"  公式卡片总数: {len(r._formula_cards)}")
print(f"  例题总数: {len(r._solved_examples)}")
