"""Batch demo script for Xiaomi MIMO Orbit application."""
from eng_solver_agent.unified_agent import UnifiedAgent

agent = UnifiedAgent()

questions = [
    {"question_id": "batch-001", "question": "求积分 ∫(x^2 + 3x) dx"},
    {"question_id": "batch-002", "question": "计算极限 lim(x→0) (sin x)/x"},
    {"question_id": "batch-003", "question": "一个质量为2kg的物体受到10N的力，求加速度"},
]

print("=== 批量解题演示 ===")
for q in questions:
    result = agent.solve_one(q)
    print(f"\n[{q['question_id']}] 答案: {result['answer'][:80]}...")

print("\n=== 演示完成 ===")
