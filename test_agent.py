from eng_solver_agent.agent import EngineeringSolverAgent

agent = EngineeringSolverAgent()

result = agent.solve_one({
    "question_id": "tmp-001",
    "question": "Find the derivative of x^x.",
    "subject": "calculus",
    "expression": "x^x"
})

print(result)