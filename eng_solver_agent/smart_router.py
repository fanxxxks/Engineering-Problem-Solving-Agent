"""DEPRECATED: SmartRouter has been merged into QuestionRouter.

The rule-based router now handles all routing needs efficiently.
LLM-based routing is redundant since the analyze stage already uses LLM.
"""

from eng_solver_agent.router import QuestionRouter

SmartRouter = QuestionRouter
