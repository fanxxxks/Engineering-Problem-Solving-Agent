"""Schema definitions for the engineering solver agent."""

from __future__ import annotations

from typing import Any, Literal, Optional

from eng_solver_agent.compat import BaseModel, Field


class QuestionInput(BaseModel):
    question_id: str
    question: str
    type: Optional[str] = None
    difficulty: Optional[str] = None
    image_path: Optional[str] = None


class ParsedQuestion(BaseModel):
    subject: Literal["physics", "circuits", "linalg", "calculus"]
    topic: str
    raw_question: str
    normalized_question: str
    knowns: list = Field(default_factory=list)
    unknowns: list = Field(default_factory=list)
    extra_context: dict = Field(default_factory=dict)


class RetrievalContext(BaseModel):
    question_id: Any
    query: str
    documents: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_name: str
    success: bool = True
    output: str = ""
    metadata: dict = Field(default_factory=dict)


class FinalAnswer(BaseModel):
    question_id: Any
    reasoning_process: str
    answer: str
    subject: Optional[str] = None
    topic: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class AnalyzeResult(BaseModel):
    subject: Literal["physics", "circuits", "linalg", "calculus"]
    topic: str
    knowns: list = Field(default_factory=list)
    unknowns: list = Field(default_factory=list)
    equations_or_theorems: list = Field(default_factory=list)
    should_use_tool: bool
    target_form: str
    possible_traps: list = Field(default_factory=list)


class DraftResult(BaseModel):
    reasoning_process: str
    answer: str


class FormulaCard(BaseModel):
    id: str
    subject: Literal["physics", "circuits", "linalg", "calculus"]
    topic: str
    keywords: list = Field(default_factory=list)
    formula: str
    conditions: list = Field(default_factory=list)
    common_traps: list = Field(default_factory=list)


class SolvedExample(BaseModel):
    question_id: str
    subject: Literal["physics", "circuits", "linalg", "calculus"]
    topic: str
    question: str
    reasoning_process: str
    answer: str
    tags: list = Field(default_factory=list)


class RetrievalResult(BaseModel):
    subject: Optional[str] = None
    topic: Optional[str] = None
    query: str = ""
    formula_cards: list = Field(default_factory=list)
    solved_examples: list = Field(default_factory=list)
    matched_terms: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


__all__ = [
    "QuestionInput",
    "ParsedQuestion",
    "RetrievalContext",
    "ToolResult",
    "FinalAnswer",
    "AnalyzeResult",
    "DraftResult",
    "FormulaCard",
    "SolvedExample",
    "RetrievalResult",
]
