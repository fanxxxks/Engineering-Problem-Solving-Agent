from pathlib import Path
from tempfile import TemporaryDirectory

from eng_solver_agent.retrieval.kb_loader import KnowledgeBaseLoader
from eng_solver_agent.retrieval.retriever import Retriever


def test_kb_loader_reads_json_and_jsonl() -> None:
    with TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        json_path = base / "cards.json"
        jsonl_path = base / "examples.jsonl"
        json_path.write_text(
            '[{"id":"1","subject":"physics","topic":"force","keywords":["force"],"formula":"F=ma","conditions":[],"common_traps":[]}]',
            encoding="utf-8",
        )
        jsonl_path.write_text(
            '{"question_id":"e1","subject":"calculus","topic":"derivative","question":"q","reasoning_process":"r","answer":"a","tags":["t"]}\n',
            encoding="utf-8",
        )

        loader = KnowledgeBaseLoader()
        json_records = loader.load(json_path)
        jsonl_records = loader.load(jsonl_path)

        assert len(json_records) == 1
        assert json_records[0]["topic"] == "force"
        assert len(jsonl_records) == 1
        assert jsonl_records[0]["question_id"] == "e1"


def test_retriever_hits_calculus_and_circuits_from_local_kb() -> None:
    loader = KnowledgeBaseLoader()
    root = Path(__file__).resolve().parents[1] / "eng_solver_agent" / "retrieval"
    formula_cards = loader.load(root / "formula_cards.json")
    solved_examples = loader.load(root / "solved_examples.jsonl")
    retriever = Retriever(formula_cards=formula_cards, solved_examples=solved_examples)

    calculus_result = retriever.retrieve("Find the derivative of x^2.", subject="calculus", topic="derivative", top_k=2)
    circuits_result = retriever.retrieve("Two resistors in series with voltage.", subject="circuits", topic="equivalent_resistance", top_k=2)

    assert calculus_result.formula_cards
    assert calculus_result.solved_examples
    assert calculus_result.formula_cards[0]["subject"] == "calculus"
    assert circuits_result.formula_cards
    assert circuits_result.solved_examples
    assert circuits_result.formula_cards[0]["subject"] == "circuits"


def test_retriever_returns_empty_results_when_no_hit() -> None:
    retriever = Retriever(formula_cards=[], solved_examples=[])

    result = retriever.retrieve("completely unrelated text", subject="physics", topic="unknown", top_k=3)

    assert result.formula_cards == []
    assert result.solved_examples == []
