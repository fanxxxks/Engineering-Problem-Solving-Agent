# eng_solver_agent

Minimal Python library scaffold for an engineering problem-solving competition.

## Structure

- `eng_solver_agent/`: package root
- `scripts/`: local smoke and evaluation scripts
- `tests/`: minimal pytest coverage
- `submission.json`: competition entrypoint declaration

## Quick Start

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q
python scripts\\smoke_test.py
python scripts\\run_local_eval.py
```

## Local Eval

Local development evaluation reads questions from `data/dev/dev.json` and writes outputs to `data/exports/dev_predictions.json` and `data/exports/dev_report.json`.

You can rerun it with:

```bash
python scripts\\run_local_eval.py
```

## Notes

- This repository intentionally implements only a runnable placeholder scaffold.
- `EngineeringSolverAgent` lives in `eng_solver_agent/agent.py`.
- The competition output format is `question_id`, `reasoning_process`, and `answer`.
- The repository currently uses an internal `eng_solver_agent.compat.pydantic_compat` shim because the machine does not have the real `pydantic` package installed.
- If the environment later supports it, prefer switching imports back to the real `pydantic` package instead of keeping the shim.
- The repository includes a tiny local `pytest -q` shim so the smoke tests can run without third-party packages.
