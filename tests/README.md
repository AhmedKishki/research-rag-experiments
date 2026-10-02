# `tests/`

Tests mirror the folders under `src/rag_experiments/`, so a change to one
concern is tested by that concern's tests alone.

| File | Holds |
|---|---|
| `conftest.py` | the fixtures, including the minimum a research-rag project must hold |
| `test_engine.py` | naming the engine under test, and resolving its settings |
| `test_guard.py` | that a source project is digested, and what a change reads as |
| `test_sandbox.py` | that a copy is a valid project and carries nothing else |
| `test_spec.py` | that a specification is refused when it says something unclear |
| `test_execute.py` | that an arm's command says what the specification asked for |
| `test_compare.py` | that the table reads a report and computes no metric |
| `test_record.py` | that a run's verdict follows from the guard |
| `test_cli.py` | the one command's statuses and its refusals |

A test builds a disposable project rather than a real one. `conftest.py` states
the minimum the app's own readers require, which is the same fact a copy depends
on: if a real project needed more than this, the copy would be incomplete and
these tests would not have noticed.
