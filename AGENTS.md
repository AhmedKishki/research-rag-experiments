# AGENTS.md

The engineering guide for AI agents working in `rag-experiments`.

## What this is

- An experiment toolkit that measures `research-rag` retrieval in a disposable
  copy of a project.
- It is a developer tool, not a server, and it is not a member of any collection
  of servers. It reaches the app as an installed dependency, which is a pinned
  dependency rather than a sibling's private state.
- The one claim it makes is that a measurement did not change the project it
  measured, and it makes that claim checkable rather than asserted.

## The rule that binds everything else

- An experiment never writes inside a source project.
  - `sandbox/guard.py` digests a project's originals and its state before a run
    and again after, and names every file that differs.
  - A run whose digest moved is `source_project_changed`, and its numbers are
    disqualified rather than hidden.
  - A source project is opened `rb` and that is all. `resolve_config` is never
    called on one, because it makes directories and rewrites the runtime pointer.
- A quality number comes from the app's own harness. Nothing here computes one.
  - A metric the app's report stopped carrying prints as a dash, not a zero.
  - The comparison does not average, weight, rank, or decide. The first arm the
    specification listed is the baseline, and it is the first arm because a
    baseline chosen after the fact is not a baseline.
- A fact about the app is read from the app.
  - On-disk names come from `research_rag.project.state_files` and from the path
    literals inside `ResearchConfig` properties, read out of their code objects.
  - Settings come from the app's own registry, resolved by the app's own layer
    stack, inside the tree under test.
  - A name this repository spells out itself is a second copy that will drift.
- A measurement is reproducible or it is not a measurement.
  - Every setting the engine declares is pinned into the sandbox, so the account
    overlay, a `RESEARCH_RAG_` variable, and a later packaged default cannot move
    a number.
  - A record names the engine revision, the corpus digest, the settings and the
    layer each came from, every command, and the exit status.
  - A settings arm and a code arm are not interchangeable, and neither is reduced
    to the other. A code arm gets its own checkout at a named revision so the app
    source is never patched.

## Documentation responsibilities

Each fact has one home. Every other file points at it.

| File | Owns |
|---|---|
| `README.md` | the user manual: what it does, how to run it, what it cannot do |
| `STORAGE.md` | the on-disk format: the workspace, the sandbox, the run record |
| `AGENTS.md` | this file: the rules that are not derivable from the code or the tests |
| `examples/` | one run specification that runs against a project the reader has |
| A folder's `README.md` | what that folder owns, and what it may never do |
| A module's docstring | why that module does what it does, and what it may never do |

- The last two rows are the ones that bind: a rule stated here and again in the
  module it governs is a rule in two places, and the two will disagree.
- `tests/` mirrors the folders under `src/rag_experiments/`, so a change to one
  concern is tested by that concern's tests alone.
- Markdown describes the present. No review logs, no change histories, no "Step
  N", and no recorded decisions. A decision that still binds is a rule, stated
  once, as a rule; a decision that no longer binds is deleted.
- Only direct, concise language. One sentence carries one fact. No preamble, no
  restatement of what the reader just read, no superlative, no selling. A number,
  a mechanism, or a limit is the argument.

## Presenting a choice to the user

- Any question needing a user choice is a numbered list of concrete options, never
  an open question.
- Each option carries a short identifier and a one-line statement, the smallest
  change it requires, pros and cons covering cost, risk, and the effect on the
  research contract, and whether it is reversible.
- Exactly one option is marked as the recommendation, and one sentence says why.
- A "no change" option is included whenever work can proceed without an answer.
- No choice that changes a project's on-disk state, a retrievable artifact, or the
  record format is implemented before the user has chosen it.

## Boundaries a contributor must not cross

- The package is split at the folders below, one concern each, and a module
  imports from the folder above it and its own siblings, never a sibling folder
  sideways.
- `engine/` is the only folder that imports `research_rag`, and it does so inside
  a function, so importing this toolkit never drags the retrieval stack in.
- No experiment writes to a directory outside its workspace, its runs directory,
  or its run directory.
- A sandbox is addressed by path and is never registered with the app.
  `research-rag init` is never called on a copy.
- The model cache is deliberately shared. Those are immutable binaries, and a run
  downloads nothing; the account *configuration* is deliberately not shared, and
  is relocated to an empty directory.
- A child process is given an explicit environment: the tree under test first on
  `PYTHONPATH`, every `RESEARCH_RAG_` variable dropped, and the configuration home
  relocated. Nothing else is rewritten, so a child inherits what the app needs.
- The corpus a reader's example names is the reader's. This repository holds no
  corpus, no judged query set, and no patch.
- A failed arm is a refusal, never a row of zeroes. A non-zero exit, a missing
  report, and an unreadable report are all refusals.
- `EXIT_SOURCE_CHANGED` is its own status because it is not a refusal: the arms
  ran and the corpus they claim to have measured is not the one on disk.
- A run never overwrites a record. A run directory name carries the
  specification's name and a timestamp, and an existing one is refused.

## Working rule

- Use `pathlib.Path`, type hints, and JSON-serializable payloads.
- Every change is committed and pushed without waiting to be asked.
  - The validation below runs first.
  - The commit message says what the change does and why.
  - A change that is not on the remote is lost.
- The work ends with a clean tree.

## Validation

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q src tests
```

- A change to the isolation contract is proved against a real project with a real
  generation, not only against the test fixture: `rag-experiments run --dry-run`
  against a project the reader names, and then a real run whose record's
  `verdict` is `verified`.
- A change to the pinned settings is proved by reading the written file back with
  the app's own layer machinery, not by parsing it here.
- A change to the comparison is proved against a report shaped like the app's,
  with a column removed, because a column that silently prints as a dash is the
  failure this code is most able to produce.
