---
name: README.md
description: Install and run isolated research-rag experiments and interpret their retained evidence.
---

# rag-experiments

- Copy project sources, review state, and selected generations into disposable projects.
- Run settings overlays or patched engine copies against those projects.
- Retain judged inputs, settings, patches, logs, reports, and content hashes per run.
- Read quality metrics from the app's evaluation harness; the toolkit does not implement a second scoring system.

## Installation

```bash
uv sync
uv run rag-experiments --version
```

- `pyproject.toml` names the research-rag installation used for dependencies.
- `--app-source` names the checkout whose code is measured.
- Each run records that checkout's revision and content identity.
- A code arm must use compatible dependencies; changing code is not an automatic environment rebuild.

## Isolation boundary

- The runner prepares independent file copies and addresses projects by path.
- It does not register a copied project with `research-rag init`.
- The copy retains its source project ID so source-relative identities resolve.
- Process files, locks, staging, old logs, and the machine-local runtime pointer do not travel.
- The account settings are resolved into pinned settings; measured children use an isolated configuration home.
- The cached model binaries remain shared.
- Source and engine guards detect changes; a failed or unreadable guard cannot report success.
- This is cooperative filesystem isolation, not an OS sandbox.
  - Arbitrary patches and preparation commands run with the account's permissions.
  - They are not physically prevented from accessing other files or the network.
  - Use a container or restricted account before executing untrusted experiments.
- Offline measurements require cached models and fail rather than downloading missing models.

## Inspect and verify

```bash
uv run rag-experiments inspect --project /path/to/project
uv run rag-experiments verify --project /path/to/project
uv run rag-experiments verify --project /path/to/project --expect <digest>
```

- `inspect` describes the project and selected generation.
- `verify` hashes the original project without following links outside it.
- A standalone digest comparison detects a content change, not which process caused it.
- Concurrent edits to the original project invalidate a run's unchanged-source claim.

## Sandboxes

```bash
uv run rag-experiments sandbox create --project /path/to/project \
    --workspace workspaces --name probe
uv run rag-experiments sandbox list --workspace workspaces
uv run rag-experiments sandbox remove --workspace workspaces --name probe
```

- `sandbox remove` removes only a copy the toolkit owns.
- Run-owned copies use unique names; a dry run does not reserve the next real run's name.
- Retained reports stay in the run directory when disposable project copies are removed.
- `STORAGE.md` defines the layout and record fields.

## Judged inputs

```bash
uv run python examples/make-splits.py --judged /path/to/queries.json \
    --out judgments/partitions \
    --exclude-target 'target-id=explicit adjudication reason'
```

- Omit `--exclude-target` when all targets remain eligible.
- No benchmark-specific target is dropped by default.
- Each partition declares only the targets its queries use.
- Target families and declared question families stay together.
- The partition manifest records the input digest, allocation, and exclusion reasons.
- Outputs are exploratory partitions of an inspected input, not untouched holdout data.
- Keep private annotations and passage samples outside tracked source.
- New relevance, usability, contradiction, and no-answer labels require author adjudication.

## Run a specification

```bash
uv run rag-experiments run --spec /path/to/spec.json \
    --app-source /path/to/research-rag --workspace workspaces --runs runs --dry-run
uv run rag-experiments run --spec /path/to/spec.json \
    --app-source /path/to/research-rag --workspace workspaces --runs runs
```

- Both commands retain their own records under unique run directories.
- Dry runs copy inputs, prepare engine code, and validate pinned settings without searching.
- Real runs validate targets before measurement by default.
- `--no-validate` disables that separate pass; target resolution still occurs in the measurement harness.
- `--no-keep-sandboxes` removes disposable projects after successful measurements, not reports or logs.
- A specification names `source_project`, `judgments`, `harness`, and `arms`.
- `judgments` is one path or a list of `{"name", "path"}` partitions.

| Arm key | Meaning |
|---|---|
| `name` | Unique safe label. |
| `kind` | `settings` or `code`. |
| `overlay` | Declared engine settings applied to the resolved baseline. |
| `base` | Base revision for a code arm. |
| `patch` | Patch retained and applied to a disposable engine copy. |
| `prepare` | Commands executed inside the disposable project before measurement. |

- Preparation tokens name disposable paths: `{project}`, `{source}`, `{tree}`, `{sandbox}`.
- An arm-specific overlay takes precedence over shared engine-setting entries in `harness`.
- Each pinned settings file names every declared setting; the record separates baseline layers from overrides.
- Preparation failures and interrupted or incomplete runs retain their diagnostics and guard outcome.
- Hard process termination or power loss cannot execute a Python finalizer; inspect retained partial artifacts and verify the original separately.

## Correct budget experiments

- `examples/retrieval-depth-sweep.json` requests branch depths 40 and 80.
- `examples/candidate-window-ablation.json` requests a 40/80/160 by 20/30/50 budget grid.
- Replace their portable input paths with your project and generated judgment paths.
- Both candidate minimum and maximum are pinned to the requested branch depth.
- The rerank multiple and floor allow each grid cap to bind.
- A small candidate pool can still produce a shorter scored window.
- Verify the observed depth, pool size, scored window, and applied-reranking status before interpreting differences.

## Read a run

```bash
uv run rag-experiments compare --run runs/<run-id>
uv run rag-experiments compare --run runs/<run-id> --split exploratory-a
uv run rag-experiments compare --run runs/<run-id> --mode hybrid+rerank
uv run rag-experiments compare --run runs/<run-id> --json
```

- Report hashes are checked when the record carries them.
- Deltas require compatible report schemas.
- Known-item metrics retain their designated-passage meaning.
- `texts`, `dup`, and `lexov` are normalized-text and lexical-overlap diagnostics, not counts of independent claims.
- `xfam%` measures repeated slots across distinct target families, not repetition among paraphrases of the same question.
- `rej` records dense score rejection; `withheld` records answer-level withholding.
- p50 and p95 come from the app report; p95 is a percentile, not necessarily the slowest query.
- Legacy reports remain readable, but superseded word-set diagnostics are not relabelled as corrected metrics.
- Missing measurements print as dashes, not zeros.
- Failed, incomplete, changed-source, and unknown-guard records do not claim a complete verified measurement.

## Validation

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q src tests
```

- Integration tests exercise the run orchestrator, failure finalization, repeated runs, evidence retention, and source guards.
- Synthetic contract tests do not establish retrieval quality on a real corpus.
- An algorithm decision needs author-judged evidence, paired target-family uncertainty, and an untouched confirmation set.

## Acknowledgement and licence

- [research-rag](https://github.com/AhmedKishki/research-rag) uses [UltraRAG](https://github.com/OpenBMB/UltraRAG) by THUNLP, NEUIR, OpenBMB, AI9stars, and the upstream contributors.
- This independent toolkit applies experimental changes to disposable copies; it does not imply upstream endorsement.
- Apache-2.0; see `LICENSE`.
