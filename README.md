# rag-experiments

An experiment toolkit for `research-rag` retrieval. It copies a project into a
disposable sandbox, measures one or more variants inside the copy, and records
what each variant ran and measured. It never writes inside the project it
measures, and it proves that by digesting that project's bytes before a run and
again after it.

A quality number comes from the app's own evaluation harness, run as a
subprocess against the sandbox. This toolkit computes no metric of its own, so a
number here and a number in the app's own `MEASUREMENTS.md` cannot come from two
implementations.

The engine under test is a `research-rag` checkout, installed as an editable
dependency. A variant is either a change to the settings that engine already
declares, or a patch applied to a disposable copy of that checkout. Nothing about
an experiment reaches the app unless it is promoted into it afterwards.

## Installing

```bash
uv sync
```

`[tool.uv.sources]` names the `research-rag` checkout to measure. It points at
the working tree beside this repository so a measurement names the code being
edited. Point it at a clone and a tag to measure a published engine instead:

```toml
[tool.uv.sources]
research-rag = { git = "https://github.com/AhmedKishki/research-rag", tag = "v1.0.0" }
```

## What it can and cannot do

- It can sweep every setting the engine declares, at a pinned value, with the
  account overlay and the environment held out of the way.
- It can measure engine code that does not exist yet, by patching a disposable
  copy and naming the base revision and the diff in the record.
- It can rebuild a generation inside a sandbox, through a `prepare` command, so
  an extraction change is measured without touching the real corpus.
- It can say exactly which engine revision, corpus digest, settings, and commands
  produced a number.
- It cannot answer a retrieval question on its own. It has no corpus, no index,
  and no query set of its own; it copies a project and hands it to the app.
- It cannot produce a measurement it can attribute. An arm that fails, a report
  that will not parse, and a source project whose bytes moved are all refusals
  rather than rows of zeroes.
- It cannot reach the network. `--prefetch-models` and `--repair-runtime` are the
  app's own operations, and a run that needs a model that is not cached fails.

## Before a run: `inspect`

```bash
uv run rag-experiments inspect --project /path/to/project
```

This writes nothing. It says where the originals are, which generation the
project selects, what a copy would carry, and what would not be carried.

## `verify`

```bash
uv run rag-experiments verify --project /path/to/project
uv run rag-experiments verify --project /path/to/project --expect <digest>
```

Prints the project's digest, or compares it to one printed earlier. This is the
guard on its own, for work that was not done through this harness. Hashing a
corpus and its generations is under a second, so there is no cheaper mode that
would miss the one write that mattered.

## Sandboxes

```bash
uv run rag-experiments sandbox create --project /path/to/project \
    --workspace workspaces --name probe
uv run rag-experiments sandbox list --workspace workspaces
uv run rag-experiments sandbox remove --workspace workspaces --name probe
```

A sandbox is a real `research-rag` project: the same originals under the same
source-relative paths, the same review state, and a selected generation. Under
the runtime directory only the selected-generation pointer and the named
generations are carried, so a process's id, port, terminal, project lock, a build
in progress, and its journal are excluded by construction rather than by a list
of names. The runtime pointer that names a relocated runtime is not carried
either: it is machine-local, and a copy holding it would read the original's
generations.

A copy keeps the original's `project_id`, which is what makes every `source_id`
in the review files still resolve and a re-ingestion produce the same ids. For
that reason `research-rag init` is never called on a copy: `init` registers a
project in the account's pointer file, and registering a copy would evict the
original's record. A sandbox is addressed by path and is never registered.

The record of a sandbox is written beside the project, not inside it, so the
project root holds nothing but what the app itself writes.

## Running an experiment

A run specification is JSON. `examples/candidate-window-ablation.json` is a
two-split one and `examples/retrieval-depth-sweep.json` a single-set one.

```bash
uv run rag-experiments run --spec examples/retrieval-depth-sweep.json \
    --app-source /path/to/research-rag --dry-run
```

`--dry-run` performs the whole preparation a real run performs — copying the
corpus, applying any patch, resolving and pinning the settings — and stops. It is
the check to run before an expensive run, and what it refuses is what a real run
would refuse.

A real run asks the app's harness to resolve every judged target before it
searches, once per arm and once per split, and reports each outcome. A target that
does not resolve uniquely would make every number for that split meaningless, and
finding that costs seconds here rather than after an hour of cross-encoding.
`--no-validate` skips the check, which is only safe when the judged set and the
generation have not changed since the last run.

Without `--dry-run`, each arm is measured in its own sandbox, and the run writes
one record under `--runs`. Each arm may declare:

| Key | Meaning |
|---|---|
| `name` | How the arm is labelled in the table and the record. Required and unique. |
| `kind` | `settings` or `code`. |
| `overlay` | Settings to apply on top of the resolved baseline. |
| `base` | A revision for a code arm to move to before its patch applies. |
| `patch` | A unified diff applied to the disposable checkout. |
| `prepare` | Commands to run before measuring, for an arm that rebuilds a generation. |

`judgments` is one path, or a list of `{"name", "path"}` splits. Two splits in
one run is the point: a development number and a held-out number are only
comparable when the same arms produced both, and two specifications listing the
same arms can drift apart between two runs. The split names have to differ,
because the record and the table key on them.

The `harness` block holds the per-run conditions. Four of its entries are
settings rather than arguments, and they are folded into each arm's overlay
before the settings are pinned, so the pinned file and the command agree about
the engine: `offline`, `dense_backend`, `model_cache_root`, `reranker_model`.
The rest are passed as flags.

`prepare` commands may use four tokens, and only four: `{project}`, `{source}`,
`{tree}`, `{sandbox}`. An unknown token is left as written, so a command that
expected a substitution this harness cannot make fails inside the command and
names the token.

## The pinned settings

Every arm gets a settings file naming **every** setting the engine under test
declares, not only the ones the arm changed. That is what makes a run
reproducible: the account overlay, a `RESEARCH_RAG_` variable, and a later change
to a packaged default are all outranked by a project file that states every
value. The values are resolved by the engine under test, through its own registry
and its own layer stack, and the record states which layer each baseline value
came from.

An unknown key or an out-of-range value is refused by the engine and reported
verbatim, before anything is measured.

## Reading a run

```bash
uv run rag-experiments compare --run runs/<run-directory>
uv run rag-experiments compare --run runs/<run-directory> --json
```

The table is one block per split and per mode, because a split is a different
set of questions and a mode is a different question asked the same queries. The
first arm the specification listed is the row every other arm is read against. A
column the app's report stopped carrying prints as a dash, because a missing
metric and a zero metric are not the same fact.

Each block states every arm's candidate window and rerank budget above it, read
from that arm's pinned settings. When they differ across arms, the block says a
difference is a difference in budget as well as in policy: a candidate-window
sweep varies the window on purpose, and a policy ablation is only comparable at a
fixed one. Latency is shown as both p50 and p95, each a value that was measured
rather than an interpolation.

`--split` tabulates one split and `--mode` one mode, for reading a single number
in isolation.

The record names, for the run: the engine revision, branch, and whether its tree
was dirty; the source project's digest before and after; the judged set's path,
digest, and size; and for each arm, the overlay, the pinned settings and their
layers, the disposable checkout's base revision and applied diff, the sandbox,
every command run, the exit status, the duration, and the report's path.

### The verdict

A run ends in one of three states:

| Verdict | Meaning | Exit status |
|---|---|---|
| `verified` | the source project's bytes were identical before and after | `0` |
| `source_project_changed` | they were not, so the numbers do not describe the corpus on disk | `3` |
| `no_arm_measured` | no arm produced a report | `0` |

`source_project_changed` has its own status because it is not a refusal: the arms
ran, and the corpus they claim to have measured is not the one that exists. The
record is still written, and the differences are named in it and in the table.

## Validation

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q src tests
```

A change to the harness is proved against a real project with a real generation,
not only against the fixture:

```bash
uv run rag-experiments run --spec examples/retrieval-depth-sweep.json \
    --app-source ../ultra-rag-mcp-servers/research-rag --dry-run
```

## What this repository does not contain

- A corpus. Every path in an example names a project the reader has.
- A judged query set. The ones an example names belong to the app or to the
  reader, and no example names a split it does not ship.
- A patch. `patch` names one the reader wrote; a code arm with no patch and no
  base is refused.
- A quality metric. See the first paragraph.

## Acknowledgement

`research-rag` is built on [UltraRAG](https://github.com/OpenBMB/UltraRAG) by
THUNLP, NEUIR, OpenBMB, AI9stars, and the upstream contributors. This toolkit
measures that app and does not modify, patch, or redistribute it. UltraRAG does
not endorse this toolkit.

## Licence

Apache-2.0. See `LICENSE`.
