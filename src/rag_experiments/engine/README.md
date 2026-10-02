# `engine/`

Owns the research-rag tree an experiment measures, and the settings that tree
resolves. It may never copy the app, reimplement a setting, or answer a
question about the engine that the app's own code can answer.

Three things live here because all three depend on *which* tree is under test:

- `locate.py` names a tree, its git revision, whether it was dirty when the run
  started, and a content digest of the files a measurement can be moved by
  (`src`, `scripts`, and the packaged metadata). Every git question it asks is
  asked with `GIT_OPTIONAL_LOCKS=0`, because a read that refreshes an index writes
  to the tree it read, and that tree is a developer's working copy.
- `harness.py` reads the harness script's own module-level constants — the modes it
  measures, the label its reranked mode carries, and the report schema version it
  writes — from its syntax tree. A harness whose report shape moved is refused
  rather than parsed into columns that would print as a dash.
- `resolve_settings.py` runs inside the tree under test. It is a module the parent
  invokes as a subprocess with that tree first on `PYTHONPATH`, so `research_rag`
  resolves to the patched engine rather than to the installed one. That is what
  makes a code arm's new settings, and its changed defaults, visible to the harness
  instead of being validated against the wrong registry.
- `runner.py` spawns that module and states the environment a child is given.

| Module | Owns |
|---|---|
| `locate.py` | which tree, which revision, which bytes |
| `harness.py` | what the engine's harness declares it measures |
| `resolve_settings.py` | the pinned settings document, inside the tree |
| `runner.py` | the request and the environment a child measures in |

## What a child's environment says

The record names what a child was given, and nothing else: a full environment would
copy whatever the reader's shell held.

- The tree's `src` is prepended to `PYTHONPATH`, so the app and the gateway it
  spawns both import the tree under test.
- Every `RESEARCH_RAG_` and `RESEARCH_ULTRARAG_` variable is dropped. The first is
  the app's settings layer, which outranks the pinned file; the second is what the
  app's own harness reads its cache root, runtime root, and dense backend from.
  Either would decide a measurement without appearing anywhere in the record.
- `XDG_CONFIG_HOME` points at a directory with no file in it, so the account overlay
  resolves somewhere empty at measurement time.
- `PYTHONDONTWRITEBYTECODE` is set, because the tree under test is frequently a
  developer's working checkout and an import must not write beside it.
- The model cache is deliberately **not** relocated: those are immutable binaries, a
  run downloads nothing, and the absolute path the app resolves is pinned into the
  sandbox's settings so a copy does not depend on an inherited cache home.
- Nothing else is touched, so a child inherits the machine's own paths for whatever
  the app legitimately needs.

## What the resolution answers

Two resolutions, because they answer different questions. The baseline applies no
overlay at all, so its layer for every key is where that key's value came from
before the arm touched anything: the packaged defaults, the account overlay
included, and the project's own file, with the environment layer dropped. The arm's
resolution then applies the overlay and its layers show what moved. Reporting the
arm's layers as the baseline would name the command line for every overridden key,
which is true of the arm and says nothing about what it was compared against.

The output document names every setting the tree declares, not the handful an arm
changed, because pinning the whole set is what makes a run reproducible: the account
overlay, a shell variable, and a later change to a packaged default are all below a
project file that states every value.