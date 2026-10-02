# STORAGE

The on-disk format. Every file and directory a run or a sandbox creates, and the
rule that decides what travels from a source project and what does not.

## Two directories, both disposable

A run writes into two places, and neither is the project being measured.

| Directory | Holds | Default |
|---|---|---|
| the workspace | one directory per sandbox, each holding a project and a record | `workspaces` |
| the runs directory | one directory per run, holding a record and each arm's evidence | `runs` |

Both are ignored by git. A sandbox holds a copy of a corpus and a generation, and
is large; a run holds generated reports. Regenerate one instead of editing it.

## A sandbox

```text
<workspace>/<name>/
  sandbox.json      what this harness made, and from where
  project/          the project root: a real research-rag project, byte for byte
  xdg/              an empty configuration home for the children a run spawns
  <arm>/            only for a run: the arm's own directory
    arm.log         every line this harness emitted for the arm
    report.json     the report the app's own harness wrote
    engine/         only for a code arm: the disposable checkout
```

`project/` holds nothing but what the app itself writes, so it is a valid project
with nothing removed. The record sits beside it because a file this harness added
to a project root would be a file the app does not know about.

`xdg/` is empty and is named by `XDG_CONFIG_HOME` for every child a run spawns.
The app resolves its account-wide settings overlay beneath that variable, so
pointing it here is what keeps one account's settings out of a measured run. The
model cache is deliberately **not** relocated: those are immutable binaries, and
re-deriving them would need a network this harness does not use.

### What a copy carries, and what it does not

The copy is an allowlist. That is the whole rule, and it is why the two exclusions
below are the only two names this harness has to know.

Carried:

- The source directory, every file in it, at the same source-relative path. The
  directory's name is read from the project's descriptor rather than assumed, so a
  project whose originals are not in `sources` is copied correctly.
- The portable review state: the descriptor, the source catalogue, the reviewed
  metadata, the source exclusions, the passage exclusions, and the bundles
  directory, each at the name the app gives it.
- Under the runtime directory: only the selected-generation pointer, and the
  generation directories the caller named. Empty directories for the logs, the
  staging area, the failure area, and the gateway's runtime are created, because
  the app expects them to exist.
- The `project_id`, because it is what makes every `source_id` in the review files
  still resolve and a re-ingestion in the sandbox produce the same ids as one in
  the original would.

Not carried:

- Everything else under the runtime directory. That is the process's id, port,
  and terminal; the project lock; a build in progress under the staging area; the
  build journal that names a pending activation; and every generation that was not
  named. All excluded by construction rather than by a list of names.
- The runtime pointer that names a relocated runtime directory. It is
  machine-local, and a copy carrying it would read the original's generations.

A symbolic link anywhere this copies is refused rather than followed or
recreated. The app refuses a symlinked original and a symlinked settings file, so
a sandbox holding one would be a project the app cannot measure, and one that
silently dropped one would be a project that differs from the source in a way no
record would show.

### `sandbox.json`

Written with the same rule the app uses for its own records: readable, stable
key order, and one trailing newline.

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | integer | `1`. |
| `toolkit_digest` | string | A digest of everything below, so a record cannot be edited without the digest moving. |
| `name` | string | The one path segment this sandbox is called. |
| `root` | string | The project root inside the workspace. |
| `source_root` | string | The originals inside the copy. |
| `project_id` | string | The identifier the copy shares with the source project. |
| `generation_ids` | list of string | The generation directories that were copied. |
| `settings_file` | string | Where a run pins the settings. Nothing has written it yet. |
| `config_home` | string | The empty configuration home a child is pointed at. |
| `created_at` | string | UTC, ISO 8601 with microseconds and a `Z`. |
| `byte_count` | integer | Bytes copied, under the project root. |
| `file_count` | integer | Files copied, under the project root. |
| `elapsed_seconds` | number | How long the copy took. |
| `origin` | object | The source project, the layout read from the app, and the generation the source selected. |

A directory in a workspace with no `sandbox.json` is not reported by `sandbox
list` and is not removed by `sandbox remove`. The marker is what makes destruction
safe.

## A run

```text
<runs>/<specification-name>-<timestamp>/
  run.json          the record
  <arm-name>/
    arm.log         every line this harness emitted, in order
    report.json     the report the app's own harness wrote
    engine/         only for a code arm: the disposable checkout
```

The run directory name carries the specification's name and a timestamp, and a run
never overwrites an existing one. A run that fails leaves its record: a failure
with its provenance is more useful than no failure at all.

### `run.json`

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | integer | `1`. |
| `toolkit_version` | string | The installed distribution's version. |
| `verdict` | string | `verified`, `source_project_changed`, or `no_arm_measured`. |
| `run` | object | `name`, `started_at`, `finished_at`, `elapsed_seconds`, `directory`. |
| `specification` | object | The specification as read: path, name, source project, judged set, generations, harness, arm count. |
| `engine` | object | The tree under test: `root`, `revision`, `branch`, `dirty`, `untracked_file_count`, `harness`, `version`. |
| `judgments` | object | `path`, `sha256`, `byte_count`. |
| `source_project` | object | `project_root`, `pointed_generation`, and `guard`. |
| `arms` | list of object | One entry per arm, in the order the specification listed them. |
| `source_project.guard` | object | `guarded_entries`, `before`, `after`, `unchanged`, `differences`. |
| `source_project.guard.before` | object | `root`, `file_count`, `byte_count`, `digest`, `elapsed_seconds`. |
| `arms[].arm` | object | The arm as read: `name`, `kind`, `overlay`, `base`, `patch`, `prepare`. |
| `arms[].engine` | object | The engine that arm measured through. |
| `arms[].checkout` | object or null | For a code arm: `root`, `base_revision`, `patch`, `patch_sha256`, `applied_diff`, `dirty_before_patch`, `elapsed_seconds`. |
| `arms[].sandbox` | object | The sandbox as `sandbox.json` describes it. |
| `arms[].settings` | object | `key_count`, `document_sha256`, `overridden`, `layers`, `values`, `written_to`, `config_home`. |
| `arms[].settings.layers` | object | Which layer each baseline value came from, keyed by setting name. |
| `arms[].prepare` | list of object | One entry per preparation command: `command`, `exit_code`, `elapsed_seconds`, `stdout`, `stderr`. |
| `arms[].measure` | object | The measurement command's `command`, `exit_code`, `elapsed_seconds`, `stdout`, `stderr`, `log`. |
| `arms[].report` | string | Where the report was written. |
| `arms[].log` | string | Where the arm's log was written. |

`layers` is a map from a setting's dotted name to the layer that supplied its
baseline value. The layer names are the app's own, read from
`research_rag.project.settings_layers`, and they are recorded rather than
re-spelled here so a rename in the app does not leave this file asserting a layer
that no longer exists.

## The guard

The guard covers two entries beneath a project root, and they are the two the app
owns: the originals and every piece of state derived from them.

A digest is the SHA-256 of every regular file's relative path, its size, and its
content digest, folded together in sorted path order. A rename and a rewrite are
therefore different changes, and neither can pass as the other. A path that is a
symbolic link, a socket, or anything that is not a regular file is skipped rather
than followed, so a link out of the project cannot make the guard read outside it.
A guarded entry that is itself a symlink is refused.

Two projects holding the same bytes have the same digest, which is what lets a
digest printed on one machine be compared on another. Comparing two snapshots of
different roots is refused rather than read as "everything changed", which would be
indistinguishable from a rewrite.

## The pinned settings file

A run writes one file into each sandbox, at the path the app's own
`project_config_path` names, naming every setting the engine under test declares:

```toml
# Written by rag-experiments. Every setting the engine under test declares
# is named here, so an arm's measurement cannot be moved by the account
# overlay, an environment variable, or a later change to a packaged
# default. An arm's overlay is already applied to these values.

[chunking]
headers = false
overlap = 64
size = 384

[retrieval]
rrf_k = 120
```

Its format is TOML because that is what the app's settings layers read, and the
value domain is the app's own setting kinds, so each value is a basic string,
integer, float, or boolean. `tests/test_engine.py` reads a written file back
through the app's own layer machinery rather than parsing it here, so a document
this repository could write and the app could not read would fail a test.
