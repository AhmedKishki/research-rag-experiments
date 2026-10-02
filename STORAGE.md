# `storage/`

What a run leaves on disk, and what a reader may conclude from it.

Nothing here is authoritative about research-rag itself. The names the app owns —
its portable directory, its generation pointer, its source directory, its pinned
settings file — are read from the app's own code at run time and are deliberately
not spelled out in this repository. What follows is the layout *this* repository
owns.

## Two directories, two jobs

```
<workspace>/                                 the sandbox area: large and disposable
└── <run-id>-<arm>/
    ├── sandbox.json                         what this harness made, and from what
    ├── project/                             a real research-rag project, a copy
    │   ├── <source directory>/              the originals, byte for byte
    │   └── .research-rag/
    │       ├── project.json                 carries the original's project_id
    │       ├── source-*.json                the review state, as the app wrote it
    │       ├── config.toml                  pinned settings: every declared key
    │       └── runtime/
    │           ├── current.json             the selected generation
    │           └── generations/<id>/        the generations the run asked for
    ├── xdg/                                 the account overlay a child sees: empty
    └── (nothing else)

<runs>/                                      the run area: small and durable
└── <run-id>/                                 <specification name>-<timestamp>
    ├── run.json                             the record
    ├── judged/<split>.json                  the judged bytes this run measured
    └── <arm>/                               one directory per arm
        ├── arm.log                          every command, exit, and output
        ├── report-<split>.json              the app's harness report, per split
        ├── engine/                          a code arm's clone of the tree
        ├── engine-source.json               what the tree held when it was copied
        ├── engine-source.diff               its uncommitted tracked changes
        └── patch.diff                       the arm's patch, as applied
```

A run's directory name is never reused. Two runs of one specification are two
directories, so a dry run, a real run, and a repeated run each keep their own
reports, logs, and record, and none of them is overwritten by the next.

## The record

`run.json` is one JSON object, schema version 2. It is written on **every** exit of
a run: success, refusal, interruption, and a guard that could not be closed. A run
that measured nothing is the case most worth keeping, because its record is what
says which command refused and where its log is.

| Field | What it holds |
|---|---|
| `schema_version` | 2 |
| `toolkit_version` | the version that wrote it |
| `verdict` | the one field to read before any number; see below |
| `run` | name, start, finish, elapsed, directory, and whether the run completed |
| `error` | the message that stopped the run, verbatim |
| `interruption` | the interrupt that ended it, if one did |
| `measured_arm_count` | arms that measured every split they were asked for |
| `specification` | the specification, as read, with the run identifier |
| `engine` | root, revision, branch, dirty, and the content digest before and after |
| `judgments` | each split: its path, its digest, and the copy kept under the run |
| `source_project` | the source root, the pointed generation, and the guard |
| `arms` | one entry per arm, including the one that refused |

### `verdict`

Only `verified` counts as a measurement. It means the source project's bytes were
identical before and after, every arm measured every split, the engine's own files
did not move while the arms ran, and nothing stopped the run.

| Verdict | What it means |
|---|---|
| `verified` | all of the above held |
| `source_project_changed` | the corpus moved; the arms ran against something else |
| `engine_source_changed` | the engine's own files moved under the run |
| `interrupted` | the run was interrupted; arms already measured are kept |
| `incomplete` | some arms measured and at least one refused |
| `failed` | the run stopped with nothing measured, or the guard could not be closed |
| `no_arm_measured` | the run completed and no arm measured anything |

The order is the argument: a moved corpus says more than a moved engine, which
says more than a stopped run.

### `source_project.guard`

| Field | What it holds |
|---|---|
| `state` | `unchanged`, `changed`, or `unknown` |
| `before`, `after` | one digest each: path counts, byte counts, the digest, the excluded paths, and the account registry's digest |
| `unchanged` | the same answer as `state`, for readers that check one field |
| `differences` | `added:`, `removed:`, `changed:` per path, and the registry |
| `error` | why a digest could not be taken, when one could not |
| `note` | what was excluded and what a change means |

`unknown` is not `unchanged`. It is what the record holds when the closing digest
could not be taken at all, and nothing is verified from it.

### `engine`

`content_sha256` is a digest of the engine tree's `src`, `scripts`, and
`pyproject.toml`, taken before the arms and again after them. A git revision names
committed content and says nothing about a file someone is editing, so a code change
during a run would otherwise be measured and reported as one number.

### `arms[]`

Each arm holds `arm` (name, kind, overlay, base, patch, prepare commands), `engine`,
`checkout`, `sandbox`, `settings`, `environment`, `prepare`, `validate`, `measure`,
`reports`, `log`, `measured_splits`, `measured`, `failure`, `failure_stage`, and
`sandbox_removed`. An arm that refused holds the same shape, with whatever it had
produced and the message that stopped it.

### `settings`

`values` and `baseline_values` are the resolved settings for the arm and for the
arm's baseline; `baseline_layers` names the layer each key's value came from before
the overlay, and `arm_layers` names where each value came from after it, which is
what the overlay moved. `model_cache_root` is pinned as
an absolute path so a sandbox reads the shared binaries rather than looking inside
its own empty configuration home. `document_sha256` is the digest of the file written
into the sandbox.

### `environment`

Names dropped, names set, and nothing else: a full environment would copy whatever
the reader's shell held.

## Reading an older record

A record written before this layout keeps its own paths, and they are absolute, so
`rag-experiments compare` still reads them where they are. A record written by an
earlier schema has no `verdict` this file lists, and its `source_project.guard` has
no `state`: `compare` says so rather than presenting it as verified.