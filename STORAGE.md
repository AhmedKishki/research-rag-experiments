# Storage

What a run leaves on disk, and what a reader may conclude from it.

Nothing here is authoritative about research-rag itself. The names the app owns —
its portable directory, its generation pointer, its source directory, its pinned
settings file — are read from the app's own code at run time and are deliberately
not spelled out in this repository. What follows is the layout *this* repository
owns.

## Two directories, two jobs

- Private annotation packets use the separate format at the end of this document.

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
├── <name>-<timestamp>/                      a measurement
│   ├── run.json                             the record
│   ├── judged/<split>.json                  the judged bytes this run measured
│   └── <arm>/                               one directory per arm
│       ├── arm.log                          every command, exit, and output
│       ├── report-<split>.json              the app's harness report, per split
│       ├── engine/                          a code arm's clone of the tree
│       ├── engine-source.json               what the tree held when it was copied
│       ├── engine-source.diff               its uncommitted tracked changes
│       └── patch.diff                       the arm's patch, as applied
└── <name>-prepared-<timestamp>/             a preparation: `--dry-run`
    ├── run.json                             the same record, kind `preparation`
    ├── judged/<split>.json                  the judged bytes it would measure
    └── <arm>/                               the same layout, with no report
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
| `run` | name, kind, start, finish, elapsed, directory, and whether it completed |
| `error` | the message that stopped the run, verbatim |
| `interruption` | the interrupt that ended it, if one did |
| `measured_arm_count` | arms that measured every split they were asked for |
| `prepared_arm_count` | arms that got as far as their copy, checkout, and settings |
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
| `verified` | all of the above held, and `run.kind` is `measurement` |
| `prepared` | every arm was resolved and no search ran; no quality claim at all |
| `source_project_changed` | the corpus moved; the arms ran against something else |
| `engine_source_changed` | the engine's own files moved under the run |
| `interrupted` | the run was interrupted; arms already measured are kept |
| `incomplete` | some arms measured and at least one refused |
| `failed` | the run stopped with nothing measured, or the guard could not be closed |
| `no_arm_measured` | the run completed and no arm measured anything |

The order is the argument: a moved corpus says more than a moved engine, which
says more than a stopped run.

`run.kind` is `measurement` or `preparation`, and the two are read differently. A
preparation is what `--dry-run` produces: the same guard over the same source
project, the same copies, the same checkout and settings resolution, and this same
record, with no prepare command and no search run. Its verdict is `prepared`, which
says every arm could be built and the corpus was not written to — and says nothing
about quality, because nothing was searched. Its exit status is 0 only when the
guard was unchanged and every arm prepared; a refused arm, an interrupted
preparation, or a corpus that moved is a nonzero status with the record's path.

`prepared` is a state of its own rather than a flavour of `verified` because a
preparation holds no measurement: `measured_arm_count` is 0 and `measured` is false
for every arm, and reading it as a measurement that found nothing would be exactly
the wrong reading.

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

## Private annotation packages

```text
annotations/<name>/
  private/
    manifest.json       selected input hashes, limits, coverage, package inventory
    key.json            question/item/pair origins and hidden consistency repeats
  reviewer/
    pool.json           opaque IDs, questions, passages, scholarly source context
    rubric.json         frozen proposed grading definitions
    template.json       complete inventory with every judgment pending
    review.html         self-contained offline author interface
    originals/          independent source copies with recorded SHA-256
```

- Schema version 1 defines the pool, package manifest, and returned annotations.
- Directories use owner-only permissions; files use owner read/write permissions.
- Publication reserves a new directory exclusively and installs the private manifest last.
- Ordinary failure removes only the publisher's own staging and claimed output.
- Hard termination may leave an ignored owner-only staging directory, not a completed packet.
- Package validation requires the complete recorded inventory and checks every file hash.
- Returned judgments stay outside the frozen packet.
- Private records retain every selected report/scorer digest and frozen input digest.
- Generation identity comes from the report's measured project, not historical judged-file corpus metadata.
- Canonical text matches the immutable artifact lookup's hash; originals match source manifest hashes.
- Generation text and reviewed bibliography must agree across selected snapshots.
- Missing retained copies cause refusal, never fallback to a live project.
- Different scorer revisions may supply candidates for labeling, with private provenance; no cross-scorer deltas are computed.

### Pool specification

| Key | Meaning |
|---|---|
| `schema_version` | `1`. |
| `name` | Safe packet label. |
| `inputs` | Explicit retained-run, arm, partition, and mode selections. |
| `inputs[].run` | Run directory, relative to the specification or absolute. |
| `inputs[].arms`, `splits`, `modes` | Nonempty selected-name lists; unknown names refuse. |
| `inputs[].include_results` | Include returned passages; defaults true. |
| `inputs[].include_collapsed_pairs` | Include both endpoints of retained collapse examples; defaults true. |
| `include_designated_targets` | Include the currently resolved designated passage without revealing its status; defaults true. |
| `max_candidates_per_query` | Refusal threshold for the mandatory union; defaults 60. |
| `max_pairs_per_query` | Retained collapse pairs plus sampled other pairs; defaults 6; mandatory overflow refuses. |
| `repeat_fraction` | Hidden consistency aliases; defaults 0.1, maximum 0.25. |
| `family_limit` | Optional pilot limit retaining all queries of each selected frozen family. |
| `seed` | Optional reproducible private ordering seed; omission uses random entropy. |

- Equal text in different sources retains distinct attribution.
- Bind questions to frozen target IDs and resolved report chunk IDs, not historic measurement chunk IDs.
- Resolve families from frozen target/question declarations, not a report override.
- Retain exclusions, limits, incomplete source-pair examples, and pair-sampling scope privately.
- Source relations and human grades are never inferred automatically.

### Returned judgments

- `annotation/schema.py` owns the frozen proposed rubric and permitted typed values.
- `pool_id`, `pool_sha256`, and `rubric_sha256` bind the response to its packet.
- `rubric_acknowledged` and `annotator` record acknowledgment and identity.
- Each `items[]` row contains `item_id`, `relevance`, `usability`, `source_verified`, and text `notes`.
- Each `pairs[]` row contains `pair_id`, `relation`, and text `notes`.
- `null` means pending, never irrelevant or unusable.
- Missing, duplicate, unknown, foreign-pool, and wrong-type rows refuse validation.
- Explicit uncertainty requires adjudication.
- Contradiction and independent-corroboration assertions require notes and original-source checks for both endpoints.
- Consistency aliases map to the same primary card privately; they are not additional annotators.
- `complete` means the acknowledged inventory is filled, not that a policy improved.
- `policy_ready` remains false; no relevance or latency metric is computed here.

## Author judgment handoff

- `annotation export` produces a new private JSON file only after completed, acknowledged, adjudicated author labels pass validation.
- Relevance and usability conflicts on hidden repeat aliases require adjudication.
- Repeat aliases are excluded from the primary judgment inventory.
- `schema_version: 1` and `protocol: author_pool_v1` distinguish the handoff from retrieval reports and annotation responses.
- `provenance` retains pool, rubric, annotation, generation, and retained-report hashes.
- The frozen rubric uses SHA-256 over sorted-key UTF-8 JSON with fixed separators.
- `protected_roots` prevents outputs within packets, retained inputs, original projects, or the app checkout.
- `questions` retains the frozen question and family identifiers for paired analysis.
- `judgments` retains the author labels and each generation-bound occurrence's source path, locator, content digest, and chunk identifier.
- `relations` contains only authored pair labels; unknown relationships are not inferred.
- `conditions[].rankings` joins the scored occurrence identifiers to actual returned lists, requested reply depth, observed branch depth, and applied rerank window.
- Each scored condition covers the same frozen questions and graded pool.
- Material selected only for collapse review is not automatically a scored condition.
- Scoring after re-chunking needs new source-span resolution; occurrence labels are not blindly portable between generations.
- Handoffs and app-owned score reports are generated private artifacts, not tracked source or source documents.
