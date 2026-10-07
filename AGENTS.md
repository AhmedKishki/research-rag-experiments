---
name: AGENTS.md
description: State the experiment toolkit's safety, measurement, and engineering constraints.
---

# AGENTS.md

## Purpose and scope

- Measure research-rag through its own evaluation harness against disposable project copies.
- Keep this developer tool independent of the collection's server release history.
- Do not change retrieval defaults, activate a live generation, or invent human judgments as part of a harness repair.

## Resource priority and sequencing

- Every subprocess this toolkit launches on a measured, scored, benchmarked, or preparation path is both low priority and resource-bounded.
  - Enforcement lives in one place: `niceness.run_low_priority` sets priority before `exec` and delegates the launch to `resource_limits.launch`.
  - `resource_limits.launch` starts the whole child tree in a cgroup v2 transient scope with enforced `MemoryMax`, `MemorySwapMax=0`, `CPUQuota`, and `TasksMax`, so descendants a measured program spawns share one bound.
  - Niceness alone is insufficient: it decides who wins the CPU, not how much memory a tree may hold, so a run under `nice` can still drive the host into swap.
  - `MemoryMax` is a cgroup RSS-plus-page-cache limit; it is never `RLIMIT_AS`, which would refuse a child that has reserved addresses it does not use.
  - Do not add a raw `subprocess.run`/`Popen` on any such path.
- The rule is fail-closed at both layers: a host or child that cannot establish niceness 19, or a host that cannot enforce a cgroup bound, refuses the launch instead of running at normal or unbounded priority.
  - Capability is verified by creating a memory-limited scope and reading `memory.max` back, not by trusting the request.
- Heavy work is serialized across separate invocations by a cross-process file lock; a second run waits a bounded time and then refuses rather than overlapping.
  - A timeout or an interrupt stops the scope, which ends every descendant, and cannot leave the tree running.
- The numerical thread pools of a measured child are capped by default; a reader's own value for one is left alone.
- Run evaluations, benchmarks, preparation, and test suites sequentially, one at a time; never start a second while one is running.
- Issue validation commands through `nice -n 19`; `README.md` holds the commands and the override names.
- `STORAGE.md` owns the record schema for the applied limits.

## Source protection

- Source projects are read-only inputs.
- Never call `resolve_config` or `research-rag init` on a source project.
- Copies keep the project ID and source-relative paths but are never registered.
- Preparation tokens must name disposable paths, including `{source}`.
- Refuse path escapes, unsafe links, and output locations within the original project or engine tree.
- Do not write Python bytecode into a live engine checkout.
- Model binaries may be shared; account configuration and project registry writes may not be shared.
- Isolation is cooperative, not an OS security boundary.
  - Arbitrary code has the account's filesystem and network permissions.
  - Untrusted experiments require a separate restricted execution environment.
- Guard failures and concurrent external writes must invalidate the unchanged-source claim.
- Ordinary exceptions and interrupts must finalize the guard and retain a record.
- Hard termination cannot run a finalizer; partial evidence must not imply successful verification.

## Evidence and provenance

- Each run owns immutable input copies, reports, logs, patches, settings, and content hashes.
- Never reuse an earlier run's evidence paths.
- Removing disposable projects must not remove measurement evidence.
- Record engine revision and content identity; a dirty flag alone cannot identify changed code.
- Capture subprocess output on success and failure.
- A failed arm is not a row of zeros.
- A run is verified only when every requested measurement succeeds and its guards complete unchanged.
- An unknown guard is not a healthy guard.
- `STORAGE.md` owns record schemas, verdict definitions, and the on-disk layout.

## Measurement contract

- The app owns metric definitions and computations.
- The toolkit reads those metrics and presents differences; it does not implement a second relevance or latency metric.
- A missing measurement is not zero.
- Do not compare deltas across incompatible metric definitions.
- Preserve readable legacy records without relabelling their defective diagnostics as corrected measures.
- Record observed branch depths, candidate pools, scored windows, and rerank fallback.
- Requested caps are not measured budgets.
- Text equality, lexical overlap, and repetition are diagnostics, not judgments of independent evidence or contradiction.
- Separate repetition within a target/question family from repetition across families.
- A final result list follows existing repetition collapse; it cannot establish whether the reranker spent budget on copies.
- A partition of an inspected benchmark is exploratory, not an untouched holdout.
- No target exclusion is implicit; retain its ID and adjudication reason with the inputs.
- Annotation preparation may collect evidence and validate author input; it must not assign grades from ranks, scores, or designated targets.
- Keep provenance keys and consistency-repeat mappings outside the reviewer packet.
- Keep frozen packets separate from returned judgments; completed annotations are not policy acceptance.
- A policy decision requires author-judged evidence and target-family-aware uncertainty, not a one-query change or a count of correlated slots.

## Ownership and writing

| File | Owns |
|---|---|
| `README.md` | Installation and command usage. |
| `STORAGE.md` | Artifact and record formats. |
| `AGENTS.md` | Constraints not derivable from implementation details. |
| `examples/` | Portable specifications and reproducible partition preparation. |
| Folder README | Concern boundaries and links to the owner of a contract. |
| Module docstring | Its algorithm and limitations. |

- Keep one owner for each rule; link instead of repeating it across documents.
- Markdown states current constraints and limits; Git owns history.
- Use direct language, type hints, `pathlib.Path`, and JSON-serializable records.
- Do not commit private judgments, passage samples, corpora, or generated run artifacts.
- Test the whole run lifecycle as well as helpers.
- Validation commands live in `README.md`.
- Commit and push tested implementation changes; stage only intended files.
- Push a child repository before changing its collection pointer.
- Report residual safety and evaluation limits without claiming that passing tests establish retrieval quality.
