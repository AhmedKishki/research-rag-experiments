---
name: AGENTS.md
description: State the experiment toolkit's safety, measurement, and engineering constraints.
---

# AGENTS.md

## Purpose and scope

- Measure research-rag through its own evaluation harness against disposable project copies.
- Keep this developer tool independent of the collection's server release history.
- Do not change retrieval defaults, activate a live generation, or invent human judgments as part of a harness repair.

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
