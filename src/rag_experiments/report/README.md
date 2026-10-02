# `report/`

Owns what a run leaves behind and how a reader reads it.

What it may never do:

- Compute a quality metric. A column in the comparison table is read out of the
  report the app's harness wrote, so a number here and a number in the app's own
  `MEASUREMENTS.md` cannot come from two implementations.
- Present an unverified run as verified. The guard's verdict is a field, not a
  preamble: a run whose source digest moved says so in the record and in the
  table.
- Lose the record of a failed run. `record.py` writes a record on every exit —
  success, refusal, interruption, and a guard that could not be closed — because a
  failure with its provenance names the command that refused and where its log is.

| Module | Owns |
|---|---|
| `record.py` | the run record: what ran, against what, and what the guard said |
| `compare.py` | the table that reads a run's records side by side |

`record.py` owns the verdict, and it is the narrowest statement this repository
makes: `verified` means the source project's bytes were identical before and after,
every arm measured every split it was asked for, the engine's own files did not move
while the arms ran, and nothing stopped the run. Every other state has its own name —
`source_project_changed`, `engine_source_changed`, `interrupted`, `incomplete`,
`failed`, `no_arm_measured` — so a reader is never told a run counted when it did
not. A guard that could not be closed is `unknown`, which is not `unchanged`.

`STORAGE.md` at the repository root holds the schema this module writes and the
layout the files live in.
