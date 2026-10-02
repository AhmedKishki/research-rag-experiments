# `report/`

Owns what a run leaves behind and how a reader reads it.

What it may never do:

- Compute a quality metric. A column in the comparison table is read out of the
  report the app's harness wrote, so a number here and a number in the app's own
  `MEASUREMENTS.md` cannot come from two implementations.
- Present an unverified run as verified. The guard's verdict is a field, not a
  preamble: a run whose source digest moved says so in the record and in the
  table.

| Module | Owns |
|---|---|
| `record.py` | the run record: what ran, against what, and what the guard said |
| `compare.py` | the table that reads a run's records side by side |
