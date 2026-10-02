# `experiment/`

Owns what an experiment is: a run specification, the arms inside it, and the
execution of one arm.

What it may never do:

- Score anything. An arm runs the app's own evaluation harness as a subprocess and
  the report it wrote is the score. A metric computed here would be a second
  implementation of a number the app already publishes.
- Touch a source project. Everything below runs inside a sandbox, and the
  experiment's own view of the source is the digest `sandbox/guard.py` took. The
  only thing this folder ever resolves against the source is a read of its settings.
- Assume a variant is expressible as a setting. That is what the two arm kinds
  are for, and neither is allowed to be silently reduced to the other: an arm
  that asked for a code change and got a setting is a different measurement.

| Module | Owns |
|---|---|
| `spec.py` | the run specification, its arms, and what a valid one contains |
| `checkout.py` | a disposable clone of the engine tree, for an arm that changes code |
| `execute.py` | running one arm, and the result it returns |
| `report_schema.py` | whether the report an arm's harness wrote is the measurement asked for |

## A report that exists is not a report that answers the question

`report_schema.py` reads a report against a contract rather than sampling keys, and
an **absent** fact is refused as firmly as a wrong one: a report that names no judged
file and one that names the wrong file are equally unusable, and only the second
reads as a disagreement a reader can act on. It refuses a report whose

- schema version is not the one the engine's own harness declares;
- summary lacks a mode the run asked for, or carries a row for a mode it did not;
- judged block is absent, names another file, or states a query count or an
  evaluated count that differs from what this run's selection yields from that file;
- classes or skipped targets differ from the ones this run named;
- `top_k` or `deep_top_k` is absent, or differs from what the run asked for — zero
  being the engine's stated "no deep pass", not an absence;
- rows state no depth, state a depth other than the run's, repeat a query id, name
  no query, or cover a query set that is not this run's selection;
- per-mode summary count is absent or disagrees with the rows beneath it;
- reranked row has any query that did not rerank, fell back, or returned no
  candidates. A query with nothing to rank never reached a reranker, so it cannot
  evidence that one ran, even though it is reported as `reranked`.

The expected query set is computed here from the judged file and the run's own
`classes`, `limit`, and `skip_targets`, in the engine's order, so it is the engine's
set rather than a second implementation of a similar one. It is computed while the
arm is prepared, so a specification that selects a class the file does not hold is
refused before an hour of measurement rather than after it.

Every fact compared comes from the engine's own harness, from the command this
harness built, or from the judged file, so a mismatch means the engine and this run
disagree rather than that this file guessed.

## A refusal keeps what it had

A non-zero exit, a missing report, a report that does not match the run's own
arguments, and an invalid setting are all refusals rather than a row of zeroes. Each
one raises `ArmFailure` carrying the evidence gathered so far — the commands it ran,
the reports it wrote, the log that holds the output — so the run that stops writes a
record naming the arm that stopped it. An interrupt is not a refusal and is not
wrapped; the arm's partial evidence rides out on the exception so the run's record
still carries it.

The arm's log is appended and flushed as the arm runs, including each subprocess's
exit status, stdout, and stderr. It is the only thing that survives a closed
terminal.

## What a code arm measures

`checkout.py` clones the tree with `git clone --local --no-hardlinks`, so the clone
holds its own objects and its own index and nothing it does writes to the tree it
read. The arm's patch is applied in the clone and copied into the run's directory, so
the variant can be reproduced from the record. What the source tree held is captured
rather than assumed: its base revision, the full diff of tracked files against that
revision, and a digest for every untracked file, all written into the run. A tree
whose dirty state is applied to the clone when no revision was named, because "the
engine as it stands" is what a settings arm measures. An untracked symbolic link is
refused, and a tree with no repository of its own is refused with the reason.