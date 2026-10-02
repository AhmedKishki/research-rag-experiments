# `experiment/`

Owns what an experiment is: a run specification, the arms inside it, and the
execution of one arm.

What it may never do:

- Score anything. An arm runs the app's own evaluation harness as a subprocess and
  the report it wrote is the score. A metric computed here would be a second
  implementation of a number the app already publishes.
- Touch a source project. Everything below runs inside a sandbox, and the
  experiment's own view of the source is the digest `sandbox/guard.py` took.
- Assume a variant is expressible as a setting. That is what the two arm kinds
  are for, and neither is allowed to be silently reduced to the other: an arm
  that asked for a code change and got a setting is a different measurement.

| Module | Owns |
|---|---|
| `spec.py` | the run specification, its arms, and what a valid one contains |
| `checkout.py` | a disposable copy of the engine tree, for an arm that changes code |
| `execute.py` | running one arm, and the result it returns |
