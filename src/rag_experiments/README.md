# `rag_experiments`

What the package owns: the machinery that runs a research-rag retrieval experiment
outside the project it measures, and the record that says what ran.

What it may never do:

- Write inside a source project. A project is read to be copied and digested, and
  nothing else. `sandbox/guard.py` holds the digest that makes that checkable.
- Reimplement a metric. Quality numbers come from the app's own evaluation
  harness, so a toolkit number and an app number cannot disagree.
- Declare a setting, a state file name, or a policy version of the app. Each
  fact is read from the app's own code through `engine/`, which is the only
  folder that imports `research_rag`.
- Claim more isolation than it has. This is a cooperative isolation, not an
  operating-system sandbox: every process this package spawns is given a sandbox,
  a pinned settings file, an environment naming the tree under test, and paths
  pointing inside the copy, but the code a run measures — including a code arm's
  patch — runs with the reader's own privileges and could reach the original if it
  tried. What is proved is that *this package* never wrote to the source.

| Folder | Owns |
|---|---|
| `engine/` | the app tree under test, what its harness declares, and the settings it resolves |
| `sandbox/` | a disposable copy of a project, and the guard over the original |
| `experiment/` | a run specification, its arms, and how one arm is executed |
| `report/` | the record a run leaves and the table that compares its arms |
| `run.py` | the order a run happens in, which is the whole point of the toolkit |
| `cli.py` | the one command, and nothing the modules above could not |
