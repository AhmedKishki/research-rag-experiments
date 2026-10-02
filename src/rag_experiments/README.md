# `rag_experiments`

What the package owns: the machinery that runs a research-rag retrieval
experiment outside the project it measures, and the record that says what ran.

What it may never do:

- Write inside a source project. A project is read to be copied and digested, and
  nothing else. `sandbox/guard.py` holds the digest that makes that checkable.
- Reimplement a metric. Quality numbers come from the app's own evaluation
  harness, so a toolkit number and an app number cannot disagree.
- Declare a setting, a state file name, or a policy version of the app. Each
  fact is read from the app's own code through `engine/`, which is the only
  folder that imports `research_rag`.

| Folder | Owns |
|---|---|
| `engine/` | the app tree under test, and the settings it resolves |
| `sandbox/` | a disposable copy of a project, and the guard over the original |
| `experiment/` | a run specification, its arms, and how one arm is executed |
| `report/` | the record a run leaves and the table that compares its arms |
| `run.py` | the order a run happens in, which is the whole point of the toolkit |
| `cli.py` | the one command, and nothing the modules above could not |
