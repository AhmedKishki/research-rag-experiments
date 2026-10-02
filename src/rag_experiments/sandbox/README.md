# `sandbox/`

Owns a disposable copy of a research-rag project, and the proof that the
original was not written to.

What it may never do:

- Open a source project for writing. Every path under one is opened `rb` and that
  is all. `guard.py` is what makes the claim checkable rather than asserted.
- Guess a name the app owns. The copied files and the directories created beside
  them are read from the app's own layout through `layout.py`, so a rename in the
  app does not leave this harness copying a path that no longer exists.
- Call `research-rag init` on a copy. `init` registers a project in the account's
  pointer file, and a faithful copy carries the same `project_id`, so registering
  it would evict the original's record. A copy is addressed by path and is never
  registered.

| Module | Owns |
|---|---|
| `layout.py` | the copied and the never-copied names, read from the app's own code |
| `guard.py` | the before-and-after digest of a source project, and what changed |
| `copy.py` | materializing a sandbox, and destroying one it created |
