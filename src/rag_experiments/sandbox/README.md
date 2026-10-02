# `sandbox/`

Owns a disposable copy of a research-rag project, and the proof that the original
was not written to.

What it may never do:

- Open a source project for writing. Every path under one is opened `rb` and that
  is all. `guard.py` is what makes the claim checkable rather than asserted.
- Guess a name the app owns. The copied files and the directories created beside
  them are read from the app's own layout through `layout.py`, so a rename in the
  app does not leave this harness copying a path that no longer exists.
- Call `research-rag init` on a copy. `init` registers a project in the account's
  pointer file, and a faithful copy carries the same `project_id`, so registering
  it would evict the original's record. A copy is addressed by path and is never
  registered. The account's registry is digested by the guard, not written.
- Share storage with the tree it copied. Every file is written as bytes into a new
  inode, and the copy is checked for shared inodes afterwards. A hard link or a
  reflink copies every byte and still lets a write inside the copy rewrite the
  corpus.

| Module | Owns |
|---|---|
| `layout.py` | the copied and the never-copied names, read from the app's own code |
| `guard.py` | the before-and-after digest of a source project, and what changed |
| `copy.py` | materializing a sandbox, and destroying one it created |

## What the guard covers

The whole project: the originals wherever the descriptor says they are, the review
state, every generation, every file at the root, and every directory between them.
A project that relocated its own derived state is digested there as well, because a
change to a generation is a change to the corpus whether it was written beside the
project or elsewhere. A symbolic link is recorded by the string it points at and
never followed, so a link out of the project cannot make the guard read outside it.

Excluded, and named in the record rather than assumed: the pid, port, terminal and
project lock files, the log directory, and the gateway's own runtime directory. A
serving app rewrites all of those while a run is going, and a guard that named them
would fail every run against a live project. **Every other change disqualifies the
run**, including one made by the app or by a person: the arms read a corpus that is
then no longer the corpus on disk.

## What the copy carries, and what it does not

Carried: the originals under their own path, the review state, and the selected
generation (or the generations a caller named).

Not carried: the runtime pointer that names a relocated runtime directory, every
derived runtime file, and the account's project registry. A copy keeps its derived
state under its own root, so removing a machine-local pointer cannot make it look
like an empty project later.

## What isolation this is

Cooperative. Every process this harness spawns is given a sandbox, a pinned
settings file, an environment naming the tree under test, and paths pointing inside
the copy. There is no operating-system sandbox here: code the run measures, including
a code arm's patch, runs with the reader's own privileges and could reach the
original if it tried. What is proved is that *this harness* never wrote to the
source, which is a claim about this code and not about the code under test.

## What is refused before anything is created

- A source directory that leaves the project root, or names no directory.
- A source directory that is not a directory at all.
- A portable directory that is a symbolic link.
- A workspace, a run directory, or a sandbox area that is inside the project it
  measures, or that contains it.
- A workspace or run directory inside the engine tree.
- A sandbox name that is not one plain path segment, including `..` and `.`.
- A sandbox whose name already exists, and a `remove` of anything this harness did
  not make: no record, a name that does not match the record, or a symbolic link.