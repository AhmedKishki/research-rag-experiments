"""What an experiment is, read from a specification and refused when it is unclear.

A specification is JSON rather than TOML because a run is written and read by
people and by this harness, and JSON needs no writer to be re-emitted. It is
refused on an unknown key rather than ignored, because a mistyped `overlays` that
silently did nothing is an experiment nobody measured.

Every path in a specification is resolved against the file's own directory unless
it is absolute, so a specification beside its patch files is portable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from ..errors import ExperimentError

#: The specification's own version. A specification this harness does not
#: understand is refused rather than read as far as it happens to match.
SPEC_SCHEMA_VERSION = 1

#: The two kinds of arm, and what each one may ask for.
SETTINGS_ARM = "settings"
CODE_ARM = "code"
ARM_KINDS = (SETTINGS_ARM, CODE_ARM)

#: The keys a specification may carry, and the keys an arm may carry. An unknown
#: key in either is an error.
SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "source_project",
        "judgments",
        "generations",
        "harness",
        "arms",
    }
)
#: The `harness` entries that are settings rather than per-call arguments, and the
#: setting each one is. The executor folds these into an arm's overlay before the
#: settings are pinned, so the pinned file and the command agree about the engine a
#: run measured.
HARNESS_SETTINGS: dict[str, str] = {
    "offline": "runtime.offline",
    "dense_backend": "dense.backend",
    "model_cache_root": "runtime.model_cache_root",
    "reranker_model": "dense.reranker_model",
}

#: The `harness` entries whose flag takes one comma-separated value, and the one
#: whose flag is repeated. This is stated rather than inferred because getting it
#: wrong is silent: the app's `--modes` is a single value, so passing it twice
#: measures the last one and a run would report as though both had been asked for.
HARNESS_JOINED = frozenset({"modes", "deep_modes", "classes", "skip_targets"})
HARNESS_REPEATED = frozenset({"reranker_model"})

#: What a specification's `harness` block may carry. It is built from the three
#: sets above rather than written out, so an entry added to one of them is a legal
#: specification key in the same change.
HARNESS_FIELDS = frozenset(
    set(HARNESS_SETTINGS)
    | HARNESS_JOINED
    | HARNESS_REPEATED
    | {"top_k", "deep_top_k", "limit"}
)
ARM_FIELDS = frozenset({"name", "kind", "overlay", "base", "patch", "prepare"})

#: The tokens a `prepare` command may interpolate, and what each is. A token the
#: harness does not know is left in place rather than guessed at, so a command
#: that expected a substitution the harness cannot make fails loudly inside the
#: command instead of quietly measuring the wrong path.
#:
#: Each element of a `prepare` list is one command: a string, which is split the
#: way a shell would split it, or a list of arguments, which is used exactly as
#: written so that a path with a space in it stays one argument.
PREPARE_TOKENS = ("{project}", "{source}", "{tree}", "{sandbox}")

#: The split name a specification that names one judged path is given. It is named
#: rather than left blank so a record and a table always have a key to print.
SINGLE_SPLIT = "all"

#: The keys one split of the `judgments` list may carry.
JUDGMENT_FIELDS = frozenset({"name", "path"})

#: One prepare command: a shell command line, or its arguments already split.
PrepareCommand: TypeAlias = "str | list[str]"


def segment(name: str, *, strict: bool = True) -> str:
    """One plain path segment for a name a record keys on.

    An arm's name, a split's name, and the specification's own name all become
    directory names inside a run, so a name that reaches a parent, a root, or
    another directory is a traversal rather than a label. With `strict`, such a
    name is refused, because silently renaming it would put the record under a key
    the reader did not write. Without it, the name is reduced to one safe segment,
    which is what a caller that invented the name itself needs.
    """

    safe = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in name
    ).strip("-")
    safe = safe or "split"
    if strict and safe != name:
        raise ExperimentError(
            f"{name!r} is not one plain path segment, and this harness uses it as a "
            "directory name inside a run. Use letters, digits, hyphens, and "
            "underscores."
        )
    return safe


@dataclass(frozen=True, slots=True)
class Arm:
    """One measured configuration, and the variant that defines it."""

    name: str
    kind: str
    overlay: dict[str, Any]
    base: str | None
    patch: Path | None
    prepare: tuple[PrepareCommand, ...]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "overlay": self.overlay,
            "base": self.base,
            "patch": None if self.patch is None else str(self.patch),
            "prepare": [list(item) for item in self.prepare],
        }


@dataclass(frozen=True, slots=True)
class JudgedSet:
    """One judged query set, and the split it belongs to.

    The name is what the record and the table use, so "held-out" reads as
    held-out rather than as a file path. Two splits measured in one run is the
    point: a development number and a held-out number are only comparable when
    the same arms produced both, and two specifications listing the same arms can
    drift apart between the two runs.
    """

    name: str
    path: Path

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "path": str(self.path)}


@dataclass(frozen=True, slots=True)
class RunSpec:
    """A whole run: what is measured, where the corpus comes from, and the arms."""

    path: Path
    name: str
    source_project: Path
    judgments: tuple[JudgedSet, ...]
    generations: tuple[str, ...]
    harness: dict[str, Any]
    arms: tuple[Arm, ...]

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "source_project": str(self.source_project),
            "judgments": [item.describe() for item in self.judgments],
            "generations": list(self.generations),
            "harness": self.harness,
            "arm_count": len(self.arms),
        }


def load_spec(path: Path) -> RunSpec:
    """Read a run specification, refusing anything this harness cannot measure."""

    location = Path(path).expanduser().resolve()
    try:
        document = json.loads(location.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExperimentError(
            f"Cannot read the run specification: {location}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ExperimentError(
            f"Run specification is not JSON: {location}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise ExperimentError(f"Run specification is not an object: {location}")
    _refuse_unknown(document, SPEC_FIELDS, f"specification {location}")

    version = document.get("schema_version")
    if version != SPEC_SCHEMA_VERSION:
        raise ExperimentError(
            f"{location} declares schema_version {version!r}; this harness reads "
            f"{SPEC_SCHEMA_VERSION}."
        )
    # The name becomes a directory inside the run's own directory, so it is checked
    # before anything is read from disk and before a judged set is resolved against
    # it: a name that reaches outside the run is refused before it is used.
    name = segment(str(document.get("name") or location.stem).strip())

    base = location.parent
    source_project = _path(document, "source_project", base, location)
    judgments = _judged_sets(document.get("judgments"), base, location)
    arms_document = document.get("arms")
    if not isinstance(arms_document, list) or not arms_document:
        raise ExperimentError(f"{location} names no arms")
    arms = tuple(
        _arm(entry, base, location, position)
        for position, entry in enumerate(arms_document, 1)
    )
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise ExperimentError(f"{location} names an arm twice: {sorted(names)}")

    return RunSpec(
        path=location,
        name=name,
        source_project=source_project,
        judgments=judgments,
        generations=tuple(str(item) for item in document.get("generations") or ()),
        harness=_harness(document.get("harness"), location),
        arms=arms,
    )


def _judged_sets(entry: Any, base: Path, location: Path) -> tuple[JudgedSet, ...]:
    """Read the `judgments` field, which is one path or a list of named splits.

    A bare string is one split called `all`, which is what a specification with
    one judged set means. A list is `{"name", "path"}` per split, and the names
    have to differ because the record and the table key on them: two splits
    sharing a name would make a held-out row indistinguishable from a development
    one, which is the mistake the split exists to prevent.
    """

    if isinstance(entry, str) and entry.strip():
        path = _absolute(entry, base)
        if not path.is_file():
            raise ExperimentError(f"The judged set does not exist: {path}")
        return (JudgedSet(name=SINGLE_SPLIT, path=path),)
    if not isinstance(entry, list) or not entry:
        raise ExperimentError(
            f"{location} has no `judgments`. Name one path, or a list of "
            '{"name", "path"} splits such as development and held-out.'
        )
    if not all(isinstance(item, dict) for item in entry):
        raise ExperimentError(
            f"{location} lists a judged set that is not an object; each split is "
            '{"name", "path"}'
        )
    judged: list[JudgedSet] = []
    for position, item in enumerate(entry, 1):
        _refuse_unknown(item, JUDGMENT_FIELDS, f"{location} judged set {position}")
        name = segment(str(item.get("name") or "").strip())
        raw = item.get("path")
        if not isinstance(raw, str) or not raw.strip():
            raise ExperimentError(f"{location} judged set {name!r} has no path")
        path = _absolute(raw, base)
        if not path.is_file():
            raise ExperimentError(f"The judged set does not exist: {path}")
        judged.append(JudgedSet(name=name, path=path))
    duplicates = sorted(
        {item.name for item in judged if [j.name for j in judged].count(item.name) > 1}
    )
    if duplicates:
        raise ExperimentError(f"{location} names a judged split twice: {duplicates}")
    return tuple(judged)


def _arm(entry: Any, base: Path, location: Path, position: int) -> Arm:
    if not isinstance(entry, dict):
        raise ExperimentError(f"{location} arm {position} is not an object")
    _refuse_unknown(entry, ARM_FIELDS, f"{location} arm {position}")
    name = segment(str(entry.get("name") or "").strip())
    kind = str(entry.get("kind") or "").strip()
    if kind not in ARM_KINDS:
        raise ExperimentError(
            f"{location} arm {name!r} has kind {kind!r}; the kinds are "
            + ", ".join(ARM_KINDS)
        )

    overlay = entry.get("overlay") or {}
    if not isinstance(overlay, dict):
        raise ExperimentError(f"{location} arm {name!r} has a non-object overlay")

    base_revision = entry.get("base")
    patch = entry.get("patch")
    if kind == SETTINGS_ARM and (base_revision or patch):
        raise ExperimentError(
            f"{location} arm {name!r} is a settings arm and also asks for a code "
            "change. A settings arm runs the engine as it stands; a code arm needs "
            f"kind {CODE_ARM!r}."
        )
    if kind == CODE_ARM and not (base_revision or patch):
        raise ExperimentError(
            f"{location} arm {name!r} is a code arm that changes nothing. Give it a "
            "`base` revision, a `patch` file, or both."
        )

    patch_path: Path | None = None
    if patch:
        patch_path = _absolute(patch, base)
        if not patch_path.is_file():
            raise ExperimentError(f"{location} arm {name!r} has no patch: {patch_path}")

    prepare = _prepare(entry.get("prepare") or [], location, name)

    return Arm(
        name=name,
        kind=kind,
        overlay={str(key): value for key, value in overlay.items()},
        base=None if base_revision is None else str(base_revision),
        patch=patch_path,
        prepare=prepare,
    )


def _prepare(entry: Any, location: Path, arm: str) -> tuple[PrepareCommand, ...]:
    """Read the commands an arm runs before it is measured.

    A command is one shell command line, which is split the way a shell would, or a
    list of arguments, which is used as written. Both forms are read because both
    are natural: a one-line command is a string, and a command whose arguments carry
    spaces or quotes is a list.
    """

    if not isinstance(entry, list):
        raise ExperimentError(
            f"{location} arm {arm!r} has a `prepare` that is not a list of commands"
        )
    commands: list[PrepareCommand] = []
    for position, item in enumerate(entry, 1):
        if isinstance(item, str) and item.strip():
            commands.append(item)
            continue
        if (
            isinstance(item, list)
            and item
            and all(isinstance(part, str) and part for part in item)
        ):
            commands.append([str(part) for part in item])
            continue
        raise ExperimentError(
            f"{location} arm {arm!r} prepare command {position} is neither a "
            "non-empty command line nor a non-empty list of arguments"
        )
    return tuple(commands)


def _harness(entry: Any, location: Path) -> dict[str, Any]:
    if entry is None:
        return {}
    if not isinstance(entry, dict):
        raise ExperimentError(f"{location} has a non-object `harness`")
    _refuse_unknown(entry, HARNESS_FIELDS, f"{location} harness")
    return {str(key): value for key, value in entry.items()}


def _refuse_unknown(entry: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(entry) - allowed)
    if unknown:
        raise ExperimentError(
            f"{where} has keys this harness does not read: {unknown}. "
            f"Allowed: {sorted(allowed)}."
        )


def _path(document: dict[str, Any], key: str, base: Path, location: Path) -> Path:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(f"{location} has no {key!r}")
    return _absolute(value, base)


def _absolute(value: Any, base: Path) -> Path:
    candidate = Path(str(value)).expanduser()
    return (
        candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    )


__all__ = [
    "ARM_KINDS",
    "CODE_ARM",
    "HARNESS_FIELDS",
    "PREPARE_TOKENS",
    "SETTINGS_ARM",
    "SINGLE_SPLIT",
    "SPEC_SCHEMA_VERSION",
    "Arm",
    "JudgedSet",
    "RunSpec",
    "load_spec",
    "segment",
]
