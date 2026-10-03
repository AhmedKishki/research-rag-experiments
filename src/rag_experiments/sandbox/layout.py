"""The names a copy is made of, read from the app rather than written here.

A harness that spelled out `.research-rag`, `current.json`, and `logs` would be a
second copy of the app's layout, and the two would disagree the day either
changed. So every name below is read out of the app's own code: the directory
and process-state names come from `state_files`, the machine-local runtime
pointer from `config`, and the rest from the single path literal inside each
`ResearchConfig` property.

The copy is an allowlist rather than a denylist. Everything under the portable
directory travels except the one file that points at a relocated runtime, and
under the runtime directory only the selected-generation pointer and the named
generations travel. A file the app adds later is therefore excluded by default
rather than copied because nobody remembered to exclude it.

Two placements are refused here rather than discovered later. A source directory
that escapes its project root would make the copy read something the project does
not contain, and a workspace nested inside a project it measures would let a run
write into the corpus it claims to have left alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._imports import without_bytecode
from ..errors import ExperimentError

#: Where the copied originals live inside a project, relative to its root. This
#: is the directory the app's own descriptor names rather than the literal, so a
#: renamed source directory follows the descriptor.
SOURCE_DIRECTORY_FIELD = "source_directory"


@dataclass(frozen=True, slots=True)
class ProjectLayout:
    """Every on-disk name a copy is built from, resolved against one project root."""

    root: Path
    portable_directory: str
    runtime_directory: str
    portable_root: Path
    state_root: Path
    source_root: Path
    current_pointer: str
    generations_directory: str
    generations_root: Path
    descriptor: str
    settings_relative: str
    empty_runtime_directories: tuple[str, ...]
    runtime_pointer: str
    default_state_root: Path
    relocated_runtime: Path | None

    @property
    def state_root_is_default(self) -> bool:
        """Whether the project's derived state sits under its own root."""

        return self.relocated_runtime is None

    def portable_path(self, name: str) -> Path:
        return self.portable_root / name

    def generation_root(self, generation_id: str) -> Path:
        return self.generations_root / generation_id

    def source_relative(self, path: Path) -> str:
        """Where a path inside the copy sits, relative to the copied project root.

        The copied root keeps the source directory's own name and position, so a
        `source_id` computed from a normalized relative path resolves in the copy
        exactly as it does in the original.
        """

        return path.relative_to(self.root).as_posix()

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "source_root": str(self.source_root),
            "portable_root": str(self.portable_root),
            "state_root": str(self.state_root),
            "default_state_root": str(self.default_state_root),
            "relocated_runtime": (
                None if self.relocated_runtime is None else str(self.relocated_runtime)
            ),
            "current_pointer": self.current_pointer,
            "generations_directory": self.generations_directory,
            "settings_file": self.settings_relative,
            "empty_runtime_directories": list(self.empty_runtime_directories),
        }


@without_bytecode
def read_layout(root: Path) -> ProjectLayout:
    """Resolve the app's own on-disk names for the project at `root`.

    This reads the app's code and the project's descriptor. It creates nothing and
    writes nothing, which is what lets it be called on a source project that the
    harness promised not to touch: the app's own `resolve_config` cannot be used
    here, because it makes directories and rewrites the runtime pointer.

    A project whose derived state was relocated is read through the app's own
    reader for that record rather than by assuming the runtime directory under its
    root, and a record naming a directory that is not there is refused instead of
    being measured as an empty project.
    """

    from research_rag.project import config as app_config
    from research_rag.project import state_files
    from research_rag.project.settings import PROJECT_CONFIG_RELATIVE
    from research_rag.project.settings_layers.paths import project_config_path

    project = Path(root).expanduser().resolve()
    if not project.is_dir():
        raise ExperimentError(f"Project root is not a directory: {project}")
    if (project / state_files.PORTABLE_DIRECTORY).is_symlink():
        raise ExperimentError(
            f"{project / state_files.PORTABLE_DIRECTORY} is a symlink; a guarded "
            "path must be a real directory"
        )
    if not (project / state_files.PORTABLE_DIRECTORY).is_dir():
        raise ExperimentError(
            f"{project} is not a research-rag project: it has no "
            f"{state_files.PORTABLE_DIRECTORY} directory."
        )

    portable = project / state_files.PORTABLE_DIRECTORY
    default_state = portable / state_files.RUNTIME_DIRECTORY
    state, relocated = _state_root(project, default_state)
    generations_directory = _literal_of(app_config.ResearchConfig, "generations_root")
    descriptor = _literal_of(app_config.ResearchConfig, "project_config_path")
    # Placed by the app's own helper rather than by joining two names here, so the
    # path a pinned settings file is written to is the path the app will read.
    settings_relative = (
        project_config_path(project, PROJECT_CONFIG_RELATIVE)
        .relative_to(project)
        .as_posix()
    )
    source_directory = _source_directory(portable / descriptor)
    source_root = _resolve_source_root(project, portable / descriptor, source_directory)

    return ProjectLayout(
        root=project,
        portable_directory=state_files.PORTABLE_DIRECTORY,
        runtime_directory=state_files.RUNTIME_DIRECTORY,
        portable_root=portable,
        state_root=state,
        source_root=source_root,
        current_pointer=_literal_of(app_config.ResearchConfig, "current_path"),
        generations_directory=generations_directory,
        generations_root=state / generations_directory,
        descriptor=descriptor,
        settings_relative=settings_relative,
        empty_runtime_directories=tuple(
            _literal_of(app_config.ResearchConfig, name)
            for name in (
                "logs_root",
                "staging_root",
                "failures_root",
                "ultrarag_workspace",
            )
        ),
        runtime_pointer=app_config.RUNTIME_ROOT_RECORD,
        default_state_root=default_state,
        relocated_runtime=relocated,
    )


@without_bytecode
def relocated_runtime_of(root: Path) -> Path | None:
    """Where a project's derived state was relocated to, or None.

    Read through the app's own record rather than by joining a name here, so a
    caller that guards a source does not have to know whether the project
    relocates its state. `read_layout` refuses a record naming a directory that is
    not there; this answers the same question for a caller that only wants to know
    whether there is a second root to digest.
    """

    from research_rag.project import state_files

    project = Path(root).expanduser().resolve()
    default = project / state_files.PORTABLE_DIRECTORY / state_files.RUNTIME_DIRECTORY
    return _state_root(project, default)[1]


def refuse_nested(area: Path, root: Path, *, what: str, inside: str) -> None:
    """Refuse a directory that contains, or is contained by, the project it is for.

    A workspace or a run directory under the source project would make the guard
    digest files the run itself creates, so every run would report that it changed
    the corpus. A workspace or run directory that *contains* the project would let
    a cleanup delete the corpus it measured. Neither placement is recoverable by
    inspection afterwards, so both are refused before anything is created.
    """

    place = Path(area).expanduser().resolve()
    against = Path(root).expanduser().resolve()
    if place == against:
        raise ExperimentError(
            f"{what} is the {inside} itself: {place}. Put it somewhere else."
        )
    if _within(place, against):
        raise ExperimentError(
            f"{what} is inside the {inside} at {against}: {place}. A run writes "
            f"inside its own directories, so one under the project it measures would "
            f"make the guard report the run's own files as a change to the corpus."
        )
    if _within(against, place):
        raise ExperimentError(
            f"{what} contains the {inside} at {against}: {place}. Removing it would "
            f"remove the project the run measured."
        )


@without_bytecode
def selected_generation(layout: ProjectLayout) -> str:
    """The generation the project's own pointer names, or a reason there is none.

    Read through the app's reader rather than by opening the file, so a pointer
    naming a generation directory that is missing, or a manifest whose id does
    not match, is refused the way the app refuses it instead of being measured.
    The id comes from the manifest the reader verified, not from the pointer, so a
    pointer and a manifest that disagree are a refusal rather than a choice.
    """

    from research_rag.storage.records import load_current_generation

    try:
        _root, manifest = load_current_generation(layout.state_root)
    except Exception as exc:
        raise ExperimentError(
            f"{layout.root} has no generation this harness can copy: {exc}"
        ) from exc
    identifier = str(manifest.get("generation_id") or "")
    if not identifier:
        raise ExperimentError(
            f"{layout.state_root} names a generation whose manifest holds no identifier"
        )
    return identifier


def existing_generations(layout: ProjectLayout) -> list[str]:
    """Every generation directory present, sorted, for a caller choosing among them."""

    if not layout.generations_root.is_dir():
        return []
    return sorted(
        path.name for path in layout.generations_root.iterdir() if path.is_dir()
    )


def _state_root(project: Path, default: Path) -> tuple[Path, Path | None]:
    """Where a project's derived state actually is, read through the app's record.

    A project can record that its state was relocated outside its own root. The
    record is the app's, so the reader is the app's, and a run copies the state
    from where the project says it is rather than from an empty directory under
    its root.
    """

    from research_rag.project.config import recorded_runtime_root
    from research_rag.project.state_files import ProjectState

    portable = (
        project
        / ProjectState(project_root=project, project_name=project.name).portable_root
    )
    relocated = ProjectState(project_root=project, project_name=project.name).state_root
    if relocated == default:
        return default, None
    recorded = recorded_runtime_root(portable)
    if not relocated.is_dir():
        raise ExperimentError(
            f"{project} records its derived state at "
            f"{recorded if recorded is not None else relocated}, and that directory "
            "is not there. A run cannot copy state it cannot read; restore it, or "
            "measure a project that keeps its state under its own root."
        )
    return relocated, relocated


def _resolve_source_root(project: Path, descriptor: Path, source_directory: str | None):
    """Where a project's originals are, refused unless they are inside the project."""

    if source_directory is None:
        raise ExperimentError(
            f"{descriptor} names no {SOURCE_DIRECTORY_FIELD}; the app could not say "
            "where this project's originals are."
        )
    candidate = Path(source_directory)
    if candidate.is_absolute():
        raise ExperimentError(
            f"{descriptor} names an absolute {SOURCE_DIRECTORY_FIELD}: "
            f"{source_directory}"
        )
    resolved = (project / candidate).resolve()
    if not _within(resolved, project) or resolved == project:
        raise ExperimentError(
            f"{descriptor} names a {SOURCE_DIRECTORY_FIELD} outside the project: "
            f"{source_directory}. The originals a run measures must be inside the "
            f"project, or a change to them would be outside the guard."
        )
    if not resolved.is_dir():
        raise ExperimentError(
            f"{descriptor} names a {SOURCE_DIRECTORY_FIELD} that is not a directory: "
            f"{resolved}"
        )
    return resolved


def _literal_of(owner: type, property_name: str) -> str:
    """The one path literal inside an app property, read from its code.

    These properties join a resolved base with a name, so their code holds exactly
    one string. Anything else means the app's shape changed and this harness
    should stop rather than guess which constant is the name.
    """

    attribute = getattr(owner, property_name, None)
    getter = getattr(attribute, "fget", None)
    if getter is None:
        raise ExperimentError(
            f"The app's ResearchConfig no longer has a {property_name} property, so "
            "this harness cannot say where that state lives."
        )
    literals = [value for value in getter.__code__.co_consts if isinstance(value, str)]
    if len(literals) != 1:
        raise ExperimentError(
            f"The app's ResearchConfig.{property_name} holds {len(literals)} string "
            f"literals, so this harness cannot say which one is the name: {literals}"
        )
    return literals[0]


def _source_directory(descriptor: Path) -> str | None:
    import json

    try:
        document = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Cannot read the project descriptor: {exc}") from exc
    if not isinstance(document, dict):
        raise ExperimentError(f"Project descriptor is not an object: {descriptor}")
    value = document.get(SOURCE_DIRECTORY_FIELD)
    return str(value).strip() if value else None


def _within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = [
    "ProjectLayout",
    "existing_generations",
    "read_layout",
    "refuse_nested",
    "relocated_runtime_of",
    "selected_generation",
]
