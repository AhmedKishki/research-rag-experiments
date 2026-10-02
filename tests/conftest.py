"""The fixtures, and the minimum a research-rag project has to hold.

`project` is a real project as far as the app's own readers are concerned: the
portable descriptor, the review files, the runtime skeleton, a generation pointer,
and two generation directories. Everything this toolkit does to a project is
exercised against that, so a copy that were missing anything the app needs would
fail here rather than on a corpus of a gigabyte.

Building a synthetic project instead of copying a real one is also the statement
that this toolkit reads a project rather than a corpus: nothing below needs an
original to be a real PDF, because nothing below reads one.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

#: These tests import the installed app, which is a developer's working checkout.
#: Without this, an import writes a `__pycache__` beside the app's source, and a
#: test suite that leaves files in the tree it measures would be doing the thing
#: every other test here holds it does not do. A run's own child processes are given
#: `PYTHONDONTWRITEBYTECODE` for the same reason.
sys.dont_write_bytecode = True

#: The generation ids used by the fixture. They carry the shape the app's own
#: generation naming writes, because a name that does not is refused by the app.
FIRST_GENERATION = "20260101T000000Z-0000000a"
SECOND_GENERATION = "20260101T010000Z-0000000b"

#: The project's identifier. A faithful copy keeps it, which is what makes every
#: `source_id` in the review files still resolve.
PROJECT_ID = "11111111-2222-3333-4444-555555555555"

#: The query classes a judged set uses, as the app's judged-set schema names them.
#: The list is stated here rather than read because a fixture states what its files
#: hold; a judged set that used any other class would be refused by the app's own
#: harness before a run reached it.
QUERY_CLASSES = ("quote", "paraphrase", "entity")

#: How many queries the fixture judged sets carry. One is the smallest set a report
#: can be measured from, and the report contract checks per-query rows, so a fixture
#: with no query would pass by measuring nothing.
JUDGED_QUERY_COUNT = 2


def build_judged_set(
    path: Path, *, queries: int = JUDGED_QUERY_COUNT
) -> dict[str, Any]:
    """Write a judged set with real identities, and return what it holds.

    A judged set is not a list of strings: each query names its own id, the target it
    is judged against, and the class it belongs to, and the harness refuses a file
    missing any of those. A fixture whose queries are empty therefore tests the
    harness's refusal rather than a measurement, so this writes queries with ids and
    lets a report name them back.
    """

    document = {
        "schema_version": 1,
        "protocol": "Fixture judged set: one target per query, one class each.",
        "targets": [
            {
                "target_id": f"t{index:02d}",
                "document_id": f"d{index:02d}",
                # The chunk the target resolved to when it was judged. The app's
                # harness re-resolves it from the snippet, and a target without it is
                # not the shape of a judged set that can be measured.
                "chunk_id_at_measurement": f"c{index:02d}",
                "source_path": f"sources/a-book-{index}.pdf",
                "snippet": f"the passage for t{index:02d}",
                "chunk_text": f"the passage for t{index:02d}",
            }
            for index in range(1, queries + 1)
        ],
        "queries": [
            {
                "query_id": f"q{index:02d}",
                "target_id": f"t{index:02d}",
                "class": QUERY_CLASSES[(index - 1) % len(QUERY_CLASSES)],
                "query": f"find the passage for t{index:02d}",
            }
            for index in range(1, queries + 1)
        ],
    }
    _write_json(path, document)
    return document


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_project(root: Path, *, generations: tuple[str, ...] = ()) -> Path:
    """Create the smallest project the app's readers accept, and return its root.

    `generations` names which generation directories to create; the pointer always
    names the last of them, because a pointer naming a missing directory is a
    refusal and would make the fixture useless for the copy tests.
    """

    present = generations or (FIRST_GENERATION,)
    pointed = present[-1]
    portable = root / ".research-rag"
    state = portable / "runtime"
    for name in ("generations", "logs", "staging", "failures", "ultrarag-runtime"):
        (state / name).mkdir(parents=True, exist_ok=True)
    (root / "sources").mkdir(parents=True, exist_ok=True)
    (root / "sources" / "a-book.pdf").write_bytes(b"%PDF-1.7\nfixture\n")
    (root / "sources" / "a-book.epub").write_bytes(b"PK\x03\x04fixture\n")

    _write_json(
        portable / "project.json",
        {
            "schema_version": 1,
            "project_id": PROJECT_ID,
            "name": "Fixture project",
            "source_directory": "sources",
        },
    )
    _write_json(
        portable / "source-catalog.json",
        {"schema_version": 1, "project_id": PROJECT_ID, "sources": {}},
    )
    _write_json(portable / "source-metadata.json", {"schema_version": 1, "sources": {}})
    _write_json(
        portable / "source-exclusions.json", {"schema_version": 1, "sources": {}}
    )

    for generation_id in present:
        _write_json(
            state / "generations" / generation_id / "manifest.json",
            {
                "schema_version": 5,
                "generation_id": generation_id,
                "project_id": PROJECT_ID,
                "chunk_count": 2,
            },
        )
    _write_json(state / "current.json", {"schema_version": 1, "generation_id": pointed})
    return root


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project with one generation, which is what a copy test needs."""

    return build_project(tmp_path / "corpus")


@pytest.fixture
def two_generation_project(tmp_path: Path) -> Path:
    """A project holding two generations, for the copy that names both."""

    return build_project(
        tmp_path / "corpus", generations=(FIRST_GENERATION, SECOND_GENERATION)
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace no sandbox has been made in yet."""

    area = tmp_path / "workspaces"
    area.mkdir()
    return area


#: The variable that names the engine under test, so a machine that keeps its
#: research-rag checkout somewhere else does not need this repository's layout.
APP_SOURCE_ENV = "RESEARCH_RAG_APP_SOURCE"

#: Where a checkout is looked for when the variable is not set: the working tree
#: this repository's `pyproject.toml` names, and the two directories beside it
#: that are named for it. A list rather than one path, because a developer may
#: keep the engine somewhere this repository cannot know.
APP_SOURCE_CANDIDATES = (
    Path("../ultra-rag-mcp-servers/research-rag"),
    Path("../research-rag"),
)


def find_app_source() -> Path | None:
    """The research-rag tree the tests measure, or None when there is none.

    The tree is found by walking up from the installed package to the directory
    holding the harness script, rather than by counting parents, so an editable
    install and a copied install are both found or both reported absent. A tree
    that cannot be found skips the tests that need one rather than failing them:
    the rule they hold does not change when the engine is installed from a wheel.
    """

    import os

    import research_rag

    named = os.environ.get(APP_SOURCE_ENV, "").strip()
    if named:
        return Path(named).expanduser().resolve()
    for parent in Path(research_rag.__file__).resolve().parents:
        if (parent / "scripts" / "evaluate_retrieval.py").is_file():
            return parent
    for candidate in APP_SOURCE_CANDIDATES:
        resolved = candidate.expanduser().resolve()
        if (resolved / "scripts" / "evaluate_retrieval.py").is_file():
            return resolved
    return None


@pytest.fixture
def app_source() -> Path:
    """A research-rag tree to measure: the one this repository is installed against."""

    root = find_app_source()
    if root is None:
        pytest.skip(
            "no research-rag tree with scripts/evaluate_retrieval.py was found; set "
            f"{APP_SOURCE_ENV} at a checkout, or point [tool.uv.sources] at one"
        )
    return root


@pytest.fixture
def make_spec(tmp_path: Path, project: Path) -> Callable[..., Path]:
    """Write a run specification and return where it went.

    The judged set is written beside it, because a specification's judged set is
    resolved relative to the file and a missing one is refused before anything
    expensive happens.
    """

    def _make(arms: list[dict[str, Any]], **overrides: Any) -> Path:
        judgments = tmp_path / "judged.json"
        if not judgments.is_file():
            build_judged_set(judgments)
        document: dict[str, Any] = {
            "schema_version": 1,
            "name": "fixture",
            "source_project": str(project),
            "judgments": judgments.name,
            "generations": [],
            "harness": {"top_k": 10, "offline": True},
            "arms": arms,
        }
        document.update(overrides)
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        return path

    return _make
