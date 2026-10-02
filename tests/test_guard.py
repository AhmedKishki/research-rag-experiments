"""That a source project is digested, and what a change reads as.

The guard is the only thing standing between an experiment and a corpus that has
been quietly modified, so these tests hold the property that matters: a change
anywhere under a project's originals or its state is named, and an unchanged
project produces an empty list.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FIRST_GENERATION, build_project

from rag_experiments.errors import ExperimentError
from rag_experiments.sandbox.guard import GUARDED_ENTRIES, differences, snapshot


def test_the_guard_covers_the_originals_and_every_piece_of_derived_state() -> None:
    assert GUARDED_ENTRIES == (".research-rag", "sources")


def test_an_untouched_project_has_no_differences(project: Path) -> None:
    before = snapshot(project)
    after = snapshot(project)
    assert differences(before, after) == []
    assert before.digest() == after.digest()
    # Every file the fixture holds is guarded, and nothing else is: an unguarded
    # file would be a file a run could write without the run noticing.
    assert set(before.entries) == {
        "sources/a-book.pdf",
        "sources/a-book.epub",
        ".research-rag/project.json",
        ".research-rag/source-catalog.json",
        ".research-rag/source-metadata.json",
        ".research-rag/source-exclusions.json",
        ".research-rag/runtime/current.json",
        f".research-rag/runtime/generations/{FIRST_GENERATION}/manifest.json",
    }
    assert before.byte_count > 0


def test_the_digest_is_one_value_that_covers_content_and_name(project: Path) -> None:
    before = snapshot(project)
    renamed = before.digest()
    (project / "sources" / "a-book.pdf").unlink()
    (project / "sources" / "renamed.pdf").write_bytes(b"%PDF-1.7\nfixture\n")
    after = snapshot(project)
    assert renamed != after.digest()
    # Same bytes under a new name is a different project, not an identical one.
    assert differences(before, after) == [
        "removed: sources/a-book.pdf",
        "added: sources/renamed.pdf",
    ]


def test_one_changed_byte_is_named(project: Path) -> None:
    before = snapshot(project)
    original = project / "sources" / "a-book.pdf"
    original.write_bytes(b"%PDF-1.7\nfixture!")
    after = snapshot(project)
    assert differences(before, after) == ["changed: sources/a-book.pdf"]


def test_a_file_added_beside_the_guarded_ones_is_named(project: Path) -> None:
    before = snapshot(project)
    (project / "sources" / "another.pdf").write_bytes(b"%PDF-1.7\n")
    assert differences(before, snapshot(project)) == ["added: sources/another.pdf"]


def test_a_change_deep_in_a_generation_is_named(project: Path) -> None:

    before = snapshot(project)
    manifest = (
        project / ".research-rag" / "runtime" / "generations" / FIRST_GENERATION
    ) / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert differences(before, snapshot(project)) == [
        f"changed: .research-rag/runtime/generations/{FIRST_GENERATION}/manifest.json"
    ]


def test_a_symlink_is_not_followed_out_of_the_project(
    project: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("not part of the project", encoding="utf-8")
    (project / "sources" / "link.txt").symlink_to(outside)
    before = snapshot(project)
    outside.write_text("changed outside the project", encoding="utf-8")
    assert differences(before, snapshot(project)) == []


def test_the_digest_covers_content_and_not_where_the_project_lives(
    project: Path, tmp_path: Path
) -> None:
    other = build_project(tmp_path / "other")
    # Two projects holding the same bytes are the same corpus, which is what lets
    # a digest printed on one machine be compared on another.
    assert snapshot(project).digest() == snapshot(other).digest()
    # Comparing snapshots of two different roots is refused rather than read as
    # "everything changed", which would be indistinguishable from a rewrite.
    assert differences(snapshot(project), snapshot(other)) == [
        f"root changed: {project} -> {other}"
    ]


def test_a_missing_project_is_a_condition_not_a_crash(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError) as caught:
        snapshot(tmp_path / "absent")
    assert "not a directory" in str(caught.value)


def test_a_guarded_path_that_is_a_symlink_is_refused(project: Path) -> None:
    portable = project / ".research-rag"
    moved = project / "elsewhere"
    portable.rename(moved)
    portable.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ExperimentError) as caught:
        snapshot(project)
    assert "symlink" in str(caught.value)
