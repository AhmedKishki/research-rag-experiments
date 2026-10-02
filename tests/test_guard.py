"""That a source project is digested, and what a change reads as.

The guard is the only thing standing between an experiment and a corpus that has
been quietly modified, so these tests hold the property that matters: a change
anywhere in the project is named, an unchanged project produces an empty list, and
the state a serving app rewrites on its own does not make every run fail.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from conftest import FIRST_GENERATION, build_project

from rag_experiments.errors import ExperimentError
from rag_experiments.sandbox.guard import (
    DIRECTORY,
    FILE,
    LINK,
    RELOCATED_PREFIX,
    differences,
    snapshot,
    volatile_paths,
)


@pytest.fixture(autouse=True)
def _account_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the account's registry out of these digests.

    The guard watches the account's project registry as well as the project,
    because a sandbox must never register itself. A test's digest would otherwise
    depend on whatever projects the reader's own account holds.
    """

    empty = tmp_path / "account-config"
    empty.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(empty))


def test_the_guard_covers_the_whole_project_and_names_what_it_skips(
    project: Path,
) -> None:
    guarded = snapshot(project)
    # Nothing inside a project is unguarded but the process state a serving app
    # rewrites, and the exclusions are stated in the record so a reader can see
    # them rather than trust them.
    assert guarded.volatile == volatile_paths()
    assert all(
        path.startswith(".research-rag/runtime/") for path in guarded.volatile
    ), guarded.volatile
    assert "sources/a-book.pdf" in guarded.entries
    assert guarded.unreadable == ()


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
        f".research-rag/runtime/generations/{FIRST_GENERATION}",
        ".research-rag/runtime/generations",
        ".research-rag/runtime/staging",
        ".research-rag/runtime/failures",
        ".research-rag/runtime",
        ".research-rag",
        "sources",
    }
    kinds = {kind for kind, _size, _digest in before.entries.values()}
    assert kinds == {FILE, DIRECTORY}
    assert before.byte_count > 0


def test_the_digest_is_one_value_that_covers_content_and_name(
    project: Path,
) -> None:
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


def test_a_file_at_the_project_root_is_guarded(project: Path) -> None:
    before = snapshot(project)
    (project / "README.md").write_text("notes\n", encoding="utf-8")
    assert differences(before, snapshot(project)) == ["added: README.md"]


def test_a_change_at_the_project_root_is_named(project: Path) -> None:
    (project / "AGENTS.md").write_text("rules\n", encoding="utf-8")
    before = snapshot(project)
    (project / "AGENTS.md").write_text("different rules\n", encoding="utf-8")
    assert differences(before, snapshot(project)) == ["changed: AGENTS.md"]


def test_an_emptied_directory_is_a_change_rather_than_an_absence(
    project: Path,
) -> None:
    empty = project / "notes"
    empty.mkdir()
    (empty / "one.md").write_text("one\n", encoding="utf-8")
    before = snapshot(project)
    (empty / "one.md").unlink()
    assert differences(before, snapshot(project)) == ["removed: notes/one.md"]
    # The directory itself is still recorded, so removing it is a further change.
    removed_directory = snapshot(project)
    empty.rmdir()
    assert differences(removed_directory, snapshot(project)) == ["removed: notes"]


def test_a_change_deep_in_a_generation_is_named(project: Path) -> None:
    before = snapshot(project)
    manifest = (
        project / ".research-rag" / "runtime" / "generations" / FIRST_GENERATION
    ) / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert differences(before, snapshot(project)) == [
        f"changed: .research-rag/runtime/generations/{FIRST_GENERATION}/manifest.json"
    ]


def test_a_symlink_is_recorded_by_its_target_and_never_followed(
    project: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("not part of the project", encoding="utf-8")
    link = project / "sources" / "link.txt"
    link.symlink_to(outside)
    before = snapshot(project)
    assert before.entries["sources/link.txt"][0] == LINK
    outside.write_text("changed outside the project", encoding="utf-8")
    # The link's own target did not move, so the project did not either.
    assert differences(before, snapshot(project)) == []
    # Repointing the link is a change to the project.
    link.unlink()
    link.symlink_to(tmp_path / "elsewhere.txt")
    assert differences(before, snapshot(project)) == ["changed: sources/link.txt"]


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


def test_a_symlinked_review_state_is_recorded_and_never_followed(
    project: Path,
) -> None:
    portable = project / ".research-rag"
    moved = project / "elsewhere"
    portable.rename(moved)
    portable.symlink_to(moved, target_is_directory=True)
    guarded = snapshot(project)
    # The link is a path in the project like any other, and nothing behind it was
    # read: the guard followed it nowhere, so a corpus kept outside the project
    # could not be read through it and could not be mistaken for the project's.
    assert guarded.entries[".research-rag"][0] == LINK
    assert not any(key.startswith(".research-rag/") for key in guarded.entries)


def test_a_project_whose_review_state_is_a_symlink_is_refused(project: Path) -> None:
    from rag_experiments.sandbox import read_layout

    portable = project / ".research-rag"
    moved = project / "elsewhere"
    portable.rename(moved)
    portable.symlink_to(moved, target_is_directory=True)
    # Reading a project's layout resolves the names a copy is made of, and a
    # symlink there would make the copy follow it out of the project.
    with pytest.raises(ExperimentError) as caught:
        read_layout(project)
    assert "symlink" in str(caught.value)


def test_the_process_state_a_serving_app_rewrites_is_not_a_false_positive(
    project: Path,
) -> None:
    before = snapshot(project)
    state = project / ".research-rag" / "runtime"
    (state / "research-rag-ui.pid").write_text("4242", encoding="utf-8")
    (state / "research-rag-ui.port").write_text("5051", encoding="utf-8")
    (state / "research-rag-ui.tty").write_text("/dev/pts/4", encoding="utf-8")
    (state / "project.lock").write_text("4242", encoding="utf-8")
    (state / "logs" / "vanilla-gateway-stderr.log").write_text(
        "noise", encoding="utf-8"
    )
    gateway = state / "ultrarag-runtime" / "corpus"
    gateway.mkdir(parents=True)
    (gateway / "index.mmap").write_bytes(b"\0" * 64)
    after = snapshot(project)
    # An app that is being served rewrites all of that while a run is going, and a
    # guard that named it would fail every run against a live project.
    assert differences(before, after) == []


def test_anything_else_the_app_writes_while_a_run_goes_is_named(
    project: Path,
) -> None:
    before = snapshot(project)
    state = project / ".research-rag" / "runtime"
    (state / "generations" / "20260202T000000Z-cafebabe").mkdir(parents=True)
    (state / "generations" / "20260202T000000Z-cafebabe" / "manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    assert any(
        line.startswith("added:") and "20260202T000000Z-cafebabe" in line
        for line in differences(before, snapshot(project))
    )


def test_a_relocated_runtime_directory_is_guarded_too(
    project: Path, tmp_path: Path
) -> None:
    relocated = tmp_path / "state-elsewhere"
    moved_generation = relocated / "generations" / FIRST_GENERATION
    moved_generation.mkdir(parents=True)
    original = project / ".research-rag" / "runtime" / "generations" / FIRST_GENERATION
    for path in original.iterdir():
        (moved_generation / path.name).write_bytes(path.read_bytes())
    shutil.rmtree(original)
    (project / ".research-rag" / "runtime-root").write_text(
        f"{relocated}\n", encoding="utf-8"
    )

    before = snapshot(project)
    assert f"{RELOCATED_PREFIX}generations/{FIRST_GENERATION}/manifest.json" in (
        before.entries
    )
    assert before.relocated_root == relocated
    (moved_generation / "manifest.json").write_text("{}\n", encoding="utf-8")
    assert differences(before, snapshot(project)) == [
        f"changed: {RELOCATED_PREFIX}generations/{FIRST_GENERATION}/manifest.json"
    ]


def test_a_relocation_record_naming_nothing_is_refused(
    project: Path, tmp_path: Path
) -> None:
    (project / ".research-rag" / "runtime-root").write_text(
        f"{tmp_path / 'absent'}\n", encoding="utf-8"
    )
    with pytest.raises(ExperimentError) as caught:
        snapshot(project)
    assert "records its derived state" in str(caught.value)


def test_the_account_registry_is_guarded_because_a_copy_must_never_join_it(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import platformdirs

    registry = platformdirs.user_config_path("research-rag") / "projects.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text('{"projects": []}\n', encoding="utf-8")
    before = snapshot(project)
    assert before.registry_digest is not None
    assert before.describe()["account_registry"]["path"] == str(registry)
    registry.write_text('{"projects": ["a"]}\n', encoding="utf-8")
    assert differences(before, snapshot(project)) == [
        f"account registry changed: {registry}"
    ]
