"""That a copy is a project the app can measure, and carries nothing else.

These are the tests that hold the isolation contract. They read the copy the way
the app's own readers do, so a copy that satisfied this harness but not the app
would fail here rather than on a real corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import (
    FIRST_GENERATION,
    PROJECT_ID,
    SECOND_GENERATION,
    build_project,
)

from rag_experiments.errors import ExperimentError
from rag_experiments.sandbox import (
    SANDBOX_RECORD,
    create,
    list_sandboxes,
    pin_settings,
    read_layout,
    remove,
    selected_generation,
    snapshot,
)
from rag_experiments.sandbox.guard import differences


def test_a_copy_is_a_project_the_apps_own_readers_accept(
    project: Path, workspace: Path
) -> None:
    sandbox = create(workspace, name="one", source=project)
    from research_rag.storage.records import load_current_generation

    layout = read_layout(sandbox.root)
    _root, manifest = load_current_generation(layout.state_root)
    assert manifest["generation_id"] == FIRST_GENERATION
    assert selected_generation(layout) == FIRST_GENERATION


def test_a_copy_keeps_the_project_id_so_source_ids_still_resolve(
    project: Path, workspace: Path
) -> None:
    sandbox = create(workspace, name="one", source=project)
    assert sandbox.project_id == PROJECT_ID
    descriptor = json.loads(
        (sandbox.root / ".research-rag" / "project.json").read_text(encoding="utf-8")
    )
    assert descriptor["project_id"] == PROJECT_ID
    catalog = json.loads(
        (sandbox.root / ".research-rag" / "source-catalog.json").read_text(
            encoding="utf-8"
        )
    )
    assert catalog["project_id"] == PROJECT_ID


def test_the_originals_are_copied_byte_for_byte(project: Path, workspace: Path) -> None:
    sandbox = create(workspace, name="one", source=project)
    for original in sorted((project / "sources").iterdir()):
        copied = sandbox.source_root / original.name
        assert copied.is_file()
        assert copied.read_bytes() == original.read_bytes()


def test_the_review_state_is_copied_and_the_runtime_is_not(
    project: Path, workspace: Path
) -> None:
    state = project / ".research-rag" / "runtime"
    (state / "research-rag-ui.pid").write_text("4242", encoding="utf-8")
    (state / "research-rag-ui.port").write_text("5051", encoding="utf-8")
    (state / "research-rag-ui.tty").write_text("/dev/pts/4", encoding="utf-8")
    (state / "project.lock").write_text("4242", encoding="utf-8")
    (state / "pending-activation.json").write_text("{}", encoding="utf-8")
    (state / "staging" / "leftover").mkdir(parents=True)
    (state / "logs" / "old.log").write_text("noise", encoding="utf-8")
    (project / ".research-rag" / "runtime-root").write_text(
        "/elsewhere\n", encoding="utf-8"
    )

    sandbox = create(workspace, name="one", source=project)
    portable = sandbox.root / ".research-rag"
    for carried in ("project.json", "source-catalog.json", "source-metadata.json"):
        assert (portable / carried).is_file()

    copied_state = portable / "runtime"
    for refused in (
        "research-rag-ui.pid",
        "research-rag-ui.port",
        "research-rag-ui.tty",
        "project.lock",
        "pending-activation.json",
    ):
        assert not (copied_state / refused).exists(), refused
    assert not (portable / "runtime-root").exists()
    assert not any((copied_state / "staging").iterdir())
    assert not any((copied_state / "logs").iterdir())


def test_only_the_named_generations_are_copied(
    two_generation_project: Path, workspace: Path
) -> None:
    sandbox = create(workspace, name="one", source=two_generation_project)
    generations = sandbox.root / ".research-rag" / "runtime" / "generations"
    assert [item.name for item in generations.iterdir()] == [SECOND_GENERATION]

    both = create(
        workspace,
        name="two",
        source=two_generation_project,
        generations=(FIRST_GENERATION, SECOND_GENERATION),
    )
    copied = both.root / ".research-rag" / "runtime" / "generations"
    assert sorted(item.name for item in copied.iterdir()) == [
        FIRST_GENERATION,
        SECOND_GENERATION,
    ]


def test_a_generation_the_project_does_not_hold_is_refused(
    two_generation_project: Path, workspace: Path
) -> None:
    with pytest.raises(ExperimentError) as caught:
        create(
            workspace,
            name="one",
            source=two_generation_project,
            generations=("20260101T000000Z-deadbeef",),
        )
    assert "has no generation" in str(caught.value)
    assert FIRST_GENERATION in str(caught.value)


def test_the_runtime_skeleton_the_app_makes_is_made_here(
    project: Path, workspace: Path
) -> None:
    sandbox = create(workspace, name="one", source=project)
    state = sandbox.root / ".research-rag" / "runtime"
    for name in ("generations", "logs", "staging", "failures", "ultrarag-runtime"):
        assert (state / name).is_dir()


def test_a_copy_carries_nothing_the_harness_added_to_the_project_root(
    project: Path, workspace: Path
) -> None:
    sandbox = create(workspace, name="one", source=project)
    # The record sits beside the project, so the copy's root holds only what the
    # app writes and a sandbox is a valid project with nothing removed.
    assert sorted(item.name for item in sandbox.root.iterdir()) == [
        ".research-rag",
        "sources",
    ]
    assert sandbox.record_path.is_file()
    assert sandbox.record_path.parent == sandbox.workspace


def test_a_symlinked_original_is_refused_rather_than_copied(
    project: Path, workspace: Path
) -> None:
    (project / "sources" / "linked.pdf").symlink_to(project / "sources" / "a-book.pdf")
    with pytest.raises(ExperimentError) as caught:
        create(workspace, name="one", source=project)
    assert "symlink" in str(caught.value)
    # Nothing is left behind: a half-built sandbox is not a project.
    assert not (workspace / "one").exists()


def test_a_name_that_is_a_path_is_refused(project: Path, workspace: Path) -> None:
    for name in ("", "a/b", ".."):
        with pytest.raises(ExperimentError):
            create(workspace, name=name, source=project)


def test_a_second_copy_under_one_name_is_refused(
    project: Path, workspace: Path
) -> None:
    create(workspace, name="one", source=project)
    with pytest.raises(ExperimentError) as caught:
        create(workspace, name="one", source=project)
    assert "already exists" in str(caught.value)
    assert "sandbox remove" in str(caught.value)


def test_only_a_copy_this_harness_made_is_removed(
    project: Path, workspace: Path, tmp_path: Path
) -> None:
    create(workspace, name="one", source=project)
    listed = list_sandboxes(workspace)
    assert [entry["name"] for entry in listed] == ["one"]
    assert listed[0]["generation_ids"] == [FIRST_GENERATION]
    assert listed[0]["project_id"] == PROJECT_ID

    stranger = workspace / "stranger"
    stranger.mkdir()
    with pytest.raises(ExperimentError) as caught:
        remove(workspace, "stranger")
    assert SANDBOX_RECORD in str(caught.value)
    assert stranger.exists()

    remove(workspace, "one")
    assert not (workspace / "one").exists()
    assert list_sandboxes(workspace) == []


def test_a_copy_leaves_the_source_project_byte_identical(
    project: Path, workspace: Path
) -> None:
    before = snapshot(project)
    create(workspace, name="one", source=project)
    assert differences(before, snapshot(project)) == []


def test_pinning_settings_writes_where_the_app_reads_them(
    project: Path, workspace: Path
) -> None:
    from research_rag.project.settings import PROJECT_CONFIG_RELATIVE

    sandbox = create(workspace, name="one", source=project)
    written = pin_settings(sandbox, "[chunking]\nsize = 512\n")
    assert written == sandbox.root / PROJECT_CONFIG_RELATIVE
    assert written.read_text(encoding="utf-8") == "[chunking]\nsize = 512\n"


def test_the_config_home_a_child_is_pointed_at_is_empty(
    project: Path, workspace: Path
) -> None:
    sandbox = create(workspace, name="one", source=project)
    assert sandbox.config_home.is_dir()
    assert list(sandbox.config_home.iterdir()) == []


def test_a_project_that_is_not_one_is_refused(tmp_path: Path, workspace: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ExperimentError) as caught:
        create(workspace, name="one", source=plain)
    assert "not a research-rag project" in str(caught.value)


def test_a_descriptor_naming_no_source_directory_is_refused(
    tmp_path: Path, workspace: Path
) -> None:
    root = build_project(tmp_path / "corpus")
    descriptor = root / ".research-rag" / "project.json"
    document = json.loads(descriptor.read_text(encoding="utf-8"))
    del document["source_directory"]
    descriptor.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ExperimentError) as caught:
        create(workspace, name="one", source=root)
    assert "source_directory" in str(caught.value)


def test_the_names_come_from_the_app_not_from_here(project: Path) -> None:
    from research_rag.project.settings import PROJECT_CONFIG_RELATIVE
    from research_rag.project.state_files import PORTABLE_DIRECTORY, RUNTIME_DIRECTORY

    layout = read_layout(project)
    assert layout.portable_directory == PORTABLE_DIRECTORY
    assert layout.runtime_directory == RUNTIME_DIRECTORY
    assert layout.settings_relative == PROJECT_CONFIG_RELATIVE.as_posix()
    assert layout.source_root == project / "sources"
    assert layout.generations_root.name == "generations"
