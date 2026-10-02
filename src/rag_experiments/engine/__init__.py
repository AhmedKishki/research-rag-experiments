"""Locating the engine under test, and running inside it."""

from __future__ import annotations

from .harness import HarnessContract, harness_contract
from .locate import (
    ENGINE_HARNESS_ENV,
    EngineSource,
    git_diff_of,
    git_revision_of,
    git_untracked_of,
    locate_engine,
    tree_digest,
)
from .runner import (
    SETTINGS_PREFIXES,
    SettingsAnswer,
    child_environment,
    describe_child_environment,
    resolve_settings_for_tree,
)

__all__ = [
    "ENGINE_HARNESS_ENV",
    "SETTINGS_PREFIXES",
    "EngineSource",
    "HarnessContract",
    "SettingsAnswer",
    "child_environment",
    "describe_child_environment",
    "git_diff_of",
    "git_revision_of",
    "git_untracked_of",
    "harness_contract",
    "locate_engine",
    "resolve_settings_for_tree",
    "tree_digest",
]
