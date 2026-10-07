"""That every measured child runs at the lowest priority, or does not run.

The rule this toolkit holds is not "usually nice": a child that could not be
lowered must fail rather than measure at normal priority. These tests exercise the
launch wrapper directly -- the priority the child actually observes, the priority a
grandchild inherits, the unchanged argv, and both refusal paths.
"""

from __future__ import annotations

import sys

import pytest

from rag_experiments import niceness
from rag_experiments.errors import ExperimentError
from rag_experiments.niceness import NICE_LEVEL, run_low_priority, supported

pytestmark = pytest.mark.skipif(
    not supported(), reason="this host cannot establish a POSIX process priority"
)

PRIORITY_PROBE = "import os; print(os.getpriority(os.PRIO_PROCESS, 0))"


def test_the_wrapper_runs_the_child_at_the_lowest_priority() -> None:
    completed = run_low_priority(
        [sys.executable, "-c", PRIORITY_PROBE], capture_output=True, text=True
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == str(NICE_LEVEL)


def test_a_grandchild_inherits_the_wrapper_priority() -> None:
    # The measured program spawns its own children without this wrapper; niceness
    # is inherited across fork/exec, so those children are low priority too.
    nested = (
        "import subprocess, sys, os\n"
        f"out = subprocess.run([sys.executable, '-c', {PRIORITY_PROBE!r}],"
        " capture_output=True, text=True)\n"
        "print(out.stdout.strip())\n"
    )
    completed = run_low_priority(
        [sys.executable, "-c", nested], capture_output=True, text=True
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == str(NICE_LEVEL)


def test_the_wrapper_does_not_rewrite_the_command() -> None:
    # Provenance records the argv the caller asked for; priority is applied by the
    # kernel, not by prepending a runner to the command.
    marker = ["alpha", "beta gamma", "--flag=value"]
    completed = run_low_priority(
        [sys.executable, "-c", "import sys; print(sys.argv[1:])", *marker],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == str(marker)


def test_a_host_that_cannot_set_priority_refuses_the_launch(monkeypatch) -> None:
    monkeypatch.setattr(niceness, "supported", lambda: False)
    with pytest.raises(ExperimentError, match="low-priority"):
        run_low_priority([sys.executable, "-c", "print('never')"])


def test_a_child_that_cannot_reach_the_level_refuses_the_launch(monkeypatch) -> None:
    def _cannot() -> None:
        raise RuntimeError("simulated inability to lower priority")

    monkeypatch.setattr(niceness, "_apply_low_priority", _cannot)
    with pytest.raises(ExperimentError, match="refusing to run it at normal priority"):
        run_low_priority(
            [sys.executable, "-c", "print('never')"], capture_output=True, text=True
        )
