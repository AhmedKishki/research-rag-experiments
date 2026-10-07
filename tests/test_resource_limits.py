"""That a measured child is resource-bounded and serialized, or does not run.

Niceness is tested beside this file. These tests hold the stronger rule: a child is
started inside an enforcing cgroup, a second heavy invocation refuses while the
first holds the lock, a timeout ends the whole tree, and a host that cannot enforce
a bound refuses the launch rather than running it unbounded.

The enforcement tests run a tiny interpreter, not a model. One deliberately asks
for more memory than its cgroup allows, so the kill is observed inside the cgroup
and never touches the host.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_experiments import resource_limits
from rag_experiments.engine.runner import child_environment, describe_child_environment
from rag_experiments.errors import ExperimentError
from rag_experiments.niceness import run_low_priority

pytestmark = pytest.mark.skipif(
    not resource_limits.capability().available,
    reason=f"cgroup enforcement unavailable: {resource_limits.capability().reason}",
)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Keep the lock and any override out of the reader's real state for every test."""

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    for name in (
        resource_limits.MEMORY_OVERRIDE_ENV,
        resource_limits.CPU_QUOTA_OVERRIDE_ENV,
        resource_limits.THREADS_OVERRIDE_ENV,
        resource_limits.TASKS_OVERRIDE_ENV,
        resource_limits.LOCK_TIMEOUT_ENV,
        *resource_limits.THREAD_ENV_VARS,
    ):
        monkeypatch.delenv(name, raising=False)


def test_capability_proves_the_memory_bound_is_enforced() -> None:
    cap = resource_limits.capability(refresh=True)
    assert cap.available, cap.reason
    assert "cgroup v2" in cap.reason


def test_the_derived_bound_keeps_a_reserve_and_a_ceiling() -> None:
    bounds = resource_limits.default_bounds()
    total = resource_limits._meminfo_bytes("MemTotal") or 0
    assert bounds.memory_max_bytes >= resource_limits.CAP_FLOOR_BYTES
    assert bounds.memory_max_bytes <= resource_limits.CAP_CEILING_BYTES
    assert bounds.memory_max_bytes <= total // 2
    assert bounds.memory_swap_max_bytes == 0
    assert bounds.threads <= resource_limits.MAX_THREADS
    assert bounds.cpu_quota_percent == bounds.threads * 100
    assert bounds.tasks_max >= resource_limits.MIN_TASKS_MAX


def test_a_host_without_enforcement_refuses_the_launch(monkeypatch) -> None:
    monkeypatch.setattr(
        resource_limits,
        "capability",
        lambda **_kwargs: resource_limits.Capability(
            False, "simulated host without cgroup enforcement"
        ),
    )
    with pytest.raises(ExperimentError, match="unavailable"):
        run_low_priority(
            [sys.executable, "-c", "print('never')"], capture_output=True, text=True
        )


def test_the_lock_refuses_a_second_overlapping_run() -> None:
    # A second *invocation* is a second process, so the refusal is tested across
    # processes: the child inherits the lock path from the environment.
    script = (
        "from rag_experiments import resource_limits\n"
        "try:\n"
        "    with resource_limits.heavy_lock(timeout=0.3):\n"
        "        print('acquired')\n"
        "except Exception as exc:\n"
        "    print(f'{type(exc).__name__}: {exc}')\n"
        "    raise SystemExit(3)\n"
    )
    with resource_limits.heavy_lock(timeout=1.0):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    assert completed.returncode == 3
    assert "heavy-work lock" in completed.stdout


def test_the_runner_enforces_a_memory_bound(monkeypatch) -> None:
    monkeypatch.setenv(resource_limits.MEMORY_OVERRIDE_ENV, "256")
    code = (
        "chunks=[]\n"
        "for _ in range(400):\n"
        "    b=bytearray(1024*1024); b[0]=1; b[-1]=2; chunks.append(b)\n"
        "print('allocated')"
    )
    completed = run_low_priority(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode != 0
    assert "allocated" not in (completed.stdout or "")


def test_a_timeout_ends_the_scope_and_its_descendants(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(resource_limits.LOCK_TIMEOUT_ENV, "5")
    marker = tmp_path / "child.pid"
    child = "import time; time.sleep(120)"
    parent = (
        "import subprocess, sys, time;"
        f" p=subprocess.Popen([sys.executable, '-c', {child!r}]);"
        f" open({str(marker)!r}, 'w').write(str(p.pid));"
        " time.sleep(120)"
    )
    pid = None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            run_low_priority(
                [sys.executable, "-c", parent],
                capture_output=True,
                text=True,
                timeout=3,
            )
        # The scope was stopped, so the grandchild must be gone shortly after.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if marker.exists():
                pid = int(marker.read_text().strip())
                if not _alive(pid):
                    break
            time.sleep(0.1)
        assert pid is not None, "the parent never recorded its child"
        assert not _alive(pid), f"descendant {pid} survived the scope stop"
    finally:
        if pid is not None and _alive(pid):
            os.kill(pid, 9)


def test_the_thread_environment_is_capped() -> None:
    values = set(resource_limits.thread_environment().values())
    assert values == {str(resource_limits._default_threads())}
    assert set(resource_limits.thread_environment()) == set(
        resource_limits.THREAD_ENV_VARS
    )
    assert resource_limits._default_threads() <= resource_limits.MAX_THREADS


def test_child_environment_caps_threads_and_records_the_bound() -> None:
    engine = SimpleNamespace(source_relative="src")
    environment = child_environment(engine)
    for name in resource_limits.THREAD_ENV_VARS:
        assert environment[name] == str(resource_limits._default_threads())
    described = describe_child_environment(engine)
    assert described["resource_limits"]["capability"]["available"] is True
    assert "bounds" in described["resource_limits"]
    assert described["given"]["OMP_NUM_THREADS"] == str(
        resource_limits._default_threads()
    )


def _alive(pid: int) -> bool:
    """Whether a process is still running, treating a reaped-or-zombie as gone."""

    try:
        fields = (Path(f"/proc/{pid}/stat")).read_text(encoding="ascii").split()
    except OSError:
        return False
    return len(fields) > 2 and fields[2] != "Z"
