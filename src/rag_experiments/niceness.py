"""Run every measured child at the lowest scheduler priority, or refuse to run it.

Experiments share the machine with whatever the reader is doing. A benchmark that
competes at normal priority changes the very latency it is trying to measure and
slows every other process. This module is the one place that decides how a child
is started on that count, so no call site can forget it.

On Linux (and any POSIX host) a process changes its own niceness with
``os.nice``. The wrapper runs that between ``fork`` and ``exec`` through
``preexec_fn``, so ``19`` is in force before the measured program's first
instruction. Niceness is inherited across ``fork`` and preserved across ``exec``,
so every descendant the measured program spawns -- a gateway, a thread pool, an
in-process fork -- runs at ``19`` too, without cooperating.

Niceness is necessary but not sufficient. It decides who wins the CPU, not how
much memory a child may hold or how many threads it may start, and a run under
``nice`` can still drive the host into swap and freeze it. So the launch itself is
delegated to :mod:`rag_experiments.resource_limits`, which starts the child inside
a cgroup v2 scope with an enforced memory, swap, CPU, and task bound, and
serializes heavy work across invocations. That module owns those limits; this one
owns priority. Both are fail-closed: a host that cannot set priority, or cannot
enforce a bound, refuses the launch rather than running it at normal or unbounded
priority.

Nothing here changes the command line. The provenance record keeps the argv the
caller asked for, because the priority and the limits are applied by the kernel to
the process tree, not by rewriting what was run.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

from . import resource_limits
from .errors import ExperimentError

#: The niceness every measured child runs at. The maximum a normal user can set,
#: so a measured run cannot outrank the reader's interactive work.
NICE_LEVEL = 19


def supported() -> bool:
    """Whether this host can establish and verify a child's priority.

    POSIX hosts expose ``os.nice`` and ``os.getpriority``. A host without both
    cannot be held to the rule, so the launch is refused rather than approximated.
    """

    return (
        os.name == "posix"
        and hasattr(os, "nice")
        and hasattr(os, "getpriority")
        and hasattr(os, "PRIO_PROCESS")
    )


def _apply_low_priority() -> None:
    """Set this process to :data:`NICE_LEVEL`, refusing to continue otherwise.

    This runs in the child between ``fork`` and ``exec``. Raising here makes
    ``subprocess`` abort the launch with ``SubprocessError`` rather than run the
    measured program at normal priority; ``run_low_priority`` turns that into an
    ``ExperimentError`` naming the command.
    """

    if not supported():
        raise RuntimeError("this host cannot set a POSIX process priority")

    try:
        current = os.getpriority(os.PRIO_PROCESS, 0)
    except OSError as exc:  # pragma: no cover - getpriority rarely fails
        raise RuntimeError(f"cannot read the process priority: {exc}") from exc

    if current < NICE_LEVEL:
        try:
            # ``os.nice`` adds to the current value and clamps at ``19``.
            os.nice(NICE_LEVEL - current)
        except OSError as exc:
            raise RuntimeError(f"cannot lower the process priority: {exc}") from exc

    try:
        achieved = os.getpriority(os.PRIO_PROCESS, 0)
    except OSError as exc:  # pragma: no cover - getpriority rarely fails
        raise RuntimeError(f"cannot confirm the process priority: {exc}") from exc
    if achieved < NICE_LEVEL:
        raise RuntimeError(
            f"the process priority is {achieved}, not the requested {NICE_LEVEL}"
        )


def priority_preexec() -> Any:
    """The ``preexec_fn`` a measured child is started with."""

    if not supported():
        raise ExperimentError(
            "This host cannot establish low-priority (nice 19) execution, so a "
            "measured or preparation command is refused rather than run at normal "
            "priority."
        )
    return _apply_low_priority


def run_low_priority(command: Any, **kwargs: Any) -> subprocess.CompletedProcess:
    """Run a child at :data:`NICE_LEVEL` inside :mod:`~rag_experiments.resource_limits`.

    The caller's arguments pass through unchanged, including the argv that lands in
    the provenance record. Priority is applied between ``fork`` and ``exec``; the
    resource bound is applied by the cgroup the child tree is started in. A child
    that cannot reach ``19``, or a host that cannot enforce a bound, is never
    executed at normal or unbounded priority: the failure is ``ExperimentError``.
    """

    preexec = priority_preexec()
    check = kwargs.pop("check", False)
    stdin_data = kwargs.pop("input", None)
    try:
        return resource_limits.launch(
            command,
            preexec_fn=preexec,
            check=check,
            stdin_data=stdin_data,
            **kwargs,
        )
    except subprocess.CalledProcessError:
        # An explicit check=True is the caller's own contract.
        raise
    except subprocess.TimeoutExpired:
        # A timeout is the caller's own contract and carries its own message.
        raise
    except ExperimentError:
        # A missing enforcement capability is already a clear, fail-closed refusal.
        raise
    except subprocess.SubprocessError as exc:
        raise ExperimentError(
            f"Cannot establish low-priority, resource-bounded execution for "
            f"{command[0]!r}; refusing to run it at normal priority: {exc}"
        ) from exc


__all__ = [
    "NICE_LEVEL",
    "priority_preexec",
    "run_low_priority",
    "supported",
]
