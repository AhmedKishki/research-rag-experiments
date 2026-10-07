"""Bound and serialize every measured child, or refuse to run it.

Niceness alone does not protect the machine. A benchmark that loads two ONNX
models and starts a gateway competes for memory as much as for CPU, and a process
that reaches normal priority but still allocates without bound can drive the whole
host into swap and freeze it. Running *nice* changes who wins the CPU; it says
nothing about how much memory a child may hold or how many threads it may start.

This module is the one place that decides those limits. A measured child is
started inside a cgroup v2 transient scope created through the reader's own
``systemd --user`` manager, so the whole process tree the child spawns shares one
enforceable subset of the machine:

- ``MemoryMax`` is the total RSS plus page cache the tree may hold. It is not
  ``RLIMIT_AS``: an address-space limit counts reserved-but-untouched mappings and
  would refuse a child that is nowhere near its real footprint, while the cgroup
  bound counts what the kernel actually charges. When the bound is reached the
  cgroup's own OOM killer ends the workload inside its cgroup; the host is not
  driven into a global OOM.
- ``MemorySwapMax=0`` forbids the tree from swapping at all, so a heavy workload
  cannot fill the swap file and stall every interactive process.
- ``CPUQuota`` caps total CPU across the whole tree, and ``TasksMax`` caps how many
  threads and processes it may create, together a conservative ceiling on a
  numerical library that would otherwise use every core.
- ``KillMode`` is the scope default (``control-group``): stopping the scope ends
  every descendant, not just the process the toolkit started.

The rule is fail-closed, and this module verifies enforcement rather than trusting
the request. The capability probe creates a real memory-limited scope and reads
``memory.max`` back from the child's own cgroup; a host whose user manager accepts
the property but does not delegate the memory controller is refused. A host
without cgroup v2, ``systemd-run``, or a running user manager is refused. An
unbounded child is never a fallback.

Heavy work is serialized across separate invocations by a file lock, so two runs
started from two shells cannot overlap and double the load. The lock is advisory
and cross-process: a second run waits a bounded time and then refuses, naming the
lock it could not take.

Nothing here rewrites the command. The provenance record keeps the argv the caller
asked for, because the limits are applied by the kernel to the process tree, not by
changing what was run.
"""

from __future__ import annotations

import fcntl
import itertools
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ExperimentError

MIB = 1024 * 1024
GIB = 1024 * MIB

#: How much of the machine is kept for the reader. A heavy run may use available
#: RAM minus this reserve, and never more than half of total RAM. The reserve is
#: the larger of a fixed floor and a fraction of total, so a small machine keeps a
#: usable margin and a large one does not reserve a token amount.
RESERVE_FLOOR_BYTES = 2 * GIB
RESERVE_DIVISOR = 8

#: The smallest and largest bound the derivation or an override may produce. The
#: ceiling keeps a mistyped override from disabling the safety it exists for.
CAP_FLOOR_BYTES = 1 * GIB
CAP_CEILING_BYTES = 8 * GIB
CAP_FRACTION_DENOMINATOR = 2
OVERRIDE_FLOOR_BYTES = 256 * MIB
OVERRIDE_CEILING_DENOMINATOR = 2

#: Conservative thread and CPU defaults. One quarter of the cores, never more
#: than four, so a machine with many cores is not fully consumed by one measured
#: child. The CPU quota is the thread budget expressed as a percentage.
MIN_THREADS = 1
MAX_THREADS = 4
THREADS_PER_CORE_DIVISOR = 4
DEFAULT_TASKS_MAX = 256
MIN_TASKS_MAX = 64
MAX_TASKS_MAX = 1024

#: How long a second heavy invocation waits for the first before refusing. The
#: work this lock guards is minutes to hours, so a short wait is a courtesy to a
#: run that is about to finish, not a queue position.
DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0

#: The environment overrides a reader may set to raise (or, to a point, lower) a
#: bound. Names are stated rather than read from a table so a documentation change
#: and a code change are the same edit.
MEMORY_OVERRIDE_ENV = "RAG_EXPERIMENTS_MEMORY_MAX_MIB"
CPU_QUOTA_OVERRIDE_ENV = "RAG_EXPERIMENTS_CPU_QUOTA_PERCENT"
THREADS_OVERRIDE_ENV = "RAG_EXPERIMENTS_THREADS"
TASKS_OVERRIDE_ENV = "RAG_EXPERIMENTS_TASKS_MAX"
LOCK_TIMEOUT_ENV = "RAG_EXPERIMENTS_LOCK_TIMEOUT_SECONDS"

#: The variables a numerical library reads for its default thread pool. They are
#: set for a measured child so a library that honours them does not oversubscribe
#: the machine; the cgroup quota is the enforcement when a library ignores them.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class Bounds:
    """The limits one measured child tree is started under, as they will be set."""

    memory_max_bytes: int
    memory_swap_max_bytes: int
    cpu_quota_percent: int
    tasks_max: int
    threads: int
    origin: str

    def systemd_properties(self) -> list[str]:
        """The ``-p`` assignments that carry these bounds to ``systemd-run``."""

        return [
            f"MemoryMax={self.memory_max_bytes}",
            f"MemorySwapMax={self.memory_swap_max_bytes}",
            f"CPUQuota={self.cpu_quota_percent}%",
            f"TasksMax={self.tasks_max}",
        ]

    def describe(self) -> dict[str, Any]:
        """The record form, in bytes and in a unit a reader can read."""

        return {
            "memory_max_bytes": self.memory_max_bytes,
            "memory_max_mib": round(self.memory_max_bytes / MIB, 1),
            "memory_swap_max_bytes": self.memory_swap_max_bytes,
            "cpu_quota_percent": self.cpu_quota_percent,
            "tasks_max": self.tasks_max,
            "threads": self.threads,
            "origin": self.origin,
            "enforced_by": "systemd-run --user --scope (cgroup v2)",
            "swap_note": (
                "MemorySwapMax=0: the tree cannot swap, so it cannot fill the swap "
                "file and stall the host; the cgroup OOM killer ends it instead."
            ),
            "memory_note": (
                "MemoryMax is a cgroup RSS+page-cache limit, not RLIMIT_AS: it "
                "counts memory the workload actually holds."
            ),
        }


@dataclass(frozen=True, slots=True)
class Capability:
    """Whether this host can enforce a bounded child tree, and why not if not."""

    available: bool
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


_CAPABILITY: Capability | None = None


def capability(*, refresh: bool = False) -> Capability:
    """The host's ability to enforce cgroup limits, probed once and remembered."""

    global _CAPABILITY
    if _CAPABILITY is not None and not refresh:
        return _CAPABILITY
    _CAPABILITY = _probe_capability()
    return _CAPABILITY


def _probe_capability() -> Capability:
    """Check for cgroup v2 and a user manager, then prove the memory bound applies."""

    if not sys.platform.startswith("linux"):
        return Capability(False, "cgroup enforcement requires Linux", {})
    if not Path("/sys/fs/cgroup/cgroup.controllers").is_file():
        return Capability(False, "cgroup v2 is not mounted at /sys/fs/cgroup", {})
    systemd_run = shutil.which("systemd-run")
    systemctl = shutil.which("systemctl")
    if not systemd_run or not systemctl:
        return Capability(False, "systemd-run and systemctl are both required", {})
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    if not Path(runtime).is_dir():
        return Capability(False, f"no user systemd runtime directory at {runtime}", {})

    expected = 64 * MIB
    code = (
        "import pathlib;"
        " p=pathlib.Path('/proc/self/cgroup').read_text().split('::',1)[1].strip();"
        " print((pathlib.Path('/sys/fs/cgroup')/p.lstrip('/')/'memory.max')"
        ".read_text().strip())"
    )
    command = [
        systemd_run,
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"MemoryMax={expected}",
        "-p",
        "MemorySwapMax=0",
        sys.executable,
        "-c",
        code,
    ]
    detail = {
        "systemd_run": systemd_run,
        "systemctl": systemctl,
        "runtime_dir": runtime,
    }
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Capability(False, f"the cgroup probe could not run: {exc}", detail)
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        return Capability(
            False,
            f"systemd-run refused a memory-limited scope: {message or completed.returncode}",
            detail,
        )
    observed = completed.stdout.strip()
    if observed != str(expected):
        return Capability(
            False,
            "the scope did not enforce MemoryMax "
            f"(memory.max={observed!r}, expected {expected}); the memory controller "
            "is not delegated to this user manager",
            detail,
        )
    return Capability(True, "cgroup v2 via systemd-run --user --scope", detail)


def _env_int(name: str) -> int | None:
    """A positive integer override, or ``None`` when unset or unusable."""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ExperimentError(f"{name} must be a whole number, not {raw!r}") from exc
    if value <= 0:
        raise ExperimentError(f"{name} must be greater than zero, not {value}")
    return value


def _meminfo_bytes(key: str) -> int | None:
    """One ``/proc/meminfo`` value in bytes, or ``None`` when it cannot be read."""

    try:
        lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith(key + ":"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def _default_threads() -> int:
    """The conservative thread budget: a quarter of the cores, capped."""

    cores = os.cpu_count() or 1
    derived = max(MIN_THREADS, min(MAX_THREADS, cores // THREADS_PER_CORE_DIVISOR))
    override = _env_int(THREADS_OVERRIDE_ENV)
    if override is None:
        return derived
    return max(MIN_THREADS, min(override, cores))


def thread_environment(threads: int | None = None) -> dict[str, str]:
    """The thread-pool variables a measured child is given."""

    value = str(_default_threads() if threads is None else threads)
    return {name: value for name in THREAD_ENV_VARS}


def default_bounds() -> Bounds:
    """The bounds a child is started under, derived from the machine, or overridden.

    The memory bound is available RAM minus a reserve for the reader, clamped to a
    floor and a ceiling and never above half of total RAM. The derivation is stable
    within one process, so two children of one run are bounded the same way.
    """

    total = _meminfo_bytes("MemTotal") or 0
    available = _meminfo_bytes("MemAvailable") or _meminfo_bytes("MemFree") or 0
    if total <= 0 or available <= 0:
        raise ExperimentError(
            "Cannot read /proc/meminfo, so the memory a measured run may use cannot "
            "be bounded; refusing to run an unbounded child."
        )

    reserve = max(RESERVE_FLOOR_BYTES, total // RESERVE_DIVISOR)
    derived = available - reserve
    derived = max(
        CAP_FLOOR_BYTES,
        min(derived, CAP_CEILING_BYTES, total // CAP_FRACTION_DENOMINATOR),
    )

    memory_override = _env_int(MEMORY_OVERRIDE_ENV)
    memory_max = derived
    origin = "derived"
    if memory_override is not None:
        memory_max = max(
            OVERRIDE_FLOOR_BYTES,
            min(
                memory_override * MIB,
                total // OVERRIDE_CEILING_DENOMINATOR,
            ),
        )
        origin = "environment"

    threads = _default_threads()
    cores = os.cpu_count() or 1
    cpu_override = _env_int(CPU_QUOTA_OVERRIDE_ENV)
    cpu_quota = cpu_override if cpu_override is not None else threads * 100
    cpu_quota = max(100, min(cpu_quota, cores * 100))
    if cpu_override is not None:
        origin = "environment"

    tasks_override = _env_int(TASKS_OVERRIDE_ENV)
    tasks_max = tasks_override if tasks_override is not None else DEFAULT_TASKS_MAX
    tasks_max = max(MIN_TASKS_MAX, min(tasks_max, MAX_TASKS_MAX))
    if tasks_override is not None:
        origin = "environment"

    return Bounds(
        memory_max_bytes=memory_max,
        memory_swap_max_bytes=0,
        cpu_quota_percent=cpu_quota,
        tasks_max=tasks_max,
        threads=threads,
        origin=origin,
    )


def lock_path() -> Path:
    """Where the cross-invocation heavy-work lock lives."""

    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "rag-experiments" / "heavy.lock"


def lock_timeout() -> float:
    """How long a heavy invocation waits for the lock before refusing."""

    override = _env_int(LOCK_TIMEOUT_ENV)
    if override is None:
        return DEFAULT_LOCK_TIMEOUT_SECONDS
    return float(override)


@contextmanager
def heavy_lock(timeout: float | None = None) -> Iterator[Path]:
    """Hold the cross-process heavy-work lock, or refuse after ``timeout`` seconds.

    ``flock`` is used rather than a lock file's existence: the kernel releases it
    when the holder exits, so a crashed run does not leave the machine locked out.
    """

    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    deadline = time.monotonic() + (lock_timeout() if timeout is None else timeout)
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise ExperimentError(
                        "Another resource-bounded run holds the heavy-work lock at "
                        f"{path}; refusing to start a second overlapping workload. "
                        "Wait for it to finish, or raise "
                        f"{LOCK_TIMEOUT_ENV}."
                    ) from None
                time.sleep(0.2)
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


_UNITS = itertools.count()


def _unit_name() -> str:
    """A unique, valid transient-unit name for this invocation's scope."""

    return f"rag-experiments-heavy-{os.getpid()}-{next(_UNITS)}"


def _systemctl() -> str | None:
    return capability().detail.get("systemctl") or shutil.which("systemctl")


def _stop_unit(unit: str) -> None:
    """End the whole scope, and every descendant in it, when a child overruns."""

    systemctl = _systemctl()
    if not systemctl:
        return
    # `systemctl` infers `.service` from a bare name, so the scope suffix is named
    # explicitly; a stop that resolved to the wrong unit would leave the tree alive.
    scope = unit if unit.endswith(".scope") else f"{unit}.scope"
    for arguments in (
        ["stop", scope],
        ["kill", "--signal=SIGKILL", "--kill-whom=all", scope],
    ):
        try:
            subprocess.run(
                [systemctl, "--user", *arguments],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue


def _terminate(process: subprocess.Popen) -> None:
    """Kill the launcher if it is still alive, without masking the real error."""

    if process.poll() is not None:
        return
    with suppress(OSError):
        process.kill()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)


def launch(
    command: Sequence[str],
    *,
    preexec_fn: Any,
    timeout: int | None = None,
    check: bool = False,
    capture_output: bool = False,
    text: bool = False,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    stdin_data: str | bytes | None = None,
) -> subprocess.CompletedProcess:
    """Run ``command`` inside a bounded, serialized cgroup, or refuse.

    The child runs at the caller's priority (applied by ``preexec_fn``) *and*
    inside a transient scope whose memory, swap, CPU, and task limits the kernel
    enforces across the whole tree. The completed process carries the applied
    bounds on ``resource_bounds`` and the scope name on ``resource_unit``.
    """

    cap = capability()
    if not cap.available:
        raise ExperimentError(
            "Resource enforcement is unavailable on this host "
            f"({cap.reason}), so a measured or preparation command is refused "
            "rather than run unbounded. Install/again enable cgroup v2 with a "
            f"running systemd --user manager. See README. Command: {command[0]!r}"
        )
    bounds = default_bounds()
    systemd_run = cap.detail["systemd_run"]

    with heavy_lock():
        unit = _unit_name()
        argv = [
            systemd_run,
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            f"--unit={unit}",
            "-p",
            f"MemoryMax={bounds.memory_max_bytes}",
            "-p",
            f"MemorySwapMax={bounds.memory_swap_max_bytes}",
            "-p",
            f"CPUQuota={bounds.cpu_quota_percent}%",
            "-p",
            f"TasksMax={bounds.tasks_max}",
            "--",
            *command,
        ]
        kwargs: dict[str, Any] = {
            "preexec_fn": preexec_fn,
            "cwd": cwd,
            "env": env,
            "text": text,
        }
        if capture_output:
            kwargs["stdout"] = subprocess.PIPE
            kwargs["stderr"] = subprocess.PIPE
        if stdin_data is not None:
            kwargs["stdin"] = subprocess.PIPE
        try:
            process = subprocess.Popen(argv, **kwargs)
        except subprocess.SubprocessError as exc:
            # A priority or resource probe that failed between fork and exec lands
            # here; the child was never started at normal or unbounded priority.
            raise ExperimentError(
                "Cannot establish low-priority, resource-bounded execution for "
                f"{command[0]!r}; refusing to run it at normal priority: {exc}"
            ) from exc

        stdout: Any = None
        stderr: Any = None
        try:
            stdout, stderr = process.communicate(stdin_data, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _stop_unit(unit)
            _terminate(process)
            with suppress(subprocess.TimeoutExpired, OSError, ValueError):
                process.communicate(timeout=5)
            raise subprocess.TimeoutExpired(
                command, timeout, output=exc.output, stderr=exc.stderr
            ) from exc
        except BaseException:
            # An interrupt must not leave the scope (and its descendants) running.
            _stop_unit(unit)
            _terminate(process)
            raise
        finally:
            if process.poll() is None:
                _terminate(process)

    completed = subprocess.CompletedProcess(
        command, int(process.returncode), stdout, stderr
    )
    completed.resource_bounds = bounds  # type: ignore[attr-defined]
    completed.resource_unit = unit  # type: ignore[attr-defined]
    if check and process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode, command, output=stdout, stderr=stderr
        )
    return completed


def describe() -> dict[str, Any]:
    """The capability, bounds, and lock a record carries to explain a launch."""

    cap = capability()
    record: dict[str, Any] = {
        "capability": {"available": cap.available, "reason": cap.reason},
        "lock_path": str(lock_path()),
        "lock_timeout_seconds": lock_timeout(),
    }
    if cap.available:
        try:
            record["bounds"] = default_bounds().describe()
        except ExperimentError as exc:
            record["bounds_error"] = str(exc)
    return record


__all__ = [
    "Bounds",
    "Capability",
    "capability",
    "default_bounds",
    "describe",
    "heavy_lock",
    "launch",
    "lock_path",
    "lock_timeout",
    "thread_environment",
]
