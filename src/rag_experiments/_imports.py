"""Suppress import caches while library entry points read an editable app.

CLI startup and measured subprocesses already disable bytecode. Programmatic
layout and guard calls need protection without changing the caller's interpreter
setting after they return. A lock orders nested and concurrent protected calls.
"""

import sys
from collections.abc import Callable
from functools import wraps
from threading import RLock
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")
_LOCK = RLock()


def without_bytecode(function: Callable[P, T]) -> Callable[P, T]:
    @wraps(function)
    def protected(*args: P.args, **kwargs: P.kwargs) -> T:
        with _LOCK:
            previous = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                return function(*args, **kwargs)
            finally:
                sys.dont_write_bytecode = previous

    return protected
