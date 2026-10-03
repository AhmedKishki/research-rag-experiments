"""Cold library reads suppress bytecode and restore the caller's setting."""

import importlib.util
import sys

import pytest

from rag_experiments._imports import without_bytecode


def test_cold_import_does_not_write_beside_source(tmp_path, monkeypatch):
    source = tmp_path / "editable_source.py"
    source.write_text("VALUE = 42\n", encoding="utf-8")
    monkeypatch.setattr(sys, "dont_write_bytecode", False)

    @without_bytecode
    def read():
        spec = importlib.util.spec_from_file_location("editable_source", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.VALUE

    assert read() == 42
    assert not (tmp_path / "__pycache__").exists()
    assert sys.dont_write_bytecode is False


def test_nested_calls_and_failures_restore_flag(monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", False)

    @without_bytecode
    def inner():
        assert sys.dont_write_bytecode is True
        raise RuntimeError("probe")

    @without_bytecode
    def outer():
        with pytest.raises(RuntimeError, match="probe"):
            inner()
        assert sys.dont_write_bytecode is True

    outer()
    assert sys.dont_write_bytecode is False


def test_snapshot_protects_parent_imports_before_scanning(project, monkeypatch):
    from rag_experiments.sandbox import guard

    original = guard.volatile_paths

    def checked():
        assert sys.dont_write_bytecode is True
        return original()

    monkeypatch.setattr(guard, "volatile_paths", checked)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    assert guard.snapshot(project).entries
    assert sys.dont_write_bytecode is False
