"""Report optional CUDA import failures without masking the original exception."""

import builtins
import runpy
from pathlib import Path

import pytest


@pytest.fixture
def import_external(monkeypatch):
    original_import = builtins.__import__

    def load(error):
        def failing_import(name, *args, **kwargs):
            if name == "tinycudann":
                raise error
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", failing_import)
        return runpy.run_path(str(Path(__file__).resolve().parents[2] / "nerfstudio/utils/external.py"))

    return load


@pytest.mark.parametrize("error", [OSError(2, "missing shared library"), OSError()])
def test_unexpected_tcnn_oserror_is_preserved(import_external, error):
    with pytest.raises(OSError) as caught:
        import_external(error)
    assert caught.value is error


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named tinycudann"),
        ImportError("undefined symbol"),
        OSError("Unknown compute capability"),
        OSError(22, "Unknown compute capability"),
    ],
)
def test_optional_tcnn_failure_is_deferred_with_original_cause(import_external, error):
    module = import_external(error)
    assert not module["TCNN_EXISTS"]
    assert module["tcnn_import_exception"] is error
    for operation in (lambda: module["tcnn"].Network(), lambda: module["tcnn"].Network.some_attribute):
        with pytest.raises(RuntimeError, match="Could not load package tinycudann") as caught:
            operation()
        assert caught.value.__cause__ is error
