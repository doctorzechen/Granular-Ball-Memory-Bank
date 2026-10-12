"""Failed timing-quality checks must not produce a completed-stage marker."""
from types import SimpleNamespace
import pytest
import reproduction
import experiments.query_benchmark as benchmark

def test_failed_query_quality_gate_does_not_mark_execution_complete(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    output = tmp_path / "run"
    args = SimpleNamespace(out_root=output, assets_root=assets,
                           dataset=assets / "data", frozen_root=assets / "frozen",
                           device="cpu", mode="frozen", categories=("toothbrush",),
                           stages=("query",))
    monkeypatch.setattr(reproduction, "doctor", lambda args: {})
    monkeypatch.setattr(reproduction, "clone_inputs", lambda *args: None)
    monkeypatch.setattr(benchmark, "run_benchmark", lambda *args, **kwargs: {"analysis": {"validation": {"pass": False}}})
    with pytest.raises(RuntimeError, match="Query timing quality gate failed"):
        reproduction.run(args)
    assert not (output / "execution_complete.json").exists()
