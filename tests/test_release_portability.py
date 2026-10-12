"""Release contracts: source-only CLI, input guards, dependencies, packaging."""
import ast
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_entry():
    spec = importlib.util.spec_from_file_location("release_entry_audit", ROOT / "reproduction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_runs_without_site_packages_from_another_directory(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "reproduction.py"), "--help"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--assets-root" in result.stdout
    assert "--mode" in result.stdout


def test_fresh_defaults_resolve_readonly_assets_without_importing_asset_code(tmp_path, monkeypatch):
    entry = load_entry()
    assets, output = tmp_path / "assets", tmp_path / "output"
    captured = {}
    monkeypatch.setattr(entry, "configure", lambda root: captured.update(configured_assets=root))
    monkeypatch.setattr(entry, "doctor", lambda args: captured.update(args=args))
    monkeypatch.setattr(sys, "argv", [str(ROOT / "reproduction.py"), "doctor",
                                     "--assets-root", str(assets), "--out-root", str(output)])
    entry.main()
    args = captured["args"]
    assert args.mode == "fresh"
    assert args.dataset == assets / "data" / "mvtec"
    assert args.frozen_root == assets / "frozen"
    assert args.out_root == output
    assert captured["configured_assets"] == assets


def test_output_guard_rejects_source_and_input_directories(tmp_path):
    entry = load_entry()
    for folder in (ROOT, ROOT.parent, ROOT / "src", ROOT / "src/gb_memory/output",
                   ROOT / "third_party/patchcore", ROOT / "reference/output"):
        with pytest.raises(ValueError):
            entry.guard_output(folder)
    data, weights = tmp_path / "input/data", tmp_path / "input/weights"
    for folder in (data, data / "output", weights / "output"):
        with pytest.raises(ValueError):
            entry.guard_output(folder, (data, weights))
    entry.guard_output(ROOT / "runs/portability_smoke")
    entry.guard_output(tmp_path / "fresh_output", (data, weights))


def test_active_modules_do_not_import_historical_or_author_asset_modules():
    paths = [ROOT / "reproduction.py"]
    for folder in ("src", "experiments", "scripts"):
        paths.extend((ROOT / folder).rglob("*.py"))
    forbidden = ("patchcore.stage", "patchcore.granular_ball",
                 "patchcore.intra_ball_redundancy", "patchcore.gb_", "legacy")
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any(name.startswith(forbidden) for name in modules), str(path)


def test_offline_wheel_contains_core_vendor_notices_and_no_large_assets(tmp_path, monkeypatch):
    # Build from a scratch copy: never create build/ or egg-info/ in the release.
    import setuptools.build_meta

    shutil.copy2(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    for folder in ("src", "experiments", "third_party"):
        shutil.copytree(ROOT / folder, tmp_path / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    output = tmp_path / "artifacts"
    output.mkdir()
    monkeypatch.chdir(tmp_path)
    wheel = setuptools.build_meta.build_wheel(str(output))
    with zipfile.ZipFile(output / wheel) as archive:
        names = archive.namelist()
        assert "gb_memory/bank.py" in names
        assert "patchcore/datasets/mvtec.py" in names
        assert "experiments/main_comparison.py" in names
        assert any(name.endswith("/LICENSE") for name in names)
        assert any(name.endswith("/NOTICE") for name in names)
        assert any(name.endswith("/MODIFICATIONS.md") for name in names)
        assert not any(name.endswith((".npy", ".npz", ".pth", ".png")) for name in names)
        assert all(info.file_size < 1_000_000 for info in archive.infolist())
