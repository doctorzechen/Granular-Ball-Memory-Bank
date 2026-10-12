"""Guard responsibility boundaries rather than counting authorship percentages."""
import ast
from pathlib import Path
import pytest
import reproduction

ROOT = Path(__file__).resolve().parents[1]

def test_no_active_historical_module_dependencies():
    files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "experiments").rglob("*.py"))
    files.append(ROOT / "reproduction.py")
    forbidden = ("patchcore.stage", "patchcore.granular_ball", "patchcore.intra_ball", "revision_evidence.")
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(m.startswith(forbidden) for m in modules), (path, modules)

def test_core_has_no_adapter_or_search_dependency():
    for path in (ROOT / "src/gb_memory").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(m.split(".")[0] in {"patchcore", "adapters", "faiss", "experiments"} for m in modules)

def test_output_guard_rejects_package_ancestors_and_source():
    for path in (ROOT.parent, ROOT / "src/new", ROOT / "experiments/new", ROOT / "third_party/new"):
        with pytest.raises(ValueError):
            reproduction.guard_output(path)
    reproduction.guard_output(ROOT / "runs/test")
