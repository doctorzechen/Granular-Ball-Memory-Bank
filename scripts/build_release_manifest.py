"""Record exact public source/evidence hashes, excluding external/run assets."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

EXCLUDED_ROOTS = {".git", ".pytest_cache", "build", "dist", "runs", "verification", "data", "weights", "frozen"}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def build(root, upstream=None):
    root = root.resolve()
    vendor = root / "third_party/patchcore"
    hashes = []
    for path in sorted((vendor / "src").rglob("*.py")):
        item = {"path": path.relative_to(vendor).as_posix(), "sha256": digest(path)}
        if upstream is not None:
            original = upstream / item["path"]
            if not original.is_file():
                raise FileNotFoundError(original)
            item["inspected_upstream_sha256"] = digest(original)
            item["same_ast_as_inspected_upstream"] = ast.dump(ast.parse(path.read_text(encoding="utf-8"))) == ast.dump(ast.parse(original.read_text(encoding="utf-8")))
        hashes.append(item)
    dump(root / "docs/vendor_source_hashes.json", {"upstream_repository": "amazon-science/patchcore-inspection", "upstream_commit": None, "scope": "Exact vendored source; optional SHA/AST comparison with the previously inspected upstream files", "files": hashes})
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if (rel.parts[0] in EXCLUDED_ROOTS or rel.parts[0].startswith(".pytest-runtime")
                or rel.parts[:2] == ("legacy", "archive")
                or any(part == "__pycache__" or part.endswith(".egg-info") for part in rel.parts)):
            continue
        if rel.as_posix() == "manifest.json" or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError("Large file unexpectedly included in public release: " + str(rel))
        entries.append({"path": rel.as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {"schema_version": 1, "scope": "Public source and curated numeric records only; excludes data, weights, frozen features and run artifacts", "files": entries, "file_count": len(entries), "total_bytes": sum(i["bytes"] for i in entries)}
    dump(root / "manifest.json", manifest)
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--upstream-root", type=Path, help="Optional local, already-inspected upstream checkout; no download")
    args = parser.parse_args()
    result = build(args.root, args.upstream_root)
    print("Manifest:", result["file_count"], "files;", result["total_bytes"], "bytes")
