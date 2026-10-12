"""Curated archival runs are evidence, unlike root-level generated runs."""
from build_release_manifest import build

def test_manifest_keeps_reference_runs_but_excludes_generated_runs(tmp_path):
    (tmp_path / "third_party/patchcore/src").mkdir(parents=True)
    evidence = tmp_path / "reference/patchcore-inspection/results/cat/runs/original/result.json"
    generated = tmp_path / "runs/generated/result.json"
    for path in (evidence, generated):
        path.parent.mkdir(parents=True)
        path.write_text('{"status": "complete"}', encoding="utf-8")
    manifest = build(tmp_path)
    paths = {entry["path"] for entry in manifest["files"]}
    assert evidence.relative_to(tmp_path).as_posix() in paths
    assert generated.relative_to(tmp_path).as_posix() not in paths
