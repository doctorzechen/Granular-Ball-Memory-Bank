"""Offline contract tests for the paper's reproduction package.

No test downloads a backbone or relies on the author's absolute paths.
Synthetic inputs test numerical behavior rather than recorded paper results.
"""

import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys

import faiss
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/patchcore/src"))
sys.path.insert(0, str(ROOT / "src"))

from gb_memory.partition import build_granular_balls
from experiments.hybrid_exploration import hybrid_bank
from experiments.retrieval import query


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    """Small offline tests should not launch dozens of CPU worker threads."""
    torch_threads = torch.get_num_threads()
    faiss_threads = faiss.omp_get_max_threads()
    torch.set_num_threads(2)
    faiss.omp_set_num_threads(2)
    yield
    torch.set_num_threads(torch_threads)
    faiss.omp_set_num_threads(faiss_threads)


def vectors(first_coordinates):
    output = np.zeros((len(first_coordinates), 1024), dtype=np.float32)
    output[:, 0] = first_coordinates
    return output


def test_exact_patch_scores_are_squared_l2_and_image_score_is_maximum():
    bank = np.zeros((2, 1024), dtype=np.float32)
    bank[1, :2] = (3, 4)
    patches = np.zeros((3, 1024), dtype=np.float32)
    patches[:, :2] = ((0, 1), (5, 4), (3, 4))
    expected = np.array((1, 4, 0), dtype=np.float32)
    scores = query(bank, patches)
    np.testing.assert_array_equal(scores, expected)
    assert scores.max() == 4
    # A square-rooted L2 implementation would incorrectly return 2 here.
    assert scores[1] != np.sqrt(expected[1])


def test_query_accepts_noncontiguous_feature_rows():
    bank = vectors((0, 3, 10))
    patches = vectors((1, 99, 5, 99, 11))[::2]
    assert not patches.flags.c_contiguous
    np.testing.assert_array_equal(query(bank, patches), (1, 4, 1))


def test_hybrid_preserves_capacity_and_rescues_farthest_training_patch():
    bank = vectors((0, 1, 10, 20))
    real_patches = vectors((0, 5, 15, 30))
    bank_before, real_before = bank.copy(), real_patches.copy()
    combined, replaced = hybrid_bank(bank, real_patches)
    # The stable redundancy ordering removes center 0 before center 1.
    # Patch 30 has the largest residual distance from centers 1, 10, 20.
    np.testing.assert_array_equal(combined, vectors((1, 10, 20, 30)))
    assert replaced == 1
    assert combined.shape == bank.shape
    assert combined.dtype == np.float32
    np.testing.assert_array_equal(bank, bank_before)
    np.testing.assert_array_equal(real_patches, real_before)


def test_hybrid_ties_are_stable_and_selection_has_training_only_inputs():
    bank = vectors((0, 1, 10, 20))
    real_patches = vectors((-9, 30, 10, 20))
    first, replaced = hybrid_bank(bank, real_patches)
    second, _ = hybrid_bank(bank, real_patches)
    # -9 and 30 are equally far from the retained gallery; input order wins.
    np.testing.assert_array_equal(first, vectors((1, 10, 20, -9)))
    np.testing.assert_array_equal(first, second)
    assert replaced == 1
    assert tuple(inspect.signature(hybrid_bank).parameters) == ("gb", "ff")


def test_granular_ball_partition_is_complete_disjoint_and_uses_exact_means():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(512, 32)).astype(np.float32)
    result = build_granular_balls(
        features,
        min_ball_size=8,
        seed=0,
        initial_ball_count=4,
        coarse_fit_sample_size=256,
        coarse_kmeans_iterations=3,
        assignment_batch_size=64,
        device=torch.device("cpu"),
    )
    members = np.concatenate(result.members)
    np.testing.assert_array_equal(np.sort(members), np.arange(len(features)))
    assert len(np.unique(members)) == len(features)
    assert np.all(result.assignments >= 0)
    assert result.sizes.sum() == len(features)
    assert result.configuration["full_n_by_n_matrix_created"] is False
    for ball_id, indices in enumerate(result.members):
        np.testing.assert_array_equal(result.assignments[indices], ball_id)
        np.testing.assert_array_equal(
            result.centers[ball_id],
            features[indices].mean(axis=0, dtype=np.float64).astype(np.float32),
        )
        assert result.sizes[ball_id] == len(indices)
        assert result.representative_indices[ball_id] in indices
    assert np.isfinite(result.centers).all()
    assert result.split_statistics["missing_assignments"] == 0
    assert result.split_statistics["duplicate_assignments"] == 0


def test_seeded_cpu_partition_is_repeatable():
    features = np.random.default_rng(13).normal(size=(128, 16)).astype(np.float32)
    options = dict(
        min_ball_size=8,
        seed=0,
        initial_ball_count=4,
        coarse_fit_sample_size=128,
        coarse_kmeans_iterations=2,
        assignment_batch_size=32,
        device=torch.device("cpu"),
    )
    first = build_granular_balls(features, **options)
    second = build_granular_balls(features, **options)
    np.testing.assert_array_equal(first.assignments, second.assignments)
    np.testing.assert_array_equal(first.centers, second.centers)


def test_offline_wr50_patch_embedding_has_784_rows_and_1024_dimensions():
    from torchvision import models
    from patchcore.patchcore import PatchCore

    # weights=None explicitly prevents any download or hidden cached-weight use.
    torch.manual_seed(0)
    backbone = models.wide_resnet50_2(weights=None)
    backbone.name, backbone.seed = "wideresnet50", 0
    model = PatchCore(torch.device("cpu"))
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=torch.device("cpu"),
        input_shape=[3, 224, 224],
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
    )
    with torch.inference_mode():
        embedding, patch_shapes = model._embed(
            torch.zeros((1, 3, 224, 224)), provide_patch_shapes=True
        )
    assert np.asarray(embedding).shape == (784, 1024)
    assert patch_shapes == [[28, 28], [14, 14]]
    assert np.isfinite(embedding).all()


def test_cli_help_works_from_a_different_working_directory(tmp_path):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "reproduction.py"), "--help"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage" in completed.stdout.lower()


def load_spade_module():
    path = ROOT / "experiments" / "spade_retrieval.py"
    spec = importlib.util.spec_from_file_location("spade_reproduction_portability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spade_image_score_uses_mean_of_five_euclidean_distances():
    module = load_spade_module()
    gallery = np.zeros((6, 2), dtype=np.float32)
    gallery[:, 0] = np.arange(6)
    image_feature = np.array([[2.5, 0]], dtype=np.float32)
    # Five exact Euclidean distances are 0.5, 0.5, 1.5, 1.5, 2.5.
    np.testing.assert_allclose(module.score(gallery, image_feature), [1.3], atol=1e-6)


def test_saved_windows_and_posix_paths_have_portable_basenames():
    module = load_spade_module()
    assert module.portable_basename(r"C:\old\datasets\screw\test\good\000.png") == "000.png"
    assert module.portable_basename("/old/datasets/screw/test/good/000.png") == "000.png"


def load_entry_module():
    spec = importlib.util.spec_from_file_location("portable_entry_test", ROOT / "reproduction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_guard_protects_inputs_and_their_descendants(tmp_path):
    module = load_entry_module()
    for path in (ROOT, ROOT / "frozen", ROOT / "data/mvtec/new_run",
                 ROOT / "reference/new_run", ROOT / "weights/new_run",
                 ROOT / "third_party/new_run", ROOT / "verification/new_run"):
        with pytest.raises(ValueError):
            module.guard_output(path)
    module.guard_output(ROOT / "runs/new_run")
    module.guard_output(ROOT.parent / "external_output")


def test_frozen_resume_accepts_regenerated_baseline_but_rejects_changed_cache(tmp_path, monkeypatch):
    import json
    module = load_entry_module()
    monkeypatch.setattr(module, "validate_cache", lambda root, category: None)
    source, output = tmp_path / "source", tmp_path / "output"
    category = module.RESULTS_REL / "IM224_WR50_STAGE5B" / "toothbrush"
    cache = source / category / "cache"
    baseline = source / category / "runs/original"
    cache.mkdir(parents=True)
    baseline.mkdir(parents=True)
    (cache / "training_cache.json").write_text('{"seed": 0}', encoding="utf-8")
    (baseline / "result.json").write_text(
        json.dumps({"status": "complete", "metrics": {}, "timing": 1}), encoding="utf-8")
    module.clone_inputs(source, output, ("toothbrush",))
    regenerated = output / category / "runs/original/result.json"
    regenerated.write_text(
        json.dumps({"status": "complete", "metrics": {}, "timing": 2}), encoding="utf-8")
    module.clone_inputs(source, output, ("toothbrush",))
    assert json.loads(regenerated.read_text(encoding="utf-8"))["timing"] == 2
    (output / category / "cache/training_cache.json").write_text('{"seed": 1}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="different inputs"):
        module.clone_inputs(source, output, ("toothbrush",))


def test_failed_reference_verification_returns_nonzero_exit(monkeypatch):
    from types import SimpleNamespace
    module = load_entry_module()
    monkeypatch.setitem(sys.modules, "verify_reference", SimpleNamespace(
        verify=lambda *args: {"passed": False, "checks_failed": 1}))
    monkeypatch.setattr(sys, "argv", ["reproduction.py", "verify-reference"])
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 1


def test_incremental_frozen_cloning_preserves_all_category_provenance(tmp_path, monkeypatch):
    import json
    module = load_entry_module()
    monkeypatch.setattr(module, "validate_cache", lambda root, category: None)
    source, output = tmp_path / "source", tmp_path / "output"
    for category in ("bottle", "toothbrush"):
        cache = source / module.RESULTS_REL / "IM224_WR50_STAGE5B" / category / "cache"
        cache.mkdir(parents=True)
        (cache / "training_cache.json").write_text('{"seed": 0}', encoding="utf-8")
        module.clone_inputs(source, output, (category,))
    recorded = json.loads((output / "frozen_input_provenance.json").read_text(encoding="utf-8"))
    assert recorded["categories"] == ["bottle", "toothbrush"]
