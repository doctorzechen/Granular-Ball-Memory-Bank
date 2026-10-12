"""Offline tests of PatchCore integration and shared experimental cache I/O."""
from pathlib import Path
import sys
from types import SimpleNamespace

import faiss
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "patchcore" / "src"))

from adapters.patchcore import _TrainingCapture, _score_cache
from experiments.cache_io import _cache_test, _load_test_cache, _save_json
from experiments.main_comparison import run_banks
from patchcore.common import RescaleSegmentor
from patchcore.patchcore import PatchMaker


@pytest.fixture(autouse=True)
def cpu_threads():
    old_torch, old_faiss = torch.get_num_threads(), faiss.omp_get_max_threads()
    torch.set_num_threads(2)
    faiss.omp_set_num_threads(2)
    yield
    torch.set_num_threads(old_torch)
    faiss.omp_set_num_threads(old_faiss)


def test_same_scorer_returns_squared_l2_image_max_and_official_maps():
    model = SimpleNamespace(
        patch_maker=PatchMaker(3, stride=1),
        anomaly_segmentor=RescaleSegmentor(torch.device("cpu"), (4, 4)),
    )
    bank = np.array([[0, 0], [2, 2]], dtype=np.float32)
    features = np.vstack([
        np.zeros((4, 2), dtype=np.float32),
        np.full((4, 2), 5, dtype=np.float32),
    ])
    masks = np.zeros((2, 1, 4, 4), dtype=np.float32)
    masks[1, :, :2] = 1
    cached = {
        "batches": [{"features": features, "batch_size": 2, "patch_shape": [2, 2]}],
        "labels": np.array([0, 1], dtype=np.int32),
        "masks_gt": list(masks),
        "report": {"feature_extraction_seconds": 0.0},
    }
    result = _score_cache(model, bank, cached, torch.device("cpu"), 2, True)
    np.testing.assert_array_equal(result["scores"], [0, 18])
    np.testing.assert_allclose(result["maps"][0], 0, atol=1e-6)
    np.testing.assert_allclose(result["maps"][1], 18, atol=1e-5)
    assert result["metrics"]["instance_auroc"] == 1.0
    assert result["metrics"]["anomaly_pixel_auroc"] == 0.5
    assert result["faiss_validation"]["pass"] is True


def test_feature_cache_roundtrip_preserves_batch_order_labels_and_masks(tmp_path):
    features = np.arange(2 * 784 * 3, dtype=np.float32).reshape(2 * 784, 3)
    masks = np.zeros((2, 1, 224, 224), dtype=np.float32)
    masks[1, :, :112] = 1
    cache = {
        "batches": [
            {"features": features[:784], "batch_size": 1, "patch_shape": [28, 28]},
            {"features": features[784:], "batch_size": 1, "patch_shape": [28, 28]},
        ],
        "labels": np.array([0, 1], dtype=np.int32),
        "masks_gt": list(masks),
        "paths": [r"C:\old\good\000.png", r"C:\old\defect\001.png"],
        "subtypes": ["good", "defect"],
        "report": {"feature_extraction_seconds": 0.0},
    }
    _cache_test(tmp_path, cache, "fixed-config-hash")
    loaded = _load_test_cache(tmp_path)
    np.testing.assert_array_equal(
        np.concatenate([batch["features"] for batch in loaded["batches"]]), features
    )
    np.testing.assert_array_equal(loaded["labels"], cache["labels"])
    np.testing.assert_array_equal(loaded["masks_gt"], masks)
    assert loaded["paths"] == cache["paths"]
    assert loaded["subtypes"] == ["good", "defect"]
    assert [batch["batch_size"] for batch in loaded["batches"]] == [1, 1]


def test_training_capture_retains_raw_features_without_changing_baseline_bank():
    raw = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
    bank = raw[[0, 2]]
    capture = _TrainingCapture()
    result = capture(raw, bank, np.array([0, 2]))
    assert capture.raw_features is raw
    assert result["memory_bank"] is bank
    assert result["selected_indices_available"] is True


def test_same_capacity_banks_use_centers_or_real_training_rows(tmp_path):
    training = tmp_path / "training" / "synthetic" / "cache"
    training.mkdir(parents=True)
    raw = np.zeros((9, 1024), dtype=np.float32)
    raw[:, 0] = [0, 1, 2, 10, 11, 12, 20, 21, 22]
    assignments = np.repeat(np.arange(3), 3).astype(np.int32)
    centers = raw[[1, 4, 7]].copy()
    np.save(training / "training_embeddings.npy", raw)
    np.save(training / "original_memory_bank.npy", raw[[0, 3, 6]])
    np.save(training / "selected_indices.npy", [0, 3, 6])
    np.savez(training / "gb_partition.npz", assignments=assignments,
             centers=centers, sizes=np.array([3, 3, 3]))
    _save_json(training / "training_cache.json", {
        "number_of_balls": 3,
        "training_embedding_shape": [9, 1024],
    })
    output = tmp_path / "compared"
    result = run_banks(tmp_path / "training", output, "synthetic",
                       torch.device("cpu"), 2)
    assert result["k_gb"] == 3
    cached = output / "synthetic" / "cache"
    np.testing.assert_array_equal(np.load(cached / "gb_center_bank.npy"), centers)
    np.testing.assert_array_equal(np.load(cached / "gb_medoid_indices.npy"), [1, 4, 7])
    order = np.load(cached / "ff_raw_order.npy")
    assert len(np.unique(order)) == 3
    np.testing.assert_array_equal(np.load(cached / "ff_raw_bank.npy"), raw[order])
    # Reusing the same protocol must reuse the complete bank checkpoint.
    repeated = run_banks(tmp_path / "training", output, "synthetic",
                         torch.device("cpu"), 2)
    assert repeated == result
