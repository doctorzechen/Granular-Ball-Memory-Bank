"""Missing generated arrays must not pass a resume via JSON markers alone."""
import numpy as np
import json
import pytest
from experiments.main_comparison import _validate_generated_banks, _validate_scored_checkpoint
from experiments.result_io import validate_existing_result
from experiments.capacity_study import _validate_capacity_order

def write_banks(root):
    for name in ("gb_center_bank.npy", "gb_medoid_bank.npy", "ff_raw_bank.npy"):
        np.save(root / name, np.zeros((2, 1024), np.float32))
    for name in ("gb_medoid_indices.npy", "ff_raw_order.npy"):
        np.save(root / name, np.array([0, 1], np.int64))
    return {"k_gb": 2, "raw_patch_count": 3}

def test_valid_bank_checkpoint_and_missing_array(tmp_path):
    manifest = write_banks(tmp_path)
    _validate_generated_banks(tmp_path, manifest)
    (tmp_path / "gb_center_bank.npy").unlink()
    with pytest.raises(RuntimeError, match="Incomplete bank checkpoint"):
        _validate_generated_banks(tmp_path, manifest)

def test_invalid_bank_shape_and_duplicate_indices(tmp_path):
    manifest = write_banks(tmp_path)
    np.save(tmp_path / "ff_raw_bank.npy", np.zeros((1, 1024), np.float32))
    with pytest.raises(RuntimeError, match="Invalid bank checkpoint"):
        _validate_generated_banks(tmp_path, manifest)
    manifest = write_banks(tmp_path)
    np.save(tmp_path / "ff_raw_order.npy", np.array([0, 0], np.int64))
    with pytest.raises(RuntimeError, match="Invalid bank indices"):
        _validate_generated_banks(tmp_path, manifest)

def test_detection_checkpoint_requires_complete_finite_scores(tmp_path):
    with pytest.raises(RuntimeError, match="Incomplete detection checkpoint"):
        _validate_scored_checkpoint(tmp_path, 2)
    np.save(tmp_path / "image_scores.npy", [0.0, 1.0])
    _validate_scored_checkpoint(tmp_path, 2)
    np.save(tmp_path / "image_scores.npy", [np.nan, 1.0])
    with pytest.raises(RuntimeError, match="Invalid detection checkpoint"):
        _validate_scored_checkpoint(tmp_path, 2)

def test_capacity_checkpoint_requires_scores_and_saved_gb_bank(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"status": "complete", "method": "gb_center_x2", "k": 2, "metrics": {}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Incomplete result checkpoint"):
        validate_existing_result(tmp_path, 2, "gb_center_x2", "bank.npy")
    np.save(tmp_path / "image_scores.npy", [0., 1.])
    with pytest.raises(RuntimeError, match="Incomplete capacity bank checkpoint"):
        validate_existing_result(tmp_path, 2, "gb_center_x2", "bank.npy")
    np.save(tmp_path / "bank.npy", np.zeros((2, 1024), np.float32))
    validate_existing_result(tmp_path, 2, "gb_center_x2", "bank.npy")

@pytest.mark.parametrize("order", [np.array([0, 0]), np.array([0., 1.]), np.array([0, 3]), np.array([[0, 1]]), np.array([0])])
def test_capacity_order_rejects_invalid_indices(order):
    with pytest.raises(RuntimeError, match="Invalid capacity FF order"):
        _validate_capacity_order(order, 2, 3)

def test_capacity_order_accepts_distinct_original_coreset_indices():
    _validate_capacity_order(np.array([2, 0], np.int64), 2, 3)
