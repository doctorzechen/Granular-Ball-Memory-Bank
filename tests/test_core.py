"""Small CPU checks for the independent GB core and established controls."""

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.farthest_first import global_farthest_first_order
from baselines.medoid import medoid_indices
from gb_memory import GranularBallMemoryBank, build_granular_balls


def features():
    rng = np.random.default_rng(42)
    return np.vstack([rng.normal(center, .25, (24, 8)) for center in (-3., 0., 3.)]).astype(np.float32)


def settings():
    return dict(min_ball_size=4, seed=0, initial_ball_count=3,
                coarse_fit_sample_size=60, coarse_kmeans_iterations=8,
                two_means_iterations=2, assignment_batch_size=16,
                device="cpu")


def test_public_defaults_match_frozen_partition():
    core = inspect.signature(build_granular_balls).parameters
    facade = inspect.signature(GranularBallMemoryBank).parameters
    for name, parameter in facade.items():
        assert parameter.default == core[name].default


def test_partition_cover_and_public_fit_match_direct_algorithm():
    x = features()
    kwargs = settings()
    direct = build_granular_balls(x, **dict(kwargs, device=torch.device("cpu")))
    repeated = build_granular_balls(x, **dict(kwargs, device=torch.device("cpu")))
    assert np.array_equal(direct.assignments, repeated.assignments)
    assert np.array_equal(direct.centers, repeated.centers)
    assert np.array_equal(np.bincount(direct.assignments), direct.sizes)
    assert direct.sizes.sum() == len(x)
    assert np.array_equal(np.sort(np.concatenate(direct.members)), np.arange(len(x)))
    for center, indices in zip(direct.centers, direct.members):
        expected = np.mean(x[indices], axis=0, dtype=np.float64).astype(np.float32)
        assert np.array_equal(center, expected)
    bank = GranularBallMemoryBank(**kwargs)
    assert bank.fit(x) is bank
    assert np.array_equal(bank.centers, direct.centers)
    assert bank.configuration == direct.configuration
    assert bank.feature_dimension == 8 and bank.size == len(direct.centers)
    assert not hasattr(bank, "members") and not hasattr(bank, "assignments")
    arrays = [value for value in vars(bank).values() if isinstance(value, np.ndarray)]
    assert len(arrays) == 1 and arrays[0] is bank.centers
    assert not np.shares_memory(bank.centers, x)


def test_saved_inference_bank_has_only_centers_and_config(tmp_path):
    bank = GranularBallMemoryBank(**settings()).fit(features())
    saved = bank.save(tmp_path / "nested" / "bank.npz")
    with np.load(saved, allow_pickle=False) as archive:
        assert set(archive.files) == {"centers", "config"}
        metadata = json.loads(str(archive["config"].item()))
        assert metadata["center_count"] == bank.size
        assert metadata["embedding_dimension"] == bank.feature_dimension
        assert "members" not in metadata and "assignments" not in metadata
    restored = GranularBallMemoryBank.load(saved)
    assert np.array_equal(restored.centers, bank.centers)
    assert restored.configuration == bank.configuration
    assert restored.parameters == bank.parameters
    assert not list(saved.parent.glob("*.tmp"))


@pytest.mark.parametrize("value", [
    np.empty((0, 8)), np.empty((2, 0)), np.zeros(8), np.zeros((2, 3, 4)),
    np.full((2, 8), np.nan), np.full((2, 8), np.inf),
    np.full((2, 8), 1e100), np.ones((2, 8), dtype=np.complex64),
    np.array([["x", "y"]]), np.ones((2, 8), dtype=bool),
])
def test_invalid_embeddings_rejected(value):
    with pytest.raises(ValueError):
        GranularBallMemoryBank(device="cpu").fit(value)


@pytest.mark.parametrize("kwargs", [
    {"min_ball_size": 0}, {"min_ball_size": True}, {"min_ball_size": 2.5},
    {"seed": -1}, {"seed": True}, {"initial_ball_count": 0},
    {"coarse_fit_sample_size": 0}, {"coarse_kmeans_iterations": 0},
    {"two_means_iterations": 0}, {"assignment_batch_size": 0},
    {"center_consistency_threshold": 1},
    {"center_consistency_threshold": float("nan")},
    {"center_consistency_threshold": float("inf")},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        GranularBallMemoryBank(**kwargs)


def test_unfitted_bank_and_malformed_save_rejected(tmp_path):
    bank = GranularBallMemoryBank()
    with pytest.raises(RuntimeError):
        bank.save(tmp_path / "unfitted.npz")
    with pytest.raises(RuntimeError):
        _ = bank.centers
    malformed = tmp_path / "malformed.npz"
    np.savez(malformed, centers=np.ones((2, 4), np.float32), config=np.asarray("{}"))
    with pytest.raises(ValueError):
        GranularBallMemoryBank.load(malformed)


def test_core_import_has_no_patchcore_or_faiss_dependency():
    # Fresh interpreter makes this robust when other test modules import the
    # explicit PatchCore adapter during collection.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    program = (
        "import sys; from gb_memory import GranularBallMemoryBank; "
        "assert not any(name == 'patchcore' or name.startswith('patchcore.') "
        "for name in sys.modules); assert 'faiss' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", program], env=env, check=True)


def test_ff_raw_contract_matches_simple_numpy_traversal():
    x = np.array([[0., 0.], [1., 0.], [3., 0.], [8., 0.], [0., 1.]], dtype=np.float32)
    center = x.mean(axis=0)
    current = int(np.argmin(np.sum((x - center) ** 2, axis=1)))
    closest = np.full(len(x), np.inf, dtype=np.float32)
    chosen = np.zeros(len(x), dtype=bool)
    expected = []
    for _ in range(len(x)):
        expected.append(current)
        chosen[current] = True
        closest = np.minimum(closest, np.sum((x - x[current]) ** 2, axis=1))
        closest[chosen] = -1
        current = int(np.argmax(closest))
    order, report = global_farthest_first_order(x, len(x), 2, 0, torch.device("cpu"))
    assert np.array_equal(order, expected)
    assert len(np.unique(order)) == len(x)
    assert report["projection_dimension"] == 2
    assert report["full_memory_pairwise_matrix_created"] is False
    shorter, _ = global_farthest_first_order(x, 3, 2, 0, torch.device("cpu"))
    assert np.array_equal(shorter, order[:3])
    with pytest.raises(ValueError):
        global_farthest_first_order(x, 0, 2, 0, torch.device("cpu"))
    with pytest.raises(ValueError):
        global_farthest_first_order(x, len(x) + 1, 2, 0, torch.device("cpu"))


def test_ff_projected_repetition_and_nearest_center_representatives():
    x = features()
    first, _ = global_farthest_first_order(x, 8, 3, 7, torch.device("cpu"))
    second, _ = global_farthest_first_order(x, 8, 3, 7, torch.device("cpu"))
    assert np.array_equal(first, second)
    partition = build_granular_balls(x, **dict(settings(), device=torch.device("cpu")))
    medoids = medoid_indices(x, partition.assignments, partition.centers, torch.device("cpu"))
    assert len(medoids) == len(partition.centers)
    assert len(np.unique(medoids)) == len(medoids)
    assert np.array_equal(medoids, partition.representative_indices)
