"""Shared read/write helpers for training-only assets and frozen test features.

Directory and JSON keys retain the original experimental schema so archived
inputs and regenerated numerical results remain directly comparable.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
)

def canonical_config(category, batch_size, faiss_workers, device="cuda:0"):
    return {
        "format_version": FORMAT_VERSION,
        "stage": "5B",
        "category": category,
        "backbone": "wideresnet50",
        "layers": ["layer2", "layer3"],
        "input_size": 224,
        "resize": 256,
        "pretrain_embed_dimension": 1024,
        "target_embed_dimension": 1024,
        "patch_size": 3,
        "patch_stride": 1,
        "anomaly_scorer_num_nn": 1,
        "original_sampler": "ApproximateGreedyCoresetSampler",
        "original_coreset_percentage": 0.1,
        "patchcore_seed": 0,
        "gb_representation": "raw_1024d",
        "gb_seed": 0,
        "min_ball_size": 64,
        "retentions": [0.75, 0.50, 0.25],
        "selection": "within_ball_farthest_first",
        "random_seeds": [0, 1, 2, 3, 4],
        "batch_size": batch_size,
        "faiss_workers": faiss_workers,
        "faiss_backend": "CPU IndexFlatL2",
        "device": str(device),
    }


def config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True) + "\n", encoding="utf-8")


def _cache_test(path, cache, config_hash_value):
    path.mkdir(parents=True, exist_ok=True)
    features = np.concatenate([batch["features"] for batch in cache["batches"]])
    np.save(path / "test_embeddings.npy", features)
    np.save(path / "labels.npy", cache["labels"])
    np.save(path / "masks_gt.npy", np.asarray(cache["masks_gt"], dtype=np.float32))
    metadata = {
        "status": "complete", "config_hash": config_hash_value,
        "report": cache["report"], "paths": cache["paths"],
        "subtypes": cache["subtypes"], "patch_shape": [28, 28],
        "batch_sizes": [batch["batch_size"] for batch in cache["batches"]],
    }
    _save_json(path / "test_cache.json", metadata)


def _load_test_cache(path):
    metadata = json.loads((path / "test_cache.json").read_text(encoding="utf-8"))
    features = np.load(path / "test_embeddings.npy", mmap_mode="r")
    batches, offset = [], 0
    for batch_size in metadata["batch_sizes"]:
        count = int(batch_size) * 784
        batches.append({"features": features[offset:offset + count],
                        "batch_size": int(batch_size), "patch_shape": [28, 28]})
        offset += count
    return {
        "batches": batches, "labels": np.load(path / "labels.npy"),
        "masks_gt": list(np.load(path / "masks_gt.npy", mmap_mode="r")),
        "paths": metadata["paths"], "subtypes": metadata["subtypes"],
        "report": metadata["report"],
    }


def _stage5b_category_root(stage5b_root, category):
    root = Path(stage5b_root) / category
    if not (root / "cache" / "training_cache.json").is_file():
        raise FileNotFoundError(
            "Stage 5B cache missing for {}: {}".format(category, root))
    return root


def load_train_assets(stage5b_root, category):
    """Training-normal assets only. This function must never touch test
    cache files (task-book data-leakage rule for Stage 8A-1)."""
    root = _stage5b_category_root(stage5b_root, category) / "cache"
    training_cache = json.loads(
        (root / "training_cache.json").read_text(encoding="utf-8"))
    partition = np.load(root / "gb_partition.npz")
    return {
        "training_cache": training_cache,
        "training_embeddings_path": root / "training_embeddings.npy",
        "memory": np.load(root / "original_memory_bank.npy", mmap_mode="r"),
        "selected": np.load(root / "selected_indices.npy"),
        "assignments": partition["assignments"],
        "gb_centers": partition["centers"],
        "gb_sizes": partition["sizes"],
        "number_of_balls": int(training_cache["number_of_balls"]),
        "files_read": [
            "training_cache.json", "training_embeddings.npy",
            "original_memory_bank.npy", "selected_indices.npy",
            "gb_partition.npz",
        ],
    }


def _load_raw(path):
    return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
