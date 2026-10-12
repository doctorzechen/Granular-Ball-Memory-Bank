"""Same-capacity GB center, medoid, and real-patch farthest-first experiments.

This module materializes memory banks, scores the identical frozen queries,
and times their construction. Historical stress/reporting stages are omitted.
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from experiments import construction_time as con
from gb_memory.partition import build_granular_balls
from baselines.farthest_first import global_farthest_first_order
from adapters.patchcore import _build_original_model, _score_cache
from experiments.cache_io import (
    _load_test_cache, _save_json, config_hash, _load_raw, load_train_assets,
)

LOGGER = logging.getLogger(__name__)
FORMAT_VERSION = 1
METHODS = ("gb_center", "gb_medoid", "ff_raw")
FF_PROJECTION_DIMENSION = 1024
TIME_BUDGET_SECONDS = 200.0

def stage8c_config(category, device, faiss_workers=8):
    return {
        "format_version": FORMAT_VERSION,
        "stage": "8C",
        "category": category,
        "seed": con.SEED,
        "min_ball_size": con.MIN_BALL_SIZE,
        "gb_threshold": con.GB_THRESHOLD,
        "timing_warmup": con.TIMING_WARMUP,
        "timing_repeats": con.TIMING_REPEATS,
        "ff_projection_dimension": FF_PROJECTION_DIMENSION,
        "ff_selection_space": "exact_raw",
        "stress_category": con.STRESS_CATEGORY,
        "stress_scales": list(con.STRESS_SCALES),
        "partition_source": "stage5b_frozen_global_gb_partition (banks) / "
                            "fresh seed-0 rebuild (timing, stress)",
        "device": str(device),
        "faiss_workers": int(faiss_workers),
    }


def _rss_bytes():
    return int(psutil.Process().memory_info().rss)


def _validate_generated_banks(cache_dir, manifest):
    """Do not treat a JSON marker alone as a complete checkpoint."""
    k = int(manifest["k_gb"])
    raw_count = int(manifest["raw_patch_count"])
    for filename in ("gb_center_bank.npy", "gb_medoid_bank.npy", "ff_raw_bank.npy"):
        path = Path(cache_dir) / filename
        if not path.is_file():
            raise RuntimeError("Incomplete bank checkpoint; missing " + str(path) + "; use a new output directory.")
        bank = np.load(path, mmap_mode="r", allow_pickle=False)
        if bank.shape != (k, 1024) or bank.dtype != np.float32 or not np.isfinite(bank).all():
            raise RuntimeError("Invalid bank checkpoint: " + str(path))
    for filename in ("gb_medoid_indices.npy", "ff_raw_order.npy"):
        path = Path(cache_dir) / filename
        if not path.is_file():
            raise RuntimeError("Incomplete bank checkpoint; missing " + str(path))
        indices = np.load(path, allow_pickle=False)
        if indices.shape != (k,) or not np.issubdtype(indices.dtype, np.integer) or len(np.unique(indices)) != k or indices.min() < 0 or indices.max() >= raw_count:
            raise RuntimeError("Invalid bank indices: " + str(path))


def _validate_scored_checkpoint(method_dir, expected_images):
    path = Path(method_dir) / "image_scores.npy"
    if not path.is_file():
        raise RuntimeError("Incomplete detection checkpoint; missing " + str(path))
    scores = np.load(path, allow_pickle=False)
    if scores.shape != (expected_images,) or not np.isfinite(scores).all():
        raise RuntimeError("Invalid detection checkpoint: " + str(path))


def run_banks(stage5b_root, out_root, category, device, faiss_workers=8):
    """Materialize the three same-K banks from the frozen Stage 5B cache."""
    stage5b_root, out_root = Path(stage5b_root), Path(out_root)
    config = stage8c_config(category, device, faiss_workers)
    base_hash = config_hash(config)
    cache_dir = out_root / category / "cache"
    manifest_path = cache_dir / "banks_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (existing.get("config_hash") == base_hash
                and existing.get("status") == "complete"):
            _validate_generated_banks(cache_dir, existing)
            LOGGER.info("[%s] banks checkpoint reused.", category)
            return existing

    assets = load_train_assets(stage5b_root, category)
    raw = _load_raw(assets["training_embeddings_path"])
    assignments = np.asarray(assets["assignments"], dtype=np.int64)
    centers = np.asarray(assets["gb_centers"], dtype=np.float32)
    k_gb = int(assets["number_of_balls"])
    if len(centers) != k_gb:
        raise RuntimeError("{}: cached centers != K_GB.".format(category))

    cache_dir.mkdir(parents=True, exist_ok=True)

    # GB-Center bank: the frozen ball centers.
    center_bank = np.array(centers, dtype=np.float32, order="C", copy=True)
    np.save(cache_dir / "gb_center_bank.npy", center_bank)

    # GB-Medoid bank: per-ball medoid of the raw patches.
    medoids = con.medoid_indices(raw, assignments, centers, device)
    medoid_bank = np.array(raw[medoids], dtype=np.float32, order="C",
                           copy=True)
    np.save(cache_dir / "gb_medoid_indices.npy", medoids)
    np.save(cache_dir / "gb_medoid_bank.npy", medoid_bank)

    # FF bank: exact raw-space farthest-first at K = K_GB.
    ff_order, ff_report = global_farthest_first_order(
        raw, k_gb, FF_PROJECTION_DIMENSION, con.SEED, device)
    ff_bank = np.array(raw[ff_order], dtype=np.float32, order="C", copy=True)
    np.save(cache_dir / "ff_raw_order.npy", ff_order)
    np.save(cache_dir / "ff_raw_bank.npy", ff_bank)
    _save_json(cache_dir / "ff_raw_report.json", ff_report)

    audits = {
        "k_gb_equals_training_cache": k_gb == int(
            assets["training_cache"]["number_of_balls"]),
        "center_bank_rows": int(len(center_bank)),
        "centers_finite": bool(np.all(np.isfinite(center_bank))),
        "medoid_count_equals_k_gb": int(len(medoids)) == k_gb,
        "medoid_unique": bool(len(np.unique(medoids)) == len(medoids)),
        "medoid_range_valid": bool(
            medoids.min() >= 0 and medoids.max() < len(raw)),
        "medoid_bank_equals_source_rows": bool(
            np.array_equal(medoid_bank, raw[medoids])),
        "ff_count_equals_k_gb": int(len(ff_order)) == k_gb,
        "ff_unique": bool(len(np.unique(ff_order)) == len(ff_order)),
        "ff_range_valid": bool(
            ff_order.min() >= 0 and ff_order.max() < len(raw)),
        "ff_bank_equals_source_rows": bool(np.array_equal(ff_bank,
                                                          raw[ff_order])),
        "ff_selection_space": ff_report["selection_space"],
    }
    manifest = {
        "status": "complete",
        "config_hash": base_hash,
        "config": config,
        "category": category,
        "raw_patch_count": int(len(raw)),
        "k_gb": k_gb,
        "audits": audits,
    }
    if not all(v for k, v in audits.items() if k != "ff_selection_space"):
        raise RuntimeError("{}: bank contract audit failed: {}".format(
            category, audits))
    _save_json(manifest_path, manifest)
    return manifest


def _construct_once(method, raw, k_gb, device):
    """One full construction; returns (bank, stages, audit)."""
    stages = {}
    peak_gpu = 0
    peak_rss = _rss_bytes()
    audit = {}

    if method in ("gb_center", "gb_medoid"):
        with con.ConstructionTimer(device) as timer:
            result = build_granular_balls(
                raw, con.MIN_BALL_SIZE, con.GB_THRESHOLD, seed=con.SEED,
                device=device)
        stages["partition_seconds"] = float(timer.seconds)
        peak_gpu = max(peak_gpu, timer.peak_gpu_allocated_bytes)
        peak_rss = max(peak_rss, _rss_bytes())
        audit["final_ball_count"] = int(len(result.centers))

        if method == "gb_medoid":
            with con.ConstructionTimer(device) as timer:
                medoids = con.medoid_indices(
                    raw, result.assignments, result.centers, device)
            stages["representative_seconds"] = float(timer.seconds)
            peak_gpu = max(peak_gpu, timer.peak_gpu_allocated_bytes)
            peak_rss = max(peak_rss, _rss_bytes())
            internal = np.asarray(result.representative_indices,
                                  dtype=np.int64)
            audit["medoid_matches_internal_fraction"] = float(
                np.mean(medoids == internal)) if len(internal) == len(
                medoids) else None
            with con.ConstructionTimer(device) as timer:
                bank = np.array(raw[medoids], dtype=np.float32, order="C",
                                copy=True)
            stages["assembly_seconds"] = float(timer.seconds)
            audit["bank_rows"] = int(len(bank))
        else:
            stages["representative_seconds"] = 0.0
            with con.ConstructionTimer(device) as timer:
                bank = np.array(result.centers, dtype=np.float32, order="C",
                                copy=True)
            stages["assembly_seconds"] = float(timer.seconds)
            audit["bank_rows"] = int(len(bank))
        peak_gpu = max(peak_gpu, timer.peak_gpu_allocated_bytes)

    elif method == "ff_raw":
        stages["partition_seconds"] = 0.0
        with con.ConstructionTimer(device) as timer:
            order, _ = global_farthest_first_order(
                raw, k_gb, FF_PROJECTION_DIMENSION, con.SEED, device)
        stages["selection_seconds"] = float(timer.seconds)
        peak_gpu = max(peak_gpu, timer.peak_gpu_allocated_bytes)
        peak_rss = max(peak_rss, _rss_bytes())
        with con.ConstructionTimer(device) as timer:
            bank = np.array(raw[order], dtype=np.float32, order="C",
                            copy=True)
        stages["assembly_seconds"] = float(timer.seconds)
        peak_gpu = max(peak_gpu, timer.peak_gpu_allocated_bytes)
        audit["selected_count"] = int(len(order))
        audit["bank_rows"] = int(len(bank))
    else:
        raise ValueError("Unknown method: " + method)

    peak_rss = max(peak_rss, _rss_bytes())
    total = float(sum(stages.values()))
    record = {
        "stages": stages,
        "total_seconds": total,
        "peak_gpu_allocated_bytes": int(peak_gpu),
        "peak_rss_bytes": int(peak_rss),
        "audit": audit,
    }
    return bank, record


def _timing_path(out_category, method):
    return out_category / "timings_{}.json".format(method)


def _timing_done(payload):
    return (payload.get("status") == "complete"
            and len(payload.get("repeats", [])) >= con.TIMING_REPEATS)


def run_time(stage5b_root, out_root, category, method, device,
             faiss_workers=8):
    """Warm-up + TIMING_REPEATS timed constructions, checkpointed per unit.

    Each invocation works until the repeats are complete or the time budget
    is exhausted; re-invoking resumes from the checkpoint."""
    stage5b_root, out_root = Path(stage5b_root), Path(out_root)
    if method not in METHODS:
        raise ValueError("Unknown method: " + method)
    config = stage8c_config(category, device, faiss_workers)
    base_hash = config_hash({"base": config_hash(config), "method": method,
                             "phase": "time"})
    out_category = out_root / category
    path = _timing_path(out_category, method)
    payload = None
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("config_hash") == base_hash:
            payload = existing
            if _timing_done(payload):
                LOGGER.info("[%s/%s] timing checkpoint reused.", category,
                            method)
                return payload
    if payload is None:
        payload = {"status": "in_progress", "config_hash": base_hash,
                   "config": config, "category": category, "method": method,
                   "warmup": None, "repeats": []}

    assets = load_train_assets(stage5b_root, category)
    raw = _load_raw(assets["training_embeddings_path"])
    k_gb = int(assets["number_of_balls"])

    started = time.perf_counter()
    while True:
        if payload["warmup"] is None:
            _, record = _construct_once(method, raw, k_gb, device)
            payload["warmup"] = record
            _save_json(path, payload)
            LOGGER.info("[%s/%s] warmup %.2fs", category, method,
                        record["total_seconds"])
        elif len(payload["repeats"]) < con.TIMING_REPEATS:
            _, record = _construct_once(method, raw, k_gb, device)
            payload["repeats"].append(record)
            _save_json(path, payload)
            LOGGER.info("[%s/%s] repeat %d/%d %.2fs", category, method,
                        len(payload["repeats"]), con.TIMING_REPEATS,
                        record["total_seconds"])
        else:
            break
        if time.perf_counter() - started > TIME_BUDGET_SECONDS:
            LOGGER.info("[%s/%s] time budget reached; resume later.",
                        category, method)
            return payload

    totals = [r["total_seconds"] for r in payload["repeats"]]
    payload["summary"] = con.summarize_repeats(totals)
    payload["stage_summaries"] = {
        stage: con.summarize_repeats([r["stages"][stage]
                                      for r in payload["repeats"]])
        for stage in payload["repeats"][0]["stages"]}
    payload["k_gb"] = k_gb
    payload["raw_patch_count"] = int(len(raw))
    payload["peak_gpu_allocated_bytes_max"] = int(max(
        [payload["warmup"]["peak_gpu_allocated_bytes"]]
        + [r["peak_gpu_allocated_bytes"] for r in payload["repeats"]]))
    payload["peak_rss_bytes_max"] = int(max(
        [payload["warmup"]["peak_rss_bytes"]]
        + [r["peak_rss_bytes"] for r in payload["repeats"]]))
    if method in ("gb_center", "gb_medoid"):
        counts = ([payload["warmup"]["audit"]["final_ball_count"]]
                  + [r["audit"]["final_ball_count"]
                     for r in payload["repeats"]])
        # Fresh rebuilds are NOT bit-deterministic (faiss CPU coarse KMeans
        # varies run-to-run); record the observed spread instead of
        # asserting equality with the frozen Stage 5B partition.
        payload["audit_final_ball_counts"] = [int(c) for c in counts]
        payload["audit_k_gb_matches_cache"] = bool(
            all(c == k_gb for c in counts))
        payload["audit_k_gb_spread_percent"] = float(
            100.0 * (max(counts) - min(counts)) / k_gb)
    payload["status"] = "complete"
    _save_json(path, payload)
    return payload


def run_detect(stage5b_root, out_root, category, device, faiss_workers=8):
    """Score the three same-K banks with the frozen detection pipeline."""
    stage5b_root, out_root = Path(stage5b_root), Path(out_root)
    config = stage8c_config(category, device, faiss_workers)
    base_hash = config_hash(config)
    out_category = out_root / category
    cache_dir = out_category / "cache"
    manifest = json.loads(
        (cache_dir / "banks_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("config_hash") != base_hash:
        raise RuntimeError("banks manifest hash mismatch for " + category)
    _validate_generated_banks(cache_dir, manifest)
    k_gb = int(manifest["k_gb"])

    assets = load_train_assets(stage5b_root, category)
    raw = _load_raw(assets["training_embeddings_path"])
    model = _build_original_model(device, faiss_workers)
    test_cache = _load_test_cache(Path(stage5b_root) / category / "cache")

    bank_files = {"gb_center": "gb_center_bank.npy",
                  "gb_medoid": "gb_medoid_bank.npy",
                  "ff_raw": "ff_raw_bank.npy"}
    index_files = {"gb_medoid": "gb_medoid_indices.npy",
                   "ff_raw": "ff_raw_order.npy"}
    summaries = {}
    for method in METHODS:
        method_hash = config_hash({"base": base_hash, "method": method,
                                   "phase": "detect"})
        method_dir = out_category / "runs" / method
        result_path = method_dir / "result.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if (existing.get("config_hash") == method_hash
                    and existing.get("status") == "complete"):
                _validate_scored_checkpoint(method_dir, len(test_cache["labels"]))
                LOGGER.info("[%s/%s] detect checkpoint reused.", category,
                            method)
                summaries[method] = existing
                continue
        bank = np.load(cache_dir / bank_files[method])
        if len(bank) != k_gb:
            raise RuntimeError("{}: {} bank size {} != K_GB {}.".format(
                category, method, len(bank), k_gb))
        scored = _score_cache(model, bank, test_cache, device,
                              faiss_workers, keep_maps=False)
        method_dir.mkdir(parents=True, exist_ok=True)
        np.save(method_dir / "image_scores.npy", scored["scores"])
        public = {key: value for key, value in scored.items()
                  if key not in ("scores", "maps")}
        audit = {
            "bank_rows_equals_k_gb": True,
            "bank_finite": bool(np.all(np.isfinite(bank))),
        }
        if method in index_files:
            indices = np.load(cache_dir / index_files[method])
            np.save(method_dir / "indices_training_embeddings.npy", indices)
            audit["indices_unique"] = bool(
                len(np.unique(indices)) == len(indices))
            audit["indices_range_valid"] = bool(
                indices.min() >= 0 and indices.max() < len(raw))
            audit["bank_equals_source_rows"] = bool(
                np.array_equal(bank, raw[indices]))
        else:
            audit["bank_equals_cached_centers"] = bool(np.array_equal(
                bank, np.asarray(assets["gb_centers"], dtype=np.float32)))
        public.update({
            "status": "complete", "config_hash": method_hash,
            "method": method, "k": k_gb,
            "memory_bank_size": int(len(bank)), "audits": audit,
        })
        _save_json(result_path, public)
        summaries[method] = public

    detection = {
        "status": "complete", "config_hash": base_hash, "k_gb": k_gb,
        "methods": {name: {
            "k": int(r["k"]),
            "memory_bank_size": int(r["memory_bank_size"]),
            "metrics": r["metrics"],
            "timing": r["timing"],
            "resources": r.get("resources", {}),
            "audits": r.get("audits", {}),
            "faiss_validation_pass": bool(
                r.get("faiss_validation", {}).get("pass", False)),
        } for name, r in summaries.items()},
    }
    _save_json(out_category / "stage8c_detection.json", detection)
    return detection
