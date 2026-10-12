"""Controlled FAISS query-latency benchmark for frozen PatchCore features.

This module fixes the timing weakness in the original validation pipeline:
each bank used to be measured once, often in a different run.  Here every
bank is indexed in one process and evaluated repeatedly with a randomized,
position-balanced schedule.  The timed path matches
``NearestNeighbourScorer.predict``: concatenate the query feature list,
perform exact ``IndexFlatL2`` search, and average the returned 1-NN distance.

The benchmark intentionally measures cached-feature query latency only.  It
does not include backbone feature extraction or segmentation post-processing.
"""

import csv
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch  # Initialize the PyTorch DLL runtime before FAISS on Windows.
import faiss
import numpy as np
import psutil


LOGGER = logging.getLogger(__name__)

METHODS = ("original", "gb_center", "gb_medoid", "ff_raw")
SAME_CAPACITY_METHODS = ("gb_center", "gb_medoid", "ff_raw")
CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)
FORMAT_VERSION = 2


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return os.fspath(value)
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _array_sha256(array, rows_per_chunk=2048):
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    for start in range(0, len(value), rows_per_chunk):
        digest.update(
            np.ascontiguousarray(value[start : start + rows_per_chunk]).tobytes()
        )
    return digest.hexdigest()


def balanced_schedule(methods, repeats, seed):
    """Return randomized orders with balanced ordinal positions.

    Every complete block of ``len(methods)`` repetitions is a randomized
    Latin-square block, so each method appears exactly once in each ordinal
    position.  The benchmark defaults to a repeat count divisible by the
    number of methods.
    """

    methods = tuple(methods)
    if not methods:
        raise ValueError("At least one method is required.")
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    rng = np.random.default_rng(seed)
    schedule = []
    while len(schedule) < repeats:
        base = list(rng.permutation(methods))
        rotations = [base[offset:] + base[:offset] for offset in range(len(base))]
        rng.shuffle(rotations)
        schedule.extend(rotations)
    return [list(order) for order in schedule[:repeats]]


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("A nonempty one-dimensional vector is required.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Timing values must be finite and nonnegative.")
    mean = float(np.mean(values))
    std = float(np.std(values))
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {
        "n": int(len(values)),
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean else 0.0,
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "iqr": float(q75 - q25),
    }


def _bootstrap_median_ci(values, seed, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return {
        "median": float(np.median(values)),
        "ci95_low": float(np.percentile(medians, 2.5)),
        "ci95_high": float(np.percentile(medians, 97.5)),
        "bootstrap_samples": int(samples),
    }


def _load_query_batches(cache_dir):
    cache_dir = Path(cache_dir)
    metadata = json.loads(
        (cache_dir / "test_cache.json").read_text(encoding="utf-8")
    )
    features = np.load(cache_dir / "test_embeddings.npy", mmap_mode="r")
    batches = []
    offset = 0
    for batch_size in metadata["batch_sizes"]:
        count = int(batch_size) * 784
        batches.append(features[offset : offset + count])
        offset += count
    if offset != len(features):
        raise RuntimeError(
            "Cached batch sizes cover {} query embeddings, expected {}.".format(
                offset, len(features)
            )
        )
    return features, batches, metadata


def _load_banks(stage5b_root, stage8c_root, category):
    stage5b_root = Path(stage5b_root)
    stage8c_root = Path(stage8c_root)
    files = {
        "original": stage5b_root / category / "cache/original_memory_bank.npy",
        "gb_center": stage8c_root / category / "cache/gb_center_bank.npy",
        "gb_medoid": stage8c_root / category / "cache/gb_medoid_bank.npy",
        "ff_raw": stage8c_root / category / "cache/ff_raw_bank.npy",
    }
    banks = {}
    metadata = {}
    for method, path in files.items():
        bank = np.load(path, mmap_mode="r")
        if bank.ndim != 2 or bank.dtype != np.float32:
            raise RuntimeError(
                "{} bank must be a float32 matrix; got {} {}.".format(
                    method, bank.shape, bank.dtype
                )
            )
        banks[method] = np.ascontiguousarray(bank)
        metadata[method] = {
            "path": str(path.resolve()),
            "shape": list(bank.shape),
            "dtype": str(bank.dtype),
            "c_contiguous": bool(banks[method].flags.c_contiguous),
            "bytes": int(banks[method].nbytes),
            "sha256": _array_sha256(banks[method]),
        }
    dimensions = {bank.shape[1] for bank in banks.values()}
    if len(dimensions) != 1:
        raise RuntimeError("All banks must have the same feature dimension.")
    same_k = {len(banks[name]) for name in SAME_CAPACITY_METHODS}
    if len(same_k) != 1:
        raise RuntimeError("The three same-capacity banks do not have equal K.")
    return banks, metadata


def _prefault(array, chunk_rows=8192):
    """Read every array chunk once before benchmarking to avoid page faults."""

    checksum = 0.0
    for start in range(0, len(array), chunk_rows):
        chunk = np.asarray(array[start : start + chunk_rows])
        checksum += float(np.sum(chunk, dtype=np.float64))
    return checksum


def _build_indices(banks, workers):
    faiss.omp_set_num_threads(int(workers))
    indices = {}
    build = {}
    for method in METHODS:
        bank = banks[method]
        started = time.perf_counter_ns()
        index = faiss.IndexFlatL2(bank.shape[1])
        index.add(bank)
        elapsed = (time.perf_counter_ns() - started) / 1e9
        if int(index.ntotal) != len(bank):
            raise RuntimeError("{} FAISS index size mismatch.".format(method))
        indices[method] = index
        build[method] = {
            "seconds": float(elapsed),
            "ntotal": int(index.ntotal),
            "dimension": int(index.d),
        }
    return indices, build


def query_once(index, batches):
    """Measure one complete cached-test-set query pass.

    Component timers reproduce the three operations performed by
    ``NearestNeighbourScorer.predict``.  The returned pipeline time excludes
    loop bookkeeping but includes merge, exact FAISS search, and reduction.
    """

    merge_ns = 0
    search_ns = 0
    reduction_ns = 0
    checksum = 0.0
    index_checksum = 0
    query_count = 0
    wall_started = time.perf_counter_ns()
    for features in batches:
        started = time.perf_counter_ns()
        query = np.concatenate([features], axis=1)
        merge_ns += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        distances, neighbours = index.search(query, 1)
        search_ns += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        scores = np.mean(distances, axis=-1)
        checksum += float(np.sum(scores, dtype=np.float64))
        index_checksum += int(np.sum(neighbours, dtype=np.int64))
        reduction_ns += time.perf_counter_ns() - started
        query_count += len(query)
    wall_ns = time.perf_counter_ns() - wall_started
    pipeline_ns = merge_ns + search_ns + reduction_ns
    return {
        "query_count": int(query_count),
        "image_count": int(sum(len(batch) // 784 for batch in batches)),
        "merge_seconds": float(merge_ns / 1e9),
        "faiss_search_seconds": float(search_ns / 1e9),
        "reduction_seconds": float(reduction_ns / 1e9),
        "pipeline_seconds": float(pipeline_ns / 1e9),
        "wall_seconds": float(wall_ns / 1e9),
        "distance_checksum": float(checksum),
        "neighbour_index_checksum": int(index_checksum),
    }


def _add_normalized_timings(record):
    images = record["image_count"]
    queries = record["query_count"]
    record["faiss_search_ms_per_image"] = (
        1000.0 * record["faiss_search_seconds"] / images
    )
    record["pipeline_ms_per_image"] = (
        1000.0 * record["pipeline_seconds"] / images
    )
    record["wall_ms_per_image"] = 1000.0 * record["wall_seconds"] / images
    record["faiss_search_us_per_query"] = (
        1e6 * record["faiss_search_seconds"] / queries
    )
    record["pipeline_us_per_query"] = 1e6 * record["pipeline_seconds"] / queries
    return record


def _position_effect(runs, metric):
    output = {}
    for method in METHODS:
        positions = {}
        for position in range(1, len(METHODS) + 1):
            values = [
                row[metric]
                for row in runs
                if row["method"] == method and row["position"] == position
            ]
            positions[str(position)] = _summary(values) if values else None
        medians = [
            value["median"] for value in positions.values() if value is not None
        ]
        output[method] = {
            "positions": positions,
            "max_to_min_median_ratio": (
                float(max(medians) / min(medians)) if min(medians) else 1.0
            ),
        }
    return output


def analyze_runs(runs, bank_metadata, schedule_seed):
    metrics = (
        "faiss_search_ms_per_image",
        "pipeline_ms_per_image",
        "wall_ms_per_image",
    )
    summaries = {}
    for method in METHODS:
        method_rows = [row for row in runs if row["method"] == method]
        summaries[method] = {
            metric: _summary([row[metric] for row in method_rows])
            for metric in metrics
        }
        summaries[method]["stable_distance_checksum"] = bool(
            len({row["distance_checksum"] for row in method_rows}) == 1
        )
        summaries[method]["stable_neighbour_index_checksum"] = bool(
            len({row["neighbour_index_checksum"] for row in method_rows}) == 1
        )

    by_repeat = {}
    for row in runs:
        by_repeat.setdefault(row["repeat"], {})[row["method"]] = row
    speedups = {}
    for method in SAME_CAPACITY_METHODS:
        speedups[method] = {}
        for metric in ("faiss_search_ms_per_image", "pipeline_ms_per_image"):
            ratios = [
                values["original"][metric] / values[method][metric]
                for values in by_repeat.values()
            ]
            speedups[method][metric] = {
                **_summary(ratios),
                "median_ci95": _bootstrap_median_ci(
                    ratios, schedule_seed + 100 + METHODS.index(method)
                ),
            }

    same_k_spread = {}
    for metric in ("faiss_search_ms_per_image", "pipeline_ms_per_image"):
        medians = {
            method: summaries[method][metric]["median"]
            for method in SAME_CAPACITY_METHODS
        }
        same_k_spread[metric] = {
            "medians": medians,
            "max_to_min_ratio": float(max(medians.values()) / min(medians.values())),
        }

    position_effect = {
        metric: _position_effect(runs, metric)
        for metric in ("faiss_search_ms_per_image", "pipeline_ms_per_image")
    }
    same_k_sizes = {
        method: bank_metadata[method]["shape"][0]
        for method in SAME_CAPACITY_METHODS
    }
    checks = {
        "same_k_contract": len(set(same_k_sizes.values())) == 1,
        "stable_outputs": all(
            summaries[method]["stable_distance_checksum"]
            and summaries[method]["stable_neighbour_index_checksum"]
            for method in METHODS
        ),
        "same_k_faiss_median_spread_le_10pct": same_k_spread[
            "faiss_search_ms_per_image"
        ]["max_to_min_ratio"]
        <= 1.10,
        "same_k_pipeline_median_spread_le_10pct": same_k_spread[
            "pipeline_ms_per_image"
        ]["max_to_min_ratio"]
        <= 1.10,
        "per_method_faiss_cv_le_10pct": all(
            summaries[method]["faiss_search_ms_per_image"]["cv"] <= 0.10
            for method in METHODS
        ),
        "position_effect_le_10pct": all(
            position_effect["faiss_search_ms_per_image"][method][
                "max_to_min_median_ratio"
            ]
            <= 1.10
            for method in METHODS
        ),
    }
    checks["pass"] = bool(all(checks.values()))
    return {
        "summaries": summaries,
        "speedup_vs_original": speedups,
        "same_k_spread": same_k_spread,
        "position_effect": position_effect,
        "validation": checks,
    }


def _system_metadata(workers):
    process = psutil.Process()
    try:
        affinity = process.cpu_affinity()
    except (AttributeError, psutil.Error):
        affinity = None
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": sys.version,
        "numpy": np.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "process_cpu_affinity": affinity,
        "faiss_omp_threads": int(workers),
    }


def _write_runs_csv(path, runs):
    fields = (
        "repeat",
        "position",
        "method",
        "query_count",
        "image_count",
        "merge_seconds",
        "faiss_search_seconds",
        "reduction_seconds",
        "pipeline_seconds",
        "wall_seconds",
        "faiss_search_ms_per_image",
        "pipeline_ms_per_image",
        "wall_ms_per_image",
        "faiss_search_us_per_query",
        "pipeline_us_per_query",
        "distance_checksum",
        "neighbour_index_checksum",
    )
    with open(path, "w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: row[name] for name in fields} for row in runs)


def _report_markdown(result):
    analysis = result["analysis"]
    lines = [
        "# Controlled Query-Latency Benchmark",
        "",
        "Category: `{}`. Cached test features: {} images / {} patch queries. ".format(
            result["config"]["category"],
            result["query_data"]["image_count"],
            result["query_data"]["query_count"],
        )
        + "FAISS threads: {}. Measured repeats: {}.".format(
            result["config"]["faiss_workers"], result["config"]["repeats"]
        ),
        "",
        "All indices were built once in the same process. Method order was "
        "randomized in position-balanced blocks. Each timing includes the exact "
        "cached-feature query path: merge, `IndexFlatL2.search`, and 1-NN "
        "distance reduction. Feature extraction and segmentation are excluded.",
        "",
        "| Method | K | FAISS median ms/img | FAISS IQR | Pipeline median ms/img | CV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        search = analysis["summaries"][method]["faiss_search_ms_per_image"]
        pipeline = analysis["summaries"][method]["pipeline_ms_per_image"]
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.2%} |".format(
                method,
                result["banks"][method]["shape"][0],
                search["median"],
                search["iqr"],
                pipeline["median"],
                search["cv"],
            )
        )
    lines.extend(
        [
            "",
            "## Same-capacity sanity check",
            "",
            "FAISS median max/min ratio: `{:.4f}`. Pipeline median max/min ratio: "
            "`{:.4f}`.".format(
                analysis["same_k_spread"]["faiss_search_ms_per_image"][
                    "max_to_min_ratio"
                ],
                analysis["same_k_spread"]["pipeline_ms_per_image"][
                    "max_to_min_ratio"
                ],
            ),
            "",
            "Overall validation: **{}**.".format(
                "PASS" if analysis["validation"]["pass"] else "FAIL"
            ),
            "",
            "## Speed-up versus the 10% coreset baseline",
            "",
            "| Method | Median FAISS speed-up | 95% bootstrap CI | Median pipeline speed-up |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in SAME_CAPACITY_METHODS:
        faiss_speed = analysis["speedup_vs_original"][method][
            "faiss_search_ms_per_image"
        ]
        pipeline_speed = analysis["speedup_vs_original"][method][
            "pipeline_ms_per_image"
        ]
        ci = faiss_speed["median_ci95"]
        lines.append(
            "| {} | {:.3f}x | [{:.3f}, {:.3f}] | {:.3f}x |".format(
                method,
                faiss_speed["median"],
                ci["ci95_low"],
                ci["ci95_high"],
                pipeline_speed["median"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_benchmark(
    stage5b_root,
    stage8c_root,
    out_root,
    category="screw",
    repeats=32,
    warmup_rounds=10,
    warmup_batches=4,
    faiss_workers=8,
    schedule_seed=20260922,
):
    """Run and persist one controlled category benchmark."""

    if repeats < len(METHODS) or repeats % len(METHODS):
        raise ValueError(
            "repeats must be a positive multiple of {} for positional balance.".format(
                len(METHODS)
            )
        )
    if warmup_rounds < 1 or warmup_batches < 1:
        raise ValueError("Warm-up rounds and batches must be positive.")
    out_dir = Path(out_root) / category
    out_dir.mkdir(parents=True, exist_ok=True)

    test_cache_dir = Path(stage5b_root) / category / "cache"
    features, batches, test_metadata = _load_query_batches(test_cache_dir)
    if warmup_batches > len(batches):
        raise ValueError("warmup_batches exceeds the number of cached batches.")
    banks, bank_metadata = _load_banks(stage5b_root, stage8c_root, category)
    if features.dtype != np.float32 or features.shape[1] != banks["original"].shape[1]:
        raise RuntimeError("Test embeddings and banks have incompatible contracts.")

    LOGGER.info("Prefaulting cached queries and four banks.")
    prefault_checksums = {"test_embeddings": _prefault(features)}
    for method in METHODS:
        prefault_checksums[method] = _prefault(banks[method])

    indices, build = _build_indices(banks, faiss_workers)
    warmup_schedule = balanced_schedule(METHODS, warmup_rounds, schedule_seed - 1)
    LOGGER.info(
        "Warm-up: %d rounds x %d batches x %d methods.",
        warmup_rounds,
        warmup_batches,
        len(METHODS),
    )
    for order in warmup_schedule:
        for method in order:
            query_once(indices[method], batches[:warmup_batches])

    schedule = balanced_schedule(METHODS, repeats, schedule_seed)
    runs = []
    garbage_collection_enabled = None
    try:
        import gc

        garbage_collection_enabled = gc.isenabled()
        gc.disable()
        for repeat_index, order in enumerate(schedule, start=1):
            repeat_started = time.perf_counter()
            for position, method in enumerate(order, start=1):
                record = query_once(indices[method], batches)
                record.update(
                    {
                        "repeat": int(repeat_index),
                        "position": int(position),
                        "method": method,
                    }
                )
                runs.append(_add_normalized_timings(record))
            LOGGER.info(
                "Measured repeat %d/%d in %.2f s; order=%s",
                repeat_index,
                repeats,
                time.perf_counter() - repeat_started,
                ",".join(order),
            )
    finally:
        if garbage_collection_enabled:
            gc.enable()

    analysis = analyze_runs(runs, bank_metadata, schedule_seed)
    image_count = int(sum(len(batch) // 784 for batch in batches))
    result = {
        "status": "complete",
        "format_version": FORMAT_VERSION,
        "config": {
            "category": category,
            "stage5b_root": str(Path(stage5b_root).resolve()),
            "stage8c_root": str(Path(stage8c_root).resolve()),
            "repeats": int(repeats),
            "warmup_rounds": int(warmup_rounds),
            "warmup_batches_per_round": int(warmup_batches),
            "schedule_seed": int(schedule_seed),
            "faiss_workers": int(faiss_workers),
            "index": "CPU IndexFlatL2",
            "timed_scope": "cached-feature 1-NN query only",
            "schedule": schedule,
        },
        "system": _system_metadata(faiss_workers),
        "query_data": {
            "path": str((test_cache_dir / "test_embeddings.npy").resolve()),
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "batch_count": int(len(batches)),
            "batch_sizes": list(test_metadata["batch_sizes"]),
            "query_count": int(len(features)),
            "image_count": image_count,
        },
        "banks": bank_metadata,
        "index_build": build,
        "prefault_checksums": prefault_checksums,
        "runs": runs,
        "analysis": analysis,
    }
    _write_json(out_dir / "query_latency_benchmark.json", result)
    _write_runs_csv(out_dir / "query_latency_runs.csv", runs)
    (out_dir / "query_latency_report.md").write_text(
        _report_markdown(result) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "Benchmark complete: validation=%s; output=%s",
        analysis["validation"]["pass"],
        out_dir,
    )
    return result


def benchmark_checkpoint_matches(
    path,
    category,
    repeats,
    warmup_rounds,
    warmup_batches,
    faiss_workers,
    schedule_seed,
):
    """Return true only for a complete checkpoint with the requested protocol."""

    path = Path(path)
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    config = value.get("config", {})
    return bool(
        value.get("status") == "complete"
        and value.get("format_version") == FORMAT_VERSION
        and config.get("category") == category
        and config.get("repeats") == repeats
        and config.get("warmup_rounds") == warmup_rounds
        and config.get("warmup_batches_per_round") == warmup_batches
        and config.get("faiss_workers") == faiss_workers
        and config.get("schedule_seed") == schedule_seed
    )


def _aggregate_markdown(aggregate):
    headline = aggregate["headline"]
    lines = [
        "# All-Category Controlled Query-Latency Benchmark",
        "",
        "All {} MVTec AD categories were measured independently with the same "
        "single-process, position-balanced repeated protocol.".format(
            len(aggregate["categories"])
        ),
        "",
        "## Headline",
        "",
        "- New GB-center pipeline speed-up: mean **{:.3f}x**, median **{:.3f}x**, "
        "range **{:.3f}--{:.3f}x**.".format(
            headline["new_pipeline_speedup"]["mean"],
            headline["new_pipeline_speedup"]["median"],
            headline["new_pipeline_speedup"]["min"],
            headline["new_pipeline_speedup"]["max"],
        ),
        "- New pure-FAISS speed-up: mean **{:.3f}x**, median **{:.3f}x**.".format(
            headline["new_faiss_speedup"]["mean"],
            headline["new_faiss_speedup"]["median"],
        ),
        "- Previous single-pass speed-up: mean **{:.3f}x**, median **{:.3f}x**.".format(
            headline["old_single_pass_speedup"]["mean"],
            headline["old_single_pass_speedup"]["median"],
        ),
        "- Mean compression ratio: **{:.3f}x**.".format(
            headline["compression_ratio"]["mean"]
        ),
        "- Categories passing all timing sanity checks: **{}/{}**.".format(
            aggregate["validation"]["passing_categories"],
            len(aggregate["categories"]),
        ),
        "",
        "## Per-category comparison",
        "",
        "| Category | Compression | Old speed-up | New pipeline speed-up | "
        "New FAISS speed-up | Same-K spread | Pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in aggregate["rows"]:
        lines.append(
            "| {category} | {compression_ratio:.2f}x | "
            "{old_single_pass_speedup:.2f}x | {new_pipeline_speedup:.2f}x | "
            "{new_faiss_speedup:.2f}x | {same_k_pipeline_spread:.3f} | {passed} |".format(
                passed="yes" if row["validation_pass"] else "no", **row
            )
        )
    lines.extend(
        [
            "",
            "The new pipeline number covers cached-feature merge, exact "
            "`IndexFlatL2` search and 1-NN reduction. It is not end-to-end "
            "inference latency.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate_benchmarks(out_root, stage5b_root, stage8c_root, categories=CATEGORIES):
    """Assemble completed per-category benchmarks and compare old timings."""

    out_root = Path(out_root)
    stage5b_root = Path(stage5b_root)
    stage8c_root = Path(stage8c_root)
    rows = []
    category_results = {}
    for category in categories:
        result_path = out_root / category / "query_latency_benchmark.json"
        if not result_path.is_file():
            raise RuntimeError("Missing completed benchmark: {}".format(result_path))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise RuntimeError("Incomplete benchmark: {}".format(result_path))
        analysis = result["analysis"]
        old_original = json.loads(
            (
                stage5b_root / category / "runs/original/result.json"
            ).read_text(encoding="utf-8")
        )
        old_center = json.loads(
            (
                stage8c_root / category / "runs/gb_center/result.json"
            ).read_text(encoding="utf-8")
        )
        original_k = int(result["banks"]["original"]["shape"][0])
        center_k = int(result["banks"]["gb_center"]["shape"][0])
        row = {
            "category": category,
            "images": int(result["query_data"]["image_count"]),
            "patch_queries": int(result["query_data"]["query_count"]),
            "original_k": original_k,
            "gb_center_k": center_k,
            "compression_ratio": float(original_k / center_k),
            "old_original_ms_per_image": float(
                old_original["timing"]["query_ms_per_image"]
            ),
            "old_gb_center_ms_per_image": float(
                old_center["timing"]["query_ms_per_image"]
            ),
            "old_single_pass_speedup": float(
                old_original["timing"]["query_ms_per_image"]
                / old_center["timing"]["query_ms_per_image"]
            ),
            "new_original_pipeline_ms_per_image": float(
                analysis["summaries"]["original"]["pipeline_ms_per_image"][
                    "median"
                ]
            ),
            "new_gb_center_pipeline_ms_per_image": float(
                analysis["summaries"]["gb_center"]["pipeline_ms_per_image"][
                    "median"
                ]
            ),
            "new_original_faiss_ms_per_image": float(
                analysis["summaries"]["original"][
                    "faiss_search_ms_per_image"
                ]["median"]
            ),
            "new_gb_center_faiss_ms_per_image": float(
                analysis["summaries"]["gb_center"][
                    "faiss_search_ms_per_image"
                ]["median"]
            ),
            "new_pipeline_speedup": float(
                analysis["speedup_vs_original"]["gb_center"][
                    "pipeline_ms_per_image"
                ]["median"]
            ),
            "new_faiss_speedup": float(
                analysis["speedup_vs_original"]["gb_center"][
                    "faiss_search_ms_per_image"
                ]["median"]
            ),
            "same_k_pipeline_spread": float(
                analysis["same_k_spread"]["pipeline_ms_per_image"][
                    "max_to_min_ratio"
                ]
            ),
            "same_k_faiss_spread": float(
                analysis["same_k_spread"]["faiss_search_ms_per_image"][
                    "max_to_min_ratio"
                ]
            ),
            "validation_pass": bool(analysis["validation"]["pass"]),
        }
        rows.append(row)
        category_results[category] = {
            "validation": analysis["validation"],
            "speedup_vs_original": analysis["speedup_vs_original"],
            "same_k_spread": analysis["same_k_spread"],
        }

    def summarize_field(name):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summary = _summary(values)
        summary["geometric_mean"] = float(
            math.exp(np.mean(np.log(values)))
        )
        return summary

    total_old_original = sum(
        row["images"] * row["old_original_ms_per_image"] for row in rows
    )
    total_old_center = sum(
        row["images"] * row["old_gb_center_ms_per_image"] for row in rows
    )
    total_new_original = sum(
        row["images"] * row["new_original_pipeline_ms_per_image"] for row in rows
    )
    total_new_center = sum(
        row["images"] * row["new_gb_center_pipeline_ms_per_image"] for row in rows
    )
    aggregate = {
        "status": "complete",
        "format_version": FORMAT_VERSION,
        "categories": list(categories),
        "rows": rows,
        "headline": {
            "compression_ratio": summarize_field("compression_ratio"),
            "old_single_pass_speedup": summarize_field(
                "old_single_pass_speedup"
            ),
            "new_pipeline_speedup": summarize_field("new_pipeline_speedup"),
            "new_faiss_speedup": summarize_field("new_faiss_speedup"),
            "old_weighted_aggregate_speedup": float(
                total_old_original / total_old_center
            ),
            "new_weighted_aggregate_pipeline_speedup": float(
                total_new_original / total_new_center
            ),
        },
        "validation": {
            "passing_categories": int(
                sum(row["validation_pass"] for row in rows)
            ),
            "all_categories_pass": bool(
                all(row["validation_pass"] for row in rows)
            ),
            "failed_categories": [
                row["category"] for row in rows if not row["validation_pass"]
            ],
        },
        "category_results": category_results,
    }
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary_dir / "query_latency_all_categories.json", aggregate)
    fields = tuple(rows[0].keys())
    with open(
        summary_dir / "query_latency_all_categories.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (summary_dir / "query_latency_all_categories.md").write_text(
        _aggregate_markdown(aggregate) + "\n", encoding="utf-8"
    )
    return aggregate
