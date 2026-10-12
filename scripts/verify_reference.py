#!/usr/bin/env python3
"""Audit the curated public numeric reference records without third-party packages.

This verifies internal numeric consistency of the archived experiments. It does
not read a manuscript, rerun experiments, or validate external assets.
--check-files additionally checks the current lightweight public manifest when
one exists; it never requires the original full-package manifest.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import statistics
import sys


CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
)
METHODS = ("original", "gb_center", "gb_medoid", "ff_raw")


def _read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _mean(values):
    return statistics.mean(values)


def _summary(values):
    return {
        "count": len(values), "mean": _mean(values),
        "median": statistics.median(values), "min": min(values), "max": max(values),
    }


class _Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, actual=None, expected=None):
        item = {"name": name, "passed": bool(passed)}
        if actual is not None:
            item["actual"] = actual
        if expected is not None:
            item["expected"] = expected
        self.checks.append(item)

    def equal(self, name, actual, expected):
        self.check(name, actual == expected, actual, expected)

    def close(self, name, actual, expected, tolerance=1e-10):
        self.check(name, math.isfinite(float(actual)) and abs(actual - expected) <= tolerance,
                   actual, {"value": expected, "absolute_tolerance": tolerance})

    def rounding(self, name, actual, expected, digits):
        rounded = format(actual, ".{}f".format(digits))
        self.equal(name, rounded, format(expected, ".{}f".format(digits)))


def _metrics(record):
    result = record["metrics"]
    for key in ("instance_auroc", "full_pixel_auroc", "anomaly_pixel_auroc"):
        if not 0 <= result[key] <= 1 or not math.isfinite(result[key]):
            raise ValueError("Invalid AUROC {}: {}".format(key, result[key]))
    return result


def _by_category(rows, field="category"):
    result = {row[field]: row for row in rows}
    if len(result) != len(rows) or set(result) != set(CATEGORIES):
        raise ValueError("Expected exactly one record for each of 15 categories")
    return result


def _verify_manifest(package_root, audit):
    manifest_path = package_root / "manifest.json"
    if not manifest_path.is_file():
        return {"checked": False, "reason": "No current public lightweight manifest is present"}
    manifest = _read_json(manifest_path)
    entries = manifest if isinstance(manifest, list) else manifest.get("files", manifest.get("entries"))
    if not isinstance(entries, list):
        raise ValueError("manifest.json must contain a files/entries list")
    seen = set()
    for entry in entries:
        relative = entry["path"]
        if Path(relative).is_absolute() or PureWindowsPath(relative).drive:
            raise ValueError("Public manifest path must be relative: " + relative)
        candidate = (package_root / relative).resolve()
        try:
            candidate.relative_to(package_root)
        except ValueError:
            raise ValueError("Manifest path escapes package: " + relative)
        audit.check("manifest_unique:" + relative, relative not in seen)
        seen.add(relative)
        audit.check("manifest_file_exists:" + relative, candidate.is_file())
        if not candidate.is_file():
            continue
        audit.equal("manifest_bytes:" + relative, candidate.stat().st_size, entry["bytes"])
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        audit.equal("manifest_sha256:" + relative, digest.hexdigest(), entry["sha256"])
    return {"checked": True, "scope": "Current lightweight public manifest only; external data, frozen features and weights are not verified here",
            "files_checked": len(entries), "total_bytes": sum(e["bytes"] for e in entries)}



def verify(package_root, output_path=None, check_files=False):
    """Return the audit report, and write it when ``output_path`` is supplied.

    Ordinary validation failures are collected into ``checks``. Missing or
    malformed input records raise an exception, so a missing archive cannot
    silently be reported as a passing reference verification.
    """
    package_root = Path(package_root).resolve()
    results = package_root / "reference" / "patchcore-inspection" / "results" / "MVTecAD_Results"
    evidence = package_root / "reference" / "revision_evidence"
    stage5b = results / "IM224_WR50_STAGE5B"
    stage8b = results / "IM224_WR50_STAGE8B"
    stage8c = results / "IM224_WR50_STAGE8C"
    query_root = results / "IM224_WR50_QUERYBENCH_V2"
    audit = _Audit()
    warnings = []
    rows = []
    consistency_values = []
    consistency_stops = 0
    matched_results = {}
    for category in CATEGORIES:
        training = _read_json(stage5b / category / "cache" / "training_cache.json")
        original = _read_json(stage5b / category / "runs" / "original" / "result.json")
        detected = {method: _read_json(stage8c / category / "runs" / method / "result.json")
                    for method in METHODS[1:]}
        matched_results[category] = detected
        baseline_metrics = _metrics(original)
        gb = _metrics(detected["gb_center"])
        ff = _metrics(detected["ff_raw"])
        medoid = _metrics(detected["gb_medoid"])
        k_gb = training["number_of_balls"]
        baseline_k = original["memory_bank_size"]
        audit.equal("training_rows:" + category, training["original_memory_bank_shape"][0], baseline_k)
        audit.equal("training_dimension:" + category, training["training_embedding_shape"][1], 1024)
        audit.check("training_selected_mapping:" + category, training["selected_mapping_exact"])
        for method, result in detected.items():
            audit.equal("matched_k:{}:{}".format(category, method), result["memory_bank_size"], k_gb)
            audit.check("faiss_vectors:{}:{}".format(category, method), result["faiss_validation"]["pass"])
        config = training["gb_configuration"]
        for key, value in {"seed": 0, "min_ball_size": 64,
                           "center_consistency_threshold": 1.3,
                           "coarse_kmeans_iterations": 8, "two_means_iterations": 2,
                           "assignment_batch_size": 4096}.items():
            audit.equal("gb_config:{}:{}".format(category, key), config[key], value)
        split = training["gb_split_statistics"]
        audit.equal("gb_missing_assignments:" + category, split["missing_assignments"], 0)
        audit.equal("gb_duplicate_assignments:" + category, split["duplicate_assignments"], 0)
        consistency_values.append(split["finite_log_consistency_min"])
        consistency_stops += sum(value for key, value in split["reason_counts"].items()
                                 if key.endswith("_consistent"))

        benchmark = _read_json(query_root / category / "query_latency_benchmark.json")
        audit.check("query_complete:" + category, benchmark["status"] == "complete")
        protocol = benchmark["config"]
        for key, value in {"repeats": 32, "warmup_rounds": 10,
                           "warmup_batches_per_round": 4, "faiss_workers": 8,
                           "schedule_seed": 20260922}.items():
            audit.equal("query_protocol:{}:{}".format(category, key), protocol[key], value)
        runs = benchmark["runs"]
        audit.equal("query_record_count:" + category, len(runs), 32 * 4)
        positions = Counter((r["method"], r["position"]) for r in runs)
        audit.check("query_position_balance:" + category,
                    all(positions[(m, p)] == 8 for m in METHODS for p in range(1, 5)))
        grouped = {}
        for run in runs:
            key = (run["repeat"], run["method"])
            if key in grouped:
                raise ValueError("Duplicate query repeat/method record for " + category)
            grouped[key] = run
        ratios = [grouped[(repeat, "original")]["pipeline_ms_per_image"] /
                  grouped[(repeat, "gb_center")]["pipeline_ms_per_image"]
                  for repeat in range(1, 33)]
        speedup = statistics.median(ratios)
        saved_speedup = benchmark["analysis"]["speedup_vs_original"]["gb_center"]["pipeline_ms_per_image"]["median"]
        audit.close("query_recomputed_speedup:" + category, speedup, saved_speedup)
        audit.check("query_saved_validation:" + category, benchmark["analysis"]["validation"]["pass"])
        latencies = {}
        construction = {}
        for method in METHODS:
            method_runs = [r for r in runs if r["method"] == method]
            latencies[method] = statistics.median(r["pipeline_ms_per_image"] for r in method_runs)
            audit.close("query_recomputed_latency:{}:{}".format(category, method), latencies[method],
                        benchmark["analysis"]["summaries"][method]["pipeline_ms_per_image"]["median"])
            if method != "original":
                timing = _read_json(stage8c / category / ("timings_" + method + ".json"))
                audit.check("construction_complete:{}:{}".format(category, method), timing["status"] == "complete")
                audit.equal("construction_repeats:{}:{}".format(category, method), len(timing["repeats"]), 5)
                construction[method] = statistics.median(r["total_seconds"] for r in timing["repeats"])
                audit.close("construction_recomputed:{}:{}".format(category, method), construction[method],
                            timing["summary"]["median"])
        rows.append({
            "category": category, "k_gb": k_gb, "baseline_k": baseline_k,
            "baseline_image_auroc": baseline_metrics["instance_auroc"],
            "center_image_auroc": gb["instance_auroc"], "ff_raw_image_auroc": ff["instance_auroc"],
            "medoid_image_auroc": medoid["instance_auroc"],
            "baseline_pixel_auroc": baseline_metrics["full_pixel_auroc"],
            "center_pixel_auroc": gb["full_pixel_auroc"],
            "ff_raw_pixel_auroc": ff["full_pixel_auroc"], "medoid_pixel_auroc": medoid["full_pixel_auroc"],
            "image_drop_pp": 100 * (baseline_metrics["instance_auroc"] - gb["instance_auroc"]),
            "pixel_drop_pp": 100 * (baseline_metrics["full_pixel_auroc"] - gb["full_pixel_auroc"]),
            "compression": baseline_k / k_gb, "query_speedup": speedup,
            "query_ms_per_image_medians": latencies, "construction_seconds_medians": construction,
            "gb_center_construction_over_ff": construction["gb_center"] / construction["ff_raw"],
        })

    main = {name: _summary([row[key] for row in rows]) for name, key in {
        "image_drop_pp": "image_drop_pp", "pixel_drop_pp": "pixel_drop_pp",
        "compression": "compression", "query_speedup": "query_speedup",
        "gb_center_construction_over_ff": "gb_center_construction_over_ff"}.items()}
    main["mean_image_auroc"] = {name: _mean([row[key] for row in rows]) for name, key in {
        "baseline": "baseline_image_auroc", "center": "center_image_auroc",
        "ff_raw": "ff_raw_image_auroc", "medoid": "medoid_image_auroc"}.items()}
    main["mean_pixel_auroc"] = {name: _mean([row[key] for row in rows]) for name, key in {
        "baseline": "baseline_pixel_auroc", "center": "center_pixel_auroc",
        "ff_raw": "ff_raw_pixel_auroc", "medoid": "medoid_pixel_auroc"}.items()}
    main["ff_vs_center"] = {
        "wins": sum(r["ff_raw_image_auroc"] > r["center_image_auroc"] for r in rows),
        "losses": sum(r["ff_raw_image_auroc"] < r["center_image_auroc"] for r in rows),
        "ties": sum(r["ff_raw_image_auroc"] == r["center_image_auroc"] for r in rows),
    }
    main["image_drop_within_5pp_count"] = sum(r["image_drop_pp"] <= 5 for r in rows)
    main["macro_query_ms_per_image"] = {method: _mean([r["query_ms_per_image_medians"][method] for r in rows])
                                         for method in METHODS}
    audit.equal("ff_wins_losses_ties", main["ff_vs_center"], {"wins": 8, "losses": 4, "ties": 3})
    audit.equal("5pp_tolerance_count", main["image_drop_within_5pp_count"], 14)
    audit.rounding("headline_compression", main["compression"]["mean"], 13.5, 1)
    audit.rounding("headline_query_speedup", main["query_speedup"]["mean"], 8.9, 1)
    audit.rounding("headline_mean_image_drop_pp", main["image_drop_pp"]["mean"], 2.78, 2)
    audit.rounding("headline_mean_pixel_drop_pp", main["pixel_drop_pp"]["mean"], 0.32, 2)
    audit.rounding("headline_query_baseline_ms", main["macro_query_ms_per_image"]["original"], 47.24, 2)
    audit.rounding("headline_query_gb_ms", main["macro_query_ms_per_image"]["gb_center"], 5.22, 2)
    audit.rounding("headline_medoid_image_loss_pp", 100 * (main["mean_image_auroc"]["center"] - main["mean_image_auroc"]["medoid"]), 1.44, 2)
    inactive = {"minimum_finite_log_consistency": min(consistency_values),
                "log_threshold": math.log(1.3), "consistency_stopped_splits": consistency_stops}
    audit.equal("gb_inactive_consistency_stops", consistency_stops, 0)
    audit.check("gb_inactive_all_logs_above_threshold", min(consistency_values) > math.log(1.3))
    audit.rounding("headline_minimum_log_consistency", min(consistency_values), 35.5, 1)

    screw_row = next(row for row in rows if row["category"] == "screw")
    screw8b = _read_json(stage8b / "screw" / "stage8b_category_summary.json")
    curve_centers = [{"k": screw_row["k_gb"], "image_auroc": screw_row["center_image_auroc"], "min_ball_size": 64}]
    for multiplier, minimum in ((2, 32), (4, 16)):
        record = _read_json(stage8c / "screw" / "runs" / ("gb_center_x" + str(multiplier)) / "result.json")
        curve_centers.append({"k": record["k"], "image_auroc": _metrics(record)["instance_auroc"], "min_ball_size": minimum})
    curve_ff = [{"k": screw8b["methods"][method]["k"],
                 "image_auroc": screw8b["methods"][method]["image_auroc"]}
                for method in ("gb_adaptive", "gb_x2", "gb_x4")]
    screw = {"center_curve": curve_centers,
             "figure_ff_curve": {"population": "original 10% coreset", "points": curve_ff},
             "same_k_ff_raw": {"population": "all raw training embeddings", "k": screw_row["k_gb"],
                               "image_auroc": screw_row["ff_raw_image_auroc"]}}
    audit.equal("screw_center_capacity", [p["k"] for p in curve_centers], [1969, 4037, 8104])
    audit.equal("screw_coreset_ff_capacity", [p["k"] for p in curve_ff], [1969, 3938, 7876])
    audit.check("screw_ff_populations_distinct", abs(curve_ff[0]["image_auroc"] - screw_row["ff_raw_image_auroc"]) > 1e-3)
    audit.rounding("headline_screw_coreset_ff_x2_auroc", curve_ff[1]["image_auroc"], 0.981, 3)

    hybrid_json = _read_json(evidence / "hybrid_fixed_rule_15cat.json")
    hybrid_rows = _by_category(hybrid_json["category_results"])
    hybrid = {"mean_image_auroc": {method: _mean([r[key] for r in hybrid_rows.values()])
              for method, key in {"hybrid": "image_auroc_hybrid", "gb_center": "image_auroc_gb_center",
                                  "ff_real": "image_auroc_ff_real"}.items()},
              "screw_image_auroc": hybrid_rows["screw"]["image_auroc_hybrid"]}
    for row in rows:
        h = hybrid_rows[row["category"]]
        audit.equal("hybrid_same_capacity:" + row["category"], h["k"], row["k_gb"])
        audit.close("hybrid_center_control:" + row["category"], h["image_auroc_gb_center"], row["center_image_auroc"])
        audit.close("hybrid_ff_control:" + row["category"], h["image_auroc_ff_real"], row["ff_raw_image_auroc"])
    for method, mean in hybrid["mean_image_auroc"].items():
        audit.close("hybrid_saved_macro:" + method, mean, hybrid_json["mean_image_auroc"][method])
    audit.rounding("headline_hybrid_screw", hybrid["screw_image_auroc"], 0.895, 3)
    audit.rounding("headline_hybrid_macro", hybrid["mean_image_auroc"]["hybrid"], 0.980, 3)

    spade_json = _read_json(evidence / "spade_image_stage_15cat_summary.json")
    spade_rows = _by_category(spade_json["categories"])
    spade = {"mean_image_auroc": {method: _mean([r["image_auroc"][method] for r in spade_rows.values()])
             for method in ("full_normal_gallery", "gb_centers", "same_k_ff_real_images")},
             "mean_compression": _mean([r["n_train_images"] / r["k_gb"] for r in spade_rows.values()])}
    for method, mean in spade["mean_image_auroc"].items():
        audit.close("spade_saved_macro:" + method, mean, spade_json["mean_image_auroc"][method])
    audit.close("spade_saved_compression", spade["mean_compression"], spade_json["mean_global_gallery_compression_ratio"])
    audit.rounding("headline_spade_compression", spade["mean_compression"], 6.97, 2)
    for method, expected in (("full_normal_gallery", .845), ("gb_centers", .821), ("same_k_ff_real_images", .837)):
        audit.rounding("headline_spade_auroc:" + method, spade["mean_image_auroc"][method], expected, 3)

    feature_json = _read_json(evidence / "screw_feature_audit_v1.json")
    feature = {"gb_normal_patch_median": feature_json["methods"]["gb_center"]["good_patch"]["median"],
               "ff_normal_patch_median": feature_json["methods"]["ff_raw"]["good_patch"]["median"],
               "gb_normal_patch_p999": feature_json["methods"]["gb_center"]["good_patch"]["p999"],
               "ff_normal_patch_p999": feature_json["methods"]["ff_raw"]["good_patch"]["p999"],
               "defect_core_center_smaller_fraction": 1 - feature_json["paired_center_vs_ff"]["anomaly_defect_core"]["fraction_center_larger"],
               "gb_max_in_defect_core_rate": feature_json["methods"]["gb_center"]["max_patch_inside_defect_core_rate"],
               "ff_max_in_defect_core_rate": feature_json["methods"]["ff_raw"]["max_patch_inside_defect_core_rate"]}
    for method in ("gb_center", "gb_medoid", "ff_raw"):
        audit.close("screw_feature_control:" + method, feature_json["methods"][method]["image_auroc"],
                    _metrics(matched_results["screw"][method])["instance_auroc"])
    for key, expected, digits in (("gb_normal_patch_median", .524, 3), ("ff_normal_patch_median", 1.318, 3),
                                  ("gb_normal_patch_p999", 2.660, 3), ("ff_normal_patch_p999", 2.194, 3)):
        audit.rounding("headline_feature:" + key, feature[key], expected, digits)
    for key, expected in (("defect_core_center_smaller_fraction", 81.9),
                           ("gb_max_in_defect_core_rate", 49.6), ("ff_max_in_defect_core_rate", 73.9)):
        audit.rounding("headline_feature:" + key, 100 * feature[key], expected, 1)

    query_summary = _read_json(query_root / "summary" / "query_latency_all_categories.json")
    audit.close("query_saved_macro_speedup", main["query_speedup"]["mean"], query_summary["headline"]["new_pipeline_speedup"]["mean"])
    audit.close("query_saved_macro_compression", main["compression"]["mean"], query_summary["headline"]["compression_ratio"]["mean"])
    manifest_result = _verify_manifest(package_root, audit) if check_files else {"checked": False}
    failed = [item for item in audit.checks if not item["passed"]]
    report = {"schema_version": 1, "passed": not failed,
              "scope": "Curated archived numeric records only; no experiment rerun, manuscript validation, external asset verification or figure pixel audit",
              "package_root": ".", "checks_total": len(audit.checks),
              "checks_failed": len(failed), "checks": audit.checks,
              "warnings": warnings,
              "computed": {"main": main, "per_category": rows, "inactive_consistency": inactive,
                           "screw_capacity": screw, "screw_features": feature, "hybrid": hybrid, "spade": spade},
              "manifest": manifest_result}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-files", action="store_true", help="Hash files in the current public lightweight manifest, if present; external assets are excluded")
    args = parser.parse_args()
    output = args.out or args.root / "verification" / "reference_check.json"
    try:
        report = verify(args.root, output, args.check_files)
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        print("Reference verification could not complete: {}".format(error), file=sys.stderr)
        return 2
    print("Reference verification: {} ({} checks, {} failures)".format(
        "PASS" if report["passed"] else "FAIL", report["checks_total"], report["checks_failed"]))
    print("Report: " + str(output.resolve()))
    for warning in report["warnings"]:
        print("WARNING: " + warning["name"], file=sys.stderr)
    for check in report["checks"]:
        if not check["passed"]:
            print("FAIL: " + check["name"], file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
