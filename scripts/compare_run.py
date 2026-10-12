"""Check stage coverage, score agreement and fresh GPU capacity drift separately.

Missing declared-stage outputs fail coverage. Deliberately omitted stages are not
certified. Hardware-dependent timing values are never forced to match an archive.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

STAGES = ("prepare", "banks", "detect", "capacity", "construction", "query",
          "screw", "hybrid", "spade", "report")
METRICS = ("instance_auroc", "full_pixel_auroc", "anomaly_pixel_auroc")
METHODS = ("gb_center", "gb_medoid", "ff_raw")
SPADE_METHODS = ("full_normal_gallery", "gb_centers", "same_k_ff_real_images")
RELATIVE = Path("patchcore-inspection/results/MVTecAD_Results")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def compare(root, run_root, tolerance=1e-5, write=True):
    root, run_root = Path(root).resolve(), Path(run_root).resolve()
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    config, evidence = read(root / "configs/paper.json"), root / "reference"
    checks, missing, failures, unreferenced, capacity_checks = [], [], [], [], []

    def relative(path):
        try:
            return path.relative_to(run_root).as_posix()
        except ValueError:
            return str(path)

    def failure(path, reason, **details):
        failures.append({"output": relative(path), "reason": reason, **details})

    def record(path, required=False):
        if not path.is_file():
            if required:
                missing.append(relative(path))
            return None
        try:
            value = read(path)
            if not isinstance(value, dict) or value.get("status", "complete") != "complete":
                raise ValueError("JSON object is not a complete result")
            return value
        except (ValueError, OSError) as error:
            failure(path, str(error))
            return None

    def require_file(path):
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative(path))
            return False
        return True

    def numeric(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def close(experiment, category, method, metric, actual, expected):
        delta = actual - expected if numeric(actual) and numeric(expected) else None
        checks.append({"experiment": experiment, "category": category, "method": method,
                       "metric": metric, "regenerated": actual, "reference": expected,
                       "delta": delta, "passed": delta is not None and abs(delta) <= tolerance})

    def compare_metrics(actual, expected, experiment, category, method):
        for metric in METRICS:
            close(experiment, category, method, metric,
                  actual.get("metrics", {}).get(metric), expected.get("metrics", {}).get(metric))

    def same_capacity(actual, expected, path):
        k = actual.get("memory_bank_size", actual.get("k"))
        target = expected.get("memory_bank_size", expected.get("k"))
        if not numeric(k) or k < 1 or k != int(k) or k != target:
            failure(path, "Capacity differs from reference", regenerated_k=k, reference_k=target)

    protocol = record(run_root / "run_protocol.json", True)
    categories = list(protocol.get("categories", [])) if protocol else []
    if not categories or len(categories) != len(set(categories)) or any(
            category not in config["categories"] for category in categories):
        failure(run_root / "run_protocol.json", "Invalid or missing category list")
    marker_path = run_root / "execution_complete.json"
    marker = record(marker_path, True)
    stages = marker.get("executed_stages", []) if marker else []
    if not isinstance(stages, list) or len(stages) != len(set(stages)) or any(stage not in STAGES for stage in stages):
        failure(marker_path, "Invalid executed_stages")
        stages = []
    if marker and (marker.get("categories") != categories or marker.get("mode") != (protocol or {}).get("mode")):
        failure(marker_path, "Completion marker disagrees with run protocol")
    if marker is not None and not stages:
        failure(marker_path, "No completed stages declared")
    executed, results = set(stages), run_root / RELATIVE
    main_records = {}

    for category in categories:
        for experiment, methods in (("IM224_WR50_STAGE5B", ("original",)),
                                    ("IM224_WR50_STAGE8C", METHODS)):
            for method in methods:
                suffix = RELATIVE / experiment / category / "runs" / method / "result.json"
                path = run_root / suffix
                actual = record(path, "detect" in executed)
                if actual is None:
                    continue
                expected = record(evidence / suffix, True)
                if expected is None:
                    continue
                main_records[(category, method)] = actual
                compare_metrics(actual, expected, experiment, category, method)
                same_capacity(actual, expected, path)
                if actual.get("faiss_validation", {}).get("pass") is False:
                    failure(path, "FAISS bank validation failed")
                if "detect" in executed:
                    require_file(path.parent / "image_scores.npy")

        if executed & {"prepare", "banks"}:
            cached = results / "IM224_WR50_STAGE5B" / category / "cache"
            for filename in ("training_embeddings.npy", "test_embeddings.npy", "labels.npy",
                             "masks_gt.npy", "original_memory_bank.npy", "selected_indices.npy",
                             "gb_partition.npz", "training_cache.json", "test_cache.json"):
                require_file(cached / filename)
        if "banks" in executed:
            cached = results / "IM224_WR50_STAGE8C" / category / "cache"
            manifest = record(cached / "banks_manifest.json", True)
            for filename in ("gb_center_bank.npy", "gb_medoid_bank.npy", "ff_raw_bank.npy",
                             "gb_medoid_indices.npy", "ff_raw_order.npy"):
                require_file(cached / filename)
            if manifest:
                for name, value in manifest.get("audits", {}).items():
                    if isinstance(value, bool) and not value:
                        failure(cached / "banks_manifest.json", "Bank audit failed: " + name)
                center = main_records.get((category, "gb_center"))
                if center and manifest.get("k_gb") != center.get("memory_bank_size", center.get("k")):
                    failure(cached / "banks_manifest.json", "Manifest K differs from score result")
        if "construction" in executed:
            for method in METHODS:
                path = results / "IM224_WR50_STAGE8C" / category / ("timings_" + method + ".json")
                actual = record(path, True)
                if actual:
                    summary = actual.get("summary", {})
                    if (not numeric(summary.get("median")) or summary["median"] <= 0 or
                            summary.get("n_repeats") != config["construction_repeats"] or
                            len(actual.get("repeats", [])) != config["construction_repeats"]):
                        failure(path, "Incomplete repeated construction timing")
        if "query" in executed:
            path = results / "IM224_WR50_QUERYBENCH_V2" / category / "query_latency_benchmark.json"
            actual = record(path, True)
            if actual:
                expected = {"category": category, "faiss_workers": config["faiss_workers"],
                            "repeats": config["query_repeats"], "warmup_rounds": config["query_warmup_rounds"],
                            "warmup_batches_per_round": config["query_warmup_batches"],
                            "schedule_seed": config["query_schedule_seed"]}
                for field, value in expected.items():
                    if actual.get("config", {}).get(field) != value:
                        failure(path, "Query protocol mismatch: " + field)
                if actual.get("analysis", {}).get("validation", {}).get("pass") is not True:
                    failure(path, "Query benchmark validation did not pass")
                analysis = actual.get("analysis", {})
                for method in ("original",) + METHODS:
                    summary = analysis.get("summaries", {}).get(method, {}).get("pipeline_ms_per_image", {})
                    if (not numeric(summary.get("median")) or summary["median"] <= 0 or
                            summary.get("n") != config["query_repeats"]):
                        failure(path, "Incomplete positive query latency summary: " + method)
                    if method != "original":
                        speed = analysis.get("speedup_vs_original", {}).get(method, {}).get("pipeline_ms_per_image", {})
                        if (not numeric(speed.get("median")) or speed["median"] <= 0 or
                                speed.get("n") != config["query_repeats"]):
                            failure(path, "Incomplete paired query speed-up summary: " + method)
                if len(actual.get("runs", [])) != config["query_repeats"] * (1 + len(METHODS)):
                    failure(path, "Query run count does not cover every method and repeat")
    if "query" in executed:
        require_file(results / "IM224_WR50_QUERYBENCH_V2/summary/query_latency_all_categories.json")

    revision = run_root / "revision_evidence"
    path = revision / "hybrid_fixed_rule_15cat.json"
    actual = record(path, "hybrid" in executed)
    if actual:
        expected = record(evidence / "revision_evidence/hybrid_fixed_rule_15cat.json", True)
        rows = actual.get("category_results", [])
        by_category = {row.get("category"): row for row in rows}
        if len(rows) != len(by_category) or set(by_category) != set(categories):
            failure(path, "Hybrid category coverage differs from run protocol")
        if expected:
            targets = {row["category"]: row for row in expected["category_results"]}
            for category in categories:
                if category not in by_category:
                    continue
                row, target = by_category[category], targets[category]
                for method in ("gb_center", "hybrid", "ff_real"):
                    key = "image_auroc_" + method
                    close("hybrid", category, method, key, row.get(key), target[key])
                for field in ("k", "replaced", "n_test_images"):
                    close("hybrid_selection", category, "hybrid", field, row.get(field), target[field])
            if set(by_category) == set(categories):
                for method in ("gb_center", "hybrid", "ff_real"):
                    target = statistics.fmean(targets[c]["image_auroc_" + method] for c in categories)
                    close("hybrid_macro", None, method, "image_auroc",
                          actual.get("mean_image_auroc", {}).get(method), target)

    spade_records = {}
    for category in categories:
        path = revision / "spade_image_stage" / category / "report.json"
        actual = record(path, "spade" in executed)
        if actual is None:
            continue
        expected = record(evidence / "revision_evidence/spade_image_stage" / category / "report.json", True)
        if expected is None:
            continue
        spade_records[category] = actual
        for method in SPADE_METHODS:
            close("spade_image_retrieval", category, method, "image_auroc",
                  actual.get("image_auroc", {}).get(method), expected["image_auroc"][method])
        for field in ("n_train_images", "n_test_images", "k_gb", "compression_ratio"):
            close("spade_gallery", category, None, field, actual.get(field), expected[field])
        if "spade" in executed:
            for filename in ("train_global.npy", "test_global.npy", "image_scores.npz",
                             "gb_image_centers.npy", "ff_image_indices.npy"):
                require_file(path.parent / filename)

    if "screw" in categories:
        path = revision / "screw_feature_audit_v1.json"
        actual = record(path, "screw" in executed)
        if actual:
            expected = record(evidence / "revision_evidence/screw_feature_audit_v1.json", True)
            if expected:
                for method in METHODS:
                    row, target = actual.get("methods", {}).get(method, {}), expected["methods"][method]
                    for metric in ("image_auroc", "max_patch_inside_defect_core_rate"):
                        close("screw_feature_audit", "screw", method, metric, row.get(metric), target[metric])
                    for metric in ("median", "p999"):
                        close("screw_normal_patch", "screw", method, metric,
                              row.get("good_patch", {}).get(metric), target["good_patch"][metric])
                for field in ("n_images", "n_good", "n_anomalous", "n_core_patches"):
                    close("screw_feature_grouping", "screw", None, field, actual.get(field), expected[field])
                close("screw_defect_response", "screw", "gb_center_vs_ff", "fraction_center_larger",
                      actual.get("paired_center_vs_ff", {}).get("anomaly_defect_core", {}).get("fraction_center_larger"),
                      expected["paired_center_vs_ff"]["anomaly_defect_core"]["fraction_center_larger"])
        path = revision / "screw_hybrid_pilot.json"
        actual = record(path, "screw" in executed)
        if actual:
            expected = record(evidence / "revision_evidence/screw_hybrid_pilot.json", True)
            if expected:
                for metric in ("image_auroc", "max_patch_inside_defect_core_rate", "capacity",
                               "retained_center_count", "added_real_patch_count"):
                    close("screw_hybrid_pilot", "screw", None, metric, actual.get(metric), expected[metric])
        if "screw" in executed:
            for filename in ("screw_feature_image_rows_v1.csv", "screw_hybrid_bank.npy",
                             "screw_hybrid_image_scores.npy", "screw_hybrid_removed_center_indices.npy",
                             "screw_hybrid_added_raw_indices.npy"):
                require_file(revision / filename)

    for category in ("screw", "capsule", "cable", "pill"):
        if category not in categories:
            continue
        methods = ["gb_center_x2"] + (["gb_center_x4"] if category == "screw" else [])
        if category == "screw":
            methods.extend("ff_coreset_x" + str(x) for x in (1, 2, 4))
        for method in methods:
            suffix = RELATIVE / "IM224_WR50_STAGE8C" / category / "runs" / method / "result.json"
            path = run_root / suffix
            actual = record(path, "capacity" in executed)
            if actual is None:
                continue
            target_path = evidence / suffix
            if method.startswith("ff_coreset_") and not target_path.is_file():
                target_path = (evidence / RELATIVE / "IM224_WR50_STAGE8B" / category / "runs" /
                               ("gb_x" + method.rsplit("x", 1)[1]) / "result.json")
            expected = record(target_path)
            if expected is None:
                unreferenced.append({"output": relative(path),
                                     "reason": "No archived capacity point; no reference metric invented"})
                continue
            if method.startswith("ff_coreset_"):
                compare_metrics(actual, expected, "ff_coreset_capacity", category, method)
                same_capacity(actual, expected, path)
            else:
                k = actual.get("k", actual.get("memory_bank_size"))
                target_k = expected.get("k", expected.get("memory_bank_size"))
                drift = {}
                for metric in METRICS:
                    value, target = actual.get("metrics", {}).get(metric), expected.get("metrics", {}).get(metric)
                    drift[metric] = value - target if numeric(value) and numeric(target) else None
                within = k == target_k and all(delta is not None and abs(delta) <= tolerance for delta in drift.values())
                capacity_checks.append({"category": category, "method": method, "regenerated_k": k,
                    "reference_k": target_k, "k_delta": k - target_k if numeric(k) and numeric(target_k) else None,
                    "regenerated_auroc": actual.get("metrics", {}).get("instance_auroc"),
                    "reference_auroc": expected["metrics"]["instance_auroc"], "delta": drift["instance_auroc"],
                    "metric_drift": drift, "within_main_tolerance": within,
                    "comparison_status": "match" if within else "fresh_gpu_partition_drift"})

    if "report" in executed:
        folder = run_root / "paper_report"
        actual = record(folder / "summary.json", True)
        require_file(folder / "paper_main_table.csv")
        require_file(folder / "scope_notes.md")
        if actual:
            if actual.get("categories") != categories or actual.get("n_categories") != len(categories):
                failure(folder / "summary.json", "Report category coverage differs from run protocol")
            for filename in actual.get("generated_figures", []):
                require_file(folder / filename)
            figure_names = []
            if "query" in executed:
                figure_names.extend(("fig2_speedup_drop_600dpi.png", "fig2_speedup_drop.pdf"))
            if "capacity" in executed and "screw" in categories:
                figure_names.extend(("fig3_screw_capacity_600dpi.png", "fig3_screw_capacity.pdf"))
            for filename in figure_names:
                require_file(folder / filename)
            if "spade" in executed:
                aggregate = record(folder / "spade_image_stage_15cat_summary.json", True)
                if aggregate:
                    if aggregate.get("n_categories") != len(categories):
                        failure(folder / "spade_image_stage_15cat_summary.json", "SPADE aggregate incomplete")
                    if len(spade_records) == len(categories):
                        for method in SPADE_METHODS:
                            target = statistics.fmean(spade_records[c]["image_auroc"][method] for c in categories)
                            close("spade_report_aggregation", None, method, "image_auroc",
                                  aggregate.get("mean_image_auroc", {}).get(method), target)
            for method, key in (("original", "image_auroc_original"),
                                ("gb_center", "image_auroc_gb_center"), ("ff_raw", "image_auroc_ff_raw")):
                selected = [main_records[(c, method)]["metrics"]["instance_auroc"]
                            for c in categories if (c, method) in main_records]
                if selected:
                    close("main_report_aggregation", None, method, "image_auroc",
                          actual.get("main", {}).get(key, {}).get("mean"), statistics.fmean(selected))
        table_path = folder / "paper_main_table.csv"
        if table_path.is_file():
            try:
                with table_path.open(encoding="utf-8-sig", newline="") as stream:
                    table = list(csv.DictReader(stream))
                if len(table) != len(categories) or {row.get("category") for row in table} != set(categories):
                    failure(table_path, "Main table category coverage incomplete")
            except (OSError, ValueError) as error:
                failure(table_path, str(error))

    missing = sorted(set(missing))
    completed = marker is not None and bool(stages) and not missing and not failures
    scores_match = all(row["passed"] for row in checks) if checks else None
    capacity_match = all(row["within_main_tolerance"] for row in capacity_checks) if capacity_checks else None
    output = {"passed": completed and scores_match is not False and capacity_match is not False,
        "executed_stages_complete": completed, "score_match_to_reference": scores_match,
        "fresh_gpu_capacity_match_to_reference": capacity_match,
        "comparison_scope": "Declared completed stages only. Omitted stages are not certified. Unarchived outputs have no claimed reference match.",
        "executed_stages": stages, "intentionally_omitted_stages": [s for s in STAGES if s not in executed],
        "completion_marker_present": marker is not None, "check_count": len(checks),
        "absolute_tolerance": tolerance, "mode": (protocol or {}).get("mode"), "categories": categories,
        "checks": checks, "fresh_gpu_capacity_checks": capacity_checks, "unreferenced_outputs": unreferenced,
        "missing_requested_outputs": missing,
        "missing_main_results": [p for p in missing if p.endswith("/result.json") and
            any("/runs/" + m + "/" in p for m in ("original",) + METHODS)],
        "validation_failures": failures,
        "timing_note": "Hardware-dependent timings are checked for coverage and protocol, not identical milliseconds."}
    if write:
        (run_root / "comparison_to_reference.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Regenerated/reference checks:", len(checks), "score passes:", sum(r["passed"] for r in checks),
          "missing requested:", len(missing), "validation failures:", len(failures),
          "completed:", completed, "GPU capacity match:", capacity_match)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--no-write", action="store_true", help="Read-only audit without comparison JSON output")
    args = parser.parse_args()
    report = compare(args.root, args.run_root, args.tolerance, write=not args.no_write)
    raise SystemExit(0 if report["passed"] else 1)
