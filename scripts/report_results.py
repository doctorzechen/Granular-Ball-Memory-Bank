"""Derive paper tables and figures from measured experiment JSON files.

No accuracy, capacity, latency, or speed-up value is supplied by this module.
Missing optional experiments remain null and are listed in scope_notes.md.
Only cached-feature benchmark timings are used for query speed-ups; one-pass
scoring timings are deliberately not substituted for missing benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
)
METHODS = ("gb_center", "gb_medoid", "ff_raw")
SPADE_METHODS = (
    "full_normal_gallery", "gb_centers", "same_k_ff_real_images",
)


def _load(path, required=False):
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError("Required measured result is missing: " + str(path))
        return None
    with path.open(encoding="utf-8-sig") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    if result.get("status", "complete") != "complete":
        raise ValueError("Result is not complete: " + str(path))
    return result


def _number(value, name, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a numeric " + name + ": " + repr(value))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite " + name)
    return value


def _auc(result, name="instance_auroc", optional=False):
    value = _number(result.get("metrics", {}).get(name), name, optional)
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError("AUROC must be between zero and one: " + name)
    return value


def _capacity(result):
    value = result.get("memory_bank_size", result.get("k"))
    numeric = _number(value, "memory_bank_size")
    if numeric < 1 or numeric != int(numeric):
        raise ValueError("Memory-bank capacity must be a positive integer")
    return int(numeric)


def _stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values), "mean": statistics.fmean(values),
        "median": statistics.median(values), "min": min(values), "max": max(values),
    }


def _first_path(paths):
    return next((path for path in paths if path.is_file()), None)


def _resolve_root(run_root):
    supplied = Path(run_root).expanduser().resolve()
    candidates = (
        supplied / "patchcore-inspection/results/MVTecAD_Results",
        supplied / "results/MVTecAD_Results",
        supplied / "MVTecAD_Results",
        supplied,
    )
    results = next((path for path in candidates
                    if (path / "IM224_WR50_STAGE5B").is_dir()), None)
    if results is None:
        raise FileNotFoundError("Cannot locate IM224_WR50_STAGE5B beneath " + str(supplied))
    revision_candidates = [supplied / "revision_evidence", results / "revision_evidence"]
    for ancestor in results.parents:
        if ancestor.name == "patchcore-inspection":
            revision_candidates.append(ancestor.parent / "revision_evidence")
            break
    revision = next((path for path in revision_candidates if path.is_dir()),
                    revision_candidates[0])
    return supplied, results, revision


def _source(path, supplied):
    try:
        return path.resolve().relative_to(supplied).as_posix()
    except ValueError:
        return str(path.resolve())


def _query_metrics(results, category, notes):
    paths = [results / stage / category / "query_latency_benchmark.json"
             for stage in ("QUERYBENCH_V2", "IM224_WR50_QUERYBENCH_V2")]
    path = _first_path(paths)
    empty = {
        "source": None, "timed_scope": None,
        "original_ms_per_image": None, "gb_center_ms_per_image": None,
        "gb_medoid_ms_per_image": None, "ff_raw_ms_per_image": None,
        "speedup_gb_center": None, "speedup_gb_medoid": None, "speedup_ff_raw": None,
    }
    if path is None:
        notes.append(category + ": no query benchmark; latency and speed-up are null.")
        return empty
    benchmark = _load(path, required=True)
    analysis = benchmark.get("analysis", {})
    summaries = analysis.get("summaries", {})
    speedups = analysis.get("speedup_vs_original", {})
    timed_scope = benchmark.get("config", {}).get("timed_scope")
    value = dict(empty, source=path, timed_scope=timed_scope)
    for method in ("original",) + METHODS:
        measured = summaries.get(method, {}).get("pipeline_ms_per_image", {}).get("median")
        value[method + "_ms_per_image"] = _number(
            measured, category + "/" + method + " benchmark latency", optional=True)
        if value[method + "_ms_per_image"] is not None and value[method + "_ms_per_image"] <= 0:
            raise ValueError("Benchmark latencies must be positive: " + str(path))
        if method != "original":
            paired = speedups.get(method, {}).get("pipeline_ms_per_image", {}).get("median")
            value["speedup_" + method] = _number(
                paired, category + "/" + method + " paired benchmark speed-up", optional=True)
            if value["speedup_" + method] is not None and value["speedup_" + method] <= 0:
                raise ValueError("Benchmark speed-up must be positive: " + str(path))
    if value["speedup_gb_center"] is None:
        notes.append(category + ": benchmark lacks paired median GB speed-up; speed-up stays null.")
    validation = analysis.get("validation", {})
    if validation.get("pass") is False:
        raise ValueError("Query benchmark validation reports failure: " + str(path))
    return value


def _construction_metrics(results, category, notes):
    output = {}
    for method in METHODS:
        path = results / "IM224_WR50_STAGE8C" / category / ("timings_" + method + ".json")
        result = _load(path)
        output[method + "_seconds"] = (
            _number(result.get("summary", {}).get("median"), "construction median", True)
            if result is not None else None
        )
        if method == "gb_center":
            output["gb_partition_seconds"] = (
                _number(result.get("stage_summaries", {}).get("partition_seconds", {}).get("median"),
                        "partition median", True) if result is not None else None
            )
    gb, ff = output["gb_center_seconds"], output["ff_raw_seconds"]
    output["gb_vs_ff_construction_time_ratio"] = gb / ff if gb is not None and ff and ff > 0 else None
    if gb is None:
        notes.append(category + ": no repeated GB construction timing; build time stays null.")
    return output


def _capacity_results(results, category, baseline, center, notes, supplied):
    base_k = _capacity(center)
    original_auc = _auc(baseline)
    rows = []
    center_source = results / "IM224_WR50_STAGE8C" / category / "runs/gb_center/result.json"
    specifications = [("gb_center", 1, [center_source])]
    for multiplier in (2, 4):
        specifications.append(("gb_center", multiplier, [
            results / "IM224_WR50_STAGE8C" / category / "runs" /
            ("gb_center_x" + str(multiplier)) / "result.json",
        ]))
    for multiplier in (1, 2, 4):
        specifications.append(("ff_coreset", multiplier, [
            results / "IM224_WR50_STAGE8C" / category / "runs" /
            ("ff_coreset_x" + str(multiplier)) / "result.json",
            # Legacy Stage 8B's gb_x* names contain actual coreset patches,
            # not ball means. They must not be confused with gb_center_x*.
            results / "IM224_WR50_STAGE8B" / category / "runs" /
            ("gb_x" + str(multiplier)) / "result.json",
        ]))
    for method, requested, paths in specifications:
        path = _first_path(paths)
        if path is None:
            if category == "screw":
                notes.append("screw: missing " + method + " x" + str(requested) +
                             " capacity point; it is omitted from Fig. 3.")
            continue
        result = _load(path, True)
        actual_k = _capacity(result)
        image_auc = _auc(result)
        rows.append({
            "category": category, "method": method,
            "requested_capacity_multiplier": requested,
            "memory_bank_size": actual_k, "actual_capacity_ratio": actual_k / base_k,
            "image_auroc": image_auc,
            "image_auroc_drop_pp": 100.0 * (original_auc - image_auc),
            "source": _source(path, supplied),
        })
    return rows


def aggregate_spade(revision_dir, categories=None):
    """Aggregate category report.json files, never an existing aggregate JSON.

    Returns None when the experiment is absent. A partial run is identified by
    n_categories and missing_categories; it is never labelled as 15 categories.
    """
    requested = list(CATEGORIES if categories is None else categories)
    rows, missing = [], []
    for category in requested:
        report = _load(Path(revision_dir) / "spade_image_stage" / category / "report.json")
        if report is None:
            missing.append(category)
            continue
        if report.get("category") != category:
            raise ValueError("SPADE report category mismatch: " + category)
        count = _number(report.get("n_train_images"), "SPADE n_train_images")
        k = _number(report.get("k_gb"), "SPADE k_gb")
        if count < 1 or k < 1 or k > count or count != int(count) or k != int(k):
            raise ValueError("Invalid SPADE gallery capacities: " + category)
        metrics = {}
        for method in SPADE_METHODS:
            value = _number(report.get("image_auroc", {}).get(method), "SPADE " + method)
            if not 0.0 <= value <= 1.0:
                raise ValueError("Invalid SPADE AUROC: " + category)
            metrics[method] = value
        rows.append({"category": category, "n_train_images": int(count),
                     "k_gb": int(k), "image_auroc": metrics})
    if not rows:
        return None
    ratios = [row["n_train_images"] / row["k_gb"] for row in rows]
    return {
        "scope": "SPADE paper Section 3.2 / DN2 image-level retrieval only; no pyramid correspondence or segmentation",
        "scorer": "mean of five exact Euclidean nearest-neighbor distances",
        "n_categories": len(rows),
        "mean_image_auroc": {
            method: statistics.fmean(row["image_auroc"][method] for row in rows)
            for method in SPADE_METHODS
        },
        "mean_global_gallery_compression_ratio": statistics.fmean(ratios),
        "compression_ratio_range": [min(ratios), max(ratios)],
        "gb_wins_vs_same_k_ff": sum(row["image_auroc"]["gb_centers"] >
                                     row["image_auroc"]["same_k_ff_real_images"] for row in rows),
        "categories": rows, "missing_categories": missing,
    }


def _revision_results(revision_dir, categories, notes):
    output = {}
    feature = _load(revision_dir / "screw_feature_audit_v1.json")
    if feature is not None:
        methods = feature.get("methods", {})
        paired = feature.get("paired_center_vs_ff", {})
        fraction_larger = paired.get("anomaly_defect_core", {}).get("fraction_center_larger")
        output["screw_feature_audit"] = {
            "n_images": feature.get("n_images"),
            "n_anomalous": feature.get("n_anomalous"),
            "mask_rule": feature.get("mask_rule"),
            "methods": {method: {
                "image_auroc": value.get("image_auroc"),
                "normal_patch_median": value.get("good_patch", {}).get("median"),
                "normal_patch_p999": value.get("good_patch", {}).get("p999"),
                "max_patch_inside_defect_core_rate": value.get("max_patch_inside_defect_core_rate"),
            } for method, value in methods.items()},
            "fraction_defect_core_center_not_larger_than_ff": (
                1.0 - float(fraction_larger) if fraction_larger is not None else None),
            "caution": feature.get("caution"),
        }
    else:
        notes.append("Reviewer screw feature diagnostic is absent.")
    hybrid = _load(revision_dir / "hybrid_fixed_rule_15cat.json")
    if hybrid is not None:
        rows = [row for row in hybrid.get("category_results", []) if row.get("category") in categories]
        by_category = {row["category"]: row for row in rows}
        if len(rows) != len(by_category):
            raise ValueError("Duplicate categories in hybrid result")
        mean = {}
        for method in ("gb_center", "hybrid", "ff_real"):
            values = [_number(row.get("image_auroc_" + method), "hybrid " + method) for row in rows]
            mean[method] = statistics.fmean(values) if values else None
        output["hybrid"] = {
            "n_categories": len(rows), "mean_image_auroc": mean,
            "category_results": rows,
            "missing_categories": [category for category in categories if category not in by_category],
            "caution": hybrid.get("caution"),
        }
    else:
        notes.append("Reviewer fixed-rule hybrid experiment is absent.")
    pilot = _load(revision_dir / "screw_hybrid_pilot.json")
    if pilot is not None:
        output["screw_hybrid_pilot"] = {
            key: pilot.get(key) for key in (
                "capacity", "retained_center_count", "added_real_patch_count",
                "image_auroc", "max_patch_inside_defect_core_rate", "caution",
            )
        }
    spade = aggregate_spade(revision_dir, categories)
    output["spade_image_stage"] = spade
    if spade is None:
        notes.append("SPADE/DN2 image-retrieval experiment is absent.")
    elif spade["missing_categories"]:
        notes.append("SPADE aggregate covers only " + str(spade["n_categories"]) +
                     " requested categories; missing: " + ", ".join(spade["missing_categories"]) + ".")
    return output


def _figures(rows, capacities, output_dir, notes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False})
    paths = []
    if any(row["query_speedup"] is not None for row in rows):
        figure, axis = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
        positions = list(range(len(rows)))
        speed = [row["query_speedup"] if row["query_speedup"] is not None else math.nan
                 for row in rows]
        bars = axis.bar(positions, speed, width=.66, color="#316da8", label="Cached-feature query speed-up")
        axis.axhline(5, color="#316da8", linestyle=":", linewidth=1)
        axis.set_ylabel("Query speed-up (times)", color="#245681")
        axis.set_xticks(positions, [row["category"] for row in rows], rotation=48, ha="right")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=.15)
        axis2 = axis.twinx()
        line, = axis2.plot(positions, [row["image_auroc_drop_pp"] for row in rows],
                           color="#ac322b", marker="o", markersize=3.5,
                           linewidth=1.5, label="Image-AUROC drop")
        threshold = axis2.axhline(5, color="#ac322b", linestyle="--", linewidth=1,
                                 label="5 pp reporting tolerance")
        axis2.set_ylabel("Image-AUROC drop (percentage points)", color="#ac322b")
        axis2.set_ylim(min(-1, min(row["image_auroc_drop_pp"] for row in rows) - 1),
                       max(6, max(row["image_auroc_drop_pp"] for row in rows) * 1.18))
        figure.legend([bars, line, threshold], [bars.get_label(), line.get_label(), threshold.get_label()],
                      loc="outside upper center", ncols=2, frameon=False, fontsize=8)
        for extension in ("pdf", "png"):
            path = output_dir / ("fig2_speedup_drop" + ("_600dpi.png" if extension == "png" else ".pdf"))
            figure.savefig(path, dpi=600)
            paths.append(path.name)
        plt.close(figure)
    else:
        notes.append("Fig. 2 speed-up plot not generated: all query benchmark values are missing.")

    screw = [row for row in capacities if row["category"] == "screw"]
    if screw:
        figure, axis = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
        styles = {"gb_center": ("GB means", "#316da8", "o"),
                  "ff_coreset": ("FF real patches from 10% coreset", "#b43c2f", "s")}
        for method, (label, color, marker) in styles.items():
            selected = sorted((row for row in screw if row["method"] == method),
                              key=lambda row: row["actual_capacity_ratio"])
            if selected:
                axis.plot([row["actual_capacity_ratio"] for row in selected],
                          [row["image_auroc_drop_pp"] for row in selected],
                          label=label, color=color, marker=marker, linewidth=1.6)
        axis.axhspan(0, 5, color="#629c66", alpha=.12, label="0-5 pp reporting band")
        axis.axhline(5, color="#58835b", linestyle="--", linewidth=1)
        axis.set_xlabel("Actual bank size / original GB bank size")
        axis.set_ylabel("Image-AUROC drop vs. 10% coreset (pp)")
        axis.set_title("Screw: capacity-controlled storage comparison")
        axis.grid(alpha=.15)
        axis.legend(frameon=False, fontsize=8)
        for extension in ("pdf", "png"):
            path = output_dir / ("fig3_screw_capacity" + ("_600dpi.png" if extension == "png" else ".pdf"))
            figure.savefig(path, dpi=600)
            paths.append(path.name)
        plt.close(figure)
    else:
        notes.append("Fig. 3 not generated: no requested screw capacity results.")
    return paths


def build_report(run_root, output_dir=None, categories=None):
    """Create a report from one experiment tree, returning its summary dictionary.

    The original 10% coreset and GB-center result for every requested category
    are required. Medoid/FF controls, construction/query timings, capacity
    extensions, and reviewer experiments are optional and explicitly labelled.
    """
    supplied, results, revision = _resolve_root(run_root)
    requested = list(CATEGORIES if categories is None else categories)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Provide at least one category, without duplicates")
    unknown = [category for category in requested if category not in CATEGORIES]
    if unknown:
        raise ValueError("Unknown MVTec AD categories: " + ", ".join(unknown))
    destination = Path(output_dir).resolve() if output_dir is not None else supplied / "paper_report"
    notes, rows, capacities = [], [], []
    for category in requested:
        original_path = results / "IM224_WR50_STAGE5B" / category / "runs/original/result.json"
        center_path = results / "IM224_WR50_STAGE8C" / category / "runs/gb_center/result.json"
        original, center = _load(original_path, True), _load(center_path, True)
        for result in (original, center):
            declared = result.get("category", result.get("config", {}).get("category"))
            if declared is not None and declared != category:
                raise ValueError("Result category does not match directory: " + category)
        k, original_k = _capacity(center), _capacity(original)
        manifest_path = results / "IM224_WR50_STAGE8C" / category / "cache/banks_manifest.json"
        manifest = _load(manifest_path)
        if manifest is not None and _number(manifest.get("k_gb"), "manifest k_gb") != k:
            raise ValueError("GB capacity disagrees with bank manifest: " + category)
        image_original, image_center = _auc(original), _auc(center)
        pixel_original = _auc(original, "full_pixel_auroc", True)
        pixel_center = _auc(center, "full_pixel_auroc", True)
        if pixel_original is None or pixel_center is None:
            notes.append(category + ": missing full pixel-AUROC; pixel difference stays null.")
        query = _query_metrics(results, category, notes)
        construction = _construction_metrics(results, category, notes)
        row = {
            "category": category, "k_gb": k, "k_10pct_coreset": original_k,
            "image_auroc_original": image_original, "image_auroc_gb_center": image_center,
            "image_auroc_drop_pp": 100.0 * (image_original - image_center),
            "pixel_auroc_original": pixel_original, "pixel_auroc_gb_center": pixel_center,
            "pixel_auroc_drop_pp": (100.0 * (pixel_original - pixel_center)
                                    if pixel_original is not None and pixel_center is not None else None),
            "memory_bank_compression_ratio": original_k / k,
            "query_speedup": query["speedup_gb_center"],
            "query_original_ms_per_image": query["original_ms_per_image"],
            "query_gb_center_ms_per_image": query["gb_center_ms_per_image"],
            "query_timed_scope": query["timed_scope"],
            "query_source": _source(query["source"], supplied) if query["source"] else None,
            "original_source": _source(original_path, supplied),
            "gb_center_source": _source(center_path, supplied),
            **construction,
        }
        for method in ("gb_medoid", "ff_raw"):
            control_path = results / "IM224_WR50_STAGE8C" / category / "runs" / method / "result.json"
            control = _load(control_path)
            if control is not None and _capacity(control) != k:
                raise ValueError("Same-capacity control differs from GB K: " + category + "/" + method)
            row["image_auroc_" + method] = _auc(control) if control is not None else None
            row["pixel_auroc_" + method] = _auc(control, "full_pixel_auroc", True) if control else None
            if control is None:
                notes.append(category + ": missing " + method + " equal-capacity control.")
        row["meets_5pp_image_reporting_tolerance"] = row["image_auroc_drop_pp"] <= 5.0 + 1e-12
        row["meets_5x_query_reporting_threshold"] = (
            row["query_speedup"] >= 5 if row["query_speedup"] is not None else None)
        rows.append(row)
        capacities.extend(_capacity_results(results, category, original, center, notes, supplied))
    reviewer = _revision_results(revision, requested, notes)
    ff_comparable = [row for row in rows if row["image_auroc_ff_raw"] is not None]
    summary = {
        "schema_version": 1, "n_categories": len(rows), "categories": requested,
        "aggregation": "Equal category weights; available-value counts accompany every macro statistic.",
        "main": {
            "image_auroc_original": _stats(row["image_auroc_original"] for row in rows),
            "image_auroc_gb_center": _stats(row["image_auroc_gb_center"] for row in rows),
            "image_auroc_ff_raw": _stats(row["image_auroc_ff_raw"] for row in rows),
            "image_auroc_drop_pp": _stats(row["image_auroc_drop_pp"] for row in rows),
            "pixel_auroc_drop_pp": _stats(row["pixel_auroc_drop_pp"] for row in rows),
            "memory_bank_compression_ratio": _stats(row["memory_bank_compression_ratio"] for row in rows),
            "cached_query_speedup": _stats(row["query_speedup"] for row in rows),
            "cached_query_original_ms_per_image": _stats(row["query_original_ms_per_image"] for row in rows),
            "cached_query_gb_center_ms_per_image": _stats(row["query_gb_center_ms_per_image"] for row in rows),
            "gb_partition_seconds": _stats(row["gb_partition_seconds"] for row in rows),
            "gb_vs_ff_construction_time_ratio": _stats(row["gb_vs_ff_construction_time_ratio"] for row in rows),
            "within_5pp_image_tolerance_count": sum(row["meets_5pp_image_reporting_tolerance"] for row in rows),
            "image_tolerance_violation_categories": [row["category"] for row in rows
                                                     if not row["meets_5pp_image_reporting_tolerance"]],
            "same_capacity_ff_comparable_categories": len(ff_comparable),
            "ff_beats_centers_count": sum(row["image_auroc_ff_raw"] > row["image_auroc_gb_center"]
                                           for row in ff_comparable),
            "centers_beat_ff_count": sum(row["image_auroc_gb_center"] > row["image_auroc_ff_raw"]
                                         for row in ff_comparable),
            "centers_tie_ff_count": sum(row["image_auroc_gb_center"] == row["image_auroc_ff_raw"]
                                        for row in ff_comparable),
        },
        "category_results": rows, "capacity_study": capacities,
        "reviewer_experiments": reviewer,
    }
    destination.mkdir(parents=True, exist_ok=True)
    summary["generated_figures"] = _figures(rows, capacities, destination, notes)
    summary["scope_notes"] = notes
    with (destination / "paper_main_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if reviewer["spade_image_stage"] is not None:
        (destination / "spade_image_stage_15cat_summary.json").write_text(
            json.dumps(reviewer["spade_image_stage"], indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    text = (
        "# Reproduction report scope\n\n"
        "Every reported accuracy, capacity and timing is read from measured result files. "
        "Category means use equal weights. Nulls and missing categories are retained explicitly.\n\n"
        "Query speed-up is the median paired original/GB ratio of the cached-feature query "
        "pipeline benchmark. The pipeline includes feature-array conversion, search, merge and "
        "score reduction; it excludes neural feature extraction, bank construction and anomaly-map "
        "postprocessing. The median paired ratio is distinct from the ratio of median latencies.\n\n"
        "The 5x and 5 percentage-point lines are descriptive reporting conventions. "
        "Hardware, runtime versions and thread counts affect timings.\n\n"
        "Equal-capacity Stage 8C FF is selected from all raw training patches. "
        "The capacity-study FF control is selected from the original 10% coreset; these candidate "
        "pools are separate. Legacy Stage 8B gb_x* result names refer to that real-patch control. "
        "Fig. 3 uses actual measured K/K_GB, and omits absent points.\n\n"
        "The screw mask analysis is post-hoc and descriptive. The hybrid rule was motivated by "
        "screw test diagnostics and has no independent held-out validation. SPADE/DN2 evaluates "
        "only five-neighbor global image retrieval, with min_ball_size=4, and does not evaluate "
        "SPADE's local correspondence, segmentation or complete original protocol.\n\n"
        "Fig. 1 is an author-supplied conceptual artwork, not an automatically generated empirical "
        "plot. Only Fig. 2 and Fig. 3 are generated by this module.\n\n"
        "## Missing data and qualifications\n\n"
    )
    text += "\n".join("- " + note for note in notes) if notes else "No optional data omissions were detected."
    (destination / "scope_notes.md").write_text(text + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Package/run root, or the direct MVTecAD_Results directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES)
    args = parser.parse_args()
    summary = build_report(args.run_root, args.output_dir, args.categories)
    print(json.dumps({"n_categories": summary["n_categories"],
                      "generated_figures": summary["generated_figures"],
                      "query_speedup": summary["main"]["cached_query_speedup"]}, indent=2))


if __name__ == "__main__":
    main()
