"""PatchCore integration for frozen patch-feature extraction and scoring.

These helpers preserve the paper's ImageNet V1 / layer2-layer3 / squared-L2
protocol. The GB memory representation lives independently in gb_memory.
"""
import hashlib
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import psutil
import torch
import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
from patchcore.datasets.mvtec import DatasetSplit, MVTecDataset

def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _array_sha256(array: np.ndarray, rows_per_chunk: int = 2048) -> str:
    digest = hashlib.sha256()
    value = np.asarray(array)
    digest.update(str(value.shape).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    if value.ndim == 0:
        digest.update(value.tobytes())
    else:
        for start in range(0, len(value), rows_per_chunk):
            digest.update(np.ascontiguousarray(value[start : start + rows_per_chunk]))
    return digest.hexdigest()


def _process_memory() -> Dict[str, int]:
    info = psutil.Process(os.getpid()).memory_info()
    return {
        "rss_bytes": int(info.rss),
        "vms_bytes": int(info.vms),
        "peak_working_set_bytes": int(getattr(info, "peak_wset", info.rss)),
    }


def _build_original_model(
    device: torch.device, faiss_workers: int
) -> patchcore.patchcore.PatchCore:
    backbone = patchcore.backbones.load("wideresnet50")
    backbone.name, backbone.seed = "wideresnet50", None
    sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(0.1, device)
    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, 224, 224),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=sampler,
        nn_method=patchcore.common.FaissNN(False, faiss_workers),
    )
    return model


def _extract_test_cache(model, dataloader, device):
    first_batch = next(iter(dataloader))
    probe_images = first_batch["image"].clone()
    with torch.no_grad():
        _ = model._embed(probe_images.to(torch.float).to(device))
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    cache = []
    labels = []
    masks_gt = []
    paths = []
    subtypes = []
    feature_seconds = 0.0
    start_total = time.perf_counter()
    for batch in dataloader:
        images = batch["image"]
        _synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            features, patch_shapes = model._embed(
                images.to(torch.float).to(device), provide_patch_shapes=True
            )
        _synchronize(device)
        feature_seconds += time.perf_counter() - started
        cache.append(
            {
                "features": np.asarray(features, dtype=np.float32),
                "batch_size": int(len(images)),
                "patch_shape": [int(x) for x in patch_shapes[0]],
            }
        )
        labels.extend(batch["is_anomaly"].numpy().astype(int).tolist())
        masks_gt.extend(batch["mask"].numpy().tolist())
        paths.extend(list(batch["image_path"]))
        subtypes.extend(list(batch["anomaly"]))
    total_seconds = time.perf_counter() - start_total
    features = np.concatenate([item["features"] for item in cache], axis=0)
    report = {
        "number_of_images": int(len(labels)),
        "number_of_patch_embeddings": int(len(features)),
        "embedding_shape": list(features.shape),
        "dtype": str(features.dtype),
        "contains_nan": bool(np.isnan(features).any()),
        "contains_inf": bool(np.isinf(features).any()),
        "sha256": _array_sha256(features),
        "feature_extraction_seconds": feature_seconds,
        "feature_extraction_ms_per_image": 1000 * feature_seconds / len(labels),
        "cache_pass_total_seconds": total_seconds,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "peak_gpu_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else 0
        ),
    }
    return {
        "batches": cache,
        "labels": np.asarray(labels, dtype=np.int32),
        "masks_gt": masks_gt,
        "paths": paths,
        "subtypes": subtypes,
        "probe_images": probe_images,
        "probe_embeddings": np.array(cache[0]["features"], copy=True),
        "report": report,
    }


def _index_vectors(index, shape) -> np.ndarray:
    try:
        reconstructed = index.reconstruct_n(0, shape[0])
        if reconstructed is not None:
            return np.asarray(reconstructed, dtype=np.float32)
    except TypeError:
        pass
    reconstructed = np.empty(shape, dtype=np.float32)
    index.reconstruct_n(0, shape[0], reconstructed)
    return reconstructed


def _score_cache(
    model,
    memory_bank: np.ndarray,
    test_cache: Dict[str, object],
    device: torch.device,
    faiss_workers: int,
    keep_maps: bool,
) -> Dict[str, object]:
    scorer = patchcore.common.NearestNeighbourScorer(
        1, patchcore.common.FaissNN(False, faiss_workers)
    )
    started = time.perf_counter()
    scorer.fit([memory_bank])
    faiss_build_seconds = time.perf_counter() - started
    index = scorer.nn_method.search_index
    reconstructed = _index_vectors(index, memory_bank.shape)
    faiss_validation = {
        "ntotal": int(index.ntotal),
        "ntotal_matches_memory_bank": int(index.ntotal) == len(memory_bank),
        "reconstructed_shape": list(reconstructed.shape),
        "vectors_equal_memory_bank": bool(np.array_equal(reconstructed, memory_bank)),
        "max_absolute_difference": float(
            np.max(np.abs(reconstructed - memory_bank))
        ),
    }
    faiss_validation["pass"] = bool(
        faiss_validation["ntotal_matches_memory_bank"]
        and faiss_validation["vectors_equal_memory_bank"]
    )

    warm = test_cache["batches"][0]
    warm_scores = scorer.predict([warm["features"]])[0]
    warm_scores = model.patch_maker.unpatch_scores(
        warm_scores, batchsize=warm["batch_size"]
    ).reshape(warm["batch_size"], *warm["patch_shape"])
    _ = model.anomaly_segmentor.convert_to_segmentation(warm_scores)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    image_scores = []
    segmentations = []
    query_seconds = 0.0
    postprocess_seconds = 0.0
    for batch in test_cache["batches"]:
        features = batch["features"]
        batch_size = batch["batch_size"]
        _synchronize(device)
        started = time.perf_counter()
        patch_scores = scorer.predict([features])[0]
        _synchronize(device)
        query_seconds += time.perf_counter() - started

        started = time.perf_counter()
        scores = model.patch_maker.unpatch_scores(
            patch_scores, batchsize=batch_size
        )
        scores = scores.reshape(*scores.shape[:2], -1)
        scores = model.patch_maker.score(scores)
        patch_grid = model.patch_maker.unpatch_scores(
            patch_scores, batchsize=batch_size
        ).reshape(batch_size, *batch["patch_shape"])
        masks = model.anomaly_segmentor.convert_to_segmentation(patch_grid)
        _synchronize(device)
        postprocess_seconds += time.perf_counter() - started
        image_scores.extend(np.asarray(scores).tolist())
        segmentations.extend(masks)

    scores = np.asarray(image_scores, dtype=np.float32)
    maps = np.asarray(segmentations, dtype=np.float32)
    labels = test_cache["labels"]
    image_metrics = patchcore.metrics.compute_imagewise_retrieval_metrics(
        scores, labels
    )
    full_pixel = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        maps, test_cache["masks_gt"]
    )
    anomaly_indices = [
        index
        for index, mask in enumerate(test_cache["masks_gt"])
        if np.sum(mask) > 0
    ]
    anomaly_pixel = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        maps[anomaly_indices],
        [test_cache["masks_gt"][index] for index in anomaly_indices],
    )
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    feature_seconds = test_cache["report"]["feature_extraction_seconds"]
    total_seconds = feature_seconds + query_seconds + postprocess_seconds
    score_analysis = {
        "normal": {
            "count": int(len(normal_scores)),
            "mean": float(np.mean(normal_scores)),
            "median": float(np.median(normal_scores)),
            "p95": float(np.percentile(normal_scores, 95)),
            "max": float(np.max(normal_scores)),
        },
        "anomaly": {
            "count": int(len(anomaly_scores)),
            "mean": float(np.mean(anomaly_scores)),
            "median": float(np.median(anomaly_scores)),
            "p5": float(np.percentile(anomaly_scores, 5)),
            "max": float(np.max(anomaly_scores)),
        },
        "median_margin": float(
            np.median(anomaly_scores) - np.median(normal_scores)
        ),
    }
    return {
        "metrics": {
            "instance_auroc": float(image_metrics["auroc"]),
            "full_pixel_auroc": float(full_pixel["auroc"]),
            "anomaly_pixel_auroc": float(anomaly_pixel["auroc"]),
            "aupro_available": False,
            "aupro": None,
        },
        "timing": {
            "faiss_build_seconds": faiss_build_seconds,
            "feature_extraction_seconds": feature_seconds,
            "feature_extraction_ms_per_image": 1000
            * feature_seconds
            / len(labels),
            "nearest_neighbor_query_seconds": query_seconds,
            "query_ms_per_image": 1000 * query_seconds / len(labels),
            "segmentation_postprocess_seconds": postprocess_seconds,
            "segmentation_postprocess_ms_per_image": 1000
            * postprocess_seconds
            / len(labels),
            "end_to_end_component_sum_seconds": total_seconds,
            "end_to_end_ms_per_image": 1000 * total_seconds / len(labels),
            "timing_method": (
                "common measured feature extraction plus method-specific measured "
                "FAISS query and official segmentation/postprocess; one warm-up batch"
            ),
        },
        "resources": {
            "peak_gpu_allocated_bytes_scoring": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "peak_gpu_reserved_bytes_scoring": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else 0
            ),
            **_process_memory(),
        },
        "faiss_validation": faiss_validation,
        "score_analysis": score_analysis,
        "scores": scores,
        "maps": maps if keep_maps else None,
    }


class _TrainingCapture:
    def __init__(self):
        self.raw_features = None

    def __call__(self, raw_features, original_memory_bank, selected_indices):
        self.raw_features = raw_features
        return {
            "memory_bank": original_memory_bank,
            "configuration": {"method": "original_training_capture"},
            "selected_indices_available": selected_indices is not None,
        }


def _loaders(dataset_path, category, batch_size, num_workers):
    train = MVTecDataset(dataset_path, category, resize=256, imagesize=224,
                         split=DatasetSplit.TRAIN, seed=0)
    test = MVTecDataset(dataset_path, category, resize=256, imagesize=224,
                        split=DatasetSplit.TEST, seed=0)
    options = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers,
                   pin_memory=True)
    return (torch.utils.data.DataLoader(train, **options),
            torch.utils.data.DataLoader(test, **options), train, test)
