# Granular-Ball Memory Bank

A center-based representation of normal training embeddings for industrial anomaly detection. The method builds a size-guarded granular-ball partition over **all normal training patch embeddings** and retains only its mean centers for nearest-neighbor inference.

`normal embeddings → coarse k-means → size-guarded recursive 2-means → mean centers → compact memory bank`

The representation is independent of the feature extractor. The paper's controlled experiments use an explicitly attributed PatchCore adapter; they do not establish that the complete method works with every anomaly-detection framework. A separate SPADE-style **image-retrieval stage** tests a limited transfer setting, not full SPADE localization.

## What to read first

- `src/gb_memory/partition.py`: the actual partition and mean-center construction.
- `src/gb_memory/bank.py`: standalone fit/save/load API; inference artifacts contain no member patches.
- `src/adapters/patchcore.py`: feature extraction, cache scoring and integration with the reused pipeline.
- `experiments/`: same-capacity controls, capacity study, query benchmarking, screw diagnostics, exploratory hybrid and image-retrieval transfer.
- `third_party/patchcore/`: clearly separated upstream implementation with its LICENSE, NOTICE and local-change record.

See [contributions](docs/contributions.md), [code origins](docs/code_origins.md), and [reproducibility](docs/reproducibility.md). The distinction is about algorithm responsibilities and evidence, not file-count percentages.

## Standalone API

Install NumPy and PyTorch, then install this source tree with `python -m pip install -e . --no-deps`. The core does not import PatchCore, FAISS or the experiment runners. For an entirely offline install, preinstall setuptools≥77 and add `--no-build-isolation`.

```python
import numpy as np
from gb_memory import GranularBallMemoryBank

normal = np.random.default_rng(0).normal(size=(512, 32)).astype(np.float32)
bank = GranularBallMemoryBank(min_ball_size=64, seed=0, device="cpu").fit(normal)
centers = bank.centers  # [K_GB, D]; use these as the downstream search gallery
bank.save("normal_centers.npz")
restored = GranularBallMemoryBank.load("normal_centers.npz")
np.testing.assert_array_equal(centers, restored.centers)
```

This synthetic example checks the API, not the paper's accuracy. The paper uses 1024-dimensional embeddings and the frozen configuration in `configs/paper.json`.

## Reproduce the experiments

The tested environment is Python 3.9.25, PyTorch 2.8.0, TorchVision 0.23.0, CUDA 12.8 and the pinned dependencies in `requirements.txt`. Install PyTorch/TorchVision with the wheel appropriate for your hardware, then:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/run_tests.py -q
python reproduction.py --help
python reproduction.py verify-reference
```

Data, checkpoint and multi-gigabyte feature caches are **not** distributed in this source directory. Use the original MVTec AD train/test/ground-truth layout and the exact TorchVision ImageNet V1 WR50 checkpoint:

```text
assets/
  data/mvtec/<category>/{train,test,ground_truth}/...
  weights/hub/checkpoints/wide_resnet50_2-95faca4d.pth
  frozen/patchcore-inspection/results/MVTecAD_Results/...  # optional
```

The checkpoint SHA256 is recorded in `reference/weight.json`. Download data and weights from their original providers under their respective licenses; this runner does not silently download or switch to ImageNet V2. If you already have the complete `reproduction_code` package, it can serve as `--assets-root` without modification.

Fresh extraction, the default mode:

```bash
python reproduction.py doctor --assets-root /path/to/assets --categories toothbrush
python reproduction.py run --assets-root /path/to/assets --categories toothbrush --stages prepare banks detect hybrid spade report --out-root runs/fresh_toothbrush
python reproduction.py run --assets-root /path/to/assets --out-root runs/fresh_15
```

Recompute banks and scores from frozen embeddings (does **not** rerun the backbone):

```bash
python reproduction.py run --assets-root /path/to/reproduction_code --mode frozen --stages banks detect report --out-root runs/frozen_15
python scripts/compare_run.py --root . --run-root runs/frozen_15
```

Run all stages to obtain capacity controls, construction timing, cached-feature query benchmarking, screw diagnostics, hybrid exploration and SPADE-style image retrieval. Timing-quality gates can reject a noisy benchmark; this is not an accuracy failure, and its speed-ups must not be cited. `--stages` supports selective/resumable work. Exact commands, inputs, output interpretation and limitations are in [reproducibility.md](docs/reproducibility.md).

## Interpretation and attribution

The study observes smaller banks and faster cached-feature querying, with reduced image AUROC in some categories. Matched-capacity real-patch farthest-first is a necessary control and is **not** claimed to have worse accuracy. Hybrid results are exploratory, not the main method.

PatchCore is an actual code/experimental dependency of the adapter, not merely an unnamed inspiration. WR50, k-means, farthest-first, FAISS, maximum patch scoring and Gaussian smoothing are established components, not inventions claimed by this repository. See `third_party/patchcore/MODIFICATIONS.md` for exact reuse boundaries.

The third-party code retains its Apache-2.0 license. No dataset/weight license is overridden. Before public release, the project owners should choose the license for the project's own code and add the final publication citation; neither is invented here.
