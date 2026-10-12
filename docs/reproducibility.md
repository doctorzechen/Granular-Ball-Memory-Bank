# Reproducibility protocol

## Inputs and modes

`--assets-root` selects a read-only asset folder with `data/mvtec`, `weights/hub/checkpoints`, and optional `frozen`. Dataset and frozen directories can be overridden independently. Source code never imports Python modules from that folder. Do not put output inside an input directory. The entry point resolves paths relative to its own file, so it also runs from another working directory.

Fresh mode (default) checks dataset PNG hashes, uses the exact V1 checkpoint, extracts normal/test features, builds the 10% reference coreset, builds the GB partition and computes baseline scores. Frozen mode reuses saved raw features, test features, labels, masks and partition; it is not evidence of a new full image-to-feature run. Frozen-only banks/detect/report do not require the raw PNGs; image-retrieval transfer does.

Binary frozen inputs are hard-linked when possible and otherwise copied to the run directory. They are treated as read-only; do not edit the linked cache files manually. JSON/results are separate writable copies. The original full reproduction package is not modified by these runners.

## Stage mapping

| `--stages` | Responsibility |
| --- | --- |
| prepare | Fresh image-to-feature extraction, provenance and original baseline. |
| banks | GB centers, nearest-center observed patches and exact raw-space farthest-first at K_GB. |
| detect | Re-score the reference and three same-K banks on identical cached queries. |
| capacity | GB minimum-child sizes 32/16; screw 1/2/4× FF over the original coreset. |
| construction | 1 warm-up +5 measured rebuilds, CUDA synchronization, resumable checkpoints. |
| query | Exact CPU IndexFlatL2, 8 threads, interleaved schedule, 10 warm-up rounds×4 batches, 32 repeats. |
| screw | Paired feature/score/localization diagnostics and the preliminary hybrid pilot. |
| hybrid | Fixed training-only 25% replacement rule, evaluated separately from the main method. |
| spade | 2048-D global image-retrieval features; mean of five Euclidean neighbor distances. No full SPADE localization. |
| report | Summaries and scientific plots from generated results, not copied headline times. |

Historical `IM224_WR50_STAGE5B` / `IM224_WR50_STAGE8C` names remain **only as output/reference schema identifiers**, preserving recorded configuration hashes. Their historical modules are not runtime imports.

## Checks and examples

```bash
python reproduction.py run --assets-root /path/to/assets --categories toothbrush --stages prepare banks detect report --out-root runs/fresh_smoke
python reproduction.py run --assets-root /path/to/assets --mode frozen --stages banks detect report --out-root runs/main15
python scripts/compare_run.py --root . --run-root runs/main15
python reproduction.py run --assets-root /path/to/assets --mode frozen --categories screw --stages banks detect capacity screw hybrid report --out-root runs/screw
python reproduction.py run --assets-root /path/to/assets --mode frozen --stages banks detect capacity construction query screw hybrid spade report --out-root runs/full
```

Different stage selections can be resumed only with the same category/mode/device protocol and unchanged configuration. Fresh and frozen output folders should be separate. `execution_complete.json` lists only stages actually completed, cumulatively under that protocol. Missing stages are not implied successful.

Complete bank/detection/capacity checkpoints must still have their required arrays with valid shapes; a surviving JSON marker alone is insufficient. Missing or malformed artifacts fail visibly. Use a new output directory after changing code/library versions, since recorded configuration hashes do not fingerprint the complete runtime environment.

The reference verifier validates curated numerical evidence and, when requested, release-file integrity. It is not a replacement for running experiments. The comparison utility checks generated results and coverage; hardware timings are measured anew, not forced to equal archived timings. Dataset hash validation anchors the particular input archive, and may reject a differently encoded distribution even when images look the same.

Query benchmarking retains the original ≤10% CV/order-effect/same-K-spread quality gates. A complete set of 32 measurements can still fail these gates despite stable distances and identical bank sizes. The report deliberately refuses failed benchmarks; do not relax the threshold or select a favorable run for a performance claim. Inspect `analysis.validation` and rerun only after removing known background load. For accuracy-only verification, use a separate output directory without the failed query records; report leaves timing fields null instead of inventing values. A failed benchmark is not silently counted as a validated speed-up.

The optional `scripts/validate_release.py` audits this refactoring against the preserved **complete** `reproduction_code` archive, including its historical source and prior `runs/verified_15_accuracy`. It is not required to run a fresh source-only clone and cannot use an assets folder containing only data/weights.

## Known evidence limits

CPU/GPU geometry and library versions can affect tie breaking and small floating-point differences. Keep the frozen paper environment for exact comparisons; use CPU tests for contracts, not timing claims. Construction and raw-space farthest-first can take significant time and GPU memory.

Use `python scripts/run_tests.py -q` for the supplied tests. This initializes PyTorch before pytest and FAISS and disables unrelated ambient pytest plugins, avoiding the native-loader failure observed with this Windows environment's plain pytest invocation. No system packages are modified.

The complete historical package contains much more exploratory research than the final-paper stages. This release does not promise to reproduce unused historical analyses. Ancillary old reporting conventions lacking source scripts are not claimed as reproduced.

See `validation.md` for the checks actually executed on this reorganized release. A source-only clone needs separately obtained data/weights; it is deliberately not described as an offline self-contained asset bundle.
