# Upstream code and local modifications

The retained `src/patchcore` files originate from `amazon-science/patchcore-inspection` (Apache-2.0). Its original LICENSE and NOTICE are retained unchanged. This is a vendored subset, not a claim of a new feature extractor or a new PatchCore implementation.

Compared with the upstream source inspected for this release:

- `patchcore.py`: exposes a memory-bank callback receiving all raw training features, the sampled bank, and its row indices. The callback supports training-feature capture for controlled experiments. Historical export-to-disk hooks from the research checkout are removed to avoid unnecessary dependencies.
- `sampler.py`: records selected row indices. Sampling and distance-update algorithms are unchanged.
- Other retained source files keep the upstream implementation, including patch-grid alignment, feature aggregation, FAISS querying, image scoring, map upsampling/smoothing and the MVTec loader.

Research algorithms and experiment runners are outside this directory. The main GB representation consumes **all normal training embeddings**, not the PatchCore coreset. Upstream CLI/test files are not needed by the new entry point and are not bundled here.

The original research checkout had no Git metadata, so no upstream commit identifier can be responsibly claimed. `docs/vendor_source_hashes.json` records SHA256 for the exact retained files; results and tests, rather than an invented revision identifier, anchor this release.
