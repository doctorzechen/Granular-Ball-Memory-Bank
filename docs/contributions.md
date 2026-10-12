# Contribution boundaries

## Method represented here

The research object is a granular-ball **mean-center memory representation** of all normal patch embeddings. The construction combines coarse initialization, size-guarded recursive splitting and mean-center retention. The searchable bank contains centers rather than selected training patches; this is the substantive representation change studied in the paper.

`gb_memory` can consume embeddings from another extractor without importing the PatchCore code. This API separation is not experimental evidence of universal transfer or a claim that every partitioning primitive is new.

## Project implementations and evidence

The project implements batched construction, explicit partition-coverage checks, center-only persistence, feature/cache provenance, matched-capacity real-patch controls, repeated synchronized construction timing and interleaved cached-feature query benchmarking. The reviewer additions examine the screw failure mode, a fixed training-only hybrid rule and a separate image-retrieval transfer setting.

These implementations make the study reproducible. Writing more experiment code does not automatically create an algorithmic novelty claim.

## Established and reused components

Pretrained WR50, feature neighborhoods, feature aggregation/alignment, nearest-neighbor search, max patch scoring and map postprocessing are not claimed as new. The evaluated feature/scoring implementation is reused from the attributed PatchCore source. K-means, recursive two-means and farthest-first are established primitives; the paper must position the particular representation and demonstrated trade-off against prior work.

The capacity controls deliberately distinguish farthest-first over **all raw features** (same-K main comparison) from farthest-first over the **original 10% coreset** (capacity curve). Conflating those pools changes the comparison.

## Claims not supported

- No claim of new backbone, new FAISS search algorithm, radius-aware inference or improved accuracy at matched capacity.
- No claim that the center-only bank is a universally effective replacement in all anomaly detectors.
- No claim that the exploratory hybrid is the final main method or fully resolves screw.
- No guarantee of identical speed-ups on another CPU, GPU, thread setting or operating system.
