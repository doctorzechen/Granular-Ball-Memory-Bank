# Code origins

| Location | Role and origin |
| --- | --- |
| `src/gb_memory` | Project granular-ball partition implementation, extracted from the research checkout without changing its numerical algorithm; new standalone center-bank API. |
| `src/baselines` | Project implementation of established farthest-first selection and nearest-center real-patch representatives, extracted without changing the controls. Not claimed as invented algorithms. |
| `src/adapters` | Project integration/cache scoring helpers. Feature extraction and search use the retained upstream PatchCore implementation. |
| `experiments` | Project experiment/diagnostic/benchmark runners, renamed and separated from historical Stage 5–8 modules. Shared retrieval helpers are independent of diagnostic scripts. |
| `scripts` | Project report, numerical comparison and archived-result validation utilities. |
| `third_party/patchcore` | Vendored subset of amazon-science/patchcore-inspection. Original Apache-2.0 LICENSE and NOTICE retained; modified files identified in MODIFICATIONS.md. |
| `reference` | Curated small numerical records and asset checksums. Historical local absolute paths are redacted; manuscript, author contact details, datasets and binary caches are excluded. |

The upstream research checkout had no Git metadata. A specific upstream commit is therefore not asserted. Exact retained-source hashes are in `vendor_source_hashes.json`; the full original reproduction package remains the historical source archive.

The new namespace does not turn reused code into original inventions. Conversely, using standard CNN/k-means/search components does not erase the distinct center-memory representation. Method novelty should be judged from the representation, construction rule, related work and experiments—not from the percentage of Python files authored locally.

The `gb_medoid` label follows the experiment archive. Its implementation picks the **observed patch closest to the mean center** in each ball; it is not a general pairwise-distance medoid optimizer.

Only upstream's license is currently supplied. The project owners must confirm a license for their own code before inviting public reuse. Data/checkpoints remain subject to their own licenses.
