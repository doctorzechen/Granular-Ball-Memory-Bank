# Release validation — 2026-10-12

Validation applies to the reorganized source release, not a new set of paper claims. The complete original `reproduction_code` package was treated as read-only and remains separate.

## Executed checks

- Independent core, scoring/cache contracts, hybrid behavior, CLI portability, offline wheel contents, required-array resume integrity, capacity-index validation, release-manifest coverage and rejection of failed timing stages passed **68 tests**. The compact result is recorded in `validation_summary.json`; the 731-check contract report is in `release_contract_audit.json`.
- Refactoring audit: **731 checks passed**, covering 15 categories, four banks, source-row/center contracts, sampled saved image scores and internal map consistency. Of these, 41 are source AST comparisons. Two runner comparisons explicitly exclude the newly added checkpoint-integrity calls; their bank/scoring calculations remain unchanged. New operational guards are tested separately.
- Largest sampled saved-score difference: `1.1444091796875e-05`; largest internal map difference: `1.3768672943115234e-05`. Tests use absolute and relative tolerance `1e-5`; these are floating-point comparisons, not a claim of bit-identical whole-run output.
- Curated archived numerical records: **686 checks passed**. The lightweight reference records also matched the previous actual 15-category run in **412 read-only comparisons**. This checks preserved evidence, not a fresh rerun of all 15 categories.
- Fresh toothbrush verification used 60 training images and 42 test images, checked 132 dataset PNG hashes and the exact V1 weight checksum, and actually extracted image features. Bank construction, detection, 1+5 construction-timing repeats, hybrid and image-retrieval stages were executed and then resumed. The clean accuracy/report run completed **34 reference comparisons**, all passing. Absent query times stay null in its report.
- Offline wheel construction retained the core, adapter, vendored modules, LICENSE, NOTICE and MODIFICATIONS, without data/cache/weight binaries. Core import does not import PatchCore or FAISS.

## Timing result that was not accepted

The 32-repeat query benchmark was executed, but repeated checks failed the original timing-quality gates (CV and/or same-capacity timing spread). Bank-size contracts and distance/neighbor checksums were stable. These measurements were **not** accepted as valid speed-up evidence, and the report refused them. Thresholds were not relaxed; archived paper timings were not replaced. The entry point now also refuses to mark a failed query-quality stage complete.

The accuracy-only completed run is distinct from the directory containing failed timing measurements. This avoids reporting a full all-stage pass that did not occur.

## Scope not claimed

This release verification did not freshly extract features or repeat construction/query timing for all 15 categories, rerun every historical research stage, or validate another computer. The optional refactoring validator requires the preserved complete archive, including prior generated score arrays. New source-only users separately obtain data and weights.

The Windows environment showed a native FAISS-loader failure for plain pytest startup. The supplied `scripts/run_tests.py` initializes the runtime before pytest and disables unrelated plugin autoload; this tested entry point does not change any installed packages.

## Source integrity

`manifest.json` hashes the public source and curated records. `docs/vendor_source_hashes.json` identifies the exact retained vendor files and compares them with the previously inspected upstream source. Neither manifest includes external assets or claims an unavailable upstream Git commit.
