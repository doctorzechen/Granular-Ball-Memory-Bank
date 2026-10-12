# Historical research

The original complete `reproduction_code` package is preserved separately, unchanged. It contains the earlier Stage 5–8 research modules, large frozen caches, data, weights, and historical evidence.

This publication directory does not copy that archive or import its Python modules. Supply it as `--assets-root` to use its read-only data, checkpoint and frozen features. New code imports only `gb_memory`, `baselines`, `adapters`, `experiments`, and the explicitly attributed vendored subset.

If you later copy an archive here, place it under `legacy/archive/` (ignored by Git). It is not a runtime dependency.
