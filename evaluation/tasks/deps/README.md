# Dependency parsing (DEPS)

The current protocol uses `encoder_deps_finetune.py` for supervised encoder parsing and `deps_stepwise_eval.py` plus `deps_stepwise_common.py` for the three-step generative protocol. `deps_stepwise_rescore.py` can reparse saved raw answers without new model calls.

The `legacy/` directory preserves the earlier single-turn JSON baseline and its launchers. It is retained for history but should not be mixed with the current stepwise results.

See `DEPS_ALL_MODELS_MANIFEST.md` and `DEPS_STEPWISE_RUN.md` for the reporting split between supervised and prompted systems.
