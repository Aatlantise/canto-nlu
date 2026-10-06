# Repository inventory

## Export strategy

This is a curated copy, not an in-place reorganization. Keeping the original workspace unchanged protects working experiment commands and makes the GitHub review reversible.

## Included by task

- `sa`: the six requested model-family evaluators, encoder fine-tuning scripts, summarizers, Slurm jobs, run guides, and dependencies. An older DeepSeek-V3/OpenRouter evaluator is retained under `legacy/`.
- `wsd`: DeepSeek, encoder, and local-LLM evaluators plus WSD Slurm jobs.
- `pos`: DeepSeek, Qwen, Gemma, Llama, mBERT, and XLM-R evaluators plus held-out/smoke/full jobs and run guides.
- `laj`: masked-LM and causal-LM surprisal evaluators, all LAJ jobs, and evaluation notes.
- `ld`: shared metrics, prompted/API/encoder evaluators, all LD jobs, manifest, run guide, and dependencies.
- `nli`: encoder, prompted, and DeepSeek evaluators, summarizer, all NLI jobs, run guides, and dependencies.
- `deps`: shared parsing/scoring modules, current stepwise protocol, encoder parser, legacy JSON baseline evaluators, all DEPS jobs, and manifests.
- Local-only, ignored by default: compact LAJ/NLI reports and the `sa`/`deps` legacy evaluators.

## Deliberately excluded

| Source item | Reason |
|---|---|
| `.venv/`, `node_modules/`, `__pycache__/` | Local dependency/cache artifacts |
| `outputs/`, `results/`, `tmp/` | Generated artifacts and duplicate bundles |
| `*.zip` | Duplicate transport bundles or data archives |
| `ld_*.jsonl`, benchmark TSV/JSONL/CSV files | Dataset redistribution needs a separate decision |
| `source/canto-nlu-main/` | Third-party benchmark snapshot; link/citation is safer than vendoring without a license review |
| `paper_text.txt` | Extracted paper text, not source code |
| Gemini evaluator and label-audit helpers | Outside the requested 7-task × 6-model evaluation scope |

## Known portability issue

The Slurm launchers retain personal cluster paths (`/home/USER/...`) and a site-specific Conda setup path. This is intentional for provenance. See `CLUSTER_PORTABILITY.md` before reuse on another cluster.

## Secret review

The source tree uses environment-variable lookups for API credentials. No literal key matching common OpenAI, Hugging Face, Google, or generic bearer-token formats was found during staging. Run a dedicated secret scanner again immediately before publishing.
