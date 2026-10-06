# Cluster portability checklist

The `.sbatch` files capture the original experiment environment and are not yet cluster-neutral.

Before submitting a job elsewhere, review:

1. `#SBATCH --partition`, GPU, CPU, memory, and wall-time requests.
2. `#SBATCH --output` paths.
3. The site-specific Conda initialization and environment name.
4. `/home/USER/...` code, dataset, and result paths.
5. Availability of gated Hugging Face models and `HF_TOKEN`.
6. `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY` for API runs.
7. CUDA, PyTorch, Transformers, and bitsandbytes compatibility on the target GPU.

For a portable follow-up, parameterize each launcher with variables such as `REPO_ROOT`, `DATA_ROOT`, and `RESULTS_ROOT`, while preserving the original launchers in version history for reproducibility.
