# mBERT Cantonese POS fine-tuning

Unlike the three generative LLM evaluations, mBERT requires supervised fine-tuning.
The supplied data is used as follows:

- `pos_train.jsonl`: 903 labelled sentences. A group-aware 90/10 split creates
  the training and validation sets with seed 42.
- `pos_test.jsonl`: 101 labelled sentences used only for final evaluation.

Exact duplicate token sequences in the training file are kept in the same split.
The best checkpoint is selected using validation Macro-F1; the supplied test set
is evaluated once.

## Upload and verify

```bash
cd /home/USER
unzip -o mbert_pos_v3_bundle.zip
grep EVALUATOR_VERSION mbert_pos_finetune.py
```

## Smoke test

```bash
sbatch run_mbert_pos_smoke.sbatch
squeue -u "$USER" -o "%.18i %.30j %.2t %.10M %.30R"
```

After completion:

```bash
cat /home/USER/pos-results/mbert-smoke-v3/summary.json
```

## Full training and evaluation

```bash
sbatch run_mbert_pos_full.sbatch
```

Final artifacts:

```text
/home/USER/pos-results/mbert-full-v3/split_manifest.json
/home/USER/pos-results/mbert-full-v3/test_predictions.jsonl
/home/USER/pos-results/mbert-full-v3/summary.json
/home/USER/pos-results/mbert-full-v3/best_model/
```

The split manifest records both input hashes and the exact train/validation IDs.
