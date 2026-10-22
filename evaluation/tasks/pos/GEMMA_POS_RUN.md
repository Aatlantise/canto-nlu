# Gemma-3-12B-it 粤语 POS 评测

本实验与 Qwen 使用相同的 prompt、标签集合、解析规则和指标。由于 NVIDIA L4
无法稳妥容纳 Gemma 12B 的 BF16 权重及推理开销，作业使用
bitsandbytes NF4 4-bit，计算类型为 bfloat16。

## 上传与解压

上传 `gemma_pos_v1_bundle.zip` 到 `/home/USER`，然后执行：

```bash
cd /home/USER
unzip -o gemma_pos_v1_bundle.zip
grep EVALUATOR_VERSION gemma3_pos_eval.py
```

应看到：`gemma3-pos-zero-shot-v1`。

## Smoke test

```bash
sbatch run_pos_gemma_smoke.sbatch
squeue -u "$USER" -o "%.18i %.30j %.2t %.10M %.30R"
```

完成后：

```bash
cat /home/USER/pos-results/gemma3-12b-pos-smoke.summary.json
```

## Full evaluation

```bash
sbatch run_pos_gemma_full.sbatch
```

实时日志（把 JOB_ID 替换为实际编号）：

```bash
tr '\r' '\n' < /home/USER/pos-gemma-full-v1-JOB_ID.out | tail -n 10
```

最终文件：

```text
/home/USER/pos-results/gemma3-12b-pos-full-v1.jsonl
/home/USER/pos-results/gemma3-12b-pos-full-v1.summary.json
```
