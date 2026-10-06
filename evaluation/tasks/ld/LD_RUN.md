# LD（Language Detection）评测

本包覆盖截图中标黄的五个模型：mBERT、Qwen2.5-7B-Instruct、
Gemma-3-12B-it、DeepSeek-V4-Flash、Llama-3.1-8B-Instruct。

## 统一任务定义

- `0 = cantonese`：纯香港粤语书面语
- `1 = mandarin`：纯标准书面汉语
- `2 = mixed`：同一句混合粤语与标准书面汉语
- 主指标：三类 `Macro-F1`；无效生成按错误预测计入
- mBERT：使用 train 训练、val 按 Macro-F1 选 checkpoint、test 只作最终评测
- 其余模型：对 test 做 zero-shot、greedy/temperature 0 评测

数据规模为 train 42,753、val 2,425、test 2,341；三个 split 的
`source_id` 无交集。脚本会再次验证 mBERT 三个 split 无 source 泄漏。

## 上传和解压

```bash
mkdir -p /home/USER/ld-eval
cd /home/USER/ld-eval
unzip -o /home/USER/ld_eval_bundle.zip
python -m py_compile *.py
```

环境应包含 `torch transformers accelerate datasets scikit-learn`；Gemma 的
4-bit 运行还需要 `bitsandbytes`。Llama 是 gated 模型，首次运行前需准备
`HF_TOKEN` 或执行 Hugging Face 登录。

## 先做 smoke test

下面命令都只跑少量样本，不会改写正式输出路径：

```bash
python mbert_ld_finetune.py --data-dir . --output-dir results/smoke-mbert \
  --limit-train 96 --limit-eval 24 --epochs 1 --batch-size 8

python ld_prompted_eval.py --data ld_test.jsonl \
  --output results/smoke-qwen.jsonl --model Qwen/Qwen2.5-7B-Instruct \
  --family causal --batch-size 4 --limit 20

python ld_prompted_eval.py --data ld_test.jsonl \
  --output results/smoke-gemma.jsonl --model google/gemma-3-12b-it \
  --family gemma3 --batch-size 2 --load-in-4bit --limit 20

python ld_prompted_eval.py --data ld_test.jsonl \
  --output results/smoke-llama.jsonl \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --family causal --batch-size 4 --limit 20

python deepseek_v4_ld_eval_stdlib.py --data ld_test.jsonl \
  --output results/smoke-deepseek.jsonl --model deepseek-v4-flash --limit 5
```

DeepSeek 运行前设置密钥：

```bash
export DEEPSEEK_API_KEY='你的密钥'
```

如果 API 实际开放的 model id 与表格名称不同，可在命令中改 `--model`；
结果会同时记录 requested model 和服务端返回的 served model，避免误报。

## 正式提交

```bash
cd /home/USER/ld-eval
sbatch run_ld_mbert.sbatch
sbatch run_ld_qwen.sbatch
sbatch run_ld_gemma.sbatch
sbatch run_ld_llama.sbatch
sbatch run_ld_deepseek.sbatch
squeue -u "$USER"
```

DeepSeek 的 `DEEPSEEK_API_KEY` 必须能传入 Slurm 作业环境。若集群不允许计算
节点访问公网，应在允许联网的机器直接运行同一 Python 命令。

## 结果

所有结果在 `/home/USER/ld-eval/results/`。每个 prompted 模型都会生成
`predictions.jsonl` 和 `predictions.summary.json`；mBERT 生成
`results/mbert/summary.json`、逐条预测和最佳模型。填表时取 summary 中的
`macro_f1 * 100`（mBERT 取 `test_metrics.test_macro_f1 * 100`）。

四个 prompted 脚本支持 `--resume`。同一个输出文件不可混用不同模型、数据或
prompt；若要改变这些设置，请使用新的输出路径。
