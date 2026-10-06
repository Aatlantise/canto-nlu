# 黄色部分：NLI Accuracy 评测

## 评测范围

| 分组 | 模型 | 评测设置 |
|---|---|---|
| fine-tuned encoder | mBERT (`google-bert/bert-base-multilingual-cased`) | 在 `hon9kon9ize/yue-all-nli` 训练集上完整 fine-tune，再测 100 条 sample |
| fine-tuned encoder | XLM-R-base (`FacebookAI/xlm-roberta-base`) | 同上 |
| prompted LLM | Qwen2.5-7B-Instruct | zero-shot、temperature 0 / greedy、4-bit |
| prompted LLM | Gemma-3-12B-it | zero-shot、temperature 0 / greedy、4-bit |
| prompted LLM | DeepSeek-V4-Flash | 官方 API、zero-shot、temperature 0、关闭 thinking |
| prompted LLM | Llama-3.1-8B-Instruct | zero-shot、temperature 0 / greedy、4-bit |

未标黄的 `DeepSeek-R1-Distill-Qwen-14B` 不在本次范围内。

## 输入检查

`sample_nli.jsonl` 共 100 条，字段为 `id / premise / hypothesis / label`：

- `entailment`: 50
- `not_entailment`: 50
- id 与 premise-hypothesis pair 均无重复
- Accuracy 的最小变化单位为 1 个百分点

脚本只把 premise 和 hypothesis 传给模型，不把含有 `_ent` / `_non_ent` 的 id 传入，避免标签泄漏。

## 上传和环境

把 `nli_eval_bundle.zip` 上传到服务器后：

```bash
mkdir -p /home/USER/nli-eval
unzip -o ~/nli_eval_bundle.zip -d /home/USER/nli-eval

source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
python -m pip install -r /home/USER/nli-eval/requirements_nli.txt
```

Gemma 和 Llama 的仓库需要先在 Hugging Face 接受许可，并在服务器登录：

```bash
huggingface-cli login
```

## 先做 smoke test

Prompted 模型可任选一个先测 5 条：

```bash
python /home/USER/nli-eval/nli_prompted_eval.py \
  --data /home/USER/nli-eval/sample_nli.jsonl \
  --output-dir /home/USER/nli-eval/results/qwen-smoke \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-samples 5 \
  --load-in-4bit
```

Encoder 先用小训练子集验证管线；这个 smoke 分数不能填表：

```bash
python /home/USER/nli-eval/nli_encoder_finetune.py \
  --test-data /home/USER/nli-eval/sample_nli.jsonl \
  --output-dir /home/USER/nli-eval/results/mbert-smoke \
  --model google-bert/bert-base-multilingual-cased \
  --max-train-triplets 200 \
  --max-validation-triplets 100 \
  --epochs 1 \
  --bf16
```

## 提交五个 GPU 作业

```bash
cd /home/USER/nli-eval
sbatch run_nli_mbert.sbatch
sbatch run_nli_xlmr.sbatch
sbatch run_nli_qwen.sbatch
sbatch run_nli_gemma.sbatch
sbatch run_nli_llama.sbatch
```

如果 GPU 不支持 BF16，把两个 encoder sbatch 文件末尾的 `--bf16` 改为 `--fp16`。

## DeepSeek-V4-Flash

该脚本只依赖 Python 标准库，可在 Windows PowerShell 或服务器 CPU 节点运行。先在当前 shell 临时设置 key；不要把 key 写入脚本或提交到仓库。

PowerShell：

```powershell
$env:DEEPSEEK_API_KEY = "你的 key"
python nli_deepseek_eval_stdlib.py `
  --data sample_nli.jsonl `
  --output results/deepseek-v4-flash/predictions.jsonl `
  --resume
```

Bash：

```bash
export DEEPSEEK_API_KEY='你的 key'
python nli_deepseek_eval_stdlib.py \
  --data sample_nli.jsonl \
  --output results/deepseek-v4-flash/predictions.jsonl \
  --resume
```

`--resume` 会跳过已经完成的 id，可安全续跑。API 返回的 `served_model` 会写入结果；填表前要确认它确实是目标模型。

## 汇总

所有作业完成后：

```bash
python summarize_nli_results.py \
  --results-root results \
  --csv nli_results.csv \
  --markdown NLI_RESULTS.md
```

表格里的 NLI Acc. 应填 `accuracy_percent`，例如 `0.83` 会显示为 `83.00`。保留每个目录里的 `predictions.jsonl` 和 `summary.json`，它们是逐条审计与复现实验所需的原始证据。

## 口径说明

- encoder 遵循图中 “fine-tuned” 分组：学习率 `2e-5`、batch size `16`、3 epochs，训练数据的每个 triplet 展开为一个 entailment 和一个 not-entailment 样本。
- prompted LLM 不使用 sample 标签做 prompt 选择或 few-shot 示例。
- 4-bit 量化是显存折衷；如果要和 FP16/BF16 报告严格比较，应另跑未量化版本并单独标注。
- 100 条只是 sample 成绩，不应冒充完整 6.6k test split 成绩。
