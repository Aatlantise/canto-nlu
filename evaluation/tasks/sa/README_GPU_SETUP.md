# 学校 GPU：Qwen2.5-7B-Instruct 运行指南

本指南先运行 CantoNLU 的 OpenRice Cantonese sentiment analysis (SA) 测试集。

## 1. 连接服务器

从 Google Cloud 页面点击 `c12-controller` 右侧的 **SSH**。进入黑色终端后，不要立即下载模型，先执行：

```bash
hostname
whoami
nvidia-smi
command -v srun
command -v sbatch
```

- 如果 `nvidia-smi` 显示 GPU：当前机器可能是 GPU 节点，但仍须遵守学校使用规则。
- 如果 `nvidia-smi` 报错，而 `srun`/`sbatch` 有路径：这是 Slurm 登录/控制节点，需要申请 GPU。
- 如果三者都没有：把完整输出发给我，学校可能使用其他调度系统。

不要在 controller/login 节点直接运行模型，除非管理员明确允许。

## 2. Slurm 交互式申请 GPU（如学校使用 Slurm）

先查看可用分区：

```bash
sinfo
```

常见申请方式如下；`gpu` 分区名和时间限制需要按学校配置修改：

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --pty bash
```

进入计算节点后再次确认：

```bash
hostname
nvidia-smi
```

如果学校要求 account：

```bash
srun --account=YOUR_ACCOUNT --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --pty bash
```

## 3. 上传项目

在 Windows PowerShell 上传时，需要学校提供 SSH 主机名和用户名：

```powershell
scp "C:\path\to\downloads\canto-nlu-main.zip" USER@HOST:~/
scp "C:\path\to\LLM-evaluation\qwen_sa_eval.py" USER@HOST:~/
```

如果浏览器 SSH 页面提供 **Upload file**，也可上传这两个文件。

服务器端解压：

```bash
unzip -q ~/canto-nlu-main.zip -d ~/cantonlu
```

## 4. 建立 Python 环境

如果服务器有 Conda：

```bash
conda create -n cantonlu-qwen python=3.10 -y
conda activate cantonlu-qwen
python -m pip install --upgrade pip
python -m pip install "torch>=2.3" "transformers>=4.37" accelerate pandas scikit-learn tqdm
```

如果没有 Conda：

```bash
python3 -m venv ~/venvs/cantonlu-qwen
source ~/venvs/cantonlu-qwen/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.3" "transformers>=4.37" accelerate pandas scikit-learn tqdm
```

16 GB 显存需要尝试 4-bit：

```bash
python -m pip install bitsandbytes
```

## 5. 先跑 10 条 smoke test

FP16/BF16（通常适合 24 GB 或更大显存）：

```bash
python ~/qwen_sa_eval.py \
  --data ~/cantonlu/canto-nlu-main/openrice-senti/test.tsv \
  --output ~/qwen-results/qwen2.5-7b-instruct-sa-smoke.jsonl \
  --max-samples 10
```

16 GB 显存：

```bash
python ~/qwen_sa_eval.py \
  --data ~/cantonlu/canto-nlu-main/openrice-senti/test.tsv \
  --output ~/qwen-results/qwen2.5-7b-instruct-sa-smoke.jsonl \
  --max-samples 10 \
  --load-in-4bit
```

确认输出中有 accuracy、macro-F1 和 invalid-output rate 后，再跑完整测试集：

```bash
python ~/qwen_sa_eval.py \
  --data ~/cantonlu/canto-nlu-main/openrice-senti/test.tsv \
  --output ~/qwen-results/qwen2.5-7b-instruct-sa-full.jsonl
```

## 6. 实验定义

- 模型：`Qwen/Qwen2.5-7B-Instruct`
- 设置：zero-shot, deterministic greedy decoding
- 标签：`smile`, `ok`, `cry`
- 主指标：macro-F1（与论文 SA 指标一致）
- 同时记录：accuracy、invalid-output rate、每条原始回答
- `test.tsv` 只用于最终评测，不用于选择 prompt 或 few-shot 示例

这属于生成式 zero-shot LLM 评测；不能声称完全复现论文中的 BERT fine-tuning 设置。
