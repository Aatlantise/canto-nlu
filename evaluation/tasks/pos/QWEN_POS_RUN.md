# Qwen2.5-7B-Instruct 粤语 POS 评测

## 上传

在 Google Cloud 页面点击 `c12-controller` 右侧的 **SSH**。在浏览器 SSH
窗口右上角菜单选择 **Upload file**，上传 `qwen_pos_bundle.zip`。

## 解压并检查数据

```bash
cd /home/USER
unzip -o qwen_pos_bundle.zip
python - <<'PY'
import json
rows = [json.loads(x) for x in open('pos_train.jsonl', encoding='utf-8') if x.strip()]
assert len(rows) == 903
assert all(len(x['tokens']) == len(x['upos']) for x in rows)
print('rows:', len(rows), 'tokens:', sum(len(x['tokens']) for x in rows))
PY
```

期望输出：`rows: 903 tokens: 12139`。

评测同时保存两套结果：顶层指标允许恢复无歧义的逗号标签序列、无引号数组和
连接标签；`strict_json_metrics` 只接受严格 JSON 数组。长度不匹配、非法标签和
无法唯一恢复的输出在两套指标中都记为无效。

## 先提交 10 条 smoke test

```bash
cd /home/USER
sbatch run_pos_qwen_smoke.sbatch
squeue -u "$USER"
```

作业结束后查看：

```bash
cat /home/USER/pos-results/qwen2.5-7b-pos-smoke.summary.json
```

如果作业失败，查看：

```bash
ls -t /home/USER/pos-qwen-smoke-*.out | head -1 | xargs cat
```

## Smoke test 正常后提交全量 903 条

```bash
cd /home/USER
sbatch run_pos_qwen_full.sbatch
squeue -u "$USER"
```

最终结果：

```text
/home/USER/pos-results/qwen2.5-7b-pos-full.jsonl
/home/USER/pos-results/qwen2.5-7b-pos-full.summary.json
```

当前正式版本使用独立文件名，避免覆盖旧版输出：

```text
/home/USER/pos-results/qwen2.5-7b-pos-full-v2.jsonl
/home/USER/pos-results/qwen2.5-7b-pos-full-v2.summary.json
```

## 显存不足时

在两个 `.sbatch` 文件最后的 Python 命令中增加：

```text
--load-in-4bit
```

需要环境已经安装 `bitsandbytes`。正式比较时必须记录是否使用了 4-bit。
