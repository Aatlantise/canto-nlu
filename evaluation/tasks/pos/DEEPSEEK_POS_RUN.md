# DeepSeek-V4-Flash 粤语 POS API 评测

本实验通过 DeepSeek 官方 API 运行，不申请 GPU。请求模型名保留为
`deepseek-v4-flash`，同时在每条记录中保存 API 返回的 `served_model`、
`system_fingerprint` 和时间戳。当前官方服务可能把旧名称路由至更新的 Flash
版本，因此报告时必须同时写明评测日期和 `served_model`。

## 上传与解压

上传 `deepseek_pos_v1_bundle.zip` 至 `/home/USER`：

```bash
cd /home/USER
unzip -o deepseek_pos_v1_bundle.zip
grep EVALUATOR_VERSION deepseek_v4_pos_eval.py
```

## 安全设置 API key

不要把 key 发到聊天中，也不要直接写入脚本：

```bash
read -s -p "DeepSeek API key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
test -n "$DEEPSEEK_API_KEY" && echo "API key is set"
```

## Smoke test

```bash
sbatch run_pos_deepseek_smoke.sbatch
squeue -u "$USER" -o "%.18i %.30j %.2t %.10M %.30R"
```

完成后：

```bash
cat /home/USER/pos-results/deepseek-v4-flash-pos-smoke.summary.json
```

## Full evaluation

在同一个已导出 API key 的 shell 中提交：

```bash
sbatch run_pos_deepseek_full.sbatch
```

日志和结果：

```text
/home/USER/pos-deepseek-full-v1-JOB_ID.out
/home/USER/pos-results/deepseek-v4-flash-pos-full-v1.jsonl
/home/USER/pos-results/deepseek-v4-flash-pos-full-v1.summary.json
```

断点续跑时，在 full sbatch 的 Python 命令末尾增加 `--resume`。
