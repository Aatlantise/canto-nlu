#!/usr/bin/env python3
"""Dependency-free DeepSeek V4 zero-shot evaluation on CantoNLU WSD JSONL."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("similar", "not_similar")
MODEL = "deepseek-v4-flash"
API_URL = "https://api.deepseek.com/chat/completions"
SYSTEM_PROMPT = (
    "你是一個香港粵語詞義辨析分類器。你必須只輸出一個標籤："
    "similar 或 not_similar，不要解釋。"
)
USER_TEMPLATE = """判斷目標詞在兩個句子中的詞義是否相同。
similar：目標詞在兩句中的詞義相同。
not_similar：目標詞在兩句中的詞義不同。

目標詞：{target}
句子一：{sentence1}
句子二：{sentence2}

只輸出 similar 或 not_similar："""


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--samples-per-label", type=int)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def extract_label(answer):
    value = answer.strip().lower().replace("-", "_")
    exact = re.fullmatch(
        r"(?:label\s*:\s*)?(not_similar|similar)[.!。！\s]*", value
    )
    if exact:
        return exact.group(1)
    found = re.findall(r"(?<![a-z_])(not_similar|similar)(?![a-z_])", value)
    return found[0] if len(set(found)) == 1 else None


def load_rows(path, limit, samples_per_label):
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    required = {"id", "target", "sentence1", "sentence2", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"JSONL requires {sorted(required)}")
    unknown = sorted({row["label"] for row in rows} - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    if samples_per_label:
        selected = []
        for name in LABELS:
            group = [row for row in rows if row["label"] == name]
            if len(group) < samples_per_label:
                raise ValueError(f"Only {len(group)} examples available for {name}")
            selected.extend(group[:samples_per_label])
        rows = sorted(selected, key=lambda row: int(row["id"]))
    return rows[:limit] if limit else rows


def read_jsonl(path):
    records = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    records[int(record["id"])] = record
    return records


def request_prediction(api_key, model, row, max_retries):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(**row)},
        ],
        "temperature": 0,
        "max_tokens": 8,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = None
    for attempt in range(max_retries):
        request = urllib.request.Request(
            API_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            answer = result["choices"][0]["message"].get("content") or ""
            usage = result.get("usage") or {}
            return {
                "raw_answer": answer.strip(),
                "prediction": extract_label(answer),
                "latency_seconds": time.perf_counter() - started,
                "served_model": result.get("model"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code in (400, 401, 402, 422):
                raise RuntimeError(f"DeepSeek API HTTP {error.code}: {detail}") from error
            last_error = RuntimeError(f"DeepSeek API HTTP {error.code}: {detail}")
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
        if attempt + 1 < max_retries:
            time.sleep(min(2**attempt, 16) + random.random())
    raise RuntimeError(f"API failed after {max_retries} attempts") from last_error


def metrics(records):
    counts = {name: defaultdict(int) for name in LABELS}
    invalid = sum(row.get("prediction") is None for row in records)
    for row in records:
        gold, pred = row["label"], row.get("prediction")
        for name in LABELS:
            if gold == name and pred == name: counts[name]["tp"] += 1
            elif gold != name and pred == name: counts[name]["fp"] += 1
            elif gold == name and pred != name: counts[name]["fn"] += 1
    report, f1s = {}, []
    for name in LABELS:
        tp, fp, fn = counts[name]["tp"], counts[name]["fp"], counts[name]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(row["label"] == name for row in records)}
        f1s.append(f1)
    return {
        "invalid_predictions": invalid,
        "invalid_output_rate": invalid / len(records),
        "accuracy": sum(row["label"] == row.get("prediction") for row in records) / len(records),
        "macro_f1": sum(f1s) / len(f1s),
        "classification_report": report,
    }


def main():
    args = arguments()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    rows = load_rows(args.data, args.max_samples, args.samples_per_label)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    prior = read_jsonl(output) if args.resume else {}
    mode = "a" if args.resume and output.exists() else "w"
    run_started = datetime.now(timezone.utc); wall_started = time.perf_counter(); calls = 0
    with output.open(mode, encoding="utf-8") as stream:
        for number, row in enumerate(rows, 1):
            row_id = int(row["id"])
            if row_id in prior: continue
            print(f"[{number}/{len(rows)}]", end=" ", flush=True)
            result = request_prediction(api_key, args.model, row, args.max_retries)
            record = {**row, **result, "requested_model": args.model, "provider": "DeepSeek official API", "setting": "zero-shot-temperature-0-non-thinking", "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n"); stream.flush()
            prior[row_id] = record; calls += 1
            print(result["prediction"] or "INVALID", flush=True)
    elapsed = time.perf_counter() - wall_started
    selected = [prior[int(row["id"])] for row in rows]
    prompt_tokens = sum(row.get("prompt_tokens") or 0 for row in selected)
    completion_tokens = sum(row.get("completion_tokens") or 0 for row in selected)
    summary = {
        "requested_model": args.model,
        "served_models": sorted({row.get("served_model") for row in selected}),
        "provider": "DeepSeek official API",
        "setting": "zero-shot-temperature-0-non-thinking",
        "samples": len(selected),
        "class_distribution": dict(Counter(row["label"] for row in selected)),
        "prediction_distribution": dict(Counter(str(row.get("prediction")) for row in selected)),
        **metrics(selected),
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds_this_invocation": elapsed,
        "requests_this_invocation": calls,
        "throughput_samples_per_second_this_invocation": calls / elapsed if elapsed else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Predictions: {output}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
