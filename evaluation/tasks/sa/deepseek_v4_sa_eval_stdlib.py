#!/usr/bin/env python3
"""Dependency-free DeepSeek-V4-Flash OpenRice SA evaluation for restricted Windows PCs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


LABELS = ("smile", "ok", "cry")
MODEL = "deepseek-v4-flash"
API_URL = "https://api.deepseek.com/chat/completions"
SYSTEM_PROMPT = (
    "你是一個香港粵語文本分類器。你必須嚴格遵守輸出格式，"
    "只輸出一個英文標籤，不要解釋。"
)
USER_TEMPLATE = """判斷以下 OpenRice 香港粵語餐廳評論的整體情感。

標籤定義：
- smile：正面
- ok：中性或好壞參半
- cry：負面

只可輸出 smile、ok 或 cry。

評論：
{text}

標籤："""


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def extract_label(answer):
    normalized = answer.strip().lower()
    exact = re.fullmatch(r"(?:label\s*:\s*)?(smile|ok|cry)[.!。！\s]*", normalized)
    if exact:
        return exact.group(1)
    found = re.findall(r"(?<![a-z])(smile|ok|cry)(?![a-z])", normalized)
    return found[0] if len(set(found)) == 1 else None


def load_tsv(path, limit):
    with open(path, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or not {"label", "text_a"}.issubset(rows[0]):
        raise ValueError("TSV must contain label and text_a columns")
    unknown = sorted({row["label"] for row in rows} - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return rows[:limit] if limit is not None else rows


def read_jsonl(path):
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[int(record["row_id"])] = record
    return records


def request_prediction(api_key, model, text, max_retries):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=text)},
        ],
        "temperature": 0,
        "max_tokens": 8,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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
            latency = time.perf_counter() - started
            answer = result["choices"][0]["message"].get("content") or ""
            usage = result.get("usage") or {}
            return {
                "raw_answer": answer.strip(),
                "prediction": extract_label(answer),
                "latency_seconds": latency,
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
    counts = {label: defaultdict(int) for label in LABELS}
    correct = 0
    invalid = 0
    for record in records:
        gold, pred = record["gold"], record.get("prediction")
        correct += pred == gold
        invalid += pred is None
        for label in LABELS:
            if gold == label and pred == label:
                counts[label]["tp"] += 1
            elif gold != label and pred == label:
                counts[label]["fp"] += 1
            elif gold == label and pred != label:
                counts[label]["fn"] += 1

    report = {}
    f1_values = []
    for label in LABELS:
        tp, fp, fn = counts[label]["tp"], counts[label]["fp"], counts[label]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": sum(r["gold"] == label for r in records),
        }
    return {
        "valid_predictions": len(records) - invalid,
        "invalid_predictions": invalid,
        "invalid_output_rate": invalid / len(records) if records else 0.0,
        "accuracy": correct / len(records) if records else None,
        "macro_f1": sum(f1_values) / len(f1_values) if records else None,
        "classification_report": report,
    }


def main():
    args = arguments()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    rows = load_tsv(args.data, args.max_samples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = read_jsonl(output) if args.resume else {}
    mode = "a" if args.resume and output.exists() else "w"
    invocation_count = 0
    run_started = datetime.now(timezone.utc)
    wall_started = time.perf_counter()

    with output.open(mode, encoding="utf-8") as stream:
        for row_id, row in enumerate(rows):
            if row_id in prior:
                continue
            print(f"[{row_id + 1}/{len(rows)}]", end=" ", flush=True)
            result = request_prediction(
                api_key, args.model, row["text_a"], args.max_retries
            )
            record = {
                "row_id": row_id,
                "gold": row["label"],
                **result,
                "requested_model": args.model,
                "provider": "DeepSeek official API",
                "setting": "zero-shot-temperature-0-non-thinking",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            invocation_count += 1
            print(result["prediction"] or "INVALID", flush=True)

    elapsed = time.perf_counter() - wall_started
    all_records = read_jsonl(output)
    selected = [all_records[i] for i in range(len(rows)) if i in all_records]
    prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in selected)
    completion_tokens = sum(r.get("completion_tokens") or 0 for r in selected)
    summary = {
        "requested_model": args.model,
        "served_models": sorted({r.get("served_model") for r in selected}),
        "provider": "DeepSeek official API",
        "samples": len(selected),
        **metrics(selected),
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds_this_invocation": elapsed,
        "requests_this_invocation": invocation_count,
        "throughput_samples_per_second_this_invocation": (
            invocation_count / elapsed if elapsed else None
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "pricing_reference_usd_per_million_input_cache_miss": 0.14,
        "pricing_reference_usd_per_million_output": 0.28,
        "estimated_upper_bound_cost_usd": prompt_tokens / 1_000_000 * 0.14
        + completion_tokens / 1_000_000 * 0.28,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Predictions: {output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
