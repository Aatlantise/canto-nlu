#!/usr/bin/env python3
"""Dependency-free Cantonese NLI evaluation through the DeepSeek chat API."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


LABELS = ("entailment", "not_entailment")
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
SYSTEM_PROMPT = (
    "你是一個香港粵語自然語言推理分類器。你必須只輸出一個英文標籤，"
    "只可輸出 entailment 或 not_entailment，唔好解釋。"
)
USER_TEMPLATE = """判斷「前提」係咪必然推出「假設」。

entailment：前提必然支持或推出假設。
not_entailment：前提唔能夠必然推出假設，包括矛盾、無關或資料不足。

前提：{premise}
假設：{hypothesis}

只輸出 entailment 或 not_entailment："""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def extract_label(answer):
    normalized = answer.strip().lower().replace("-", "_")
    match = re.fullmatch(
        r"(?:label|標籤)?\s*[:：]?\s*(not_entailment|entailment)[.!。！\s]*",
        normalized,
    )
    if match:
        return match.group(1)
    found = re.findall(r"(?<![a-z_])(not_entailment|entailment)(?![a-z_])", normalized)
    return found[0] if len(set(found)) == 1 else None


def load_rows(path, limit):
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"id", "premise", "hypothesis", "label"} - set(row)
            if missing:
                raise ValueError(f"Line {line_number} is missing {sorted(missing)}")
            if row["label"] not in LABELS:
                raise ValueError(f"Unknown label on line {line_number}: {row['label']}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate id: {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    return rows[:limit] if limit is not None else rows


def read_prior(path):
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[str(record["id"])] = record
    return records


def request_prediction(api_key, api_url, model, row, max_retries):
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
    for attempt in range(max_retries):
        request = urllib.request.Request(
            api_url,
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
                "prediction": extract_label(answer),
                "raw_answer": answer.strip(),
                "latency_seconds": time.perf_counter() - started,
                "served_model": result.get("model"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code in (400, 401, 402, 403, 404, 422):
                raise RuntimeError(f"DeepSeek API HTTP {error.code}: {detail}") from error
            last_error = RuntimeError(f"DeepSeek API HTTP {error.code}: {detail}")
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
        if attempt + 1 < max_retries:
            time.sleep(min(2**attempt, 16) + random.random())
    raise RuntimeError(f"API failed after {max_retries} attempts") from last_error


def summarize(records, model):
    correct = sum(record.get("prediction") == record["gold"] for record in records)
    invalid = sum(record.get("prediction") is None for record in records)
    return {
        "evaluator_version": "cantonese-nli-deepseek-api-v1",
        "requested_model": model,
        "served_models": sorted({str(r.get("served_model")) for r in records}),
        "provider": "DeepSeek official API",
        "setting": "zero-shot temperature-0 non-thinking",
        "primary_metric": "accuracy",
        "samples": len(records),
        "label_counts": dict(Counter(record["gold"] for record in records)),
        "correct": correct,
        "accuracy": correct / len(records),
        "invalid_predictions": invalid,
        "invalid_output_rate": invalid / len(records),
        "prompt_tokens": sum(record.get("prompt_tokens") or 0 for record in records),
        "completion_tokens": sum(record.get("completion_tokens") or 0 for record in records),
        "total_tokens": sum(record.get("total_tokens") or 0 for record in records),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    args = parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    rows = load_rows(args.data, args.max_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prior = read_prior(args.output) if args.resume else {}
    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as stream:
        for index, row in enumerate(rows, start=1):
            if str(row["id"]) in prior:
                continue
            result = request_prediction(
                api_key, args.api_url, args.model, row, args.max_retries
            )
            record = {
                "id": str(row["id"]),
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "gold": row["label"],
                **result,
                "correct": result["prediction"] == row["label"],
                "requested_model": args.model,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(rows)}] {record['prediction'] or 'INVALID'}", flush=True)

    all_records = read_prior(args.output)
    selected = [all_records[str(row["id"])] for row in rows if str(row["id"]) in all_records]
    if len(selected) != len(rows):
        raise RuntimeError(f"Only {len(selected)}/{len(rows)} predictions are present")
    summary = summarize(selected, args.model)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
