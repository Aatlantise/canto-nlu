#!/usr/bin/env python3
"""Dependency-free DeepSeek-V4-Flash zero-shot LD evaluation via API."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ld_metrics import LABELS, classification_metrics, read_ld_jsonl, sha256


EVALUATOR_VERSION = "ld-deepseek-v4-api-v1"
API_URL = "https://api.deepseek.com/chat/completions"
PROMPT = """你是一個粵語／書面漢語語言分類器。請判斷下列句子的語言類型。

標籤定義：
- cantonese：整句是香港粵語書面語。
- mandarin：整句是標準書面漢語（普通話／國語書面語）。
- mixed：同一句同時混合粵語和標準書面漢語的詞彙或語法。

繁體、簡體字本身不是判斷標準。只可輸出一個英文標籤：cantonese、mandarin 或 mixed；不要解釋。

句子：{sentence}

標籤："""


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def extract_label(answer: str) -> str | None:
    normalized = answer.strip().lower()
    exact = re.fullmatch(
        r"(?:label\s*:\s*)?(cantonese|mandarin|mixed)[.!。！\s]*", normalized
    )
    if exact:
        return exact.group(1)
    found = re.findall(r"(?<![a-z])(cantonese|mandarin|mixed)(?![a-z])", normalized)
    return found[0] if len(set(found)) == 1 else None


def read_predictions(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    records[int(record["id"])] = record
    return records


def request_prediction(api_key, api_url, model, sentence, max_retries):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(sentence=sentence)}],
        "temperature": 0,
        "max_tokens": 8,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = None
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
            elapsed = time.perf_counter() - started
            answer = result["choices"][0]["message"].get("content") or ""
            usage = result.get("usage") or {}
            return {
                "raw_answer": answer.strip(),
                "prediction": extract_label(answer),
                "latency_seconds": elapsed,
                "served_model": result.get("model"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code in (400, 401, 402, 403, 404, 422):
                raise RuntimeError(f"API HTTP {error.code}: {detail}") from error
            last_error = RuntimeError(f"API HTTP {error.code}: {detail}")
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
        if attempt + 1 < max_retries:
            time.sleep(min(2**attempt, 16) + random.random())
    raise RuntimeError(f"API failed after {max_retries} attempts") from last_error


def main() -> None:
    args = arguments()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    rows = read_ld_jsonl(args.data, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prior = read_predictions(args.output) if args.resume else {}
    mode = "a" if args.resume and args.output.exists() else "w"
    run_started = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    generated = 0

    with args.output.open(mode, encoding="utf-8") as stream:
        for position, row in enumerate(rows, start=1):
            if row["id"] in prior:
                continue
            result = request_prediction(
                api_key, args.api_url, args.model, row["sentence"], args.max_retries
            )
            record = {
                "id": row["id"],
                "source_id": row["source_id"],
                "gold_id": row["label"],
                "gold": LABELS[row["label"]],
                **result,
                "requested_model": args.model,
                "setting": "zero-shot-temperature-0-non-thinking",
                "evaluator_version": EVALUATOR_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            generated += 1
            print(f"[{position}/{len(rows)}] {result['prediction'] or 'INVALID'}", flush=True)

    all_records = read_predictions(args.output)
    selected = [all_records[row["id"]] for row in rows if row["id"] in all_records]
    prompt_tokens = sum(record.get("prompt_tokens") or 0 for record in selected)
    completion_tokens = sum(record.get("completion_tokens") or 0 for record in selected)
    summary = {
        "requested_model": args.model,
        "served_models": sorted(
            {record.get("served_model") for record in selected if record.get("served_model")}
        ),
        "task": "Cantonese/Mandarin/mixed language detection",
        "setting": "zero-shot-temperature-0-non-thinking",
        "primary_metric": "macro_f1",
        "label_mapping": {str(i): label for i, label in enumerate(LABELS)},
        "data_path": str(args.data),
        "data_sha256": sha256(args.data),
        **classification_metrics(selected),
        "generated_this_invocation": generated,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "wall_seconds_this_invocation": time.perf_counter() - wall_started,
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_version": EVALUATOR_VERSION,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Predictions: {args.output.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
