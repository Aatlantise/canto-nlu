#!/usr/bin/env python3
"""Dependency-free DeepSeek API evaluation for Cantonese UPOS tagging."""

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
from functools import lru_cache
from pathlib import Path


LABELS = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "VERB",
)
LABEL_SET = set(LABELS)
MODEL = "deepseek-v4-flash"
API_URL = "https://api.deepseek.com/chat/completions"
EVALUATOR_VERSION = "deepseek-v4-pos-zero-shot-v1"
SYSTEM_PROMPT = """You are a Cantonese Universal Dependencies part-of-speech tagger.

The input contains a JSON array of tokens. The tokens have already been segmented.
Assign exactly one UPOS tag to every token.

Allowed tags:
ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, VERB

Return only a valid JSON array of UPOS tags. The output array must have exactly the
same length and order as the input. Do not add explanations, Markdown, token text,
or additional fields.

Required syntax: ["PRON","VERB","PUNCT"]
Wrong syntax: PRON,VERB,PUNCT
Wrong syntax: [PRON, VERB, PUNCT]"""


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_data(path, limit=None):
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No input rows")
    seen = set()
    for row in rows:
        if not {"id", "tokens", "upos"}.issubset(row):
            raise ValueError("Every row requires id, tokens, and upos")
        if row["id"] in seen:
            raise ValueError(f"Duplicate id: {row['id']}")
        seen.add(row["id"])
        if len(row["tokens"]) != len(row["upos"]):
            raise ValueError(f"Token/tag length mismatch in id {row['id']}")
        unknown = set(row["upos"]) - LABEL_SET
        if unknown:
            raise ValueError(f"Unknown gold tags in id {row['id']}: {sorted(unknown)}")
    return rows


def read_predictions(path):
    records = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    records[int(record["id"])] = record
    return records


def strict_parse_answer(answer, expected_length):
    try:
        value = json.loads(answer.strip())
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(value, list) or not all(isinstance(tag, str) for tag in value):
        return None, "invalid_schema"
    tags = [tag.strip().upper() for tag in value]
    if len(tags) != expected_length:
        return None, "wrong_length"
    if any(tag not in LABEL_SET for tag in tags):
        return tags, "unknown_label"
    return tags, "valid"


def split_concatenated_labels(text, expected_length):
    labels = tuple(sorted(LABELS, key=len, reverse=True))

    @lru_cache(maxsize=None)
    def search(position, remaining):
        if position == len(text):
            return ((),) if remaining == 0 else ()
        if remaining == 0:
            return ()
        solutions = []
        for label in labels:
            if text.startswith(label, position):
                for suffix in search(position + len(label), remaining - 1):
                    solutions.append((label,) + suffix)
                    if len(solutions) > 1:
                        return tuple(solutions)
        return tuple(solutions)

    solutions = search(0, expected_length)
    return list(solutions[0]) if len(solutions) == 1 else None


def recover_answer(answer, expected_length):
    strict_tags, strict_status = strict_parse_answer(answer, expected_length)
    if strict_status == "valid":
        return strict_tags, "valid_strict", strict_tags, strict_status

    text = answer.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    bracketed = re.search(r"\[([^\[\]]*)\]", text, flags=re.DOTALL)
    body = bracketed.group(1) if bracketed else text
    if "," in body:
        tags = [part.strip().strip("\"'").upper() for part in body.split(",")]
        tags = [tag for tag in tags if tag]
        method = "recovered_delimited"
    else:
        compact = re.sub(r"[\s\[\]\"'`]", "", body).upper()
        tags = split_concatenated_labels(compact, expected_length)
        method = "recovered_concatenated"
    if not isinstance(tags, list):
        return None, "unrecoverable", strict_tags, strict_status
    if len(tags) != expected_length:
        return None, "wrong_length", strict_tags, strict_status
    if any(tag not in LABEL_SET for tag in tags):
        return tags, "unknown_label", strict_tags, strict_status
    return tags, method, strict_tags, strict_status


def calculate_metrics(records, prediction_key):
    per_label = {label: defaultdict(int) for label in LABELS}
    confusion = {label: Counter() for label in LABELS}
    total_tokens = correct_tokens = 0
    non_punct_tokens = correct_non_punct = 0
    exact_sentences = valid_sentences = 0
    for record in records:
        gold = record["upos"]
        pred = record.get(prediction_key)
        aligned = isinstance(pred, list) and len(pred) == len(gold)
        valid = aligned and all(tag in LABEL_SET for tag in pred)
        valid_sentences += int(valid)
        exact_sentences += int(aligned and pred == gold)
        for index, gold_tag in enumerate(gold):
            pred_tag = pred[index] if aligned else "__INVALID_SENTENCE__"
            total_tokens += 1
            correct_tokens += int(pred_tag == gold_tag)
            if gold_tag != "PUNCT":
                non_punct_tokens += 1
                correct_non_punct += int(pred_tag == gold_tag)
            confusion[gold_tag][pred_tag] += 1
            for label in LABELS:
                if gold_tag == label and pred_tag == label:
                    per_label[label]["tp"] += 1
                elif gold_tag != label and pred_tag == label:
                    per_label[label]["fp"] += 1
                elif gold_tag == label and pred_tag != label:
                    per_label[label]["fn"] += 1
    report, f1_values = {}, []
    for label in LABELS:
        tp, fp, fn = (
            per_label[label]["tp"], per_label[label]["fp"], per_label[label]["fn"]
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(r["upos"].count(label) for r in records),
        }
    sentence_count = len(records)
    invalid_count = sentence_count - valid_sentences
    return {
        "sentences": sentence_count,
        "tokens": total_tokens,
        "token_accuracy": correct_tokens / total_tokens if total_tokens else None,
        "non_punct_tokens": non_punct_tokens,
        "non_punct_accuracy": correct_non_punct / non_punct_tokens if non_punct_tokens else None,
        "macro_f1": sum(f1_values) / len(f1_values),
        "sentence_exact_match": exact_sentences / sentence_count if sentence_count else None,
        "invalid_outputs": invalid_count,
        "invalid_output_rate": invalid_count / sentence_count if sentence_count else None,
        "per_label": report,
        "confusion": {gold: dict(counts) for gold, counts in confusion.items()},
    }


def request_prediction(api_key, model, row, max_retries):
    user_content = json.dumps({"tokens": row["tokens"]}, ensure_ascii=False)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": 512,
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
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
            latency = time.perf_counter() - started
            choice = result["choices"][0]
            answer = choice["message"].get("content") or ""
            prediction, status, strict_prediction, strict_status = recover_answer(
                answer, len(row["tokens"])
            )
            usage = result.get("usage") or {}
            return {
                "raw_answer": answer.strip(),
                "prediction": prediction,
                "parse_status": status,
                "strict_prediction": strict_prediction,
                "strict_parse_status": strict_status,
                "latency_seconds": latency,
                "served_model": result.get("model"),
                "system_fingerprint": result.get("system_fingerprint"),
                "finish_reason": choice.get("finish_reason"),
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
            delay = min(2**attempt, 16) + random.random()
            print(f"retrying in {delay:.1f}s: {last_error}", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"API failed after {max_retries} attempts") from last_error


def main():
    args = arguments()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    rows = read_data(args.data, args.max_samples)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prior = read_predictions(output_path) if args.resume else {}
    mode = "a" if args.resume and output_path.exists() else "w"
    run_started = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    calls = 0
    with output_path.open(mode, encoding="utf-8") as stream:
        for number, row in enumerate(rows, 1):
            row_id = int(row["id"])
            if row_id in prior:
                continue
            print(f"[{number}/{len(rows)}]", end=" ", flush=True)
            result = request_prediction(api_key, args.model, row, args.max_retries)
            record = {
                **row,
                **result,
                "requested_model": args.model,
                "provider": "DeepSeek official API",
                "setting": "zero-shot-temperature-0-non-thinking",
                "evaluator_version": EVALUATOR_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            prior[row_id] = record
            calls += 1
            print(result["parse_status"], flush=True)
    selected = [prior[int(row["id"])] for row in rows]
    elapsed = time.perf_counter() - wall_started
    prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in selected)
    completion_tokens = sum(r.get("completion_tokens") or 0 for r in selected)
    summary = {
        "requested_model": args.model,
        "served_models": sorted({str(r.get("served_model")) for r in selected}),
        "system_fingerprints": sorted(
            {str(r.get("system_fingerprint")) for r in selected}
        ),
        "provider": "DeepSeek official API",
        "setting": "zero-shot-temperature-0-non-thinking",
        "evaluator_version": EVALUATOR_VERSION,
        "metrics_policy": (
            "Top-level metrics use uniquely recoverable label sequences; "
            "strict_json_metrics require an exact JSON array."
        ),
        **calculate_metrics(selected, "prediction"),
        "parse_status_counts": dict(Counter(r["parse_status"] for r in selected)),
        "strict_json_metrics": calculate_metrics(selected, "strict_prediction"),
        "strict_parse_status_counts": dict(
            Counter(r["strict_parse_status"] for r in selected)
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds_this_invocation": elapsed,
        "requests_this_invocation": calls,
        "throughput_samples_per_second_this_invocation": calls / elapsed if elapsed else None,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Predictions: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
