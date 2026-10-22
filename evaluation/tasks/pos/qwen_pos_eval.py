#!/usr/bin/env python3
"""Zero-shot Cantonese UPOS evaluation for Qwen2.5-Instruct models."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = (
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "VERB",
)
LABEL_SET = set(LABELS)
EVALUATOR_VERSION = "qwen-pos-zero-shot-v2"
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="JSONL with id, tokens, and upos")
    parser.add_argument("--output", required=True, help="Prediction JSONL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
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


def load_model(model_id, load_in_4bit):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"device_map": "auto"}
    precision = "auto"
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        precision = f"bitsandbytes-nf4-{str(compute_dtype).replace('torch.', '')}"
    else:
        kwargs["torch_dtype"] = "auto"
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    if not load_in_4bit:
        precision = str(model.dtype).replace("torch.", "")
    return tokenizer, model, precision, time.perf_counter() - started


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
    """Return a unique segmentation of concatenated UPOS labels, if one exists."""
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
        recovery_method = "recovered_delimited"
    else:
        compact = re.sub(r"[\s\[\]\"'`]", "", body).upper()
        tags = split_concatenated_labels(compact, expected_length)
        recovery_method = "recovered_concatenated"

    if not isinstance(tags, list):
        return None, "unrecoverable", strict_tags, strict_status
    if len(tags) != expected_length:
        return None, "wrong_length", strict_tags, strict_status
    if any(tag not in LABEL_SET for tag in tags):
        return tags, "unknown_label", strict_tags, strict_status
    return tags, recovery_method, strict_tags, strict_status


def predict(row, tokenizer, model, max_input_tokens, max_new_tokens):
    user_content = json.dumps({"tokens": row["tokens"]}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - started
    generated = output[0, model_inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    tags, status, strict_tags, strict_status = recover_answer(
        answer, len(row["tokens"])
    )
    return answer, tags, status, strict_tags, strict_status, latency


def calculate_metrics(records, prediction_key):
    per_label = {label: defaultdict(int) for label in LABELS}
    confusion = {label: Counter() for label in LABELS}
    total_tokens = correct_tokens = 0
    non_punct_tokens = correct_non_punct = 0
    exact_sentences = 0
    valid_sentences = 0

    for record in records:
        gold = record["upos"]
        pred = record.get(prediction_key)
        aligned = isinstance(pred, list) and len(pred) == len(gold)
        valid = aligned and all(tag in LABEL_SET for tag in pred)
        valid_sentences += int(valid)
        sentence_correct = aligned and pred == gold
        exact_sentences += int(sentence_correct)

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

    report = {}
    f1_values = []
    for label in LABELS:
        tp = per_label[label]["tp"]
        fp = per_label[label]["fp"]
        fn = per_label[label]["fn"]
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
        "non_punct_accuracy": (
            correct_non_punct / non_punct_tokens if non_punct_tokens else None
        ),
        "macro_f1": sum(f1_values) / len(f1_values),
        "sentence_exact_match": exact_sentences / sentence_count if sentence_count else None,
        "invalid_outputs": invalid_count,
        "invalid_output_rate": invalid_count / sentence_count if sentence_count else None,
        "per_label": report,
        "confusion": {gold: dict(counts) for gold, counts in confusion.items()},
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_data(args.data, args.max_samples)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prior = read_predictions(output_path) if args.resume else {}
    mode = "a" if args.resume and output_path.exists() else "w"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    run_started = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    tokenizer, model, precision, load_seconds = load_model(
        args.model, args.load_in_4bit
    )
    inference_seconds = 0.0
    generated_count = 0

    with output_path.open(mode, encoding="utf-8") as stream:
        for row in tqdm(rows, desc="POS evaluation"):
            row_id = int(row["id"])
            if row_id in prior:
                continue
            answer, prediction, status, strict_prediction, strict_status, latency = predict(
                row,
                tokenizer,
                model,
                args.max_input_tokens,
                args.max_new_tokens,
            )
            inference_seconds += latency
            generated_count += 1
            record = {
                **row,
                "prediction": prediction,
                "parse_status": status,
                "strict_prediction": strict_prediction,
                "strict_parse_status": strict_status,
                "raw_answer": answer,
                "latency_seconds": latency,
                "model": args.model,
                "precision": precision,
                "seed": args.seed,
                "setting": "zero-shot-greedy-strict-json",
                "evaluator_version": EVALUATOR_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            prior[row_id] = record

    selected = [prior[int(row["id"])] for row in rows]
    wall_seconds = time.perf_counter() - wall_started
    recovered_metrics = calculate_metrics(selected, "prediction")
    strict_metrics = calculate_metrics(selected, "strict_prediction")
    summary = {
        "model": args.model,
        "precision": precision,
        "setting": "zero-shot-greedy-strict-json",
        "evaluator_version": EVALUATOR_VERSION,
        "seed": args.seed,
        "metrics_policy": (
            "Top-level metrics use uniquely recoverable label sequences; "
            "strict_json_metrics require an exact JSON array."
        ),
        **recovered_metrics,
        "parse_status_counts": dict(Counter(r["parse_status"] for r in selected)),
        "strict_json_metrics": strict_metrics,
        "strict_parse_status_counts": dict(
            Counter(r["strict_parse_status"] for r in selected)
        ),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        ),
        "model_load_seconds": load_seconds,
        "inference_seconds_this_invocation": inference_seconds,
        "wall_time_seconds_this_invocation": wall_seconds,
        "samples_generated_this_invocation": generated_count,
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
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
