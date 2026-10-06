#!/usr/bin/env python3
"""Frozen zero-shot evaluator for Llama 3.1 8B Instruct on Cantonese SA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


EVALUATOR_VERSION = "llama-3.1-8b-instruct-cantonlu-sa-zero-shot-v1"
LABELS = ("smile", "ok", "cry")
SYSTEM_PROMPT = (
    "你係一個香港粵語情感分類器。嚴格按指示判斷整段評論嘅整體情感。"
    "最後只可以輸出一個英文標籤，唔可以解釋。"
)
USER_TEMPLATE = """判斷以下 OpenRice 香港粵語餐廳評論的整體情感。

標籤定義：
- smile：整體正面
- ok：整體中性，或者正負評價明顯混合
- cry：整體負面

只可輸出 smile、ok 或 cry。

評論：
{text}

標籤："""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def normalize_row(row_id, label, text) -> dict:
    label = str(label).strip()
    if label not in LABELS:
        raise ValueError(f"Unknown label {label!r} at row {row_id}")
    text = str(text)
    return {
        "row_id": row_id,
        "gold": label,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def load_data(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    rows: list[dict] = []
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                item = json.loads(line)
                rows.append(normalize_row(item.get("id", index), item["label"], item["sentence"]))
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for index, item in enumerate(csv.DictReader(stream)):
                rows.append(normalize_row(index, item["Gold Label"], item["Text"]))
    elif suffix == ".tsv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            for index, item in enumerate(csv.DictReader(stream, delimiter="\t")):
                rows.append(normalize_row(index, item["label"], item["text_a"]))
    else:
        raise ValueError("Data must be .jsonl, .csv, or .tsv")
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def extract_label(answer: str) -> str | None:
    normalized = answer.strip().lower()
    exact = re.fullmatch(r"(?:label\s*:\s*)?(smile|ok|cry)[.!。！\s]*", normalized)
    if exact:
        return exact.group(1)
    found = re.findall(r"(?<![a-z])(smile|ok|cry)(?![a-z])", normalized)
    return found[0] if len(set(found)) == 1 else None


def load_completed(path: Path) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = str(item["row_id"])
            if key in completed:
                raise ValueError(f"Duplicate row_id {key!r} in {path}:{line_number}")
            completed[key] = item
    return completed


def build_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kwargs = {"device_map": "auto"}
    if args.load_in_4bit:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        kwargs["torch_dtype"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    return tokenizer, model


def render_messages(text: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_input(text: str, tokenizer, max_input_tokens: int):
    """Truncate only the review, preserving the instructions and generation cue."""
    text_token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    keep = len(text_token_ids)
    was_truncated = False
    while True:
        if keep < len(text_token_ids):
            was_truncated = True
            head = keep // 2
            tail = keep - head
            kept_ids = text_token_ids[:head] + text_token_ids[-tail:]
            review = tokenizer.decode(kept_ids, skip_special_tokens=True)
        else:
            review = text
        rendered = render_messages(review, tokenizer)
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_length = int(inputs["input_ids"].shape[1])
        if input_length <= max_input_tokens:
            return inputs, was_truncated, input_length
        excess = input_length - max_input_tokens
        new_keep = keep - excess - 16
        if new_keep >= keep:
            new_keep = keep - 1
        if new_keep <= 0:
            raise ValueError("max-input-tokens is too small for the prompt instructions")
        keep = new_keep


def predict(
    text: str, tokenizer, model, max_input_tokens: int
) -> tuple[str, str | None, bool, int]:
    inputs, was_truncated, input_length = prepare_input(text, tokenizer, max_input_tokens)
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer, extract_label(answer), was_truncated, input_length


def summarize(rows: list[dict], model_name: str, elapsed_seconds: float) -> dict:
    gold = [row["gold"] for row in rows]
    predictions = [row.get("prediction") for row in rows]
    scored_predictions = [value if value in LABELS else "__invalid__" for value in predictions]
    invalid_count = sum(value not in LABELS for value in predictions)
    report = classification_report(
        gold,
        scored_predictions,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "model": model_name,
        "protocol": "frozen zero-shot; deterministic greedy decoding; test labels used only for scoring",
        "labels": list(LABELS),
        "samples": len(rows),
        "valid_predictions": len(rows) - invalid_count,
        "invalid_predictions": invalid_count,
        "invalid_output_rate": invalid_count / len(rows),
        "accuracy": accuracy_score(gold, scored_predictions),
        "macro_f1": f1_score(
            gold, scored_predictions, labels=list(LABELS), average="macro", zero_division=0
        ),
        "classification_report": report,
        "confusion_matrix_rows_gold_columns_prediction": confusion_matrix(
            gold, scored_predictions, labels=list(LABELS)
        ).tolist(),
        "elapsed_seconds": elapsed_seconds,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load_data(args.data)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(args.output) if args.resume else {}
    expected = {str(row["row_id"]): row for row in data}
    unexpected = set(completed) - set(expected)
    if unexpected:
        raise ValueError(f"Resume file contains row IDs not in current data: {sorted(unexpected)[:5]}")
    for key, item in completed.items():
        if item.get("text_sha256") != expected[key]["text_sha256"]:
            raise ValueError(f"Resume hash mismatch for row_id {key}")

    pending = [row for row in data if str(row["row_id"]) not in completed]
    started = time.time()
    tokenizer = model = None
    if pending:
        tokenizer, model = build_model(args)
        mode = "a" if args.resume and args.output.exists() else "w"
        with args.output.open(mode, encoding="utf-8") as stream:
            for row in tqdm(pending, desc="Evaluating"):
                item_started = time.time()
                answer, prediction, was_truncated, input_tokens = predict(
                    row["text"], tokenizer, model, args.max_input_tokens
                )
                record = {
                    "row_id": row["row_id"],
                    "text_sha256": row["text_sha256"],
                    "gold": row["gold"],
                    "prediction": prediction,
                    "raw_answer": answer,
                    "input_tokens": input_tokens,
                    "input_truncated": was_truncated,
                    "latency_seconds": time.time() - item_started,
                    "model": args.model,
                    "evaluator_version": EVALUATOR_VERSION,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()

    completed = load_completed(args.output)
    ordered = [completed[str(row["row_id"])] for row in data]
    summary = summarize(ordered, args.model, time.time() - started)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
