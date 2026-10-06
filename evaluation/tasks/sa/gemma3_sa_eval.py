#!/usr/bin/env python3
"""Zero-shot CantoNLU OpenRice SA evaluation for Gemma 3 IT in 4-bit."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration


LABELS = ("smile", "ok", "cry")
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def extract_label(answer):
    normalized = answer.strip().lower()
    exact = re.fullmatch(r"(?:label\s*:\s*)?(smile|ok|cry)[.!。！\s]*", normalized)
    if exact:
        return exact.group(1)
    found = re.findall(r"(?<![a-z])(smile|ok|cry)(?![a-z])", normalized)
    return found[0] if len(set(found)) == 1 else None


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


def load_model(model_id):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    load_started = time.perf_counter()
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, time.perf_counter() - load_started


def predict(text, model, processor, max_input_tokens):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": USER_TEMPLATE.format(text=text)}],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    latency = time.perf_counter() - started
    generated = output[0, inputs["input_ids"].shape[-1] :]
    answer = processor.decode(generated, skip_special_tokens=True).strip()
    return answer, extract_label(answer), latency


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = pd.read_csv(args.data, sep="\t", encoding="utf-8")
    if args.max_samples is not None:
        data = data.head(args.max_samples)

    torch.cuda.reset_peak_memory_stats()
    run_started = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    model, processor, load_seconds = load_model(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prior = read_jsonl(output_path) if args.resume else {}
    mode = "a" if args.resume and output_path.exists() else "w"
    inference_seconds = 0.0
    generated_count = 0

    with output_path.open(mode, encoding="utf-8") as stream:
        for row_id, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating"):
            row_id = int(row_id)
            if row_id in prior:
                continue
            answer, prediction, latency = predict(
                str(row["text_a"]), model, processor, args.max_input_tokens
            )
            inference_seconds += latency
            generated_count += 1
            record = {
                "row_id": row_id,
                "gold": str(row["label"]),
                "prediction": prediction,
                "raw_answer": answer,
                "latency_seconds": latency,
                "model": args.model,
                "quantization": "bitsandbytes-nf4-4bit-double-quant",
                "load_in_4bit": True,
                "compute_dtype": "bfloat16",
                "seed": args.seed,
                "setting": "zero-shot-greedy",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

    records = read_jsonl(output_path)
    selected = [records[int(i)] for i in data.index if int(i) in records]
    gold = [r["gold"] for r in selected]
    predictions = [r.get("prediction") for r in selected]
    scored = [p if p is not None else "__invalid__" for p in predictions]
    invalid = sum(p is None for p in predictions)
    wall_seconds = time.perf_counter() - wall_started
    summary = {
        "model": args.model,
        "samples": len(selected),
        "valid_predictions": len(selected) - invalid,
        "invalid_predictions": invalid,
        "invalid_output_rate": invalid / len(selected) if selected else 0.0,
        "accuracy": accuracy_score(gold, scored) if gold else None,
        "macro_f1": (
            f1_score(gold, scored, labels=list(LABELS), average="macro", zero_division=0)
            if gold
            else None
        ),
        "classification_report": (
            classification_report(
                gold, scored, labels=list(LABELS), output_dict=True, zero_division=0
            )
            if gold
            else None
        ),
        "quantization": "bitsandbytes-nf4-4bit-double-quant",
        "load_in_4bit": True,
        "compute_dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "model_load_seconds": load_seconds,
        "inference_seconds_this_invocation": inference_seconds,
        "wall_time_seconds_this_invocation": wall_seconds,
        "requests_this_invocation": generated_count,
        "throughput_samples_per_second_this_invocation": (
            generated_count / inference_seconds if inference_seconds else None
        ),
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
