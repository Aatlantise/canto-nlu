#!/usr/bin/env python3
"""Zero-shot Cantonese UPOS evaluation for Gemma 3 instruction models."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

from qwen_pos_eval import (
    SYSTEM_PROMPT,
    calculate_metrics,
    read_data,
    read_predictions,
    recover_answer,
)


EVALUATOR_VERSION = "gemma3-pos-zero-shot-v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="JSONL with id, tokens, and upos")
    parser.add_argument("--output", required=True, help="Prediction JSONL")
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_model(model_id, load_in_4bit):
    kwargs = {"device_map": "auto"}
    if load_in_4bit:
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["torch_dtype"] = compute_dtype
        precision = f"bitsandbytes-nf4-{str(compute_dtype).replace('torch.', '')}"
    else:
        kwargs["torch_dtype"] = torch.bfloat16
        precision = "bfloat16"
    started = time.perf_counter()
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, **kwargs
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return processor, model, precision, time.perf_counter() - started


def predict(row, processor, model, max_input_tokens, max_new_tokens):
    user_content = json.dumps({"tokens": row["tokens"]}, ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_content}],
        },
    ]
    model_inputs = processor.apply_chat_template(
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
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    latency = time.perf_counter() - started
    generated = output[0, model_inputs["input_ids"].shape[-1] :]
    answer = processor.decode(generated, skip_special_tokens=True).strip()
    tags, status, strict_tags, strict_status = recover_answer(
        answer, len(row["tokens"])
    )
    return answer, tags, status, strict_tags, strict_status, latency


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
    processor, model, precision, load_seconds = load_model(
        args.model, args.load_in_4bit
    )
    inference_seconds = 0.0
    generated_count = 0

    with output_path.open(mode, encoding="utf-8") as stream:
        for row in tqdm(rows, desc="Gemma POS evaluation"):
            row_id = int(row["id"])
            if row_id in prior:
                continue
            answer, prediction, status, strict_prediction, strict_status, latency = predict(
                row,
                processor,
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
