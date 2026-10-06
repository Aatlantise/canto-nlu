#!/usr/bin/env python3
"""Zero-shot prompted LD evaluation for Qwen, Gemma 3, and Llama."""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from ld_metrics import LABELS, classification_metrics, read_ld_jsonl, sha256


EVALUATOR_VERSION = "ld-prompted-v1"
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
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--family", choices=("causal", "gemma3"), default="causal")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
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
    unique = set(found)
    return found[0] if len(unique) == 1 else None


def read_predictions(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[int(record["id"])] = record
    return records


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available() else torch.float32
    )
    kwargs = {
        "device_map": "auto",
        "dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    quantization = "none"
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise SystemExit("--load-in-4bit requires CUDA")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        quantization = "4-bit NF4 double quantization"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    started = time.perf_counter()
    if args.family == "gemma3":
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model, **kwargs
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    return tokenizer, model, dtype, quantization, time.perf_counter() - started


def predict_batch(rows, tokenizer, model, args):
    import torch

    conversations = [
        [{"role": "user", "content": PROMPT.format(sentence=row["sentence"])}]
        for row in rows
    ]
    rendered = [
        tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        for chat in conversations
    ]
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    ).to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.perf_counter() - started
    input_width = inputs["input_ids"].shape[1]
    answers = tokenizer.batch_decode(outputs[:, input_width:], skip_special_tokens=True)
    return [(answer.strip(), extract_label(answer)) for answer in answers], elapsed


def main() -> None:
    args = arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    random.seed(args.seed)
    import torch
    import transformers

    torch.manual_seed(args.seed)
    rows = read_ld_jsonl(args.data, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prior = read_predictions(args.output) if args.resume else {}
    pending = [row for row in rows if row["id"] not in prior]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    run_started = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    tokenizer, model, dtype, quantization, load_seconds = load_model(args)
    mode = "a" if args.resume and args.output.exists() else "w"
    inference_seconds = 0.0
    generated = 0

    with args.output.open(mode, encoding="utf-8") as stream:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            results, latency = predict_batch(batch, tokenizer, model, args)
            inference_seconds += latency
            for row, (answer, prediction) in zip(batch, results):
                record = {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "gold_id": row["label"],
                    "gold": LABELS[row["label"]],
                    "prediction": prediction,
                    "raw_answer": answer,
                    "model": args.model,
                    "setting": "zero-shot-greedy",
                    "evaluator_version": EVALUATOR_VERSION,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                generated += 1
            stream.flush()
            done = min(start + len(batch), len(pending))
            print(f"generated {done}/{len(pending)}", flush=True)

    all_records = read_predictions(args.output)
    selected = [all_records[row["id"]] for row in rows if row["id"] in all_records]
    summary = {
        "model": args.model,
        "family": args.family,
        "task": "Cantonese/Mandarin/mixed language detection",
        "setting": "zero-shot-greedy",
        "primary_metric": "macro_f1",
        "label_mapping": {str(i): name for i, name in enumerate(LABELS)},
        "data_path": str(args.data),
        "data_sha256": sha256(args.data),
        **classification_metrics(selected),
        "quantization": quantization,
        "compute_dtype": str(dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "model_load_seconds": load_seconds,
        "inference_seconds_this_invocation": inference_seconds,
        "wall_seconds_this_invocation": time.perf_counter() - wall_started,
        "generated_this_invocation": generated,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        ),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
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
