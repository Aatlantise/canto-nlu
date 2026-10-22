#!/usr/bin/env python3
"""Zero-shot Cantonese NLI evaluation for local instruction-tuned LMs."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABELS = ("entailment", "not_entailment")
EVALUATOR_VERSION = "cantonese-nli-zero-shot-greedy-v1"
PROMPT_TEMPLATE = """你是一個香港粵語自然語言推理分類器。判斷「前提」係咪必然推出「假設」。

標籤定義：
- entailment：前提必然支持或推出假設
- not_entailment：前提唔能夠必然推出假設（包括矛盾、無關或資料不足）

只可以輸出 entailment 或 not_entailment，唔好解釋。

前提：{premise}
假設：{hypothesis}
標籤："""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"id", "premise", "hypothesis", "label"} - set(row)
            if missing:
                raise ValueError(f"Line {line_number} is missing {sorted(missing)}")
            if row["label"] not in LABELS:
                raise ValueError(f"Line {line_number} has unknown label {row['label']!r}")
            row_id = str(row["id"])
            if row_id in seen:
                raise ValueError(f"Duplicate id {row_id!r}")
            seen.add(row_id)
            rows.append({key: str(row[key]) for key in ("id", "premise", "hypothesis", "label")})
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No examples found")
    return rows


def extract_label(answer: str) -> str | None:
    normalized = answer.strip().lower().replace("-", "_")
    exact = re.fullmatch(
        r"(?:label|標籤)?\s*[:：]?\s*(not_entailment|entailment)[.!。！\s]*",
        normalized,
    )
    if exact:
        return exact.group(1)
    found = re.findall(r"(?<![a-z_])(not_entailment|entailment)(?![a-z_])", normalized)
    return found[0] if len(set(found)) == 1 else None


def calculate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(record["prediction"] == record["gold"] for record in records)
    invalid = sum(record["prediction"] is None for record in records)
    confusion = {
        gold: {
            predicted: sum(
                record["gold"] == gold and record["prediction"] == predicted
                for record in records
            )
            for predicted in (*LABELS, "INVALID")
        }
        for gold in LABELS
    }
    for gold in LABELS:
        confusion[gold]["INVALID"] = sum(
            record["gold"] == gold and record["prediction"] is None
            for record in records
        )
    return {
        "samples": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "invalid_predictions": invalid,
        "invalid_output_rate": invalid / len(records),
        "confusion_matrix": confusion,
    }


def main() -> None:
    args = parse_args()
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install torch, transformers, accelerate, and optionally bitsandbytes.") from exc

    rows = load_rows(args.data, args.max_samples)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    quantization = "none"
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise SystemExit("--load-in-4bit requires CUDA")
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        quantization = "4-bit NF4 double quantization"

    is_multimodal_gemma3 = "gemma-3-" in args.model.lower() and "-1b-" not in args.model.lower()
    processor = None
    if is_multimodal_gemma3:
        from transformers import Gemma3ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
        tokenizer = processor.tokenizer
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model, **model_kwargs
        ).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        if processor is not None:
            conversations = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT_TEMPLATE.format(**row)}
                        ],
                    }
                ]
                for row in batch
            ]
            prompts = [
                processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                for conversation in conversations
            ]
        else:
            conversations = [
                [{"role": "user", "content": PROMPT_TEMPLATE.format(**row)}]
                for row in batch
            ]
            prompts = tokenizer.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=True
            )
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        answers = tokenizer.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )
        for row, answer in zip(batch, answers):
            prediction = extract_label(answer)
            records.append(
                {
                    "id": row["id"],
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "gold": row["label"],
                    "prediction": prediction,
                    "correct": prediction == row["label"],
                    "raw_answer": answer.strip(),
                }
            )
        print(f"Scored {len(records)}/{len(rows)}", flush=True)

    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output_dir / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "evaluator_version": EVALUATOR_VERSION,
        "model": args.model,
        "setting": "zero-shot deterministic greedy decoding",
        "labels": list(LABELS),
        "primary_metric": "accuracy",
        **calculate_metrics(records),
        "quantization": quantization,
        "compute_dtype": str(dtype).replace("torch.", ""),
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": elapsed,
        "samples_per_second": len(records) / elapsed if elapsed else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
