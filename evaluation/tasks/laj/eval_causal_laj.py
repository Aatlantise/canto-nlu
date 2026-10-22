#!/usr/bin/env python3
"""Evaluate an autoregressive LM on LAJ forced-choice sentence pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-length", type=int)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pairs(pairs_path: Path, source_path: Path | None) -> tuple[list[dict], int]:
    pairs = read_jsonl(pairs_path)
    if source_path is None:
        return pairs, 0
    valid_ids = {
        item["id"]
        for item in read_jsonl(source_path)
        if ((item.get("annotations") or {}).get("annotatedSpans") or [])
    }
    kept = [pair for pair in pairs if pair["source_id"] in valid_ids]
    return kept, len(pairs) - len(kept)


def score_sentences(model, tokenizer, texts: list[str], batch_size: int, max_length: int | None):
    import torch
    import torch.nn.functional as F

    scores = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "add_special_tokens": True,
            "truncation": max_length is not None,
        }
        if max_length is not None:
            kwargs["max_length"] = max_length
        enc = tokenizer(batch, **kwargs)
        enc = {key: value.to(model.device) for key, value in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()
        token_log_probs = F.log_softmax(logits.float(), dim=-1).gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)
        token_nll = (-token_log_probs).masked_fill(~mask, 0.0)
        totals = token_nll.sum(dim=1)
        counts = mask.sum(dim=1)
        for total, count in zip(totals.tolist(), counts.tolist()):
            scores.append({
                "surprisal": total,
                "token_count": count,
                "mean_surprisal": total / count if count else math.nan,
            })
    return scores


def choice(score1: float, score2: float) -> int:
    if math.isclose(score1, score2, rel_tol=0.0, abs_tol=1e-12):
        return 0
    return 1 if score1 < score2 else 2


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install torch, transformers and accelerate first.") from exc

    pairs, excluded = load_pairs(args.pairs, args.source)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    texts = [text for pair in pairs for text in (pair["sentence1"], pair["sentence2"])]

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
    scores = score_sentences(model, tokenizer, texts, args.batch_size, args.max_length)

    rows = []
    for i, pair in enumerate(pairs):
        s1, s2 = scores[2 * i], scores[2 * i + 1]
        pred_total = choice(float(s1["surprisal"]), float(s2["surprisal"]))
        pred_mean = choice(float(s1["mean_surprisal"]), float(s2["mean_surprisal"]))
        rows.append({
            **pair,
            "surprisal1": s1["surprisal"], "surprisal2": s2["surprisal"],
            "token_count1": s1["token_count"], "token_count2": s2["token_count"],
            "mean_surprisal1": s1["mean_surprisal"],
            "mean_surprisal2": s2["mean_surprisal"],
            "pred_total": pred_total, "pred_mean": pred_mean,
            "correct_total": pred_total == pair["label"],
            "correct_mean": pred_mean == pair["label"],
        })

    if not rows:
        raise SystemExit("No pairs to score")
    n = len(rows)
    correct_total = sum(row["correct_total"] for row in rows)
    correct_mean = sum(row["correct_mean"] for row in rows)
    summary = {
        "model": args.model,
        "model_type": "causal_language_model",
        "scoring": "autoregressive_surprisal",
        "primary_metric": "accuracy_mean_token_surprisal",
        "quantization": quantization,
        "compute_dtype": str(dtype).replace("torch.", ""),
        "pairs_scored": n,
        "pairs_excluded_no_mt_errors": excluded,
        "correct_total_surprisal": correct_total,
        "accuracy_total_surprisal": correct_total / n,
        "correct_mean_token_surprisal": correct_mean,
        "accuracy_mean_token_surprisal": correct_mean / n,
        "ties_total": sum(row["pred_total"] == 0 for row in rows),
        "ties_mean": sum(row["pred_mean"] == 0 for row in rows),
        "max_length": args.max_length,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
