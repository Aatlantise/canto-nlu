#!/usr/bin/env python3
"""Evaluate a causal LM on CantoNLU/LAJ forced-choice pairs."""

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
    p.add_argument(
        "--source",
        type=Path,
        help="Original cantonese.jsonl; excludes MT items without error annotations.",
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--family", choices=("qwen", "gemma3"), default="qwen")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("laj_qwen_results"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Score only the first N valid pairs.")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def valid_source_ids(path: Path | None) -> set[Any] | None:
    if path is None:
        return None
    valid = set()
    for item in read_jsonl(path):
        spans = (item.get("annotations") or {}).get("annotatedSpans") or []
        if spans:
            valid.add(item["id"])
    return valid


def load_pairs(pairs_path: Path, source_path: Path | None) -> tuple[list[dict], int]:
    pairs = read_jsonl(pairs_path)
    valid_ids = valid_source_ids(source_path)
    if valid_ids is None:
        return pairs, 0
    kept = [x for x in pairs if x["source_id"] in valid_ids]
    return kept, len(pairs) - len(kept)


def score_sentences(model, tokenizer, texts: list[str], batch_size: int, max_length: int | None):
    import torch
    import torch.nn.functional as F

    scores: list[dict[str, float | int]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        kwargs = dict(
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            truncation=max_length is not None,
        )
        if max_length is not None:
            kwargs["max_length"] = max_length
        enc = tokenizer(batch, **kwargs)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()
        token_lp = F.log_softmax(logits.float(), dim=-1).gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)
        token_nll = (-token_lp).masked_fill(~mask, 0.0)
        totals = token_nll.sum(dim=1)
        counts = mask.sum(dim=1)
        for total, count in zip(totals.tolist(), counts.tolist()):
            scores.append(
                {
                    "surprisal": total,
                    "token_count": count,
                    "mean_surprisal": total / count if count else math.nan,
                }
            )
    return scores


def choice(a: float, b: float) -> int:
    if a == b:
        return 0
    return 1 if a < b else 2


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install with: pip install 'torch>=2.1' "
            "'transformers>=4.37' accelerate"
        ) from exc

    pairs, excluded = load_pairs(args.pairs, args.source)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    texts = [s for x in pairs for s in (x["sentence1"], x["sentence2"])]

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    quantization_config = None
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise SystemExit("--load-in-4bit requires a CUDA GPU")
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {
        "device_map": "auto",
        "dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if args.family == "gemma3":
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

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
        model = AutoModelForCausalLM.from_pretrained(
            args.model, **model_kwargs
        ).eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    scores = score_sentences(
        model, tokenizer, texts, args.batch_size, args.max_length
    )

    rows = []
    for i, pair in enumerate(pairs):
        s1, s2 = scores[2 * i], scores[2 * i + 1]
        pred_total = choice(float(s1["surprisal"]), float(s2["surprisal"]))
        pred_mean = choice(float(s1["mean_surprisal"]), float(s2["mean_surprisal"]))
        rows.append(
            {
                **pair,
                "surprisal1": s1["surprisal"],
                "surprisal2": s2["surprisal"],
                "token_count1": s1["token_count"],
                "token_count2": s2["token_count"],
                "mean_surprisal1": s1["mean_surprisal"],
                "mean_surprisal2": s2["mean_surprisal"],
                "pred_total": pred_total,
                "pred_mean": pred_mean,
                "correct_total": pred_total == pair["label"],
                "correct_mean": pred_mean == pair["label"],
            }
        )

    n = len(rows)
    summary = {
        "model": args.model,
        "family": args.family,
        "quantization": "4-bit NF4 double quantization" if args.load_in_4bit else "none",
        "compute_dtype": str(dtype).replace("torch.", ""),
        "pairs_scored": n,
        "pairs_excluded_no_mt_errors": excluded,
        "accuracy_total_surprisal": sum(x["correct_total"] for x in rows) / n,
        "accuracy_mean_token_surprisal": sum(x["correct_mean"] for x in rows) / n,
        "ties_total": sum(x["pred_total"] == 0 for x in rows),
        "ties_mean": sum(x["pred_mean"] == 0 for x in rows),
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
