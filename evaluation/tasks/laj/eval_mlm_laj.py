#!/usr/bin/env python3
"""Evaluate masked language models on LAJ forced-choice sentence pairs.

Each sentence is scored with pseudo-log-likelihood (PLL): mask every lexical
token once and sum the negative log probability assigned to its original token.
The primary decision uses mean token pseudo-surprisal to reduce length bias.
"""

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
    p.add_argument("--model", default="bert-base-multilingual-cased")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32, help="Masked variants per GPU batch")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-length", type=int, default=512)
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


def score_sentence(model, tokenizer, text: str, batch_size: int, max_length: int) -> dict[str, float | int]:
    import torch
    import torch.nn.functional as F

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"][0]
    special_mask = tokenizer.get_special_tokens_mask(
        input_ids.tolist(), already_has_special_tokens=True
    )
    positions = [i for i, is_special in enumerate(special_mask) if not is_special]
    if not positions:
        return {"pseudo_surprisal": math.nan, "token_count": 0, "mean_pseudo_surprisal": math.nan}

    losses: list[float] = []
    device = next(model.parameters()).device
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        masked = input_ids.unsqueeze(0).repeat(len(batch_positions), 1)
        targets = torch.tensor(
            [int(input_ids[pos]) for pos in batch_positions], dtype=torch.long
        )
        for row, pos in enumerate(batch_positions):
            masked[row, pos] = tokenizer.mask_token_id
        attention = torch.ones_like(masked)
        masked = masked.to(device)
        attention = attention.to(device)
        targets = targets.to(device)
        with torch.inference_mode():
            logits = model(input_ids=masked, attention_mask=attention).logits
        row_ids = torch.arange(len(batch_positions), device=device)
        pos_ids = torch.tensor(batch_positions, device=device)
        selected = logits[row_ids, pos_ids, :].float()
        nll = F.cross_entropy(selected, targets, reduction="none")
        losses.extend(nll.cpu().tolist())

    total = float(sum(losses))
    return {
        "pseudo_surprisal": total,
        "token_count": len(losses),
        "mean_pseudo_surprisal": total / len(losses),
    }


def choice(score1: float, score2: float) -> int:
    if math.isclose(score1, score2, rel_tol=0.0, abs_tol=1e-12):
        return 0
    return 1 if score1 < score2 else 2


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install torch, transformers and accelerate first.") from exc

    pairs, excluded = load_pairs(args.pairs, args.source)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.mask_token_id is None:
        raise SystemExit(f"{args.model} has no mask token and is not a masked LM")
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows = []
    for index, pair in enumerate(pairs, start=1):
        s1 = score_sentence(model, tokenizer, pair["sentence1"], args.batch_size, args.max_length)
        s2 = score_sentence(model, tokenizer, pair["sentence2"], args.batch_size, args.max_length)
        pred_total = choice(float(s1["pseudo_surprisal"]), float(s2["pseudo_surprisal"]))
        pred_mean = choice(float(s1["mean_pseudo_surprisal"]), float(s2["mean_pseudo_surprisal"]))
        rows.append({
            **pair,
            "pseudo_surprisal1": s1["pseudo_surprisal"],
            "pseudo_surprisal2": s2["pseudo_surprisal"],
            "token_count1": s1["token_count"],
            "token_count2": s2["token_count"],
            "mean_pseudo_surprisal1": s1["mean_pseudo_surprisal"],
            "mean_pseudo_surprisal2": s2["mean_pseudo_surprisal"],
            "pred_total": pred_total,
            "pred_mean": pred_mean,
            "correct_total": pred_total == pair["label"],
            "correct_mean": pred_mean == pair["label"],
        })
        if index % 50 == 0 or index == len(pairs):
            print(f"scored {index}/{len(pairs)} pairs", flush=True)

    if not rows:
        raise SystemExit("No pairs to score")
    n = len(rows)
    correct_total = sum(row["correct_total"] for row in rows)
    correct_mean = sum(row["correct_mean"] for row in rows)
    summary = {
        "model": args.model,
        "model_type": "masked_language_model",
        "scoring": "pseudo_log_likelihood",
        "primary_metric": "accuracy_mean_token_pseudo_surprisal",
        "device": str(device),
        "pairs_scored": n,
        "pairs_excluded_no_mt_errors": excluded,
        "correct_total_pseudo_surprisal": correct_total,
        "accuracy_total_pseudo_surprisal": correct_total / n,
        "correct_mean_token_pseudo_surprisal": correct_mean,
        "accuracy_mean_token_pseudo_surprisal": correct_mean / n,
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
