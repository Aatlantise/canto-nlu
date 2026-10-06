#!/usr/bin/env python3
"""Paper-style encoder evaluation for the CantoNLU WSD benchmark.

No model training, validation set, classification head, or cosine threshold is
used.  For each target word, every two same-sense sentence pairs form one test:

  mean(similarity within the two senses) >
  mean(similarity across the two senses)

The target is replaced by the tokenizer's mask token and the final-layer hidden
state at that position is compared with cosine similarity.
"""

import argparse
import json
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# A few source sentences are simplified or mixed-script although ``target`` is
# traditional.  Keep the benchmark file untouched and resolve only the attested
# target surface forms during masking.
TARGET_SURFACE_ALIASES = {
    "細佬": ("细佬",),
    "西北風": ("西北风",),
    "點": ("点",),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="WSD JSONL test file")
    p.add_argument("--output", required=True, help="Per-comparison JSONL output")
    p.add_argument(
        "--model",
        default="google-bert/bert-base-multilingual-cased",
    )
    p.add_argument("--batch-size", type=int, default=32,
                   help="Inference batching only; it is not tuned")
    return p.parse_args()


def load_same_sense_pairs(path):
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["label"] == "similar":
                for key in ("target", "sentence1", "sentence2"):
                    if not row.get(key):
                        raise ValueError(f"Missing {key!r} on line {line_no}")
                groups[row["target"]].append(row)
    return groups


def masked_sentences(groups, mask_token):
    sentences = []
    locations = {}
    for target, rows in groups.items():
        for row_idx, row in enumerate(rows):
            for side in ("sentence1", "sentence2"):
                sentence = row[side]
                surface = target if target in sentence else None
                if surface is None:
                    surface = next(
                        (x for x in TARGET_SURFACE_ALIASES.get(target, ())
                         if x in sentence),
                        None,
                    )
                if surface is None:
                    raise ValueError(
                        f"Target {target!r} absent from id={row.get('id')} {side}"
                    )
                # This matches the original notebook's str.replace behavior.
                locations[(target, row_idx, side)] = len(sentences)
                sentences.append(sentence.replace(surface, mask_token))
    return sentences, locations


def encode_masks(sentences, tokenizer, model, device, batch_size):
    vectors = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        mask_positions = encoded["input_ids"].eq(tokenizer.mask_token_id)
        counts = mask_positions.sum(dim=1)
        if torch.any(counts == 0):
            bad = torch.where(counts == 0)[0][0].item()
            raise ValueError(f"No mask token after tokenization: {batch[bad]!r}")

        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state

        # Original code takes the first mask if replacement created more than one.
        first_position = mask_positions.to(torch.int64).argmax(dim=1)
        batch_index = torch.arange(hidden.size(0), device=device)
        selected = hidden[batch_index, first_position].float().cpu()
        vectors.extend(selected.unbind(0))
    return vectors


def main():
    args = parse_args()
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
        raise ValueError(f"{args.model} has no mask token")
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    groups = load_same_sense_pairs(args.data)
    sentences, locations = masked_sentences(groups, tokenizer.mask_token)
    vectors = encode_masks(
        sentences, tokenizer, model, device, args.batch_size
    )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    right = 0
    total = 0
    with output.open("w", encoding="utf-8") as f:
        for target, rows in groups.items():
            for i, j in combinations(range(len(rows)), 2):
                a1 = vectors[locations[(target, i, "sentence1")]]
                a2 = vectors[locations[(target, i, "sentence2")]]
                b1 = vectors[locations[(target, j, "sentence1")]]
                b2 = vectors[locations[(target, j, "sentence2")]]
                same1 = F.cosine_similarity(a1, a2, dim=0).item()
                same2 = F.cosine_similarity(b1, b2, dim=0).item()
                diff1 = F.cosine_similarity(a1, b1, dim=0).item()
                diff2 = F.cosine_similarity(a2, b2, dim=0).item()
                same_mean = (same1 + same2) / 2
                diff_mean = (diff1 + diff2) / 2
                correct = same_mean > diff_mean
                right += int(correct)
                total += 1
                record = {
                    "target": target,
                    "sense_pair_ids": [rows[i].get("id"), rows[j].get("id")],
                    "same_mean": same_mean,
                    "different_mean": diff_mean,
                    "margin": same_mean - diff_mean,
                    "correct": correct,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "protocol": "paper-style masked-position cosine comparison",
        "training": "none",
        "validation": "none",
        "threshold": "none",
        "layer": "last hidden layer",
        "targets": len(groups),
        "same_sense_source_pairs": sum(len(x) for x in groups.values()),
        "comparisons": total,
        "correct": right,
        "wrong": total - right,
        "accuracy": right / total if total else None,
        "batch_size_inference_only": args.batch_size,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "wall_seconds": time.time() - started,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Comparisons: {output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
