"""Canonicalize the LD (language discrimination) dataset to .jsonl.

Reimplements the pipeline in acceptability-dataset.ipynb:
  - label 0: original Cantonese sentence
  - label 1: original Mandarin sentence
  - label 2: corrupted, partially code-mixed sentence

acceptability-dataset.ipynb: https://github.com/Aatlantise/canto-nlu/blob/helena-data/acceptability-dataset.ipynb

Source corpus: HKAllen/cantonese-chinese-parallel-corpus

Usage:
    python data/ld/convert_ld.py [--n-sentences 10000] [--device mps|cpu]
"""
import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from hanziconv import HanziConv
from simalign import SentenceAligner
from tqdm import tqdm
from transformers import BertTokenizerFast

HERE = Path(__file__).parent
RANDOM_SEED = 42
TRAIN_RATIO = 0.9
VAL_RATIO = 0.05


def build_examples(n_sentences: int, device: str):
    ds = load_dataset("HKAllen/cantonese-chinese-parallel-corpus")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    aligner = SentenceAligner(model="bert-base-chinese", matching_methods="m", device=device)

    examples = []
    for source_id, sentence in enumerate(tqdm(ds["train"].select(range(n_sentences)))):
        yue_tokens = tokenizer.tokenize(sentence["yue"])
        cmn_traditional = HanziConv.toTraditional(sentence["zh"])
        if sentence["yue"] == cmn_traditional or 127 <= len(yue_tokens) <= 5 or 127 <= len(cmn_traditional) <= 5:
            continue
        cmn_tokens = tokenizer.tokenize(cmn_traditional)
        alignment = aligner.get_word_aligns(yue_tokens, cmn_tokens)["mwmf"]

        corrupted_sentences = set()
        for _ in (0.15, 0.33, 0.5):
            corrupted_sentence_yue = list(yue_tokens)
            for idx_0, idx_1 in alignment:
                if random.random() < 0.25 and yue_tokens[idx_0] != cmn_tokens[idx_1]:
                    corrupted_sentence_yue[idx_0] = cmn_tokens[idx_1]
            corrupted_sentences.add("".join(corrupted_sentence_yue).replace("##", ""))
        for _ in (0.15, 0.33, 0.5):
            corrupted_sentence_cmn = list(cmn_tokens)
            for idx_0, idx_1 in alignment:
                if random.random() < 0.25 and yue_tokens[idx_0] != cmn_tokens[idx_1]:
                    corrupted_sentence_cmn[idx_1] = yue_tokens[idx_0]
            corrupted_sentences.add("".join(corrupted_sentence_cmn).replace("[UNK]", ""))

        for corrupted_sentence in corrupted_sentences:
            if corrupted_sentence in (sentence["yue"], cmn_traditional):
                continue
            examples.append((source_id, corrupted_sentence, 2))
        examples.append((source_id, sentence["yue"], 0))
        examples.append((source_id, cmn_traditional, 1))


    filtered = []
    for source_id, sentence, label in examples:
        if "[UNK]" in sentence:
            continue
        sentence = sentence.replace("##", "")
        if any(c.isascii() and c.isalpha() for c in sentence):
            continue
        filtered.append((source_id, sentence, label))

    return filtered


def split_by_source(examples):
    """Split 90/5/5 by source_id so sentence variants derived
    from the same source pair will come in the same split."""
    source_ids = sorted(set(source_id for source_id, _, _ in examples))
    n = len(source_ids)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_ids = set(source_ids[:train_end])
    val_ids = set(source_ids[train_end:val_end])
    test_ids = set(source_ids[val_end:])

    splits = {"train": [], "val": [], "test": []}
    for row in examples:
        source_id = row[0]
        if source_id in train_ids:
            splits["train"].append(row)
        elif source_id in val_ids:
            splits["val"].append(row)
        else:
            splits["test"].append(row)
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sentences", type=int, default=10000)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    examples = build_examples(args.n_sentences, args.device)
    splits = split_by_source(examples)

    for split, rows in splits.items():
        out_path = HERE / f"ld_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for i, (source_id, sentence, label) in enumerate(rows):
                f.write(json.dumps({
                    "id": i,
                    "source_id": source_id,
                    "sentence": sentence,
                    "label": label,
                }, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} examples to {out_path}")


if __name__ == "__main__":
    main()
