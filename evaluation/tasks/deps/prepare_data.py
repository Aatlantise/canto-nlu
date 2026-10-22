"""Canonicalize the Cantonese UD dependency-parsing data to .jsonl.

Source: UD_Cantonese-HK test CoNLL-U file
https://github.com/UniversalDependencies/UD_Cantonese-HK/blob/dev/yue_hk-ud-test.conllu

Usage:
    python data/deps/convert_deps.py
"""
import json
import urllib.request
from pathlib import Path

CONLLU_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_Cantonese-HK/dev/yue_hk-ud-test.conllu"
)
HERE = Path(__file__).parent
CONLLU_PATH = HERE / "yue_hk-ud-test.conllu"
TRAIN_RATIO = 0.9


def download_conllu():
    if not CONLLU_PATH.exists():
        urllib.request.urlretrieve(CONLLU_URL, CONLLU_PATH)


def parse_conllu(file_path):
    sentences, dep_labels, head_labels = [], [], []
    current_sentence, current_dep, current_head = [], [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if not line and current_sentence:
                    sentences.append(current_sentence)
                    dep_labels.append(current_dep)
                    head_labels.append(current_head)
                    current_sentence, current_dep, current_head = [], [], []
                continue

            parts = line.split("\t")
            if len(parts) != 10:
                continue

            token, head, dep = parts[1], int(parts[6]), parts[7]
            current_sentence.append(token)
            current_dep.append(dep)
            current_head.append(head)

    if current_sentence:
        sentences.append(current_sentence)
        dep_labels.append(current_dep)
        head_labels.append(current_head)

    return sentences, dep_labels, head_labels


def main():
    """Split 90/10 by rows, no shuffle, and write to .jsonl files."""
    download_conllu()
    sentences, dep_labels, head_labels = parse_conllu(CONLLU_PATH)

    train_end = int(len(sentences) * TRAIN_RATIO)
    splits = {
        "train": (sentences[:train_end], dep_labels[:train_end], head_labels[:train_end]),
        "test": (sentences[train_end:], dep_labels[train_end:], head_labels[train_end:]),
    }

    for split, (sents, deps, heads) in splits.items():
        out_path = HERE / f"deps_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for i, (tokens, deprels, hd) in enumerate(zip(sents, deps, heads)):
                f.write(json.dumps({
                    "id": i,
                    "tokens": tokens,
                    "deprels": deprels,
                    "heads": hd,
                }, ensure_ascii=False) + "\n")
        print(f"Wrote {len(sents)} examples to {out_path}")


if __name__ == "__main__":
    main()
