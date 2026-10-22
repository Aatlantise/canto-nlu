"""Canonicalize the Cantonese UD POS-tagging data to .jsonl.

Source: UD_Cantonese-HK test CoNLL-U file
https://github.com/UniversalDependencies/UD_Cantonese-HK/blob/dev/yue_hk-ud-test.conllu

Usage:
    python data/pos/convert_pos.py
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
    sentences, pos_labels = [], []
    current_sentence, current_pos = [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if not line and current_sentence:
                    sentences.append(current_sentence)
                    pos_labels.append(current_pos)
                    current_sentence, current_pos = [], []
                continue

            parts = line.split("\t")
            if len(parts) != 10:
                continue

            token, upos = parts[1], parts[3]
            current_sentence.append(token)
            current_pos.append(upos)

    if current_sentence:
        sentences.append(current_sentence)
        pos_labels.append(current_pos)

    return sentences, pos_labels


def main():
    """Split 90/10 by rows, no shuffle, and write to .jsonl files."""
    download_conllu()
    sentences, pos_labels = parse_conllu(CONLLU_PATH)

    train_end = int(len(sentences) * TRAIN_RATIO)
    splits = {
        "train": (sentences[:train_end], pos_labels[:train_end]),
        "test": (sentences[train_end:], pos_labels[train_end:]),
    }

    for split, (sents, tags) in splits.items():
        out_path = HERE / f"pos_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for i, (tokens, upos) in enumerate(zip(sents, tags)):
                f.write(json.dumps({
                    "id": i,
                    "tokens": tokens,
                    "upos": upos,
                }, ensure_ascii=False) + "\n")
        print(f"Wrote {len(sents)} examples to {out_path}")


if __name__ == "__main__":
    main()
