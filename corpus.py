from argparse import ArgumentParser
import os
from collections import Counter

import ftfy
from ftfy import TextFixerConfig
import regex
from bs4 import BeautifulSoup
from cantofilter import judge
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_from_disk,
)
from opencc import OpenCC
import pandas as pd

CANTO_DATASETS = ('babylmyue', 'hkcancor', 'lihkg', 'tatoeba', 'wiki', 'wordshk', 'zoengjyutgaai', 'cantomap.tsv', 'captions.csv')

DATA_DIR = "./data"
OUT_DIR = "./data/canto-corpus"

MIN_LENGTH = 4
MAX_LENGTH = 126

KEEP_LABELS = {"cantonese"}

_ALLOWED_RE = regex.compile(r"[^\p{Han}\p{Latin}\p{Common}\p{Emoji_Presentation}]|(\p{P}|=)\1{3,}")
_URL_RE = regex.compile(r"https?://[^\s<>\"')\]]+|www\.[^\s<>\"')\]]+")
_MD_LINK_RE = regex.compile(r"\[([^\]]*)\]\((?:[^)]*)\)")
_BBCODE_TAG_RE = regex.compile(r"\[/?[a-zA-Z0-9=#\"' ]+\]")
_WHITESPACE_RE = regex.compile(r"\p{Zs}+")
_SENT_SPLIT_RE = regex.compile(r"(?<=[。.！？!?；;…」\n])")

_opencc = OpenCC("s2hk")  # converts simplified to traditional

_FTFY_CONFIG = TextFixerConfig(fix_character_width=False)


def _clean(text: str) -> str:
    """Everything that operates on a whole document, before splitting."""
    if not text:
        return ""

    # fix encoding
    text = ftfy.fix_text(text, config=_FTFY_CONFIG, normalization="NFC")

    # strip html
    text = BeautifulSoup(text, "html.parser").get_text()

    # strip markup stuff from forums
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("", text)
    text = _BBCODE_TAG_RE.sub("", text)

    text = _ALLOWED_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    # script normalization
    return _opencc.convert(text)


def _split(text: str, max_length: int) -> list:
    """
    Greedily pack sentences into chunks of at most max_length characters.
    """
    if len(text) <= max_length:
        return [text]

    chunks, current = [], ""
    for sentence in _SENT_SPLIT_RE.split(text):
        if not sentence:
            continue

        if len(sentence) > max_length:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(sentence[i:i + max_length] for i in range(0, len(sentence), max_length))
            continue

        if len(current) + len(sentence) <= max_length:
            current += sentence
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def process(batch: dict) -> dict:
    out_text, out_source, out_label = [], [], []

    for text, source in zip(batch["text"], batch["source"]):
        cleaned = _clean(text)
        if not cleaned:
            continue

        for chunk in _split(cleaned, MAX_LENGTH):
            if len(chunk) < MIN_LENGTH:
                continue
            label = str(judge(chunk))
            if source in ['wiki', 'lihkg', 'appledaily'] and label not in KEEP_LABELS:
                continue
            out_text.append(chunk)
            out_source.append(source)
            out_label.append(label)

    return {"text": out_text, "source": out_source, "label": out_label}


def _to_text_dataset(ds, column: str, source_name: str, splits=None) -> Dataset:
    if isinstance(ds, DatasetDict):
        keys = splits if splits is not None else list(ds.keys())
        ds = concatenate_datasets([ds[k] for k in keys])
    if column != "text":
        ds = ds.rename_column(column, "text")
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    ds = ds.add_column("source", [source_name] * len(ds))
    return ds


def canto_corpus():
    for dataset in CANTO_DATASETS:
        if not os.path.exists(f"{DATA_DIR}/canto-{dataset}"):
            print(f"Dataset canto-{dataset} is missing. Please run download.py first.")
            return

    datasets_to_combine = []

    # canto-babylmyue
    ds = load_from_disk(f"{DATA_DIR}/canto-babylmyue")
    datasets_to_combine.append(_to_text_dataset(ds, "text", "babylmyue"))

    # canto-hkcancor
    ds = load_from_disk(f"{DATA_DIR}/canto-hkcancor")
    datasets_to_combine.append(
        _to_text_dataset(ds, "sentence", "hkcancor", splits=["train", "test", "validation"])
    )

    # canto-lihkg
    ds = load_from_disk(f"{DATA_DIR}/canto-lihkg")
    datasets_to_combine.append(_to_text_dataset(ds, "text", "lihkg"))

    # canto-tatoeba
    tatoeba = load_from_disk(f"{DATA_DIR}/canto-tatoeba")["train"]
    yue_texts = [row["yue"] for row in tatoeba["translation"]]
    datasets_to_combine.append(
        Dataset.from_dict({"text": yue_texts, "source": ["tatoeba"] * len(yue_texts)})
    )

    # canto-wiki
    ds = load_from_disk(f"{DATA_DIR}/canto-wiki")
    datasets_to_combine.append(_to_text_dataset(ds, "text", "wiki"))

    # canto-wordshk
    ds = load_from_disk(f"{DATA_DIR}/canto-wordshk")
    datasets_to_combine.append(_to_text_dataset(ds, "text", "wordshk"))

    # canto-zoengjyutgaai
    ds = load_from_disk(f"{DATA_DIR}/canto-zoengjyutgaai")
    datasets_to_combine.append(
        _to_text_dataset(
            ds,
            "transcription",
            "zoengjyutgaai",
            splits=["lukdinggei", "mouzaakdung", "saamgwokjinji", "seoiwuzyun"],
        )
    )

    # canto-cantomap.tsv
    cantomap_df = pd.read_csv(f"{DATA_DIR}/canto-cantomap.tsv", sep="\t")
    datasets_to_combine.append(
        Dataset.from_dict(
            {
                "text": cantomap_df["sentence"].astype(str).tolist(),
                "source": ["cantomap"] * len(cantomap_df),
            }
        )
    )

    # canto-appledaily
    ds = load_from_disk(f"{DATA_DIR}/canto-appledaily")
    datasets_to_combine.append(_to_text_dataset(ds, "text", "appledaily"))

    # canto-captions.csv
    captions_df = pd.read_csv(f"{DATA_DIR}/canto-captions.csv")
    datasets_to_combine.append(
        Dataset.from_dict(
            {
                "text": captions_df["text"].astype(str).tolist(),
                "source": ["captions"] * len(captions_df),
            }
        )
    )

    corpus = concatenate_datasets(datasets_to_combine)
    print(f"Loaded {len(corpus)} raw examples across {len(datasets_to_combine)} sources.")

    corpus = corpus.map(
        process,
        batched=True,
        remove_columns=corpus.column_names,
        num_proc=os.cpu_count()
    )
    print(f"{len(corpus)} examples after cleaning, splitting and filtering.")

    chars = Counter()
    rows = Counter()
    labels = Counter()
    for source, text, label in zip(corpus["source"], corpus["text"], corpus["label"]):
        chars[source] += len(text)
        rows[source] += 1
        labels[label] += 1

    print(f"\n{'source':<16}{'rows':>12}{'chars':>16}{'mean len':>10}")
    for source, n_chars in chars.most_common():
        print(f"{source:<16}{rows[source]:>12,}{n_chars:>16,}{n_chars / rows[source]:>10.1f}")
    print(f"{'TOTAL':<16}{sum(rows.values()):>12,}{sum(chars.values()):>16,}")
    print(f"\nlabels: {dict(labels)}")

    os.makedirs(os.path.dirname(OUT_DIR), exist_ok=True)
    corpus.save_to_disk(OUT_DIR)
    corpus.to_json(f"{OUT_DIR}.jsonl", force_ascii=False)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--lang", default="yue", choices=["yue"], help="Language to prepare corpus for"
    )
    args = parser.parse_args()
    if args.lang == "yue":
        canto_corpus()
