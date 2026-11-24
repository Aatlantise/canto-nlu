from datasets import load_from_disk, Dataset, DatasetDict
from collections import Counter
import pandas as pd
import os
import torch
from pathlib import Path
from tokenizers import SentencePieceUnigramTokenizer
from transformers import (
    PreTrainedTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def lexical_similarity(s1: str, s2: str) -> float:
    """
    Calculate the lexical similarity between two strings using Jaccard similarity,
    considering only non-Latin characters.
    :param s1: str in a Sinitic language
    :param s2: str in a Sinitic language
    :return: float between [0, 1]
    """
    s1 = ''.join(filter(lambda x: x.isalpha() or x.isspace(), s1))
    s2 = ''.join(filter(lambda x: x.isalpha() or x.isspace(), s2))
    s1_counter = Counter(list(s1))
    s2_counter = Counter(list(s2))

    intersection = sum((s1_counter & s2_counter).values())
    union = sum((s1_counter | s2_counter).values())

    if union == 0:
        return 0.0
    return intersection / union


def parse_conllu(file_path):
    sentences = []
    pos_labels = []
    dep_labels = []
    head_labels = []

    current_sentence = []
    current_pos = []
    current_dep = []
    current_head = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip metadata and empty lines
            if not line or line.startswith('#'):
                if not line:
                    if current_sentence:  # end of sentence
                        sentences.append(current_sentence)
                        pos_labels.append(current_pos)
                        dep_labels.append(current_dep)
                        head_labels.append(current_head)
                        current_sentence = []
                        current_pos = []
                        current_dep = []
                        current_head = []
                continue

            parts = line.split('\t')
            if len(parts) != 10:
                continue  # malformed line

            token, upos, head, dep = parts[1], parts[3], int(parts[6]), parts[7]
            current_sentence.append(token)
            current_pos.append(upos)
            current_dep.append(dep)
            current_head.append(head)

        # Catch any sentence that wasn’t followed by a blank line
        if current_sentence:
            sentences.append(current_sentence)
            pos_labels.append(current_pos)
            dep_labels.append(current_dep)
            head_labels.append(current_head)

    return sentences, pos_labels, dep_labels, head_labels


def build_pos_dataset_dict(sentences, labels, train_ratio=0.9):
    total = len(sentences)
    train_end = int(total * train_ratio)

    dataset_splits = {
        "train": {
            "sentence": sentences[:train_end],
            "labels": labels[:train_end]
        },
        "test": {
            "sentence": sentences[train_end:],
            "labels": labels[train_end:]
        }
    }

    return DatasetDict({
        split: Dataset.from_dict(data)
        for split, data in dataset_splits.items()
    })

def build_deps_dataset_dict(sentences, dep, heads, train_ratio=0.9):
    total = len(sentences)
    train_end = int(total * train_ratio)

    dataset_splits = {
        "train": {
            "sentence": sentences[:train_end],
            "deps": dep[:train_end],
            "heads": heads[:train_end],
        },
        "test": {
            "sentence": sentences[train_end:],
            "deps": dep[train_end:],
            "heads": heads[train_end:]
        }
    }

    return DatasetDict({
        split: Dataset.from_dict(data)
        for split, data in dataset_splits.items()
    })


def conllu_to_dataset():
    # Usage
    conllu_path = "./data/yue_hk-ud-test.conllu"  # Adjust this path
    sents, pos, dep, head = parse_conllu(conllu_path)
    pos_dataset = build_pos_dataset_dict(sents, pos)
    deps_dataset = build_deps_dataset_dict(sents, dep, head)

    # Optionally save for reuse
    pos_dataset.save_to_disk("./data/yue-pos")
    deps_dataset.save_to_disk("./data/yue-deps")

def train_canto_tokenizer():
    dataset = load_from_disk("./data/yue-wiki-full-local")
    tokenizer_dir = Path("./models/cantonese_tokenizer")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    if not (tokenizer_dir / "tokenizer.json").exists():
        print("Training tokenizer...")
        # Save all text to a temporary file for SP training
        with open("cantonese_wiki_corpus.txt", "w", encoding="utf-8") as f:
            for example in dataset["train"]:
                if example["text"] is not None:
                    f.write(example["text"].replace("\n", " ") + "\n")

        sp_tokenizer = SentencePieceUnigramTokenizer(
            # unk_token="[UNK]"
        )
        sp_tokenizer.train(
            files=["cantonese_wiki_corpus.txt"],
            vocab_size=32000,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            show_progress=True,
        )
        sp_tokenizer.save(str(tokenizer_dir / "tokenizer.json"))

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenizer.model_max_length = 512
    tokenizer.save_pretrained(tokenizer_dir)

if __name__ == '__main__':
    train_canto_tokenizer()
