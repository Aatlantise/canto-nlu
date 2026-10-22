#!/usr/bin/env python3
"""Zero-shot OpenRice sentiment evaluation for Qwen2.5 Instruct models."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = ("smile", "ok", "cry")
SYSTEM_PROMPT = (
    "你是一個香港粵語文本分類器。你必須嚴格遵守輸出格式，"
    "只輸出一個英文標籤，不要解釋。"
)
USER_TEMPLATE = """判斷以下 OpenRice 香港粵語餐廳評論的整體情感。

標籤定義：
- smile：正面
- ok：中性或好壞參半
- cry：負面

只可輸出 smile、ok 或 cry。

評論：
{text}

標籤："""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to OpenRice test.tsv")
    parser.add_argument("--output", required=True, help="JSONL prediction output")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def extract_label(answer: str) -> str | None:
    normalized = answer.strip().lower()
    exact = re.fullmatch(r"(?:label\s*:\s*)?(smile|ok|cry)[.!。！\s]*", normalized)
    if exact:
        return exact.group(1)
    found = re.findall(r"(?<![a-z])(smile|ok|cry)(?![a-z])", normalized)
    return found[0] if len(set(found)) == 1 else None


def load_model(model_id: str, load_in_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"device_map": "auto"}
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
    else:
        kwargs["torch_dtype"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    return tokenizer, model


def predict(text: str, tokenizer, model, max_input_tokens: int) -> tuple[str, str | None]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer, extract_label(answer)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = pd.read_csv(args.data, sep="\t", encoding="utf-8")
    required = {"label", "text_a"}
    if not required.issubset(data.columns):
        raise ValueError(f"Expected columns {sorted(required)}; found {list(data.columns)}")
    unknown = sorted(set(data["label"].astype(str)) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    if args.max_samples is not None:
        data = data.head(args.max_samples)

    tokenizer, model = load_model(args.model, args.load_in_4bit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gold: list[str] = []
    predicted: list[str | None] = []
    with output_path.open("w", encoding="utf-8") as stream:
        for row_id, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating"):
            answer, label = predict(
                str(row["text_a"]), tokenizer, model, args.max_input_tokens
            )
            gold_label = str(row["label"])
            gold.append(gold_label)
            predicted.append(label)
            record = {
                "row_id": int(row_id),
                "gold": gold_label,
                "prediction": label,
                "raw_answer": answer,
                "model": args.model,
                "seed": args.seed,
                "setting": "zero-shot-greedy",
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

    valid_indices = [i for i, label in enumerate(predicted) if label is not None]
    valid_gold = [gold[i] for i in valid_indices]
    valid_pred = [predicted[i] for i in valid_indices]
    invalid_count = len(predicted) - len(valid_indices)
    scored_pred = [label if label is not None else "__invalid__" for label in predicted]

    summary = {
        "model": args.model,
        "samples": len(gold),
        "valid_predictions": len(valid_indices),
        "invalid_predictions": invalid_count,
        "invalid_output_rate": invalid_count / len(gold) if gold else 0.0,
        "accuracy": accuracy_score(gold, scored_pred) if gold else None,
        "macro_f1": (
            f1_score(
                gold,
                scored_pred,
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
            if gold
            else None
        ),
        "accuracy_on_valid": accuracy_score(valid_gold, valid_pred) if valid_gold else None,
        "macro_f1_on_valid": (
            f1_score(valid_gold, valid_pred, labels=list(LABELS), average="macro", zero_division=0)
            if valid_gold
            else None
        ),
        "classification_report_on_valid": (
            classification_report(
                valid_gold,
                valid_pred,
                labels=list(LABELS),
                output_dict=True,
                zero_division=0,
            )
            if valid_gold
            else None
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Predictions: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
