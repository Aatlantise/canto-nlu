#!/usr/bin/env python3
"""Zero-shot evaluation of instruction-tuned LMs on the seven CantoNLU tasks.

Local Hugging Face models share one generation backend. DeepSeek-compatible APIs
share the same task prompts, parsers, and metrics so results remain comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABELS = {
    "nli": ("entailment", "not_entailment"),
    "sentiment": ("smile", "ok", "cry"),
    "ld": ("cantonese", "mandarin", "mixed"),
    "laj": ("unacceptable", "acceptable"),
    "wsd": ("similar", "not_similar"),
}
POS_TAGS = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "VERB",
)
DEFAULT_DATA = {
    "nli": Path("data/nli/sample_nli.jsonl"),
    "sentiment": Path("data/sentiment/test.jsonl"),
    "ld": Path("data/ld/ld_test.jsonl"),
    "laj": Path("data/laj/laj_finetune/finetune_test.jsonl"),
    "pos": Path("data/pos/pos_test.jsonl"),
    "deps": Path("data/deps/deps_test.jsonl"),
    "wsd": Path("data/wsd/test.jsonl"),
}
LD_LABELS = {0: "cantonese", 1: "mandarin", 2: "mixed"}
LAJ_LABELS = {0: "unacceptable", 1: "acceptable"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(DEFAULT_DATA), required=True)
    parser.add_argument("--model", required=True, help="HF model ID/path or API model name")
    parser.add_argument("--backend", choices=("local", "deepseek"), default="local")
    parser.add_argument("--data", type=Path, help="Defaults to the task test file under data/")
    parser.add_argument("--output", type=Path, required=True, help="Prediction JSONL path")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Defaults to 16 for classification and 512 for POS/DEPS",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-url", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def read_rows(path: Path, task: str, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id", len(rows)))
            if row_id in seen:
                raise ValueError(f"Duplicate id {row_id!r} at {path}:{line_number}")
            seen.add(row_id)
            row["id"] = row_id
            validate_row(row, task, path, line_number)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def validate_row(row: dict[str, Any], task: str, path: Path, line_number: int) -> None:
    required = {
        "nli": {"premise", "hypothesis", "label"},
        "sentiment": {"sentence", "label"},
        "ld": {"sentence", "label"},
        "laj": {"sentence", "label"},
        "wsd": {"target", "sentence1", "sentence2", "label"},
        "pos": {"tokens", "upos"},
        "deps": {"tokens", "heads", "deprels"},
    }[task]
    missing = required - set(row)
    if missing:
        raise ValueError(f"Missing {sorted(missing)} at {path}:{line_number}")
    if task == "pos" and len(row["tokens"]) != len(row["upos"]):
        raise ValueError(f"Token/tag length mismatch at {path}:{line_number}")
    if task == "deps" and not (
        len(row["tokens"]) == len(row["heads"]) == len(row["deprels"])
    ):
        raise ValueError(f"Dependency length mismatch at {path}:{line_number}")


def gold_value(row: dict[str, Any], task: str) -> Any:
    if task == "ld":
        return LD_LABELS[int(row["label"])]
    if task == "laj":
        return LAJ_LABELS[int(row["label"])]
    if task == "pos":
        return row["upos"]
    if task == "deps":
        return {"heads": row["heads"], "deprels": row["deprels"]}
    return row["label"]


def build_prompt(row: dict[str, Any], task: str, dep_labels: tuple[str, ...]) -> str:
    if task == "nli":
        return f"""你是一個香港粵語自然語言推理分類器。判斷「前提」係咪必然推出「假設」。
entailment 表示必然支持；not_entailment 包括矛盾、無關或資料不足。
只輸出 entailment 或 not_entailment，不要解釋。

前提：{row['premise']}
假設：{row['hypothesis']}
標籤："""
    if task == "sentiment":
        return f"""判斷以下香港粵語餐廳評論的整體情感。
smile：整體正面；ok：整體中性或正負混合；cry：整體負面。
只輸出 smile、ok 或 cry，不要解釋。

評論：{row['sentence']}
標籤："""
    if task == "ld":
        return f"""判斷句子的語言類型。
cantonese：香港粵語；mandarin：標準書面漢語；mixed：兩者混合。
繁簡字不是判斷標準。只輸出 cantonese、mandarin 或 mixed，不要解釋。

句子：{row['sentence']}
標籤："""
    if task == "laj":
        return f"""判斷以下句子在香港粵語中是否自然、可接受。
acceptable：自然可接受；unacceptable：不自然或有明顯語法問題。
只輸出 acceptable 或 unacceptable，不要解釋。

句子：{row['sentence']}
標籤："""
    if task == "wsd":
        return f"""判斷目標詞在兩個句子中的詞義是否相同。
similar：詞義相同；not_similar：詞義不同。
只輸出 similar 或 not_similar，不要解釋。

目標詞：{row['target']}
句子一：{row['sentence1']}
句子二：{row['sentence2']}
標籤："""
    if task == "pos":
        tokens = json.dumps(row["tokens"], ensure_ascii=False)
        return f"""Tag the pre-segmented Cantonese tokens with Universal Dependencies UPOS tags.
Allowed tags: {', '.join(POS_TAGS)}.
Return only a valid JSON array with exactly one tag per input token, in the same order.

Tokens: {tokens}
Tags:"""
    tokens = json.dumps(row["tokens"], ensure_ascii=False)
    return f"""Parse the pre-segmented Cantonese tokens using Universal Dependencies.
HEAD values are 1-based token indices; 0 marks the single root. Return exactly one
head and dependency relation per token. Allowed relations: {', '.join(dep_labels)}.
Return only valid JSON with this schema:
{{"heads":[3,0,2],"deprels":["nsubj","root","obj"]}}

Tokens: {tokens}
Parse:"""


def strip_code_fence(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_prediction(answer: str, task: str, length: int | None = None) -> tuple[Any, str]:
    if task in LABELS:
        normalized = strip_code_fence(answer).lower().replace("-", "_")
        choices = sorted(LABELS[task], key=len, reverse=True)
        alternation = "|".join(map(re.escape, choices))
        exact = re.fullmatch(
            rf"(?:label|標籤)?\s*[:：]?\s*({alternation})[.!。！\s]*", normalized
        )
        if exact:
            return exact.group(1), "valid"
        hits = re.findall(rf"(?<![a-z_])({alternation})(?![a-z_])", normalized)
        return (hits[0], "recovered") if len(set(hits)) == 1 else (None, "invalid")

    try:
        value = json.loads(strip_code_fence(answer))
    except json.JSONDecodeError:
        return None, "invalid_json"

    if task == "pos":
        if not isinstance(value, list) or not all(isinstance(tag, str) for tag in value):
            return None, "invalid_schema"
        tags = [tag.strip().upper() for tag in value]
        if len(tags) != length:
            return None, "wrong_length"
        if any(tag not in POS_TAGS for tag in tags):
            return None, "unknown_label"
        return tags, "valid"

    if not isinstance(value, dict) or set(value) != {"heads", "deprels"}:
        return None, "invalid_schema"
    heads, deprels = value["heads"], value["deprels"]
    if not isinstance(heads, list) or not isinstance(deprels, list):
        return None, "invalid_schema"
    if len(heads) != length or len(deprels) != length:
        return None, "wrong_length"
    if not all(isinstance(head, int) and 0 <= head <= length for head in heads):
        return None, "invalid_head"
    if not all(isinstance(label, str) for label in deprels):
        return None, "invalid_relation"
    return {"heads": heads, "deprels": [label.lower() for label in deprels]}, "valid"


def load_local_model(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before local evaluation") from exc

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": "auto",
        "trust_remote_code": args.trust_remote_code,
    }
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise SystemExit("--load-in-4bit requires CUDA and bitsandbytes")
        from transformers import BitsAndBytesConfig

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    if "gemma-3-" in args.model.lower() and "-1b-" not in args.model.lower():
        from transformers import Gemma3ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
        model = Gemma3ForConditionalGeneration.from_pretrained(args.model, **kwargs).eval()
        return {"kind": "gemma3", "handler": processor, "model": model, "torch": torch}

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    return {"kind": "causal", "handler": tokenizer, "model": model, "torch": torch}


def generate_local(state: dict[str, Any], prompt: str, args: argparse.Namespace) -> str:
    handler, model, torch = state["handler"], state["model"], state["torch"]
    if state["kind"] == "gemma3":
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = handler.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(model.device)
        decode = handler.decode
        pad_token_id = handler.tokenizer.pad_token_id or handler.tokenizer.eos_token_id
    else:
        messages = [{"role": "user", "content": prompt}]
        rendered = handler.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = handler(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(model.device)
        decode = handler.decode
        pad_token_id = handler.pad_token_id or handler.eos_token_id

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[-1]:]
    return decode(generated, skip_special_tokens=True).strip()


def generate_deepseek(prompt: str, args: argparse.Namespace, api_key: str) -> str:
    payload = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": args.max_new_tokens,
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(args.max_retries):
        request = urllib.request.Request(
            args.api_url,
            data=payload,
            method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
            return (result["choices"][0]["message"].get("content") or "").strip()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in (400, 401, 402, 403, 422):
                raise RuntimeError(f"API HTTP {exc.code}: {detail}") from exc
            last_error = RuntimeError(f"API HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        if attempt + 1 < args.max_retries:
            time.sleep(min(2**attempt, 16) + random.random())
    raise RuntimeError(f"API failed after {args.max_retries} attempts") from last_error


def classification_metrics(records: list[dict[str, Any]], labels: tuple[str, ...]) -> dict[str, Any]:
    correct = sum(row["prediction"] == row["gold"] for row in records)
    f1_values = []
    for label in labels:
        tp = sum(row["gold"] == label and row["prediction"] == label for row in records)
        fp = sum(row["gold"] != label and row["prediction"] == label for row in records)
        fn = sum(row["gold"] == label and row["prediction"] != label for row in records)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "accuracy": correct / len(records),
        "macro_f1": sum(f1_values) / len(f1_values),
    }


def token_metrics(records: list[dict[str, Any]], task: str) -> dict[str, Any]:
    total = correct = labeled_correct = sentence_exact = 0
    gold_tags: list[str] = []
    predicted_tags: list[str | None] = []
    for row in records:
        gold, prediction = row["gold"], row["prediction"]
        if task == "pos":
            gold_sequence = gold
            predicted_sequence = prediction if isinstance(prediction, list) else [None] * len(gold)
            gold_tags.extend(gold_sequence)
            predicted_tags.extend(predicted_sequence)
            total += len(gold_sequence)
            correct += sum(a == b for a, b in zip(gold_sequence, predicted_sequence))
            sentence_exact += int(gold_sequence == predicted_sequence)
        else:
            predicted_heads = prediction.get("heads", []) if isinstance(prediction, dict) else []
            predicted_rels = prediction.get("deprels", []) if isinstance(prediction, dict) else []
            sentence_ok = len(predicted_heads) == len(gold["heads"])
            for index, (gold_head, gold_rel) in enumerate(zip(gold["heads"], gold["deprels"])):
                pred_head = predicted_heads[index] if index < len(predicted_heads) else None
                pred_rel = predicted_rels[index] if index < len(predicted_rels) else None
                total += 1
                correct += int(pred_head == gold_head)
                labeled_correct += int(pred_head == gold_head and pred_rel == gold_rel)
                sentence_ok = sentence_ok and pred_head == gold_head and pred_rel == gold_rel
            sentence_exact += int(sentence_ok)
    if task == "deps":
        return {
            "uas": correct / total,
            "las": labeled_correct / total,
            "sentence_labeled_exact_match": sentence_exact / len(records),
        }

    f1_values = []
    for tag in POS_TAGS:
        tp = sum(gold == tag and pred == tag for gold, pred in zip(gold_tags, predicted_tags))
        fp = sum(gold != tag and pred == tag for gold, pred in zip(gold_tags, predicted_tags))
        fn = sum(gold == tag and pred != tag for gold, pred in zip(gold_tags, predicted_tags))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "token_accuracy": correct / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "sentence_exact_match": sentence_exact / len(records),
    }


def summarize(records: list[dict[str, Any]], task: str) -> dict[str, Any]:
    summary = {
        "samples": len(records),
        "invalid_predictions": sum(row["prediction"] is None for row in records),
        "parse_status": dict(Counter(row["parse_status"] for row in records)),
    }
    if task in LABELS:
        summary.update(classification_metrics(records, LABELS[task]))
    else:
        summary.update(token_metrics(records, task))
    summary["invalid_output_rate"] = summary["invalid_predictions"] / len(records)
    return summary


def read_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    completed[str(record["id"])] = record
    return completed


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be positive")
    if args.max_new_tokens is None:
        args.max_new_tokens = 512 if args.task in ("pos", "deps") else 16
    random.seed(args.seed)
    data_path = args.data or DEFAULT_DATA[args.task]
    rows = read_rows(data_path, args.task, args.max_samples)
    dep_labels = tuple(sorted({label for row in rows for label in row.get("deprels", [])}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(args.output) if args.resume else {}
    mode = "a" if args.resume and args.output.exists() else "w"

    if args.backend == "local":
        state = load_local_model(args)
        api_key = None
    else:
        state = None
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"Environment variable {args.api_key_env} is not set")

    started = time.perf_counter()
    generated = 0
    with args.output.open(mode, encoding="utf-8") as stream:
        for number, row in enumerate(rows, 1):
            if row["id"] in completed:
                continue
            prompt = build_prompt(row, args.task, dep_labels)
            sample_started = time.perf_counter()
            if state is not None:
                answer = generate_local(state, prompt, args)
            else:
                answer = generate_deepseek(prompt, args, api_key)
            prediction, parse_status = parse_prediction(
                answer, args.task, len(row.get("tokens", [])) or None
            )
            record = {
                "id": row["id"],
                "gold": gold_value(row, args.task),
                "prediction": prediction,
                "parse_status": parse_status,
                "raw_answer": answer,
                "latency_seconds": time.perf_counter() - sample_started,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            completed[row["id"]] = record
            generated += 1
            print(f"[{number}/{len(rows)}] id={row['id']} {parse_status}", flush=True)

    selected = [completed[row["id"]] for row in rows]
    summary = {
        "task": args.task,
        "model": args.model,
        "backend": args.backend,
        "data": str(data_path),
        "setting": "zero-shot-greedy",
        "seed": args.seed,
        "samples_generated_this_invocation": generated,
        "wall_time_seconds_this_invocation": time.perf_counter() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        **summarize(selected, args.task),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Predictions: {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
