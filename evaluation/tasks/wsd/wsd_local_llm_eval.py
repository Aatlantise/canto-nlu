#!/usr/bin/env python3
"""Zero-shot WSD evaluation for Qwen, Gemma, or Llama on CantoNLU JSONL."""

import argparse, json, random, re, time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

LABELS = ("similar", "not_similar")
SYSTEM = (
    "你是一個香港粵語詞義辨析分類器。你必須只輸出一個標籤："
    "similar 或 not_similar，不要解釋。"
)
PROMPT = """判斷目標詞在兩個句子中的詞義是否相同。
similar：目標詞在兩句中的詞義相同。
not_similar：目標詞在兩句中的詞義不同。

目標詞：{target}
句子一：{sentence1}
句子二：{sentence2}

只輸出 similar 或 not_similar："""


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--family", choices=("qwen", "gemma", "llama"), required=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument(
        "--samples-per-label",
        type=int,
        help="Deterministically select this many examples from each label (for a balanced smoke test).",
    )
    p.add_argument("--max-input-tokens", type=int, default=3072)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def label(text):
    value = text.strip().lower().replace("-", "_")
    exact = re.fullmatch(r"(?:label\s*:\s*)?(not_similar|similar)[.!。！\s]*", value)
    if exact:
        return exact.group(1)
    hits = re.findall(r"(?<![a-z_])(not_similar|similar)(?![a-z_])", value)
    return hits[0] if len(set(hits)) == 1 else None


def load_rows(path, limit, samples_per_label):
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    need = {"id", "target", "sentence1", "sentence2", "label"}
    if not rows or not need.issubset(rows[0]):
        raise ValueError(f"JSONL requires {sorted(need)}")
    unknown = sorted({x["label"] for x in rows} - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    if samples_per_label:
        selected = []
        for name in LABELS:
            group = [row for row in rows if row["label"] == name]
            if len(group) < samples_per_label:
                raise ValueError(f"Only {len(group)} examples available for {name}")
            selected.extend(group[:samples_per_label])
        rows = sorted(selected, key=lambda row: int(row["id"]))
    return rows[:limit] if limit else rows


def prior(path):
    done = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line); done[int(row["id"])] = row
    return done


def scores(gold, pred):
    report = {}
    f1s = []
    for name in LABELS:
        tp = sum(g == name and p == name for g, p in zip(gold, pred))
        fp = sum(g != name and p == name for g, p in zip(gold, pred))
        fn = sum(g == name and p != name for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(g == name for g in gold)}
        f1s.append(f1)
    return sum(g == p for g, p in zip(gold, pred)) / len(gold), sum(f1s) / len(f1s), report


def main():
    a = args(); random.seed(a.seed); torch.manual_seed(a.seed)
    rows = load_rows(a.data, a.max_samples, a.samples_per_label)
    if a.family == "gemma":
        from transformers import Gemma3ForConditionalGeneration
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = Gemma3ForConditionalGeneration.from_pretrained(a.model, device_map="auto", quantization_config=quant, torch_dtype=torch.bfloat16).eval()
        handler = AutoProcessor.from_pretrained(a.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, device_map="auto", torch_dtype="auto").eval()
        handler = AutoTokenizer.from_pretrained(a.model)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    done = prior(out) if a.resume else {}
    started = time.perf_counter()
    with out.open("a" if a.resume else "w", encoding="utf-8") as f:
        for row in tqdm(rows):
            if int(row["id"]) in done: continue
            user = PROMPT.format(**row)
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            if a.family == "gemma":
                messages = [{"role": x["role"], "content": [{"type": "text", "text": x["content"]}]} for x in messages]
                inp = handler.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", truncation=True, max_length=a.max_input_tokens).to(model.device)
                pad = handler.tokenizer.pad_token_id
            else:
                text = handler.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inp = handler(text, return_tensors="pt", truncation=True, max_length=a.max_input_tokens).to(model.device)
                pad = handler.eos_token_id
            tick = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inp, max_new_tokens=8, do_sample=False, pad_token_id=pad)
            answer = handler.decode(generated[0, inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            rec = {**row, "raw_answer": answer, "prediction": label(answer), "latency_seconds": time.perf_counter() - tick}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); done[int(row["id"])] = rec
    selected = [done[int(x["id"])] for x in rows]
    gold = [x["label"] for x in selected]; pred = [x["prediction"] for x in selected]
    acc, macro, report = scores(gold, pred)
    summary = {"model": a.model, "setting": "zero-shot-greedy", "samples": len(rows), "class_distribution": dict(Counter(gold)), "invalid_predictions": sum(x is None for x in pred), "invalid_output_rate": sum(x is None for x in pred)/len(pred), "accuracy": acc, "macro_f1": macro, "classification_report": report, "seed": a.seed, "max_input_tokens": a.max_input_tokens, "max_new_tokens": 8, "quantization": "4-bit NF4 double quantization, BF16 compute" if a.family == "gemma" else "none, automatic dtype", "wall_seconds": time.perf_counter()-started, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    summary_path = out.with_suffix(".summary.json"); summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)); print(f"Predictions: {out}\nSummary: {summary_path}")


if __name__ == "__main__": main()
