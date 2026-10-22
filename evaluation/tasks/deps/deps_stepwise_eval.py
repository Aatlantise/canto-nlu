#!/usr/bin/env python3
"""Paper-inspired three-step TSV dependency evaluation for local and API LLMs."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from deps_eval_common import read_data, read_predictions
from deps_stepwise_common import (
    SYSTEM_PROMPT,
    build_user_prompt,
    calculate_stepwise_metrics,
    parse_stepwise_answer,
)


API_URL = "https://api.deepseek.com/chat/completions"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("local", "gemma", "deepseek"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--evaluator-version", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_backend(args):
    if args.backend == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return {"kind": "deepseek", "api_key": api_key}, "provider-api", 0.0

    import torch
    from transformers import BitsAndBytesConfig

    kwargs = {"device_map": "auto"}
    if args.load_in_4bit:
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["torch_dtype"] = compute_dtype
        precision = f"bitsandbytes-nf4-{str(compute_dtype).replace('torch.', '')}"
    else:
        kwargs["torch_dtype"] = "auto"
        precision = "auto"

    started = time.perf_counter()
    if args.backend == "gemma":
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model, **kwargs
        ).eval()
        processor = AutoProcessor.from_pretrained(args.model)
        state = {"kind": "gemma", "model": model, "processor": processor}
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
        if not args.load_in_4bit:
            precision = str(model.dtype).replace("torch.", "")
        state = {"kind": "local", "model": model, "tokenizer": tokenizer}
    return state, precision, time.perf_counter() - started


def _generate_local(state, user_prompt, args):
    import torch

    model = state["model"]
    if state["kind"] == "gemma":
        processor = state["processor"]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        model_inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(model.device)
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id
        decode = processor.decode
    else:
        tokenizer = state["tokenizer"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(model.device)
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        decode = tokenizer.decode

    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    latency = time.perf_counter() - started
    generated = output[0, model_inputs["input_ids"].shape[-1] :]
    answer = decode(generated, skip_special_tokens=True).strip()
    return answer, latency, {}


def _generate_deepseek(state, user_prompt, args):
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": args.max_new_tokens,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = None
    for attempt in range(args.max_retries):
        request = urllib.request.Request(
            API_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {state['api_key']}",
                "Content-Type": "application/json",
            },
        )
        try:
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
            latency = time.perf_counter() - started
            choice = result["choices"][0]
            usage = result.get("usage") or {}
            metadata = {
                "served_model": result.get("model"),
                "system_fingerprint": result.get("system_fingerprint"),
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
            return (choice["message"].get("content") or "").strip(), latency, metadata
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code in (400, 401, 402, 422):
                raise RuntimeError(f"DeepSeek API HTTP {error.code}: {detail}") from error
            last_error = RuntimeError(f"DeepSeek API HTTP {error.code}: {detail}")
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
        if attempt + 1 < args.max_retries:
            delay = min(2**attempt, 16) + random.random()
            print(f"retrying in {delay:.1f}s: {last_error}", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"API failed after {args.max_retries} attempts") from last_error


def generate_prediction(state, row, args):
    user_prompt = build_user_prompt(row["tokens"])
    if state["kind"] == "deepseek":
        answer, latency, metadata = _generate_deepseek(state, user_prompt, args)
    else:
        answer, latency, metadata = _generate_local(state, user_prompt, args)
    prediction, status, details = parse_stepwise_answer(answer, row["tokens"])
    return {
        "raw_answer": answer,
        "prediction": prediction,
        "parse_status": status,
        "parse_details": details,
        "latency_seconds": latency,
        **metadata,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None

    rows = read_data(args.data, args.max_samples)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prior = read_predictions(output_path) if args.resume else {}
    mode = "a" if args.resume and output_path.exists() else "w"
    run_started = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    state, precision, load_seconds = load_backend(args)
    inference_seconds = 0.0
    generated_count = 0

    with output_path.open(mode, encoding="utf-8") as stream:
        for number, row in enumerate(rows, 1):
            row_id = int(row["id"])
            if row_id in prior:
                continue
            print(f"[{number}/{len(rows)}]", end=" ", flush=True)
            result = generate_prediction(state, row, args)
            inference_seconds += result["latency_seconds"]
            generated_count += 1
            record = {
                **row,
                **result,
                "requested_model": args.model,
                "precision": precision,
                "seed": args.seed,
                "setting": "zero-shot-greedy-three-step-tsv",
                "evaluator_version": args.evaluator_version,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            prior[row_id] = record
            print(result["parse_status"], result["parse_details"], flush=True)

    selected = [prior[int(row["id"])] for row in rows]
    wall_seconds = time.perf_counter() - wall_started
    summary = {
        "requested_model": args.model,
        "precision": precision,
        "setting": "zero-shot-greedy-three-step-tsv",
        "evaluator_version": args.evaluator_version,
        "protocol_note": (
            "Paper-inspired post-hoc protocol: UPOS, then HEAD, then DEPREL in one "
            "response using minimal CoNLL-U-like TSV. It is separate from v1 JSON."
        ),
        "metrics_policy": (
            "Primary UAS/LAS include punctuation. Rows are aligned by ID and exact FORM; "
            "missing or malformed token rows score as incorrect only for those tokens."
        ),
        **calculate_stepwise_metrics(selected),
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "model_load_seconds": load_seconds,
        "inference_seconds_this_invocation": inference_seconds,
        "wall_time_seconds_this_invocation": wall_seconds,
        "samples_generated_this_invocation": generated_count,
    }
    if args.backend == "deepseek":
        summary.update(
            {
                "served_models": sorted({str(r.get("served_model")) for r in selected}),
                "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in selected),
                "completion_tokens": sum(r.get("completion_tokens") or 0 for r in selected),
                "total_tokens": sum(r.get("total_tokens") or 0 for r in selected),
            }
        )
    elif torch is not None and torch.cuda.is_available():
        summary["gpu"] = torch.cuda.get_device_name(0)
        summary["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Predictions: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
