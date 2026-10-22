#!/usr/bin/env python3
"""Fine-tune a multilingual encoder with a dependency parsing head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from deps_eval_common import calculate_metrics


EVALUATOR_VERSION = "encoder-deps-greedy-tree-v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--evaluator-version", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--arc-dim", type=int, default=256)
    parser.add_argument("--rel-dim", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    return parser.parse_args()


def load_rows(path):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    seen = set()
    for row in rows:
        if not {"id", "tokens", "heads", "deprels"}.issubset(row):
            raise ValueError("Each row requires id, tokens, heads, and deprels")
        row["id"] = int(row["id"])
        if row["id"] in seen:
            raise ValueError(f"Duplicate id: {row['id']}")
        seen.add(row["id"])
        n = len(row["tokens"])
        if len(row["heads"]) != n or len(row["deprels"]) != n:
            raise ValueError(f"Length mismatch in id {row['id']}")
        if any(head < 0 or head > n for head in row["heads"]):
            raise ValueError(f"Out-of-range head in id {row['id']}")
    return rows


def token_key(row):
    return "\x1f".join(row["tokens"])


def split_train_validation(rows, seed, validation_ratio):
    groups = [token_key(row) for row in rows]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_ratio, random_state=seed
    )
    train_indices, validation_indices = next(splitter.split(rows, groups=groups))
    return {
        "train": [rows[index] for index in train_indices],
        "validation": [rows[index] for index in validation_indices],
    }


def prepare_examples(rows, tokenizer, rel2id, max_length):
    examples = []
    for row in rows:
        safe_tokens = [
            token if tokenizer.tokenize(token) else tokenizer.unk_token
            for token in row["tokens"]
        ]
        encoded = tokenizer(
            safe_tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        word_ids = encoded.word_ids()
        first_positions = {}
        for token_index, word_index in enumerate(word_ids):
            if word_index is not None and word_index not in first_positions:
                first_positions[word_index] = token_index
        missing = sorted(set(range(len(row["tokens"]))) - set(first_positions))
        if missing:
            raise ValueError(
                f"max_length={max_length} omitted word indexes {missing} in id {row['id']}"
            )
        examples.append(
            {
                "row": row,
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "word_positions": [
                    first_positions[index] for index in range(len(row["tokens"]))
                ],
                "heads": [int(head) for head in row["heads"]],
                "rels": [rel2id[rel] for rel in row["deprels"]],
            }
        )
    return examples


class DependencyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        encoded = self.tokenizer.pad(
            [
                {
                    "input_ids": example["input_ids"],
                    "attention_mask": example["attention_mask"],
                }
                for example in examples
            ],
            padding=True,
            return_tensors="pt",
        )
        max_words = max(len(example["word_positions"]) for example in examples)
        batch_size = len(examples)
        word_positions = torch.zeros((batch_size, max_words), dtype=torch.long)
        word_mask = torch.zeros((batch_size, max_words), dtype=torch.bool)
        heads = torch.full((batch_size, max_words), -100, dtype=torch.long)
        rels = torch.full((batch_size, max_words), -100, dtype=torch.long)
        for index, example in enumerate(examples):
            n = len(example["word_positions"])
            word_positions[index, :n] = torch.tensor(example["word_positions"])
            word_mask[index, :n] = True
            heads[index, :n] = torch.tensor(example["heads"])
            rels[index, :n] = torch.tensor(example["rels"])
        return {
            **encoded,
            "word_positions": word_positions,
            "word_mask": word_mask,
            "heads": heads,
            "rels": rels,
            "rows": [example["row"] for example in examples],
        }


class DependencyParser(nn.Module):
    def __init__(self, model_name, num_relations, arc_dim, rel_dim):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        dropout_probability = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_probability)
        self.arc_dep = nn.Linear(hidden_size, arc_dim)
        self.arc_head = nn.Linear(hidden_size, arc_dim)
        self.rel_dep = nn.Linear(hidden_size, rel_dim)
        self.rel_head = nn.Linear(hidden_size, rel_dim)
        self.rel_classifier = nn.Linear(rel_dim * 2, num_relations)
        self.arc_scale = math.sqrt(arc_dim)

    def encode_and_score(self, input_ids, attention_mask, word_positions, word_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        hidden = self.dropout(hidden)
        gather_index = word_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        word_repr = hidden.gather(1, gather_index)
        root_repr = hidden[:, :1, :]
        candidates = torch.cat([root_repr, word_repr], dim=1)
        dep_arc = torch.relu(self.arc_dep(word_repr))
        head_arc = torch.relu(self.arc_head(candidates))
        arc_scores = torch.bmm(dep_arc, head_arc.transpose(1, 2)) / self.arc_scale

        batch_size, max_words = word_mask.shape
        word_counts = word_mask.sum(dim=1)
        candidate_indices = torch.arange(max_words + 1, device=word_mask.device)
        candidate_mask = candidate_indices.unsqueeze(0) <= word_counts.unsqueeze(1)
        arc_scores = arc_scores.masked_fill(~candidate_mask.unsqueeze(1), -1e4)
        dependent_indices = torch.arange(max_words, device=word_mask.device)
        arc_scores[:, dependent_indices, dependent_indices + 1] = -1e4
        return word_repr, candidates, arc_scores

    def relation_scores(self, word_repr, candidates, selected_heads):
        safe_heads = selected_heads.clamp(min=0, max=candidates.size(1) - 1)
        gather_index = safe_heads.unsqueeze(-1).expand(-1, -1, candidates.size(-1))
        head_repr = candidates.gather(1, gather_index)
        dep_rel = torch.relu(self.rel_dep(word_repr))
        head_rel = torch.relu(self.rel_head(head_repr))
        return self.rel_classifier(torch.cat([dep_rel, head_rel], dim=-1))

    def forward(
        self, input_ids, attention_mask, word_positions, word_mask, heads, rels
    ):
        word_repr, candidates, arc_scores = self.encode_and_score(
            input_ids, attention_mask, word_positions, word_mask
        )
        relation_logits = self.relation_scores(word_repr, candidates, heads)
        arc_loss = F.cross_entropy(
            arc_scores.reshape(-1, arc_scores.size(-1)),
            heads.reshape(-1),
            ignore_index=-100,
        )
        rel_loss = F.cross_entropy(
            relation_logits.reshape(-1, relation_logits.size(-1)),
            rels.reshape(-1),
            ignore_index=-100,
        )
        return {
            "loss": arc_loss + rel_loss,
            "arc_loss": arc_loss,
            "rel_loss": rel_loss,
            "word_repr": word_repr,
            "candidates": candidates,
            "arc_scores": arc_scores,
        }


def find_cycle(heads):
    n = len(heads)
    for start in range(1, n + 1):
        order = []
        position = {}
        current = start
        while current != 0 and current not in position:
            position[current] = len(order)
            order.append(current)
            current = heads[current - 1]
        if current in position:
            return order[position[current] :]
    return None


def decode_tree(score_matrix):
    """Greedy single-root decoding with deterministic cycle repair."""
    scores = np.asarray(score_matrix, dtype=np.float64)
    n = scores.shape[0]
    non_root_heads = []
    root_advantages = []
    for dependent in range(n):
        candidates = [head for head in range(1, n + 1) if head != dependent + 1]
        best_head = max(candidates, key=lambda head: scores[dependent, head])
        non_root_heads.append(best_head)
        root_advantages.append(scores[dependent, 0] - scores[dependent, best_head])
    root_dependent = int(np.argmax(root_advantages))
    heads = list(non_root_heads)
    heads[root_dependent] = 0

    cycle = find_cycle(heads)
    repairs = 0
    while cycle:
        # Reattach one cycle member directly to the single ROOT word. Because the
        # ROOT word points to 0, this always breaks the cycle without creating a
        # new one elsewhere. Choose the least costly such change.
        root_word = root_dependent + 1
        best_change = None
        for dependent_word in cycle:
            dependent = dependent_word - 1
            loss = scores[dependent, heads[dependent]] - scores[dependent, root_word]
            candidate = (loss, dependent, root_word)
            if best_change is None or candidate < best_change:
                best_change = candidate
        if best_change is None:
            raise RuntimeError("Unable to repair dependency cycle")
        _, dependent, head = best_change
        heads[dependent] = head
        repairs += 1
        if repairs > n:
            raise RuntimeError("Dependency cycle repair exceeded sentence length")
        cycle = find_cycle(heads)
    return heads


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def evaluate(model, loader, id2rel, device, use_bf16):
    model.eval()
    records = []
    losses = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
            ):
                output = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["word_positions"],
                    batch["word_mask"],
                    batch["heads"],
                    batch["rels"],
                )
            losses.append(float(output["loss"].detach().cpu()))
            arc_scores = output["arc_scores"].float().cpu().numpy()
            decoded = []
            for index, row in enumerate(batch["rows"]):
                n = len(row["tokens"])
                decoded.append(decode_tree(arc_scores[index, :n, : n + 1]))
            padded_heads = torch.zeros_like(batch["heads"])
            for index, heads in enumerate(decoded):
                padded_heads[index, : len(heads)] = torch.tensor(
                    heads, device=device, dtype=torch.long
                )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
            ):
                relation_logits = model.relation_scores(
                    output["word_repr"], output["candidates"], padded_heads
                )
            predicted_relations = relation_logits.argmax(dim=-1).cpu().numpy()
            for index, row in enumerate(batch["rows"]):
                n = len(row["tokens"])
                prediction = {
                    "heads": decoded[index],
                    "deprels": [
                        id2rel[int(label)] for label in predicted_relations[index, :n]
                    ],
                }
                records.append({**row, "prediction": prediction})
    metrics = calculate_metrics(records, "prediction")
    metrics["loss"] = float(np.mean(losses)) if losses else None
    return metrics, records


def main():
    args = parse_args()
    wall_started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(args.train_data)
    test_rows = load_rows(args.test_data)
    train_keys = {token_key(row) for row in train_rows}
    if any(token_key(row) in train_keys for row in test_rows):
        raise ValueError("Train/test exact-token overlap detected")
    split = split_train_validation(train_rows, args.seed, args.validation_ratio)
    split["test"] = test_rows
    all_relations = sorted({rel for row in train_rows for rel in row["deprels"]})
    test_only = sorted(
        {rel for row in test_rows for rel in row["deprels"]} - set(all_relations)
    )
    if test_only:
        raise ValueError(f"Test-only dependency relations: {test_only}")
    rel2id = {label: index for index, label in enumerate(all_relations)}
    id2rel = {index: label for label, index in rel2id.items()}

    manifest = {
        "evaluator_version": args.evaluator_version,
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "protocol": (
            "Group-aware train/validation split; exact duplicate token sequences stay "
            "together; supplied test is untouched and evaluated once."
        ),
        "sizes": {name: len(rows) for name, rows in split.items()},
        "ids": {name: [row["id"] for row in rows] for name, rows in split.items()},
        "relations": all_relations,
        "source_sha256": {
            "train": hashlib.sha256(args.train_data.read_bytes()).hexdigest(),
            "test": hashlib.sha256(args.test_data.read_bytes()).hexdigest(),
        },
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    run_split = {name: list(rows) for name, rows in split.items()}
    if args.max_train_samples is not None:
        run_split["train"] = run_split["train"][: args.max_train_samples]
    if args.max_eval_samples is not None:
        run_split["validation"] = run_split["validation"][: args.max_eval_samples]
        run_split["test"] = run_split["test"][: args.max_eval_samples]

    tokenizer_kwargs = {"use_fast": True}
    if "xlm-roberta" in args.model.lower():
        tokenizer_kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    prepared = {
        name: prepare_examples(rows, tokenizer, rel2id, args.max_length)
        for name, rows in run_split.items()
    }
    collator = DependencyCollator(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        prepared["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    validation_loader = DataLoader(
        prepared["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        prepared["test"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = DependencyParser(
        args.model, len(rel2id), args.arc_dim, args.rel_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    checkpoint_path = args.output_dir / "best_checkpoint.pt"
    best_validation_las = -1.0
    best_epoch = None
    history = []
    train_started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
            ):
                output = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["word_positions"],
                    batch["word_mask"],
                    batch["heads"],
                    batch["rels"],
                )
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(output["loss"].detach().cpu()))

        validation_metrics, _ = evaluate(
            model, validation_loader, id2rel, device, use_bf16
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_uas": validation_metrics["uas"],
            "validation_las": validation_metrics["las"],
            "validation_loss": validation_metrics["loss"],
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)
        if validation_metrics["las"] > best_validation_las:
            best_validation_las = validation_metrics["las"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "rel2id": rel2id,
                    "model_name": args.model,
                    "arc_dim": args.arc_dim,
                    "rel_dim": args.rel_dim,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

    train_seconds = time.time() - train_started
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_metrics, _ = evaluate(
        model, validation_loader, id2rel, device, use_bf16
    )
    test_started = time.time()
    test_metrics, test_records = evaluate(
        model, test_loader, id2rel, device, use_bf16
    )
    test_seconds = time.time() - test_started

    predictions_path = args.output_dir / "test_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as stream:
        for record in test_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    best_model_dir = args.output_dir / "best_model"
    best_model_dir.mkdir(exist_ok=True)
    model.encoder.save_pretrained(best_model_dir / "encoder")
    tokenizer.save_pretrained(best_model_dir)
    parser_head_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("encoder.")
    }
    torch.save(parser_head_state, best_model_dir / "parser_heads.pt")
    (best_model_dir / "parser_config.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "relations": all_relations,
                "arc_dim": args.arc_dim,
                "rel_dim": args.rel_dim,
                "decoder": "greedy-single-root-cycle-repair",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "model": args.model,
        "evaluator_version": args.evaluator_version,
        "protocol": (
            "supervised encoder fine-tuning; best checkpoint selected by validation "
            "LAS; greedy single-root acyclic decoding; held-out test evaluated once"
        ),
        "metrics_policy": "Primary UAS/LAS include punctuation.",
        "split_sizes": {name: len(rows) for name, rows in run_split.items()},
        "full_split_sizes": {name: len(rows) for name, rows in split.items()},
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "arc_dim": args.arc_dim,
            "rel_dim": args.rel_dim,
            "max_length": args.max_length,
            "validation_ratio": args.validation_ratio,
            "seed": args.seed,
        },
        "best_checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation_las": best_validation_las,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "invalid_outputs": 0,
        "invalid_output_rate": 0.0,
        "history": history,
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "wall_time_seconds": time.time() - wall_started,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else None
        ),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
