#!/usr/bin/env python3
"""Shared data, parsing, and metrics for zero-shot Cantonese dependency parsing."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DEPRELS = (
    "acl", "advcl", "advcl:coverb", "advmod", "advmod:df", "amod",
    "appos", "aux", "case", "case:loc", "cc", "ccomp", "clf",
    "clf:det", "compound", "compound:dir", "compound:ext",
    "compound:quant", "compound:vo", "compound:vv", "conj", "cop",
    "csubj", "det", "discourse", "discourse:sp", "dislocated", "flat",
    "iobj", "mark", "mark:adv", "mark:rel", "nmod", "nsubj",
    "nsubj:pass", "nsubj:periph", "nummod", "obj", "obj:periph", "obl",
    "obl:agent", "obl:patient", "obl:tmod", "parataxis", "punct",
    "reparandum", "root", "vocative", "xcomp",
)
DEPREL_SET = set(DEPRELS)

SYSTEM_PROMPT = f"""You are a Cantonese Universal Dependencies dependency parser.

The input is a JSON object containing an already segmented token array. Token
positions are 1-based. For every token, predict:
1. its syntactic head as an integer, where 0 denotes ROOT and 1..N denote token
   positions; and
2. its dependency relation.

Return only one JSON object with exactly these two keys:
"heads": an integer array with exactly N entries
"deprels": a string array with exactly N entries

Allowed dependency relations:
{', '.join(DEPRELS)}

Constraints:
- heads and deprels must follow the input token order and each have length N.
- Exactly one head must be 0.
- A token cannot be its own head.
- The result must form one connected acyclic dependency tree.
- deprel must be "root" exactly when head is 0.
- Do not include Markdown, explanations, token text, or additional keys."""


def read_data(path, max_samples=None):
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not {"id", "tokens", "heads", "deprels"}.issubset(row):
                raise ValueError("Each row requires id, tokens, heads, and deprels")
            row["id"] = int(row["id"])
            if row["id"] in seen:
                raise ValueError(f"Duplicate id: {row['id']}")
            seen.add(row["id"])
            n = len(row["tokens"])
            if len(row["heads"]) != n or len(row["deprels"]) != n:
                raise ValueError(f"Length mismatch in id {row['id']}")
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def read_predictions(path):
    records = {}
    path = Path(path)
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    records[int(record["id"])] = record
    return records


def _has_cycle(heads):
    n = len(heads)
    for start in range(1, n + 1):
        seen = set()
        current = start
        while current != 0:
            if current < 1 or current > n:
                break
            if current in seen:
                return True
            seen.add(current)
            current = heads[current - 1]
    return False


def assess_prediction(value, expected_length):
    if not isinstance(value, dict):
        return None, "invalid_schema"
    heads = value.get("heads")
    deprels = value.get("deprels")
    if not isinstance(heads, list) or not isinstance(deprels, list):
        return None, "invalid_schema"
    if len(heads) != expected_length or len(deprels) != expected_length:
        return None, "wrong_length"
    if any(isinstance(head, bool) or not isinstance(head, int) for head in heads):
        return None, "invalid_head_type"
    if any(not isinstance(rel, str) for rel in deprels):
        return None, "invalid_relation_type"

    normalized = {
        "heads": [int(head) for head in heads],
        "deprels": [rel.strip().lower() for rel in deprels],
    }
    invalid_head = any(
        head < 0 or head > expected_length or head == index
        for index, head in enumerate(normalized["heads"], 1)
    )
    if invalid_head:
        return normalized, "invalid_head"
    if any(rel not in DEPREL_SET for rel in normalized["deprels"]):
        return normalized, "unknown_relation"

    roots = [index for index, head in enumerate(normalized["heads"]) if head == 0]
    root_consistent = all(
        (head == 0) == (rel == "root")
        for head, rel in zip(normalized["heads"], normalized["deprels"])
    )
    if len(roots) != 1 or not root_consistent:
        return normalized, "invalid_root"
    if _has_cycle(normalized["heads"]):
        return normalized, "invalid_tree"
    return normalized, "valid"


def strict_parse_answer(answer, expected_length):
    try:
        value = json.loads(answer.strip())
    except json.JSONDecodeError:
        return None, "invalid_json"
    return assess_prediction(value, expected_length)


def _extract_first_object(text):
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        return value
    except json.JSONDecodeError:
        return None


def recover_answer(answer, expected_length):
    strict_prediction, strict_status = strict_parse_answer(answer, expected_length)
    if strict_status == "valid":
        return strict_prediction, "valid_strict", strict_prediction, strict_status

    text = answer.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    value = _extract_first_object(text)
    if value is None:
        return strict_prediction, strict_status, strict_prediction, strict_status
    prediction, status = assess_prediction(value, expected_length)
    recovered_status = "valid_recovered" if status == "valid" else status
    return prediction, recovered_status, strict_prediction, strict_status


def _head_tree_is_valid(heads):
    if not isinstance(heads, list):
        return False
    n = len(heads)
    if any(
        isinstance(head, bool)
        or not isinstance(head, int)
        or head < 0
        or head > n
        or head == index
        for index, head in enumerate(heads, 1)
    ):
        return False
    return sum(head == 0 for head in heads) == 1 and not _has_cycle(heads)


def calculate_metrics(records, prediction_key):
    total = uas_correct = las_correct = rel_correct = 0
    non_punct_total = non_punct_uas = non_punct_las = 0
    sentence_uem = sentence_lem = 0
    valid_outputs = valid_head_trees = 0
    per_relation = {label: defaultdict(int) for label in DEPRELS}

    for record in records:
        gold_heads = record["heads"]
        gold_rels = record["deprels"]
        prediction = record.get(prediction_key)
        pred_heads = prediction.get("heads") if isinstance(prediction, dict) else None
        pred_rels = prediction.get("deprels") if isinstance(prediction, dict) else None
        aligned_heads = (
            isinstance(pred_heads, list)
            and len(pred_heads) == len(gold_heads)
            and all(isinstance(head, int) and not isinstance(head, bool) for head in pred_heads)
        )
        aligned_rels = (
            isinstance(pred_rels, list)
            and len(pred_rels) == len(gold_rels)
            and all(isinstance(rel, str) for rel in pred_rels)
        )
        if aligned_rels:
            pred_rels = [rel.strip().lower() for rel in pred_rels]

        head_tree_valid = aligned_heads and _head_tree_is_valid(pred_heads)
        valid_head_trees += int(head_tree_valid)
        output_valid = (
            head_tree_valid
            and aligned_rels
            and all(rel in DEPREL_SET for rel in pred_rels)
            and all(
                (head == 0) == (rel == "root")
                for head, rel in zip(pred_heads, pred_rels)
            )
        )
        valid_outputs += int(output_valid)

        heads_exact = aligned_heads and pred_heads == gold_heads
        labeled_exact = heads_exact and aligned_rels and pred_rels == gold_rels
        sentence_uem += int(heads_exact)
        sentence_lem += int(labeled_exact)

        for index, (gold_head, gold_rel) in enumerate(zip(gold_heads, gold_rels), 1):
            pred_head = pred_heads[index - 1] if aligned_heads else None
            pred_rel = pred_rels[index - 1] if aligned_rels else None
            head_ok = pred_head == gold_head
            rel_ok = pred_rel == gold_rel
            labeled_ok = head_ok and rel_ok
            total += 1
            uas_correct += int(head_ok)
            las_correct += int(labeled_ok)
            rel_correct += int(rel_ok)
            if gold_rel != "punct":
                non_punct_total += 1
                non_punct_uas += int(head_ok)
                non_punct_las += int(labeled_ok)
            per_relation[gold_rel]["support"] += 1
            per_relation[gold_rel]["uas_correct"] += int(head_ok)
            per_relation[gold_rel]["las_correct"] += int(labeled_ok)

    sentence_count = len(records)
    per_relation_metrics = {}
    for label in DEPRELS:
        support = per_relation[label]["support"]
        if support:
            per_relation_metrics[label] = {
                "uas": per_relation[label]["uas_correct"] / support,
                "las": per_relation[label]["las_correct"] / support,
                "support": support,
            }
    return {
        "sentences": sentence_count,
        "tokens": total,
        "uas": uas_correct / total if total else None,
        "las": las_correct / total if total else None,
        "relation_accuracy": rel_correct / total if total else None,
        "non_punct_tokens": non_punct_total,
        "non_punct_uas": non_punct_uas / non_punct_total if non_punct_total else None,
        "non_punct_las": non_punct_las / non_punct_total if non_punct_total else None,
        "sentence_unlabeled_exact_match": sentence_uem / sentence_count if sentence_count else None,
        "sentence_labeled_exact_match": sentence_lem / sentence_count if sentence_count else None,
        "valid_outputs": valid_outputs,
        "invalid_outputs": sentence_count - valid_outputs,
        "invalid_output_rate": (sentence_count - valid_outputs) / sentence_count if sentence_count else None,
        "well_formed_head_trees": valid_head_trees,
        "well_formed_head_tree_rate": valid_head_trees / sentence_count if sentence_count else None,
        "per_relation": per_relation_metrics,
    }
