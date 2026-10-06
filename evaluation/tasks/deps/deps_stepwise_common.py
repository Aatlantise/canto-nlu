#!/usr/bin/env python3
"""Paper-inspired step-by-step TSV prompting and evaluation for Cantonese DEPS."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from deps_eval_common import DEPRELS, DEPREL_SET, _has_cycle, assess_prediction


UPOS = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)
UPOS_SET = set(UPOS)

SYSTEM_PROMPT = f"""You are a Cantonese linguist specializing in Universal Dependencies.

The input has already been segmented into tokens. Preserve every token exactly and
perform all three tasks in one response, in this order: UPOS, HEAD, DEPREL.

Use plain TSV rows separated by newlines. Use one literal TAB between fields. Do not
use JSON, Markdown code fences, prose, a header row, or extra commentary.

Allowed UPOS tags:
{', '.join(UPOS)}

Allowed dependency relations:
{', '.join(DEPRELS)}

HEAD uses 1-based token indices; 0 denotes the single ROOT. A token cannot be its
own head. The final heads must form one connected acyclic tree. DEPREL must be
"root" exactly when HEAD is 0.
"""


def build_user_prompt(tokens):
    n = len(tokens)
    sentence = " ".join(tokens)
    words = "\n".join(tokens)
    return f"""We will now perform dependency parsing on one Cantonese sentence.
The tokenization below is fixed. There are exactly {n} tokens.

Execute the following three tasks in a single response:

- Task 1
Create exactly {n} TSV rows with three fields:
ID<TAB>FORM<TAB>UPOS

- Task 2
Repeat exactly the same {n} rows and add HEAD as the fourth field:
ID<TAB>FORM<TAB>UPOS<TAB>HEAD

- Task 3
Repeat exactly the same {n} rows and add DEPREL as the fifth field:
ID<TAB>FORM<TAB>UPOS<TAB>HEAD<TAB>DEPREL

Write the section labels exactly as "- Task 1", "- Task 2", and "- Task 3".
IDs must run from 1 through {n}. FORM must exactly copy the corresponding input
token. The main predicate must have HEAD 0 and DEPREL root.

input sentence:
{sentence}

words:
{words}"""


_TASK_RE = re.compile(r"^\s*(?:[-*#]+\s*)?task\s*([123])\s*:?\s*$", re.I)


def _normalize_line(line):
    line = line.strip()
    if not line or line.startswith("```") or line == "`":
        return ""
    if line.startswith("|") and line.endswith("|"):
        line = line[1:-1].strip()
    return line.replace("|", "\t")


def _split_row(line, tokens):
    line = _normalize_line(line)
    match = re.match(r"^(\d+)\s+(.+)$", line)
    if not match:
        return None
    index = int(match.group(1))
    if not 1 <= index <= len(tokens):
        return None
    expected_form = str(tokens[index - 1])
    remainder = match.group(2).strip()
    if remainder == expected_form:
        tail = ""
    elif remainder.startswith(expected_form):
        boundary = len(expected_form)
        if boundary >= len(remainder) or not remainder[boundary].isspace():
            return None
        tail = remainder[boundary:].strip()
    else:
        return None
    return index, expected_form, tail.split()


def _parse_row_strict(line, task, tokens):
    split = _split_row(line, tokens)
    if split is None:
        return None
    index, expected_form, fields = split
    if task == 1 and len(fields) == 1:
        return index, {"form": expected_form, "upos": fields[0].upper()}
    if task == 2 and len(fields) == 2:
        try:
            head = int(fields[1])
        except ValueError:
            return None
        return index, {
            "form": expected_form,
            "upos": fields[0].upper(),
            "head": head,
        }
    if task == 3 and len(fields) == 3:
        try:
            head = int(fields[1])
        except ValueError:
            return None
        return index, {
            "form": expected_form,
            "upos": fields[0].upper(),
            "head": head,
            "deprel": fields[2].lower(),
        }
    return None


def _rightmost_int(fields):
    for position in range(len(fields) - 1, -1, -1):
        try:
            return position, int(fields[position])
        except ValueError:
            continue
    return None, None


def _parse_row_recovered(line, task, tokens):
    """Recover over-complete TSV rows while preserving ID and exact FORM."""
    split = _split_row(line, tokens)
    if split is None:
        return None
    index, expected_form, fields = split
    if not fields:
        return None

    upos_position = next(
        (i for i, field in enumerate(fields) if field.upper() in UPOS_SET), None
    )
    if upos_position is None:
        upos_position = 0
    upos = fields[upos_position].upper()
    if task == 1:
        return index, {"form": expected_form, "upos": upos}

    if task == 2:
        _, head = _rightmost_int(fields[upos_position + 1 :])
        if head is None:
            return None
        return index, {"form": expected_form, "upos": upos, "head": head}

    # For Task 3, take the rightmost relation-looking field as DEPREL. If no
    # allowed relation exists, retain the last field so unknown labels remain
    # visible to the evaluator instead of silently disappearing.
    relation_position = next(
        (
            i
            for i in range(len(fields) - 1, upos_position, -1)
            if fields[i].lower() in DEPREL_SET
        ),
        len(fields) - 1,
    )
    deprel = fields[relation_position].lower()
    _, head = _rightmost_int(fields[upos_position + 1 : relation_position])
    if head is None:
        return None
    return index, {
        "form": expected_form,
        "upos": upos,
        "head": head,
        "deprel": deprel,
    }


def parse_stepwise_answer(answer, tokens):
    """Recover task rows by their token ID and exact FORM, as in the paper."""
    strict_rows = {1: {}, 2: {}, 3: {}}
    rows = {1: {}, 2: {}, 3: {}}
    current_task = None
    saw_task_heading = False
    all_lines = answer.splitlines()
    for raw_line in all_lines:
        normalized = _normalize_line(raw_line)
        if not normalized:
            continue
        heading = _TASK_RE.match(normalized)
        if heading:
            current_task = int(heading.group(1))
            saw_task_heading = True
            continue
        if current_task is None:
            continue
        strict = _parse_row_strict(normalized, current_task, tokens)
        if strict is not None:
            index, value = strict
            strict_rows[current_task][index] = value
        recovered = _parse_row_recovered(normalized, current_task, tokens)
        if recovered is not None:
            index, value = recovered
            rows[current_task][index] = value

    # Recovery fallback: accept bare five-field final rows when section labels were
    # omitted. Never scan other sections as Task 3 when headings are present.
    if not saw_task_heading:
        for raw_line in all_lines:
            strict = _parse_row_strict(raw_line, 3, tokens)
            if strict is not None:
                index, value = strict
                strict_rows[3][index] = value
            recovered = _parse_row_recovered(raw_line, 3, tokens)
            if recovered is not None:
                index, value = recovered
                rows[3][index] = value

    n = len(tokens)
    final_rows = rows[3]
    upos = [final_rows[i]["upos"] if i in final_rows else None for i in range(1, n + 1)]
    heads = [final_rows[i]["head"] if i in final_rows else None for i in range(1, n + 1)]
    deprels = [
        final_rows[i]["deprel"] if i in final_rows else None
        for i in range(1, n + 1)
    ]
    prediction = {"upos": upos, "heads": heads, "deprels": deprels}
    details = {
        "task1_rows": len(rows[1]),
        "task2_rows": len(rows[2]),
        "task3_rows": len(rows[3]),
        "strict_task1_rows": len(strict_rows[1]),
        "strict_task2_rows": len(strict_rows[2]),
        "strict_task3_rows": len(strict_rows[3]),
        "complete_three_step_format": all(
            len(strict_rows[t]) == n for t in (1, 2, 3)
        ),
        "complete_recovered_three_step": all(len(rows[t]) == n for t in (1, 2, 3)),
        "recovery_used": any(len(rows[t]) != len(strict_rows[t]) for t in (1, 2, 3)),
        "unknown_upos_rows": sum(tag not in UPOS_SET for tag in upos if tag is not None),
    }

    if len(final_rows) == 0:
        status = "no_final_rows"
    elif len(final_rows) != n:
        status = "partial_final_rows"
    else:
        structural, structural_status = assess_prediction(
            {"heads": heads, "deprels": deprels}, n
        )
        if structural_status == "valid":
            if any(tag not in UPOS_SET for tag in upos):
                status = "valid_tree_unknown_upos"
            elif details["recovery_used"]:
                status = "valid_recovered_table"
            else:
                status = "valid_table"
        else:
            status = structural_status
        if structural is not None:
            prediction["heads"] = structural["heads"]
            prediction["deprels"] = structural["deprels"]
    return prediction, status, details


def _complete_valid_tree(heads, deprels):
    n = len(heads)
    if not all(isinstance(head, int) and not isinstance(head, bool) for head in heads):
        return False
    if not all(isinstance(rel, str) and rel in DEPREL_SET for rel in deprels):
        return False
    if any(head < 0 or head > n or head == i for i, head in enumerate(heads, 1)):
        return False
    if sum(head == 0 for head in heads) != 1 or _has_cycle(heads):
        return False
    return all((head == 0) == (rel == "root") for head, rel in zip(heads, deprels))


def calculate_stepwise_metrics(records):
    total = uas_correct = las_correct = relation_correct = 0
    non_punct_total = non_punct_uas = non_punct_las = 0
    sentence_uem = sentence_lem = valid_outputs = 0
    task1_rows = task2_rows = task3_rows = 0
    strict_task1_rows = strict_task2_rows = strict_task3_rows = 0
    complete_three_step = 0
    complete_recovered_three_step = 0
    valid_upos_rows = predicted_upos_rows = 0
    per_relation = {label: defaultdict(int) for label in DEPRELS}

    for record in records:
        gold_heads = record["heads"]
        gold_rels = record["deprels"]
        prediction = record.get("prediction") or {}
        pred_heads = prediction.get("heads") or [None] * len(gold_heads)
        pred_rels = prediction.get("deprels") or [None] * len(gold_rels)
        pred_upos = prediction.get("upos") or [None] * len(gold_heads)
        details = record.get("parse_details") or {}
        task1_rows += int(details.get("task1_rows") or 0)
        task2_rows += int(details.get("task2_rows") or 0)
        task3_rows += int(details.get("task3_rows") or 0)
        strict_task1_rows += int(details.get("strict_task1_rows") or 0)
        strict_task2_rows += int(details.get("strict_task2_rows") or 0)
        strict_task3_rows += int(details.get("strict_task3_rows") or 0)
        complete_three_step += int(bool(details.get("complete_three_step_format")))
        complete_recovered_three_step += int(
            bool(details.get("complete_recovered_three_step"))
        )
        predicted_upos_rows += sum(tag is not None for tag in pred_upos)
        valid_upos_rows += sum(tag in UPOS_SET for tag in pred_upos if tag is not None)

        complete_heads = len(pred_heads) == len(gold_heads) and all(
            isinstance(head, int) and not isinstance(head, bool) for head in pred_heads
        )
        complete_rels = len(pred_rels) == len(gold_rels) and all(
            isinstance(rel, str) for rel in pred_rels
        )
        output_valid = complete_heads and complete_rels and _complete_valid_tree(
            pred_heads, pred_rels
        )
        valid_outputs += int(output_valid)
        heads_exact = complete_heads and pred_heads == gold_heads
        labeled_exact = heads_exact and complete_rels and pred_rels == gold_rels
        sentence_uem += int(heads_exact)
        sentence_lem += int(labeled_exact)

        for index, (gold_head, gold_rel) in enumerate(zip(gold_heads, gold_rels)):
            pred_head = pred_heads[index] if index < len(pred_heads) else None
            pred_rel = pred_rels[index] if index < len(pred_rels) else None
            if isinstance(pred_rel, str):
                pred_rel = pred_rel.strip().lower()
            head_ok = pred_head == gold_head
            rel_ok = pred_rel == gold_rel
            las_ok = head_ok and rel_ok
            total += 1
            uas_correct += int(head_ok)
            las_correct += int(las_ok)
            relation_correct += int(rel_ok)
            if gold_rel != "punct":
                non_punct_total += 1
                non_punct_uas += int(head_ok)
                non_punct_las += int(las_ok)
            per_relation[gold_rel]["support"] += 1
            per_relation[gold_rel]["uas_correct"] += int(head_ok)
            per_relation[gold_rel]["las_correct"] += int(las_ok)

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
        "relation_accuracy": relation_correct / total if total else None,
        "non_punct_tokens": non_punct_total,
        "non_punct_uas": non_punct_uas / non_punct_total if non_punct_total else None,
        "non_punct_las": non_punct_las / non_punct_total if non_punct_total else None,
        "sentence_unlabeled_exact_match": sentence_uem / sentence_count if sentence_count else None,
        "sentence_labeled_exact_match": sentence_lem / sentence_count if sentence_count else None,
        "task1_token_recall": task1_rows / total if total else None,
        "task2_token_recall": task2_rows / total if total else None,
        "task3_token_recall": task3_rows / total if total else None,
        "strict_task1_token_recall": strict_task1_rows / total if total else None,
        "strict_task2_token_recall": strict_task2_rows / total if total else None,
        "strict_task3_token_recall": strict_task3_rows / total if total else None,
        "complete_three_step_outputs": complete_three_step,
        "complete_three_step_rate": complete_three_step / sentence_count if sentence_count else None,
        "complete_recovered_three_step_outputs": complete_recovered_three_step,
        "complete_recovered_three_step_rate": (
            complete_recovered_three_step / sentence_count if sentence_count else None
        ),
        "valid_upos_output_rate": valid_upos_rows / predicted_upos_rows if predicted_upos_rows else None,
        "valid_outputs": valid_outputs,
        "invalid_outputs": sentence_count - valid_outputs,
        "invalid_output_rate": (sentence_count - valid_outputs) / sentence_count if sentence_count else None,
        "per_relation": per_relation_metrics,
        "parse_status_counts": dict(Counter(r.get("parse_status") for r in records)),
    }
