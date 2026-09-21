import csv
import json
import argparse
from collections import defaultdict
from itertools import combinations


def build_wsd_test_jsonl(input_csv, output_jsonl):
    """
    Rules:
    1. For each row (one sense): generate one positive example (sentence1, sentence2) with label="similar"
    2. For each pair of different senses of the same target word: generate all cross-sense sentence pairs (4 pairs total) with label="not_similar"
    """
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    by_target = defaultdict(list)
    for row in rows:
        by_target[row['traditional']].append(row)
    
    examples = []
    example_id = 0

    for row in rows:
        examples.append({
            "id": example_id,
            "target": row['traditional'],
            "sentence1": row['sentence1'],
            "sentence2": row['sentence2'],
            "label": "similar"
        })
        example_id += 1
    
    for target, sense_rows in by_target.items():
        if len(sense_rows) < 2:
            continue
        
        for sense_a, sense_b in combinations(sense_rows, 2):
            sent_a1 = sense_a['sentence1']
            sent_a2 = sense_a['sentence2']
            sent_b1 = sense_b['sentence1']
            sent_b2 = sense_b['sentence2']
            
            cross_pairs = [
                (sent_a1, sent_b1),
                (sent_a1, sent_b2),
                (sent_a2, sent_b1),
                (sent_a2, sent_b2),
            ]
            
            for s1, s2 in cross_pairs:
                examples.append({
                    "id": example_id,
                    "target": target,
                    "sentence1": s1,
                    "sentence2": s2,
                    "label": "not_similar"
                })
                example_id += 1
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    positive = len(rows)
    negative = len(examples) - positive
    
    return len(examples), positive, negative


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build WSD test set JSONL')
    parser.add_argument('--input', default='senses.csv', help='Input CSV')
    parser.add_argument('--output', default='test.jsonl', help='Output JSONL')
    args = parser.parse_args()
    
    total, pos, neg = build_wsd_test_jsonl(args.input, args.output)
    print(f'Generated {total} examples to {args.output}')
    print(f'  Positive (similar):     {pos}')
    print(f'  Negative (not_similar): {neg}')
