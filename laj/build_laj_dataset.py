import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class LabelingConfig:
    semantic_error_types: set = None
    
    min_major_errors: int = 1      # If >= this many Major errors, label=0
    min_semantic_errors: int = 1   # If >= this many semantic errors, label=0
    min_total_errors: int = 1      # If >= this many total errors, label=0
    
    def __post_init__(self):
        if self.semantic_error_types is None:
            self.semantic_error_types = {
                "Mistranslation",
                "Omission",
                "Addition",
                "Grammar",
            }

DEFAULT_CONFIG = LabelingConfig(
    semantic_error_types={
        "Mistranslation",
        "Omission",
        "Addition",
        "Grammar",
    },
    min_major_errors=1,
    min_semantic_errors=1,
    min_total_errors=1,
)

def compute_label(annotations: Dict[str, Any], config: LabelingConfig) -> int:
    
    if not annotations or "annotatedSpans" not in annotations:
        return 1
    
    annotated_spans = annotations["annotatedSpans"]
    
    if not annotated_spans:
        return 1
    
    major_error_count = 0
    semantic_error_count = 0
    total_error_count = len(annotated_spans)
    
    for span in annotated_spans:
        error_type = span.get("error_type", "")
        error_severity = span.get("error_severity", "")
        
        if error_severity == "Major":
            major_error_count += 1
        
        if error_type in config.semantic_error_types:
            semantic_error_count += 1
    
    if major_error_count >= config.min_major_errors:
        return 0
    
    if semantic_error_count >= config.min_semantic_errors:
        return 0
    
    if total_error_count >= config.min_total_errors:
        return 0

    return 1


def build_laj_dataset(
    input_jsonl_path: str,
    output_jsonl_path: str,
    config: LabelingConfig = None,
) -> None:

    if config is None:
        config = DEFAULT_CONFIG
    
    input_path = Path(input_jsonl_path)
    output_path = Path(output_jsonl_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_jsonl_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = []
    example_id = 0
    
    with open(input_path, "r", encoding="utf-8") as f_in:
        for line_idx, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_idx}: {e}")
                continue
            
            entry_id = entry.get("id")
            ref = entry.get("ref", "")
            mt = entry.get("mt", "")
            annotations = entry.get("annotations", {})
            
            ref_example = {
                "id": example_id,
                "source_id": entry_id,
                "sentence": ref,
                "label": 1,
            }
            dataset.append(ref_example)
            example_id += 1

            mt_label = compute_label(annotations, config)
            mt_example = {
                "id": example_id,
                "source_id": entry_id,
                "sentence": mt,
                "label": mt_label,
            }
            dataset.append(mt_example)
            example_id += 1
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for example in dataset:
            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"✓ Dataset conversion complete!")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Total examples: {len(dataset)}")
    
    label_counts = {}
    for example in dataset:
        label = example["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"  Label distribution: {label_counts}")
    print(f"  Config: {config}")


if __name__ == "__main__":
    import sys
    
    input_file = Path(__file__).parent / "cantonese.jsonl"
    output_file = Path(__file__).parent / "laj_dataset.jsonl"

    build_laj_dataset(str(input_file), str(output_file), DEFAULT_CONFIG)
