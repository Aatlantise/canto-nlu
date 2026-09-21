from pathlib import Path
import csv
import json
import sys
import argparse


def convert_tsv_to_jsonl(input_path: Path, output_path: Path, source: str = "openrice"):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8-sig", newline="") as inf:
        reader = csv.reader(inf, delimiter="\t")
        try:
            first = next(reader)
        except StopIteration:
            with output_path.open("w", encoding="utf-8") as outf:
                pass
            return

        lower = [c.strip().lower() for c in first]
        has_header = ("label" in lower and "text_a" in lower) or ("sentence" in lower)

        inf.seek(0)

        if has_header:
            dict_reader = csv.DictReader(inf, delimiter="\t")
            rows = list(dict_reader)
            text_key = None
            if "text_a" in dict_reader.fieldnames:
                text_key = "text_a"
            elif "sentence" in dict_reader.fieldnames:
                text_key = "sentence"
            else:
                for f in dict_reader.fieldnames:
                    if f and f.strip().lower() == "text_a":
                        text_key = f
                        break

            label_key = None
            if "label" in dict_reader.fieldnames:
                label_key = "label"
            else:
                for f in dict_reader.fieldnames:
                    if f and f.strip().lower() == "label":
                        label_key = f
                        break

            with output_path.open("w", encoding="utf-8") as outf:
                for i, row in enumerate(rows):
                    sentence = row.get(text_key, "") if text_key else row.get("sentence", "")
                    label = row.get(label_key, "") if label_key else row.get("label", "")
                    obj = {"id": i, "sentence": sentence, "label": label, "source": source}
                    outf.write(json.dumps(obj, ensure_ascii=False) + "\n")

        else:
            inf.seek(0)
            reader = csv.reader(inf, delimiter="\t")
            with output_path.open("w", encoding="utf-8") as outf:
                for i, parts in enumerate(reader):
                    if not parts:
                        continue
                    if len(parts) == 1:
                        label = ""
                        sentence = parts[0]
                    else:
                        label = parts[0]
                        sentence = parts[1]
                    obj = {"id": i, "sentence": sentence, "label": label, "source": source}
                    outf.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert TSV (label, text_a) to JSONL with id, sentence, label, source")
    parser.add_argument("paths", nargs="*", help="TSV input files to convert (optional). If omitted, script will convert train/valid/test.tsv in the script directory.")
    parser.add_argument("-s", "--source", default="openrice", help="value to fill into the `source` field")
    args = parser.parse_args(argv)

    if not args.paths:
        base = Path(__file__).parent
        candidates = [base / "train.tsv", base / "valid.tsv", base / "test.tsv"]
        for p in candidates:
            if p.exists():
                out = p.with_suffix(".jsonl")
                print(f"Converting {p} -> {out}")
                convert_tsv_to_jsonl(p, out, source=args.source)
    else:
        paths = [Path(p) for p in args.paths]
        for p in paths:
            if not p.exists():
                print(f"Warning: {p} does not exist, skipping", file=sys.stderr)
                continue
            out = p.with_suffix(".jsonl")
            print(f"Converting {p} -> {out}")
            convert_tsv_to_jsonl(p, out, source=args.source)


if __name__ == "__main__":
    main()
