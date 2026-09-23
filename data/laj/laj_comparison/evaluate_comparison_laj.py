import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Evaluate a (masked-LM) checkpoint on the comparison LAJ task, "
                 "BLiMP-style: for each sentence pair, predict the sentence with the "
                 "higher pseudo-log-likelihood as more acceptable."
)
parser.add_argument("model_dir", help="HF hub model name or local checkpoint directory to evaluate", default="./models/yue-monolingual")
parser.add_argument("--max_length", type=int, default=128, help="Max tokens per sentence (truncated beyond this)")
args = parser.parse_args()

model_path = args.model_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)

pairs = []
with open("data/laj/laj_comparison/comparison_test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            pairs.append(json.loads(line))


@torch.no_grad()
def pseudo_log_likelihood(sentence):
    """Sum of log P(token | context) for every non-special token in the sentence,
    each computed by masking that one token and reading off the model's log-prob
    for the true token at that position."""
    encoding = tokenizer(sentence, truncation=True, max_length=args.max_length, return_tensors="pt")
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]

    special_tokens_mask = tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
    maskable_positions = [i for i, is_special in enumerate(special_tokens_mask) if not is_special]
    if not maskable_positions:
        return 0.0

    num_positions = len(maskable_positions)
    batch_input_ids = input_ids.unsqueeze(0).repeat(num_positions, 1).to(device)
    batch_attention_mask = attention_mask.unsqueeze(0).repeat(num_positions, 1).to(device)
    target_ids = input_ids[maskable_positions].to(device)

    row_idx = torch.arange(num_positions)
    pos_idx = torch.tensor(maskable_positions)
    batch_input_ids[row_idx, pos_idx] = tokenizer.mask_token_id

    logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[row_idx, pos_idx, target_ids]

    return token_log_probs.sum().item()


correct = 0
total = 0

for pair in tqdm(pairs, desc="Scoring comparison pairs"):
    pll1 = pseudo_log_likelihood(pair["sentence1"])
    pll2 = pseudo_log_likelihood(pair["sentence2"])
    prediction = 1 if pll1 >= pll2 else 2

    total += 1
    if prediction == pair["label"]:
        correct += 1

accuracy = correct / total if total else 0.0
print(f"Comparison pairs: {total}")
print(f"Comparison LAJ Accuracy: {accuracy}")
