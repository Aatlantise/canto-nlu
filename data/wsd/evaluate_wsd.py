import argparse
import warnings
from hanziconv import HanziConv
import polars as pl
from itertools import combinations
from torch import cosine_similarity
from termcolor import colored
from collections import Counter
import torch
from transformers import BertForMaskedLM, AlbertTokenizer, BertTokenizer

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Evaluate a (masked-LM) checkpoint on the WSD task.")
parser.add_argument("model_dir", help="HF hub model name or local checkpoint directory to evaluate")
args = parser.parse_args()

df = pl.read_csv("data/wsd/senses.csv")

df = df.with_columns([
    pl.col("traditional").map_elements(
        lambda x: HanziConv.toTraditional(x) if x is not None else None,
        return_dtype=pl.Utf8
    ),
    pl.col("sentence1").map_elements(
        lambda x: HanziConv.toTraditional(x) if x is not None else None,
        return_dtype=pl.Utf8
    ),
    pl.col("sentence2").map_elements(
        lambda x: HanziConv.toTraditional(x) if x is not None else None,
        return_dtype=pl.Utf8
    )
])

for row in df.iter_rows(named=True):
    assert row["traditional"] in row["sentence1"]
    assert row["traditional"] in row["sentence2"]


model_path = args.model_dir

if "yue-scratch" in model_path:
    tokenizer_class = AlbertTokenizer
else:
    tokenizer_class = BertTokenizer

model = BertForMaskedLM.from_pretrained(model_path)
tokenizer = tokenizer_class.from_pretrained(model_path)



df = pl.read_csv("data/wsd/senses-traditional.csv")

total_pairs = 0

predictions = []


def get_embedding(sentence, word, tokenizer, model):
    """Extract BERT embedding for a masked word in a sentence."""
    sentence = sentence.replace(word, "[MASK]")
    input_ids = tokenizer.encode(sentence, return_tensors='pt')

    mask_token_indices = torch.where(input_ids == tokenizer.mask_token_id)[1]
    mask_token_index = mask_token_indices[0].item()

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    last_hidden_state = outputs.hidden_states[-1]
    mask_embedding = last_hidden_state[0, mask_token_index, :]

    return mask_embedding


for word, group in df.with_row_index().group_by("traditional"):
    # Step 1: Create all embeddings for this word group
    embeddings_cache = {}  # {row_index: (embedding1, embedding2)}

    for row in group.iter_rows(named=True):
        idx = row["index"]
        sent1 = row["sentence1"]
        sent2 = row["sentence2"]

        emb1 = get_embedding(sent1, word[0], tokenizer, model)
        emb2 = get_embedding(sent2, word[0], tokenizer, model)

        embeddings_cache[idx] = (emb1, emb2)

    # Step 2: Generate combinations and compare
    indices = group.select("index").to_series().to_list()

    for index1, index2 in combinations(indices, r=2):
        # Get precomputed embeddings
        def1_emb1, def1_emb2 = embeddings_cache[index1]
        def2_emb1, def2_emb2 = embeddings_cache[index2]

        # Calculate similarities
        same1 = cosine_similarity(def1_emb1, def1_emb2, dim=0)
        same2 = cosine_similarity(def2_emb1, def2_emb2, dim=0)

        diff1 = cosine_similarity(def1_emb1, def2_emb1, dim=0)
        diff2 = cosine_similarity(def1_emb2, def2_emb2, dim=0)

        averaged_same_score = (same1 + same2) / 2
        averaged_diff_score = (diff1 + diff2) / 2

        if averaged_same_score > averaged_diff_score:
            predictions.append("right")
        else:
            predictions.append("wrong")
            # Get original sentences for debugging
            def1_row = df.row(index1, named=True)
            def2_row = df.row(index2, named=True)
            sentences = (def1_row["sentence1"], def1_row["sentence2"],
                         def2_row["sentence1"], def2_row["sentence2"])
            sentences = [s.replace(word[0], f"_[{word[0]}]_") for s in sentences]
            print(sentences)

# print(predictions)



result_counter = Counter(predictions)
acc = result_counter["right"] / (result_counter["right"] + result_counter["wrong"])
print(f"WSD Accuracy: {acc}")
