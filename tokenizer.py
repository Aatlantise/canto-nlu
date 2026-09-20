import pandas as pd
import sentencepiece as spm
from tqdm import tqdm
import random
from datasets import Dataset, interleave_datasets
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast


def weighted_ds(in_file: str, out_file: str, t: float = 2.0, r: int = 16):
    # read corpus
    df = pd.read_json(in_file, lines=True)
    df['char_count'] = df['text'].apply(len)

    # drop min{10%, 10k samples} for each source
    val_indices = []
    for source, group in df.groupby('source'):
        n_drop = min(int(len(group) * 0.1), 10000)
        val_indices.extend(group.sample(n=n_drop, random_state=42).index.tolist())
    df = df.drop(index=val_indices).reset_index(drop=True)
    # save the validation indices to a file for later use
    with open('val_indices.txt', 'w') as f:
        for idx in val_indices:
            f.write(f"{idx}\n")

    # calculate sampling probability per source with temperature scaling
    chars_by_source_df = df.groupby('source')['char_count'].sum().reset_index().rename(columns={'char_count': 'chars'})
    chars_by_source_df['prob'] = chars_by_source_df['chars'] / chars_by_source_df['chars'].sum()
    capped = pd.Series(False, index=chars_by_source_df.index)

    p = chars_by_source_df['prob']
    w = p ** (1 / t)
    w = w / w.sum()

    while True:
        over = (~capped) & (w > r * p)
        if not over.any():
            break
        capped |= over
        w[capped] = r * p[capped]
        free = ~capped
        w[free] = w[free] / w[free].sum() * (1 - w[capped].sum())

    chars_by_source_df['scaled_prob'] = w
    chars_by_source_df['effective_ratio'] = w / p

    # adjust probs by mean length of each source
    mean_len = df.groupby('source')['char_count'].mean()
    ex = chars_by_source_df['scaled_prob'].values / mean_len[chars_by_source_df['source']].values
    chars_by_source_df['ex_prob'] = ex / ex.sum()

    # debugging prints
    print(f"\n{'source':<16}{'chars':>12}{'prob':>10}{'scaled_prob':>16}{'effective_ratio':>16}")
    for _, row in chars_by_source_df.iterrows():
        print(f"{row['source']:<16}{row['chars']:>12,}{row['prob']:>10.4f}{row['scaled_prob']:>16.4f}{row['effective_ratio']:>16.4f}")

    # produce Datasets for each source
    datasets = []
    for source in chars_by_source_df['source']:
        source_df = df[df['source'] == source].reset_index(drop=True)
        ds = Dataset.from_pandas(source_df)
        datasets.append(ds)

    # produce weighted dataset with interleave
    weights = chars_by_source_df.set_index('source')['ex_prob'].to_dict()
    dataset = interleave_datasets(
        datasets,
        probabilities=[weights[source] for source in chars_by_source_df['source']],
        stopping_strategy='all_exhausted'
    )

    # save weighted dataset
    with open(out_file, "w") as f:
        for t in dataset["text"]:
            f.write(t.replace("\n", " ") + "\n")


def _get_cifu_words(filename: str) -> list:
    df = pd.read_csv(filename, sep='\t')
    print(f'got {len(df)} entries in cifu')
    # select the most common 2000 multi-character items in cifu
    df = df[df['Word'].astype(str).str.len() > 1]
    df = df.sort_values('SpokenAdult', ascending=False).head(2000)
    df[['Word', 'SpokenAdult']].to_csv('cifu_top.tsv', sep='\t', index=False)
    return df['Word'].tolist()


def train_tokenizer(corpus_file: str, vocab_size: int = 20000, model_prefix: str = "cantonese_tokenizer/canto_tokenizer"):
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        model_type='unigram',
        vocab_size=vocab_size,
        character_coverage=0.9995,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        input_sentence_size=10000000,
        shuffle_input_sentence=True,
        byte_fallback=True,
        user_defined_symbols=_get_cifu_words("data/Cifu-v1.tsv"),
        control_symbols=["[CLS]", "[SEP]", "[MASK]"],
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
        max_sentencepiece_length=4,
    )


def save_hf_tokenizer(model_file: str = "cantonese_tokenizer/canto_tokenizer.model",
                      out_dir: str = "cantonese_tokenizer/canto_tokenizer_hf"):
    sp = spm.SentencePieceProcessor(model_file=model_file)
    backend = Tokenizer(models.Unigram(
        [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())],
        unk_id=sp.unk_id(),
        byte_fallback=True,
    ))
    backend.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="never")  # add_dummy_prefix=False
    backend.decoder = decoders.Metaspace(prepend_scheme="never")
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", sp.piece_to_id("[CLS]")), ("[SEP]", sp.piece_to_id("[SEP]"))],
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=512,
    )
    tokenizer.save_pretrained(out_dir)
    return tokenizer


if __name__ == "__main__":
    weighted_ds('data/canto-corpus.jsonl', 'data/tokenizer_sample.txt', t=2.0, r=16)
    train_tokenizer('data/tokenizer_sample.txt')
    save_hf_tokenizer()
