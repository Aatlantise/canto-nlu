from datasets import load_from_disk, Dataset, load_dataset, interleave_datasets
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,

    EarlyStoppingCallback,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertPreTrainedModel,
    BertModel,
    BertConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DebertaV2Tokenizer,
)
from transformers.modeling_outputs import TokenClassifierOutput
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, train_test_split
import evaluate
from utils import get_subset
import torch.nn as nn
import re
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SiniticPreTrainer:
    def __init__(self, lang="", model_dir="./models/bert-base-chinese-local", scratch=False, data=None):
        self.ds = None
        self.tokenizer = None
        self.lang = lang
        self.model_dir = model_dir
        self.tokenized_ds = None
        self.lm_dataset = None
        self.from_scratch = scratch
        self.data = data

    def preprocess_data(self):
        self.ds = self.ds.filter(lambda x: len(x["text"]) > 100)  # Remove stubs/empty pages

        def tokenize(example):
            return self.tokenizer(example["text"], return_special_tokens_mask=True, truncation=False)

        tokenized_ds = self.ds.map(tokenize, batched=True, remove_columns=["text", "title", "id", "url"])

        # For example, into 512-token chunks
        block_size = 512

        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = (len(concatenated["input_ids"]) // block_size) * block_size
            result = {
                k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated.items()
            }
            return result

        train_dataset, validation_dataset = tokenized_ds["train"].train_test_split(test_size=0.1).values()
        self.lm_dataset = {
            "train": train_dataset.map(group_texts, batched=True),
            "validation": validation_dataset.map(group_texts, batched=True)
        }

    def train(self):
        self.preprocess_data()

        # if any({split not in self.lm_dataset for split in ["train", "validation"]}):
        #     raise ValueError(f"'train' and 'validation' splits must be present in lm_dataset."
        #                      f"Found: {self.lm_dataset.keys()}")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        config = BertConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=514,  # 512 + 2
            type_vocab_size=2,
            pad_token_id=self.tokenizer.pad_token_id,
)

        if self.from_scratch:
            model = BertForMaskedLM(config=config)
            model.resize_token_embeddings(len(self.tokenizer))
        else:
            model = BertForMaskedLM.from_pretrained(self.model_dir)
            model.resize_token_embeddings(len(self.tokenizer))

        output_dir_name = f"./{self.lang}-scratch" if self.from_scratch else f"./{self.lang}-transfer"

        training_args = TrainingArguments(
            num_train_epochs=2,
            per_device_train_batch_size=128,
            dataloader_num_workers=8,
            learning_rate=1e-5,
            warmup_steps=10000,
            weight_decay=0.01,
            save_steps=10000,
            save_total_limit=2,
            logging_steps=100,
            report_to="tensorboard",
            eval_strategy="steps",
            eval_steps=1000,
            fp16=False,
            bf16=True,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.lm_dataset["train"],
            eval_dataset=self.lm_dataset["validation"],
            data_collator=data_collator,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()
        trainer.save_model(output_dir_name)

class CantoPreTrainer(SiniticPreTrainer):
    # `canto-corpus`: pre-training over corpus.py's corpus, weighted as in tokenizer.py
    CORPUS_FILE = "./data/canto-corpus.jsonl"
    VAL_INDICES_FILE = "./val_indices.txt"
    TOKENIZER_DIR = "./cantonese_tokenizer/canto_tokenizer_hf"
    MAX_LENGTH = 128
    TEMPERATURE = 2.0  # t in tokenizer.py
    MAX_BOOST = 16  # r in tokenizer.py: no source is sampled past 16x its natural share
    MAX_STEPS = 250000
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 10000

    def __init__(self, lang="yue", model_dir="./models/bert-base-chinese-local", scratch=False, data=None):
        super().__init__(lang, model_dir, scratch, data)
        # checking data happens in run.py
        if data == "canto-corpus":
            for path, command in [(self.CORPUS_FILE, "python corpus.py --lang=yue"),
                                  (self.TOKENIZER_DIR, "python tokenizer.py"),
                                  (self.VAL_INDICES_FILE, "python tokenizer.py")]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} not found. Please first run `{command}`.")
            if not self.from_scratch and not os.path.exists(self.model_dir):
                raise FileNotFoundError(
                    f"Model directory {self.model_dir} not found."
                    f"Please first run `python download.py --lang=yue --model_dir={self.model_dir}`."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_DIR)
            return

        if data == "cantonese-sentences":
            if not os.path.exists("./data/cantonese-sentences"):
                raise FileNotFoundError(
                    "Cantonese Sentences dataset not found. Please first run `python download.py --lang=yue`."
                )
            self.ds = load_from_disk("./data/cantonese-sentences")
        else:
            if not os.path.exists("./data/yue-wiki-full-local"):
                raise FileNotFoundError(
                    "Cantonese Wiki dataset not found. Please first run `python download.py --lang=yue`."
                )
            self.ds = load_from_disk("./data/yue-wiki-full-local")

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model directory {self.model_dir} not found."
                f"Please first run `python download.py --lang=yue --model_dir={self.model_dir}`."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

    def preprocess_data(self):
        if self.data == "wiki":
            self.wiki_preprocess_data()
        elif self.data == "cantonese-sentences":
            self.wiki_preprocess_data() # for future use if there's a need to implement something different
        elif self.data == "canto-corpus":
            self.canto_corpus_preprocess_data()
        else:
            raise ValueError(f"{self.data} not supported--please choose between `wiki`, "
                             f"`cantonese-sentences` or `canto-corpus`.")


    def canto_sentences_preprocess_data(self):
        pass

    def wiki_preprocess_data(self):
        PARENS_JUNK = re.compile(r"\(\s*[, -]*\s*\)")

        def clean_parentheses(text: str) -> str:
            return PARENS_JUNK.sub("", text)

        def tokenize_function(batch):
            # Clean all texts in batch
            _batch = batch['text'] if self.data == "wiki" else batch['content']
            cleaned = [clean_parentheses(t) for t in _batch]

            # Tokenize in batch
            tokens = self.tokenizer(
                cleaned,
                truncation=False,
                return_attention_mask=True,
                add_special_tokens=True
            )

            # Split each sequence into 128-token chunks
            input_batch = []
            attn_batch = []
            for input_ids, attention_mask in zip(tokens["input_ids"], tokens["attention_mask"]):
                for i in range(0, len(input_ids), 128):
                    input_batch.append(input_ids[i:i + 128])
                    attn_batch.append(attention_mask[i:i + 128])
            return {"input_ids": input_batch, "attention_mask": attn_batch}

        dataset = self.ds["train"] if self.data == "cantonese-sentences" else self.ds
        print(f"Number of documents: {len(dataset)}")

        # Use map with batched=True to avoid full in-memory loading
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        print(f"Number of 128-token chunks: {len(tokenized)}")

        # Train/validation split
        split_dataset = tokenized.train_test_split(test_size=0.01, seed=42)
        self.lm_dataset = {
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        }

        # Optional: save preprocessed dataset to disk
        tokenized.save_to_disk(f"data/pretokenized_{self.data}")

    def mixture(self, meta):
        stats = meta.groupby("source", observed=True)["n_chars"].agg(rows="count", chars="sum")

        p = stats["chars"] / stats["chars"].sum()
        w = p ** (1 / self.TEMPERATURE)
        w = w / w.sum()

        capped = pd.Series(False, index=stats.index)
        while True:
            over = (~capped) & (w > self.MAX_BOOST * p)
            if not over.any():
                break
            capped |= over
            w[capped] = self.MAX_BOOST * p[capped]
            free = ~capped
            if not free.any():
                break
            w[free] = w[free] / w[free].sum() * (1 - w[capped].sum())

        stats["prob"] = p
        stats["scaled_prob"] = w
        stats["effective_ratio"] = w / p

        print(f"\n{'source':<16}{'rows':>12}{'chars':>14}{'prob':>10}{'scaled_prob':>14}{'ratio':>8}")
        for source, row in stats.iterrows():
            print(f"{source:<16}{row['rows']:>12,.0f}{row['chars']:>14,.0f}{row['prob']:>10.4f}"
                  f"{row['scaled_prob']:>14.4f}{row['effective_ratio']:>8.2f}")
        return stats

    def pack(self, ds, desc):
        # rows arrive as [CLS] ... [SEP], so dropping each row's [CLS] keeps the [SEP]
        # between neighbours and every block still opens the way fine-tuning expects
        cls_id, body_size = self.tokenizer.cls_token_id, self.MAX_LENGTH - 1

        def group(batch):
            flat = [i for ids in batch["input_ids"] for i in ids[1:]]
            total = (len(flat) // body_size) * body_size  # the tail of each batch is dropped
            return {"input_ids": [[cls_id] + flat[i:i + body_size]
                                  for i in range(0, total, body_size)]}

        packed = ds.map(group, batched=True, batch_size=10000, desc=f"Packing {desc}")
        print(f"{desc}: {len(ds):,} rows -> {len(packed):,} blocks")
        return packed

    def canto_corpus_preprocess_data(self):
        cache = f"data/pretokenized_{self.data}"
        if os.path.exists(cache):
            tokenized = load_from_disk(cache)
            print(f"Reusing {cache} ({len(tokenized):,} rows; delete it to rebuild).")
        else:
            ds = load_dataset("json", data_files=self.CORPUS_FILE, split="train")
            print(f"Loaded {len(ds):,} rows from {self.CORPUS_FILE}.")

            def tokenize_function(batch):
                return {
                    "input_ids": self.tokenizer(batch["text"], truncation=True,
                                                max_length=self.MAX_LENGTH)["input_ids"],
                    "n_chars": [len(text) for text in batch["text"]],
                }

            tokenized = ds.map(
                tokenize_function,
                batched=True,
                remove_columns=["text", "label"],
                desc="Tokenizing dataset",
            )
            tokenized.save_to_disk(cache)

        meta = tokenized.select_columns(["source", "n_chars"]).to_pandas()
        meta["source"] = meta["source"].astype("category")
        tokenized = tokenized.select_columns(["input_ids"])

        # hold out validation set
        is_train = np.ones(len(tokenized), dtype=bool)
        is_train[np.loadtxt(self.VAL_INDICES_FILE, dtype=np.int64)] = False
        stats = self.mixture(meta[is_train])

        parts, validation = [], {}
        for source in stats.index:
            of_source = (meta["source"] == source).to_numpy()
            parts.append(self.pack(tokenized.select(np.flatnonzero(of_source & is_train)),
                                   f"train/{source}").shuffle(seed=42))
            validation[source] = self.pack(tokenized.select(np.flatnonzero(of_source & ~is_train)),
                                           f"val/{source}")

        # create dataset stream
        train = interleave_datasets(
            parts,
            probabilities=list(stats["scaled_prob"] / stats["scaled_prob"].sum()),
            seed=42,
            stopping_strategy="all_exhausted",
        )
        unique = sum(len(part) for part in parts)
        print(f"\nTraining stream: {len(train):,} blocks ({len(train) / unique:.2f}x the {unique:,} "
              f"unique blocks), validation: {sum(len(v) for v in validation.values()):,} blocks "
              f"over {len(validation)} sources.")

        self.lm_dataset = {"train": train, "validation": validation}

    def train(self):
        if self.data != "canto-corpus":
            return super().train()

        self.preprocess_data()

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        config = BertConfig(
            vocab_size=len(self.tokenizer),
            pad_token_id=self.tokenizer.pad_token_id,
        )

        if self.from_scratch:
            model = BertForMaskedLM(config=config)
        else:
            model = BertForMaskedLM.from_pretrained(self.model_dir)
        model.resize_token_embeddings(len(self.tokenizer))

        output_dir_name = f"./models/{self.lang}-canto-corpus"
        print(f"\n{model.num_parameters() / 1e6:.1f}M parameters, {self.MAX_STEPS:,} steps of "
              f"{self.BATCH_SIZE} blocks, {self.MAX_STEPS * self.BATCH_SIZE * self.MAX_LENGTH / 1e9:.1f}B "
              f"tokens (~{self.MAX_STEPS * self.BATCH_SIZE / len(self.lm_dataset['train']):.1f} "
              f"epochs over the stream).")

        training_args = TrainingArguments(
            output_dir=output_dir_name,
            max_steps=self.MAX_STEPS,
            per_device_train_batch_size=self.BATCH_SIZE,
            per_device_eval_batch_size=self.BATCH_SIZE,
            dataloader_num_workers=4,
            learning_rate=self.LEARNING_RATE,
            warmup_steps=self.WARMUP_STEPS,
            weight_decay=0.01,
            save_steps=10000,
            save_total_limit=2,
            logging_steps=100,
            report_to="tensorboard",
            eval_strategy="steps",
            eval_steps=5000,
            fp16=False,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            load_best_model_at_end=False,
            seed=42,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.lm_dataset["train"],
            eval_dataset=self.lm_dataset["validation"],  # a dict: one eval_<source>_loss each
            data_collator=data_collator,
            processing_class=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(output_dir_name)
        self.tokenizer.save_pretrained(output_dir_name)
        print(f"Saved the model and tokenizer to {output_dir_name}.")


class WuPreTrainer(SiniticPreTrainer):
    def __init__(self, lang="wuu", model_dir="./models/bert-base-chinese-local"):
        super().__init__(lang, model_dir)
        if not os.path.exists("./data/wuu-wiki-local"):
            raise FileNotFoundError(
                "Wu Wiki dataset not found. Please first run `python download.py --lang=wuu`."
            )
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model directory {self.model_dir} not found."
                f"Please first run `python download.py --lang=wuu --model_dir={self.model_dir}`."
            )
        self.ds = load_from_disk("./data/wuu-wiki-local")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_dir)


def compute_nli_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


class CantoNLIFineTuner(CantoPreTrainer):
    def __init__(self, lang, model_dir, eval_only=False):
        super().__init__(lang, model_dir)
        self.finetune_dataset = None
        self.preprocess_data(eval_only=eval_only)
        self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
        self.training_args = TrainingArguments(
            output_dir=f"./models/{self.lang}-nlu-{[f for f in self.model_dir.split('/') if f][-1]}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            optim="adamw_torch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            save_steps=1000,
            save_total_limit=2,
            logging_steps=100,
            report_to="tensorboard",
            eval_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        )

    def preprocess_data(self, eval_only=False):
        nli_data = load_from_disk("./data/yue-nli-local")

        def tokenize_function(examples):
            return self.tokenizer(examples["input_text"], return_special_tokens_mask=True, truncation=True,
                                  padding="max_length", max_length=256)

        def yue_nli_collator(split):
            nested_nli_list = [
                [
              {"input_text": f"{s['anchor']} [SEP] {s['positive']}", "label": 0},
              {"input_text": f"{s['anchor']} [SEP] {s['negative']}", "label": 1}
              ]
                               for s in split]

            return Dataset.from_list([e for s in nested_nli_list for e in s])

        if eval_only:
            test = get_subset(nli_data["test"])
            test_set = yue_nli_collator(test).map(tokenize_function, batched=True)
            print(f"{len(test_set)} test examples.")
            self.finetune_dataset = {
                "train": None,
                "validation": None,
                "test": test_set
            }
        else:
            train, val, test = (
                nli_data["train"],
                nli_data["dev"],
                get_subset(nli_data["test"])
            )

            train_set, val_set, test_set = [
                yue_nli_collator(train).map(tokenize_function, batched=True),
                yue_nli_collator(val).map(tokenize_function, batched=True),
                yue_nli_collator(test).map(tokenize_function, batched=True)
            ]

            print(f"A total of {len(train_set)} training examples,")
            print(f"{len(val_set)} validation examples,")
            print(f"{len(test_set)} test examples.")

            self.finetune_dataset = {
                "train": train_set,
                "validation": val_set,
                "test": test_set
            }

    def eval(self, trainer):
        trainer.compute_metrics = compute_nli_metrics
        metrics = trainer.evaluate(
            eval_dataset=self.finetune_dataset["test"],
        )
        print(f"Accuracy: {metrics['eval_accuracy']}")
        print(f"Confusion_matrix: {metrics['eval_confusion_matrix']}")

    def finetune(self):
        if any({split not in self.finetune_dataset for split in ["train", "validation", "test"]}):
            raise ValueError(f"'train' and 'validation' splits must be present in finetune_dataset."
                             f"Found: {self.finetune_dataset.keys()}")

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.finetune_dataset["train"],
            eval_dataset=self.finetune_dataset["validation"],
        )

        trainer.train()
        trainer.save_model(f"./models/{self.lang}-nlu-{[f for f in self.model_dir.split('/') if f][-1]}")
        self.eval(trainer)


class CantoPOSFineTuner(CantoPreTrainer):
    def __init__(self, lang, model_dir):
        super().__init__(lang, model_dir)
        self.finetune_dataset = None
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir,
                                                           unk_token="[UNK]",
                                                           pad_token="[PAD]",
                                                           cls_token="[CLS]",
                                                           sep_token="[SEP]",
                                                           mask_token="[MASK]",
                                                           )
        self.pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
                    'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        self.tag2id = {tag: i for i, tag in enumerate(self.pos_tags)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}

    def preprocess_data(self):
        # Load raw dataset (adjust this path to your actual source)
        raw_data = load_from_disk("./data/yue-pos")  # Should return DatasetDict
        # Expected format: {"train": [{"sentence": [...], "labels": [...]}], ...}

        def align_labels_with_tokens(examples):
            tokenized = self.tokenizer(
                examples["sentence"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=128
            )

            labels = []
            for i, word_ids in enumerate(tokenized.word_ids(batch_index=i) for i in range(len(examples["sentence"]))):
                word_labels = examples["labels"][i]
                label_ids = []
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(self.tag2id.get(word_labels[word_idx], -100))
                        previous_word_idx = word_idx
                    else:
                        label_ids.append(-100)  # Mask subsequent subwords
                labels.append(label_ids)

            tokenized["labels"] = labels
            return tokenized

        # Tokenize and align labels
        tokenized_data = raw_data.map(align_labels_with_tokens, batched=True)
        self.finetune_dataset = {
            "train": tokenized_data["train"],
            "test": tokenized_data.get("test", tokenized_data["train"])
        }

    def finetune(self):
        self.preprocess_data()

        if any(split not in self.finetune_dataset for split in ["train", "test"]):
            raise ValueError(f"'train' and 'validation' splits must be present in finetune_dataset."
                             f"Found: {self.finetune_dataset.keys()}")

        model = BertForTokenClassification.from_pretrained(
            self.model_dir,
            num_labels=len(self.tag2id),
            id2label=self.id2tag,
            label2id=self.tag2id
        )

        training_args = TrainingArguments(
            output_dir=f"./models/{self.lang}-pos-{self.model_dir.strip('/').split('/')[-1]}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=100,
            report_to="tensorboard",
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
        )

        def pos_compute_metrics(pred):
            predictions, labels = pred
            predictions = np.argmax(predictions, axis=-1)

            true_labels = [
                [self.id2tag[l] for (p, l) in zip(pred_row, label_row) if l != -100]
                for pred_row, label_row in zip(predictions, labels)
            ]
            true_preds = [
                [self.id2tag[p] for (p, l) in zip(pred_row, label_row) if l != -100]
                for pred_row, label_row in zip(predictions, labels)
            ]

            # Simple accuracy and F1 (could replace with seqeval)
            flat_preds = [p for row in true_preds for p in row]
            flat_labels = [l for row in true_labels for l in row]

            return {
                "accuracy": accuracy_score(flat_labels, flat_preds),
                "macro_f1": f1_score(flat_labels, flat_preds, average="macro"),
                "micro_f1": f1_score(flat_labels, flat_preds, average="micro"),
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.finetune_dataset["train"],
            eval_dataset=self.finetune_dataset["test"],
            tokenizer=self.tokenizer,
            compute_metrics=pos_compute_metrics,
        )

        trainer.train()
        trainer.save_model(f"./models/{self.lang}-pos-{self.model_dir.strip('/').split('/')[-1]}")

        metrics = trainer.evaluate(self.finetune_dataset["test"])
        print(f"Final test accuracy: {metrics['eval_accuracy']}")
        print(f"Final test macro F1: {metrics['eval_macro_f1']}")
        print(f"Final test micro F1: {metrics['eval_micro_f1']}")



class BertForDependencyParsing(BertPreTrainedModel):
    """
    head_classifier: predicts head index in [0..max_length-1] (we map ROOT -> [CLS] position 0)
    rel_classifier: predicts dependency relation label for each token
    """
    def __init__(self, config, num_rel_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Predict head index among max_length token positions
        self.head_classifier = nn.Linear(config.hidden_size, config.max_position_embeddings)
        # Predict relation label
        self.rel_classifier = nn.Linear(config.hidden_size, num_rel_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_head=None,   # [B, L] with indices in [0..L-1], -100 to ignore
        labels_rel=None,    # [B, L] with rel ids, -100 to ignore
        **kwargs
    ):
        outputs = self.bert(input_ids, attention_mask=attention_mask, **kwargs)
        seq = self.dropout(outputs.last_hidden_state)  # [B, L, H]

        head_logits = self.head_classifier(seq)  # [B, L, max_pos] (use as [B, L, L] effectively)
        rel_logits  = self.rel_classifier(seq)   # [B, L, R]

        loss = None
        if labels_head is not None and labels_rel is not None:
            ce = nn.CrossEntropyLoss(ignore_index=-100)
            # Flatten over tokens
            head_loss = ce(head_logits.view(-1, head_logits.size(-1)), labels_head.view(-1))
            rel_loss  = ce(rel_logits.view(-1, rel_logits.size(-1)), labels_rel.view(-1))
            loss = head_loss + rel_loss

        # Return a tuple so Trainer hands both logits to compute_metrics
        return {"loss": loss, "logits": (head_logits, rel_logits)}


class CantoDEPSFineTuner(CantoPreTrainer):
    def __init__(self, lang, model_dir):
        super().__init__(lang, model_dir)
        self.finetune_dataset = None
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)

        # You can expand/adjust to your UD label set (incl. language-specific subtypes like discourse:sp)
        self.dep_labels = [
            "root","nsubj","obj","iobj","obl","vocative","expl","dislocated",
            "advcl","advmod","discourse","aux","cop","mark","nmod","appos",
            "nummod","acl","amod","det","clf","case","conj","cc","fixed",
            "flat","compound","list","parataxis","orphan","goeswith","reparandum",
            "punct","dep","csubj","xcomp","ccomp"
        ]
        self.rel2id = {r: i for i, r in enumerate(self.dep_labels)}
        self.id2rel = {i: r for r, i in self.rel2id.items()}

    def preprocess_data(self):
        """
        Expect dataset with fields:
          - 'sentence': list[str] words
          - 'heads':   list[int] UD heads (0=ROOT, 1..n word indices)
          - 'rels':    list[str] dependency relations, 'root' for head==0
        We align to subwords and:
          - place gold labels only on first subword of each word
          - map head word index -> tokenized sequence index of that head's FIRST subword
          - map ROOT (0) -> [CLS] position index (usually 0)
        """
        raw = load_from_disk("./data/yue-deps")

        def align(examples):
            # Tokenize with word alignment
            enc = self.tokenizer(
                examples["sentence"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_offsets_mapping=False,
            )

            B = len(examples["sentence"])
            labels_head = []
            labels_rel  = []

            for i in range(B):
                word_ids = enc.word_ids(batch_index=i)  # len = seq_len (incl CLS/SEP/PAD)
                words = examples["sentence"][i]
                heads = examples["heads"][i]  # UD heads: 0..len(words)
                rels  = examples["deps"][i]

                # Build map: word_idx -> token_idx of FIRST subword
                # word indices in UD are 1-based; we'll keep that in mind
                first_tok_idx_of_word = {}
                prev_w = None
                for tok_idx, w in enumerate(word_ids):
                    if w is None:
                        continue
                    if w != prev_w:
                        first_tok_idx_of_word[w] = tok_idx
                        prev_w = w

                # Choose a ROOT anchor: map to [CLS] token's position
                # Typically [CLS] is at index 0 with BERT tokenizers
                root_tok_idx = next((i for i, w in enumerate(word_ids) if w is None), 0)

                # Now create aligned labels for each token position
                seq_heads = []
                seq_rels  = []
                prev_w = None
                for tok_idx, w in enumerate(word_ids):
                    if w is None:
                        # Special or padding positions
                        seq_heads.append(-100)
                        seq_rels.append(-100)
                        continue

                    if w != prev_w:
                        # First subword for word w
                        gold_head_word = heads[w]  # if words indexed 0..n-1 in your data, adjust accordingly
                        # If your 'heads' are UD-style (0..n), but our word_ids are 0..n-1,
                        # then gold_head_word==0 means ROOT, otherwise (1..n) -> map to word (gold_head_word-1).
                        if max(heads) <= len(words) and min(heads) == 0:
                            # UD-style 1-based heads
                            if gold_head_word == 0:
                                gold_head_tok = root_tok_idx
                            else:
                                gold_head_tok = first_tok_idx_of_word.get(gold_head_word - 1, root_tok_idx)
                        else:
                            # Already 0-based word indices with -1/None for root? (less common)
                            if gold_head_word < 0:
                                gold_head_tok = root_tok_idx
                            else:
                                gold_head_tok = first_tok_idx_of_word.get(gold_head_word, root_tok_idx)

                        seq_heads.append(gold_head_tok)
                        seq_rels.append(self.rel2id.get(rels[w], -100))
                        prev_w = w
                    else:
                        # Non-first subword -> ignore
                        seq_heads.append(-100)
                        seq_rels.append(-100)

                labels_head.append(seq_heads)
                labels_rel.append(seq_rels)

            enc["labels_head"] = labels_head
            enc["labels_rel"]  = labels_rel
            return enc

        tokenized = raw.map(align, batched=True)
        self.finetune_dataset = {
            "train": tokenized["train"],
            "test": tokenized.get("test", tokenized["train"]),
         }

    def finetune(self):
        self.preprocess_data()

        model = BertForDependencyParsing.from_pretrained(
            self.model_dir,
            num_rel_labels=len(self.rel2id),
        )

        args = TrainingArguments(
            output_dir=f"./models/{self.lang}-deps-{self.model_dir.strip('/').split('/')[-1]}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=50,
            report_to="tensorboard",
            load_best_model_at_end=True,
            metric_for_best_model="eval_las",
            greater_is_better=True,
        )

        def compute_deps_metrics(eval_pred):
            """
            UAS: pred_head == gold_head
            LAS: pred_head == gold_head AND pred_rel == gold_rel
            Evaluated only where gold labels != -100 (i.e., first subwords).
            """
            preds = eval_pred.predictions
            labels = eval_pred.label_ids

            # Predictions: tuple(head_logits, rel_logits)
            if isinstance(preds, tuple) and len(preds) == 2:
                head_logits, rel_logits = preds
            elif isinstance(preds, dict) and "logits" in preds:
                head_logits, rel_logits = preds["logits"]
            else:
                raise ValueError("Unexpected predictions structure")

            # Labels may be dict or tuple depending on HF version
            if isinstance(labels, dict):
                gold_heads = labels["labels_head"]
                gold_rels  = labels["labels_rel"]
            elif isinstance(labels, (list, tuple)) and len(labels) == 2:
                gold_heads, gold_rels = labels
            else:
                # Some HF versions pass a single array; not our case.
                raise ValueError("Unexpected label_ids structure")

            # Argmax
            pred_heads = np.argmax(head_logits, axis=-1)  # [B, L]
            pred_rels  = np.argmax(rel_logits, axis=-1)   # [B, L]

            gold_heads = np.array(gold_heads)
            gold_rels  = np.array(gold_rels)

            # Valid positions: where gold_rel != -100 (equivalently gold_head != -100)
            valid_mask = gold_rels != -100

            total = valid_mask.sum()
            if total == 0:
                return {"uas": 0.0, "las": 0.0}

            uas_correct = ((pred_heads == gold_heads) & valid_mask).sum()
            las_correct = ((pred_heads == gold_heads) & (pred_rels == gold_rels) & valid_mask).sum()

            uas = float(uas_correct) / float(total)
            las = float(las_correct) / float(total)
            return {"uas": uas, "las": las}

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self.finetune_dataset["train"],
            eval_dataset=self.finetune_dataset["test"],
            tokenizer=self.tokenizer,
            compute_metrics=compute_deps_metrics,
        )

        trainer.train()
        trainer.save_model(f"./models/{self.lang}-deps-{self.model_dir.strip('/').split('/')[-1]}")



class CantoTokenClassificationFineTuner(CantoNLIFineTuner):
    def __init__(self, lang="yue", model_dir="./bert-base-chinese-local"):
        super().__init__(lang, model_dir)

    def preprocess_data(self):
        nlu_data = load_from_disk('./data/nlptea_dataset')['train']
        k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
        indices = list(k_fold.split(np.arange(len(nlu_data))))

        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

            labels = []
            for i, label in enumerate(examples[f"cantonese_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        self.finetune_dataset = []

        for fold, (indices_train, indices_test) in enumerate(indices):
            train_set = nlu_data.select(indices_train)
            valid_set = nlu_data.select(indices_test)

            train_set = train_set.map(tokenize_and_align_labels, batched=True)
            valid_set = valid_set.map(tokenize_and_align_labels, batched=True)

            self.finetune_dataset.append({
                "train": train_set,
                "validation": valid_set
            })

    def finetune(self):
        self.preprocess_data()

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        seqeval = evaluate.load("seqeval")
        label_list = ['Chinese', 'Cantonese']
        id2label = {
            0: 'Chinese',
            1: 'Cantonese'
        }
        label2id = {
            'Chinese': 0,
            'Cantonese': 1
        }

        def compute_nlu_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = seqeval.compute(predictions=true_predictions, references=true_labels)
            _, _, f1_scores, _ = precision_recall_fscore_support(
                [l for sublist in true_labels for l in sublist],
                [p for sublist in true_predictions for p in sublist],
                labels=["Cantonese"]
            )
            f1_positive = f1_scores[0]

            return {
                "f1_positive": f1_positive,
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        training_args = TrainingArguments(
            output_dir=f"./models/{self.lang}-nlu-{[f for f in self.model_dir.split('/') if f][-1]}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            optim="adamw_torch",
            learning_rate=1e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_steps=50,
            report_to="tensorboard",
        )

        cross_validation_results = {
            'f1_positive': [],
            'f1': [],
            'accuracy': [],
        }

        for fold, dataset in enumerate(self.finetune_dataset):
            print(f"Training on fold {fold + 1}/{len(self.finetune_dataset)}")
            model = BertForTokenClassification.from_pretrained(
                self.model_dir,
                num_labels=2,
                id2label=id2label,
                label2id=label2id
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                processing_class=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_nlu_metrics,
            )

            trainer.train()
            # trainer.save_model(f"./models/{self.lang}-nlu-{[f for f in self.model_dir.split('/') if f][-1]}-fold-{fold}")

            metrics = trainer.evaluate(
                eval_dataset=dataset["validation"],
            )
            print(metrics)

            cross_validation_results['accuracy'].append(metrics['eval_accuracy'])
            cross_validation_results['f1_positive'].append(metrics['eval_f1_positive'])
            cross_validation_results['f1'].append(metrics['eval_f1'])
            print(f"Fold {fold + 1} - Accuracy: {metrics['eval_accuracy']}")
            print(f"Fold {fold + 1} - F1: {metrics['eval_f1']}")
            print(f"Fold {fold + 1} - F1 Positive: {metrics['eval_f1_positive']}")

        print("Cross-validation results:")
        print(f"Average Accuracy: {np.mean(cross_validation_results['accuracy'])}")
        print(f"Average F1: {np.mean(cross_validation_results['f1'])}")
        print(f"Average F1 Positive: {np.mean(cross_validation_results['f1_positive'])}")

class CantoAcceptabilityFineTuner(CantoNLIFineTuner):
    def __init__(self, lang="yue", model_dir="./bert-base-chinese-local"):
        super().__init__(lang, model_dir)

    def preprocess_data(self):
        data = load_from_disk('data/acceptability-dataset-2')
        data = data.shuffle(seed=42)

        def tokenize(example):
            return self.tokenizer(example["text"], return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=128)

        self.finetune_dataset = []
        train_set, temp_set = data.train_test_split(test_size=0.1, seed=42).values()
        valid_set, test_set = temp_set.train_test_split(test_size=0.5, seed=42).values()

        train_set = train_set.map(tokenize, batched=True, remove_columns=["text"])
        valid_set = valid_set.map(tokenize, batched=True, remove_columns=["text"])
        test_set = test_set.map(tokenize, batched=True, remove_columns=["text"])

        self.finetune_dataset = {
            "train": train_set,
            "validation": valid_set,
            "test": test_set
        }
          
        model = BertForSequenceClassification.from_pretrained(
            self.model_dir,
            num_labels=3,
            id2label={0: "unacceptable", 1: "acceptable", 2: "mix"},
            label2id={"unacceptable": 0, "acceptable": 1, "mix": 2}
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)

            acc = accuracy_score(labels, preds)
            cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
            macro_f1 = f1_score(labels, preds, average='macro')
            weighted_f1 = f1_score(labels, preds, average='weighted')

            # Get per-class precision, recall, f1
            _, _, f1s, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1])
            f1_positive = f1s[1]  # label=1

            return {
                "accuracy": acc,
                "confusion_matrix": cm.tolist(),
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                # "f1_positive": f1_positive  # new key
            }

        for param in model.bert.parameters():
            param.requires_grad = False
        for param in model.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in model.bert.pooler.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

        training_args = TrainingArguments(
            output_dir=f"./models/{self.lang}-acceptability2-{[f for f in self.model_dir.split('/') if f][-1]}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            optim="adamw_torch",
            learning_rate=1e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            logging_steps=20,
            report_to="tensorboard",
            eval_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            # metric_for_best_model="eval_f1_positive",
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
            save_steps=1000,
        )
        seqeval = evaluate.load("seqeval")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.finetune_dataset["train"],
            eval_dataset=self.finetune_dataset["validation"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        trainer.train()

        metrics = trainer.evaluate(
            eval_dataset=self.finetune_dataset["test"],
        )
        print(f"Accuracy: {metrics['eval_accuracy']}")
        print(f"Macro F1: {metrics['eval_macro_f1']}")     
          
