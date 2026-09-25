from datasets import load_from_disk, Dataset, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForTokenClassification,
)
from transformers.modeling_outputs import TokenClassifierOutput
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, train_test_split
import evaluate
from utils import get_subset
import torch.nn as nn
import re
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SiniticPreTrainer:
    def __init__(self, lang="", model_dir="./models/yue-monolingual", scratch=False, data=None):
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

        if self.from_scratch:
            if os.path.exists(os.path.join(self.model_dir, "config.json")):
                # Reuse the architecture (BERT, XLM-R, ModernBERT, ...) of model_dir, with fresh weights
                config = AutoConfig.from_pretrained(
                    self.model_dir,
                    vocab_size=len(self.tokenizer),
                    pad_token_id=self.tokenizer.pad_token_id,
                    trust_remote_code=True,
                )
            else:
                # model_dir only holds a tokenizer: fall back to a BERT-base architecture
                config = AutoConfig.for_model(
                    "bert",
                    vocab_size=len(self.tokenizer),
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    max_position_embeddings=514,  # 512 + 2
                    type_vocab_size=2,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
        else:
            model = AutoModelForMaskedLM.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
            )
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
    def __init__(self, lang="yue", model_dir="./models/yue-monolingual", scratch=False, data=None):
        super().__init__(lang, model_dir, scratch, data)
        # checking data happens in run.py
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir,
            trust_remote_code=True,)

    def preprocess_data(self):
        if self.data == "wiki":
            self.wiki_preprocess_data()
        elif self.data == "cantonese-sentences":
            self.wiki_preprocess_data() # for future use if there's a need to implement something different
        else:
            raise ValueError(f"{self.data} not supported--please choose between `wiki` or `cantonese-sentences`.")


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


class WuPreTrainer(SiniticPreTrainer):
    def __init__(self, lang="wuu", model_dir="./models/yue-monolingual"):
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir,
            trust_remote_code=True,)


def compute_classification_metrics(num_labels):
    """Shared accuracy/F1/confusion-matrix metrics for single-label sequence classification."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        acc = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds, labels=list(range(num_labels)))
        macro_f1 = f1_score(labels, preds, average='macro')
        weighted_f1 = f1_score(labels, preds, average='weighted')

        return {
            "accuracy": acc,
            "confusion_matrix": cm.tolist(),
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
    return compute_metrics


def load_jsonl_splits(paths):
    """Load {split_name: jsonl_path} into HF Datasets, keyed by split name."""
    return {split: load_dataset("json", data_files=str(path), split="train") for split, path in paths.items()}


class CantoFineTuningBase:
    """Lightweight base for fine-tuning tasks: only needs a tokenizer and the base model
    directory, unlike CantoPreTrainer, which also requires the Cantonese Wikipedia corpus."""

    def __init__(self, lang="yue", model_dir="./models/yue-monolingual"):
        self.lang = lang
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model directory {self.model_dir} not found."
                f"Please first run `python download.py --lang={lang} --model_dir={model_dir}`."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir,
            trust_remote_code=True,)


class CantoSequenceClassificationFineTuner(CantoFineTuningBase):
    """Shared base for single- and paired-sentence classification tasks (nli, sentiment,
    ld, laj), all of which use AutoModelForSequenceClassification and differ only in input
    shape, label space, and where their data lives."""

    task_name = None
    text_fields = ("sentence",)
    id2label = {}
    split_paths = {}
    learning_rate = 2e-5
    num_train_epochs = 3
    per_device_batch_size = 64
    max_length = 128

    def __init__(self, lang="yue", model_dir="./models/yue-monolingual", eval_only=False):
        super().__init__(lang, model_dir)
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = len(self.id2label)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            trust_remote_code=True,
        )
        self.finetune_dataset = None
        self.preprocess_data(eval_only=eval_only)

        model_basename = [f for f in self.model_dir.split('/') if f][-1]
        self.training_args = TrainingArguments(
            output_dir=f"./models/{self.lang}-{self.task_name}-{model_basename}",
            overwrite_output_dir=True,
            num_train_epochs=self.num_train_epochs,
            optim="adamw_torch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_batch_size,
            per_device_eval_batch_size=self.per_device_batch_size,
            logging_steps=100,
            report_to="tensorboard",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
        )

    def load_raw_splits(self, eval_only=False):
        """Return {'train', 'validation', 'test'} raw (untokenized) examples. Default
        implementation reads self.split_paths; subclasses with unusual data sources
        (e.g. nli) override this entirely."""
        paths = {"test": self.split_paths["test"]}
        if not eval_only:
            paths.update({k: v for k, v in self.split_paths.items() if k != "test" and v is not None})

        raw = load_jsonl_splits(paths)
        for split in ("train", "validation", "test"):
            raw.setdefault(split, None)
        return raw

    def _map_label(self, example):
        label = example["label"]
        if isinstance(label, str):
            example["label"] = self.label2id[label]
        return example

    def preprocess_data(self, eval_only=False):
        raw = self.load_raw_splits(eval_only=eval_only)

        def tokenize(examples):
            args = [examples[field] for field in self.text_fields]
            return self.tokenizer(*args, truncation=True, padding="max_length", max_length=self.max_length)

        self.finetune_dataset = {}
        for split, ds in raw.items():
            if ds is None:
                self.finetune_dataset[split] = None
                continue
            ds = ds.map(self._map_label)
            ds = ds.map(tokenize, batched=True)
            self.finetune_dataset[split] = ds

    def finetune(self):
        eval_dataset = self.finetune_dataset["validation"]
        if eval_dataset is None:
            eval_dataset = self.finetune_dataset["test"]

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.finetune_dataset["train"],
            eval_dataset=eval_dataset,
            compute_metrics=compute_classification_metrics(self.num_labels),
        )

        trainer.train()
        model_basename = [f for f in self.model_dir.split('/') if f][-1]
        trainer.save_model(f"./models/{self.lang}-{self.task_name}-{model_basename}")
        self.eval(trainer)

    def eval(self, trainer):
        trainer.compute_metrics = compute_classification_metrics(self.num_labels)
        metrics = trainer.evaluate(eval_dataset=self.finetune_dataset["test"])
        print(f"Accuracy: {metrics['eval_accuracy']}")
        print(f"Macro F1: {metrics['eval_macro_f1']}")
        print(f"Confusion matrix: {metrics['eval_confusion_matrix']}")


class CantoNLIFineTuner(CantoSequenceClassificationFineTuner):
    task_name = "nli"
    text_fields = ("premise", "hypothesis")
    id2label = {0: "entailment", 1: "not_entailment"}

    def load_raw_splits(self, eval_only=False):
        nli_data = load_from_disk("./data/yue-nli-local")

        def unroll(split):
            rows = []
            for s in split:
                rows.append({"premise": s["anchor"], "hypothesis": s["positive"], "label": 0})
                rows.append({"premise": s["anchor"], "hypothesis": s["negative"], "label": 1})
            return Dataset.from_list(rows)

        test = unroll(get_subset(nli_data["test"]))
        if eval_only:
            return {"train": None, "validation": None, "test": test}
        return {
            "train": unroll(nli_data["train"]),
            "validation": unroll(nli_data["dev"]),
            "test": test,
        }


class CantoSentimentFineTuner(CantoSequenceClassificationFineTuner):
    task_name = "sentiment"
    text_fields = ("sentence",)
    id2label = {0: "smile", 1: "ok", 2: "cry"}
    split_paths = {
        "train": "data/sentiment/train.jsonl",
        "validation": "data/sentiment/valid.jsonl",
        "test": "data/sentiment/test.jsonl",
    }


class CantoLangDetectFineTuner(CantoSequenceClassificationFineTuner):
    task_name = "ld"
    text_fields = ("sentence",)
    id2label = {0: "cantonese", 1: "mandarin", 2: "corrupted"}
    split_paths = {
        "train": "data/ld/ld_train.jsonl",
        "validation": "data/ld/ld_val.jsonl",
        "test": "data/ld/ld_test.jsonl",
    }


class CantoLAJFineTuner(CantoSequenceClassificationFineTuner):
    task_name = "laj"
    text_fields = ("sentence",)
    id2label = {0: "unacceptable", 1: "acceptable"}
    split_paths = {
        "train": "data/laj/laj_finetune/finetune_train.jsonl",
        "validation": None,
        "test": "data/laj/laj_finetune/finetune_test.jsonl",
    }


class CantoPOSFineTuner(CantoFineTuningBase):
    def __init__(self, lang, model_dir):
        super().__init__(lang, model_dir)
        self.finetune_dataset = None
        self.pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
                    'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        self.tag2id = {tag: i for i, tag in enumerate(self.pos_tags)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}

    def preprocess_data(self):
        # Expected format: {"train": [{"tokens": [...], "upos": [...]}], ...}
        raw_data = load_jsonl_splits({
            "train": "data/pos/pos_train.jsonl",
            "test": "data/pos/pos_test.jsonl",
        })

        def align_labels_with_tokens(examples):
            tokenized = self.tokenizer(
                examples["tokens"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=128
            )

            labels = []
            for i, word_ids in enumerate(tokenized.word_ids(batch_index=i) for i in range(len(examples["tokens"]))):
                word_labels = examples["upos"][i]
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
        tokenized_data = {split: ds.map(align_labels_with_tokens, batched=True) for split, ds in raw_data.items()}
        self.finetune_dataset = {
            "train": tokenized_data["train"],
            "test": tokenized_data.get("test", tokenized_data["train"])
        }

    def finetune(self):
        self.preprocess_data()

        if any(split not in self.finetune_dataset for split in ["train", "test"]):
            raise ValueError(f"'train' and 'validation' splits must be present in finetune_dataset."
                             f"Found: {self.finetune_dataset.keys()}")

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_dir,
            num_labels=len(self.tag2id),
            id2label=self.id2tag,
            label2id=self.tag2id,
            trust_remote_code=True,
        )

        model.resize_token_embeddings(len(self.tokenizer))

        training_args = TrainingArguments(
            output_dir=f"./models/{self.lang}-pos-{self.model_dir.strip('/').split('/')[-1]}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
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



class EncoderForDependencyParsing(nn.Module):
    """
    Architecture-agnostic encoder (BERT, XLM-R, ModernBERT, ...) with two token-level heads:
    head_classifier: predicts head index in [0..max_length-1] (we map ROOT -> [CLS] position 0)
    rel_classifier: predicts dependency relation label for each token
    """
    def __init__(self, model_dir, num_rel_labels, max_length=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        config = self.encoder.config
        # Dropout config names differ across architectures (ModernBERT has no hidden_dropout_prob)
        dropout = next(
            (getattr(config, name) for name in ("classifier_dropout", "hidden_dropout_prob", "dropout")
             if getattr(config, name, None) is not None),
            0.1,
        )
        self.dropout = nn.Dropout(dropout)

        # Predict head index among max_length token positions
        self.head_classifier = nn.Linear(config.hidden_size, max_length)
        # Predict relation label
        self.rel_classifier = nn.Linear(config.hidden_size, num_rel_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_head=None,   # [B, L] with indices in [0..L-1], -100 to ignore
        labels_rel=None,    # [B, L] with rel ids, -100 to ignore
        **kwargs
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        seq = self.dropout(outputs.last_hidden_state)  # [B, L, H]

        head_logits = self.head_classifier(seq)  # [B, L, max_length] (i.e. [B, L, L])
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


class CantoDEPSFineTuner(CantoFineTuningBase):
    def __init__(self, lang, model_dir):
        super().__init__(lang, model_dir)
        self.finetune_dataset = None
        self.max_length = 128

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
          - 'tokens':  list[str] words
          - 'heads':   list[int] UD heads (0=ROOT, 1..n word indices)
          - 'deprels': list[str] dependency relations, 'root' for head==0
        We align to subwords and:
          - place gold labels only on first subword of each word
          - map head word index -> tokenized sequence index of that head's FIRST subword
          - map ROOT (0) -> [CLS] position index (usually 0)
        """
        raw = load_jsonl_splits({
            "train": "data/deps/deps_train.jsonl",
            "test": "data/deps/deps_test.jsonl",
        })

        def align(examples):
            # Tokenize with word alignment
            enc = self.tokenizer(
                examples["tokens"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_offsets_mapping=False,
            )

            B = len(examples["tokens"])
            labels_head = []
            labels_rel  = []

            for i in range(B):
                word_ids = enc.word_ids(batch_index=i)  # len = seq_len (incl CLS/SEP/PAD)
                words = examples["tokens"][i]
                heads = examples["heads"][i]  # UD heads: 0..len(words)
                rels  = examples["deprels"][i]

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

        tokenized = {split: ds.map(align, batched=True) for split, ds in raw.items()}
        self.finetune_dataset = {
            "train": tokenized["train"],
            "test": tokenized.get("test", tokenized["train"]),
         }

    def finetune(self):
        self.preprocess_data()

        model = EncoderForDependencyParsing(
            self.model_dir,
            num_rel_labels=len(self.rel2id),
            max_length=self.max_length,
        )

        model.encoder.resize_token_embeddings(len(self.tokenizer))

        args = TrainingArguments(
            output_dir=f"./models/{self.lang}-deps-{self.model_dir.strip('/').split('/')[-1]}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
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

        metrics = trainer.evaluate(self.finetune_dataset["test"])
        print(f"Final UAS: {metrics['eval_uas']}")
        print(f"Final LAS: {metrics['eval_las']}")

        trainer.save_model(f"./models/{self.lang}-deps-{self.model_dir.strip('/').split('/')[-1]}")



class CantoTokenClassificationFineTuner(CantoFineTuningBase):
    def __init__(self, lang="yue", model_dir="./models/yue-monolingual"):
        super().__init__(lang, model_dir)
        self.finetune_dataset = None

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
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
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
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_dir,
                num_labels=2,
                id2label=id2label,
                label2id=label2id,
                trust_remote_code=True,
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
            # trainer.save_model(f"./models/{self.lang}-nli-{[f for f in self.model_dir.split('/') if f][-1]}-fold-{fold}")

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

