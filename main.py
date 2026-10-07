from datasets import load_from_disk, Dataset, load_dataset, disable_progress_bars
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
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import is_torch_bf16_gpu_available
import os
import tempfile
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import evaluate
from utils import get_subset
import torch
import torch.nn as nn
import re
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
disable_progress_bars()

# Pre-training logs training loss ~20 times per run regardless of dataset/batch size:
# transformers reads a logging_steps value below 1 as a fraction of total training steps.
# Fine-tuning prints metrics to stdout once per epoch instead (logging_strategy="epoch").
LOGGING_STEPS = 0.05

# Fine-tuning saves neither weights nor TensorBoard logs, but Trainer still requires an
# output_dir and creates it on startup, so point every run at one shared temp directory.
FINETUNE_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "sinitic-nlu-finetune")

class SiniticPreTrainer:
    def __init__(self, lang="", model_dir="./models/yue-monolingual", scratch=False, data=None, seed=42):
        self.ds = None
        self.tokenizer = None
        self.lang = lang
        self.model_dir = model_dir
        self.tokenized_ds = None
        self.lm_dataset = None
        self.from_scratch = scratch
        self.data = data
        self.seed = seed

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

        run_name = f"{self.lang}-scratch" if self.from_scratch else f"{self.lang}-transfer"
        output_dir_name = f"./{run_name}"

        training_args = TrainingArguments(
            # Checkpoints and the final model go to output_dir; TensorBoard logs go to ./logs
            output_dir=output_dir_name,
            logging_dir=f"./logs/{run_name}",
            num_train_epochs=2,
            per_device_train_batch_size=128,
            dataloader_num_workers=8,
            learning_rate=1e-5,
            warmup_steps=10000,
            weight_decay=0.01,
            save_steps=10000,
            save_total_limit=2,
            logging_steps=LOGGING_STEPS,
            disable_tqdm=True,
            report_to="tensorboard",
            seed=self.seed,
            eval_strategy="steps",
            eval_steps=1000,
            fp16=False,
            # bf16 needs an Ampere+ GPU (e.g. H100); fall back to fp32 elsewhere instead of crashing
            bf16=is_torch_bf16_gpu_available(),
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
    def __init__(self, lang="yue", model_dir="./models/yue-monolingual", scratch=False, data=None, seed=42):
        super().__init__(lang, model_dir, scratch, data, seed)
        # checking data happens in run.py
        if data == "cantonese-sentences":
            if not os.path.exists("./data/cantonese-sentences"):
                raise FileNotFoundError(
                    "Cantonese Sentences dataset not found at ./data/cantonese-sentences. Fetch it with "
                    "`load_dataset('raptorkwok/cantonese_sentences').save_to_disk('./data/cantonese-sentences')`."
                )
            self.ds = load_from_disk("./data/cantonese-sentences")
        else:
            if not os.path.exists("./data/yue-wiki-full-local"):
                raise FileNotFoundError(
                    "Cantonese Wiki dataset not found at ./data/yue-wiki-full-local. Fetch it with "
                    "`load_dataset('wikimedia/wikipedia', '20231101.zh-yue', split='train')"
                    ".save_to_disk('./data/yue-wiki-full-local')`."
                )
            self.ds = load_from_disk("./data/yue-wiki-full-local")

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model directory {self.model_dir} not found. "
                f"Run `python download.py` to fetch the base models into ./models/."
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

    def __init__(self, lang="yue", model_dir="./models/yue-monolingual", per_device_batch_size=64, seed=42,
                 gradient_accumulation_steps=1):
        self.lang = lang
        self.model_dir = model_dir
        self.seed = seed
        # Default suits base-sized encoders; lower it for large models (e.g. 16 for ModernBERT-large)
        self.per_device_batch_size = per_device_batch_size
        # Raise together with lowering per_device_batch_size to keep the effective batch size
        # (and so the number of optimizer steps) comparable across models
        self.gradient_accumulation_steps = gradient_accumulation_steps
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
    max_length = 128

    def __init__(self, lang="yue", model_dir="./models/yue-monolingual", eval_only=False, per_device_batch_size=64,
                 seed=42, gradient_accumulation_steps=1):
        super().__init__(lang, model_dir, per_device_batch_size, seed, gradient_accumulation_steps)
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

        self.training_args = TrainingArguments(
            output_dir=FINETUNE_OUTPUT_DIR,
            overwrite_output_dir=True,
            num_train_epochs=self.num_train_epochs,
            optim="adamw_torch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_batch_size,
            per_device_eval_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            disable_tqdm=True,
            report_to="none",
            seed=self.seed,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="no",
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
        self.eval(trainer)

    def eval(self, trainer):
        trainer.compute_metrics = compute_classification_metrics(self.num_labels)
        metrics = trainer.evaluate(eval_dataset=self.finetune_dataset["test"])
        print(f"Final {self.task_name} test accuracy: {metrics['eval_accuracy']}")
        print(f"Final {self.task_name} test macro F1: {metrics['eval_macro_f1']}")
        print(f"Final {self.task_name} test confusion matrix: {metrics['eval_confusion_matrix']}")


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
    def __init__(self, lang, model_dir, per_device_batch_size=64, seed=42, gradient_accumulation_steps=1):
        super().__init__(lang, model_dir, per_device_batch_size, seed, gradient_accumulation_steps)
        self.finetune_dataset = None
        # The 15 UPOS tags attested in UD_Cantonese-HK (SYM and X are unused):
        # https://universaldependencies.org/treebanks/yue_hk/index.html
        self.pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
                    'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'VERB']
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
            output_dir=FINETUNE_OUTPUT_DIR,
            overwrite_output_dir=True,
            num_train_epochs=20,
            learning_rate=2e-5,
            per_device_train_batch_size=self.per_device_batch_size,
            per_device_eval_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="no",
            disable_tqdm=True,
            report_to="none",
            seed=self.seed,
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

        metrics = trainer.evaluate(self.finetune_dataset["test"])
        print(f"Final pos test accuracy: {metrics['eval_accuracy']}")
        print(f"Final pos test macro F1: {metrics['eval_macro_f1']}")
        print(f"Final pos test micro F1: {metrics['eval_micro_f1']}")



class Biaffine(nn.Module):
    """Biaffine scorer (Dozat & Manning, 2017): x^T W y, with an optional constant 1 appended to
    x and/or y so that the same weight tensor also holds the linear and bias terms."""
    def __init__(self, in_dim, out_dim=1, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        # Zero init: every candidate starts with the same score
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim + bias_x, in_dim + bias_y))

    def _append_ones(self, t, bias):
        return torch.cat([t, torch.ones_like(t[..., :1])], dim=-1) if bias else t

    def forward(self, x, y):
        """x: [B, L, D] dependents, y: [B, L, D] heads -> [B, out_dim, L, L] (dependent x head)."""
        x = self._append_ones(x, self.bias_x)
        y = self._append_ones(y, self.bias_y)
        return torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)

    def pairwise(self, x, y):
        """Score aligned pairs only: x, y: [B, L, D] -> [B, L, out_dim]."""
        x = self._append_ones(x, self.bias_x)
        y = self._append_ones(y, self.bias_y)
        return torch.einsum("bxi,oij,bxj->bxo", x, self.weight, y)


class EncoderForDependencyParsing(nn.Module):
    """
    Architecture-agnostic encoder (BERT, XLM-R, ModernBERT, ...) with a biaffine parser on top
    (Dozat & Manning, 2017). Each word is represented by its first subword; ROOT is the [CLS]
    position. Arcs are scored for every (dependent, head) pair, and the relation label is scored
    from the (dependent, head) pair: the gold head in training, the predicted head otherwise.
    """
    mlp_dropout = 0.33

    def __init__(self, model_dir, num_rel_labels, arc_dim=500, rel_dim=100):
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

        def mlp(out_dim):
            return nn.Sequential(nn.Linear(config.hidden_size, out_dim), nn.LeakyReLU(0.1),
                                 nn.Dropout(self.mlp_dropout))

        # Separate "as dependent" and "as head" views of each token
        self.arc_dep, self.arc_head = mlp(arc_dim), mlp(arc_dim)
        self.rel_dep, self.rel_head = mlp(rel_dim), mlp(rel_dim)
        # Arc score = dep^T U head + u^T head (no head-independent dependent term)
        self.arc_attn = Biaffine(arc_dim, 1, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(rel_dim, num_rel_labels, bias_x=True, bias_y=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_candidates=None,  # [B, L] 1 where a token may be a head ([CLS]/ROOT and first subwords)
        labels_head=None,      # [B, L] with indices in [0..L-1], -100 to ignore
        labels_rel=None,       # [B, L] with rel ids, -100 to ignore
        **kwargs
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        seq = self.dropout(outputs.last_hidden_state)  # [B, L, H]

        arc_scores = self.arc_attn(self.arc_dep(seq), self.arc_head(seq)).squeeze(1)  # [B, L, L]
        # Only ROOT and first subwords can be heads, and no token heads itself
        invalid = ~head_candidates.bool().unsqueeze(1).expand_as(arc_scores)
        invalid = invalid | torch.eye(arc_scores.size(-1), dtype=torch.bool, device=arc_scores.device)
        arc_scores = arc_scores.masked_fill(invalid, torch.finfo(arc_scores.dtype).min)

        # Score relations against the gold head when training, the predicted head otherwise
        if labels_head is not None and self.training:
            heads = labels_head.clamp(min=0)
        else:
            heads = arc_scores.argmax(-1)
        rel_dep, rel_head = self.rel_dep(seq), self.rel_head(seq)
        rel_head = rel_head.gather(1, heads.unsqueeze(-1).expand(-1, -1, rel_head.size(-1)))
        rel_logits = self.rel_attn.pairwise(rel_dep, rel_head)  # [B, L, R]

        loss = None
        if labels_head is not None and labels_rel is not None:
            ce = nn.CrossEntropyLoss(ignore_index=-100)
            head_loss = ce(arc_scores.view(-1, arc_scores.size(-1)), labels_head.view(-1))
            rel_loss  = ce(rel_logits.view(-1, rel_logits.size(-1)), labels_rel.view(-1))
            loss = head_loss + rel_loss

        # Return a tuple so Trainer hands both logits to compute_metrics
        return {"loss": loss, "logits": (arc_scores, rel_logits)}


class CantoDEPSFineTuner(CantoFineTuningBase):
    encoder_learning_rate = 2e-5
    parser_learning_rate = 1e-3

    def __init__(self, lang, model_dir, per_device_batch_size=64, seed=42, gradient_accumulation_steps=1):
        super().__init__(lang, model_dir, per_device_batch_size, seed, gradient_accumulation_steps)
        self.finetune_dataset = None
        self.max_length = 128

        # The 49 relations attested in UD_Cantonese-HK: 31 universal relations (expl, fixed, list,
        # orphan, goeswith and dep are unused) plus 18 language-specific subtypes. Relations missing
        # here get label -100 and are dropped from the relation loss and from UAS/LAS.
        # https://universaldependencies.org/treebanks/yue_hk/index.html
        self.dep_labels = [
            "root","nsubj","obj","iobj","obl","vocative","dislocated",
            "advcl","advmod","discourse","aux","cop","mark","nmod","appos",
            "nummod","acl","amod","det","clf","case","conj","cc",
            "flat","compound","parataxis","reparandum",
            "punct","csubj","xcomp","ccomp",
            # Language-specific subtypes
            "advcl:coverb","advmod:df","case:loc","clf:det",
            "compound:dir","compound:ext","compound:quant","compound:vo","compound:vv",
            "discourse:sp","mark:adv","mark:rel",
            "nsubj:pass","nsubj:periph","obj:periph",
            "obl:agent","obl:patient","obl:tmod",
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
            head_candidates = []

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
                first_tok_idx_set = set(first_tok_idx_of_word.values())

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
                head_candidates.append([int(tok_idx == root_tok_idx or tok_idx in first_tok_idx_set)
                                        for tok_idx in range(len(word_ids))])

            enc["labels_head"] = labels_head
            enc["labels_rel"]  = labels_rel
            enc["head_candidates"] = head_candidates
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
        )

        model.encoder.resize_token_embeddings(len(self.tokenizer))

        args = TrainingArguments(
            output_dir=FINETUNE_OUTPUT_DIR,
            overwrite_output_dir=True,
            num_train_epochs=30,
            learning_rate=self.encoder_learning_rate,
            warmup_ratio=0.1,
            per_device_train_batch_size=self.per_device_batch_size,
            per_device_eval_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="no",
            disable_tqdm=True,
            report_to="none",
            seed=self.seed,
        )

        # The randomly initialized parser layers need a far higher learning rate than the
        # pre-trained encoder; Trainer applies its linear warmup/decay schedule to both groups.
        encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
        parser_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": self.encoder_learning_rate},
            {"params": parser_params, "lr": self.parser_learning_rate},
        ], weight_decay=0.0)  # Trainer's own default; torch's AdamW would otherwise use 0.01

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
            optimizers=(optimizer, None),
        )

        trainer.train()

        metrics = trainer.evaluate(self.finetune_dataset["test"])
        print(f"Final deps UAS: {metrics['eval_uas']}")
        print(f"Final deps LAS: {metrics['eval_las']}")
