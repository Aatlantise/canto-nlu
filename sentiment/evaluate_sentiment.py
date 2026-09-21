import pandas as pd 
import torch
from transformers import (
    BertForSequenceClassification,
    AlbertTokenizer,
    BertTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os 
import wandb
import evaluate 
import argparse
from pathlib import Path


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Load and preprocess data
data = {
    "train": pd.read_csv("./train.tsv", sep='\t'),
    "test": pd.read_csv("./test.tsv", sep='\t'),
    "valid": pd.read_csv("./valid.tsv", sep='\t'),
}

# Convert polars dataframes to huggingface datasets
def convert_to_hf_dataset(df):
    label_map = {"smile": 0, "ok": 1, "cry": 2}
    labels = df['label'].map(label_map).astype(int).to_list()
    return Dataset.from_dict({
        'text': df['text_a'].astype(str).to_list(),
        'label': labels
    })

datasets = {split: convert_to_hf_dataset(df) for split, df in data.items()}

model_options = {
    "yue-scratch": "./yue-scratch-2/",
    "yue-transfer": "./yue-transfer",
    "bert-base-chinese": "google-bert/bert-base-chinese",
    "bert-base-cantonese": "indiejoseph/bert-base-cantonese"
}



parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='Flag to indicate training')
parser.add_argument('--test', action='store_true', help='Flag to indicate testing')
parser.add_argument('--wandb', action='store_true', help='Log to wandb')
args = parser.parse_args()


for model_name, model_path in model_options.items():
    print(f"Training model: {model_name}")

    if args.wandb:
        wandb.init(
            project="openrice-senti",
            name=f"{model_name}-train{args.train}-test{args.test}",  # Give each run a distinct name
            reinit=True  # Allow multiple runs in the same script
        )

    if args.test:
        model_path = f"results/{model_name}"
        model_path = list(Path(model_path).iterdir())[0]
        assert model_path.name.startswith("checkpoint-")

    if "yue-scratch" in str(model_path):
        tokenizer_class = AlbertTokenizer
    else:
        tokenizer_class = BertTokenizer

    # Initialize tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=128)

    # Tokenize datasets
    tokenized_datasets = {
        split: dataset.map(tokenize_function, batched=True)
        for split, dataset in datasets.items()
    }

    # Initialize model for sequence classification
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=3,  # Binary classification
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=250,  # Evaluate every 250 steps
        save_strategy="steps",
        save_steps=250,  # Save every 250 steps
        save_total_limit=1,  # Only keep the best checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
        report_to="wandb"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    if args.train:
        # Train the model
        trainer.train()
    
    if args.test:
        # Evaluate the model
        f1_metric = evaluate.load("f1")
        test_set = tokenized_datasets["test"]
        predictions = trainer.predict(test_dataset=test_set)
        predicted_labels = np.argmax(predictions.predictions, axis=1)
        f1 = f1_metric.compute(predictions=predicted_labels, 
            references=predictions.label_ids, average="macro")
        print(f"Test F1 Score for {model_name}: {f1['f1']}")

