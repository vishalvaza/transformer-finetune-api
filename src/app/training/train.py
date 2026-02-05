from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from .data import load_jsonl

def fine_tune(
    train_path: str,
    eval_path: str,
    base_model: str,
    out_dir: str,
) -> None:
    train_ex = load_jsonl(train_path)
    eval_ex = load_jsonl(eval_path)

    ds_train = Dataset.from_dict({"text": [e.text for e in train_ex], "label": [e.label for e in train_ex]})
    ds_eval = Dataset.from_dict({"text": [e.text for e in eval_ex], "label": [e.label for e in eval_ex]})

    tok = AutoTokenizer.from_pretrained(base_model)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=256)

    ds_train = ds_train.map(tok_fn, batched=True)
    ds_eval = ds_eval.map(tok_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        return {"accuracy": acc}

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,         # keep small; you can tune
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        compute_metrics=compute_metrics,
        tokenizer=tok,
    )
    trainer.train()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

if __name__ == "__main__":
    fine_tune(
        train_path="tests/data/tiny_train.jsonl",
        eval_path="tests/data/tiny_eval.jsonl",
        base_model="distilbert-base-uncased",
        out_dir="artifacts/model",
    )
