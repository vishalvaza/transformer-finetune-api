from __future__ import annotations

from pathlib import Path
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from .data import load_jsonl

@torch.inference_mode()
def evaluate(model_dir: str, eval_path: str) -> dict[str, float]:
    model_path = Path(model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    ex = load_jsonl(eval_path)
    ds = Dataset.from_dict({"text": [e.text for e in ex], "label": [e.label for e in ex]})

    correct = 0
    total = 0
    for row in ds:
        inputs = tok(row["text"], return_tensors="pt", truncation=True, max_length=256)
        logits = model(**inputs).logits
        pred = int(torch.argmax(logits, dim=-1).item())
        correct += int(pred == int(row["label"]))
        total += 1

    return {"accuracy": float(correct / max(1, total))}

if __name__ == "__main__":
    print(evaluate("artifacts/model", "tests/data/tiny_eval.jsonl"))
