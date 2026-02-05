from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

@dataclass(frozen=True)
class Prediction:
    label: str
    score: float

class Predictor:
    def __init__(self, model_name_or_path: str, device: str = "cpu") -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(device)
        self.model.eval()

        # Build id2label map safely
        cfg = self.model.config
        self.id2label = getattr(cfg, "id2label", None) or {0: "NEGATIVE", 1: "POSITIVE"}

    @torch.inference_mode()
    def predict(self, text: str) -> Prediction:
        inputs: dict[str, Any] = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        idx = int(torch.argmax(probs).item())
        return Prediction(label=self.id2label.get(idx, str(idx)), score=float(probs[idx].item()))
