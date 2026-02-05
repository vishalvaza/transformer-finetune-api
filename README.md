# Transformer Fine-tune + Inference API (FastAPI)

Production-style template for:
- Fine-tuning a Transformer classifier (Hugging Face Trainer)
- Saving artifacts to `artifacts/model`
- Serving predictions via FastAPI
- Tests + CI + Docker

## Quickstart (local)

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install ".[dev]"
