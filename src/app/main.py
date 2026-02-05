from fastapi import FastAPI
from .config import settings
from .schemas import PredictRequest, PredictResponse
from .inference.predictor import Predictor

app = FastAPI(title="Transformer Inference API", version="0.1.0")

predictor = Predictor(model_name_or_path=settings.model_dir, device=settings.device)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    pred = predictor.predict(req.text)
    return PredictResponse(label=pred.label, score=pred.score)
