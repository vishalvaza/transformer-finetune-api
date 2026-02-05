import os
import pytest
from app.inference.predictor import Predictor

@pytest.mark.skipif(not os.path.isdir("artifacts/model"), reason="model not trained yet")
def test_predictor_runs() -> None:
    p = Predictor("artifacts/model", device="cpu")
    pred = p.predict("I love it!")
    assert pred.label
    assert 0.0 <= pred.score <= 1.0
