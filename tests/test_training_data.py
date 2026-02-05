from app.training.data import load_jsonl

def test_load_jsonl() -> None:
    ex = load_jsonl("tests/data/tiny_train.jsonl")
    assert len(ex) >= 2
    assert ex[0].text
    assert ex[0].label in (0, 1)
