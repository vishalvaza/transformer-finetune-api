from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Example:
    text: str
    label: int  # 0/1

def load_jsonl(path: str | Path) -> list[Example]:
    path = Path(path)
    out: list[Example] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out.append(Example(text=row["text"], label=int(row["label"])))
    return out
