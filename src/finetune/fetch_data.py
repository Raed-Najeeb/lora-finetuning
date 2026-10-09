# src/finetune/fetch_data.py
import csv
import io
import json
from collections import Counter
from pathlib import Path

import requests

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
OUT_DIR = Path("data/raw")


def download_split(split: str) -> list[dict]:
    resp = requests.get(f"{BASE}/{split}.csv", timeout=60)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    rows = list(csv.DictReader(io.StringIO(resp.text)))
    assert "text" in rows[0] and "category" in rows[0], f"unexpected columns: {list(rows[0].keys())}"
    return [{"text": r["text"], "label": r["category"]} for r in rows]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        examples = download_split(split)
        path = OUT_DIR / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        counts = Counter(ex["label"] for ex in examples)
        print(f"{split}: {len(examples)} examples, {len(counts)} labels")
        print(f"  smallest class: {min(counts.values())}, largest class: {max(counts.values())}")
        print(f"  example: {examples[0]}")


if __name__ == "__main__":
    main()