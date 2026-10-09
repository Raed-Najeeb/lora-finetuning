import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import build_baseline_messages, load_jsonl

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1:8b"
VAL_PATH = Path("data/processed/val.jsonl")
LABELS_PATH = Path("data/processed/labels.json")
OUT_PATH = Path("results/baseline_val.jsonl")


def normalize(reply: str) -> str:
    return reply.strip().strip("`'\".").strip().lower()


def classify(text: str, labels: list[str]) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": build_baseline_messages(text, labels),
        "stream": False,
        "options": {"temperature": 0.0},
    }
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of validation examples (0 = all)")
    args = parser.parse_args()

    val = load_jsonl(VAL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    label_set = set(labels)

    if args.n > 0:
        val = random.Random(0).sample(val, args.n)

    results = []
    start = time.time()
    for i, ex in enumerate(val, 1):
        reply = classify(ex["text"], labels)
        pred = normalize(reply)
        results.append({
            "text": ex["text"],
            "label": ex["label"],
            "reply": reply,
            "pred": pred,
            "correct": pred == ex["label"],
            "valid_label": pred in label_set,
        })
        mark = "OK " if results[-1]["correct"] else "BAD"
        print(f"[{i}/{len(val)}] {mark} | correct: {ex['label']} | model said: {pred}")

    elapsed = time.time() - start
    n = len(results)
    accuracy = sum(r["correct"] for r in results) / n
    invalid = sum(not r["valid_label"] for r in results)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nExamples: {n}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Replies that were not a valid label at all: {invalid}")
    print(f"Average time per example: {elapsed / n:.2f}s")


if __name__ == "__main__":
    main()