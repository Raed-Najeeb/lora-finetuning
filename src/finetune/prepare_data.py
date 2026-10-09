import json
import random
from collections import defaultdict
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
SEED = 42
VAL_FRACTION = 0.1

BASELINE_SYSTEM = (
    "You are a banking customer-support classifier. Classify the customer "
    "message into exactly one of these intents:\n{labels}\n"
    "Reply with the intent name only."
)
TRAIN_SYSTEM = (
    "Classify the banking customer message into one intent. "
    "Reply with the intent name only."
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(examples, val_fraction=VAL_FRACTION, seed=SEED):
    """Split so every label keeps roughly the same share in train and val."""
    by_label = defaultdict(list)
    for ex in examples:
        by_label[ex["label"]].append(ex)

    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


def build_baseline_messages(text, labels):
    """Prompting method: the label list travels with every request."""
    return [
        {"role": "system", "content": BASELINE_SYSTEM.format(labels="\n".join(labels))},
        {"role": "user", "content": text},
    ]


def build_train_messages(text, label):
    """Fine-tuning method: short instruction, plus the correct answer to learn."""
    return [
        {"role": "system", "content": TRAIN_SYSTEM},
        {"role": "user", "content": text},
        {"role": "assistant", "content": label},
    ]


def main():
    train_all = load_jsonl(RAW_DIR / "train.jsonl")
    labels = sorted({ex["label"] for ex in train_all})

    train, val = stratified_split(train_all)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)
    (OUT_DIR / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    print(f"train: {len(train)}  val: {len(val)}  labels: {len(labels)}")

    example = val[0]
    baseline_prompt = build_baseline_messages(example["text"], labels)[0]["content"]
    train_prompt = build_train_messages(example["text"], example["label"])[0]["content"]
    print(f"\nbaseline system prompt: {len(baseline_prompt)} characters")
    print(f"training system prompt: {len(train_prompt)} characters")
    print(f"\nexample message: {example['text']}")
    print(f"correct label:   {example['label']}")


if __name__ == "__main__":
    main()