import json
import sys
from collections import Counter
from pathlib import Path


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ft_2k"
    labels = set(json.loads(Path("data/processed/labels.json").read_text(encoding="utf-8")))
    rows = [json.loads(line) for line in open(f"results/eval_{name}.jsonl", encoding="utf-8")]

    wrong = [r for r in rows if r["pred"] != r["label"]]
    invalid = [r for r in wrong if r["pred"] not in labels]
    confused = [r for r in wrong if r["pred"] in labels]

    print(f"{name}: {len(rows) - len(wrong)}/{len(rows)} correct")
    print(f"wrong answers: {len(wrong)}  (invented labels: {len(invalid)}, real-but-wrong labels: {len(confused)})")

    print("\nInvented labels (model reply | correct label):")
    for r in invalid:
        print(f"  {r['pred']!r} | {r['label']}")

    print("\nMost common real-but-wrong confusions (count, correct -> predicted):")
    pairs = Counter((r["label"], r["pred"]) for r in confused)
    for (gold, pred), count in pairs.most_common(10):
        print(f"  {count}  {gold} -> {pred}")


if __name__ == "__main__":
    main()