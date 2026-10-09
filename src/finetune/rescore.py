import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import canonicalize, summarize


def main():
    labels = json.loads(Path("data/processed/labels.json").read_text(encoding="utf-8"))
    for path in sorted(Path("results").glob("eval_*.jsonl")):
        name = path.stem.removeprefix("eval_")
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        for r in rows:
            r["pred"] = canonicalize(r["reply"], labels)

        new = summarize(rows, labels)
        summary_path = Path("results") / f"eval_{name}_summary.json"
        old = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        print(f"{name}: accuracy {old.get('accuracy')} -> {new['accuracy']:.4f}, "
              f"invalid {old.get('invalid_replies')} -> {new['invalid_replies']}")

        old.update(new)
        old["name"] = name
        summary_path.write_text(json.dumps(old, indent=2), encoding="utf-8")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()