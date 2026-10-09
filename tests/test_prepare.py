import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "finetune"))

from prepare_data import build_baseline_messages, build_train_messages, stratified_split


def _fake_examples():
    return [
        {"text": f"{label}-{i}", "label": label}
        for label in ["zzz_one", "zzz_two", "zzz_three"]
        for i in range(20)
    ]


def test_split_has_no_overlap_and_keeps_everything():
    examples = _fake_examples()
    train, val = stratified_split(examples)
    assert {e["text"] for e in train}.isdisjoint({e["text"] for e in val})
    assert len(train) + len(val) == len(examples)


def test_every_label_appears_in_val():
    _, val = stratified_split(_fake_examples())
    assert {e["label"] for e in val} == {"zzz_one", "zzz_two", "zzz_three"}


def test_split_is_reproducible():
    assert stratified_split(_fake_examples()) == stratified_split(_fake_examples())


def test_baseline_lists_all_labels_and_has_no_answer():
    msgs = build_baseline_messages("hello", ["zzz_one", "zzz_two"])
    assert "zzz_one" in msgs[0]["content"] and "zzz_two" in msgs[0]["content"]
    assert all(m["role"] != "assistant" for m in msgs)


def test_train_messages_end_with_label_and_skip_label_list():
    msgs = build_train_messages("hello", "card_arrival")
    assert msgs[-1] == {"role": "assistant", "content": "card_arrival"}
    assert "intents:" not in msgs[0]["content"]