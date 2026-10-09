import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "finetune"))

from scoring import canonicalize, normalize, summarize


def test_normalize_strips_quotes_spaces_and_case():
    assert normalize("  `Card_Arrival`. ") == "card_arrival"


def test_canonicalize_restores_dataset_casing():
    labels = ["Refund_not_showing_up", "card_arrival"]
    assert canonicalize("Refund_not_showing_up", labels) == "Refund_not_showing_up"
    assert canonicalize(" `card_arrival`. ", labels) == "card_arrival"
    assert canonicalize("made_up_label", labels) == "made_up_label"


def test_canonicalize_handles_label_with_question_mark():
    labels = ["reverted_card_payment?", "card_arrival"]
    assert canonicalize("reverted_card_payment", labels) == "reverted_card_payment?"
    assert canonicalize("reverted_card_payment?", labels) == "reverted_card_payment?"


def test_canonicalize_rejects_ambiguous_labels():
    with pytest.raises(ValueError):
        canonicalize("x", ["same?", "same"])


def test_summarize_counts_accuracy_and_invalid():
    labels = ["a", "b"]
    results = [
        {"pred": "a", "label": "a"},
        {"pred": "b", "label": "a"},
        {"pred": "nonsense", "label": "b"},
        {"pred": "b", "label": "b"},
    ]
    s = summarize(results, labels)
    assert s["n"] == 4 and s["accuracy"] == 0.5 and s["invalid_replies"] == 1