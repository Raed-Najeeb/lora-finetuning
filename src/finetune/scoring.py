def normalize(reply: str) -> str:
    """Strip whitespace, quotes and trailing punctuation; lowercase."""
    return reply.strip().strip("`'\".").strip().lower()


def _match_key(text: str) -> str:
    """Key used to match a reply to a label: case-insensitive, ignoring a trailing '?'.

    The dataset has quirks: one label is spelled with a capital letter
    (Refund_not_showing_up) and one ends in a question mark
    (reverted_card_payment?). A model that names the right intent should not be
    marked wrong over that punctuation or casing.
    """
    return normalize(text).rstrip("?").strip()


def canonicalize(reply: str, labels: list[str]) -> str:
    """Map a reply onto the exact dataset label it names, if any.

    Returns the label exactly as the dataset spells it. Replies that match no
    label come back cleaned but unmapped, so they still count as invalid.
    """
    lookup = {_match_key(label): label for label in labels}
    if len(lookup) != len(labels):
        raise ValueError("two labels collide after normalisation; matching would be ambiguous")
    return lookup.get(_match_key(reply), normalize(reply))


def summarize(results: list[dict], labels: list[str]) -> dict:
    label_set = set(labels)
    n = len(results)
    correct = sum(r["pred"] == r["label"] for r in results)
    invalid = sum(r["pred"] not in label_set for r in results)
    return {"n": n, "accuracy": correct / n, "invalid_replies": invalid}