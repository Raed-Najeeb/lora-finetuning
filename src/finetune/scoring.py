def normalize(reply: str) -> str:
    """Strip whitespace, quotes and trailing punctuation; lowercase."""
    return reply.strip().strip("`'\".").strip().lower()


def canonicalize(reply: str, labels: list[str]) -> str:
    """Map a reply onto the exact dataset label it names (ignoring case), if any.

    Some dataset labels contain capital letters (e.g. Refund_not_showing_up),
    so we match case-insensitively but return the label exactly as the
    dataset spells it. Unrecognised replies come back cleaned but unmapped.
    """
    lookup = {label.lower(): label for label in labels}
    cleaned = normalize(reply)
    return lookup.get(cleaned, cleaned)


def summarize(results: list[dict], labels: list[str]) -> dict:
    label_set = set(labels)
    n = len(results)
    correct = sum(r["pred"] == r["label"] for r in results)
    invalid = sum(r["pred"] not in label_set for r in results)
    return {"n": n, "accuracy": correct / n, "invalid_replies": invalid}