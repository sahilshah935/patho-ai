import re
from typing import List


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_words(text: str) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def edit_distance(source, target) -> int:
    if source == target:
        return 0

    if len(source) < len(target):
        source, target = target, source

    previous = list(range(len(target) + 1))

    for i, source_item in enumerate(source, start=1):
        current = [i]

        for j, target_item in enumerate(target, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (source_item != target_item)
            current.append(min(insert_cost, delete_cost, replace_cost))

        previous = current

    return previous[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)

    if not reference:
        return 0.0 if not prediction else 1.0

    return edit_distance(reference, prediction) / len(reference)


def word_error_rate(reference: str, prediction: str) -> float:
    reference_words = tokenize_words(reference)
    prediction_words = tokenize_words(prediction)

    if not reference_words:
        return 0.0 if not prediction_words else 1.0

    return edit_distance(reference_words, prediction_words) / len(reference_words)

