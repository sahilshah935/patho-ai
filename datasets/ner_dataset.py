"""BIO Tagging Dataset Formatter for Token Classification (NER) fine-tuning.

Converts pathology report AnnotationRecords into token-level BIO tagged
sequences for training HuggingFace AutoModelForTokenClassification models.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from pathology_ai.datasets.loader import AnnotationLoader, AnnotationRecord

# ---------------------------------------------------------------------------
# Label Taxonomy
# ---------------------------------------------------------------------------

LABEL_LIST: List[str] = [
    "O",
    "B-TEST_NAME",
    "I-TEST_NAME",
    "B-VALUE",
    "I-VALUE",
    "B-UNIT",
    "I-UNIT",
    "B-REF_RANGE",
    "I-REF_RANGE",
    "B-PATIENT_NAME",
    "I-PATIENT_NAME",
    "B-PATIENT_ID",
    "I-PATIENT_ID",
]

LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(LABEL_LIST)}


# ---------------------------------------------------------------------------
# BIO Tag Generator
# ---------------------------------------------------------------------------

def generate_bio_tags(record: AnnotationRecord) -> Tuple[List[str], List[str]]:
    """Convert an AnnotationRecord into word tokens and corresponding BIO tags.

    Args:
        record: An ``AnnotationRecord`` from ``AnnotationLoader``.

    Returns:
        Tuple of ``(words, bio_tags)``.
    """
    text = record.ground_truth_text or ""
    if not text:
        return [], []

    # Simple whitespace/punctuation word tokenization while tracking char spans
    words: List[str] = []
    word_spans: List[Tuple[int, int]] = []

    for m in re.finditer(r"\S+", text):
        words.append(m.group(0))
        word_spans.append((m.start(), m.end()))

    tags = ["O"] * len(words)

    # Helper to tag a string value in the text
    def _tag_entity(entity_val: str, b_tag: str, i_tag: str):
        if not entity_val or len(entity_val) < 2:
            return
        entity_clean = entity_val.strip()
        idx = text.find(entity_clean)
        if idx == -1:
            return
        end_idx = idx + len(entity_clean)

        first = True
        for i, (w_start, w_end) in enumerate(word_spans):
            # If word falls inside entity character span
            if w_start >= idx and w_end <= end_idx:
                if first:
                    tags[i] = b_tag
                    first = False
                else:
                    tags[i] = i_tag

    # Tag Patient Info
    if record.patient:
        _tag_entity(record.patient.name, "B-PATIENT_NAME", "I-PATIENT_NAME")
        _tag_entity(record.patient.patient_id, "B-PATIENT_ID", "I-PATIENT_ID")

    # Tag Test Results
    for test in record.all_tests():
        _tag_entity(test.test_name_raw, "B-TEST_NAME", "I-TEST_NAME")
        _tag_entity(test.value_raw, "B-VALUE", "I-VALUE")
        _tag_entity(test.unit_raw, "B-UNIT", "I-UNIT")
        _tag_entity(test.reference_range_raw, "B-REF_RANGE", "I-REF_RANGE")

    return words, tags


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PathologyNERDataset(Dataset):
    """PyTorch Dataset for fine-tuning token classification models."""

    def __init__(
        self,
        records: List[AnnotationRecord],
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Dict[str, torch.Tensor]] = []

        for record in records:
            words, tags = generate_bio_tags(record)
            if not words:
                continue

            # Tokenize words with subword alignment
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            word_ids = encoding.word_ids(batch_index=0)
            label_ids: List[int] = []

            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # PyTorch ignore index for special tokens
                elif word_idx != previous_word_idx:
                    tag = tags[word_idx] if word_idx < len(tags) else "O"
                    label_ids.append(LABEL2ID.get(tag, 0))
                else:
                    tag = tags[word_idx] if word_idx < len(tags) else "O"
                    # For subwords of the same word, use I- tag or -100
                    if tag.startswith("B-"):
                        tag = "I-" + tag[2:]
                    label_ids.append(LABEL2ID.get(tag, 0))
                previous_word_idx = word_idx

            item = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(label_ids, dtype=torch.long),
            }
            self.samples.append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
