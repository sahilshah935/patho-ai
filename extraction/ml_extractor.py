"""ML-based Named Entity Extraction using fine-tuned Transformer model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pathology_ai.configs.paths import PACKAGE_ROOT
from pathology_ai.extraction.schema import ExtractedPatient, ExtractedSection, ExtractedTest
from pathology_ai.datasets.ner_dataset import ID2LABEL

DEFAULT_CHECKPOINT_DIR = PACKAGE_ROOT / "checkpoints" / "ner_model"


class MLExtractor:
    """Named Entity Extractor powered by fine-tuned Transformer weights."""

    def __init__(self, model_dir: Optional[str | Path] = None, use_gpu: bool = True):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        ckpt_dir = Path(model_dir) if model_dir else DEFAULT_CHECKPOINT_DIR
        if not ckpt_dir.exists() or not (ckpt_dir / "config.json").exists():
            raise FileNotFoundError(
                f"No trained model checkpoint found at {ckpt_dir}. "
                "Run 'python pathology_ai/train.py' first to train the model."
            )

        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(ckpt_dir))
        self.model.to(self.device)
        self.model.eval()

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Run ML token classification on plain or OCR text to extract named entities.

        Args:
            text: Raw or OCR text from document page.

        Returns:
            Dict mapping entity types (TEST_NAME, VALUE, UNIT, REF_RANGE, PATIENT_NAME, PATIENT_ID)
            to list of extracted string spans.
        """
        if not text or not text.strip():
            return {}

        words = text.split()
        if not words:
            return {}

        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

        word_ids = inputs.word_ids(batch_index=0)

        # Collect entity spans by word
        entities: Dict[str, List[str]] = {
            "TEST_NAME": [],
            "VALUE": [],
            "UNIT": [],
            "REF_RANGE": [],
            "PATIENT_NAME": [],
            "PATIENT_ID": [],
        }

        current_label = None
        current_tokens = []

        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue

            pred_id = predictions[i]
            label = ID2LABEL.get(pred_id, "O")
            word = words[word_idx]

            if label.startswith("B-"):
                if current_label and current_tokens:
                    ent_key = current_label.replace("B-", "").replace("I-", "")
                    if ent_key in entities:
                        entities[ent_key].append(" ".join(current_tokens))
                current_label = label
                current_tokens = [word]
            elif label.startswith("I-") and current_label and label[2:] == current_label[2:]:
                current_tokens.append(word)
            else:
                if current_label and current_tokens:
                    ent_key = current_label.replace("B-", "").replace("I-", "")
                    if ent_key in entities:
                        entities[ent_key].append(" ".join(current_tokens))
                current_label = None
                current_tokens = []

            previous_word_idx = word_idx

        if current_label and current_tokens:
            ent_key = current_label.replace("B-", "").replace("I-", "")
            if ent_key in entities:
                entities[ent_key].append(" ".join(current_tokens))

        return entities
