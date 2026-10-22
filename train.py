"""Fine-tune a Transformer model for Pathology Named Entity Recognition (NER).

Trains a token classification model (e.g. bert-base-uncased or layoutlm-base-uncased)
on the pathology_dataset annotations to extract test names, values, units,
reference ranges, and patient demographics.

Usage:
    python pathology_ai/train.py --epochs 1 --max-samples 500 --batch-size 8
    python pathology_ai/train.py --epochs 3 --max-samples 5000 --model-name bert-base-uncased
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathology_ai.configs.paths import PACKAGE_ROOT
from pathology_ai.datasets.loader import AnnotationLoader
from pathology_ai.datasets.ner_dataset import (
    ID2LABEL,
    LABEL2ID,
    LABEL_LIST,
    PathologyNERDataset,
)

DEFAULT_CHECKPOINT_DIR = PACKAGE_ROOT / "checkpoints" / "ner_model"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Transformer model for Pathology Report NER.")
    parser.add_argument(
        "--model-name",
        default="bert-base-uncased",
        help="HuggingFace model backbone (e.g. bert-base-uncased, distilbert-base-uncased).",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per step.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max-samples", type=int, default=500, help="Maximum training samples to load.")
    parser.add_argument("--eval-samples", type=int, default=100, help="Maximum evaluation samples to load.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory to save fine-tuned model checkpoint.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU instead of GPU.")
    return parser.parse_args()


def train_model(
    model_name: str = "bert-base-uncased",
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 5e-5,
    max_samples: int = 500,
    eval_samples: int = 100,
    output_dir: str = str(DEFAULT_CHECKPOINT_DIR),
    use_gpu: bool = True,
):
    """Fine-tune token classification model on pathology report dataset."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer & Model
    print(f"Loading backbone: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # Load Annotations
    loader = AnnotationLoader()
    print(f"Loading {max_samples} train records...")
    train_records = list(loader.iterate(split="train", limit=max_samples))
    print(f"Loading {eval_samples} val records...")
    val_records = list(loader.iterate(split="validation", limit=eval_samples))

    # Build PyTorch Datasets
    print("Formatting NER token datasets...")
    train_dataset = PathologyNERDataset(train_records, tokenizer, max_length=512)
    val_dataset = PathologyNERDataset(val_records, tokenizer, max_length=512)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\nStarting training ({epochs} epoch(s), {len(train_dataset)} samples)...")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader) if train_loader else 0.0

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_tokens = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                mask = labels != -100
                correct_tokens += (preds[mask] == labels[mask]).sum().item()
                total_tokens += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader) if val_loader else 0.0
        val_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | Val Token Acc = {val_accuracy:.4f}")

    elapsed = round(time.time() - t0, 2)
    print(f"\nTraining completed in {elapsed}s.")

    # Save Checkpoint
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving checkpoint to: {out_path.resolve()}")
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    # Save metadata
    meta = {
        "model_name": model_name,
        "epochs": epochs,
        "samples_trained": len(train_dataset),
        "val_samples": len(val_dataset),
        "val_accuracy": round(val_accuracy, 4),
        "elapsed_seconds": elapsed,
        "labels": LABEL_LIST,
    }
    (out_path / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Checkpoint and metadata saved successfully!")


def main():
    args = parse_args()
    train_model(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        eval_samples=args.eval_samples,
        output_dir=args.output_dir,
        use_gpu=not args.cpu,
    )


if __name__ == "__main__":
    main()
