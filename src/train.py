"""
Fine-tuning script for the RoBERTa plausibility regressor.

Training procedure
------------------
1. Load and tokenise train.json and dev.json.
2. Fine-tune ``roberta-base`` with MSE loss for NUM_EPOCHS epochs.
3. Apply a linear warm-up over WARMUP_RATIO of total steps, followed by
   linear decay (standard schedule for transformer fine-tuning).
4. After each epoch, evaluate on the development set and report both
   official metrics (Spearman ρ and Acc±σ).
5. Save the model checkpoint that achieves the best Spearman ρ on dev.

Run this module directly to start training:

    python -m src.train
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_NAME,
    DEV_PATH,
    LEARNING_RATE,
    MAX_SEQ_LEN,
    MODEL_DIR,
    NUM_EPOCHS,
    OUTPUT_DIR,
    PRETRAINED_MODEL_NAME,
    RANDOM_SEED,
    TRAIN_PATH,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.data_loader import extract_samples, load_json_dataset
from src.dataset import AmbiStoryDataset
from src.evaluation import evaluate
from src.model import RobertaPlausibilityRegressor


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Custom collate to handle the non-tensor sample_id field ──────────────────

def collate_fn(batch: list[dict]) -> dict:
    """
    Stack tensor fields and gather string sample IDs into a list.

    The default PyTorch collate function cannot handle a mix of tensors and
    plain strings, so we handle the two field types separately.
    """
    input_ids      = torch.stack([item["input_ids"]      for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels         = torch.stack([item["label"]          for item in batch])
    sample_ids     = [item["sample_id"] for item in batch]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "label":          labels,
        "sample_id":      sample_ids,
    }


# ── Epoch-level helpers ───────────────────────────────────────────────────────

def train_one_epoch(
    model:      RobertaPlausibilityRegressor,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    scheduler,
    criterion:  nn.Module,
    device:     torch.device,
) -> float:
    """
    Run one full pass over the training set.

    Returns
    -------
    float
        Mean MSE loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()

        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        loss        = criterion(predictions, labels)

        loss.backward()

        # Gradient clipping prevents exploding gradients in the transformer layers
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_on_dev(
    model:  RobertaPlausibilityRegressor,
    loader: DataLoader,
    device: torch.device,
    data:   dict,
) -> dict[str, float]:
    """
    Run inference on the development set and compute both official metrics.

    Parameters
    ----------
    model : RobertaPlausibilityRegressor
        Model in eval mode.
    loader : DataLoader
        DataLoader wrapping the dev dataset.
    device : torch.device
        Device for inference.
    data : dict
        Original dev JSON dict, used to look up per-sample standard deviations.

    Returns
    -------
    dict with keys ``"spearman"`` and ``"accuracy_within_stdev"``.
    """
    model.eval()
    all_predictions: list[float] = []
    all_targets:     list[float] = []
    all_stdevs:      list[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"]
            sample_ids     = batch["sample_id"]

            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = predictions.cpu().tolist()

            all_predictions.extend(predictions)
            all_targets.extend(labels.tolist())

            # Retrieve standard deviations from the original data dict
            for sid in sample_ids:
                stdev = data.get(sid, {}).get("stdev", 1.0)
                all_stdevs.append(float(stdev))

    return evaluate(all_predictions, all_targets, all_stdevs)


# ── Main training entry point ─────────────────────────────────────────────────

def train() -> None:
    """
    Run the full fine-tuning pipeline and save the best model checkpoint.
    """
    set_seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading datasets …")
    train_data = load_json_dataset(TRAIN_PATH)
    dev_data   = load_json_dataset(DEV_PATH)

    train_samples = extract_samples(train_data)
    dev_samples   = extract_samples(dev_data)

    print(f"  Train samples : {len(train_samples)}")
    print(f"  Dev   samples : {len(dev_samples)}")

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    print(f"Loading tokeniser: {PRETRAINED_MODEL_NAME} …")
    tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    train_dataset = AmbiStoryDataset(train_samples, tokenizer, MAX_SEQ_LEN)
    dev_dataset   = AmbiStoryDataset(dev_samples,   tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Initialising model from {PRETRAINED_MODEL_NAME} …")
    model = RobertaPlausibilityRegressor(pretrained_name=PRETRAINED_MODEL_NAME)
    model.to(device)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps  = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_spearman    = -1.0
    checkpoint_path  = MODEL_DIR / CHECKPOINT_NAME

    print(f"\nStarting training for {NUM_EPOCHS} epochs …\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        dev_metrics = evaluate_on_dev(model, dev_loader, device, dev_data)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"spearman={dev_metrics['spearman']:.4f}  "
            f"acc_within_stdev={dev_metrics['accuracy_within_stdev']:.4f}"
        )

        # Save checkpoint whenever dev Spearman improves
        if dev_metrics["spearman"] > best_spearman:
            best_spearman = dev_metrics["spearman"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ New best Spearman {best_spearman:.4f} — checkpoint saved.")

    print(f"\nTraining complete.  Best dev Spearman: {best_spearman:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    train()
