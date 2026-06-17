

import json
import sys
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

# Ensure the project root is on the path when called directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_NAME,
    MAX_SEQ_LEN,
    MODEL_DIR,
    PRETRAINED_MODEL_NAME,
    SCORE_MAX,
    SCORE_MIN,
)
from src.data_loader import extract_samples
from src.dataset import AmbiStoryDataset
from src.model import RobertaPlausibilityRegressor, load_model_checkpoint
from src.train import collate_fn


# ── Fallback prediction ───────────────────────────────────────────────────────

FALLBACK_SCORE = 3  # integer mid-point; used only when no checkpoint exists


# ── Helpers ───────────────────────────────────────────────────────────────────

def round_and_clip(score: float, lo: int = SCORE_MIN, hi: int = SCORE_MAX) -> int:
    return int(max(lo, min(hi, round(score))))


def run_model_inference(
    data:   dict,
    device: torch.device,
) -> dict[str, int]:
    
    checkpoint_path = MODEL_DIR / CHECKPOINT_NAME

    if not checkpoint_path.exists():
        warnings.warn(
            f"Checkpoint not found at {checkpoint_path}.  "
            f"Using fallback score {FALLBACK_SCORE} for all samples."
        )
        return {str(k): FALLBACK_SCORE for k in data.keys()}

    # Load tokeniser and model
    tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    model     = load_model_checkpoint(str(checkpoint_path), PRETRAINED_MODEL_NAME, device)

    samples = extract_samples(data)
    dataset = AmbiStoryDataset(samples, tokenizer, MAX_SEQ_LEN)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,   # preserve order for deterministic output
        collate_fn=collate_fn,
    )

    predictions: dict[str, int] = {}

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sample_ids     = batch["sample_id"]

            scores = model(input_ids=input_ids, attention_mask=attention_mask)

            for sid, score in zip(sample_ids, scores.cpu().tolist()):
                predictions[str(sid)] = round_and_clip(score)

    return predictions


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python predict.py <input_json> <output_jsonl>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    # ── Load input ────────────────────────────────────────────────────────────
    with open(input_path, encoding="utf-8") as fh:
        data = json.load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Inference ─────────────────────────────────────────────────────────────
    predictions = run_model_inference(data, device)

    # ── Write output ──────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        for sample_id in data.keys():
            prediction = predictions.get(str(sample_id), FALLBACK_SCORE)
            line = json.dumps({"id": str(sample_id), "prediction": prediction})
            fh.write(line + "\n")

    print(f"Predictions written to: {output_path}  ({len(data)} samples)")


if __name__ == "__main__":
    main()
