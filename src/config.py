"""
Central configuration for the AmbiStory plausibility prediction pipeline.

All hyperparameters, file paths, and model settings are defined here so that
the rest of the codebase can import constants rather than scattering magic
numbers across files.
"""

from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR  = BASE_DIR / "model_checkpoint"

TRAIN_PATH = DATA_DIR / "train.json"
DEV_PATH   = DATA_DIR / "dev.json"

# ── Model ─────────────────────────────────────────────────────────────────────

# Pretrained checkpoint used for fine-tuning.
# roberta-base has 125 M parameters and works well for sentence-pair regression.
PRETRAINED_MODEL_NAME = "roberta-base"

# Maximum number of tokens fed to the transformer (including special tokens).
# RoBERTa supports up to 512; 256 comfortably fits the AmbiStory inputs.
MAX_SEQ_LEN = 256

# ── Training ──────────────────────────────────────────────────────────────────

BATCH_SIZE       = 16
NUM_EPOCHS       = 5
LEARNING_RATE    = 2e-5
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.1   # fraction of total steps used for linear warm-up
RANDOM_SEED      = 42

# Score range expected by the task (inclusive on both ends).
SCORE_MIN = 1
SCORE_MAX = 5

# ── Evaluation ────────────────────────────────────────────────────────────────

# Saved checkpoint name written after training.
CHECKPOINT_NAME = "roberta_ambistory.pt"
