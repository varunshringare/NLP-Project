"""
predict.py — AmbiStory SemEval 2026 Task 5 submission entry point.

Usage:
    python predict.py <input_json> <output_jsonl>

The script loads the fine-tuned RoBERTa regression model from
'best_roberta_model.pt' and runs inference on the provided test samples.
Falls back to the global mean predictor if the model file is not found.

No external API keys required — the model checkpoint is loaded locally.
"""

import sys
import json
import os
import math
import numpy as np

# ── Third-party imports (must be listed in requirements.txt) ──────────────
import torch
from torch import nn

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME     = 'roberta-base'
MODEL_WEIGHTS  = os.path.join(os.path.dirname(__file__), 'best_roberta_model.pt')
MAX_LEN        = 256
BATCH_SIZE     = 32
GLOBAL_MEAN    = 3.14  # fallback: global mean of training scores


# ══════════════════════════════════════════════════════════════════════════
# Model definition — must match the architecture in notebook3
# ══════════════════════════════════════════════════════════════════════════
class PlausibilityRegressor(nn.Module):
    """
    RoBERTa-base with a two-layer regression head.
    Input: tokenised (story_context, judged_meaning) pair.
    Output: plausibility score in [1, 5].
    """

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        from transformers import RobertaModel
        self.roberta   = RobertaModel.from_pretrained(model_name)
        hidden_size    = self.roberta.config.hidden_size  # 768
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs  = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        cls_repr = self.dropout(cls_repr)
        score    = self.regressor(cls_repr).squeeze(-1)
        # Scale sigmoid output to [1, 5]
        score    = 1.0 + 4.0 * torch.sigmoid(score)
        return score


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def build_story_text(sample: dict) -> str:
    """Concatenate narrative parts into a single story string."""
    parts = [
        sample.get('precontext', ''),
        sample.get('sentence', ''),
        sample.get('ending', '') or '',
    ]
    return ' '.join(p for p in parts if p).strip()


def predict_with_model(model, tokenizer, samples, device):
    """
    Run batch inference and return a dict {sample_id: float_score}.
    """
    model.eval()
    ids_list      = list(samples.keys())
    story_texts   = [build_story_text(samples[i])          for i in ids_list]
    meaning_texts = [samples[i].get('judged_meaning', '')  for i in ids_list]

    all_preds = []
    for start in range(0, len(ids_list), BATCH_SIZE):
        batch_stories  = story_texts[start:start + BATCH_SIZE]
        batch_meanings = meaning_texts[start:start + BATCH_SIZE]

        encoding = tokenizer(
            batch_stories,
            batch_meanings,
            max_length=MAX_LEN,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        input_ids   = encoding['input_ids'].to(device)
        attn_mask   = encoding['attention_mask'].to(device)

        with torch.no_grad():
            preds = model(input_ids, attn_mask).cpu().numpy()

        all_preds.extend(preds.tolist())

    return {sid: score for sid, score in zip(ids_list, all_preds)}


def predict_fallback(samples):
    """Return the global mean for every sample (safety fallback)."""
    print('[predict.py] WARNING: model weights not found — using global mean fallback.')
    return {sid: GLOBAL_MEAN for sid in samples}


def score_to_int(score: float) -> int:
    """Round a float score in [1, 5] to the nearest integer, clamped."""
    return max(1, min(5, round(float(score))))


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) != 3:
        print('Usage: python predict.py <input_json> <output_jsonl>', file=sys.stderr)
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    # ── Read input ─────────────────────────────────────────────────────
    with open(input_path, encoding='utf-8') as f:
        samples = json.load(f)
    print(f'[predict.py] Loaded {len(samples)} samples from {input_path}')

    # ── Load model if available ────────────────────────────────────────
    if os.path.exists(MODEL_WEIGHTS):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[predict.py] Loading model on {device} ...')

        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
        model     = PlausibilityRegressor(MODEL_NAME)
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        model.to(device)

        float_preds = predict_with_model(model, tokenizer, samples, device)
    else:
        float_preds = predict_fallback(samples)

    # ── Write output ──────────────────────────────────────────────────
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for sample_id in samples:
            prediction = score_to_int(float_preds[sample_id])
            f_out.write(json.dumps({'id': str(sample_id), 'prediction': prediction}) + '\n')

    print(f'[predict.py] Predictions written to {output_path}')


if __name__ == '__main__':
    main()
