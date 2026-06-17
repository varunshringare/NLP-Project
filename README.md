# AmbiStory Plausibility Prediction

**SemEval 2026 Task 5 — Rating Plausibility of Word Senses in Ambiguous Sentences**

Team: Rishit Khadawala · Warren Lloyd Rodrigues · Varun Dayanand Shringare · Naga Suresh Bonam

---

## Approach

The task requires predicting how plausible a given word sense is in the context of a short
five-sentence story. We frame this as a **sentence-pair regression** problem and fine-tune
`roberta-base` on the AmbiStory training data.

### Input representation

Each sample is encoded as a sentence pair:

```
Segment A: <precontext>  <ambiguous sentence>  <ending>
Segment B: <judged_meaning>  <example_sentence>
```

Segment A carries the full narrative context (including the optional ending, which often
disambiguates the homonym). Segment B carries the candidate word sense being rated,
augmented by the illustrative example sentence provided with each sample.

### Model

- **Backbone**: `roberta-base` (125 M parameters, 12 transformer layers)
- **Head**: Linear(768 → 1) followed by a scaled sigmoid, mapping the output to [1, 5]
- **Loss**: Mean Squared Error against the average human rating
- **Optimiser**: AdamW with linear warm-up (10 % of total steps) and linear decay

Using a scaled sigmoid (rather than an unconstrained linear head) keeps predictions
within the valid annotation range throughout training and avoids the need for output
clipping during loss computation.

### Evaluation metrics

| Metric | Description |
|---|---|
| Spearman ρ | Rank-order correlation with average human rating |
| Acc±σ | Proportion of predictions within one standard deviation of the average (floor σ = 1) |

---


## Setup

```bash
pip install -r requirements.txt
```


## Training

```bash
python -m src.train
```

This will:
1. Fine-tune `roberta-base` for 5 epochs on `data/train.json`
2. Evaluate on `data/dev.json` after each epoch
3. Save the best checkpoint (by dev Spearman ρ) to `model_checkpoint/roberta_ambistory.pt`

---

## Prediction

```bash
python predict.py data/dev.json outputs/dev_predictions.jsonl
```

Each line of the output file has the format:
```json
{"id": "0", "prediction": 4}
```

The continuous regression score is rounded to the nearest integer and clipped to [1, 5].

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Pretrained model | `roberta-base` |
| Max sequence length | 256 |
| Batch size | 16 |
| Epochs | 5 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warm-up ratio | 10 % |
| Random seed | 42 |
