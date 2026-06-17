# AmbiStory Plausibility Prediction

**SemEval 2026 Task 5 — Rating Plausibility of Word Senses in Ambiguous Sentences**

Team: Rishit Khadawala · Warren Lloyd Rodrigues · Varun Dayanand Shringare

---

## Approach

The task requires predicting how plausible a given word sense is within the context of a short story. We formulate this as a **sentence-pair regression task** and fine-tune **RoBERTa-base** on the AmbiStory dataset.

### Input Representation

Each sample is encoded as a pair of sequences:

**Sequence A (Story Context)**

```text
Target Word: <homonym>

Story Context:
<precontext>

Ambiguous Sentence:
<sentence>

Story Continuation:
<ending>
```

**Sequence B (Candidate Sense)**

```text
Candidate Meaning:
<judged_meaning>

Example Usage:
<example_sentence>
```

This structured format explicitly separates the target homonym, narrative context, candidate meaning, and example usage, allowing the model to better align the intended word sense with the story context.

---

## Model

### Backbone

* **Model:** `roberta-base`
* **Parameters:** 125M
* **Maximum sequence length:** 384 tokens

### Pooling Strategy

Instead of using the first token representation, the model applies **mean pooling** over all contextual token embeddings:

```text
Token Embeddings → Mean Pooling → Dropout → Regression Head
```

Mean pooling captures information from the entire story and generally performs better than relying solely on the first token representation for semantic similarity and ranking tasks.

### Regression Head

```text
Linear(768 → 1)
```

The model predicts a continuous plausibility score.

### Loss Function

Training combines two objectives:

#### 1. Smooth L1 Loss (Huber Loss)

Measures the difference between predicted and gold plausibility scores while being more robust to outliers than Mean Squared Error.

#### 2. Pairwise Ranking Loss

Encourages the model to preserve the relative ordering of plausibility scores between samples, directly supporting the Spearman correlation evaluation metric.

Total training loss:

```text
Loss = SmoothL1Loss + 0.2 × PairwiseRankingLoss
```

### Optimisation

* Optimiser: AdamW
* Learning rate scheduling:

  * Linear warm-up (10% of training steps)
  * Linear decay for the remaining steps
* Gradient clipping:

  * Maximum norm = 1.0

---

## Evaluation Metrics

| Metric     | Description                                                                                           |
| ---------- | ----------------------------------------------------------------------------------------------------- |
| Spearman ρ | Rank-order correlation with average human ratings                                                     |
| Acc±σ      | Proportion of predictions within one standard deviation of the average rating (minimum tolerance = 1) |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python -m src.train
```

Training will:

1. Load the AmbiStory training and development datasets.
2. Fine-tune RoBERTa-base on the plausibility prediction task.
3. Evaluate on the development set after each epoch.
4. Save the checkpoint with the highest development Spearman correlation.

Checkpoint location:

```text
model_checkpoint/roberta_ambistory.pt
```

---

## Prediction

```bash
python predict.py data/dev.json outputs/dev_predictions.jsonl
```

Output format:

```json
{"id": "0", "prediction": 4}
```

Predictions are rounded to the nearest integer and clipped to the valid score range [1, 5].

---

## Hyperparameters

| Parameter           | Value                                |
| ------------------- | ------------------------------------ |
| Pretrained model    | `roberta-base`                       |
| Max sequence length | 384                                  |
| Batch size          | 8                                    |
| Epochs              | 5                                    |
| Learning rate       | 2e-5                                 |
| Weight decay        | 0.01                                 |
| Warm-up ratio       | 10%                                  |
| Loss function       | SmoothL1Loss + Pairwise Ranking Loss |
| Pooling strategy    | Mean Pooling                         |
| Random seed         | 42                                   |

---

## Results

The final model achieved approximately **0.48 Spearman correlation**  and **0.72 Accuracy within StandardDeviation ** on the development set after introducing:

* Structured input formatting
* Mean pooling
* Smooth L1 regression loss
* Pairwise ranking loss

These modifications improved ranking performance compared to a standard RoBERTa regression baseline.
