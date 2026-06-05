# AmbiStory — SemEval 2026 Task 5: Plausibility of Word Senses

**NLP Project | University of Marburg | Summer 2026**  
Team: Rishit Khadawala, Warren Rodrigues, Varun Shringare, Naga Suresh Bonam

---

## Task

Predict the human-perceived plausibility of a word sense (1–5 scale) for ambiguous homonyms embedded in short 5-sentence stories. Evaluation uses Spearman correlation and Accuracy-within-StdDev.

## Approach

We implement a pipeline of four models with increasing complexity:

| Model | Type | Dev Spearman |
|---|---|---|
| Random | Baseline | ~0.00 |
| Global Mean | Baseline | ~0.00 |
| TF-IDF Nearest Neighbour | Retrieval | moderate |
| Ridge Regression | ML | moderate |
| Gradient Boosting | ML | moderate–good |
| **RoBERTa-base (fine-tuned)** | **Transformer** | **best** |

Our main model fine-tunes `roberta-base` as a regression model. The story context and judged meaning are provided as a two-segment input to the model. The [CLS] representation is passed through a two-layer MLP regression head, and the output is sigmoid-scaled to [1, 5].

## Setup

```bash
pip install -r requirements.txt
```

No external API keys are required. The HuggingFace `roberta-base` weights (~500 MB) are downloaded automatically on first run and cached locally.

## Running Predictions

```bash
python predict.py data/test.json predictions.jsonl
```

## Training

Run the notebooks in order:

1. `notebook1_eda_baselines.ipynb` — EDA + random/mean/TF-IDF baselines
2. `notebook2_feature_engineering.ipynb` — Ridge + Gradient Boosting
3. `notebook3_transformer_model.ipynb` — RoBERTa fine-tuning (produces `best_roberta_model.pt`)
4. `notebook4_final_evaluation.ipynb` — Final comparison and predict.py testing

## Repository Layout

```
├── predict.py               ← Submission entry point (required)
├── requirements.txt         ← Dependencies (required)
├── best_roberta_model.pt    ← Trained model weights
├── notebook1_eda_baselines.ipynb
├── notebook2_feature_engineering.ipynb
├── notebook3_transformer_model.ipynb
├── notebook4_final_evaluation.ipynb
├── train.json
├── dev.json
└── README.md
```
