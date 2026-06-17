
import torch
import torch.nn as nn
from transformers import RobertaModel


class RobertaPlausibilityRegressor(nn.Module):
    

    def __init__(
        self,
        pretrained_name: str = "roberta-base",
        dropout_prob: float  = 0.1,
        score_min: float     = 1.0,
        score_max: float     = 5.0,
    ) -> None:
        super().__init__()

        self.score_min   = score_min
        self.score_range = score_max - score_min  # 4.0 for the 1-5 scale

        # Pretrained transformer backbone
        self.roberta = RobertaModel.from_pretrained(pretrained_name)

        hidden_size = self.roberta.config.hidden_size  # 768 for roberta-base

        # Regression head: dropout → linear → scaled sigmoid
        self.dropout         = nn.Dropout(p=dropout_prob)
        self.regression_head = nn.Linear(hidden_size, 1)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        # The [CLS]-equivalent token is the first token (index 0) for RoBERTa
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1)

        pooled = (
            token_embeddings * mask
        ).sum(1) / mask.sum(1)

        pooled = self.dropout(pooled)

        logits = self.regression_head(pooled).squeeze(-1)

        return logits


def load_model_checkpoint(
    checkpoint_path: str,
    pretrained_name: str,
    device: torch.device,
) -> RobertaPlausibilityRegressor:
    
    model = RobertaPlausibilityRegressor(pretrained_name=pretrained_name)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
