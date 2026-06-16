"""
RoBERTa-based regression model for plausibility score prediction.

Architecture
------------
- Backbone : ``roberta-base`` (12 layers, 768-dim hidden, 125 M parameters)
- Head      : Linear(768 → 1) with a sigmoid activation scaled to [1, 5]

Using a scaled sigmoid rather than a raw linear output ensures predictions
always stay within the valid annotation range and makes training more stable
than relying solely on loss clamping.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel


class RobertaPlausibilityRegressor(nn.Module):
    """
    Fine-tunable RoBERTa model that predicts a plausibility score in [1, 5].

    Parameters
    ----------
    pretrained_name : str
        HuggingFace model identifier, e.g. ``"roberta-base"``.
    dropout_prob : float
        Dropout probability applied before the regression head.
    score_min : float
        Lower bound of the output range (default 1.0).
    score_max : float
        Upper bound of the output range (default 5.0).
    """

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
        """
        Run a forward pass and return predicted plausibility scores.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs, shape (batch_size, seq_len).
        attention_mask : torch.Tensor
            Attention mask, shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Predicted scores in [score_min, score_max], shape (batch_size,).
        """
        # The [CLS]-equivalent token is the first token (index 0) for RoBERTa
        outputs         = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]   # (batch, hidden)

        pooled  = self.dropout(cls_hidden_state)
        logits  = self.regression_head(pooled).squeeze(-1)       # (batch,)

        # Scale sigmoid output from (0, 1) to (score_min, score_max)
        scores = torch.sigmoid(logits) * self.score_range + self.score_min

        return scores


def load_model_checkpoint(
    checkpoint_path: str,
    pretrained_name: str,
    device: torch.device,
) -> RobertaPlausibilityRegressor:
    """
    Instantiate the model and load saved weights from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pt`` file saved by :func:`src.train.train`.
    pretrained_name : str
        HuggingFace identifier used when the model was originally created.
    device : torch.device
        Device on which to load the weights.

    Returns
    -------
    RobertaPlausibilityRegressor
        Model in evaluation mode with weights loaded.
    """
    model = RobertaPlausibilityRegressor(pretrained_name=pretrained_name)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
