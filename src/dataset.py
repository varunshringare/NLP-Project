"""
PyTorch Dataset for the AmbiStory plausibility prediction task.

Each sample is encoded as a sentence-pair input understood by RoBERTa:

    <s> precontext  sentence  ending </s></s> judged_meaning </s>

The first segment carries the narrative context; the second segment carries
the candidate word sense being rated.  The tokeniser handles truncation and
padding automatically.

The regression target is the ``average`` human rating (a float in [1, 5]).
For inference samples that lack an ``average`` field, a sentinel value of
-1.0 is returned so that the training loop can detect the absence of labels.
"""

from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class AmbiStoryDataset(Dataset):
    """
    Tokenised dataset for fine-tuning RoBERTa on the AmbiStory task.

    Parameters
    ----------
    samples : list[dict]
        List of sample dicts as produced by
        :func:`src.data_loader.extract_samples`.
    tokenizer : RobertaTokenizer
        Tokeniser instance (already loaded from the pretrained checkpoint).
    max_seq_len : int
        Maximum total token length (including special tokens).
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: RobertaTokenizer,
        max_seq_len: int,
    ) -> None:
        self.samples     = samples
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_context_text(sample: dict[str, Any]) -> str:
        """
        Concatenate the three narrative parts into a single context string.

        The ending field is optional (may be an empty string or absent); when
        present it often implies a specific word sense, so it is always
        included.
        """
        precontext = sample.get("precontext", "").strip()
        sentence   = sample.get("sentence",   "").strip()
        ending     = sample.get("ending",     "").strip()

        parts = [precontext, sentence]
        if ending:
            parts.append(ending)

        return (
            f"Context: {precontext} "
            f"Sentence: {sentence} "
            f"Continuation: {ending}"
        )   

    @staticmethod
    def _build_meaning_text(sample):
        homonym = sample.get("homonym", "").strip()
        judged_meaning = sample.get("judged_meaning", "").strip()
        example_sentence = sample.get("example_sentence", "").strip()

        return (
            f"Target word: {homonym}. "
            f"Meaning: {judged_meaning}. "
            f"Example: {example_sentence}"
        )

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Tokenise a single sample and return a dict of tensors.

        Returns
        -------
        dict with keys:
            ``input_ids``      – token IDs, shape (max_seq_len,)
            ``attention_mask`` – 1 for real tokens, 0 for padding, shape (max_seq_len,)
            ``label``          – scalar float tensor; -1.0 when no label is available
            ``sample_id``      – string sample ID (not a tensor; used for output)
        """
        sample = self.samples[index]

        context_text = self._build_context_text(sample)
        meaning_text = self._build_meaning_text(sample)

        encoding = self.tokenizer(
            context_text,
            meaning_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Squeeze batch dimension added by return_tensors="pt"
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Average rating is the regression target; -1.0 signals missing label
        label = float(sample.get("average", -1.0))

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "label":          torch.tensor(label, dtype=torch.float),
            "sample_id":      str(sample.get("id", index)),
        }
