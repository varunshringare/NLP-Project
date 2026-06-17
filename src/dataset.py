from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class AmbiStoryDataset(Dataset):
    

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
        
        sample = self.samples[index]

        context_text = f"""
        Target Word: {sample['homonym']}

        Story Context:
        {sample['precontext']}

        Ambiguous Sentence:
        {sample['sentence']}

        Story Continuation:
        {sample['ending']}
        """

        meaning_text = f"""
        Candidate Meaning:
        {sample['judged_meaning']}

        Example Usage:
        {sample['example_sentence']}
        """

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
