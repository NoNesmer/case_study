"""DistilBERT-NER inference wrapper, off-the-shelf, fine-tuned on CoNLL-2003.

Single-pass per sentence (or batched) with proper subword-to-token alignment.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

import torch

# Pinned model id. We use elastic's CoNLL-2003-fine-tuned DistilBERT.
# Revision is intentionally not hardcoded as a SHA — we try `main` and rely on the
# `transformers` library's local cache for reproducibility within a single environment.
DEFAULT_MODEL_ID = "elastic/distilbert-base-cased-finetuned-conll03-english"
DEFAULT_REVISION = "main"


class DistilBertNER:
    """Wraps an HF token-classification model for word-aligned NER inference.

    Convention: tags are returned in IOB2 form (`O`, `B-PER`, `I-PER`, ...).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID,
                 revision: Optional[str] = DEFAULT_REVISION,
                 num_threads: int = 1):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        torch.set_num_threads(num_threads)

        self.model_id = model_id
        self.revision = revision
        kwargs = {}
        if revision is not None:
            kwargs["revision"] = revision

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        self.model = AutoModelForTokenClassification.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.id2label = self.model.config.id2label
        # Sanity: the elastic model uses standard IOB2 labels.
        assert any(v == "O" for v in self.id2label.values()), (
            f"Unexpected label set in {model_id}: {self.id2label}"
        )

    @torch.no_grad()
    def predict(self, tokens: List[str]) -> List[str]:
        """Word-aligned IOB2 prediction. len(out) == len(tokens)."""
        if len(tokens) == 0:
            return []
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        word_ids = enc.word_ids(batch_index=0)
        logits = self.model(**enc).logits[0]  # (T_sub, num_labels)
        preds = logits.argmax(dim=-1).tolist()

        out = ["O"] * len(tokens)
        seen = [False] * len(tokens)
        for sub_idx, w_idx in enumerate(word_ids):
            if w_idx is None:
                continue
            if 0 <= w_idx < len(tokens) and not seen[w_idx]:
                out[w_idx] = self.id2label[preds[sub_idx]]
                seen[w_idx] = True
        return out

    @torch.no_grad()
    def predict_batch(self, token_lists: List[List[str]], batch_size: int = 8) -> List[List[str]]:
        """Batched word-aligned prediction with padding."""
        out: List[List[str]] = [None] * len(token_lists)  # type: ignore
        for start in range(0, len(token_lists), batch_size):
            batch_tokens = token_lists[start:start + batch_size]
            enc = self.tokenizer(
                batch_tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            logits = self.model(**enc).logits  # (B, T_sub, L)
            preds = logits.argmax(dim=-1).tolist()
            for i, toks in enumerate(batch_tokens):
                w_ids = enc.word_ids(batch_index=i)
                tags = ["O"] * len(toks)
                seen = [False] * len(toks)
                for sub_idx, w_idx in enumerate(w_ids):
                    if w_idx is None:
                        continue
                    if 0 <= w_idx < len(toks) and not seen[w_idx]:
                        tags[w_idx] = self.id2label[preds[i][sub_idx]]
                        seen[w_idx] = True
                out[start + i] = tags
        return out  # type: ignore


def make_distilbert_predict(model_id: str = DEFAULT_MODEL_ID,
                            revision: Optional[str] = DEFAULT_REVISION,
                            num_threads: int = 1):
    """Convenience: build the wrapper and return (wrapper, predict_fn, load_seconds)."""
    t0 = time.perf_counter()
    wrap = DistilBertNER(model_id=model_id, revision=revision, num_threads=num_threads)
    load_s = time.perf_counter() - t0
    return wrap, wrap.predict, load_s
