"""HMM (NLTK) and CRF (sklearn-crfsuite) sequence labelers.

CRF features are richer than the fast version: ±2 context window, suffix/prefix 2-4,
word shape, plus standard casing/digit flags.
"""
from __future__ import annotations

import pickle
import time
from typing import List, Tuple

Sentence = List[Tuple[str, str]]


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

def lidstone_estimator(fd, bins):
    """Top-level (pickleable) Lidstone estimator factory."""
    from nltk.probability import LidstoneProbDist
    return LidstoneProbDist(fd, 0.1, bins)


def train_hmm(train_sents: List[Sentence]):
    from nltk.tag import hmm
    trainer = hmm.HiddenMarkovModelTrainer()
    t0 = time.perf_counter()
    model = trainer.train_supervised(train_sents, estimator=lidstone_estimator)
    elapsed = time.perf_counter() - t0
    return model, elapsed


def save_hmm(model, path: str) -> None:
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    except (pickle.PicklingError, AttributeError) as e:
        print(f"  HMM full pickle failed: {e}; falling back to dill")
        import dill
        with open(path, "wb") as f:
            dill.dump(model, f)


def make_hmm_predict(model):
    def predict(tokens: List[str]) -> List[str]:
        try:
            return [tag for _, tag in model.tag(tokens)]
        except Exception:
            # NLTK HMM can crash on pathological inputs; fall back to all-O.
            return ["O"] * len(tokens)
    return predict


# ---------------------------------------------------------------------------
# CRF
# ---------------------------------------------------------------------------

def _word_shape(w: str) -> str:
    """Map characters to a coarse shape: X for upper, x for lower, d for digit, - for other."""
    out = []
    last = ""
    for c in w:
        if c.isupper():
            ch = "X"
        elif c.islower():
            ch = "x"
        elif c.isdigit():
            ch = "d"
        else:
            ch = "-"
        if ch != last:
            out.append(ch)
            last = ch
    return "".join(out)


def word2features(tokens: List[str], i: int) -> dict:
    w = tokens[i]
    feats = {
        "bias": 1.0,
        "word.lower()": w.lower(),
        "word.shape()": _word_shape(w),
        "word[-2:]": w[-2:],
        "word[-3:]": w[-3:],
        "word[-4:]": w[-4:] if len(w) >= 4 else w,
        "word[:2]": w[:2],
        "word[:3]": w[:3] if len(w) >= 3 else w,
        "word[:4]": w[:4] if len(w) >= 4 else w,
        "word.isupper()": w.isupper(),
        "word.istitle()": w.istitle(),
        "word.islower()": w.islower(),
        "word.isdigit()": w.isdigit(),
        "word.has_digit()": any(c.isdigit() for c in w),
        "word.has_hyphen()": "-" in w,
        "word.len": len(w),
    }
    # ±1 context
    for offset in (-1, 1):
        j = i + offset
        if 0 <= j < len(tokens):
            ctx = tokens[j]
            tag = f"{offset:+d}"
            feats[f"{tag}:word.lower()"] = ctx.lower()
            feats[f"{tag}:word.istitle()"] = ctx.istitle()
            feats[f"{tag}:word.isupper()"] = ctx.isupper()
            feats[f"{tag}:word.shape()"] = _word_shape(ctx)
        else:
            feats["BOS" if offset < 0 else "EOS"] = True
    # ±2 context (only lower + shape — keep feature explosion in check)
    for offset in (-2, 2):
        j = i + offset
        if 0 <= j < len(tokens):
            ctx = tokens[j]
            tag = f"{offset:+d}"
            feats[f"{tag}:word.lower()"] = ctx.lower()
            feats[f"{tag}:word.shape()"] = _word_shape(ctx)
    return feats


def sent2features(tokens: List[str]) -> List[dict]:
    return [word2features(tokens, i) for i in range(len(tokens))]


def train_crf(train_sents: List[Sentence], max_iter: int = 100, c1: float = 0.1, c2: float = 0.1):
    from sklearn_crfsuite import CRF
    X = [sent2features([t for t, _ in s]) for s in train_sents]
    y = [[tag for _, tag in s] for s in train_sents]
    crf = CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=max_iter,
        all_possible_transitions=True,
    )
    t0 = time.perf_counter()
    crf.fit(X, y)
    elapsed = time.perf_counter() - t0
    return crf, elapsed


def save_crf(crf, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(crf, f)


def make_crf_predict(crf):
    def predict(tokens: List[str]) -> List[str]:
        return crf.predict_single(sent2features(tokens))
    return predict
