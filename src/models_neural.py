"""Char-aware BiLSTM tagger with GloVe initialization, dynamic int8 quantization,
batching utilities, and post-hoc vocabulary pruning.
"""
from __future__ import annotations

import copy
import os
import time
import urllib.request
import zipfile
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

Sentence = List[Tuple[str, str]]


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

PAD = "<PAD>"
UNK = "<UNK>"
CHAR_PAD = "\0"
CHAR_UNK = "\1"


def build_word_vocab(train_sents: List[Sentence], min_freq: int = 2) -> Tuple[List[str], Dict[str, int], Counter]:
    """Returns (vocab_words, word2id, word_freq_lowercased)."""
    counter: Counter = Counter()
    for s in train_sents:
        for tok, _ in s:
            counter[tok.lower()] += 1
    vocab_words = [PAD, UNK] + [w for w, c in counter.items() if c >= min_freq]
    word2id = {w: i for i, w in enumerate(vocab_words)}
    return vocab_words, word2id, counter


def build_char_vocab(train_sents: List[Sentence]) -> Tuple[List[str], Dict[str, int]]:
    chars = set()
    for s in train_sents:
        for tok, _ in s:
            chars.update(tok)
    char_list = [CHAR_PAD, CHAR_UNK] + sorted(chars)
    char2id = {c: i for i, c in enumerate(char_list)}
    return char_list, char2id


# ---------------------------------------------------------------------------
# GloVe loader
# ---------------------------------------------------------------------------

def load_glove_embeddings(
    vocab_words: List[str],
    emb_dim: int = 100,
    cache_path: str = "embeddings/glove.6B.100d.txt",
) -> Tuple[np.ndarray, int]:
    """Build an embedding matrix aligned to vocab_words, initialized from GloVe.

    Strategy:
      1. If `cache_path` exists, parse it directly (fastest).
      2. Else try gensim's downloader (`glove-wiki-gigaword-100`) — different corpus
         but same dimensionality, well-cached.
      3. Else download `glove.6B.zip` from Stanford, extract `glove.6B.100d.txt`.

    Returns (matrix [V, emb_dim] float32, n_initialized).
    """
    rng = np.random.default_rng(0)
    matrix = rng.normal(0.0, 0.1, size=(len(vocab_words), emb_dim)).astype(np.float32)
    matrix[0] = 0.0  # PAD row stays zero

    glove_table: Dict[str, np.ndarray] = {}

    if os.path.exists(cache_path):
        print(f"Loading GloVe from cache: {cache_path}")
        glove_table = _parse_glove_txt(cache_path, emb_dim)
    else:
        try:
            print("Trying gensim.downloader (glove-wiki-gigaword-100)...")
            import gensim.downloader as gdl
            kv = gdl.load("glove-wiki-gigaword-100")
            for w in kv.key_to_index:
                glove_table[w] = kv[w].astype(np.float32)
            print(f"  Loaded {len(glove_table):,} glove vectors via gensim")
        except Exception as e:
            print(f"  gensim path failed: {e}")
            print("  Falling back to Stanford glove.6B.zip direct download...")
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            zip_path = os.path.join(os.path.dirname(cache_path) or ".", "glove.6B.zip")
            if not os.path.exists(zip_path):
                url = "https://nlp.stanford.edu/data/glove.6B.zip"
                print(f"  Downloading {url} (~822MB) ...")
                urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extract("glove.6B.100d.txt", os.path.dirname(cache_path) or ".")
            glove_table = _parse_glove_txt(cache_path, emb_dim)

    n_init = 0
    for i, w in enumerate(vocab_words):
        if w in (PAD, UNK):
            continue
        if w in glove_table and glove_table[w].shape[0] == emb_dim:
            matrix[i] = glove_table[w]
            n_init += 1
    print(f"GloVe coverage: {n_init} / {len(vocab_words) - 2} ({100 * n_init / max(1, len(vocab_words) - 2):.1f}%)")
    return matrix, n_init


def _parse_glove_txt(path: str, expected_dim: int) -> Dict[str, np.ndarray]:
    table: Dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != expected_dim + 1:
                continue
            table[parts[0]] = np.array(parts[1:], dtype=np.float32)
    return table


# ---------------------------------------------------------------------------
# Char-aware BiLSTM
# ---------------------------------------------------------------------------

class CharCNNEncoder(nn.Module):
    """Per-token char-CNN: embed chars → 1D conv → max-pool over chars."""

    def __init__(self, num_chars: int, char_emb_dim: int = 25, num_filters: int = 30,
                 kernel_size: int = 3, char_pad_id: int = 0):
        super().__init__()
        self.char_emb = nn.Embedding(num_chars, char_emb_dim, padding_idx=char_pad_id)
        self.conv = nn.Conv1d(char_emb_dim, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.out_dim = num_filters

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: (B, T, C)
        B, T, C = char_ids.shape
        flat = char_ids.reshape(B * T, C)
        emb = self.char_emb(flat)  # (B*T, C, e)
        emb = emb.transpose(1, 2)  # (B*T, e, C)
        h = F.relu(self.conv(emb))  # (B*T, F, C)
        pooled = h.max(dim=-1).values  # (B*T, F)
        return pooled.reshape(B, T, self.out_dim)


class CharAwareBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, num_chars: int, num_tags: int,
                 word_emb_dim: int = 100, char_emb_dim: int = 25,
                 char_filters: int = 30, char_kernel: int = 3,
                 hidden_dim: int = 200, dropout: float = 0.5,
                 word_pad_id: int = 0, char_pad_id: int = 0):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=word_pad_id)
        self.char_enc = CharCNNEncoder(num_chars, char_emb_dim, char_filters, char_kernel, char_pad_id)
        self.dropout = nn.Dropout(dropout)
        in_dim = word_emb_dim + self.char_enc.out_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, word_ids: torch.Tensor, char_ids: torch.Tensor) -> torch.Tensor:
        we = self.word_emb(word_ids)        # (B, T, We)
        ce = self.char_enc(char_ids)        # (B, T, Cf)
        x = torch.cat([we, ce], dim=-1)
        x = self.dropout(x)
        h, _ = self.lstm(x)
        return self.fc(h)


# ---------------------------------------------------------------------------
# Tensorization
# ---------------------------------------------------------------------------

def encode_words(tokens: List[str], word2id: Dict[str, int]) -> List[int]:
    unk = word2id[UNK]
    return [word2id.get(t.lower(), unk) for t in tokens]


def encode_chars(tokens: List[str], char2id: Dict[str, int], max_char_len: int) -> List[List[int]]:
    pad = char2id[CHAR_PAD]
    unk = char2id[CHAR_UNK]
    out = []
    for tok in tokens:
        ids = [char2id.get(c, unk) for c in tok[:max_char_len]]
        ids = ids + [pad] * (max_char_len - len(ids))
        out.append(ids)
    return out


class NERDataset(Dataset):
    def __init__(self, sents: List[Sentence], word2id, char2id, tag2id, max_char_len: int = 25):
        self.sents = sents
        self.word2id = word2id
        self.char2id = char2id
        self.tag2id = tag2id
        self.max_char_len = max_char_len

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx: int):
        s = self.sents[idx]
        toks = [t for t, _ in s]
        tags = [t for _, t in s]
        x_w = torch.tensor(encode_words(toks, self.word2id), dtype=torch.long)
        x_c = torch.tensor(encode_chars(toks, self.char2id, self.max_char_len), dtype=torch.long)
        y = torch.tensor([self.tag2id[t] for t in tags], dtype=torch.long)
        return x_w, x_c, y


def make_collate(word_pad_id: int, char_pad_id: int, max_char_len: int):
    def collate(batch):
        ws, cs, ys = zip(*batch)
        T = max(w.size(0) for w in ws)
        B = len(batch)
        w_pad = torch.full((B, T), word_pad_id, dtype=torch.long)
        c_pad = torch.full((B, T, max_char_len), char_pad_id, dtype=torch.long)
        y_pad = torch.full((B, T), -100, dtype=torch.long)
        for i, (w, c, y) in enumerate(batch):
            L = w.size(0)
            w_pad[i, :L] = w
            c_pad[i, :L, :] = c
            y_pad[i, :L] = y
        return w_pad, c_pad, y_pad
    return collate


# ---------------------------------------------------------------------------
# Training with early stopping
# ---------------------------------------------------------------------------

def train_with_early_stop(
    model: nn.Module,
    train_loader: DataLoader,
    valid_sents: List[Sentence],
    word2id: Dict[str, int],
    char2id: Dict[str, int],
    id2tag: List[str],
    tag2id: Dict[str, int],
    max_char_len: int,
    *,
    epochs: int = 15,
    patience: int = 3,
    lr: float = 1e-3,
    train_threads: Optional[int] = None,
):
    from seqeval.metrics import f1_score
    from seqeval.scheme import IOB2
    import os as _os

    if train_threads is None:
        train_threads = max(1, (_os.cpu_count() or 2) // 2)
    torch.set_num_threads(train_threads)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    num_tags = model.fc.out_features

    best_f1 = -1.0
    best_state = None
    bad_epochs = 0
    history = []

    valid_y_true = [[tag for _, tag in s] for s in valid_sents]
    valid_tokens = [[t for t, _ in s] for s in valid_sents]

    t_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for w, c, y in train_loader:
            opt.zero_grad()
            logits = model(w, c)
            loss = loss_fn(logits.reshape(-1, num_tags), y.reshape(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        # dev eval
        torch.set_num_threads(1)
        model.eval()
        with torch.no_grad():
            y_pred = []
            for toks in valid_tokens:
                w_ids = torch.tensor([encode_words(toks, word2id)], dtype=torch.long)
                c_ids = torch.tensor([encode_chars(toks, char2id, max_char_len)], dtype=torch.long)
                logits = model(w_ids, c_ids)[0]
                pred = logits.argmax(dim=-1).tolist()
                y_pred.append([id2tag[i] for i in pred])
        f1 = f1_score(valid_y_true, y_pred, mode="strict", scheme=IOB2)
        torch.set_num_threads(train_threads)

        avg_loss = total_loss / max(1, n_batches)
        history.append({"epoch": epoch + 1, "loss": avg_loss, "dev_f1": f1})
        print(f"  epoch {epoch + 1:>2}/{epochs}  loss={avg_loss:.4f}  dev_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {epoch + 1} (no dev-F1 improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - t_start
    return model, elapsed, history, best_f1


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def make_predict_fn(model, word2id, char2id, id2tag, max_char_len: int):
    @torch.no_grad()
    def predict(tokens: List[str]) -> List[str]:
        model.eval()
        w_ids = torch.tensor([encode_words(tokens, word2id)], dtype=torch.long)
        c_ids = torch.tensor([encode_chars(tokens, char2id, max_char_len)], dtype=torch.long)
        logits = model(w_ids, c_ids)[0]
        pred = logits.argmax(dim=-1).tolist()
        return [id2tag[i] for i in pred]
    return predict


def make_batched_predict_fn(model, word2id, char2id, id2tag, max_char_len: int,
                            word_pad_id: int, char_pad_id: int, batch_size: int):
    """Returns a callable(list_of_token_lists) -> list_of_tag_lists, batched."""
    @torch.no_grad()
    def predict(token_lists: List[List[str]]) -> List[List[str]]:
        model.eval()
        out: List[List[str]] = [None] * len(token_lists)  # type: ignore
        for start in range(0, len(token_lists), batch_size):
            batch = token_lists[start:start + batch_size]
            T = max(len(t) for t in batch)
            B = len(batch)
            w_pad = torch.full((B, T), word_pad_id, dtype=torch.long)
            c_pad = torch.full((B, T, max_char_len), char_pad_id, dtype=torch.long)
            lengths = []
            for i, toks in enumerate(batch):
                L = len(toks)
                lengths.append(L)
                w_pad[i, :L] = torch.tensor(encode_words(toks, word2id), dtype=torch.long)
                c_pad[i, :L, :] = torch.tensor(encode_chars(toks, char2id, max_char_len), dtype=torch.long)
            logits = model(w_pad, c_pad)
            preds = logits.argmax(dim=-1).tolist()
            for i, L in enumerate(lengths):
                out[start + i] = [id2tag[t] for t in preds[i][:L]]
        return out  # type: ignore
    return predict


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_dynamic(model: nn.Module) -> nn.Module:
    return torch.quantization.quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)


# ---------------------------------------------------------------------------
# Vocabulary pruning (post-hoc, no retrain)
# ---------------------------------------------------------------------------

def prune_embedding_table(
    state_dict: dict,
    vocab_words: List[str],
    word_freq: Counter,
    word2id: Dict[str, int],
    target_size: int,
    embedding_param: str = "word_emb.weight",
) -> Tuple[dict, int]:
    """Replace embedding rows of the (V - target_size) lowest-frequency words
    with the UNK row. Returns (new_state_dict, n_unique_vectors).

    PAD and UNK are always kept. `target_size` includes PAD + UNK.
    """
    sd = {k: v.clone() for k, v in state_dict.items()}
    emb = sd[embedding_param]  # (V, emb_dim)
    V = emb.size(0)
    if target_size >= V:
        return sd, V
    unk_id = word2id[UNK]
    pad_id = word2id[PAD]
    # Sort vocab indices by frequency descending; PAD and UNK pinned to top.
    candidates = []
    for i, w in enumerate(vocab_words):
        if i in (pad_id, unk_id):
            continue
        candidates.append((word_freq.get(w, 0), i))
    candidates.sort(key=lambda kv: -kv[0])
    keep = {pad_id, unk_id}
    keep.update(idx for _, idx in candidates[: max(0, target_size - 2)])
    unk_row = emb[unk_id].clone()
    for i in range(V):
        if i not in keep:
            emb[i] = unk_row
    return sd, len(keep)
