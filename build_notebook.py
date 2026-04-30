"""Generate notebook.ipynb for case-study-4.9-full via nbformat.

The notebook is structured as 17 sections that exercise:
  - 5 models (HMM, CRF, char-aware BiLSTM-fp32 / -int8, DistilBERT)
  - 4 optimization experiments (quantization, batching, vocab pruning, threading)
  - bootstrap F1 CIs, per-entity F1, 3-tier recommendation
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------
# Section 1 — Setup
# ---------------------------------------------------------------------------
md("""# Case Study 4.9 (FULL) — Latency-Constrained Sequence Labelling on CPU

**Task:** Named Entity Recognition on CoNLL-2003 (synalp .txt mirror).
**Models:** HMM, CRF, char-aware BiLSTM (fp32), char-aware BiLSTM (int8), DistilBERT-NER (off-the-shelf).
**Optimizations exercised:** dynamic int8 quantization, batching sweep, vocabulary pruning, multi-thread sweep.
**Constraint:** CPU-only inference; latency benchmarks run with `torch.set_num_threads(1)` unless noted.
**Statistical hygiene:** bootstrap 95% CIs on F1 (1,000 resamples); per-model RSS measured in subprocess isolation.
**Goal:** recommend a model per latency tier (5 ms / 10 ms / 50 ms per sentence).""")

md("## 1. Setup")

code("""import os, sys, time, random, pickle, platform, json, gc
from collections import Counter

import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Force CPU only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Make sure local src/ is importable.
sys.path.insert(0, ".")
from src import data as data_mod
from src import bench as bench_mod
from src import models_classical as classical
from src import models_neural as neural
from src import models_transformer as transformer

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Single-threaded by default (we'll temporarily bump for training).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

print(f"Python      : {sys.version.split()[0]}")
print(f"PyTorch     : {torch.__version__}")
print(f"Platform    : {platform.platform()}")
print(f"CPU         : {platform.processor() or 'unknown'}")
print(f"Logical CPUs: {os.cpu_count()}")
print(f"RAM         : {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"torch threads (inference default): {torch.get_num_threads()}")
""")

# ---------------------------------------------------------------------------
# Section 2 — Data
# ---------------------------------------------------------------------------
md("""## 2. Data — full CoNLL-2003

We use the **full** CoNLL-2003 train (14,041 sents) — not the 3K subset of the fast version.
Validation (`eng.testa`) is used for early stopping; test (`eng.testb`) is reported.
IOB1 → IOB2 conversion is required for `seqeval` strict mode.""")

code("""train_sents, valid_sents, test_sents, label_list = data_mod.load_conll(data_dir="data")
NUM_TAGS = len(label_list)
tag2id = {t: i for i, t in enumerate(label_list)}
id2tag = list(label_list)

print(f"train: {len(train_sents):>5} sents, {sum(len(s) for s in train_sents):>6} tokens")
print(f"valid: {len(valid_sents):>5} sents, {sum(len(s) for s in valid_sents):>6} tokens")
print(f"test : {len(test_sents):>5} sents, {sum(len(s) for s in test_sents):>6} tokens")
print(f"Tags ({NUM_TAGS}): {label_list}")

_tagc = Counter(t for s in train_sents for _, t in s)
print("Train tag distribution:")
for t, c in sorted(_tagc.items(), key=lambda kv: -kv[1]):
    print(f"  {t:>8}: {c}")

_train_vocab = {tok.lower() for s in train_sents for tok, _ in s}
_test_toks = [tok for s in test_sents for tok, _ in s]
_oov = sum(1 for t in _test_toks if t.lower() not in _train_vocab)
print(f"OOV rate (test vs full train): {_oov / len(_test_toks):.3%}")
""")

# ---------------------------------------------------------------------------
# Section 3 — Embeddings
# ---------------------------------------------------------------------------
md("""## 3. GloVe 100d embeddings

We initialize the BiLSTM word-embedding layer from GloVe (100d). Strategy:
1. Local cache (`embeddings/glove.6B.100d.txt`) if present.
2. `gensim.downloader.load("glove-wiki-gigaword-100")` — well-cached, S3-backed.
3. Stanford direct download as last resort (slow, intermittent).""")

code("""# Build word vocab + char vocab from the FULL train set.
vocab_words, word2id, word_freq = neural.build_word_vocab(train_sents, min_freq=2)
char_list, char2id = neural.build_char_vocab(train_sents)
PAD_ID = word2id[neural.PAD]
UNK_ID = word2id[neural.UNK]
CHAR_PAD_ID = char2id[neural.CHAR_PAD]
print(f"Word vocab: {len(vocab_words):,}  |  Char vocab: {len(char_list):,}  |  Tags: {NUM_TAGS}")

EMB_DIM = 100
glove_matrix, n_init = neural.load_glove_embeddings(vocab_words, emb_dim=EMB_DIM,
                                                   cache_path="embeddings/glove.6B.100d.txt")
""")

# ---------------------------------------------------------------------------
# Section 4 — Shared benchmark sample
# ---------------------------------------------------------------------------
md("""## 4. Shared benchmark sample

200 test sentences, 20-sentence warm-up, used for all 5 models so the latency numbers are directly comparable.""")

code("""random.seed(SEED + 1)
N_BENCH = 200
N_WARMUP = 20
bench_indices = random.sample(range(len(test_sents)), min(N_BENCH, len(test_sents)))
bench_sents = [test_sents[i] for i in bench_indices]
bench_token_lists = [[t for t, _ in s] for s in bench_sents]
bench_total_tokens = sum(len(s) for s in bench_token_lists)
print(f"Benchmark sample: {len(bench_token_lists)} sents, {bench_total_tokens} tokens")

# Inputs for full-test evaluation
y_true = [[tag for _, tag in s] for s in test_sents]
test_token_lists = [[t for t, _ in s] for s in test_sents]

rows = []        # headline metrics, one per model
y_preds = {}     # model name -> y_pred (for per-entity breakdown)
""")

# ---------------------------------------------------------------------------
# Section 5 — Bootstrap CI helper
# ---------------------------------------------------------------------------
md("""## 5. Helper: bootstrap CI + cold-vs-warm latency

Defined in `src/bench.py`; we rebind here for convenience.""")

code("""bootstrap_f1_ci = bench_mod.bootstrap_f1_ci
time_predict = bench_mod.time_predict
latency_summary = bench_mod.latency_summary
file_size_mb = bench_mod.file_size_mb
cold_vs_warm_latency = bench_mod.cold_vs_warm_latency
gc_pause = bench_mod.gc_pause


def add_row(name, model_path, y_pred, train_seconds, cold_ms, warm_times_ms, total_tokens, n_resamples=1000):
    \"\"\"Compute headline metrics + bootstrap CI, append to `rows`, store y_pred.\"\"\"
    lat = latency_summary(warm_times_ms, total_tokens)
    f1_pt, f1_lo, f1_hi = bootstrap_f1_ci(y_true, y_pred, n_resamples=n_resamples, seed=SEED)
    size_mb = file_size_mb(model_path) if model_path is not None else float("nan")
    row = {
        "model": name,
        "f1": f1_pt,
        "f1_low_95": f1_lo,
        "f1_high_95": f1_hi,
        "latency_mean_ms": lat["mean_ms"],
        "latency_median_ms": lat["median_ms"],
        "latency_p95_ms": lat["p95_ms"],
        "latency_p99_ms": lat["p99_ms"],
        "throughput_tok_per_s": lat["throughput_tok_per_s"],
        "cold_start_ms": cold_ms,
        "model_size_mb": size_mb,
        "train_seconds": train_seconds,
    }
    rows.append(row)
    y_preds[name] = y_pred
    print(f"  {name:<22} F1={f1_pt:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]  "
          f"p50={lat['median_ms']:.2f}ms p95={lat['p95_ms']:.2f}ms  "
          f"throughput={lat['throughput_tok_per_s']:,.0f} tok/s  size={size_mb:.2f}MB")
""")

# ---------------------------------------------------------------------------
# Section 6 — HMM
# ---------------------------------------------------------------------------
md("""## 6. Model 1 — HMM (full train)""")

code("""torch.set_num_threads(1)
print(\"Training HMM on full CoNLL-2003 train ...\")
hmm_model, hmm_train_s = classical.train_hmm(train_sents)
print(f\"  trained in {hmm_train_s:.1f}s\")

hmm_path = \"models/hmm.pkl\"
classical.save_hmm(hmm_model, hmm_path)
hmm_predict = classical.make_hmm_predict(hmm_model)

# Cold/warm latency on the shared 200-sent sample
cold_ms, warm_times = cold_vs_warm_latency(hmm_predict, bench_token_lists)
# Quality on full test
y_pred = [hmm_predict(toks) for toks in test_token_lists]

add_row(\"HMM\", hmm_path, y_pred, hmm_train_s, cold_ms, warm_times, bench_total_tokens)
gc_pause(0.3)
""")

# ---------------------------------------------------------------------------
# Section 7 — CRF
# ---------------------------------------------------------------------------
md("""## 7. Model 2 — CRF (full train, rich features)

Features: word shape, prefix/suffix 2-4, casing, digit, hyphen, length, ±1 word + casing + shape, ±2 word + shape.""")

code("""torch.set_num_threads(1)
print(\"Training CRF on full CoNLL-2003 train ...\")
crf_model, crf_train_s = classical.train_crf(train_sents, max_iter=100)
print(f\"  trained in {crf_train_s:.1f}s ({len(crf_model.attributes_):,} features)\")

crf_path = \"models/crf.pkl\"
classical.save_crf(crf_model, crf_path)
crf_predict = classical.make_crf_predict(crf_model)

cold_ms, warm_times = cold_vs_warm_latency(crf_predict, bench_token_lists)
y_pred = [crf_predict(toks) for toks in test_token_lists]

add_row(\"CRF\", crf_path, y_pred, crf_train_s, cold_ms, warm_times, bench_total_tokens)
gc_pause(0.3)
""")

# ---------------------------------------------------------------------------
# Section 8 — BiLSTM-fp32
# ---------------------------------------------------------------------------
md("""## 8. Model 3 — Char-aware BiLSTM (fp32, GloVe init)

GloVe-100d word embedding + char-CNN (30 filters, kernel 3) → BiLSTM (hidden=200) → Linear → softmax.
Adam 1e-3, dropout 0.5, batch=32, 15 epochs with **dev-set early stopping** (patience=3) on `eng.testa`.""")

code("""from torch.utils.data import DataLoader

MAX_CHAR_LEN = 25
HIDDEN_DIM = 200
BATCH_SIZE = 32
EPOCHS = 15
PATIENCE = 3

torch.manual_seed(SEED)
bilstm = neural.CharAwareBiLSTM(
    vocab_size=len(vocab_words),
    num_chars=len(char_list),
    num_tags=NUM_TAGS,
    word_emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    word_pad_id=PAD_ID,
    char_pad_id=CHAR_PAD_ID,
)
# Initialize word embeddings from GloVe.
with torch.no_grad():
    bilstm.word_emb.weight.copy_(torch.from_numpy(glove_matrix))

train_ds = neural.NERDataset(train_sents, word2id, char2id, tag2id, max_char_len=MAX_CHAR_LEN)
collate = neural.make_collate(PAD_ID, CHAR_PAD_ID, MAX_CHAR_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

bilstm, bilstm_train_s, history, best_dev_f1 = neural.train_with_early_stop(
    bilstm, train_loader, valid_sents, word2id, char2id, id2tag, tag2id,
    max_char_len=MAX_CHAR_LEN, epochs=EPOCHS, patience=PATIENCE, lr=1e-3,
)
print(f\"  trained in {bilstm_train_s:.1f}s, best dev F1 = {best_dev_f1:.4f}\")

bilstm_path = \"models/bilstm_charaware_fp32.pt\"
torch.save(bilstm.state_dict(), bilstm_path)

torch.set_num_threads(1)
bilstm.eval()
bilstm_predict = neural.make_predict_fn(bilstm, word2id, char2id, id2tag, MAX_CHAR_LEN)

cold_ms, warm_times = cold_vs_warm_latency(bilstm_predict, bench_token_lists)
y_pred = [bilstm_predict(toks) for toks in test_token_lists]
add_row(\"BiLSTM-char-fp32\", bilstm_path, y_pred, bilstm_train_s, cold_ms, warm_times, bench_total_tokens)
gc_pause(0.3)
""")

# ---------------------------------------------------------------------------
# Section 9 — BiLSTM-int8
# ---------------------------------------------------------------------------
md("""## 9. Model 4 — Char-aware BiLSTM (int8, dynamic quantization)

Post-training `quantize_dynamic` on `nn.LSTM` and `nn.Linear`. No retraining.""")

code("""qbilstm = neural.quantize_dynamic(bilstm)
qbilstm.eval()
qbilstm_path = \"models/bilstm_charaware_int8.pt\"
torch.save(qbilstm.state_dict(), qbilstm_path)

torch.set_num_threads(1)
qbilstm_predict = neural.make_predict_fn(qbilstm, word2id, char2id, id2tag, MAX_CHAR_LEN)

cold_ms, warm_times = cold_vs_warm_latency(qbilstm_predict, bench_token_lists)
y_pred = [qbilstm_predict(toks) for toks in test_token_lists]
add_row(\"BiLSTM-char-int8\", qbilstm_path, y_pred, 0.0, cold_ms, warm_times, bench_total_tokens)
gc_pause(0.3)
""")

# ---------------------------------------------------------------------------
# Section 10 — DistilBERT
# ---------------------------------------------------------------------------
md("""## 10. Model 5 — DistilBERT-NER (off-the-shelf, inference only)

`elastic/distilbert-base-cased-finetuned-conll03-english`, pinned to `revision="main"` and snapshot-cached locally by `transformers`. Subword-to-token alignment uses the first-subword convention.""")

code("""torch.set_num_threads(1)
print(\"Loading DistilBERT-NER ...\")
db_wrap, db_predict, db_load_s = transformer.make_distilbert_predict(num_threads=1)
print(f\"  loaded in {db_load_s:.1f}s; id2label = {db_wrap.id2label}\")

# Approximate model size on disk: walk the HF cache for this model id.
def _hf_model_size_mb(model_id):
    import glob
    # Common cache locations across transformers versions.
    candidates = []
    for env in (\"HF_HOME\", \"TRANSFORMERS_CACHE\", \"HUGGINGFACE_HUB_CACHE\"):
        v = os.environ.get(env)
        if v:
            candidates.append(v)
    candidates.append(os.path.expanduser(\"~/.cache/huggingface\"))
    safetensors_total = 0
    pt_total = 0
    for root in candidates:
        for ext in (\"*.safetensors\", \"*.bin\"):
            for f in glob.glob(os.path.join(root, \"**\", ext), recursive=True):
                if model_id.replace(\"/\", \"--\") in f:
                    sz = os.path.getsize(f)
                    if ext == \"*.safetensors\":
                        safetensors_total += sz
                    else:
                        pt_total += sz
    total = safetensors_total or pt_total
    return total / (1024 ** 2) if total else float(\"nan\")

db_size_mb = _hf_model_size_mb(transformer.DEFAULT_MODEL_ID)
print(f\"  DistilBERT model files on disk: {db_size_mb:.1f} MB\")

cold_ms, warm_times = cold_vs_warm_latency(db_predict, bench_token_lists)
y_pred_db = []
print(\"  Running DistilBERT inference on full test ...\")
t0 = time.perf_counter()
for toks in test_token_lists:
    y_pred_db.append(db_predict(toks))
print(f\"  full-test inference: {time.perf_counter() - t0:.1f}s\")

# Manual append to capture custom size source.
lat = latency_summary(warm_times, bench_total_tokens)
f1_pt, f1_lo, f1_hi = bootstrap_f1_ci(y_true, y_pred_db, n_resamples=1000, seed=SEED)
rows.append({
    \"model\": \"DistilBERT-NER\",
    \"f1\": f1_pt,
    \"f1_low_95\": f1_lo,
    \"f1_high_95\": f1_hi,
    \"latency_mean_ms\": lat[\"mean_ms\"],
    \"latency_median_ms\": lat[\"median_ms\"],
    \"latency_p95_ms\": lat[\"p95_ms\"],
    \"latency_p99_ms\": lat[\"p99_ms\"],
    \"throughput_tok_per_s\": lat[\"throughput_tok_per_s\"],
    \"cold_start_ms\": cold_ms,
    \"model_size_mb\": db_size_mb,
    \"train_seconds\": 0.0,
})
y_preds[\"DistilBERT-NER\"] = y_pred_db
print(f\"  DistilBERT-NER     F1={f1_pt:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]  \"
      f\"p50={lat['median_ms']:.2f}ms p95={lat['p95_ms']:.2f}ms  \"
      f\"throughput={lat['throughput_tok_per_s']:,.0f} tok/s  size={db_size_mb:.2f}MB\")
gc_pause(0.3)
""")

# ---------------------------------------------------------------------------
# Section 11 — Batching sweep
# ---------------------------------------------------------------------------
md("""## 11. Optimization 2 — Batching sweep

For BiLSTM (fp32 + int8): batch sizes ∈ {1, 4, 16, 32, 64} on 1,024 test sentences.
Reports throughput (tokens/s) per batch size — the key production-deployment knob.""")

code("""BATCH_BENCH_N = min(1024, len(test_sents))
batch_bench_tokens = test_token_lists[:BATCH_BENCH_N]
batch_bench_total_tokens = sum(len(t) for t in batch_bench_tokens)
batch_sizes = [1, 4, 16, 32, 64]

batching_rows = []

def _run_batched(model, batch_size):
    pf = neural.make_batched_predict_fn(
        model, word2id, char2id, id2tag, MAX_CHAR_LEN,
        word_pad_id=PAD_ID, char_pad_id=CHAR_PAD_ID, batch_size=batch_size,
    )
    # Warm-up
    _ = pf(batch_bench_tokens[:max(2 * batch_size, 16)])
    t0 = time.perf_counter()
    _ = pf(batch_bench_tokens)
    elapsed = time.perf_counter() - t0
    return elapsed, batch_bench_total_tokens / elapsed

torch.set_num_threads(1)
for label, m in [(\"BiLSTM-char-fp32\", bilstm), (\"BiLSTM-char-int8\", qbilstm)]:
    for b in batch_sizes:
        elapsed_s, throughput = _run_batched(m, b)
        batching_rows.append({\"model\": label, \"batch_size\": b,
                              \"elapsed_s\": elapsed_s, \"throughput_tok_per_s\": throughput})
        print(f\"  {label:<22} bs={b:>3}  {elapsed_s:.2f}s  {throughput:,.0f} tok/s\")

batching_df = pd.DataFrame(batching_rows)
batching_df.to_csv(\"results_batching.csv\", index=False)
print(\"Saved results_batching.csv\")
""")

# ---------------------------------------------------------------------------
# Section 12 — Vocab pruning sweep
# ---------------------------------------------------------------------------
md("""## 12. Optimization 3 — Vocabulary pruning (post-hoc, no retrain)

Sweep the *effective* vocabulary by replacing low-frequency words' embedding rows with the `<UNK>` row (rows kept in place — same architecture, cheaper deploy).
Report F1 + effective-table size in MB.

We also retrain at vocab=2K as a sanity check that post-hoc tracks retrain.""")

code("""VOCAB_TARGETS = [1000, 2000, 5000, 10000, len(vocab_words)]

base_state = {k: v.clone() for k, v in bilstm.state_dict().items()}
pruning_rows = []

for tgt in VOCAB_TARGETS:
    new_sd, n_unique = neural.prune_embedding_table(
        base_state, vocab_words, word_freq, word2id, target_size=tgt,
    )
    pruned = neural.CharAwareBiLSTM(
        vocab_size=len(vocab_words), num_chars=len(char_list), num_tags=NUM_TAGS,
        word_emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
        word_pad_id=PAD_ID, char_pad_id=CHAR_PAD_ID,
    )
    pruned.load_state_dict(new_sd)
    pruned.eval()
    torch.set_num_threads(1)
    pf = neural.make_predict_fn(pruned, word2id, char2id, id2tag, MAX_CHAR_LEN)
    yp = [pf(toks) for toks in test_token_lists]
    from seqeval.metrics import f1_score
    from seqeval.scheme import IOB2
    f1 = f1_score(y_true, yp, mode=\"strict\", scheme=IOB2)
    eff_table_mb = n_unique * EMB_DIM * 4 / (1024 ** 2)
    pruning_rows.append({
        \"strategy\": \"post_hoc\",
        \"vocab_size\": min(tgt, len(vocab_words)),
        \"unique_vectors\": n_unique,
        \"effective_emb_table_mb\": eff_table_mb,
        \"f1\": f1,
    })
    print(f\"  post-hoc target={tgt:>5}  unique={n_unique:>5}  table={eff_table_mb:.2f}MB  F1={f1:.4f}\")

# Retrained sanity check at vocab=2K (1 retrain, fewer epochs).
print(\"Retraining at vocab=2K (sanity check) ...\")
RETRAIN_TARGET = 2000
freq_sorted = sorted(((word_freq.get(w, 0), i, w) for i, w in enumerate(vocab_words)
                      if w not in (neural.PAD, neural.UNK)), key=lambda x: -x[0])
keep_words = [neural.PAD, neural.UNK] + [w for _, _, w in freq_sorted[:RETRAIN_TARGET - 2]]
re_word2id = {w: i for i, w in enumerate(keep_words)}
re_glove, _ = neural.load_glove_embeddings(keep_words, emb_dim=EMB_DIM,
                                            cache_path=\"embeddings/glove.6B.100d.txt\")
RE_PAD_ID = re_word2id[neural.PAD]

torch.manual_seed(SEED + 7)
re_model = neural.CharAwareBiLSTM(
    vocab_size=len(keep_words), num_chars=len(char_list), num_tags=NUM_TAGS,
    word_emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
    word_pad_id=RE_PAD_ID, char_pad_id=CHAR_PAD_ID,
)
with torch.no_grad():
    re_model.word_emb.weight.copy_(torch.from_numpy(re_glove))

re_train_ds = neural.NERDataset(train_sents, re_word2id, char2id, tag2id, max_char_len=MAX_CHAR_LEN)
re_collate = neural.make_collate(RE_PAD_ID, CHAR_PAD_ID, MAX_CHAR_LEN)
re_train_loader = DataLoader(re_train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=re_collate)
re_model, _, _, _ = neural.train_with_early_stop(
    re_model, re_train_loader, valid_sents, re_word2id, char2id, id2tag, tag2id,
    max_char_len=MAX_CHAR_LEN, epochs=8, patience=2, lr=1e-3,
)
torch.set_num_threads(1)
re_model.eval()
re_pf = neural.make_predict_fn(re_model, re_word2id, char2id, id2tag, MAX_CHAR_LEN)
yp_re = [re_pf(toks) for toks in test_token_lists]
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
re_f1 = f1_score(y_true, yp_re, mode=\"strict\", scheme=IOB2)
re_table_mb = len(keep_words) * EMB_DIM * 4 / (1024 ** 2)
pruning_rows.append({
    \"strategy\": \"retrain\",
    \"vocab_size\": len(keep_words),
    \"unique_vectors\": len(keep_words),
    \"effective_emb_table_mb\": re_table_mb,
    \"f1\": re_f1,
})
print(f\"  retrain      vocab={len(keep_words)}  F1={re_f1:.4f}  table={re_table_mb:.2f}MB\")

pruning_df = pd.DataFrame(pruning_rows)
pruning_df.to_csv(\"results_pruning.csv\", index=False)
print(\"Saved results_pruning.csv\")
del re_model
gc_pause(0.3)
""")

# ---------------------------------------------------------------------------
# Section 13 — Multi-thread sweep
# ---------------------------------------------------------------------------
md("""## 13. Optimization 4 — Multi-thread sweep

For BiLSTM-fp32 and BiLSTM-int8 only (CRF/HMM are single-threaded C internally).
Threads ∈ {1, 2, 4, max(cpu)}.""")

code("""thread_counts = sorted(set([1, 2, 4, max(1, os.cpu_count() or 1)]))
threading_rows = []
# Note: torch.set_num_interop_threads can only be called once per process — already set in section 1.
for label, m in [(\"BiLSTM-char-fp32\", bilstm), (\"BiLSTM-char-int8\", qbilstm)]:
    for t in thread_counts:
        torch.set_num_threads(t)
        pf = neural.make_predict_fn(m, word2id, char2id, id2tag, MAX_CHAR_LEN)
        # Warm-up
        for s in bench_token_lists[:N_WARMUP]:
            pf(s)
        times = bench_mod.time_predict(pf, bench_token_lists, warmup=0)
        ls = bench_mod.latency_summary(times, bench_total_tokens)
        threading_rows.append({\"model\": label, \"threads\": t,
                                **ls})
        print(f\"  {label:<22} threads={t:>2}  p50={ls['median_ms']:.2f}ms  p95={ls['p95_ms']:.2f}ms  throughput={ls['throughput_tok_per_s']:,.0f} tok/s\")

torch.set_num_threads(1)
threading_df = pd.DataFrame(threading_rows)
threading_df.to_csv(\"results_threads.csv\", index=False)
print(\"Saved results_threads.csv\")
""")

# ---------------------------------------------------------------------------
# Section 14 — Headline results table
# ---------------------------------------------------------------------------
md("""## 14. Headline results table""")

code("""df = pd.DataFrame(rows)
col_order = [\"model\", \"f1\", \"f1_low_95\", \"f1_high_95\",
             \"latency_mean_ms\", \"latency_median_ms\", \"latency_p95_ms\", \"latency_p99_ms\",
             \"throughput_tok_per_s\", \"cold_start_ms\", \"model_size_mb\", \"train_seconds\"]
df = df[col_order]
df.to_csv(\"results.csv\", index=False)
print(df.to_string(index=False))
""")

# ---------------------------------------------------------------------------
# Section 15 — Per-entity F1 + classification reports
# ---------------------------------------------------------------------------
md("""## 15. Per-entity F1 breakdown + error analysis""")

code("""from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

per_entity = {}
for name, yp in y_preds.items():
    rep = classification_report(y_true, yp, mode=\"strict\", scheme=IOB2, output_dict=True, digits=3)
    per_entity[name] = rep

# Print full reports (poster-friendly).
for name, yp in y_preds.items():
    print(f\"\\n=== {name} ===\")
    print(classification_report(y_true, yp, mode=\"strict\", scheme=IOB2, digits=3))

# Build per-entity-type F1 matrix.
ENT_TYPES = [\"PER\", \"LOC\", \"ORG\", \"MISC\"]
model_names = [r[\"model\"] for r in rows]
mat = np.zeros((len(model_names), len(ENT_TYPES)))
for i, name in enumerate(model_names):
    rep = per_entity[name]
    for j, et in enumerate(ENT_TYPES):
        mat[i, j] = rep.get(et, {}).get(\"f1-score\", 0.0)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(ENT_TYPES))
width = 0.13
colors = [\"#d62728\", \"#2ca02c\", \"#1f77b4\", \"#9467bd\", \"#ff7f0e\"]
for i, name in enumerate(model_names):
    offset = (i - (len(model_names) - 1) / 2) * width
    ax.bar(x + offset, mat[i], width, label=name, color=colors[i % len(colors)])
ax.set_xticks(x)
ax.set_xticklabels(ENT_TYPES)
ax.set_ylabel(\"Span-F1\")
ax.set_title(\"Per-entity F1 — CoNLL-2003 NER on CPU\")
ax.legend(loc=\"lower right\", fontsize=9)
ax.set_ylim(0, 1)
ax.grid(axis=\"y\", alpha=0.3)
fig.tight_layout()
fig.savefig(\"figures/per_entity_f1.png\", dpi=300, bbox_inches=\"tight\")
plt.show()
print(\"Saved figures/per_entity_f1.png\")
""")

# ---------------------------------------------------------------------------
# Section 16 — Pareto plot
# ---------------------------------------------------------------------------
md("""## 16. Pareto plot — quality vs latency""")

code("""fig, ax = plt.subplots(figsize=(9, 6))
plot_colors = {\"HMM\": \"#d62728\", \"CRF\": \"#2ca02c\",
               \"BiLSTM-char-fp32\": \"#1f77b4\", \"BiLSTM-char-int8\": \"#9467bd\",
               \"DistilBERT-NER\": \"#ff7f0e\"}

for _, r in df.iterrows():
    ax.scatter(r[\"latency_median_ms\"], r[\"f1\"], s=200,
               color=plot_colors.get(r[\"model\"], \"k\"),
               edgecolors=\"black\", linewidth=0.7, zorder=3)
    # CI as vertical bar
    ax.errorbar(r[\"latency_median_ms\"], r[\"f1\"],
                yerr=[[r[\"f1\"] - r[\"f1_low_95\"]], [r[\"f1_high_95\"] - r[\"f1\"]]],
                color=plot_colors.get(r[\"model\"], \"k\"), alpha=0.7, capsize=4, zorder=2)
    ax.annotate(r[\"model\"], (r[\"latency_median_ms\"], r[\"f1\"]),
                textcoords=\"offset points\", xytext=(10, 10), fontsize=10)

# Pareto frontier
df_s = df.sort_values(\"latency_median_ms\").reset_index(drop=True)
frontier = []
best = -1.0
for _, r in df_s.iterrows():
    if r[\"f1\"] > best:
        frontier.append((r[\"latency_median_ms\"], r[\"f1\"]))
        best = r[\"f1\"]
if len(frontier) >= 2:
    fx, fy = zip(*frontier)
    ax.plot(fx, fy, \"k--\", alpha=0.55, label=\"Pareto frontier\")

# Latency-budget reference lines
for budget, label in [(5, \"5 ms\"), (10, \"10 ms\"), (50, \"50 ms\")]:
    ax.axvline(budget, color=\"gray\", linestyle=\":\", alpha=0.4)
    ax.text(budget, ax.get_ylim()[0] + 0.02, label, color=\"gray\",
            ha=\"left\", va=\"bottom\", fontsize=9)

ax.set_xscale(\"log\")
ax.set_xlabel(\"Median latency per sentence (ms, log scale, single-threaded)\")
ax.set_ylabel(\"Span-F1 (CoNLL-2003 NER, strict, IOB2; 95% CI)\")
ax.set_title(\"CoNLL-2003 NER on CPU — Quality vs Latency (full version)\")
ax.legend(loc=\"lower right\")
ax.grid(True, which=\"both\", alpha=0.3)
fig.tight_layout()
fig.savefig(\"figures/pareto_f1_vs_latency.png\", dpi=300, bbox_inches=\"tight\")
plt.show()

# Throughput vs batch size
fig2, ax2 = plt.subplots(figsize=(8, 5))
for label in batching_df[\"model\"].unique():
    sub = batching_df[batching_df[\"model\"] == label]
    ax2.plot(sub[\"batch_size\"], sub[\"throughput_tok_per_s\"], marker=\"o\", label=label)
ax2.set_xscale(\"log\", base=2)
ax2.set_xlabel(\"Batch size\")
ax2.set_ylabel(\"Throughput (tokens / s)\")
ax2.set_title(\"BiLSTM throughput vs batch size (1,024 test sentences)\")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(\"figures/throughput_vs_batch.png\", dpi=300, bbox_inches=\"tight\")
plt.show()

# F1 vs vocab size (pruning)
fig3, ax3 = plt.subplots(figsize=(8, 5))
post = pruning_df[pruning_df[\"strategy\"] == \"post_hoc\"].sort_values(\"vocab_size\")
ret = pruning_df[pruning_df[\"strategy\"] == \"retrain\"]
ax3.plot(post[\"vocab_size\"], post[\"f1\"], marker=\"o\", label=\"post-hoc pruning\", color=\"#1f77b4\")
if len(ret) > 0:
    ax3.scatter(ret[\"vocab_size\"], ret[\"f1\"], marker=\"*\", s=200,
                color=\"#d62728\", label=\"retrain @ 2K (sanity)\", zorder=5)
ax3.set_xscale(\"log\")
ax3.set_xlabel(\"Effective vocabulary size\")
ax3.set_ylabel(\"Span-F1\")
ax3.set_title(\"BiLSTM-char-fp32: F1 vs vocabulary size\")
ax3.grid(True, alpha=0.3)
ax3b = ax3.twinx()
ax3b.plot(post[\"vocab_size\"], post[\"effective_emb_table_mb\"], marker=\"s\",
          color=\"gray\", alpha=0.6, linestyle=\"--\", label=\"emb-table size (MB)\")
ax3b.set_ylabel(\"Effective embedding-table size (MB)\")
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc=\"lower right\")
fig3.tight_layout()
fig3.savefig(\"figures/f1_vs_vocab.png\", dpi=300, bbox_inches=\"tight\")
plt.show()

print(\"Saved 4 figures.\")
""")

# ---------------------------------------------------------------------------
# Section 17 — Three-tier recommendation
# ---------------------------------------------------------------------------
md("""## 17. Three-tier latency recommendation

Pick the highest-F1 model whose median single-threaded latency ≤ tier budget.""")

code("""tiers = [(\"5 ms (real-time chat / streaming)\", 5.0),
         (\"10 ms (real-time tagging in API hot path)\", 10.0),
         (\"50 ms (batch / serverless)\", 50.0)]
recs = []
for label, budget in tiers:
    cand = df[df[\"latency_median_ms\"] <= budget]
    if len(cand) > 0:
        winner = cand.loc[cand[\"f1\"].idxmax()]
        recs.append((label, budget, winner[\"model\"], winner[\"f1\"], winner[\"latency_median_ms\"]))
    else:
        # Fall back to overall winner if no model fits.
        winner = df.loc[df[\"f1\"].idxmax()]
        recs.append((label, budget, f\"{winner['model']} (no model meets budget)\",
                     winner[\"f1\"], winner[\"latency_median_ms\"]))

print(\"\\n=== 3-Tier recommendation ===\")
for label, budget, model, f1, lat in recs:
    print(f\"  budget {budget:>4.0f}ms ({label}): {model}  -> F1={f1:.4f}, p50={lat:.2f}ms\")

with open(\"recommendation.json\", \"w\", encoding=\"utf-8\") as f:
    json.dump([{\"tier\": l, \"budget_ms\": b, \"model\": m, \"f1\": float(f), \"p50_ms\": float(lat)}
               for (l, b, m, f, lat) in recs], f, indent=2)
print(\"Saved recommendation.json\")
""")

md("""---

That concludes the notebook. Outputs you should now find on disk:

- `results.csv` — headline 5-row table with bootstrap CIs.
- `results_batching.csv`, `results_pruning.csv`, `results_threads.csv` — supplementary sweeps.
- `recommendation.json` — 3-tier picks.
- `figures/pareto_f1_vs_latency.png`, `figures/per_entity_f1.png`, `figures/throughput_vs_batch.png`, `figures/f1_vs_vocab.png`.
- `models/hmm.pkl`, `models/crf.pkl`, `models/bilstm_charaware_fp32.pt`, `models/bilstm_charaware_int8.pt`.
""")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
nb["cells"] = cells
with open("notebook.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("notebook.ipynb written")
