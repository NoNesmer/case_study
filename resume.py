"""Resume the notebook from saved models (sections 13-17).

Skips training (HMM, CRF, BiLSTM, retrain) and uses the saved checkpoints.
Re-runs prediction on the full test set, threading sweep (with fix), and figures.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src import bench as bench_mod
from src import data as data_mod
from src import models_classical as classical
from src import models_neural as neural
from src import models_transformer as transformer

# Ensure single-threaded inference defaults for fair latency.
SEED = 42
import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(1)
# Note: set_num_interop_threads can only be called once. Don't call here in case it was set elsewhere.

# ---------------------------------------------------------------------------
# Re-load data + vocabs (must match the notebook's seeding)
# ---------------------------------------------------------------------------
print("Loading data ...")
train_sents, valid_sents, test_sents, label_list = data_mod.load_conll(data_dir=str(ROOT / "data"))
NUM_TAGS = len(label_list)
tag2id = {t: i for i, t in enumerate(label_list)}
id2tag = list(label_list)

print("Building vocabs ...")
vocab_words, word2id, word_freq = neural.build_word_vocab(train_sents, min_freq=2)
char_list, char2id = neural.build_char_vocab(train_sents)
PAD_ID = word2id[neural.PAD]
CHAR_PAD_ID = char2id[neural.CHAR_PAD]

# Bench sample (must match notebook seeding: SEED + 1)
random.seed(SEED + 1)
N_BENCH = 200
N_WARMUP = 20
bench_indices = random.sample(range(len(test_sents)), min(N_BENCH, len(test_sents)))
bench_sents = [test_sents[i] for i in bench_indices]
bench_token_lists = [[t for t, _ in s] for s in bench_sents]
bench_total_tokens = sum(len(s) for s in bench_token_lists)
print(f"Benchmark sample: {len(bench_token_lists)} sents, {bench_total_tokens} tokens")

y_true = [[tag for _, tag in s] for s in test_sents]
test_token_lists = [[t for t, _ in s] for s in test_sents]

EMB_DIM = 100
HIDDEN_DIM = 200
MAX_CHAR_LEN = 25

rows = []
y_preds = {}


def add_row_with_size(name, model_path, y_pred, train_seconds, cold_ms, warm_times_ms,
                      total_tokens, size_mb=None):
    lat = bench_mod.latency_summary(warm_times_ms, total_tokens)
    f1, lo, hi = bench_mod.bootstrap_f1_ci(y_true, y_pred, n_resamples=1000, seed=SEED)
    if size_mb is None:
        size_mb = bench_mod.file_size_mb(model_path) if model_path else float("nan")
    row = {
        "model": name,
        "f1": f1, "f1_low_95": lo, "f1_high_95": hi,
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
    print(f"  {name:<22} F1={f1:.4f} [{lo:.4f}, {hi:.4f}]  "
          f"p50={lat['median_ms']:.2f}ms p95={lat['p95_ms']:.2f}ms  "
          f"throughput={lat['throughput_tok_per_s']:,.0f} tok/s  size={size_mb:.2f}MB")


# ---------------------------------------------------------------------------
# Load HMM
# ---------------------------------------------------------------------------
print("\n--- HMM ---")
hmm_path = str(ROOT / "models" / "hmm.pkl")
with open(hmm_path, "rb") as f:
    try:
        hmm_model = pickle.load(f)
    except Exception:
        f.seek(0)
        import dill
        hmm_model = dill.load(f)
hmm_predict = classical.make_hmm_predict(hmm_model)
torch.set_num_threads(1)
cold_ms, warm_times = bench_mod.cold_vs_warm_latency(hmm_predict, bench_token_lists)
y_pred = [hmm_predict(toks) for toks in test_token_lists]
add_row_with_size("HMM", hmm_path, y_pred, 0.0, cold_ms, warm_times, bench_total_tokens)

# ---------------------------------------------------------------------------
# Load CRF
# ---------------------------------------------------------------------------
print("\n--- CRF ---")
crf_path = str(ROOT / "models" / "crf.pkl")
with open(crf_path, "rb") as f:
    crf_model = pickle.load(f)
crf_predict = classical.make_crf_predict(crf_model)
cold_ms, warm_times = bench_mod.cold_vs_warm_latency(crf_predict, bench_token_lists)
y_pred = [crf_predict(toks) for toks in test_token_lists]
add_row_with_size("CRF", crf_path, y_pred, 0.0, cold_ms, warm_times, bench_total_tokens)

# ---------------------------------------------------------------------------
# Load BiLSTM-fp32
# ---------------------------------------------------------------------------
print("\n--- BiLSTM-char-fp32 ---")
bilstm_path = str(ROOT / "models" / "bilstm_charaware_fp32.pt")
bilstm = neural.CharAwareBiLSTM(
    vocab_size=len(vocab_words), num_chars=len(char_list), num_tags=NUM_TAGS,
    word_emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
    word_pad_id=PAD_ID, char_pad_id=CHAR_PAD_ID,
)
bilstm.load_state_dict(torch.load(bilstm_path, map_location="cpu"))
bilstm.eval()
torch.set_num_threads(1)
bilstm_predict = neural.make_predict_fn(bilstm, word2id, char2id, id2tag, MAX_CHAR_LEN)
cold_ms, warm_times = bench_mod.cold_vs_warm_latency(bilstm_predict, bench_token_lists)
y_pred = [bilstm_predict(toks) for toks in test_token_lists]
add_row_with_size("BiLSTM-char-fp32", bilstm_path, y_pred, 0.0, cold_ms, warm_times, bench_total_tokens)

# ---------------------------------------------------------------------------
# Quantize and load int8
# ---------------------------------------------------------------------------
print("\n--- BiLSTM-char-int8 ---")
qbilstm_path = str(ROOT / "models" / "bilstm_charaware_int8.pt")
qbilstm = neural.quantize_dynamic(bilstm)
qbilstm.eval()
qbilstm_predict = neural.make_predict_fn(qbilstm, word2id, char2id, id2tag, MAX_CHAR_LEN)
cold_ms, warm_times = bench_mod.cold_vs_warm_latency(qbilstm_predict, bench_token_lists)
y_pred = [qbilstm_predict(toks) for toks in test_token_lists]
add_row_with_size("BiLSTM-char-int8", qbilstm_path, y_pred, 0.0, cold_ms, warm_times, bench_total_tokens)

# ---------------------------------------------------------------------------
# DistilBERT (off-the-shelf, inference only)
# ---------------------------------------------------------------------------
print("\n--- DistilBERT-NER ---")
torch.set_num_threads(1)
db_wrap, db_predict, db_load_s = transformer.make_distilbert_predict(num_threads=1)
print(f"  loaded in {db_load_s:.1f}s")

# Approximate size from HF cache.
def _hf_model_size_mb(model_id):
    import glob
    candidates = []
    for env in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(env)
        if v:
            candidates.append(v)
    candidates.append(os.path.expanduser("~/.cache/huggingface"))
    safetensors_total = 0
    pt_total = 0
    for root in candidates:
        for ext in ("*.safetensors", "*.bin"):
            for f in glob.glob(os.path.join(root, "**", ext), recursive=True):
                if model_id.replace("/", "--") in f:
                    sz = os.path.getsize(f)
                    if ext == "*.safetensors":
                        safetensors_total += sz
                    else:
                        pt_total += sz
    total = safetensors_total or pt_total
    return total / (1024 ** 2) if total else float("nan")

db_size_mb = _hf_model_size_mb(transformer.DEFAULT_MODEL_ID)
print(f"  DistilBERT model size: {db_size_mb:.1f} MB")

cold_ms, warm_times = bench_mod.cold_vs_warm_latency(db_predict, bench_token_lists)
print("  Running DistilBERT inference on full test ...")
t0 = time.perf_counter()
y_pred_db = [db_predict(toks) for toks in test_token_lists]
print(f"  full-test inference: {time.perf_counter() - t0:.1f}s")
add_row_with_size("DistilBERT-NER", None, y_pred_db, 0.0, cold_ms, warm_times,
                  bench_total_tokens, size_mb=db_size_mb)

# ---------------------------------------------------------------------------
# Threading sweep (FIXED — no set_num_interop_threads inside loop)
# ---------------------------------------------------------------------------
print("\n--- Threading sweep ---")
thread_counts = sorted(set([1, 2, 4, max(1, os.cpu_count() or 1)]))
threading_rows = []
for label, m in [("BiLSTM-char-fp32", bilstm), ("BiLSTM-char-int8", qbilstm)]:
    for t in thread_counts:
        torch.set_num_threads(t)
        pf = neural.make_predict_fn(m, word2id, char2id, id2tag, MAX_CHAR_LEN)
        for s in bench_token_lists[:N_WARMUP]:
            pf(s)
        times = bench_mod.time_predict(pf, bench_token_lists, warmup=0)
        ls = bench_mod.latency_summary(times, bench_total_tokens)
        threading_rows.append({"model": label, "threads": t, **ls})
        print(f"  {label:<22} threads={t:>2}  p50={ls['median_ms']:.2f}ms  "
              f"p95={ls['p95_ms']:.2f}ms  throughput={ls['throughput_tok_per_s']:,.0f} tok/s")

torch.set_num_threads(1)
threading_df = pd.DataFrame(threading_rows)
threading_df.to_csv(ROOT / "results_threads.csv", index=False)
print("Saved results_threads.csv")

# ---------------------------------------------------------------------------
# Headline results table
# ---------------------------------------------------------------------------
print("\n--- Saving results.csv ---")
df = pd.DataFrame(rows)
col_order = ["model", "f1", "f1_low_95", "f1_high_95",
             "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_p99_ms",
             "throughput_tok_per_s", "cold_start_ms", "model_size_mb", "train_seconds"]
df = df[col_order]
df.to_csv(ROOT / "results.csv", index=False)
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Per-entity F1 figure
# ---------------------------------------------------------------------------
print("\n--- Per-entity F1 ---")
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

per_entity = {}
for name, yp in y_preds.items():
    rep = classification_report(y_true, yp, mode="strict", scheme=IOB2,
                                output_dict=True, digits=3)
    per_entity[name] = rep

for name, yp in y_preds.items():
    print(f"\n=== {name} ===")
    print(classification_report(y_true, yp, mode="strict", scheme=IOB2, digits=3))

ENT_TYPES = ["PER", "LOC", "ORG", "MISC"]
model_names = [r["model"] for r in rows]
mat = np.zeros((len(model_names), len(ENT_TYPES)))
for i, name in enumerate(model_names):
    rep = per_entity[name]
    for j, et in enumerate(ENT_TYPES):
        mat[i, j] = rep.get(et, {}).get("f1-score", 0.0)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(ENT_TYPES))
width = 0.13
colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e"]
for i, name in enumerate(model_names):
    offset = (i - (len(model_names) - 1) / 2) * width
    ax.bar(x + offset, mat[i], width, label=name, color=colors[i % len(colors)])
ax.set_xticks(x)
ax.set_xticklabels(ENT_TYPES)
ax.set_ylabel("Span-F1")
ax.set_title("Per-entity F1 — CoNLL-2003 NER on CPU")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "per_entity_f1.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved figures/per_entity_f1.png")

# ---------------------------------------------------------------------------
# Pareto figure
# ---------------------------------------------------------------------------
print("\n--- Pareto figure ---")
fig, ax = plt.subplots(figsize=(9, 6))
plot_colors = {"HMM": "#d62728", "CRF": "#2ca02c",
               "BiLSTM-char-fp32": "#1f77b4", "BiLSTM-char-int8": "#9467bd",
               "DistilBERT-NER": "#ff7f0e"}
for _, r in df.iterrows():
    ax.scatter(r["latency_median_ms"], r["f1"], s=200,
               color=plot_colors.get(r["model"], "k"),
               edgecolors="black", linewidth=0.7, zorder=3)
    ax.errorbar(r["latency_median_ms"], r["f1"],
                yerr=[[r["f1"] - r["f1_low_95"]], [r["f1_high_95"] - r["f1"]]],
                color=plot_colors.get(r["model"], "k"), alpha=0.7, capsize=4, zorder=2)
    ax.annotate(r["model"], (r["latency_median_ms"], r["f1"]),
                textcoords="offset points", xytext=(10, 10), fontsize=10)

df_s = df.sort_values("latency_median_ms").reset_index(drop=True)
frontier = []
best = -1.0
for _, r in df_s.iterrows():
    if r["f1"] > best:
        frontier.append((r["latency_median_ms"], r["f1"]))
        best = r["f1"]
if len(frontier) >= 2:
    fx, fy = zip(*frontier)
    ax.plot(fx, fy, "k--", alpha=0.55, label="Pareto frontier")

for budget, label in [(5, "5 ms"), (10, "10 ms"), (50, "50 ms")]:
    ax.axvline(budget, color="gray", linestyle=":", alpha=0.4)
    ax.text(budget, ax.get_ylim()[0] + 0.02, label, color="gray",
            ha="left", va="bottom", fontsize=9)

ax.set_xscale("log")
ax.set_xlabel("Median latency per sentence (ms, log scale, single-threaded)")
ax.set_ylabel("Span-F1 (CoNLL-2003 NER, strict, IOB2; 95% CI)")
ax.set_title("CoNLL-2003 NER on CPU — Quality vs Latency (full version)")
ax.legend(loc="lower right")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "pareto_f1_vs_latency.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved figures/pareto_f1_vs_latency.png")

# ---------------------------------------------------------------------------
# Throughput vs batch (from existing CSV)
# ---------------------------------------------------------------------------
print("\n--- Batching figure ---")
batching_df = pd.read_csv(ROOT / "results_batching.csv")
fig, ax = plt.subplots(figsize=(8, 5))
for label in batching_df["model"].unique():
    sub = batching_df[batching_df["model"] == label]
    ax.plot(sub["batch_size"], sub["throughput_tok_per_s"], marker="o", label=label)
ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size")
ax.set_ylabel("Throughput (tokens / s)")
ax.set_title("BiLSTM throughput vs batch size (1,024 test sentences)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "throughput_vs_batch.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved figures/throughput_vs_batch.png")

# ---------------------------------------------------------------------------
# F1 vs vocab (from existing CSV)
# ---------------------------------------------------------------------------
print("\n--- Vocab pruning figure ---")
pruning_df = pd.read_csv(ROOT / "results_pruning.csv")
post = pruning_df[pruning_df["strategy"] == "post_hoc"].sort_values("vocab_size")
ret = pruning_df[pruning_df["strategy"] == "retrain"]
fig, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(post["vocab_size"], post["f1"], marker="o", label="post-hoc pruning", color="#1f77b4")
if len(ret) > 0:
    ax3.scatter(ret["vocab_size"], ret["f1"], marker="*", s=200,
                color="#d62728", label="retrain @ 2K (sanity)", zorder=5)
ax3.set_xscale("log")
ax3.set_xlabel("Effective vocabulary size")
ax3.set_ylabel("Span-F1")
ax3.set_title("BiLSTM-char-fp32: F1 vs vocabulary size")
ax3.grid(True, alpha=0.3)
ax3b = ax3.twinx()
ax3b.plot(post["vocab_size"], post["effective_emb_table_mb"], marker="s",
          color="gray", alpha=0.6, linestyle="--", label="emb-table size (MB)")
ax3b.set_ylabel("Effective embedding-table size (MB)")
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "f1_vs_vocab.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved figures/f1_vs_vocab.png")

# ---------------------------------------------------------------------------
# 3-tier recommendation
# ---------------------------------------------------------------------------
print("\n--- 3-tier Recommendation ---")
tiers = [("5 ms (real-time chat / streaming)", 5.0),
         ("10 ms (real-time tagging in API hot path)", 10.0),
         ("50 ms (batch / serverless)", 50.0)]
recs = []
for label, budget in tiers:
    cand = df[df["latency_median_ms"] <= budget]
    if len(cand) > 0:
        winner = cand.loc[cand["f1"].idxmax()]
        recs.append((label, budget, winner["model"], winner["f1"], winner["latency_median_ms"]))
    else:
        winner = df.loc[df["f1"].idxmax()]
        recs.append((label, budget, f"{winner['model']} (no model meets budget)",
                     winner["f1"], winner["latency_median_ms"]))

for label, budget, model, f1, lat in recs:
    print(f"  budget {budget:>4.0f}ms ({label}): {model}  -> F1={f1:.4f}, p50={lat:.2f}ms")

with open(ROOT / "recommendation.json", "w", encoding="utf-8") as f:
    json.dump([{"tier": l, "budget_ms": b, "model": m, "f1": float(f), "p50_ms": float(lat)}
               for (l, b, m, f, lat) in recs], f, indent=2)
print("Saved recommendation.json")
print("\n=== Resume complete ===")
