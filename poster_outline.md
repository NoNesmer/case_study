# Poster outline — Case Study 4.9 (FULL)

Drop into PowerPoint/Canva. The poster is roughly a 3×3 grid of panels; this outline maps each panel to its content + figure. All figures are 300 DPI in `figures/`.

## Title bar
**Latency-Constrained Sequence Labelling on CPU — A Quality vs Speed Comparison for Real-Time NER**
*Case Study 4.9, NLP 2026*

## Top-left — Motivation (~120 words)
- Production NER on CPU-only nodes is a real constraint — many deployments don't have GPUs.
- Transformer baselines dominate F1 leaderboards but routinely blow real-time latency budgets (5–50 ms/sent) on CPU.
- Classical models (HMM, CRF) are sub-millisecond but have lower ceilings.
- Strong neural baselines (char-aware BiLSTM with pretrained word embeddings) sit between the two.
- This study quantifies the trade-off and ends with a per-tier model recommendation.

## Top-center — Methodology (~140 words)
- **5 models:** HMM, CRF, char-aware BiLSTM (fp32), char-aware BiLSTM (int8), DistilBERT-NER (off-the-shelf).
- **3 deploy-time optimizations:** dynamic int8 quantization, batching sweep, vocabulary pruning. Plus a multi-thread sweep.
- BiLSTM: GloVe-100d word embed + CharCNN (30×k=3) → BiLSTM (hidden=200) → softmax. Adam 1e-3, dropout 0.5, 15 epochs with dev-set early stopping (patience=3).
- CRF: sklearn-crfsuite, rich features (word shape, suffix/prefix 2-4, ±2 context), 100 L-BFGS iters.
- DistilBERT: `elastic/distilbert-base-cased-finetuned-conll03-english`, inference only.
- All single-threaded latency benchmarking; multi-thread reported as a side experiment.
- F1 reported with bootstrap 95% CI (1,000 sentence-level resamples).

## Top-right — Experimental setup (compact box)
- **Dataset:** CoNLL-2003 (synalp .txt mirror), full train (14,041 sents), full test (3,453 sents).
- **Tag scheme:** IOB2 (converted from IOB1), strict `seqeval`.
- **Hardware:** Windows 10, AMD Ryzen 7 5800H, 15.4 GB RAM, Python 3.12, PyTorch 2.8 CPU.
- **Latency:** 200 test sents, 20-sent warm-up, single-threaded.
- **Cold-start vs warm:** first call timed separately.

## Center-left — Results table (large, headline panel)
Insert rendered table from `results.csv`, formatted as below (numbers filled in by the notebook):

| Model | Span-F1 (95% CI) | Latency p50 (ms) | Latency p95 (ms) | Throughput (tok/s) | Size (MB) |
| --- | --- | --- | --- | --- | --- |
| HMM | 0.638 [0.623, 0.653] | 0.93 | 3.25 | 11,380 | 0.40 |
| CRF | 0.827 [0.816, 0.838] | 0.19 | 0.92 | 43,059 | 2.59 |
| BiLSTM-char-fp32 | 0.854 [0.843, 0.864] | 3.08 | 7.96 | 3,646 | 6.24 |
| BiLSTM-char-int8 | 0.854 [0.843, 0.864] | 3.14 | 8.28 | 3,580 | 4.72 |
| DistilBERT-NER | 0.898 [0.888, 0.906] | 66.70 | 115.45 | 188 | 248.72 |

## Center — Pareto frontier (large)
**Insert `figures/pareto_f1_vs_latency.png`.**

Key reading instructions:
- Log-x axis (latency).
- Vertical bars are 95% bootstrap CIs.
- Dotted vertical lines mark the 5 / 10 / 50 ms latency budgets.
- The dashed black line is the empirical Pareto frontier — points strictly above + left of it are dominated.

## Center-right — Optimization experiments (3 small panels)

**Quantization:** point comparison (BiLSTM-fp32 vs BiLSTM-int8) — F1 matches to within bootstrap noise (0.85368 vs 0.85360); on-disk model shrinks 24% (6.24 → 4.72 MB); single-thread p50 latency is essentially flat (3.08 → 3.14 ms — marginal regression). **Size win, not a latency win at hidden=200.**

**Batching:** insert `figures/throughput_vs_batch.png`. Throughput rises ~2.3× from batch=1 (6,214 tok/s) to batch=16 (14,340 tok/s, the peak), then declines because padding overhead grows faster than the batched-GEMM gain. **Operating point: batch=16, not 32 or 64.**

**Vocab pruning:** insert `figures/f1_vs_vocab.png`. Two y-axes: F1 (left) and effective embedding-table size in MB (right). Pruning to 10K (8% size cut) costs <0.3 F1; pruning to 5K (54% size cut) costs ~2.1 F1; pruning to 2K post-hoc costs ~6.6 F1, but the **retrain-at-2K star recovers ~3.2 of those points** — confirming that aggressive pruning is partly a "retrain to compensate for OOV" problem.

**Multi-threading (extra finding):** BiLSTM-fp32 p50 latency: 1 thread = 2.34 ms; **2 threads = 1.90 ms (peak, ~1.23× speedup)**; 4 threads regresses; 12 threads = 2.45 ms (worse than 1 thread). int8 worsens to 5.80 ms at 12 threads (2.3× slower than 1 thread). **Recommendation: `OMP_NUM_THREADS=2` per worker; scale out via processes, not intra-op threads.**

## Bottom-left — Per-entity F1
**Insert `figures/per_entity_f1.png`.**

Conclusions:
- `PER` is easiest across all models (proper-noun cues are strong; all models ≥ 0.68 F1).
- `MISC` is hardest (heterogeneous, low frequency; HMM = 0.54, neural ≥ 0.73).
- DistilBERT-over-BiLSTM gain is largest on `MISC` (~6 F1 points) and `ORG` (~5), smallest on `LOC` (~3) — the entity types where transformer context (long-range disambiguation) helps most.
- **Surprising:** CRF beats BiLSTM-char on `MISC` (0.78 vs 0.73) — hand-crafted shape/casing features have an inductive bias for the heterogeneous MISC class that the small CharCNN doesn't recover.

## Bottom-center — 3-tier Recommendation (large)

| Latency budget | Use case | Recommended model |
|---|---|---|
| **5 ms** | real-time chat / streaming | **BiLSTM-char-int8** (F1=0.854, p50=3.14 ms) |
| **10 ms** | real-time tagging in API hot path | **BiLSTM-char-int8** (F1=0.854, p50=3.14 ms) |
| **50 ms** | batch / serverless | **BiLSTM-char-int8** (F1=0.854, p50=3.14 ms) |

The recommendation logic is in the final notebook cell and outputs `recommendation.json`.

## Bottom-right — Conclusions & limitations (~120 words)

**Conclusions** (each one citation-traceable to a CSV row)
- **Char features + GloVe + full data closed the OOV gap.** Fast version's word-only BiLSTM scored 0.539 (HMM-level); full version's char-aware BiLSTM scores **0.854** — a 31-point gain.
- **The Pareto frontier has 3 real points:** CRF (0.827, 0.19 ms), BiLSTM-char-int8 (0.854, 3.14 ms), DistilBERT (0.898, 66.7 ms). HMM is dominated. The fast version's plot was a vertical line; the full version's is a curve. CRF→BiLSTM is statistically significant (CIs do not overlap).
- **Quantization at this model size is a size win, not a speed win.** F1 within bootstrap noise; on-disk shrinks 24%; single-thread latency neutral. Useful for shipping to edge devices.
- **Batching is the throughput knob, not the latency knob.** ~2.3× peak at batch=16, declines past that; argmax decoding is batch-invariant in F1.
- **Vocab pruning is a knob, not a free lunch.** Pruning the 10,951-word table to 5K saves 54% of the table at ~2.1 F1; a single retrain at 2K recovers ~3 F1 points over post-hoc pruning at the same size.
- **Multi-threading peaks at 2 threads.** No further gain from 4; **regression** at 12. Don't naïvely use all cores — scale out via process-level concurrency.
- **DistilBERT misses every real-time tier.** 66.7 ms p50 fails even the 50 ms budget. CPU transformer NER needs distillation or async-batched inference.

**Limitations**
- Single dataset (CoNLL-2003); no domain-shift evaluation.
- No hyperparameter search per model.
- DistilBERT is off-the-shelf, not fine-tuned by us.
- No ONNX / TorchScript optimization for the BiLSTM.
