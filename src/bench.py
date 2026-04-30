"""Benchmark utilities: timing, latency stats, bootstrap CI, RSS, file size."""
from __future__ import annotations

import gc
import os
import time
from typing import Callable, List, Sequence, Tuple

import numpy as np
import psutil


def time_predict(predict_fn: Callable, sentences: Sequence, warmup: int = 20) -> np.ndarray:
    """Per-sentence prediction timing in ms. Returns numpy array of length len(sentences).

    The first `warmup` sentences are timed but discarded — they handle JIT,
    cache warm-up, etc. After warm-up, every sentence is timed.
    """
    for s in sentences[:warmup]:
        predict_fn(s)
    times_ms = []
    for s in sentences:
        t0 = time.perf_counter()
        predict_fn(s)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times_ms)


def cold_vs_warm_latency(predict_fn: Callable, sentences: Sequence) -> Tuple[float, np.ndarray]:
    """First-call (cold) latency vs warm distribution (everything after).

    Useful for serverless deployments where the first request pays setup cost.
    """
    t0 = time.perf_counter()
    predict_fn(sentences[0])
    cold_ms = (time.perf_counter() - t0) * 1000.0
    warm = []
    for s in sentences[1:]:
        t0 = time.perf_counter()
        predict_fn(s)
        warm.append((time.perf_counter() - t0) * 1000.0)
    return cold_ms, np.array(warm)


def latency_summary(times_ms: np.ndarray, total_tokens: int) -> dict:
    return {
        "mean_ms": float(times_ms.mean()),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "throughput_tok_per_s": float(total_tokens / (times_ms.sum() / 1000.0)),
    }


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


def bootstrap_f1_ci(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    n_resamples: int = 1000,
    seed: int = 0,
):
    """Bootstrap 95% CI on span-F1 by resampling sentences with replacement.

    Returns (f1_point, low_95, high_95).
    """
    from seqeval.metrics import f1_score
    from seqeval.scheme import IOB2

    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = f1_score(y_true, y_pred, mode="strict", scheme=IOB2)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = [y_true[j] for j in idx]
        yp = [y_pred[j] for j in idx]
        samples[i] = f1_score(yt, yp, mode="strict", scheme=IOB2)
    low = float(np.percentile(samples, 2.5))
    high = float(np.percentile(samples, 97.5))
    return float(point), low, high


def rss_mb_now() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def measure_rss_isolated(target_fn: Callable[[], None]) -> float:
    """Run `target_fn` in a fresh subprocess (spawn) and return its peak RSS in MB.

    Falls back to in-process measurement if multiprocessing fails (Windows quirks).
    `target_fn` must be a top-level pickleable function.
    """
    import multiprocessing as mp
    from multiprocessing import get_context

    try:
        ctx = get_context("spawn")
        q = ctx.Queue()

        def _runner(q):
            try:
                target_fn()
                q.put(rss_mb_now())
            except Exception as e:
                q.put(f"ERR:{e!r}")

        p = ctx.Process(target=_runner, args=(q,))
        p.start()
        p.join(timeout=900)
        if p.is_alive():
            p.terminate()
            return -1.0
        try:
            v = q.get_nowait()
        except Exception:
            return -1.0
        if isinstance(v, str) and v.startswith("ERR:"):
            print(f"  isolated_rss subprocess error: {v}")
            return -1.0
        return float(v)
    except Exception as e:
        print(f"  multiprocessing unavailable ({e}); falling back to in-process RSS")
        rss_pre = rss_mb_now()
        target_fn()
        rss_post = rss_mb_now()
        return max(rss_pre, rss_post)


def gc_pause(seconds: float = 0.5) -> None:
    """Force a GC pass between models to reduce RSS pollution when subprocess is unavailable."""
    gc.collect()
    time.sleep(seconds)
