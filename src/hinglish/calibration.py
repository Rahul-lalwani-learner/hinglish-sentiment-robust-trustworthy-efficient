"""Trustworthiness study: is the model's confidence honest?

- ECE (Expected Calibration Error): bucket predictions by confidence and
  average |accuracy - confidence| weighted by bucket size. 0 = perfectly honest.
- Reliability diagram: accuracy per confidence bucket vs the diagonal.
- Temperature scaling (Guo et al. 2017): divide logits by one scalar T fitted on
  the validation set. T > 1 cools an overconfident model; argmax (and so F1)
  never changes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def ece_score(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 10):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append(
            dict(
                lo=lo,
                hi=hi,
                count=int(m.sum()),
                confidence=float(conf[m].mean()) if m.any() else np.nan,
                accuracy=float(correct[m].mean()) if m.any() else np.nan,
            )
        )
    return rows


def nll(logits: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    return float(-log_softmax(logits / T, axis=1)[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(lambda t: nll(logits, y, t), bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return softmax(logits / T, axis=1)


def plot_reliability(ax, probs, y, title="", n_bins=10, color="#C0504D"):
    rows = [r for r in reliability_bins(probs, y, n_bins) if r["count"] > 0]
    width = 1.0 / n_bins
    centers = np.array([(r["lo"] + r["hi"]) / 2 for r in rows])
    accs = np.array([r["accuracy"] for r in rows])
    ax.bar(centers, accs, width=width * 0.9, color=color, alpha=0.85, label="Actual accuracy")
    ax.bar(centers, centers - accs, bottom=accs, width=width * 0.9,
           color="none", edgecolor="#555", hatch="//", label="Gap (over/under-confident)")
    ax.plot([0, 1], [0, 1], "--", color="#333", lw=1, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model says it is this sure")
    ax.set_ylabel("Actually right")
    ece = ece_score(probs, y)
    ax.set_title(f"{title}\nECE = {ece:.3f}")
    ax.legend(loc="upper left", fontsize=8)
    return ece
