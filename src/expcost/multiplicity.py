"""Corrections across several metrics at one look.

This is a different problem from looking many times over the course of one
experiment, which is what peeking.py handles. Confusing the two is common, so
they are kept in separate modules and reported in separate tables.

  Holm-Bonferroni  controls the family wise error rate, the probability of one
                   or more false positives anywhere in the family. Appropriate
                   for a guardrail set, where a single false alarm is costly.
  Benjamini-Hochberg controls the false discovery rate, the expected share of
                   the rejections that are false. Appropriate for a wide
                   exploratory scan, where some false positives are tolerable
                   as long as most findings are real.
"""

from __future__ import annotations

import numpy as np


def holm_bonferroni(p_values, alpha: float = 0.05) -> np.ndarray:
    """Step down family wise error control. Returns the rejection mask."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = p.size
    rejected = np.zeros(m, dtype=bool)
    for rank, index in enumerate(order):
        if p[index] <= alpha / (m - rank):
            rejected[index] = True
        else:
            break
    return rejected


def benjamini_hochberg(p_values, alpha: float = 0.05) -> np.ndarray:
    """Step up false discovery rate control. Returns the rejection mask."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    passing = np.flatnonzero(p[order] <= thresholds)
    rejected = np.zeros(m, dtype=bool)
    if passing.size:
        rejected[order[: passing[-1] + 1]] = True
    return rejected


def adjusted_p_values(p_values, method: str = "holm") -> np.ndarray:
    """Adjusted p values, monotone within the family."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    out = np.empty(m)
    if method == "holm":
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, (m - rank) * p[index])
            out[index] = min(1.0, running)
    elif method == "bh":
        running = 1.0
        for rank in range(m - 1, -1, -1):
            index = order[rank]
            running = min(running, m / (rank + 1) * p[index])
            out[index] = min(1.0, running)
    else:
        raise ValueError(f"unknown method {method}")
    return out
