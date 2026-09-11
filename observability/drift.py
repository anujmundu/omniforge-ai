"""OmniForge Statistical Data and Concept Drift Detection Engine.

Implements Kolmogorov-Smirnov (KS) two-sample testing, Population Stability Index (PSI),
and categorical frequency shift calculations for continuous data quality and distribution monitoring.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from observability.base import (
    DatasetDriftReport,
    DriftMethod,
    FeatureDriftResult,
)


def compute_ks_statistic(
    sample1: Union[np.ndarray, List[float]], sample2: Union[np.ndarray, List[float]]
) -> Tuple[float, float]:
    """Compute two-sample Kolmogorov-Smirnov test statistic D and asymptotic p-value."""
    s1 = np.sort(np.asarray(sample1, dtype=float))
    s2 = np.sort(np.asarray(sample2, dtype=float))
    n1, n2 = len(s1), len(s2)

    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Unified grid of evaluation points
    all_vals = np.concatenate([s1, s2])
    all_vals.sort()

    # Empirical CDFs
    cdf1 = np.searchsorted(s1, all_vals, side="right") / n1
    cdf2 = np.searchsorted(s2, all_vals, side="right") / n2

    d_stat = float(np.max(np.abs(cdf1 - cdf2)))

    # Asymptotic Kolmogorov distribution p-value approximation
    en = math.sqrt((n1 * n2) / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d_stat

    # Two-tailed asymptotic Kolmogorov p-value
    if lam <= 0:
        p_val = 1.0
    else:
        # Sum of alternating series 2 * sum_{j=1..100} (-1)^(j-1) * exp(-2 j^2 lam^2)
        p_val = 0.0
        for j in range(1, 101):
            term = 2.0 * ((-1) ** (j - 1)) * math.exp(-2.0 * (j**2) * (lam**2))
            p_val += term
            if abs(term) < 1e-12:
                break
        p_val = max(0.0, min(1.0, float(p_val)))

    return round(d_stat, 4), round(p_val, 4)


def compute_psi(
    expected: Union[np.ndarray, List[Any]], actual: Union[np.ndarray, List[Any]], num_bins: int = 10
) -> float:
    """Compute Population Stability Index (PSI) between expected (reference) and actual (current)."""
    exp_arr = np.asarray(expected)
    act_arr = np.asarray(actual)

    if len(exp_arr) == 0 or len(act_arr) == 0:
        return 0.0

    # Numerical feature quantile binning
    if np.issubdtype(exp_arr.dtype, np.number) and np.issubdtype(act_arr.dtype, np.number):
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(exp_arr, percentiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        min_val = min(float(np.min(exp_arr)), float(np.min(act_arr))) - 1e-4
        max_val = max(float(np.max(exp_arr)), float(np.max(act_arr))) + 1e-4
        bins[0] = min_val
        bins[-1] = max_val

        exp_counts, _ = np.histogram(exp_arr, bins=bins)
        act_counts, _ = np.histogram(act_arr, bins=bins)
    else:
        # Categorical feature unique value frequency binning
        categories = list(set(exp_arr).union(set(act_arr)))
        exp_counts = np.array([np.sum(exp_arr == c) for c in categories])
        act_counts = np.array([np.sum(act_arr == c) for c in categories])

    exp_pct = (exp_counts + 1e-5) / (len(exp_arr) + 1e-5 * len(exp_counts))
    act_pct = (act_counts + 1e-5) / (len(act_arr) + 1e-5 * len(act_counts))

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return round(float(psi_val), 4)


class StatisticalDriftEngine:
