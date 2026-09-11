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
