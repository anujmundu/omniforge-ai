"""Unit tests for KS-test and PSI statistical data drift calculations."""

import numpy as np
import pandas as pd

from observability.drift import StatisticalDriftEngine, compute_ks_statistic, compute_psi


def test_ks_statistic_identical_distributions():
    np.random.seed(42)
    s1 = np.random.normal(50, 10, 1000)
    s2 = np.random.normal(50, 10, 1000)

    d_stat, p_val = compute_ks_statistic(s1, s2)
    assert d_stat < 0.10
    assert p_val > 0.05


def test_ks_statistic_shifted_distribution():
    np.random.seed(42)
    s1 = np.random.normal(50, 10, 1000)
    s2 = np.random.normal(70, 10, 1000)

    d_stat, p_val = compute_ks_statistic(s1, s2)
    assert d_stat > 0.50
    assert p_val < 0.001


def test_psi_numerical_and_categorical():
    # Categorical identical
    exp_cat = ["A"] * 50 + ["B"] * 50
    act_cat = ["A"] * 48 + ["B"] * 52
    psi_cat = compute_psi(exp_cat, act_cat)
    assert psi_cat < 0.10

    # Categorical shifted
    act_cat_shifted = ["A"] * 90 + ["B"] * 10
    psi_shifted = compute_psi(exp_cat, act_cat_shifted)
    assert psi_shifted > 0.20


def test_dataset_drift_report_calculation():
    engine = StatisticalDriftEngine(significance_level=0.05, psi_threshold=0.20)
    np.random.seed(42)

    ref_df = pd.DataFrame(
        {
            "num_stable": np.random.normal(0, 1, 500),
            "num_drifted": np.random.normal(0, 1, 500),
            "cat_stable": ["low"] * 250 + ["high"] * 250,
        }
    )

    curr_df = pd.DataFrame(
        {
            "num_stable": np.random.normal(0, 1, 500),
            "num_drifted": np.random.normal(5, 1, 500),  # Heavy shift
            "cat_stable": ["low"] * 240 + ["high"] * 260,
        }
    )

    report = engine.calculate_dataset_drift(ref_df, curr_df, dataset_name="test_dataset", drift_share_threshold=0.30)
    assert report.number_of_features == 3
    assert report.drifted_features_count == 1
    assert report.feature_results["num_drifted"].drift_detected is True
    assert report.feature_results["num_stable"].drift_detected is False
    assert report.feature_results["cat_stable"].drift_detected is False
    assert report.drift_detected is True  # 1/3 = 33.3% >= 30%
