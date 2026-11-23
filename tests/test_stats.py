# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from ssr_abtest.stats.compare import cohens_d, compare_variants, top2_box


def test_bootstrap_separates_constant_arms():
    rows = compare_variants(
        {"A": np.array([1.0, 1.0, 1.0, 1.0]), "B": np.array([2.0, 2.0, 2.0, 2.0])},
        replicates=200,
        seed=1,
        confidence=0.95,
    )
    row = rows[0]
    assert row["mean_difference_b_minus_a"] == pytest.approx(1.0)
    assert row["bootstrap_ci"] == pytest.approx([1.0, 1.0])
    assert row["ci_excludes_zero"] is True
    assert row["preferred_variant"] == "B"
    assert row["ks_statistic"] == pytest.approx(1.0)
    assert row["cohens_d_b_minus_a"] is None or row["cohens_d_b_minus_a"] > 0


def test_identical_samples_include_zero():
    ratings = np.array([3.0, 3.0, 4.0, 2.0])
    rows = compare_variants({"A": ratings, "B": ratings.copy()}, replicates=300, seed=2)
    row = rows[0]
    low, high = row["bootstrap_ci"]
    assert low <= 0 <= high
    assert row["ks_statistic"] == pytest.approx(0.0)
    assert cohens_d(ratings, ratings) == pytest.approx(0.0)


def test_top2_box_and_its_interval():
    pmf = np.array([[0.0, 0.0, 0.0, 0.25, 0.75], [0.5, 0.5, 0.0, 0.0, 0.0]])
    assert top2_box(pmf) == pytest.approx([1.0, 0.0])
    rows = compare_variants(
        {"A": np.array([2.0, 2.0, 2.0]), "B": np.array([4.0, 4.0, 4.0])},
        {"A": np.array([0.1, 0.1, 0.1]), "B": np.array([0.8, 0.8, 0.8])},
        replicates=100,
        seed=3,
    )
    assert rows[0]["top2_difference_b_minus_a"] == pytest.approx(0.7)
    assert rows[0]["top2_bootstrap_ci"][0] <= rows[0]["top2_bootstrap_ci"][1]


def test_cohens_d_sign():
    effect = cohens_d(np.array([1.0, 2.0, 1.5]), np.array([3.0, 4.0, 3.5]))
    assert effect is not None and effect > 0
    assert cohens_d(np.array([1.0]), np.array([2.0, 3.0])) is None
