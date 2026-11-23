# SPDX-License-Identifier: Apache-2.0
"""Mean Likert differences, bootstrap intervals, and a two-sample KS test.

The unit of analysis is the persona. Pass one expected rating (and one
top-two-box probability) per persona. The difference is arm B minus arm A,
so a positive number means B has the higher stated intent.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import ks_2samp


def top2_box(pmf: np.ndarray) -> np.ndarray:
    """Probability of the two highest Likert points, per row."""
    array = np.asarray(pmf, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] < 2:
        raise ValueError("top-two box needs at least two Likert points")
    return array[:, -2:].sum(axis=1)


def cohens_d(arm_a: np.ndarray, arm_b: np.ndarray) -> float | None:
    """Pooled-sd Cohen's d for arm B minus arm A. None when a sd is undefined."""
    left = np.asarray(arm_a, dtype=float)
    right = np.asarray(arm_b, dtype=float)
    if left.size < 2 or right.size < 2:
        return None
    var_a = float(np.var(left, ddof=1))
    var_b = float(np.var(right, ddof=1))
    pooled = math.sqrt(
        ((left.size - 1) * var_a + (right.size - 1) * var_b) / (left.size + right.size - 2)
    )
    diff = float(right.mean() - left.mean())
    if pooled == 0.0:
        return 0.0 if diff == 0.0 else None
    return diff / pooled


def _bootstrap_diff(
    arm_a: np.ndarray,
    arm_b: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    draw_a = rng.integers(0, arm_a.size, size=(replicates, arm_a.size))
    draw_b = rng.integers(0, arm_b.size, size=(replicates, arm_b.size))
    return arm_b[draw_b].mean(axis=1) - arm_a[draw_a].mean(axis=1)


def compare_variants(
    ratings: dict[str, np.ndarray],
    top2: dict[str, np.ndarray] | None = None,
    *,
    replicates: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> list[dict]:
    """Pairwise comparison in variant-list order.

    Each value is a 1-d array of persona-level summaries for that arm.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    names = list(ratings)
    for name in names:
        if top2 is not None and len(top2[name]) != len(ratings[name]):
            raise ValueError(f"top2 and ratings differ in length for variant {name}")
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    comparisons: list[dict] = []
    for index, name_a in enumerate(names):
        for name_b in names[index + 1 :]:
            arm_a = np.asarray(ratings[name_a], dtype=float)
            arm_b = np.asarray(ratings[name_b], dtype=float)
            if arm_a.size == 0 or arm_b.size == 0:
                comparisons.append(
                    {
                        "a": name_a,
                        "b": name_b,
                        "n_a": int(arm_a.size),
                        "n_b": int(arm_b.size),
                        "error": "insufficient observations",
                    }
                )
                continue
            observed = float(arm_b.mean() - arm_a.mean())
            draws = _bootstrap_diff(arm_a, arm_b, replicates=replicates, rng=rng)
            low, high = (float(x) for x in np.quantile(draws, [alpha / 2, 1 - alpha / 2]))
            ks = ks_2samp(arm_a, arm_b, alternative="two-sided", method="auto")
            row: dict = {
                "a": name_a,
                "b": name_b,
                "n_a": int(arm_a.size),
                "n_b": int(arm_b.size),
                "mean_a": float(arm_a.mean()),
                "mean_b": float(arm_b.mean()),
                "mean_difference_b_minus_a": observed,
                "confidence": confidence,
                "bootstrap_replicates": replicates,
                "bootstrap_ci": [low, high],
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "cohens_d_b_minus_a": cohens_d(arm_a, arm_b),
            }
            if abs(observed) <= 1e-12:
                preferred = "tie"
            else:
                preferred = name_b if observed > 0 else name_a
            row["preferred_variant"] = preferred
            if low > 0:
                meaning = (
                    f"On this stated-intent measure, {name_b} is higher than {name_a} "
                    f"and the {confidence:.0%} bootstrap interval excludes zero."
                )
            elif high < 0:
                meaning = (
                    f"On this stated-intent measure, {name_a} is higher than {name_b} "
                    f"and the {confidence:.0%} bootstrap interval excludes zero."
                )
            else:
                meaning = (
                    "The bootstrap interval for the mean difference includes zero, "
                    "so this run does not separate the variants on this measure."
                )
            row["interpretation"] = meaning
            if top2 is not None:
                box_a = np.asarray(top2[name_a], dtype=float)
                box_b = np.asarray(top2[name_b], dtype=float)
                box_diff = float(box_b.mean() - box_a.mean())
                box_draws = _bootstrap_diff(box_a, box_b, replicates=replicates, rng=rng)
                box_low, box_high = (
                    float(x) for x in np.quantile(box_draws, [alpha / 2, 1 - alpha / 2])
                )
                row["top2_mean_a"] = float(box_a.mean())
                row["top2_mean_b"] = float(box_b.mean())
                row["top2_difference_b_minus_a"] = box_diff
                row["top2_bootstrap_ci"] = [box_low, box_high]
            comparisons.append(row)
    return comparisons
