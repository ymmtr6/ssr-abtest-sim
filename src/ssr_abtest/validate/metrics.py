# SPDX-License-Identifier: Apache-2.0
"""Agreement, rank correlation, and calibration summaries against click rates.

A pair has a clear CTR winner when a two-sided two-proportion z-test rejects
equal rates. Likert means are not click rates. The calibration bins are an
association plot of predicted score against observed CTR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, spearmanr


@dataclass(frozen=True)
class ScoredPackage:
    package_id: str
    test_id: str
    headline: str
    impressions: int
    clicks: int
    ctr: float
    length: int
    scores: dict[str, float]


def two_proportion_z(clicks_a: int, impressions_a: int, clicks_b: int, impressions_b: int) -> float | None:
    if impressions_a <= 0 or impressions_b <= 0:
        return None
    rate_a = clicks_a / impressions_a
    rate_b = clicks_b / impressions_b
    pooled_clicks = clicks_a + clicks_b
    pooled_n = impressions_a + impressions_b
    pooled = pooled_clicks / pooled_n
    if pooled <= 0.0 or pooled >= 1.0:
        return 0.0 if rate_a == rate_b else None
    standard_error = math.sqrt(pooled * (1.0 - pooled) * (1.0 / impressions_a + 1.0 / impressions_b))
    if standard_error == 0.0:
        return 0.0
    return (rate_a - rate_b) / standard_error


def clear_pairs(
    packages: list[ScoredPackage],
    *,
    alpha: float,
) -> list[tuple[ScoredPackage, ScoredPackage, str]]:
    """Return (lower_id_package, higher_id_package, winner_id) for significant pairs."""
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    found: list[tuple[ScoredPackage, ScoredPackage, str]] = []
    ordered = sorted(packages, key=lambda package: package.package_id)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            z_value = two_proportion_z(left.clicks, left.impressions, right.clicks, right.impressions)
            if z_value is None or abs(z_value) <= critical:
                continue
            winner = left.package_id if left.ctr > right.ctr else right.package_id
            found.append((left, right, winner))
    return found


def _predict_winner(left: ScoredPackage, right: ScoredPackage, score_name: str) -> str | None:
    left_score = left.scores.get(score_name)
    right_score = right.scores.get(score_name)
    if left_score is None or right_score is None:
        return None
    if abs(left_score - right_score) <= 1e-12:
        return "tie"
    return left.package_id if left_score > right_score else right.package_id


def agreement_on_pairs(
    pairs: list[tuple[ScoredPackage, ScoredPackage, str]],
    score_name: str,
    *,
    replicates: int,
    seed: int,
    confidence: float,
    random_choices: dict[tuple[str, str], str] | None = None,
) -> dict:
    hits: list[float] = []
    considered = 0
    for left, right, winner in pairs:
        if random_choices is not None:
            key = (left.package_id, right.package_id)
            predicted = random_choices.get(key)
        else:
            predicted = _predict_winner(left, right, score_name)
        if predicted is None:
            continue
        considered += 1
        hits.append(1.0 if predicted == winner else 0.0)
    summary = _bootstrap_mean(hits, replicates=replicates, seed=seed, confidence=confidence)
    summary["n_clear_pairs"] = len(pairs)
    summary["n_scored_pairs"] = considered
    summary["matches"] = int(sum(hits))
    summary["agreement"] = summary["mean"]
    return summary


def random_choices_for(
    pairs: list[tuple[ScoredPackage, ScoredPackage, str]],
    seed: int,
) -> dict[tuple[str, str], str]:
    rng = np.random.default_rng(seed)
    choices: dict[tuple[str, str], str] = {}
    for left, right, _winner in pairs:
        pick = int(rng.integers(0, 2))
        choices[(left.package_id, right.package_id)] = left.package_id if pick == 0 else right.package_id
    return choices


def spearman_by_test(
    packages: list[ScoredPackage],
    score_name: str,
    *,
    replicates: int,
    seed: int,
    confidence: float,
    minimum_packages: int,
) -> dict:
    rhos: list[float] = []
    per_test: list[dict] = []
    by_test: dict[str, list[ScoredPackage]] = {}
    for package in packages:
        if score_name not in package.scores:
            continue
        by_test.setdefault(package.test_id, []).append(package)
    for test_id, rows in sorted(by_test.items()):
        if len(rows) < minimum_packages:
            continue
        predicted = [row.scores[score_name] for row in rows]
        observed = [row.ctr for row in rows]
        if len(set(predicted)) < 2 or len(set(observed)) < 2:
            continue
        result = spearmanr(predicted, observed)
        coefficient = getattr(result, "statistic", None)
        if coefficient is None:
            coefficient = result[0]
        if coefficient is None or not np.isfinite(coefficient):
            continue
        value = float(coefficient)
        rhos.append(value)
        per_test.append({"test_id": test_id, "n_packages": len(rows), "rho": value})
    summary = _bootstrap_mean(rhos, replicates=replicates, seed=seed, confidence=confidence)
    summary["minimum_packages"] = minimum_packages
    summary["per_test"] = per_test
    summary["mean_rho"] = summary["mean"]
    return summary


def calibration_bins(
    packages: list[ScoredPackage],
    score_name: str,
    *,
    edges: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
) -> dict:
    points = []
    for package in packages:
        if score_name not in package.scores:
            continue
        points.append(
            {
                "test_id": package.test_id,
                "package_id": package.package_id,
                "predicted": package.scores[score_name],
                "ctr": package.ctr,
                "impressions": package.impressions,
                "clicks": package.clicks,
            }
        )
    bins = []
    for low, high in zip(edges, edges[1:]):
        last = high == edges[-1]
        members = [
            point
            for point in points
            if (low <= point["predicted"] < high) or (last and point["predicted"] == high)
        ]
        if not members:
            bins.append(
                {
                    "predicted_low": low,
                    "predicted_high": high,
                    "n": 0,
                    "mean_predicted": None,
                    "mean_ctr": None,
                }
            )
            continue
        bins.append(
            {
                "predicted_low": low,
                "predicted_high": high,
                "n": len(members),
                "mean_predicted": float(np.mean([point["predicted"] for point in members])),
                "mean_ctr": float(np.mean([point["ctr"] for point in members])),
            }
        )
    return {"points": points, "bins": bins}


def _bootstrap_mean(values: list[float], *, replicates: int, seed: int, confidence: float) -> dict:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": None, "ci": None, "confidence": confidence, "bootstrap_replicates": replicates}
    if array.size == 1 or replicates < 1:
        value = float(array.mean())
        return {
            "n": int(array.size),
            "mean": value,
            "ci": [value, value],
            "confidence": confidence,
            "bootstrap_replicates": replicates,
        }
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(replicates, array.size), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    low, high = (float(x) for x in np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0]))
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "ci": [low, high],
        "confidence": confidence,
        "bootstrap_replicates": replicates,
    }
