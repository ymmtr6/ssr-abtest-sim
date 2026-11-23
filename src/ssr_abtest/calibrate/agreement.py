# SPDX-License-Identifier: Apache-2.0
"""Win agreement against labels from a past or public A/B result.

This is a scoring helper for a follow-up calibration set. It does not
download external archives and it does not adjust ε or temperature.
"""

from __future__ import annotations


def preferred_variant(mean_by_variant: dict[str, float], *, tolerance: float = 1e-9) -> str:
    if len(mean_by_variant) < 2:
        raise ValueError("need at least two variant means")
    ranking = sorted(mean_by_variant.items(), key=lambda item: (-item[1], item[0]))
    best_name, best_mean = ranking[0]
    second_mean = ranking[1][1]
    if abs(best_mean - second_mean) <= tolerance:
        return "tie"
    return best_name


def agreement_rate(predicted: list[str], labels: list[str]) -> dict[str, float | int]:
    """Fraction of items whose predicted winner equals the label.

    Ties in either series count as disagreements. Items are paired by order.
    """
    if len(predicted) != len(labels):
        raise ValueError("predicted and labels must have the same length")
    if not predicted:
        raise ValueError("at least one comparison is required")
    matches = sum(1 for guess, label in zip(predicted, labels) if guess == label and guess != "tie")
    return {
        "n": len(predicted),
        "matches": matches,
        "agreement": matches / len(predicted),
    }
