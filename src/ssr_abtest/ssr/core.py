# SPDX-License-Identifier: Apache-2.0
"""Map response embeddings to Likert distributions.

The transformation is Semantic Similarity Rating (SSR) from Maier et al.,
arXiv:2510.08338. Similarity is the scaled cosine used by the paper's
reference implementation:

    γ(u, v) = (1 + cosine(u, v)) / 2
    p(r) = (γ_r - γ_ℓ + ε δ_{ℓ,r}) / (Σ_s (γ_s - γ_ℓ) + ε)

γ_ℓ is the lowest similarity among the anchor statements and ℓ is its
index (the first index when several anchors tie). δ_{ℓ,r} is 1 only at
that index. The reported experiments in the paper fix ε = 0 and
temperature T = 1.

When ε = 0 the probabilities are unchanged by any positive affine
transform of γ, so raw cosine and (1 + cosine) / 2 match. ε > 0 is
applied in the scaled-cosine space, which is the reference convention.

Anchor sets are averaged with equal weight. Temperature scaling then
runs once on that average, ``p_r ← p_r ** (1 / T) / Z``, matching the
reference rater's ``mean`` reference set. T = 1 leaves the PMF unchanged.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def _as_2d(embeddings: np.ndarray) -> Array:
    array = np.asarray(embeddings, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("embeddings must have shape (n, dim) or (dim,)")
    return array


def scaled_cosine(responses: np.ndarray, anchors: np.ndarray) -> Array:
    """Return γ = (1 + cosine) / 2 with one row per response.

    ``anchors`` has shape (k, dim) in Likert order, point 1 first.
    """
    left = _as_2d(responses)
    right = _as_2d(anchors)
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"embedding dimensions differ: responses {left.shape[1]}, anchors {right.shape[1]}"
        )
    if left.shape[0] == 0 or right.shape[0] == 0:
        return np.empty((left.shape[0], right.shape[0]), dtype=float)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    if np.any(left_norm == 0) or np.any(right_norm == 0):
        raise ValueError("embedding vectors must be non-zero")
    cosine = (left / left_norm[:, None]) @ (right / right_norm[:, None]).T
    cosine = np.clip(cosine, -1.0, 1.0)
    return (1.0 + cosine) / 2.0


def similarities_to_pmf(similarities: np.ndarray, epsilon: float = 0.0) -> Array:
    """Convert a similarity matrix of shape (n, k) into row-wise PMFs.

    A row whose denominator is 0 (every anchor equally similar and ε = 0)
    becomes the uniform distribution. That is the limit of the SSR ratio
    as the similarities approach each other, and it avoids a 0/0 result.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    sim = np.asarray(similarities, dtype=float)
    if sim.ndim != 2:
        raise ValueError("similarities must have shape (n, k)")
    n_rows, n_points = sim.shape
    if n_rows == 0 or n_points == 0:
        return np.empty_like(sim)
    gamma_min = sim.min(axis=1, keepdims=True)
    numerator = sim - gamma_min
    if epsilon > 0:
        min_index = np.argmin(sim, axis=1)
        numerator[np.arange(n_rows), min_index] += epsilon
    denominator = sim.sum(axis=1, keepdims=True) - n_points * gamma_min + epsilon
    undefined = np.abs(denominator.ravel()) <= 1e-15
    pmf = np.empty_like(numerator)
    defined = ~undefined
    if np.any(defined):
        pmf[defined] = numerator[defined] / denominator[defined]
    if np.any(undefined):
        pmf[undefined] = 1.0 / n_points
    pmf = np.clip(pmf, 0.0, None)
    totals = pmf.sum(axis=1, keepdims=True)
    zero_total = totals.ravel() <= 0
    if np.any(~zero_total):
        pmf[~zero_total] = pmf[~zero_total] / totals[~zero_total]
    if np.any(zero_total):
        pmf[zero_total] = 1.0 / n_points
    return pmf


def scale_pmf(pmf: np.ndarray, temperature: float) -> Array:
    """Temperature-scale a PMF or a batch of PMFs (rows).

    Temperature 0 places all mass on the first maximum. A uniform row is
    left unchanged at temperature 0, because there is no unique mode.
    """
    if temperature < 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and >= 0")
    array = np.asarray(pmf, dtype=float)
    single = array.ndim == 1
    if single:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("pmf must have shape (k,) or (n, k)")
    n_points = array.shape[1]
    if temperature == 0.0:
        out = np.zeros_like(array)
        out[np.arange(array.shape[0]), np.argmax(array, axis=1)] = 1.0
        uniform = np.all(array == array[:, :1], axis=1)
        out[uniform] = array[uniform]
    else:
        out = array ** (1.0 / temperature)
    totals = out.sum(axis=1, keepdims=True)
    empty = totals.ravel() <= 0
    scaled = np.divide(out, totals, out=np.zeros_like(out), where=totals > 0)
    if np.any(empty):
        scaled[empty] = 1.0 / n_points
    return scaled[0] if single else scaled


def response_pmf(
    response_embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    *,
    epsilon: float = 0.0,
    temperature: float = 1.0,
) -> Array:
    """PMF over anchors for each response embedding."""
    pmf = similarities_to_pmf(
        scaled_cosine(response_embeddings, anchor_embeddings),
        epsilon=epsilon,
    )
    if temperature != 1.0:
        pmf = scale_pmf(pmf, temperature)
    return pmf


def combine_anchor_set_pmfs(per_set: np.ndarray, temperature: float = 1.0) -> Array:
    """Average anchor-set PMFs, then apply temperature to that average.

    This follows the reference rater: each set is converted with ε first,
    the sets are averaged with equal weight, and temperature scaling runs
    once on the average. ``per_set`` has shape (n_sets, n_responses, k).
    """
    stacked = np.asarray(per_set, dtype=float)
    if stacked.ndim != 3 or stacked.shape[0] == 0:
        raise ValueError("per_set must have shape (n_sets, n_responses, n_points)")
    mean = stacked.mean(axis=0)
    if temperature != 1.0:
        mean = scale_pmf(mean, temperature)
    return mean


def average_anchor_set_pmfs(
    response_embeddings: np.ndarray,
    anchor_sets: list[np.ndarray],
    *,
    epsilon: float = 0.0,
    temperature: float = 1.0,
) -> tuple[Array, Array]:
    """Mean PMF across anchor sets, plus the per-set PMFs before temperature.

    The returned per-set array has shape (n_sets, n_responses, n_points).
    Temperature is applied only to the averaged PMF.
    """
    if not anchor_sets:
        raise ValueError("at least one anchor set is required")
    widths = {int(np.asarray(anchors).shape[0]) for anchors in anchor_sets}
    if len(widths) != 1:
        raise ValueError("anchor sets must share the same number of Likert points")
    pmfs = [
        similarities_to_pmf(scaled_cosine(response_embeddings, anchors), epsilon=epsilon)
        for anchors in anchor_sets
    ]
    stacked = np.stack(pmfs, axis=0)
    return combine_anchor_set_pmfs(stacked, temperature), stacked


def expected_rating(pmf: np.ndarray, start: int = 1) -> Array:
    """Expected Likert score. Column 0 is ``start`` (1 for a 1–5 scale)."""
    array = np.asarray(pmf, dtype=float)
    points = np.arange(start, start + array.shape[-1], dtype=float)
    return array @ points
