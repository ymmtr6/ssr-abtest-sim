# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from ssr_abtest.ssr.core import (
    average_anchor_set_pmfs,
    combine_anchor_set_pmfs,
    expected_rating,
    response_pmf,
    scale_pmf,
    scaled_cosine,
    similarities_to_pmf,
)


def test_scaled_cosine_pmf_matches_hand_calculation():
    anchors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    responses = np.array([[1.0, 0.0]])
    gamma = scaled_cosine(responses, anchors)
    assert gamma[0] == pytest.approx([1.0, 0.5, 0.0])
    pmf = similarities_to_pmf(gamma, epsilon=0.0)
    assert pmf[0] == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert pmf.sum(axis=1) == pytest.approx([1.0])


def test_epsilon_is_added_only_at_the_first_minimum():
    gamma = np.array([[1.0, 0.5, 0.0]])
    pmf = similarities_to_pmf(gamma, epsilon=0.1)
    assert pmf[0] == pytest.approx([1 / 1.6, 0.5 / 1.6, 0.1 / 1.6])


def test_zero_epsilon_is_invariant_to_positive_affine_rescaling():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(6, 5))
    scaled = 0.5 * raw + 0.5
    assert similarities_to_pmf(raw, epsilon=0.0) == pytest.approx(
        similarities_to_pmf(scaled, epsilon=0.0)
    )


def test_equal_similarities_become_uniform():
    pmf = similarities_to_pmf(np.array([[0.4, 0.4, 0.4, 0.4]]), epsilon=0.0)
    assert pmf[0] == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_temperature_extremes():
    base = np.array([0.2, 0.3, 0.5])
    assert scale_pmf(base, 1.0) == pytest.approx(base)
    assert scale_pmf(base, 0.0) == pytest.approx([0.0, 0.0, 1.0])
    uniform = np.full(3, 1 / 3)
    assert scale_pmf(uniform, 0.0) == pytest.approx(uniform)
    flattened = scale_pmf(base, 100.0)
    assert flattened.max() < base.max()
    assert flattened.sum() == pytest.approx(1.0)


def test_temperature_and_zero_vectors_are_rejected():
    with pytest.raises(ValueError):
        scale_pmf(np.array([1.0, 0.0]), -1)
    with pytest.raises(ValueError):
        scaled_cosine(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        similarities_to_pmf(np.ones((1, 2)), epsilon=-0.1)


def test_anchor_sets_are_averaged_with_equal_weight():
    response = np.array([[1.0, 0.0]])
    positive = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    negative = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    mean, per_set = average_anchor_set_pmfs(response, [positive, negative], epsilon=0.0, temperature=1.0)
    assert per_set.shape == (2, 1, 5)
    assert mean == pytest.approx(per_set.mean(axis=0))
    assert mean.sum() == pytest.approx(1.0)
    # The response matches point 1 of the first set and point 5 of the second.
    assert float(expected_rating(per_set[0, 0])) < float(expected_rating(per_set[1, 0]))


def test_temperature_is_applied_after_the_anchor_sets_are_averaged():
    per_set = np.array(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.6, 0.4]],
        ]
    )
    mean = combine_anchor_set_pmfs(per_set, temperature=0.0)
    # The average is [0.5, 0.3, 0.2], so temperature 0 selects the first point.
    # Scaling each set first would have averaged a one-hot on point 1 with a
    # one-hot on point 2.
    assert mean[0] == pytest.approx([1.0, 0.0, 0.0])


def test_response_pmf_uses_likert_order_of_rows():
    response = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    anchors = np.eye(5)
    pmf = response_pmf(response, anchors, epsilon=0.0, temperature=1.0)
    assert int(np.argmax(pmf[0])) == 0
    assert pmf.sum() == pytest.approx(1.0)
