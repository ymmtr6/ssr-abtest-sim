# SPDX-License-Identifier: Apache-2.0
from ssr_abtest.personas.sample import sample_personas

ATTRIBUTES = [
    {
        "name": "age",
        "levels": [
            {"label": "young", "weight": 0.25},
            {"label": "older", "weight": 0.75},
        ],
    },
    {
        "name": "region",
        "levels": [
            {"label": "north", "weight": 1.0},
            {"label": "south", "weight": 1.0},
        ],
    },
]


def test_sampling_is_reproducible_and_balanced():
    first = sample_personas(ATTRIBUTES, n=10, variants=["A", "B"], seed=7)
    second = sample_personas(ATTRIBUTES, n=10, variants=["A", "B"], seed=7)
    assert [(p.id, p.variant, p.attributes) for p in first] == [
        (p.id, p.variant, p.attributes) for p in second
    ]
    counts = {variant: sum(p.variant == variant for p in first) for variant in ("A", "B")}
    assert counts == {"A": 5, "B": 5}
    assert first[0].id == "p0000"
    assert set(first[0].attributes) == {"age", "region"}


def test_odd_counts_differ_by_one_and_weights_are_respected():
    personas = sample_personas(ATTRIBUTES, n=5, variants=["A", "B"], seed=1)
    counts = [sum(p.variant == variant for p in personas) for variant in ("A", "B")]
    assert sorted(counts) == [2, 3]
    large = sample_personas(ATTRIBUTES, n=4000, variants=["A"], seed=3)
    older = sum(p.attributes["age"] == "older" for p in large) / len(large)
    assert abs(older - 0.75) < 0.03

