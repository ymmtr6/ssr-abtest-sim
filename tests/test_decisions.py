# SPDX-License-Identifier: Apache-2.0
import pytest

from ssr_abtest.providers.decisions import (
    build_decision_questions,
    interpret_decision,
    score_to_likert,
)


def test_default_questions_match_the_documented_shapes():
    questions = build_decision_questions("このページは役に立ちますか？")
    kinds = [item["type"] for item in questions]
    assert kinds == ["score", "predicate", "predicate"]
    score = questions[0]
    assert score["name"] == "likert"
    assert [level["label"] for level in score["levels"]] == ["1", "2", "3", "4", "5"]
    assert {item["name"] for item in questions} == {"likert", "cta_intent", "persona_break"}


def test_score_values_are_zero_based_indices():
    answer = {
        "type": "score",
        "name": "likert",
        "score": 2.6,
        "confidence": 0.4,
        "probabilities": [
            {"value": 0, "label": "1", "probability": 0.1},
            {"value": 1, "label": "2", "probability": 0.1},
            {"value": 2, "label": "3", "probability": 0.2},
            {"value": 3, "label": "4", "probability": 0.3},
            {"value": 4, "label": "5", "probability": 0.3},
        ],
    }
    pmf, mean = score_to_likert(answer)
    assert pmf == pytest.approx([0.1, 0.1, 0.2, 0.3, 0.3])
    assert mean == pytest.approx(answer["score"] + 1)
    interpreted = interpret_decision({"answers": [answer]}, threshold=0.8)
    assert interpreted["likert_mean"] == pytest.approx(3.6)
    assert interpreted["exclude"] is False


def test_persona_break_excludes_and_refusal_does_not():
    excluded = interpret_decision(
        {
            "answers": [
                {"type": "refusal", "name": "likert"},
                {"type": "predicate", "name": "cta_intent", "probability": 0.2},
                {"type": "predicate", "name": "persona_break", "probability": 0.91},
            ]
        },
        threshold=0.8,
    )
    assert excluded["exclude"] is True
    assert excluded["exclude_reason"] == "persona_break"
    assert excluded["likert_mean"] is None
    assert "likert" in excluded["refusals"]
    kept = interpret_decision(
        {"answers": [{"type": "predicate", "name": "persona_break", "probability": 0.79}]},
        threshold=0.8,
    )
    assert kept["exclude"] is False


def test_out_of_range_score_value_is_an_error():
    with pytest.raises(ValueError):
        score_to_likert({"probabilities": [{"value": 5, "probability": 1.0}]})
