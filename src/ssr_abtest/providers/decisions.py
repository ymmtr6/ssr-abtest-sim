# SPDX-License-Identifier: Apache-2.0
"""Optional OpenAI Decisions API adapter.

Checked against the Decisions guide and the create-decision reference
(https://developers.openai.com/api/docs/guides/decisions and
https://developers.openai.com/api/reference/resources/decisions) on
2026-11-23:

* ``POST /v1/decisions``
* the only model is ``gpt-6-luna``
* each question is a predicate, a choice, or a score (1 to 400 questions)
* a score answer has ``score``, ``confidence``, and
  ``probabilities[{value, label, probability}]``; ``value`` is a 0-based index
* a predicate answer has ``probability``
* any question may return ``type: "refusal"`` instead

The default experiment sends one 5-level score and two predicates. Choice
is a documented question type; ``OpenAIProvider.decide`` will forward a
choice question, and the default set does not include one. SSR stays the
primary endpoint. The score is an auxiliary reading of the same free-text
answer, closer to a probabilistic follow-up rating than to SSR.
"""

from __future__ import annotations

from typing import Any


def build_decision_questions(survey_prompt: str) -> list[dict[str, Any]]:
    """Questions asked about one free-text answer, not about the page itself."""
    return [
        {
            "type": "score",
            "name": "likert",
            "instructions": (
                "Rate how positively this answer expresses the attitude in the "
                f"survey question. Survey question: {survey_prompt}"
            ),
            "levels": [
                {"label": "1", "description": "Strongly negative, or not at all."},
                {"label": "2", "description": "Somewhat negative."},
                {"label": "3", "description": "Mixed, unsure, or moderate."},
                {"label": "4", "description": "Somewhat positive."},
                {"label": "5", "description": "Strongly positive."},
            ],
        },
        {
            "type": "predicate",
            "name": "cta_intent",
            "instructions": (
                "Does the respondent sound willing to press the page's main "
                "button or other primary call to action?"
            ),
        },
        {
            "type": "predicate",
            "name": "persona_break",
            "instructions": (
                "Does the answer refuse the task, say it is an AI or a language "
                "model, or clearly abandon the assigned respondent profile?"
            ),
        },
    ]


def score_to_likert(answer: dict[str, Any], n_levels: int = 5) -> tuple[list[float], float]:
    """Map a score answer onto a 1-based Likert mean.

    Level indices from the API are 0-based, so Likert point ``i + 1`` is
    ``value == i``. The mean is one plus the probability-weighted index.
    """
    probabilities = answer.get("probabilities") or []
    vector = [0.0] * n_levels
    for item in probabilities:
        value = int(item["value"])
        if value < 0 or value >= n_levels:
            raise ValueError(f"score value {value} is outside 0..{n_levels - 1}")
        vector[value] += float(item["probability"])
    total = sum(vector)
    if total <= 0:
        raise ValueError("score probabilities sum to 0")
    if abs(total - 1.0) > 1e-3:
        vector = [item / total for item in vector]
    mean = sum((index + 1) * weight for index, weight in enumerate(vector))
    return vector, mean


def interpret_decision(payload: dict[str, Any], *, threshold: float, n_levels: int = 5) -> dict[str, Any]:
    """Read the default question names out of a decisions response."""
    answers = list(payload.get("answers") or [])
    by_name = {answer.get("name"): answer for answer in answers if answer.get("name")}
    refusals = [answer.get("name") for answer in answers if answer.get("type") == "refusal"]
    result: dict[str, Any] = {
        "likert_pmf": None,
        "likert_mean": None,
        "api_score": None,
        "confidence": None,
        "cta_probability": None,
        "persona_break_probability": None,
        "refusals": [name for name in refusals if name],
        "exclude": False,
        "exclude_reason": None,
    }
    likert = by_name.get("likert")
    if likert and likert.get("type") == "score":
        pmf, mean = score_to_likert(likert, n_levels)
        result["likert_pmf"] = pmf
        result["likert_mean"] = mean
        if likert.get("score") is not None:
            result["api_score"] = float(likert["score"])
        if likert.get("confidence") is not None:
            result["confidence"] = float(likert["confidence"])
    cta = by_name.get("cta_intent")
    if cta and cta.get("type") == "predicate":
        result["cta_probability"] = float(cta["probability"])
    broken = by_name.get("persona_break")
    if broken and broken.get("type") == "predicate":
        probability = float(broken["probability"])
        result["persona_break_probability"] = probability
        if probability >= threshold:
            result["exclude"] = True
            result["exclude_reason"] = "persona_break"
    return result
