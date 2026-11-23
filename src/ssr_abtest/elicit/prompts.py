# SPDX-License-Identifier: Apache-2.0
"""Survey prompts that ask for prose rather than a number.

SSR needs a free-text reaction. Asking the model for a 1–5 score directly
is the direct Likert rating setup the paper found collapses toward the
middle of the scale.
"""

from __future__ import annotations

LANGUAGE_NAMES = {
    "ja": "Japanese",
    "en": "English",
}


def instructions_for(language: str) -> str:
    name = LANGUAGE_NAMES.get(language, language)
    return (
        f"You are role-playing one survey respondent. Answer in {name}. "
        "Write two to four sentences of natural speech about your reaction. "
        "Do not give a numeric score, a star rating, or a labelled scale point. "
        "Stay consistent with the demographic profile you are given. "
        "Do not mention that you are a language model."
    )


def user_prompt(
    *,
    attributes: dict[str, str],
    scenario: str,
    variant_id: str,
    variant_label: str,
    page_text: str,
    question: str,
    sample_index: int,
) -> str:
    lines = ["Respondent profile:"]
    for key, value in attributes.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            f"Situation: {scenario}",
            "",
            f"Web page variant {variant_id} ({variant_label}):",
            page_text.strip(),
            "",
            f"Question: {question}",
            "Answer in your own words.",
            f"Reply id: {sample_index + 1}",
        ]
    )
    return "\n".join(lines)
