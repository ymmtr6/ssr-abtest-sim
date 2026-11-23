# SPDX-License-Identifier: Apache-2.0
"""Local stand-in used by ``--dry-run``.

Answers and embeddings are deterministic functions of the prompt. They do
not call the network and they are not model judgments. The report says so.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

from ssr_abtest.cost import estimate_tokens
from ssr_abtest.providers.base import DecisionResult, EmbeddingBatch, Generation

_PHRASES = {
    "ja": [
        "探している内容にかなり近く、このまま詳細を開いて比べたいです。",
        "一部は参考になりますが、決めるにはもう少し具体性がほしいです。",
        "今の目的にはあまり合っておらず、ボタンを押す気持ちにはなりません。",
    ],
    "en": [
        "This is close to what I am looking for, and I would open the details.",
        "Parts of this are useful, but I want something more specific before I continue.",
        "This does not match my goal, and I would not press the button.",
    ],
}


def hashed_embedding(text: str, dim: int = 64) -> list[float]:
    """Bag-of-tokens embedding. Shared wording lands in the same coordinates."""
    vector = np.zeros(dim, dtype=float)
    tokens = re.findall(r"\w+|[^\s\w]", text.lower(), flags=re.UNICODE)
    if not tokens:
        tokens = ["empty"]
    pieces = list(tokens)
    pieces.extend(f"{left} {right}" for left, right in zip(tokens, tokens[1:]))
    for token in pieces:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    if not np.any(vector):
        vector[0] = 1.0
    return vector.tolist()


class DryRunProvider:
    def __init__(self, language: str):
        self.language = language if language in _PHRASES else "en"

    def generate(
        self,
        *,
        model: str,
        instructions: str,
        user_input: str,
        temperature: float,
        max_output_tokens: int,
        sample_index: int,
        image_path: str | None = None,
        image_detail: str = "high",
        image_tokens: int = 0,
    ) -> Generation:
        del model, temperature, max_output_tokens, image_path, image_detail
        phrases = _PHRASES[self.language]
        material = f"{sample_index}\n{user_input}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        phrase = phrases[int(digest[:8], 16) % len(phrases)]
        snippet = " ".join(user_input.split())
        marker = "Web page variant"
        start = snippet.find(marker)
        excerpt = snippet[start : start + 180] if start >= 0 else snippet[:180]
        text = f"{phrase} ({excerpt})"
        return Generation(
            text=text,
            input_tokens=estimate_tokens(instructions) + estimate_tokens(user_input) + max(0, image_tokens),
            output_tokens=estimate_tokens(text),
        )

    def embed(self, texts: list[str], *, model: str) -> EmbeddingBatch:
        del model
        return EmbeddingBatch(
            vectors=[hashed_embedding(text) for text in texts],
            input_tokens=sum(estimate_tokens(text) for text in texts),
        )

    def decide(
        self,
        *,
        model: str,
        input_text: str,
        questions: list[dict[str, Any]],
    ) -> DecisionResult:
        """Neutral synthetic answers. Persona-break stays well below the default cutoff."""
        del model, input_text
        answers: list[dict[str, Any]] = []
        for question in questions:
            kind = question.get("type")
            name = question.get("name")
            if kind == "score":
                levels = question.get("levels") or []
                width = max(len(levels), 1)
                share = 1.0 / width
                answers.append(
                    {
                        "type": "score",
                        "name": name,
                        "score": (width - 1) / 2,
                        "confidence": 0.2,
                        "probabilities": [
                            {
                                "value": index,
                                "label": str(level.get("label", index + 1)),
                                "probability": share,
                            }
                            for index, level in enumerate(levels)
                        ],
                    }
                )
            elif kind == "predicate":
                probability = 0.05 if name == "persona_break" else 0.5
                answers.append({"type": "predicate", "name": name, "probability": probability})
            elif kind == "choice":
                choices = question.get("choices") or [{"value": "unset"}]
                answers.append(
                    {
                        "type": "choice",
                        "name": name,
                        "choice": choices[0].get("value"),
                        "confidence": 0.2,
                        "probabilities": [
                            {"value": choice.get("value"), "probability": 1.0 if i == 0 else 0.0}
                            for i, choice in enumerate(choices)
                        ],
                    }
                )
            else:
                answers.append({"type": "refusal", "name": name})
        return DecisionResult(
            payload={"model": "dry-run", "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}},
            input_tokens=0,
            output_tokens=0,
        )
