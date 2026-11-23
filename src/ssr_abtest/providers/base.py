# SPDX-License-Identifier: Apache-2.0
"""Shared result types for generation, embeddings, and decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass
class EmbeddingBatch:
    vectors: list[list[float]]
    input_tokens: int


@dataclass
class DecisionResult:
    payload: dict[str, Any]
    input_tokens: int
    output_tokens: int


class Provider(Protocol):
    def generate(
        self,
        *,
        model: str,
        instructions: str,
        user_input: str,
        temperature: float,
        max_output_tokens: int,
        sample_index: int,
    ) -> Generation: ...

    def embed(self, texts: list[str], *, model: str) -> EmbeddingBatch: ...

    def decide(
        self,
        *,
        model: str,
        input_text: str,
        questions: list[dict[str, Any]],
    ) -> DecisionResult: ...
