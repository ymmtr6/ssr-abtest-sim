# SPDX-License-Identifier: Apache-2.0
"""OpenAI Responses, Embeddings, and Decisions calls.

The client is constructed only when a real run needs it. Pass a fake client
in tests. ``OPENAI_API_KEY`` is read by the OpenAI SDK from the environment
and is never written to disk by this module.
"""

from __future__ import annotations

import os
from typing import Any

from ssr_abtest.cache import DiskCache, cache_key
from ssr_abtest.providers.base import DecisionResult, EmbeddingBatch, Generation

_EMBED_BATCH = 128


def _plain(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        return _plain(obj.model_dump())
    if isinstance(obj, dict):
        return {key: _plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(value) for value in obj]
    if hasattr(obj, "__dict__"):
        data = {key: value for key, value in vars(obj).items() if not key.startswith("_")}
        if data:
            return {key: _plain(value) for key, value in data.items()}
    return obj


def _usage_tokens(usage: Any) -> tuple[int, int]:
    data = _plain(usage) if usage is not None else {}
    if not isinstance(data, dict):
        return 0, 0
    input_tokens = data.get("input_tokens", data.get("prompt_tokens", 0)) or 0
    output_tokens = data.get("output_tokens", data.get("completion_tokens", 0)) or 0
    return int(input_tokens), int(output_tokens)


class OpenAIProvider:
    def __init__(self, client: Any | None = None, cache: DiskCache | None = None):
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Re-run with --dry-run to execute "
                    "the pipeline locally without calling the API."
                )
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.cache = cache

    def generate(
        self,
        *,
        model: str,
        instructions: str,
        user_input: str,
        temperature: float,
        max_output_tokens: int,
        sample_index: int,
    ) -> Generation:
        payload = {
            "op": "responses",
            "model": model,
            "instructions": instructions,
            "input": user_input,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "sample_index": sample_index,
        }
        key = cache_key(payload)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return Generation(
                    text=hit["text"],
                    input_tokens=int(hit["input_tokens"]),
                    output_tokens=int(hit["output_tokens"]),
                )
        request = {
            "model": model,
            "instructions": instructions,
            "input": user_input,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "text": {"format": {"type": "text"}},
        }
        try:
            response = self.client.responses.create(**request)
        except Exception as exc:
            # Some reasoning models reject temperature. Retry once without it
            # only when the error names that parameter.
            if "temperature" not in str(exc).lower():
                raise
            request.pop("temperature")
            response = self.client.responses.create(**request)
        text = getattr(response, "output_text", None)
        if not text:
            plain = _plain(response)
            text = plain.get("output_text") if isinstance(plain, dict) else None
        if not text or not str(text).strip():
            raise RuntimeError(
                f"model {model} returned no text. Increase max_output_tokens or "
                "check that the model can answer this prompt."
            )
        input_tokens, output_tokens = _usage_tokens(getattr(response, "usage", None))
        generation = Generation(text=str(text).strip(), input_tokens=input_tokens, output_tokens=output_tokens)
        if self.cache is not None:
            self.cache.set(
                key,
                {
                    "text": generation.text,
                    "input_tokens": generation.input_tokens,
                    "output_tokens": generation.output_tokens,
                },
            )
        return generation

    def embed(self, texts: list[str], *, model: str) -> EmbeddingBatch:
        vectors: list[list[float] | None] = [None] * len(texts)
        billed = 0
        missing: list[int] = []
        for index, text in enumerate(texts):
            if not str(text).strip():
                raise ValueError("embeddings input must be a non-empty string")
            key = cache_key({"op": "embeddings", "model": model, "text": text})
            hit = self.cache.get(key) if self.cache is not None else None
            if hit is not None:
                vectors[index] = list(hit["vector"])
            else:
                missing.append(index)
        for start in range(0, len(missing), _EMBED_BATCH):
            batch_ids = missing[start : start + _EMBED_BATCH]
            batch_texts = [texts[index] for index in batch_ids]
            response = self.client.embeddings.create(model=model, input=batch_texts)
            plain = _plain(response)
            items = plain["data"] if isinstance(plain, dict) else _plain(response.data)
            input_tokens, _output = _usage_tokens(plain.get("usage") if isinstance(plain, dict) else None)
            billed += input_tokens
            for fallback, item in enumerate(items):
                item_plain = item if isinstance(item, dict) else _plain(item)
                origin = int(item_plain.get("index", fallback))
                target = batch_ids[origin]
                vector = [float(value) for value in item_plain["embedding"]]
                vectors[target] = vector
                if self.cache is not None:
                    self.cache.set(
                        cache_key({"op": "embeddings", "model": model, "text": texts[target]}),
                        {"vector": vector, "input_tokens": 0},
                    )
        if any(vector is None for vector in vectors):
            raise RuntimeError("the embeddings response did not cover every input")
        return EmbeddingBatch(vectors=[list(vector) for vector in vectors if vector is not None], input_tokens=billed)

    def decide(
        self,
        *,
        model: str,
        input_text: str,
        questions: list[dict[str, Any]],
    ) -> DecisionResult:
        payload = {"op": "decisions", "model": model, "input": input_text, "questions": questions}
        key = cache_key(payload)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return DecisionResult(
                    payload=hit["payload"],
                    input_tokens=int(hit["input_tokens"]),
                    output_tokens=int(hit["output_tokens"]),
                )
        decisions = getattr(self.client, "decisions", None)
        if decisions is None or not hasattr(decisions, "create"):
            raise RuntimeError(
                "This OpenAI SDK has no client.decisions.create. Upgrade the openai "
                "package. The Decisions helper is optional and stays off unless enabled."
            )
        response = decisions.create(model=model, input=input_text, questions=questions)
        plain = _plain(response)
        if not isinstance(plain, dict) or "answers" not in plain:
            raise RuntimeError("Decisions response did not include answers")
        input_tokens, output_tokens = _usage_tokens(plain.get("usage"))
        result = DecisionResult(payload=plain, input_tokens=input_tokens, output_tokens=output_tokens)
        if self.cache is not None:
            self.cache.set(
                key,
                {
                    "payload": plain,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )
        return result
