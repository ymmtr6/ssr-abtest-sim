# SPDX-License-Identifier: Apache-2.0
"""HTTP calls for Chat Completions, embeddings, and Workers AI runs.

The OpenAI SDK path is unchanged for direct Responses, embeddings, and
Decisions. This client is used when the role is z.ai, Cloudflare, or
OpenAI Chat Completions.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from ssr_abtest.cache import DiskCache, cache_key
from ssr_abtest.providers.base import EmbeddingBatch, Generation
from ssr_abtest.providers.openai_provider import _image_data_url, _usage_tokens
from ssr_abtest.providers.roles import RoleSettings, endpoint_url, request_headers


class Transport(Protocol):
    def post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]: ...


class UrllibTransport:
    def post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError(f"expected a JSON object from {url}")
        if data.get("success") is False:
            raise RuntimeError(f"request failed: {data.get('errors') or data.get('error')}")
        if isinstance(data.get("error"), dict):
            raise RuntimeError(str(data["error"].get("message") or data["error"]))
        return data


class HttpRoleClient:
    def __init__(self, role: RoleSettings, *, transport: Transport | None = None, cache: DiskCache | None = None):
        self.role = role
        self.transport = transport or UrllibTransport()
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
        image_path: str | None = None,
        image_detail: str = "high",
        image_tokens: int = 0,
    ) -> Generation:
        del image_detail, image_tokens
        model_name = model or self.role.model
        body = chat_body(
            self.role,
            model=model_name,
            instructions=instructions,
            user_input=user_input,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            image_path=image_path,
        )
        payload = {
            "op": "chat" if self.role.api == "chat_completions" else self.role.api,
            "sample_index": sample_index,
            "body": _cache_body(body),
            **self.role.identity(),
            "model": model_name,
        }
        cached = self._get(payload)
        if cached is not None:
            return Generation(
                text=cached["text"],
                input_tokens=int(cached["input_tokens"]),
                output_tokens=int(cached["output_tokens"]),
            )
        data = self._post(body)
        text = _chat_text(data)
        if not text.strip():
            raise RuntimeError(f"model {model_name} returned no text")
        input_tokens, output_tokens = _usage_tokens(data.get("usage"))
        result = Generation(text=text.strip(), input_tokens=input_tokens, output_tokens=output_tokens)
        self._set(payload, {"text": result.text, "input_tokens": input_tokens, "output_tokens": output_tokens})
        return result

    def embed(self, texts: list[str], *, model: str) -> EmbeddingBatch:
        model_name = model or self.role.model
        vectors: list[list[float] | None] = [None] * len(texts)
        billed = 0
        missing: list[int] = []
        for index, text in enumerate(texts):
            if not str(text).strip():
                raise ValueError("embeddings input must be a non-empty string")
            key = {
                "op": "embeddings",
                "text": text,
                "model": model_name,
                **self.role.identity(),
            }
            hit = self._get(key)
            if hit is not None:
                vectors[index] = list(hit["vector"])
            else:
                missing.append(index)
        step = max(1, self.role.batch_size)
        for start in range(0, len(missing), step):
            batch_ids = missing[start : start + step]
            batch_texts = [texts[index] for index in batch_ids]
            body = embed_body(self.role, model=model_name, texts=batch_texts)
            data = self._post(body)
            found, used = _embed_vectors(data, len(batch_texts))
            billed += used
            for offset, vector in enumerate(found):
                target = batch_ids[offset]
                vectors[target] = vector
                self._set(
                    {"op": "embeddings", "text": texts[target], "model": model_name, **self.role.identity()},
                    {"vector": vector, "input_tokens": 0},
                )
        if any(vector is None for vector in vectors):
            raise RuntimeError("the embeddings response did not cover every input")
        return EmbeddingBatch(
            vectors=[list(vector) for vector in vectors if vector is not None],
            input_tokens=billed,
        )

    def decide(self, **_kwargs: Any):
        from ssr_abtest.providers.roles import decisions_blocked_reason

        reason = decisions_blocked_reason(self.role) or "Decisions is not available for this role."
        raise RuntimeError(reason)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = endpoint_url(self.role)
        headers = request_headers(self.role)
        if self.role.api == "chat_completions" and "model" in body and self.role.provider == "cloudflare":
            body = dict(body)
            body["model"] = _cloudflare_model_name(self.role, str(body["model"]))
        return self.transport.post(url, headers, body)

    def _get(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if self.cache is None:
            return None
        return self.cache.get(cache_key(payload))

    def _set(self, payload: dict[str, Any], value: dict[str, Any]) -> None:
        if self.cache is not None:
            self.cache.set(cache_key(payload), value)


def chat_body(
    role: RoleSettings,
    *,
    model: str,
    instructions: str,
    user_input: str,
    temperature: float,
    max_output_tokens: int,
    image_path: str | None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if instructions.strip():
        messages.append({"role": "system", "content": instructions})
    if image_path:
        content: Any = [
            {"type": "text", "text": user_input},
            {"type": "image_url", "image_url": {"url": _image_data_url(Path(image_path))}},
        ]
    else:
        content = user_input
    messages.append({"role": "user", "content": content})
    token_field = "max_tokens" if role.provider == "zai" else "max_completion_tokens"
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        token_field: max_output_tokens,
    }


def embed_body(role: RoleSettings, *, model: str, texts: list[str]) -> dict[str, Any]:
    if role.api == "workers_ai_run":
        return {"text": texts}
    body: dict[str, Any] = {"model": model, "input": texts}
    if role.dimensions is not None:
        body["dimensions"] = role.dimensions
    return body


def _cloudflare_model_name(role: RoleSettings, model: str) -> str:
    """REST compat names are ``provider/model``. Native OpenAI gateway keeps the bare model."""
    if role.gateway_mode != "rest":
        return model
    if model.startswith("@cf/") or "/" in model:
        return model
    if role.custom_slug:
        slug = role.custom_slug if role.custom_slug.startswith("custom-") else "custom-" + role.custom_slug
        return f"{slug}/{model}"
    return f"openai/{model}"


def _chat_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        output = data.get("output_text")
        if isinstance(output, str):
            return output
        raise RuntimeError("chat response did not include choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
        return "".join(parts)
    return ""


def _embed_vectors(data: dict[str, Any], count: int) -> tuple[list[list[float]], int]:
    input_tokens, _output = _usage_tokens(data.get("usage"))
    result = data.get("result") if isinstance(data.get("result"), dict) else data
    rows = result.get("data")
    if not isinstance(rows, list):
        raise RuntimeError("embeddings response did not include data")
    if rows and isinstance(rows[0], list):
        if len(rows) != count:
            raise RuntimeError("embeddings response length does not match the input")
        return [[float(value) for value in row] for row in rows], input_tokens
    ordered: list[list[float] | None] = [None] * count
    for fallback, item in enumerate(rows):
        if not isinstance(item, dict) or "embedding" not in item:
            raise RuntimeError("embeddings response items have no embedding")
        index = int(item.get("index", fallback))
        ordered[index] = [float(value) for value in item["embedding"]]
    if any(item is None for item in ordered):
        raise RuntimeError("embeddings response did not cover every input")
    return [list(item) for item in ordered if item is not None], input_tokens


def _cache_body(body: dict[str, Any]) -> dict[str, Any]:
    """Drop raw image bytes from the local cache key and keep a marker instead."""
    cloned = json.loads(json.dumps({key: value for key, value in body.items() if key != "messages"}))
    messages = []
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            safe = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = str((part.get("image_url") or {}).get("url") or "")
                    digest = hashlib.sha256(url.encode()).hexdigest()
                    safe.append({"type": "image_url", "image_url": {"url": digest}})
                else:
                    safe.append(part)
            messages.append({**message, "content": safe})
        else:
            messages.append(message)
    cloned["messages"] = messages
    return cloned
