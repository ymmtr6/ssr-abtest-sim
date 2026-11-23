# SPDX-License-Identifier: Apache-2.0
"""Build one provider per role and route generate, embed, and decide."""

from __future__ import annotations

from typing import Any

from ssr_abtest.cache import DiskCache
from ssr_abtest.providers.base import DecisionResult, EmbeddingBatch, Generation
from ssr_abtest.providers.http_client import HttpRoleClient, Transport
from ssr_abtest.providers.openai_provider import OpenAIProvider
from ssr_abtest.providers.roles import RoleSettings, decisions_blocked_reason


class RoutedProvider:
    """Dispatch each role to its own client. The public methods match ``Provider``."""

    def __init__(self, generate: Any, embed: Any, decide: Any):
        self._generate = generate
        self._embed = embed
        self._decide = decide

    def generate(self, **kwargs: Any) -> Generation:
        return self._generate.generate(**kwargs)

    def embed(self, texts: list[str], *, model: str) -> EmbeddingBatch:
        return self._embed.embed(texts, model=model)

    def decide(self, **kwargs: Any) -> DecisionResult:
        return self._decide.decide(**kwargs)


def client_for(role: RoleSettings, *, cache: DiskCache | None, transport: Transport | None):
    if role.uses_openai_sdk:
        from openai import OpenAI

        if not role.api_key_env:
            raise RuntimeError(f"{role.role} needs api_key_env")
        import os

        key = os.environ.get(role.api_key_env, "")
        if not key:
            raise RuntimeError(f"environment variable {role.api_key_env} is not set")
        client = OpenAI(api_key=key)
        return OpenAIProvider(client=client, cache=cache, identity=role.identity())
    return HttpRoleClient(role, transport=transport, cache=cache)


def build_routed_provider(
    roles: dict[str, RoleSettings],
    *,
    cache: DiskCache | None = None,
    transport: Transport | None = None,
    decisions_enabled: bool = False,
) -> RoutedProvider | OpenAIProvider:
    decide = roles["decide"]
    if decisions_enabled:
        reason = decisions_blocked_reason(decide)
        if reason:
            raise RuntimeError(reason)
    if all(role.uses_openai_sdk for role in roles.values()) and _same_openai_client(roles):
        import os

        from openai import OpenAI

        key_env = roles["generate"].api_key_env or "OPENAI_API_KEY"
        key = os.environ.get(key_env, "")
        if not key:
            raise RuntimeError(
                f"{key_env} is not set. Re-run with --dry-run to execute the pipeline locally "
                "without calling the API."
            )
        return OpenAIProvider(client=OpenAI(api_key=key), cache=cache, identity=roles["generate"].identity())
    return RoutedProvider(
        generate=client_for(roles["generate"], cache=cache, transport=transport),
        embed=client_for(roles["embed"], cache=cache, transport=transport),
        decide=client_for(roles["decide"], cache=cache, transport=transport),
    )


def _same_openai_client(roles: dict[str, RoleSettings]) -> bool:
    bases = {role.public_base() for role in roles.values()}
    keys = {role.api_key_env for role in roles.values()}
    return len(bases) == 1 and len(keys) == 1
