# SPDX-License-Identifier: Apache-2.0
"""Per-role provider settings and request URLs.

API modes are ``responses``, ``chat_completions``, ``embeddings``, and
``workers_ai_run``. Decisions is not an API mode: it exists only on the
OpenAI API called directly, not on z.ai and not on Cloudflare AI Gateway.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

OPENAI_BASE = "https://api.openai.com/v1"
ZAI_INTERNATIONAL = "https://api.z.ai"
ZAI_CHINA = "https://open.bigmodel.cn"
GATEWAY_HOST = "https://gateway.ai.cloudflare.com"
CF_API_HOST = "https://api.cloudflare.com"

APIS = {"responses", "chat_completions", "embeddings", "workers_ai_run"}
PROVIDERS = {"openai", "zai", "cloudflare"}


@dataclass
class RoleSettings:
    role: str
    provider: str
    api: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = "OPENAI_API_KEY"
    headers: dict[str, str] = field(default_factory=dict)
    batch_size: int = 1
    account_id: str | None = None
    gateway_id: str | None = None
    gateway_auth_env: str | None = None
    byok: bool = False
    skip_cache: bool = False
    custom_slug: str | None = None
    region: str = "international"
    dimensions: int | None = None
    gateway_mode: str = "native"

    @property
    def uses_openai_sdk(self) -> bool:
        """Direct OpenAI Responses, embeddings, and Decisions stay on the SDK."""
        if self.provider != "openai" or self.gateway_mode != "native":
            return False
        if self.api not in {"responses", "embeddings"}:
            return False
        base = (self.base_url or OPENAI_BASE).rstrip("/")
        return base == OPENAI_BASE

    @property
    def supports_decisions(self) -> bool:
        return self.role == "decide" and self.uses_openai_sdk

    def identity(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "base_url": self.public_base(),
            "model": self.model,
            "api": self.api,
        }

    def public_base(self) -> str:
        if self.provider == "openai":
            return (self.base_url or OPENAI_BASE).rstrip("/")
        if self.provider == "zai":
            return (self.base_url or (ZAI_CHINA if self.region == "china" else ZAI_INTERNATIONAL)).rstrip("/")
        if self.gateway_mode == "rest":
            return f"{CF_API_HOST}/client/v4/accounts/{self.account_id or ''}"
        return f"{GATEWAY_HOST}/v1/{self.account_id or ''}/{self.gateway_id or ''}"


def expand_env(value: str, environ: Mapping[str, str] | None = None) -> str:
    """Replace ``${NAME}`` from the environment. The value is never logged here."""
    source = os.environ if environ is None else environ

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        found = source.get(name, "")
        if not found:
            raise RuntimeError(f"environment variable {name} is not set")
        return found

    return _ENV.sub(replace, value)


def default_roles(generate: str, embed: str, decide: str) -> dict[str, RoleSettings]:
    return {
        "generate": RoleSettings(
            role="generate",
            provider="openai",
            api="responses",
            model=generate,
            api_key_env="OPENAI_API_KEY",
            batch_size=1,
        ),
        "embed": RoleSettings(
            role="embed",
            provider="openai",
            api="embeddings",
            model=embed,
            api_key_env="OPENAI_API_KEY",
            batch_size=128,
        ),
        "decide": RoleSettings(
            role="decide",
            provider="openai",
            api="responses",
            model=decide,
            api_key_env="OPENAI_API_KEY",
            batch_size=1,
        ),
    }


def role_from_mapping(role: str, raw: dict, *, fallback_model: str) -> RoleSettings:
    provider = str(raw.get("provider") or "openai").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"{role}.provider must be openai, zai, or cloudflare")
    api = str(raw.get("api") or _default_api(role, provider)).strip().lower()
    if api not in APIS:
        raise ValueError(f"{role}.api must be one of {', '.join(sorted(APIS))}")
    region = str(raw.get("region") or "international").strip().lower()
    if region not in {"international", "china"}:
        raise ValueError(f"{role}.region must be international or china")
    gateway_mode = str(raw.get("gateway_mode") or "native").strip().lower()
    if gateway_mode not in {"native", "rest"}:
        raise ValueError(f"{role}.gateway_mode must be native or rest")
    headers = raw.get("headers") or {}
    if not isinstance(headers, dict):
        raise ValueError(f"{role}.headers must be a mapping")
    lowered = {str(key).lower() for key in headers}
    if "cf-aig-cache-key" in lowered:
        raise ValueError(
            f"{role}.headers must not set cf-aig-cache-key. "
            "A shared cache key would collapse repeated persona samples."
        )
    if "skip_cache" in raw:
        skip_cache = bool(raw["skip_cache"])
    else:
        skip_cache = provider == "cloudflare" and role == "generate"
    batch = raw.get("batch_size")
    if batch is None:
        batch = _default_batch(provider, api)
    dimensions = raw.get("dimensions")
    if dimensions is None and provider == "zai" and api == "embeddings":
        dimensions = 2048
    return RoleSettings(
        role=role,
        provider=provider,
        api=api,
        model=str(raw.get("model") or fallback_model),
        base_url=(str(raw["base_url"]).rstrip("/") if raw.get("base_url") else None),
        api_key_env=(str(raw["api_key_env"]) if raw.get("api_key_env") else None),
        headers={str(key): str(value) for key, value in headers.items()},
        batch_size=int(batch),
        account_id=(str(raw["account_id"]) if raw.get("account_id") else None),
        gateway_id=(str(raw["gateway_id"]) if raw.get("gateway_id") else None),
        gateway_auth_env=(str(raw["gateway_auth_env"]) if raw.get("gateway_auth_env") else None),
        byok=bool(raw.get("byok", False)),
        skip_cache=skip_cache,
        custom_slug=(str(raw["custom_slug"]) if raw.get("custom_slug") else None),
        region=region,
        dimensions=(int(dimensions) if dimensions is not None else None),
        gateway_mode=gateway_mode,
    )


def _default_api(role: str, provider: str) -> str:
    if role == "embed":
        return "workers_ai_run" if provider == "cloudflare" else "embeddings"
    if provider == "zai":
        return "chat_completions"
    return "responses"


def _default_batch(provider: str, api: str) -> int:
    if api == "embeddings" and provider == "zai":
        return 64
    if api == "embeddings":
        return 128
    if api == "workers_ai_run":
        # The bge-m3 docs show a list of texts and do not publish a maximum.
        return 32
    return 1


def endpoint_url(role: RoleSettings, environ: Mapping[str, str] | None = None) -> str:
    """Absolute URL for this role. Account and gateway ids may use ``${ENV}``."""
    account = expand_env(role.account_id, environ) if role.account_id else ""
    gateway = expand_env(role.gateway_id, environ) if role.gateway_id else ""
    zai_host = role.base_url or (ZAI_CHINA if role.region == "china" else ZAI_INTERNATIONAL)
    if role.provider == "zai" and role.api == "embeddings" and "open.bigmodel.cn" not in zai_host:
        raise RuntimeError(
            "embedding-3 is documented only on the China endpoint "
            "https://open.bigmodel.cn/api/paas/v4/embeddings, not on the international z.ai API. "
            "Use provider openai or workers_ai_run for embeddings, or set the embed role region to china."
        )
    if role.provider == "openai":
        root = (role.base_url or OPENAI_BASE).rstrip("/")
        return _join(root, _openai_suffix(role.api))
    if role.provider == "zai":
        if role.base_url:
            root = role.base_url.rstrip("/")
        else:
            host = ZAI_CHINA if role.region == "china" else ZAI_INTERNATIONAL
            root = f"{host}/api/paas/v4" if role.api != "responses" else f"{host}/api/v1"
        return _join(root, _openai_suffix(role.api) if role.api != "workers_ai_run" else "")
    if role.provider == "cloudflare":
        if not account or not gateway:
            raise RuntimeError(f"{role.role} on Cloudflare needs account_id and gateway_id")
        if role.api == "workers_ai_run" or role.gateway_mode == "rest":
            if role.api == "workers_ai_run":
                model = role.model.lstrip("/")
                return f"{CF_API_HOST}/client/v4/accounts/{account}/ai/run/{model}"
            if role.api == "embeddings":
                raise RuntimeError(
                    "Cloudflare's REST API has no /ai/v1/embeddings endpoint. "
                    "Use workers_ai_run for @cf/baai/bge-m3, or the native OpenAI gateway "
                    "base URL for OpenAI embeddings."
                )
            suffix = "chat/completions" if role.api == "chat_completions" else "responses"
            return f"{CF_API_HOST}/client/v4/accounts/{account}/ai/v1/{suffix}"
        if role.custom_slug:
            slug = role.custom_slug
            if not slug.startswith("custom-"):
                slug = "custom-" + slug
            if role.api == "responses":
                path = "api/v1/responses"
            elif role.api == "embeddings":
                path = "api/paas/v4/embeddings"
            else:
                path = "api/paas/v4/chat/completions"
            return f"{GATEWAY_HOST}/v1/{account}/{gateway}/{slug}/{path}"
        return f"{GATEWAY_HOST}/v1/{account}/{gateway}/openai/{_openai_suffix(role.api).lstrip('/')}"
    raise RuntimeError(f"unknown provider {role.provider}")


def _openai_suffix(api: str) -> str:
    if api == "responses":
        return "/responses"
    if api == "chat_completions":
        return "/chat/completions"
    if api == "embeddings":
        return "/embeddings"
    raise RuntimeError(f"{api} is not an OpenAI-compatible path")


def _join(root: str, suffix: str) -> str:
    if not suffix:
        return root
    return root.rstrip("/") + (suffix if suffix.startswith("/") else "/" + suffix)


def request_headers(role: RoleSettings, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    headers = {"Content-Type": "application/json"}
    for key, value in role.headers.items():
        if key.lower() == "cf-aig-cache-key":
            raise RuntimeError("refusing to send cf-aig-cache-key")
        headers[key] = expand_env(value, source)
    url = ""
    try:
        url = endpoint_url(role, source)
    except RuntimeError:
        url = role.public_base()
    if role.byok:
        headers.pop("Authorization", None)
    elif role.api_key_env:
        token = source.get(role.api_key_env, "")
        if not token:
            raise RuntimeError(f"environment variable {role.api_key_env} is not set")
        headers["Authorization"] = f"Bearer {token}"
    if role.gateway_auth_env and "gateway.ai.cloudflare.com" in url:
        token = source.get(role.gateway_auth_env, "")
        if not token:
            raise RuntimeError(f"environment variable {role.gateway_auth_env} is not set")
        headers["cf-aig-authorization"] = f"Bearer {token}"
    if "api.cloudflare.com" in url and role.gateway_id:
        headers["cf-aig-gateway-id"] = expand_env(role.gateway_id, source)
    if role.skip_cache:
        headers["cf-aig-skip-cache"] = "true"
    if any(key.lower() == "cf-aig-cache-key" for key in headers):
        raise RuntimeError("refusing to send cf-aig-cache-key")
    return headers


def decisions_blocked_reason(role: RoleSettings) -> str | None:
    if role.supports_decisions:
        return None
    if role.provider == "zai":
        return (
            "Decisions is not available on z.ai. Disable decisions, or set the decide "
            "role to provider openai on https://api.openai.com/v1."
        )
    if role.provider == "cloudflare":
        return (
            "Decisions is not documented on Cloudflare AI Gateway. Disable decisions, "
            "or set the decide role to provider openai on https://api.openai.com/v1."
        )
    return (
        "Decisions is only available when the decide role calls the OpenAI API directly. "
        "Disable decisions or change that role to provider openai without a gateway."
    )
