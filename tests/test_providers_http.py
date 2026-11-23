# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from ssr_abtest.cache import cache_key
from ssr_abtest.cost import _row, quote_usd
from ssr_abtest.providers.http_client import HttpRoleClient
from ssr_abtest.providers.roles import (
    decisions_blocked_reason,
    endpoint_url,
    request_headers,
    role_from_mapping,
)
from ssr_abtest.render.images import write_placeholder_png


class _Transport:
    def __init__(self):
        self.calls: list[tuple[str, dict, dict]] = []

    def post(self, url, headers, body):
        self.calls.append((url, headers, body))
        if body.get("text") is not None or "input" in body:
            rows = body.get("text") or body.get("input")
            return {
                "data": [{"index": index, "embedding": [float(index), 0.5]} for index in range(len(rows))],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            }
        return {
            "choices": [{"message": {"content": "noted"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }


def test_zai_chat_url_headers_and_image_payload(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "zai-test-key")
    role = role_from_mapping(
        "generate",
        {
            "provider": "zai",
            "region": "international",
            "api": "chat_completions",
            "api_key_env": "ZAI_API_KEY",
            "model": "glm-5.3-flash",
        },
        fallback_model="unused",
    )
    image = tmp_path / "shot.png"
    write_placeholder_png(image)
    transport = _Transport()
    client = HttpRoleClient(role, transport=transport)
    result = client.generate(
        model="glm-5.3-flash",
        instructions="Stay in character.",
        user_input="What do you think?",
        temperature=0.5,
        max_output_tokens=80,
        sample_index=1,
        image_path=str(image),
    )
    assert result.text == "noted"
    url, headers, body = transport.calls[0]
    assert url == "https://api.z.ai/api/paas/v4/chat/completions"
    assert headers["Authorization"] == "Bearer zai-test-key"
    assert "cf-aig-cache-key" not in headers
    assert body["max_tokens"] == 80
    assert body["messages"][0] == {"role": "system", "content": "Stay in character."}
    user = body["messages"][1]["content"]
    assert user[0] == {"type": "text", "text": "What do you think?"}
    assert user[1]["type"] == "image_url"
    assert user[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_zai_embeddings_are_china_only_and_batched(monkeypatch):
    international = role_from_mapping(
        "embed",
        {"provider": "zai", "region": "international", "api": "embeddings", "model": "embedding-3"},
        fallback_model="embedding-3",
    )
    with pytest.raises(RuntimeError, match="China endpoint"):
        endpoint_url(international)
    monkeypatch.setenv("ZAI_API_KEY", "zai-test-key")
    role = role_from_mapping(
        "embed",
        {
            "provider": "zai",
            "region": "china",
            "api": "embeddings",
            "api_key_env": "ZAI_API_KEY",
            "model": "embedding-3",
            "batch_size": 1,
        },
        fallback_model="embedding-3",
    )
    assert role.batch_size == 1
    assert role.dimensions == 2048
    transport = _Transport()
    client = HttpRoleClient(role, transport=transport)
    batch = client.embed(["one", "two"], model="embedding-3")
    assert len(transport.calls) == 2
    assert transport.calls[0][0] == "https://open.bigmodel.cn/api/paas/v4/embeddings"
    assert transport.calls[0][2]["dimensions"] == 2048
    assert batch.vectors[0] == [0.0, 0.5]
    assert batch.vectors[1] == [0.0, 0.5]


def test_gateway_openai_skips_generate_cache_and_keeps_embeddings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("CF_AIG_TOKEN", "cf-test-token")
    monkeypatch.setenv("CF_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CF_GATEWAY_ID", "gw")
    generate = role_from_mapping(
        "generate",
        {
            "provider": "cloudflare",
            "gateway_mode": "native",
            "api": "responses",
            "model": "gpt-5.4-mini",
            "account_id": "${CF_ACCOUNT_ID}",
            "gateway_id": "${CF_GATEWAY_ID}",
            "api_key_env": "OPENAI_API_KEY",
            "gateway_auth_env": "CF_AIG_TOKEN",
        },
        fallback_model="gpt-5.4-mini",
    )
    embed = role_from_mapping(
        "embed",
        {
            "provider": "cloudflare",
            "gateway_mode": "native",
            "api": "embeddings",
            "model": "text-embedding-3-small",
            "account_id": "${CF_ACCOUNT_ID}",
            "gateway_id": "${CF_GATEWAY_ID}",
            "api_key_env": "OPENAI_API_KEY",
            "gateway_auth_env": "CF_AIG_TOKEN",
        },
        fallback_model="text-embedding-3-small",
    )
    assert endpoint_url(generate).endswith("/acct/gw/openai/responses")
    assert endpoint_url(embed).endswith("/acct/gw/openai/embeddings")
    generated_headers = request_headers(generate)
    embed_headers = request_headers(embed)
    assert generated_headers["cf-aig-skip-cache"] == "true"
    assert "cf-aig-skip-cache" not in embed_headers
    assert generated_headers["cf-aig-authorization"] == "Bearer cf-test-token"
    assert generated_headers["Authorization"] == "Bearer openai-test-key"
    assert "cf-aig-cache-key" not in generated_headers
    assert cache_key({"op": "x", **generate.identity()}) != cache_key({"op": "x", **embed.identity()})


def test_custom_zai_gateway_and_byok_headers(monkeypatch):
    monkeypatch.setenv("CF_AIG_TOKEN", "cf-test-token")
    role = role_from_mapping(
        "generate",
        {
            "provider": "cloudflare",
            "api": "chat_completions",
            "custom_slug": "zai",
            "model": "glm-5.3-flash",
            "account_id": "acct",
            "gateway_id": "gw",
            "gateway_auth_env": "CF_AIG_TOKEN",
            "byok": True,
            "skip_cache": True,
        },
        fallback_model="glm-5.3-flash",
    )
    assert endpoint_url(role) == (
        "https://gateway.ai.cloudflare.com/v1/acct/gw/custom-zai/api/paas/v4/chat/completions"
    )
    headers = request_headers(role)
    assert "Authorization" not in headers
    assert headers["cf-aig-authorization"] == "Bearer cf-test-token"
    assert headers["cf-aig-skip-cache"] == "true"
    with pytest.raises(ValueError, match="cf-aig-cache-key"):
        role_from_mapping(
            "generate",
            {"provider": "openai", "headers": {"cf-aig-cache-key": "shared"}},
            fallback_model="gpt-5.4-mini",
        )


def test_workers_ai_embed_body_and_rest_chat_model(monkeypatch):
    monkeypatch.setenv("CF_AIG_TOKEN", "cf-test-token")
    embed = role_from_mapping(
        "embed",
        {
            "provider": "cloudflare",
            "api": "workers_ai_run",
            "model": "@cf/baai/bge-m3",
            "account_id": "acct",
            "gateway_id": "gw",
            "api_key_env": "CF_AIG_TOKEN",
        },
        fallback_model="@cf/baai/bge-m3",
    )
    transport = _Transport()
    client = HttpRoleClient(embed, transport=transport)
    client.embed(["hello"], model="@cf/baai/bge-m3")
    url, headers, body = transport.calls[0]
    assert url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/baai/bge-m3"
    assert body == {"text": ["hello"]}
    assert headers["cf-aig-gateway-id"] == "gw"
    assert "cf-aig-authorization" not in headers
    assert headers["Authorization"] == "Bearer cf-test-token"
    generate = role_from_mapping(
        "generate",
        {
            "provider": "cloudflare",
            "gateway_mode": "rest",
            "api": "chat_completions",
            "model": "@cf/zai-org/glm-5.3",
            "account_id": "acct",
            "gateway_id": "gw",
            "api_key_env": "CF_AIG_TOKEN",
        },
        fallback_model="@cf/zai-org/glm-5.3",
    )
    HttpRoleClient(generate, transport=transport).generate(
        model="@cf/zai-org/glm-5.3",
        instructions="Stay in character.",
        user_input="Hello",
        temperature=0.2,
        max_output_tokens=16,
        sample_index=0,
    )
    rest_url, rest_headers, rest_body = transport.calls[1]
    assert rest_url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions"
    assert rest_body["model"] == "@cf/zai-org/glm-5.3"
    assert rest_headers["cf-aig-skip-cache"] == "true"
    assert decisions_blocked_reason(generate)


def test_prices_warn_instead_of_zero_and_keep_cny_out_of_usd():
    assert quote_usd("glm-5.3-flash", 1_000_000, 2_000_000) == pytest.approx(0.15 + 1.0)
    assert quote_usd("glm-5.3", 1_000_000, 0) == pytest.approx(1.4)
    assert quote_usd("unknown-model", 1_000_000, 1_000_000) is None
    row = _row("embed", "embedding-3", 1_000_000, 0, None)
    assert row["usd"] is None
    assert row["currency"] == "CNY"
    assert row["amount"] == pytest.approx(0.5)
    assert row["unpriced"] is False
    missing = _row("generate", "local-model", 10, 10, None)
    assert missing["usd"] is None
    assert missing["unpriced"] is True
