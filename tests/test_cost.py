# SPDX-License-Identifier: Apache-2.0
from ssr_abtest.cost import estimate_plan, estimate_tokens, quote_usd


def test_token_heuristic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("あ") == 1
    assert estimate_tokens("あ" * 3 + "abcd") == 4


def test_quote_uses_list_prices_per_million():
    # gpt-5.4-mini: input 0.75, output 4.50 USD / 1M, checked 2026-11-23.
    assert quote_usd("gpt-5.4-mini", 1_000_000, 0) == 0.75
    assert quote_usd("gpt-5.4-mini", 0, 1_000_000) == 4.5
    assert quote_usd("text-embedding-3-small", 1_000_000, 0) == 0.02
    assert quote_usd("gpt-6-luna", 1_000_000, 1_000_000) == 0.6
    assert quote_usd("unknown-model", 10, 10) is None
    assert quote_usd("custom", 1_000_000, 0, {"custom": {"input": 3.0, "output": 1.0}}) == 3.0


def test_plan_adds_roles_and_leaves_unknown_models_unpriced():
    priced = estimate_plan(
        generate_model="gpt-5.4-mini",
        embed_model="text-embedding-3-small",
        decide_model="gpt-6-luna",
        generation_input_tokens=1_000_000,
        n_generations=1,
        assumed_output_tokens=0,
        anchor_tokens=0,
        decision_input_tokens=1_000_000,
        decisions_enabled=True,
    )
    assert priced["total_usd"] == 0.75 + 0.0 + 0.10
    assert priced["basis"] == "estimated"
    missing = estimate_plan(
        generate_model="local-model",
        embed_model="text-embedding-3-small",
        decide_model="gpt-6-luna",
        generation_input_tokens=10,
        n_generations=1,
        assumed_output_tokens=1,
        anchor_tokens=1,
        decision_input_tokens=0,
        decisions_enabled=False,
    )
    assert missing["total_usd"] is None
