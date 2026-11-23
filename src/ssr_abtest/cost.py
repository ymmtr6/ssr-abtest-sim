# SPDX-License-Identifier: Apache-2.0
"""Token heuristics and list-price estimates.

Rates are standard OpenAI list prices in USD per 1,000,000 tokens, checked
on 2026-11-23 against https://developers.openai.com/api/docs/pricing and the
model pages for the embedding models. They are short-context standard rates.
Batch, Flex, Fast, cached-input discounts, and the regional-processing
uplift are not applied.

``gpt-6-luna`` is the only Decisions model. Its model card prices output
tokens. Published Decisions responses often report ``output_tokens`` of 0,
and the planner assumes that. A measured run bills whatever usage the API
returns.
"""

from __future__ import annotations

PRICE_CATALOG_DATE = "2026-11-23"

# USD per 1,000,000 tokens.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-6.1-terra": {"input": 1.00, "output": 5.00},
    "gpt-6.2-sol": {"input": 2.00, "output": 10.00},
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


def estimate_tokens(text: str) -> int:
    """Approximate token count.

    Latin and other non-CJK characters count as about four per token.
    CJK characters count as one token each. The result is a planning
    figure, not a tokenizer count.
    """
    if not text:
        return 0
    cjk = 0
    for char in text:
        code = ord(char)
        if (
            0x2E80 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or 0xFF66 <= code <= 0xFF9D
        ):
            cjk += 1
    other = len(text) - cjk
    estimated = int(round(other / 4 + cjk))
    return max(1, estimated)


def quote_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    overrides: dict[str, dict[str, float]] | None = None,
) -> float | None:
    """Return a USD estimate, or None when the model has no listed rate."""
    table = dict(DEFAULT_PRICES)
    if overrides:
        for name, rate in overrides.items():
            table[name] = {key: float(value) for key, value in rate.items()}
    rate = table.get(model)
    if rate is None or "input" not in rate:
        return None
    output_rate = float(rate.get("output", 0.0))
    usd = (input_tokens * float(rate["input"]) + output_tokens * output_rate) / 1_000_000
    return round(usd, 6)


def _row(
    role: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    overrides: dict[str, dict[str, float]] | None,
) -> dict:
    usd = quote_usd(model, input_tokens, output_tokens, overrides)
    return {
        "role": role,
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "usd": usd,
    }


def _total(rows: list[dict]) -> float | None:
    if any(row["usd"] is None for row in rows):
        return None
    return round(sum(float(row["usd"]) for row in rows), 6)


def estimate_plan(
    *,
    generate_model: str,
    embed_model: str,
    decide_model: str,
    generation_input_tokens: int,
    n_generations: int,
    assumed_output_tokens: int,
    anchor_tokens: int,
    decision_input_tokens: int,
    decisions_enabled: bool,
    price_overrides: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Planning estimate used by dry-run and ``ssr-abtest estimate``."""
    rows = [
        _row(
            "generate",
            generate_model,
            generation_input_tokens,
            n_generations * assumed_output_tokens,
            price_overrides,
        ),
        _row(
            "embed",
            embed_model,
            anchor_tokens + n_generations * assumed_output_tokens,
            0,
            price_overrides,
        ),
    ]
    notes = [
        "Token counts are approximate: about four non-CJK characters per token, "
        "and one token per CJK character.",
        "Completion size uses assumed_output_tokens from the config.",
        f"Rates are standard list prices checked on {PRICE_CATALOG_DATE}. "
        "Batch, Flex, Fast, cached-input discounts, and regional processing "
        "are not included.",
        "The figure is the cost of a stated-intent screen, not a forecast of "
        "clicks, conversions, or revenue.",
    ]
    if decisions_enabled:
        rows.append(_row("decide", decide_model, decision_input_tokens, 0, price_overrides))
        notes.append(
            "Decisions output tokens are assumed to be 0, as in the published "
            "response examples. gpt-6-luna output tokens are billed when a "
            "measured response reports them."
        )
    missing = [row["model"] for row in rows if row["usd"] is None]
    if missing:
        notes.append(
            "No list price for: "
            + ", ".join(missing)
            + ". Set models.prices.<model>.input and .output in USD per 1,000,000 tokens."
        )
    return {
        "basis": "estimated",
        "currency": "USD",
        "price_catalog_date": PRICE_CATALOG_DATE,
        "calls": n_generations,
        "by_model": rows,
        "total_usd": _total(rows),
        "notes": notes,
    }


def cost_from_usage(
    *,
    generate_model: str,
    embed_model: str,
    decide_model: str,
    generation_input_tokens: int,
    generation_output_tokens: int,
    embedding_input_tokens: int,
    decision_input_tokens: int,
    decision_output_tokens: int,
    decisions_enabled: bool,
    price_overrides: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Price measured API usage. Output tokens are whatever the API reported."""
    rows = [
        _row(
            "generate",
            generate_model,
            generation_input_tokens,
            generation_output_tokens,
            price_overrides,
        ),
        _row("embed", embed_model, embedding_input_tokens, 0, price_overrides),
    ]
    notes = [
        "Token counts come from API usage fields on this run.",
        f"Rates are standard list prices checked on {PRICE_CATALOG_DATE}.",
    ]
    if decisions_enabled:
        rows.append(
            _row(
                "decide",
                decide_model,
                decision_input_tokens,
                decision_output_tokens,
                price_overrides,
            )
        )
    missing = [row["model"] for row in rows if row["usd"] is None]
    if missing:
        notes.append("No list price for: " + ", ".join(missing) + ".")
    return {
        "basis": "api_usage",
        "currency": "USD",
        "price_catalog_date": PRICE_CATALOG_DATE,
        "by_model": rows,
        "total_usd": _total(rows),
        "notes": notes,
    }
