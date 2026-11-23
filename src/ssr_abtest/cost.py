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
    # International z.ai list prices, USD per 1M tokens, from
    # https://docs.z.ai/guides/overview/pricing (checked 2026-11-23).
    "glm-5.3": {"input": 1.4, "output": 4.4, "currency": "USD"},
    "glm-5.3-flash": {"input": 0.15, "output": 0.50, "currency": "USD"},
    "glm-5.3-flashx": {"input": 0.37, "output": 1.25, "currency": "USD"},
    "glm-4.6v": {"input": 0.30, "output": 0.90, "currency": "USD"},
    "glm-4.6v-flashx": {"input": 0.04, "output": 0.40, "currency": "USD"},
    # China embedding-3 is priced in CNY, not USD. It is not added to total_usd.
    "embedding-3": {"input": 0.5, "output": 0.0, "currency": "CNY"},
    # Workers AI bge-m3 unit price from the model page, USD per 1M input tokens.
    "@cf/baai/bge-m3": {"input": 0.0118, "output": 0.0, "currency": "USD"},
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


def lookup_rate(
    model: str,
    overrides: dict[str, dict[str, float]] | None = None,
) -> dict[str, float | str] | None:
    """Longest-name match so ``glm-5.3-flash`` does not use the ``glm-5.3`` rate."""
    table: dict[str, dict[str, float | str]] = dict(DEFAULT_PRICES)
    if overrides:
        for name, rate in overrides.items():
            table[str(name)] = dict(rate)
    exact = table.get(model) or table.get(model.lower())
    if exact is not None:
        return exact
    lowered = model.lower()
    match: dict[str, float | str] | None = None
    match_len = -1
    for name, rate in table.items():
        key = name.lower()
        if lowered == key or lowered.startswith(key + "-"):
            if len(key) > match_len:
                match = rate
                match_len = len(key)
    return match


def quote_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    overrides: dict[str, dict[str, float]] | None = None,
) -> float | None:
    """Return a USD estimate, or None when the model has no USD list price.

    A missing model is not priced as zero. CNY rates, including China
    ``embedding-3``, also return None here so they are not added to a USD total.
    """
    rate = lookup_rate(model, overrides)
    if rate is None or "input" not in rate or "output" not in rate:
        return None
    if str(rate.get("currency", "USD")).upper() != "USD":
        return None
    usd = (
        input_tokens * float(rate["input"]) + output_tokens * float(rate["output"])
    ) / 1_000_000
    return round(usd, 6)


def _row(
    role: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    overrides: dict[str, dict[str, float]] | None,
) -> dict:
    rate = lookup_rate(model, overrides)
    currency = str(rate.get("currency", "USD")).upper() if rate else None
    priced = rate is not None and "input" in rate and "output" in rate
    amount = None
    if priced:
        amount = round(
            (input_tokens * float(rate["input"]) + output_tokens * float(rate["output"])) / 1_000_000,
            6,
        )
    usd = amount if priced and currency == "USD" else None
    return {
        "role": role,
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "usd": usd,
        "amount": amount,
        "currency": currency,
        "unpriced": not priced,
    }


def _total(rows: list[dict]) -> float | None:
    if any(row.get("unpriced") for row in rows):
        return None
    return round(sum(float(row["usd"] or 0.0) for row in rows if row.get("currency") == "USD"), 6)


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
    image_input_tokens: int = 0,
    image_notes: list[str] | None = None,
    image_unpriced: bool = False,
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
    if image_input_tokens or image_unpriced:
        notes.append(
            "Image input tokens follow the vision guide checked on "
            f"{PRICE_CATALOG_DATE}: 32px patches times the model multiplier "
            "for current models, or base-plus-512px-tiles for older vision models. "
            "detail high is the default. On gpt-5.4-mini, low can cost more than high. "
            "A full-page capture that has not been rendered uses an assumed height."
        )
    if image_notes:
        notes.extend(image_notes)
    notes.extend(_price_notes(rows))
    if image_unpriced:
        notes.append(
            "Image tokens could not be priced for the generation model, so the dollar total is omitted."
        )
    unpriced = any(row.get("unpriced") for row in rows)
    return {
        "basis": "estimated",
        "currency": "USD",
        "price_catalog_date": PRICE_CATALOG_DATE,
        "calls": n_generations,
        "by_model": rows,
        "image_input_tokens": int(image_input_tokens),
        "total_usd": None if image_unpriced or unpriced else _total(rows),
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
    notes.extend(_price_notes(rows))
    return {
        "basis": "api_usage",
        "currency": "USD",
        "price_catalog_date": PRICE_CATALOG_DATE,
        "by_model": rows,
        "total_usd": _total(rows),
        "notes": notes,
    }


def _price_notes(rows: list[dict]) -> list[str]:
    notes: list[str] = []
    missing = [row["model"] for row in rows if row.get("unpriced")]
    if missing:
        notes.append(
            "Warning: no list price for "
            + ", ".join(missing)
            + ". These models are not priced as zero. Set models.prices.<model>.input "
            "and .output in USD per 1,000,000 tokens."
        )
    foreign = [
        f"{row['model']} {row['amount']} {row['currency']}"
        for row in rows
        if row.get("currency") not in (None, "USD") and not row.get("unpriced")
    ]
    if foreign:
        notes.append("Non-USD rates are excluded from total_usd: " + ", ".join(foreign) + ".")
    return notes
