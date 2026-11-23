# SPDX-License-Identifier: Apache-2.0
"""Draw personas from independent marginal weights and assign variants.

Attributes are sampled independently. That is not a joint census draw:
associations between age, region, and income in a real population are not
reproduced. Variant assignment is a seeded shuffle of a balanced label
list, so arm sizes differ by at most one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Persona:
    id: str
    variant: str
    attributes: dict[str, str]


def sample_personas(
    attributes: list[dict],
    *,
    n: int,
    variants: list[str],
    seed: int,
) -> list[Persona]:
    if n < 1:
        raise ValueError("persona count must be >= 1")
    if not variants:
        raise ValueError("at least one variant is required")
    if n < len(variants):
        raise ValueError("persona count must be at least the number of variants")
    if not attributes:
        raise ValueError("at least one persona attribute is required")
    rng = np.random.default_rng(seed)
    drawn: list[dict[str, str]] = []
    for _ in range(n):
        profile: dict[str, str] = {}
        for attribute in attributes:
            labels = [str(level["label"]) for level in attribute["levels"]]
            weights = np.asarray([float(level["weight"]) for level in attribute["levels"]], dtype=float)
            if np.any(weights < 0) or float(weights.sum()) <= 0:
                raise ValueError(f"attribute {attribute['name']} has invalid weights")
            weights = weights / weights.sum()
            profile[str(attribute["name"])] = str(rng.choice(np.asarray(labels), p=weights))
        drawn.append(profile)
    labels = [variants[index % len(variants)] for index in range(n)]
    rng.shuffle(labels)
    return [
        Persona(id=f"p{index:04d}", variant=str(labels[index]), attributes=drawn[index])
        for index in range(n)
    ]
