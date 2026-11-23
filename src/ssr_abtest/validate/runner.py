# SPDX-License-Identifier: Apache-2.0
"""Run SSR, direct Likert, and optional Decisions scores on sampled tests."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ssr_abtest.config import _load_anchors
from ssr_abtest.cost import estimate_plan, estimate_tokens, quote_usd
from ssr_abtest.elicit.prompts import instructions_for, user_prompt
from ssr_abtest.personas.sample import Persona, sample_personas
from ssr_abtest.providers.base import Generation, Provider
from ssr_abtest.providers.decisions import build_decision_questions, interpret_decision
from ssr_abtest.providers.dry_run import DryRunProvider
from ssr_abtest.ssr.core import average_anchor_set_pmfs, expected_rating
from ssr_abtest.validate.archive import (
    CITATION,
    HeadlineTest,
    build_tests,
    load_packages,
    sample_tests,
)
from ssr_abtest.validate.metrics import (
    ScoredPackage,
    agreement_on_pairs,
    calibration_bins,
    clear_pairs,
    random_choices_for,
    spearman_by_test,
)

QUESTION = "Would you click this headline to read the article?"
SCENARIO = "You are browsing a news site and deciding whether to open a story."
DLR_INSTRUCTIONS = (
    "You are role-playing one survey respondent. "
    "Reply with a single integer from 1 to 5 and no other text. "
    "1 means you would not click the headline. 5 means you would definitely click it."
)


@dataclass
class ValidateSettings:
    data_path: Path
    n: int
    seed: int
    dry_run: bool
    n_personas: int
    samples_per_persona: int
    min_impressions: int
    max_packages: int
    same_image: bool
    include_imbalance_window: bool
    alpha: float
    confidence: float
    bootstrap_replicates: int
    personas_path: Path
    anchors_path: Path
    generate_model: str
    embed_model: str
    decide_model: str
    temperature: float
    decisions: bool
    budget_usd: float | None
    image_dir: Path | None
    output_dir: Path


def load_attributes(path: Path) -> tuple[list[dict], str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("attributes"):
        raise ValueError(f"{path} needs an attributes list")
    return data["attributes"], str(data.get("source_note") or "")


def package_text(package) -> str:
    lines = [f"Headline: {package.headline}"]
    if package.excerpt:
        lines.append(f"Excerpt: {package.excerpt}")
    return "\n".join(lines)


def image_for(package, image_dir: Path | None) -> Path | None:
    if image_dir is None or not package.eyecatcher_id:
        return None
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = image_dir / f"{package.eyecatcher_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def parse_direct_likert(text: str) -> float | None:
    match = re.search(r"\b([1-5])\b", text)
    if match is None:
        return None
    return float(match.group(1))


def _child_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _assign(personas: list[Persona], package_ids: list[str], seed: int) -> dict[str, list[Persona]]:
    labels = [package_ids[index % len(package_ids)] for index in range(len(personas))]
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    assigned: dict[str, list[Persona]] = {package_id: [] for package_id in package_ids}
    for persona, label in zip(personas, labels):
        assigned[label].append(persona)
    return assigned


class _DryRunRatings(DryRunProvider):
    """Dry-run answers. Direct Likert prompts return a digit so the metric path runs."""

    def generate(self, *, instructions: str, user_input: str, sample_index: int, **kwargs):
        if "single integer" in instructions:
            digest = hashlib.sha256(f"{sample_index}:{user_input}".encode()).hexdigest()
            digit = str(int(digest[:2], 16) % 5 + 1)
            return Generation(
                text=digit,
                input_tokens=estimate_tokens(instructions) + estimate_tokens(user_input),
                output_tokens=1,
            )
        return super().generate(
            instructions=instructions,
            user_input=user_input,
            sample_index=sample_index,
            **kwargs,
        )


def estimate_validation(
    tests: list[HeadlineTest],
    settings: ValidateSettings,
    *,
    prompt_tokens: int,
    dlr_prompt_tokens: int,
) -> dict:
    """Planning cost for the sampled tests. One SSR call and one direct-Likert call per sample."""
    generations = 0
    ssr_input = 0
    dlr_input = 0
    for _test in tests:
        # Each persona is assigned to one package, once per sample.
        calls = settings.n_personas * settings.samples_per_persona
        generations += calls
        ssr_input += calls * prompt_tokens
        dlr_input += calls * dlr_prompt_tokens
    assumed_ssr_out = 80
    assumed_dlr_out = 8
    # Direct Likert is a second generation on the same model.
    plan = estimate_plan(
        generate_model=settings.generate_model,
        embed_model=settings.embed_model,
        decide_model=settings.decide_model,
        generation_input_tokens=ssr_input + dlr_input,
        n_generations=generations * 2,
        assumed_output_tokens=(assumed_ssr_out + assumed_dlr_out) // 2,
        anchor_tokens=200,
        decision_input_tokens=(generations * 400) if settings.decisions else 0,
        decisions_enabled=settings.decisions,
    )
    # The helper prices n_generations * assumed_output_tokens. Override the
    # generate output with the split assumption so SSR and direct Likert differ.
    generate = plan["by_model"][0]
    output_tokens = generations * assumed_ssr_out + generations * assumed_dlr_out
    generate["output_tokens"] = output_tokens
    generate["input_tokens"] = ssr_input + dlr_input
    generate["usd"] = quote_usd(
        settings.generate_model,
        generate["input_tokens"],
        output_tokens,
    )
    embed = plan["by_model"][1]
    embed["input_tokens"] = generations * assumed_ssr_out + 200
    embed["usd"] = quote_usd(settings.embed_model, embed["input_tokens"], 0)
    if settings.decisions:
        decide = plan["by_model"][2]
        decide["usd"] = quote_usd(settings.decide_model, decide["input_tokens"], 0)
    priced = [row["usd"] for row in plan["by_model"]]
    plan["total_usd"] = None if any(value is None for value in priced) else round(sum(priced), 6)
    plan["n_tests"] = len(tests)
    plan["ssr_calls"] = generations
    plan["direct_likert_calls"] = generations
    plan["notes"].append(
        "Validation prices one free-text answer and one direct 1–5 rating per persona sample. "
        "Headline text is short, so this uses a fixed prompt-size assumption from the sampled tests' "
        "average prompt, not a full page."
    )
    return plan


def _representative_prompt_tokens(settings: ValidateSettings, tests: list[HeadlineTest], attributes: list[dict]) -> tuple[int, int]:
    if not tests:
        return 0, 0
    package = tests[0].packages[0]
    persona = sample_personas(attributes, n=1, variants=["panel"], seed=settings.seed)[0]
    ssr = user_prompt(
        attributes=persona.attributes,
        scenario=SCENARIO,
        variant_id=package.package_id,
        variant_label="headline",
        page_text=package_text(package),
        question=QUESTION,
    )
    dlr = ssr + "\nReply with one integer from 1 to 5."
    return (
        estimate_tokens(instructions_for("en")) + estimate_tokens(ssr),
        estimate_tokens(DLR_INSTRUCTIONS) + estimate_tokens(dlr),
    )


def _score_key(name: str) -> str:
    if name == "headline_length":
        return "length"
    return name


def _score_method(packages: list[ScoredPackage], name: str, settings: ValidateSettings, pairs, random_map) -> dict:
    key = _score_key(name)
    return {
        "agreement": agreement_on_pairs(
            pairs,
            key,
            replicates=settings.bootstrap_replicates,
            seed=settings.seed,
            confidence=settings.confidence,
            random_choices=random_map if name == "random" else None,
        ),
        "spearman": None
        if name == "random"
        else spearman_by_test(
            packages,
            key,
            replicates=settings.bootstrap_replicates,
            seed=settings.seed + 1,
            confidence=settings.confidence,
            minimum_packages=2,
        ),
        "spearman_at_least_3": None
        if name == "random"
        else spearman_by_test(
            packages,
            key,
            replicates=settings.bootstrap_replicates,
            seed=settings.seed + 2,
            confidence=settings.confidence,
            minimum_packages=3,
        ),
        "calibration": None if name == "random" else calibration_bins(packages, key),
    }


def validate_archive(settings: ValidateSettings, provider: Provider | None = None) -> dict:
    packages = load_packages(settings.data_path)
    frame = build_tests(
        packages,
        min_impressions=settings.min_impressions,
        max_packages=min(settings.max_packages, settings.n_personas),
        same_image=settings.same_image,
        include_imbalance_window=settings.include_imbalance_window,
    )
    selected = sample_tests(frame, settings.n, settings.seed)
    attributes, source_note = load_attributes(settings.personas_path)
    if settings.n_personas < 2:
        raise ValueError("n_personas must be at least 2")
    panel = sample_personas(attributes, n=settings.n_personas, variants=["panel"], seed=settings.seed)
    anchor_sets = _load_anchors(settings.anchors_path)[0]
    ssr_tokens, dlr_tokens = _representative_prompt_tokens(settings, selected, attributes)
    # Scale the single-example token count across calls inside estimate_validation
    # by passing the per-call size.
    cost = estimate_validation(selected, settings, prompt_tokens=ssr_tokens, dlr_prompt_tokens=dlr_tokens)
    if (
        not settings.dry_run
        and settings.budget_usd is not None
        and cost["total_usd"] is not None
        and cost["total_usd"] > settings.budget_usd
    ):
        raise RuntimeError(
            f"estimated cost ${cost['total_usd']:.4f} exceeds the budget cap ${settings.budget_usd:.4f}. "
            "No API calls were made. Lower --n or raise --budget."
        )
    if provider is None:
        if settings.dry_run:
            provider = _DryRunRatings("en")
        else:
            from ssr_abtest.cache import DiskCache
            from ssr_abtest.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(cache=DiskCache(settings.output_dir / "cache"))
    scored, spent, stopped = _run_tests(
        selected,
        settings,
        provider,
        panel,
        anchor_sets,
        cost,
    )
    pairs = []
    for test in selected:
        rows = [row for row in scored if row.test_id == test.test_id]
        pairs.extend(clear_pairs(rows, alpha=settings.alpha))
    random_map = random_choices_for(pairs, settings.seed + 17)
    methods = {}
    for name in ("ssr", "direct_likert", "headline_length", "random"):
        methods[name] = _score_method(scored, name, settings, pairs, random_map)
    if settings.decisions:
        methods["decisions"] = _score_method(scored, "decisions", settings, pairs, random_map)
    else:
        methods["decisions"] = None
    return {
        "dry_run": settings.dry_run,
        "dataset": {
            **CITATION,
            "file": str(settings.data_path),
            "n_packages_loaded": len(packages),
            "n_tests_available": len(frame),
            "n_tests_requested": settings.n,
            "n_tests_sampled": len(selected),
            "same_image": settings.same_image,
            "min_impressions": settings.min_impressions,
            "max_packages": settings.max_packages,
            "include_imbalance_window": settings.include_imbalance_window,
            "persona_source_note": source_note,
        },
        "seed": settings.seed,
        "question": QUESTION,
        "models": {
            "generate": settings.generate_model,
            "embed": settings.embed_model,
            "decide": settings.decide_model,
            "decisions": settings.decisions,
        },
        "cost": cost,
        "budget_usd": settings.budget_usd,
        "spent_usd": round(spent, 6),
        "stopped_early": stopped,
        "alpha": settings.alpha,
        "n_clear_pairs": len(pairs),
        "methods": methods,
        "packages": [
            {
                "test_id": row.test_id,
                "package_id": row.package_id,
                "headline": row.headline,
                "impressions": row.impressions,
                "clicks": row.clicks,
                "ctr": row.ctr,
                "length": row.length,
                "scores": row.scores,
            }
            for row in scored
        ],
    }


def _run_tests(selected, settings: ValidateSettings, provider: Provider, panel, anchor_sets, cost: dict):
    scored: list[ScoredPackage] = []
    spent = 0.0
    stopped = False
    per_test_estimate = (cost["total_usd"] or 0.0) / max(len(selected), 1)
    instructions = instructions_for("en")
    anchor_texts = [statement for anchor_set in anchor_sets for statement in anchor_set.statements]
    for test in selected:
        if (
            not settings.dry_run
            and settings.budget_usd is not None
            and spent + per_test_estimate > settings.budget_usd
        ):
            stopped = True
            break
        package_ids = [package.package_id for package in test.packages]
        assigned = _assign(panel, package_ids, _child_seed(settings.seed, test.test_id))
        responses: list[dict] = []
        for package in test.packages:
            page = package_text(package)
            attached = image_for(package, settings.image_dir)
            for persona in assigned[package.package_id]:
                for sample_index in range(settings.samples_per_persona):
                    ssr_prompt = user_prompt(
                        attributes=persona.attributes,
                        scenario=SCENARIO,
                        variant_id=package.package_id,
                        variant_label="headline",
                        page_text=page,
                        question=QUESTION,
                        attach_image=attached is not None,
                    )
                    generated = provider.generate(
                        model=settings.generate_model,
                        instructions=instructions,
                        user_input=ssr_prompt,
                        temperature=settings.temperature,
                        max_output_tokens=240,
                        sample_index=sample_index,
                        image_path=str(attached) if attached else None,
                        image_detail="high",
                    )
                    if not settings.dry_run:
                        spent += _call_cost(settings.generate_model, generated.input_tokens, generated.output_tokens)
                    dlr = provider.generate(
                        model=settings.generate_model,
                        instructions=DLR_INSTRUCTIONS,
                        user_input=ssr_prompt,
                        temperature=0.0,
                        max_output_tokens=16,
                        sample_index=sample_index,
                        image_path=str(attached) if attached else None,
                        image_detail="high",
                    )
                    if not settings.dry_run:
                        spent += _call_cost(settings.generate_model, dlr.input_tokens, dlr.output_tokens)
                    responses.append(
                        {
                            "package": package,
                            "text": generated.text,
                            "direct": parse_direct_likert(dlr.text),
                        }
                    )
        if not responses:
            continue
        embed_inputs = [row["text"] for row in responses] + anchor_texts
        embedded = provider.embed(embed_inputs, model=settings.embed_model)
        if not settings.dry_run:
            spent += _call_cost(settings.embed_model, embedded.input_tokens, 0)
        lookup = dict(zip(embed_inputs, embedded.vectors, strict=True))
        response_matrix = np.asarray([lookup[row["text"]] for row in responses], dtype=float)
        anchor_matrices = [
            np.asarray([lookup[statement] for statement in anchor_set.statements], dtype=float)
            for anchor_set in anchor_sets
        ]
        mean_pmf, _per_set = average_anchor_set_pmfs(response_matrix, anchor_matrices, epsilon=0.0, temperature=1.0)
        for index, row in enumerate(responses):
            row["ssr"] = float(expected_rating(mean_pmf[index]))
            if settings.decisions:
                decided = provider.decide(
                    model=settings.decide_model,
                    input_text=row["text"],
                    questions=build_decision_questions(QUESTION),
                )
                if not settings.dry_run:
                    spent += _call_cost(settings.decide_model, decided.input_tokens, decided.output_tokens)
                interpreted = interpret_decision(decided.payload, threshold=1.1)
                row["decisions"] = interpreted["likert_mean"]
        by_package: dict[str, list[dict]] = {}
        for row in responses:
            by_package.setdefault(row["package"].package_id, []).append(row)
        for package in test.packages:
            rows = by_package.get(package.package_id, [])
            if not rows:
                continue
            scores = {"ssr": float(np.mean([row["ssr"] for row in rows])), "length": float(len(package.headline))}
            directs = [row["direct"] for row in rows if row["direct"] is not None]
            if directs:
                scores["direct_likert"] = float(np.mean(directs))
            if settings.decisions:
                decisions = [row["decisions"] for row in rows if row.get("decisions") is not None]
                if decisions:
                    scores["decisions"] = float(np.mean(decisions))
            scored.append(
                ScoredPackage(
                    package_id=package.package_id,
                    test_id=package.test_id,
                    headline=package.headline,
                    impressions=package.impressions,
                    clicks=package.clicks,
                    ctr=package.ctr,
                    length=len(package.headline),
                    scores=scores,
                )
            )
        if not settings.dry_run and settings.budget_usd is not None and spent > settings.budget_usd:
            stopped = True
            break
    return scored, spent, stopped


def _call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    value = quote_usd(model, input_tokens, output_tokens)
    return 0.0 if value is None else value
