# SPDX-License-Identifier: Apache-2.0
"""Run one between-subjects SSR screen from an experiment file."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ssr_abtest.citation import citation_dict
from ssr_abtest.config import ExperimentConfig, load_experiment
from ssr_abtest.cost import cost_from_usage, estimate_plan, estimate_tokens
from ssr_abtest.elicit.prompts import instructions_for, user_prompt
from ssr_abtest.personas.sample import Persona, sample_personas
from ssr_abtest.providers.base import Provider
from ssr_abtest.providers.decisions import build_decision_questions, interpret_decision
from ssr_abtest.providers.dry_run import DryRunProvider
from ssr_abtest.providers.openai_provider import OpenAIProvider
from ssr_abtest.render.html import load_variant_text
from ssr_abtest.render.screenshots import VariantImage, resolve_images
from ssr_abtest.report.render import write_reports
from ssr_abtest.ssr.core import average_anchor_set_pmfs, expected_rating
from ssr_abtest.stats.compare import compare_variants, top2_box

LIMITATIONS = [
    "The output is a distribution over stated intent, not a click-through rate, conversion rate, or dwell time.",
    "Use it to screen variants before a live experiment. It does not replace that experiment.",
    "Anchor wording changes the mapping. The bundled statements are a starting point, not the reference statements from the paper, and they have not been locked to human retest data.",
    "Persona attributes are sampled from independent marginals. Subgroup splits, especially gender and region, are exploratory.",
    "A direct numeric rating from the same model is a different method. The optional Decisions score is reported beside SSR and is not the primary endpoint.",
    "Results can move when the model, the embedding model, the seed, or the anchor sets change. The report records those choices.",
    "A screenshot is one view of the page at the chosen viewport. Full-page height is measured only when a browser capture runs. Until then the cost estimate uses the viewport, or the viewport height times full_page_height_factor.",
]


@dataclass
class Call:
    persona_id: str
    variant: str
    question_id: str
    sample_index: int
    attributes: dict[str, str]
    instructions: str
    user_input: str
    survey_prompt: str


@dataclass
class ResponseRecord:
    persona_id: str
    variant: str
    question_id: str
    sample_index: int
    attributes: dict[str, str]
    text: str
    input_tokens: int
    output_tokens: int
    pmf: list[float] | None = None
    per_set_expected: list[float] | None = None
    excluded: bool = False
    exclude_reason: str | None = None
    decision: dict[str, Any] | None = None


@dataclass
class Prepared:
    config: ExperimentConfig
    personas: list[Persona]
    page_text: dict[str, str]
    truncated: dict[str, bool]
    calls: list[Call]
    anchor_texts: list[str]
    images: dict[str, VariantImage]


def _page_text_for(config: ExperimentConfig, variant, extra_text: dict[str, str]) -> tuple[str, bool]:
    if variant.kind == "url":
        text = extra_text.get(variant.id, "")
        if config.input_mode == "image":
            text = "The page is attached as a screenshot."
        return text, False
    if variant.kind == "image":
        return "The page is attached as a screenshot.", False
    if config.input_mode == "image":
        return "The page is attached as a screenshot.", False
    text, truncated = load_variant_text(variant.path, variant.text, max_chars=config.max_page_chars)
    return text, truncated


def prepare(
    config: ExperimentConfig,
    *,
    output_dir: Path | None = None,
    capture: bool = False,
    cache_dir: Path | None = None,
    playwright_factory=None,
) -> Prepared:
    variant_ids = list(config.variants)
    personas = sample_personas(
        config.personas.attributes,
        n=config.personas.n,
        variants=variant_ids,
        seed=config.personas.seed,
    )
    shot_dir = None if output_dir is None else output_dir / "screenshots"
    images, extra_text = resolve_images(
        config,
        capture=capture,
        output_dir=shot_dir,
        cache_dir=cache_dir,
        playwright_factory=playwright_factory,
    )
    page_text: dict[str, str] = {}
    truncated: dict[str, bool] = {}
    for variant_id, variant in config.variants.items():
        text, was_truncated = _page_text_for(config, variant, extra_text)
        if not text:
            raise ValueError(f"variant {variant_id} has no visible text")
        page_text[variant_id] = text
        truncated[variant_id] = was_truncated
    instructions = instructions_for(config.language)
    attach_image = config.input_mode == "both"
    calls: list[Call] = []
    for persona in personas:
        variant = config.variants[persona.variant]
        for question in config.questions:
            for sample_index in range(config.samples_per_persona):
                calls.append(
                    Call(
                        persona_id=persona.id,
                        variant=persona.variant,
                        question_id=question.id,
                        sample_index=sample_index,
                        attributes=dict(persona.attributes),
                        instructions=instructions,
                        user_input=user_prompt(
                            attributes=persona.attributes,
                            scenario=config.scenario,
                            variant_id=persona.variant,
                            variant_label=variant.label,
                            page_text=page_text[persona.variant],
                            question=question.prompt,
                            attach_image=attach_image,
                        ),
                        survey_prompt=question.prompt,
                    )
                )
    anchor_texts: list[str] = []
    seen: set[str] = set()
    for question in config.questions:
        for anchor_set in question.sets:
            for statement in anchor_set.statements:
                if statement not in seen:
                    seen.add(statement)
                    anchor_texts.append(statement)
    return Prepared(
        config=config,
        personas=personas,
        page_text=page_text,
        truncated=truncated,
        calls=calls,
        anchor_texts=anchor_texts,
        images=images,
    )


def planned_cost(prepared: Prepared) -> dict[str, Any]:
    config = prepared.config
    generation_input = sum(
        estimate_tokens(call.instructions) + estimate_tokens(call.user_input) for call in prepared.calls
    )
    image_tokens = 0
    image_notes: list[str] = []
    unpriced = False
    if config.input_mode in {"image", "both"}:
        for call in prepared.calls:
            image = prepared.images[call.variant]
            if image.tokens is None:
                unpriced = True
                if image.token_error and image.token_error not in image_notes:
                    image_notes.append(image.token_error)
            else:
                image_tokens += image.tokens
                generation_input += image.tokens
        if any(image.assumed_dimensions for image in prepared.images.values()):
            image_notes.append(
                "Full-page height is not known until a browser capture. "
                "This estimate uses viewport height times "
                f"screenshot.full_page_height_factor ({config.screenshot.full_page_height_factor})."
            )
        if any(image.placeholder for image in prepared.images.values()):
            image_notes.append(
                "No browser ran for this estimate. HTML and URL screenshots use the "
                "configured viewport (or the full-page assumption above), not measured pixels."
            )
    decision_input = 0
    if config.decisions.enabled:
        for call in prepared.calls:
            questions = build_decision_questions(call.survey_prompt)
            rubric = call.survey_prompt + "\n" + json.dumps(questions, ensure_ascii=False)
            decision_input += config.assumed_output_tokens + estimate_tokens(rubric)
    return estimate_plan(
        generate_model=config.models.generate,
        embed_model=config.models.embed,
        decide_model=config.models.decide,
        generation_input_tokens=generation_input,
        n_generations=len(prepared.calls),
        assumed_output_tokens=config.assumed_output_tokens,
        anchor_tokens=sum(estimate_tokens(text) for text in prepared.anchor_texts),
        decision_input_tokens=decision_input,
        decisions_enabled=config.decisions.enabled,
        price_overrides=config.models.prices or None,
        image_input_tokens=image_tokens,
        image_notes=image_notes,
        image_unpriced=unpriced,
    )


def _embed_lookup(provider: Provider, texts: list[str], model: str) -> tuple[dict[str, list[float]], int]:
    unique = list(dict.fromkeys(texts))
    batch = provider.embed(unique, model=model)
    if len(batch.vectors) != len(unique):
        raise RuntimeError("embedding provider returned the wrong number of vectors")
    return dict(zip(unique, batch.vectors, strict=True)), batch.input_tokens


def _question_block(
    config: ExperimentConfig,
    records: list[ResponseRecord],
    *,
    synthetic_decisions: bool,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    variant_ids = list(config.variants)
    for question in config.questions:
        mine = [record for record in records if record.question_id == question.id]
        kept = [record for record in mine if not record.excluded]
        grouped: dict[tuple[str, str], list[ResponseRecord]] = defaultdict(list)
        for record in kept:
            grouped[(record.variant, record.persona_id)].append(record)
        ratings: dict[str, list[float]] = {variant_id: [] for variant_id in variant_ids}
        boxes: dict[str, list[float]] = {variant_id: [] for variant_id in variant_ids}
        pmf_rows: dict[str, list[np.ndarray]] = {variant_id: [] for variant_id in variant_ids}
        persona_rows: list[tuple[str, str, dict[str, str], float]] = []
        for (variant_id, persona_id), samples in sorted(grouped.items()):
            matrix = np.asarray([sample.pmf for sample in samples], dtype=float)
            persona_pmf = matrix.mean(axis=0)
            score = float(expected_rating(persona_pmf))
            ratings[variant_id].append(score)
            boxes[variant_id].append(float(top2_box(persona_pmf)[0]))
            pmf_rows[variant_id].append(persona_pmf)
            persona_rows.append((variant_id, persona_id, samples[0].attributes, score))
        arms: dict[str, Any] = {}
        for variant_id in variant_ids:
            rows = pmf_rows[variant_id]
            if rows:
                mean_pmf = np.mean(np.stack(rows, axis=0), axis=0)
                mean_value = float(np.mean(ratings[variant_id]))
                top2_value = float(np.mean(boxes[variant_id]))
            else:
                width = len(question.sets[0].statements)
                mean_pmf = np.full(width, 1.0 / width)
                mean_value = None
                top2_value = None
            arms[variant_id] = {
                "n": len(rows),
                "mean": mean_value,
                "mean_pmf": [float(value) for value in mean_pmf],
                "top2_box": top2_value,
            }
        rating_arrays = {key: np.asarray(value, dtype=float) for key, value in ratings.items()}
        box_arrays = {key: np.asarray(value, dtype=float) for key, value in boxes.items()}
        usable = {key: value for key, value in rating_arrays.items() if value.size}
        comparisons: list[dict[str, Any]] = []
        if len(usable) >= 2:
            comparisons = compare_variants(
                {key: rating_arrays[key] for key in variant_ids if rating_arrays[key].size},
                {key: box_arrays[key] for key in variant_ids if box_arrays[key].size},
                replicates=config.bootstrap.replicates,
                seed=config.bootstrap.seed,
                confidence=config.bootstrap.confidence,
            )
        spreads = [
            float(np.std(record.per_set_expected, ddof=0))
            for record in kept
            if record.per_set_expected
        ]
        excerpts: dict[str, list[str]] = {variant_id: [] for variant_id in variant_ids}
        for record in sorted(mine, key=lambda item: (item.variant, item.persona_id, item.sample_index)):
            bucket = excerpts[record.variant]
            if len(bucket) < 2 and record.text not in bucket:
                bucket.append(record.text[:400])
        block: dict[str, Any] = {
            "id": question.id,
            "prompt": question.prompt,
            "anchors": str(question.anchors_path),
            "n_anchor_sets": len(question.sets),
            "anchor_set_ids": [anchor_set.id for anchor_set in question.sets],
            "anchor_set_spread": float(np.mean(spreads)) if spreads else None,
            "arms": arms,
            "comparisons": comparisons,
            "excerpts": excerpts,
            "exploratory_subgroups": _subgroups(config, persona_rows),
            "decisions": _decision_summary(config, mine, synthetic_decisions),
        }
        blocks.append(block)
    return blocks


def _subgroups(
    config: ExperimentConfig,
    persona_rows: list[tuple[str, str, dict[str, str], float]],
) -> list[dict[str, Any]]:
    present = set()
    for attribute in config.personas.attributes:
        present.add(attribute["name"])
    groups: list[dict[str, Any]] = []
    for name in config.exploratory_attributes:
        if name not in present:
            continue
        levels: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for variant_id, _persona_id, attributes, score in persona_rows:
            levels[attributes[name]][variant_id].append(score)
        rendered_levels = {}
        for level, arms in sorted(levels.items()):
            rendered_levels[level] = {
                variant_id: {"n": len(values), "mean": float(np.mean(values))}
                for variant_id, values in sorted(arms.items())
            }
        groups.append(
            {
                "attribute": name,
                "note": (
                    "Exploratory only. The SSR paper did not reliably reproduce "
                    "gender or region differences."
                ),
                "levels": rendered_levels,
            }
        )
    return groups


def _decision_summary(
    config: ExperimentConfig,
    records: list[ResponseRecord],
    synthetic: bool,
) -> dict[str, Any] | None:
    if not config.decisions.enabled:
        return None
    excluded = sum(1 for record in records if record.excluded)
    arms: dict[str, dict[str, float | None]] = {}
    for variant_id in config.variants:
        chosen = [
            record.decision
            for record in records
            if record.variant == variant_id and record.decision and not record.excluded
        ]
        likert = [float(item["likert_mean"]) for item in chosen if item.get("likert_mean") is not None]
        cta = [
            float(item["cta_probability"])
            for item in chosen
            if item.get("cta_probability") is not None
        ]
        arms[variant_id] = {
            "n_responses": len(chosen),
            "mean_likert": float(np.mean(likert)) if likert else None,
            "mean_cta_probability": float(np.mean(cta)) if cta else None,
        }
    return {"synthetic": synthetic, "excluded_responses": excluded, "arms": arms}


def run_experiment(
    config: ExperimentConfig,
    provider: Provider,
    *,
    output_dir: Path,
    dry_run: bool,
    cache_dir: Path | None = None,
    playwright_factory=None,
) -> dict[str, Any]:
    prepared = prepare(
        config,
        output_dir=output_dir / config.name,
        capture=not dry_run and config.input_mode in {"image", "both"},
        cache_dir=None if dry_run else cache_dir,
        playwright_factory=playwright_factory,
    )
    estimate = planned_cost(prepared)
    records: list[ResponseRecord] = []
    generation_input = 0
    generation_output = 0
    for call in prepared.calls:
        image = prepared.images.get(call.variant)
        send_image = image is not None and image.path is not None and not image.placeholder
        generated = provider.generate(
            model=config.models.generate,
            instructions=call.instructions,
            user_input=call.user_input,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            sample_index=call.sample_index,
            image_path=str(image.path) if send_image else None,
            image_detail=image.detail if image is not None else "high",
            image_tokens=(image.tokens or 0) if image is not None else 0,
        )
        generation_input += generated.input_tokens
        generation_output += generated.output_tokens
        records.append(
            ResponseRecord(
                persona_id=call.persona_id,
                variant=call.variant,
                question_id=call.question_id,
                sample_index=call.sample_index,
                attributes=call.attributes,
                text=generated.text,
                input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
            )
        )
    texts = [record.text for record in records]
    lookup, embedding_tokens = _embed_lookup(provider, texts + prepared.anchor_texts, config.models.embed)
    decision_input = 0
    decision_output = 0
    for question in config.questions:
        selected = [record for record in records if record.question_id == question.id]
        response_matrix = np.asarray([lookup[record.text] for record in selected], dtype=float)
        anchor_matrices = [
            np.asarray([lookup[statement] for statement in anchor_set.statements], dtype=float)
            for anchor_set in question.sets
        ]
        mean_pmf, per_set = average_anchor_set_pmfs(
            response_matrix,
            anchor_matrices,
            epsilon=config.ssr.epsilon,
            temperature=config.ssr.temperature,
        )
        for row_index, record in enumerate(selected):
            record.pmf = [float(value) for value in mean_pmf[row_index]]
            record.per_set_expected = [
                float(expected_rating(per_set[set_index, row_index]))
                for set_index in range(per_set.shape[0])
            ]
        if not config.decisions.enabled:
            continue
        for record in selected:
            questions = build_decision_questions(question.prompt)
            decided = provider.decide(
                model=config.models.decide,
                input_text=record.text,
                questions=questions,
            )
            decision_input += decided.input_tokens
            decision_output += decided.output_tokens
            interpreted = interpret_decision(
                decided.payload,
                threshold=config.decisions.persona_break_threshold,
            )
            record.decision = interpreted
            if interpreted["exclude"]:
                record.excluded = True
                record.exclude_reason = interpreted["exclude_reason"]
    if dry_run:
        cost = estimate
    else:
        measured = cost_from_usage(
            generate_model=config.models.generate,
            embed_model=config.models.embed,
            decide_model=config.models.decide,
            generation_input_tokens=generation_input,
            generation_output_tokens=generation_output,
            embedding_input_tokens=embedding_tokens,
            decision_input_tokens=decision_input,
            decision_output_tokens=decision_output,
            decisions_enabled=config.decisions.enabled,
            price_overrides=config.models.prices or None,
        )
        measured["planned_total_usd"] = estimate["total_usd"]
        cost = measured
    report = {
        "experiment": config.name,
        "dry_run": dry_run,
        "created_at": datetime.now(UTC).isoformat(),
        "citation": citation_dict(),
        "method": {
            "design": "between-subjects",
            "primary_endpoint": "mean expected Likert rating (1-5) of stated intent",
            "similarity": "(1 + cosine) / 2",
            "epsilon": config.ssr.epsilon,
            "temperature": config.ssr.temperature,
            "aggregation": (
                "Each anchor set produces a PMF. Sets are averaged with equal weight. "
                "A persona's PMF is the mean of its samples. Statistics use one row per persona."
            ),
            "reference_implementation": citation_dict()["reference_implementation"],
        },
        "models": {
            "generate": config.models.generate,
            "embed": config.models.embed,
            "decide": config.models.decide,
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
            "input_mode": config.input_mode,
        },
        "language": config.language,
        "scenario": config.scenario,
        "n_personas": config.personas.n,
        "samples_per_persona": config.samples_per_persona,
        "seed": config.personas.seed,
        "persona_spec": config.personas.spec_id,
        "persona_source_note": config.personas.source_note,
        "decisions_enabled": config.decisions.enabled,
        "input_mode": config.input_mode,
        "variants": [
            {
                "id": variant.id,
                "label": variant.label,
                "kind": variant.kind,
                "text_preview": prepared.page_text[variant.id][:240],
                "text_truncated": prepared.truncated[variant.id],
                "screenshot": _screenshot_report(
                    prepared.images.get(variant.id),
                    output_dir / config.name,
                ),
            }
            for variant in config.variants.values()
        ],
        "assignment": [
            {"persona_id": persona.id, "variant": persona.variant, "attributes": persona.attributes}
            for persona in prepared.personas
        ],
        "usage": {
            "generation_input_tokens": generation_input,
            "generation_output_tokens": generation_output,
            "embedding_input_tokens": embedding_tokens,
            "decision_input_tokens": decision_input,
            "decision_output_tokens": decision_output,
        },
        "cost": cost,
        "questions": _question_block(config, records, synthetic_decisions=dry_run and config.decisions.enabled),
        "responses": [
            {
                "persona_id": record.persona_id,
                "variant": record.variant,
                "question_id": record.question_id,
                "sample_index": record.sample_index,
                "text": record.text,
                "pmf": record.pmf,
                "excluded": record.excluded,
                "exclude_reason": record.exclude_reason,
            }
            for record in records
        ],
        "limitations": LIMITATIONS,
    }
    destination = output_dir / config.name
    write_reports(report, destination)
    return report


def _screenshot_report(image: VariantImage | None, report_dir: Path) -> dict[str, Any] | None:
    if image is None:
        return None
    path: str | None = None
    if image.path is not None:
        try:
            path = str(image.path.resolve().relative_to(report_dir.resolve()))
        except ValueError:
            path = str(image.path)
    return {
        "path": path,
        "width": image.width,
        "height": image.height,
        "detail": image.detail,
        "capture": image.capture,
        "viewport": image.viewport,
        "placeholder": image.placeholder,
        "assumed_dimensions": image.assumed_dimensions,
        "image_input_tokens": image.tokens,
        "source": image.source,
    }


def build_provider(config: ExperimentConfig, *, dry_run: bool, cache_dir: Path | None) -> Provider:
    if dry_run:
        return DryRunProvider(config.language)
    from ssr_abtest.cache import DiskCache

    cache = DiskCache(cache_dir) if cache_dir is not None else None
    return OpenAIProvider(cache=cache)


def execute(
    config_path: Path,
    *,
    output_dir: Path,
    dry_run: bool,
    decisions: bool | None,
    cache_dir: Path | None,
) -> dict[str, Any]:
    config = load_experiment(config_path)
    if decisions is not None:
        config.decisions.enabled = decisions
    provider = build_provider(config, dry_run=dry_run, cache_dir=cache_dir)
    shot_cache = None if cache_dir is None else cache_dir / "screenshots"
    return run_experiment(
        config,
        provider,
        output_dir=output_dir,
        dry_run=dry_run,
        cache_dir=shot_cache,
    )


def estimate_config(config_path: Path, *, decisions: bool | None = None) -> dict[str, Any]:
    config = load_experiment(config_path)
    if decisions is not None:
        config.decisions.enabled = decisions
    return planned_cost(prepare(config, capture=False))
