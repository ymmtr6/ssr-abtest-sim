# SPDX-License-Identifier: Apache-2.0
"""Write the experiment report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ssr_abtest.citation import BIBTEX, URL, VERSION_NOTE


def write_reports(report: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    md_path = directory / "report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    title = report["experiment"]
    lines.append(f"# SSR screen: {title}")
    lines.append("")
    if report.get("dry_run"):
        lines.append(
            "> Dry run. Free-text answers and embeddings are local fixtures. "
            "They are not model judgments and they are not a substitute for a live A/B test."
        )
        lines.append("")
    lines.append(
        "This report compares stated intent (relevance, satisfaction, click intent, "
        "purchase intent, and similar attitudes). It does not estimate click-through "
        "rate, conversion rate, or time on page."
    )
    lines.append("")
    lines.append("## Citation")
    lines.append("")
    lines.append(
        f"Method: Maier et al., {report['citation']['title']}. {VERSION_NOTE}. {URL}"
    )
    lines.append("")
    lines.append("```bibtex")
    lines.append(BIBTEX.rstrip("\n"))
    lines.append("```")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Language: {report['language']}")
    lines.append(f"- Scenario: {report['scenario']}")
    lines.append(f"- Personas: {report['n_personas']} (seed {report['seed']})")
    lines.append(f"- Samples per persona: {report['samples_per_persona']}")
    lines.append(
        f"- Models: generate `{report['models']['generate']}`, "
        f"embed `{report['models']['embed']}`, decide `{report['models']['decide']}`"
    )
    lines.append(
        f"- SSR: ε = {report['method']['epsilon']}, "
        f"T = {report['method']['temperature']}, similarity `{report['method']['similarity']}`"
    )
    lines.append(f"- Decisions auxiliary score: {'on' if report['decisions_enabled'] else 'off'}")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    for variant in report["variants"]:
        lines.append(f"### {variant['id']}: {variant['label']}")
        lines.append("")
        lines.append(variant["text_preview"])
        lines.append("")
    cost = report["cost"]
    lines.append("## Cost")
    lines.append("")
    lines.append(f"Basis: {cost['basis']}. Total USD: {_fmt(cost.get('total_usd'), 4)}.")
    lines.append("")
    for row in cost["by_model"]:
        lines.append(
            f"- {row['role']} `{row['model']}`: "
            f"in {row['input_tokens']}, out {row['output_tokens']}, usd {_fmt(row['usd'], 4)}"
        )
    lines.append("")
    for question in report["questions"]:
        lines.append(f"## {question['id']}")
        lines.append("")
        lines.append(question["prompt"])
        lines.append("")
        lines.append(f"Anchor sets: {question['n_anchor_sets']}. "
                     f"Mean within-response spread of set-level expected ratings: "
                     f"{_fmt(question['anchor_set_spread'])}.")
        lines.append("")
        lines.append("| Variant | Personas | Mean Likert | Top-2 box |")
        lines.append("| --- | ---: | ---: | ---: |")
        for variant_id, arm in question["arms"].items():
            lines.append(
                f"| {variant_id} | {arm['n']} | {_fmt(arm['mean'])} | {_fmt(arm['top2_box'])} |"
            )
        lines.append("")
        for comparison in question["comparisons"]:
            if comparison.get("error"):
                lines.append(f"{comparison['a']} vs {comparison['b']}: {comparison['error']}")
                lines.append("")
                continue
            low, high = comparison["bootstrap_ci"]
            lines.append(
                f"Difference ({comparison['b']} − {comparison['a']}): "
                f"{_fmt(comparison['mean_difference_b_minus_a'])} "
                f"(bootstrap {comparison['confidence']:.0%} CI {_fmt(low)} to {_fmt(high)})."
            )
            lines.append(
                f"KS statistic {_fmt(comparison['ks_statistic'])} "
                f"(p = {_fmt(comparison['ks_pvalue'])}). "
                f"Cohen's d {_fmt(comparison['cohens_d_b_minus_a'])}."
            )
            lines.append("")
            lines.append(comparison["interpretation"])
            lines.append("")
            lines.append(
                "The preferred label is the higher mean stated intent. "
                "It is not a prediction that the variant would win a live experiment."
            )
            lines.append("")
        decisions = question.get("decisions")
        if decisions:
            synthetic = " Synthetic dry-run values." if decisions.get("synthetic") else ""
            lines.append(f"Auxiliary Decisions score.{synthetic}")
            lines.append(f"Responses excluded for persona break: {decisions['excluded_responses']}.")
            for variant_id, arm in decisions["arms"].items():
                lines.append(
                    f"- {variant_id}: Decisions Likert {_fmt(arm.get('mean_likert'))}, "
                    f"CTA probability {_fmt(arm.get('mean_cta_probability'))}"
                )
            lines.append("")
        if question["exploratory_subgroups"]:
            lines.append(
                "Subgroup means are exploratory. The SSR paper did not reliably "
                "reproduce gender or region patterns, so do not treat these splits as findings."
            )
            lines.append("")
            for subgroup in question["exploratory_subgroups"]:
                lines.append(f"### {subgroup['attribute']}")
                lines.append("")
                for level, arms in subgroup["levels"].items():
                    bits = [
                        f"{variant_id} n={arm['n']} mean={_fmt(arm['mean'])}"
                        for variant_id, arm in arms.items()
                    ]
                    lines.append(f"- {level}: " + "; ".join(bits))
                lines.append("")
        lines.append("Excerpts from the free-text answers:")
        lines.append("")
        for variant_id, excerpts in question["excerpts"].items():
            for excerpt in excerpts:
                lines.append(f"- {variant_id}: {excerpt}")
        lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(
        f"Persona file note: {report['persona_source_note']}"
    )
    lines.append("")
    return "\n".join(lines)
