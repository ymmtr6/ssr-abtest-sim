# SPDX-License-Identifier: Apache-2.0
"""Write the validation JSON and Markdown report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_validation(report: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "validation.json"
    md_path = directory / "validation.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_validation(report), encoding="utf-8")
    return json_path, md_path


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(block: dict | None) -> str:
    if not block or not block.get("ci"):
        return "n/a"
    low, high = block["ci"]
    return f"{_fmt(block.get('mean'))} ({block.get('confidence', 0.95):.0%} CI {_fmt(low)} to {_fmt(high)})"


def render_validation(report: dict[str, Any]) -> str:
    data = report["dataset"]
    lines = [
        "# Upworthy archive validation",
        "",
    ]
    if report["dry_run"]:
        lines.append(
            "> Dry run. Scores are local fixtures, not model judgments. "
            "Click rates in the file are real only when you pointed this command at the archive."
        )
        lines.append("")
    lines.extend(
        [
            "Predicted scores are stated click intent on a 1–5 scale. "
            "They are not estimates of click-through rate. "
            "Agreement uses only pairs whose click rates differ on a two-sided two-proportion z-test.",
            "",
            "## Dataset",
            "",
            f"- {data['authors']} ({data['year']}). {data['title']}. {data['venue']}. {data['article_url']}",
            f"- Correction: {data['correction_url']}",
            f"- Archive: {data['osf_url']} ({data['license']}, {data['license_url']})",
            f"- Exploratory file: {data['exploratory_download']}",
            f"- SHA-256: `{data['exploratory_sha256']}`",
            f"- Loaded packages: {data['n_packages_loaded']}. Tests after filters: {data['n_tests_available']}. "
            f"Sampled: {data['n_tests_sampled']} of {data['n_tests_requested']} requested.",
            f"- Same-image headline contrast: {data['same_image']}. "
            f"Minimum impressions: {data['min_impressions']}.",
            f"- Imbalance window (25 June 2013 through 10 January 2014) included: "
            f"{data['include_imbalance_window']}.",
            "",
            data["persona_source_note"],
            "",
            "## Cost",
            "",
            f"Planned USD: {_fmt(report['cost'].get('total_usd'), 4)}. "
            f"Recorded spend USD: {_fmt(report.get('spent_usd'), 4)}. "
            f"Budget cap: {_fmt(report.get('budget_usd'), 4)}.",
            "",
        ]
    )
    if report.get("stopped_early"):
        lines.append("The run stopped because recorded spend reached the budget cap.")
        lines.append("")
    lines.append(f"Clear CTR pairs in the sample: {report['n_clear_pairs']}.")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append("| Method | Pair agreement | Mean within-test Spearman rho |")
    lines.append("| --- | --- | --- |")
    labels = [
        ("ssr", "SSR"),
        ("direct_likert", "Direct Likert"),
        ("decisions", "Decisions score"),
        ("headline_length", "Longer headline"),
        ("random", "Random"),
    ]
    for key, label in labels:
        block = report["methods"].get(key)
        if block is None:
            lines.append(f"| {label} | not run | not run |")
            continue
        lines.append(f"| {label} | {_ci(block['agreement'])} | {_ci(block.get('spearman'))} |")
    lines.append("")
    lines.append(
        "Spearman rho is computed inside each test, then averaged across tests. "
        "The interval resamples tests. A two-package test can only produce rho of -1 or 1; "
        "`spearman_at_least_3` in the JSON drops those tests."
    )
    lines.append("")
    ssr = report["methods"]["ssr"]
    lines.append("## SSR calibration bins")
    lines.append("")
    lines.append("Bins group packages by predicted mean Likert. The CTR column is the mean observed click rate.")
    lines.append("")
    lines.append("| Predicted Likert | Packages | Mean predicted | Mean CTR |")
    lines.append("| --- | ---: | ---: | ---: |")
    for bucket in ssr["calibration"]["bins"]:
        span = f"{bucket['predicted_low']:g}–{bucket['predicted_high']:g}"
        lines.append(
            f"| {span} | {bucket['n']} | {_fmt(bucket['mean_predicted'])} | {_fmt(bucket['mean_ctr'], 4)} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in report["cost"].get("notes", []):
        lines.append(f"- {note}")
    lines.append(
        "- The archive does not include the images that were shown with the headlines. "
        "Pass `--images` only if you supply those files yourself."
    )
    lines.append("")
    return "\n".join(lines)
