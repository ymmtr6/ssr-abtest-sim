# SPDX-License-Identifier: Apache-2.0
"""Command line: ``ssr-abtest run config.yaml`` and ``ssr-abtest estimate``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ssr_abtest import __version__
from ssr_abtest.pipeline import estimate_config, execute
from ssr_abtest.validate.report import write_validation
from ssr_abtest.validate.runner import ValidateSettings, validate_archive


def _decisions_flag(args: argparse.Namespace) -> bool | None:
    enabled = bool(getattr(args, "decisions", False))
    disabled = bool(getattr(args, "no_decisions", False))
    if enabled and disabled:
        raise ValueError("pass only one of --decisions and --no-decisions")
    if enabled:
        return True
    if disabled:
        return False
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssr-abtest",
        description=(
            "Screen web-page variants with LLM personas and Semantic Similarity Rating. "
            "The result is a stated-intent comparison, not a live A/B test."
        ),
    )
    parser.add_argument("--version", action="version", version=f"ssr-abtest {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run an experiment and write JSON and Markdown reports")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use local fixtures and a price estimate; do not call the API",
    )
    run_parser.add_argument("--output", type=Path, default=Path("outputs"))
    run_parser.add_argument("--decisions", action="store_true", help="enable the Decisions API check")
    run_parser.add_argument("--no-decisions", action="store_true", help="force the Decisions API check off")
    run_parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="directory for API response cache (default: .ssr-abtest-cache on a live run)",
    )
    run_parser.add_argument("--no-cache", action="store_true", help="do not read or write the API cache")

    estimate_parser = sub.add_parser("estimate", help="print a JSON cost estimate and exit")
    estimate_parser.add_argument("config", type=Path)
    estimate_parser.add_argument("--decisions", action="store_true")
    estimate_parser.add_argument("--no-decisions", action="store_true")

    validate_parser = sub.add_parser(
        "validate",
        help="score a sample of Upworthy headline tests and compare with click rates",
    )
    validate_parser.add_argument("data", type=Path, help="archive CSV from benchmarks/upworthy/fetch_archive.py")
    validate_parser.add_argument("--n", type=int, default=20, help="headline tests to sample")
    validate_parser.add_argument("--seed", type=int, default=42)
    validate_parser.add_argument("--dry-run", action="store_true")
    validate_parser.add_argument("--output", type=Path, default=Path("outputs/upworthy-validation"))
    validate_parser.add_argument("--personas", type=Path, default=Path("configs/personas/us_readers_example.yaml"))
    validate_parser.add_argument("--anchors", type=Path, default=Path("configs/anchors/en/click_intent.yaml"))
    validate_parser.add_argument("--personas-n", type=int, default=4)
    validate_parser.add_argument("--samples", type=int, default=1)
    validate_parser.add_argument("--min-impressions", type=int, default=500)
    validate_parser.add_argument("--max-packages", type=int, default=4)
    validate_parser.add_argument("--mixed-images", action="store_true", help="score every package, not only a shared image")
    validate_parser.add_argument("--include-imbalance-window", action="store_true")
    validate_parser.add_argument("--alpha", type=float, default=0.05)
    validate_parser.add_argument("--budget", type=float, default=None, help="hard cap in USD; required for a live run")
    validate_parser.add_argument("--images", type=Path, default=None, help="optional directory of eyecatcher_id images")
    validate_parser.add_argument("--decisions", action="store_true")
    validate_parser.add_argument("--no-decisions", action="store_true")
    validate_parser.add_argument("--generate-model", default="gpt-5.4-mini")
    validate_parser.add_argument("--embed-model", default="text-embedding-3-small")
    validate_parser.add_argument("--decide-model", default="gpt-6-luna")
    return parser


def _cmd_validate(args: argparse.Namespace, decisions: bool | None) -> int:
    if not args.dry_run and args.budget is None:
        raise RuntimeError("a live validation needs --budget USD. Use --dry-run to price the sample first.")
    enabled = True if decisions is True else False
    settings = ValidateSettings(
        data_path=args.data,
        n=args.n,
        seed=args.seed,
        dry_run=args.dry_run,
        n_personas=args.personas_n,
        samples_per_persona=args.samples,
        min_impressions=args.min_impressions,
        max_packages=args.max_packages,
        same_image=not args.mixed_images,
        include_imbalance_window=args.include_imbalance_window,
        alpha=args.alpha,
        confidence=0.95,
        bootstrap_replicates=1000,
        personas_path=args.personas,
        anchors_path=args.anchors,
        generate_model=args.generate_model,
        embed_model=args.embed_model,
        decide_model=args.decide_model,
        temperature=0.5,
        decisions=enabled,
        budget_usd=args.budget,
        image_dir=args.images,
        output_dir=args.output,
    )
    report = validate_archive(settings)
    json_path, md_path = write_validation(report, args.output)
    if args.dry_run:
        print("Dry run: no API calls. Scores are local fixtures, not model judgments.")
    print(f"Planned USD: {report['cost'].get('total_usd')}")
    if report.get("budget_usd") is not None:
        print(f"Budget cap USD: {report['budget_usd']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    agreement = report["methods"]["ssr"]["agreement"]
    print(
        f"SSR agreement on clear pairs: {agreement.get('agreement')} "
        f"({agreement.get('n_scored_pairs')} pairs)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        decisions = _decisions_flag(args)
        if args.command == "validate":
            return _cmd_validate(args, decisions)
        if args.command == "estimate":
            report = estimate_config(args.config, decisions=decisions)
            json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
            sys.stdout.write("\n")
            return 0
        cache_dir: Path | None
        if args.dry_run or args.no_cache:
            cache_dir = None
        else:
            cache_dir = args.cache if args.cache is not None else Path(".ssr-abtest-cache")
        report = execute(
            args.config,
            output_dir=args.output,
            dry_run=args.dry_run,
            decisions=decisions,
            cache_dir=cache_dir,
        )
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print("Dry run: no API calls. Figures are local fixtures, not model judgments.")
    print(f"Wrote {args.output / report['experiment'] / 'report.json'}")
    print(f"Wrote {args.output / report['experiment'] / 'report.md'}")
    for question in report["questions"]:
        means = ", ".join(
            f"{variant_id}={arm['mean']:.3f}" if arm["mean"] is not None else f"{variant_id}=n/a"
            for variant_id, arm in question["arms"].items()
        )
        print(f"{question['id']}: {means}")
    return 0
