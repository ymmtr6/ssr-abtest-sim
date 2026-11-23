# SPDX-License-Identifier: Apache-2.0
"""Command line: ``ssr-abtest run config.yaml`` and ``ssr-abtest estimate``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ssr_abtest import __version__
from ssr_abtest.pipeline import estimate_config, execute


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        decisions = _decisions_flag(args)
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
