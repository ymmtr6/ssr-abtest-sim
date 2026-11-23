# SPDX-License-Identifier: Apache-2.0
import json
import socket
from pathlib import Path

import pytest

from ssr_abtest.citation import BIBTEX, URL
from ssr_abtest.cli import main
from ssr_abtest.pipeline import OpenAIProvider

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "configs/experiments/example_lp.yaml"


def _block_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket, "create_connection", refuse)


def test_readmes_include_the_arxiv_bibtex():
    for name in ("README.md", "README.ja.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert BIBTEX.strip() in text
        assert URL in text
        assert "2510.08338" in text


def test_dry_run_example_writes_reports_without_a_client(tmp_path: Path, monkeypatch):
    _block_network(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def boom(*_args, **_kwargs):
        raise AssertionError("dry-run constructed an API provider")

    monkeypatch.setattr("ssr_abtest.pipeline.OpenAIProvider", boom)
    output = tmp_path / "out"
    assert main(["run", str(EXAMPLE), "--dry-run", "--output", str(output)]) == 0
    report_path = output / "lp-hero-copy" / "report.json"
    markdown = (output / "lp-hero-copy" / "report.md").read_text(encoding="utf-8")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert report["n_personas"] == 24
    assert report["citation"]["url"] == URL
    assert report["cost"]["basis"] == "estimated"
    assert report["cost"]["total_usd"] > 0
    assert report["input_mode"] == "both"
    assert report["cost"]["image_input_tokens"] == 24 * 2 * 2 * 1200
    assert report["variants"][0]["screenshot"]["placeholder"] is True
    assert report["variants"][0]["screenshot"]["width"] == 1280
    assert "Dry run" in markdown
    assert URL in markdown
    assert len(report["responses"]) == 24 * 2 * 2
    for question in report["questions"]:
        assert question["decisions"] is None
        total_n = sum(arm["n"] for arm in question["arms"].values())
        assert total_n == 24
        for arm in question["arms"].values():
            assert sum(arm["mean_pmf"]) == pytest.approx(1.0)
            assert arm["n"] == 12
        comparison = question["comparisons"][0]
        low, high = comparison["bootstrap_ci"]
        assert low <= high
    for response in report["responses"]:
        assert sum(response["pmf"]) == pytest.approx(1.0)
    assert not (tmp_path / ".ssr-abtest-cache").exists()


def test_second_dry_run_matches_and_estimate_matches_cost(tmp_path: Path, monkeypatch, capsys):
    _block_network(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    first = tmp_path / "a"
    second = tmp_path / "b"
    assert main(["run", str(EXAMPLE), "--dry-run", "--output", str(first)]) == 0
    assert main(["run", str(EXAMPLE), "--dry-run", "--output", str(second)]) == 0
    left = json.loads((first / "lp-hero-copy" / "report.json").read_text(encoding="utf-8"))
    right = json.loads((second / "lp-hero-copy" / "report.json").read_text(encoding="utf-8"))
    left.pop("created_at")
    right.pop("created_at")
    assert left == right
    capsys.readouterr()
    assert main(["estimate", str(EXAMPLE)]) == 0
    estimate = json.loads(capsys.readouterr().out)
    assert estimate["total_usd"] == left["cost"]["total_usd"]
    assert estimate["calls"] == 24 * 2 * 2


def test_dry_run_decisions_stay_synthetic(tmp_path: Path, monkeypatch):
    _block_network(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = tmp_path / "out"
    assert main(["run", str(EXAMPLE), "--dry-run", "--decisions", "--output", str(output)]) == 0
    report = json.loads((output / "lp-hero-copy" / "report.json").read_text(encoding="utf-8"))
    assert report["decisions_enabled"] is True
    for question in report["questions"]:
        decisions = question["decisions"]
        assert decisions["synthetic"] is True
        assert decisions["excluded_responses"] == 0
        for arm in decisions["arms"].values():
            assert arm["mean_likert"] == pytest.approx(3.0)
            assert arm["mean_cta_probability"] == pytest.approx(0.5)


def test_missing_config_returns_2(tmp_path: Path):
    assert main(["run", str(tmp_path / "missing.yaml"), "--dry-run"]) == 2


def test_provider_symbol_is_the_real_class():
    assert OpenAIProvider.__name__ == "OpenAIProvider"
