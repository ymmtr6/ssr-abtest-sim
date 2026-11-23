# SPDX-License-Identifier: Apache-2.0
import json
import socket
from pathlib import Path

import pytest

from ssr_abtest.cli import main
from ssr_abtest.validate.archive import build_tests, load_packages, sample_tests
from ssr_abtest.validate.metrics import (
    ScoredPackage,
    agreement_on_pairs,
    calibration_bins,
    clear_pairs,
    spearman_by_test,
    two_proportion_z,
)
from ssr_abtest.validate.runner import ValidateSettings, validate_archive

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/upworthy_synthetic.csv"


def _package(package_id: str, clicks: int, impressions: int, score: float, headline: str) -> ScoredPackage:
    return ScoredPackage(
        package_id=package_id,
        test_id="t",
        headline=headline,
        impressions=impressions,
        clicks=clicks,
        ctr=clicks / impressions,
        length=len(headline),
        scores={"ssr": score, "length": float(len(headline))},
    )


def test_clear_pair_and_agreement():
    loud = abs(two_proportion_z(100, 2000, 300, 2000))
    quiet = abs(two_proportion_z(100, 2000, 110, 2000))
    assert loud > 1.96
    assert quiet < 1.96
    low = _package("a", 100, 2000, 2.0, "short")
    high = _package("b", 300, 2000, 4.0, "a much longer headline")
    near = _package("c", 110, 2000, 2.1, "mid")
    pairs = clear_pairs([low, high, near], alpha=0.05)
    winners = {pair[2] for pair in pairs}
    assert winners == {"b"}
    summary = agreement_on_pairs(pairs, "ssr", replicates=200, seed=1, confidence=0.95)
    assert summary["agreement"] == pytest.approx(1.0)
    length = agreement_on_pairs(pairs, "length", replicates=200, seed=1, confidence=0.95)
    assert length["agreement"] == pytest.approx(1.0)
    rho = spearman_by_test(
        [low, high, near],
        "ssr",
        replicates=200,
        seed=2,
        confidence=0.95,
        minimum_packages=3,
    )
    assert rho["mean_rho"] == pytest.approx(1.0)
    bins = calibration_bins([low, high], "ssr")
    assert sum(bucket["n"] for bucket in bins["bins"]) == 2


def test_archive_filters_drop_the_imbalance_window():
    packages = load_packages(FIXTURE)
    tests = build_tests(
        packages,
        min_impressions=100,
        max_packages=4,
        same_image=True,
        include_imbalance_window=False,
    )
    assert {test.test_id for test in tests} == {"test-clear", "test-other"}
    assert len(sample_tests(tests, n=5, seed=1)) == 2
    kept = build_tests(
        packages,
        min_impressions=100,
        max_packages=4,
        same_image=True,
        include_imbalance_window=True,
    )
    assert "test-imbalance" in {test.test_id for test in kept}


def test_validate_dry_run_and_budget_cap(tmp_path: Path, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("validation dry-run opened a connection")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = tmp_path / "out"
    code = main(
        [
            "validate",
            str(FIXTURE),
            "--dry-run",
            "--n",
            "2",
            "--personas-n",
            "2",
            "--max-packages",
            "2",
            "--min-impressions",
            "100",
            "--output",
            str(output),
            "--personas",
            str(ROOT / "configs/personas/us_readers_example.yaml"),
            "--anchors",
            str(ROOT / "configs/anchors/en/click_intent.yaml"),
            "--seed",
            "1",
        ]
    )
    assert code == 0
    report = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    markdown = (output / "validation.md").read_text(encoding="utf-8")
    assert report["dry_run"] is True
    assert report["dataset"]["n_tests_sampled"] == 2
    assert report["n_clear_pairs"] == 2
    assert report["cost"]["total_usd"] > 0
    assert report["methods"]["ssr"]["agreement"]["n_scored_pairs"] == 2
    assert report["methods"]["direct_likert"]["agreement"]["n_scored_pairs"] == 2
    assert report["methods"]["headline_length"]["agreement"]["agreement"] == pytest.approx(1.0)
    assert report["methods"]["decisions"] is None
    assert "10.17605/OSF.IO/JD64P" in markdown
    assert "CC BY 4.0" in markdown
    settings = ValidateSettings(
        data_path=FIXTURE,
        n=2,
        seed=1,
        dry_run=False,
        n_personas=2,
        samples_per_persona=1,
        min_impressions=100,
        max_packages=2,
        same_image=True,
        include_imbalance_window=False,
        alpha=0.05,
        confidence=0.95,
        bootstrap_replicates=50,
        personas_path=ROOT / "configs/personas/us_readers_example.yaml",
        anchors_path=ROOT / "configs/anchors/en/click_intent.yaml",
        generate_model="gpt-5.4-mini",
        embed_model="text-embedding-3-small",
        decide_model="gpt-6-luna",
        temperature=0.5,
        decisions=False,
        budget_usd=0.0,
        image_dir=None,
        output_dir=output,
    )
    with pytest.raises(RuntimeError, match="budget"):
        validate_archive(settings, provider=object())
