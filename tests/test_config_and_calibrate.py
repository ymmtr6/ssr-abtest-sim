# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from ssr_abtest.calibrate.agreement import agreement_rate, preferred_variant
from ssr_abtest.config import load_experiment

ROOT = Path(__file__).parents[1]


def test_example_experiment_loads():
    config = load_experiment(ROOT / "configs/experiments/example_lp.yaml")
    assert config.name == "lp-hero-copy"
    assert config.language == "ja"
    assert list(config.variants) == ["A", "B"]
    assert config.personas.n == 24
    assert config.models.generate == "gpt-5.4-mini"
    assert config.models.embed == "text-embedding-3-small"
    assert config.models.decide == "gpt-6-luna"
    assert config.decisions.enabled is False
    assert config.ssr.epsilon == 0.0
    assert config.ssr.temperature == 1.0
    assert [question.id for question in config.questions] == ["relevance", "click_intent"]
    for question in config.questions:
        assert len(question.sets) == 3
        for anchor_set in question.sets:
            assert len(anchor_set.statements) == 5
    assert "official statistics" in config.personas.source_note
    assert config.variants["A"].path is not None
    assert config.variants["A"].path.is_file()


def test_unknown_key_is_rejected(tmp_path: Path):
    source = (ROOT / "configs/experiments/example_lp.yaml").read_text(encoding="utf-8")
    broken = tmp_path / "broken.yaml"
    broken.write_text(source + "\nnot_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_experiment(broken)


def test_agreement_helper():
    assert preferred_variant({"A": 3.2, "B": 3.9}) == "B"
    assert preferred_variant({"A": 3.0, "B": 3.0}) == "tie"
    summary = agreement_rate(["A", "B", "tie"], ["A", "A", "tie"])
    assert summary["matches"] == 1
    assert summary["agreement"] == pytest.approx(1 / 3)
