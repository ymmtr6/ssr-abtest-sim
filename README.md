# ssr-abtest-sim

Screen web-page variants with LLM personas and [Semantic Similarity Rating](https://arxiv.org/abs/2510.08338) (SSR).

A persona sees one landing page, product page, or other page variant and writes a short reaction. Embeddings map that reaction onto a 1–5 distribution by its similarity to anchor statements. The report compares variants on **stated intent** (relevance, click intent, and the same kind of attitude). It does not estimate click-through rate, conversion rate, or time on page, and it is not a substitute for a live A/B test.

The example experiment is a fictional weekend-trip landing page. Run it with no API key:

```bash
pip install -e ".[dev]"
ssr-abtest run configs/experiments/example_lp.yaml --dry-run
```

Dry-run writes `outputs/lp-hero-copy/report.json` and `report.md`. The numbers are local fixtures so you can check the pipeline. They are not model judgments.

## Citation

Maier et al., *LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings*, arXiv:2510.08338v3 (27 October 2025), <https://arxiv.org/abs/2510.08338>.

The probability mapping follows that paper and the Apache-2.0 reference implementation at <https://github.com/pymc-labs/semantic-similarity-rating>. This repository does not vendor that code. Bundled anchor sentences are originals for page-level attitudes, not the paper's personal-care purchase-intent anchors.

BibTeX exported from <https://arxiv.org/bibtex/2510.08338>:

```bibtex
@misc{maier2025llmsreproducehumanpurchase,
      title={LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings}, 
      author={Benjamin F. Maier and Ulf Aslak and Luca Fiaschi and Nina Rismal and Kemble Fletcher and Christian C. Luhmann and Robbie Dow and Kli Pappas and Thomas V. Wiecki},
      year={2025},
      eprint={2510.08338},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.08338}, 
}
```

## Install

Python 3.11 or newer.

```bash
pip install -e .
```

Screenshot capture is optional and is not required for the example:

```bash
pip install -e ".[screenshot]"
playwright install chromium
```

Set a key only for a live run. The SDK reads `OPENAI_API_KEY` from the environment. Do not commit it.

```bash
export OPENAI_API_KEY="..."
ssr-abtest run configs/experiments/example_lp.yaml --output outputs
```

A repeated live run stores Responses, Embeddings, and Decisions payloads in `.ssr-abtest-cache/` (gitignored). Cached token counts are replayed so the report stays comparable; the cache does not send those calls again. `--no-cache` turns that off.

`ssr-abtest estimate configs/experiments/example_lp.yaml` prints the planning cost and does not call the API. Rates are standard list prices checked on 2026-11-23. Batch, Flex, Fast, cache discounts, and regional processing are not applied. See `src/ssr_abtest/cost.py`.

## What a run does

1. Sample personas from independent attribute marginals and assign each persona to one variant. The assignment is a seeded balanced shuffle.
2. Extract visible text from each HTML or text variant. `render: [text, screenshot]` also stores a local PNG when the optional extra is installed. Images are not sent to the model in this version.
3. Ask the Responses API (`client.responses.create`) for a free-text reaction. The prompt forbids a numeric score. Default generation model: `gpt-5.4-mini`. Default samples per persona: 2. Default temperature: 0.5.
4. Embed the reactions and the anchor statements with `text-embedding-3-small`.
5. Turn scaled cosine similarity into a 5-point PMF (ε = 0, temperature T = 1 by default) and average the anchor sets.
6. Compare persona-level mean Likert scores with a bootstrap confidence interval, a two-sample Kolmogorov-Smirnov statistic, a top-two box, and Cohen's d.
7. Write JSON and Markdown. Gender and region splits are marked exploratory.

Add `--decisions` to also call the Decisions API (`POST /v1/decisions`, model `gpt-6-luna`) on each free-text answer. Checked against the Decisions guide on 2026-11-23. The adapter sends one `score` question (five ordered levels) and two `predicate` questions (call-to-action willingness, and persona breakage). A score answer's `value` is a 0-based index; the report converts it to a 1–5 mean. Answers at or above `persona_break_threshold` (default 0.8) are dropped from the aggregate. `choice` is part of the API and can be passed through `OpenAIProvider.decide`; the default set does not send one. SSR remains the primary endpoint. The Decisions score is auxiliary and can bunch in the middle, as direct ratings did in the paper.

## Layout

```text
src/ssr_abtest/     library and the ssr-abtest CLI
configs/personas/   marginal weights (aggregates only)
configs/anchors/    Japanese and English anchor sets
configs/experiments example experiment
examples/pages/     fictional landing, product, and route-list pages
tests/              mocked providers; no network
```

Paths in an experiment file are relative to the working directory, then to the file itself. Run commands from the repository root.

The example persona file is named `jp_census_example.yaml` because the layout is a marginal table. The weights are rounded placeholders, not official statistics. Replace them with published aggregates you are allowed to redistribute. Do not commit microdata.

Page files under `examples/pages/` are fiction: a landing page pair used by the example, a product page, and a route list that is only a results-style layout for that fictional travel site.

## Configuration

`configs/experiments/example_lp.yaml` is the template. Variants need a `path` or inline `text`, not both. Each question points at an anchor file with one or more sets of exactly five statements, Likert point 1 first. The paper averaged six reference sets; this example ships three per question. Add more sets in the same shape.

To run the English anchors, point `language` at `en`, rewrite `scenario` and `questions` in English, and set `anchors` to `configs/anchors/en/relevance.yaml` and `configs/anchors/en/click_intent.yaml`.

Unknown model names need `models.prices.<model>.input` and `.output` in USD per 1,000,000 tokens or the dollar total is left empty.

## Tests

```bash
ruff check src tests
pytest
```

GitHub Actions runs that pair on Python 3.11 and 3.12, then dry-runs the example with an empty `OPENAI_API_KEY`.

## Limits

Details live in [docs/limitations.md](docs/limitations.md) and [docs/method.md](docs/method.md). Short version: stated intent only, anchors are uncalibrated, demographic slices are exploratory, and a public-archive calibration is a follow-up ([benchmarks/upworthy/README.md](benchmarks/upworthy/README.md)).

Further follow-ups: sending screenshots to the model, Batch API pricing, joint population draws, and fitting ε and T to an external outcome file.

## License

Apache-2.0. Copyright 2026 Riku. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

日本語の説明は [README.ja.md](README.ja.md) にあります。
