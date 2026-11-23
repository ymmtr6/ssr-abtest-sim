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

The example dry-run prices a desktop screenshot but does not launch a browser. A live run that sends screenshots of HTML or a URL needs the optional extra:

```bash
pip install -e ".[screenshot]"
playwright install chromium
```

Set a key only for a live run. The default provider reads `OPENAI_API_KEY`. Other roles can read `ZAI_API_KEY` or `CF_AIG_TOKEN`. Do not commit keys. Use a standard pay-as-you-go z.ai key; GLM Coding Plan keys are limited to official coding tools.

```bash
export OPENAI_API_KEY="..."
ssr-abtest run configs/experiments/example_lp.yaml --output outputs
```

A repeated live run stores Responses, Embeddings, and Decisions payloads in `.ssr-abtest-cache/` (gitignored). Cached token counts are replayed so the report stays comparable; the cache does not send those calls again. `--no-cache` turns that off.

`ssr-abtest estimate configs/experiments/example_lp.yaml` prints the planning cost and does not call the API. Rates are standard list prices checked on 2026-11-23. Batch, Flex, Fast, cache discounts, and regional processing are not applied. See `src/ssr_abtest/cost.py`.

## What a run does

1. Sample personas from independent attribute marginals and assign each persona to one variant. The assignment is a seeded balanced shuffle.
2. Read each variant from local HTML or text, an `http(s)` URL, or an existing image file. `input_mode` is `text`, `image`, or `both`. `image` and `both` attach a screenshot. HTML and URL captures use Playwright with a desktop (1280×800) or mobile (390×844) viewport, either the fold or the full page. Real captures are cached under `.ssr-abtest-cache/screenshots/`. Dry-run writes a placeholder PNG and prices the planned pixel size instead of browsing.
3. Ask the Responses API (`client.responses.create`) for a free-text reaction. An image is an `input_image` content part beside the question, with `detail` from the config (`high` by default). The prompt forbids a numeric score. Default generation model: `gpt-5.4-mini`. Default samples per persona: 2. Default temperature: 0.5. Image tokens are added to the input estimate using the vision guide checked on 2026-11-23: 32×32 patches and a model multiplier for current models (1.2 for `gpt-5.4-mini`), or a base plus 512px tiles for older vision models. On `gpt-5.4-mini`, `detail: low` can cost more than `high`. The API's own usage field is what a live run bills.
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

`configs/experiments/example_lp.yaml` is the template. Each variant needs exactly one of `path` (local HTML or text), `text`, `url`, or `image`. `input_mode: both` in the example attaches a fold-sized desktop screenshot as well as the extracted copy. Each question points at an anchor file with one or more sets of exactly five statements, Likert point 1 first. The paper averaged six reference sets; this example ships three per question. Add more sets in the same shape.

To run the English anchors, point `language` at `en`, rewrite `scenario` and `questions` in English, and set `anchors` to `configs/anchors/en/relevance.yaml` and `configs/anchors/en/click_intent.yaml`.

Unknown model names need `models.prices.<model>.input` and `.output` in USD per 1,000,000 tokens or the dollar total is left empty.

## Providers

Each of generate, embed, and decide can use a different provider. The default is direct OpenAI, with Responses for text, embeddings for SSR, and Decisions only when you pass `--decisions`. Set `models.providers` to one of the files in `configs/providers/` or inline the same mapping.

- `openai.yaml` is that default.
- `zai.yaml` calls z.ai Chat Completions (`https://api.z.ai/api/paas/v4/chat/completions`) and keeps embeddings on OpenAI. `embedding-3` is documented only on the China host `https://open.bigmodel.cn/api/paas/v4/embeddings` (batch 64, 2048 dimensions, 0.5 CNY per 1M tokens). That CNY rate is not added to the USD total.
- `cloudflare-gateway.yaml` sends OpenAI through `https://gateway.ai.cloudflare.com/v1/${CF_ACCOUNT_ID}/${CF_GATEWAY_ID}/openai`. Responses and Chat Completions are documented there. Embeddings use the same base-URL replacement and do not have a separate official example.
- `cloudflare-zai.yaml` sends z.ai through a custom provider. Create slug `zai` with base URL `https://api.z.ai`. The call is `.../custom-zai/api/paas/v4/chat/completions`. Embeddings use Workers AI `@cf/baai/bge-m3`.
- `cloudflare-workers.yaml` uses Cloudflare-hosted `@cf/zai-org/glm-5.3` and `@cf/baai/bge-m3`. Workers AI `glm-5.3-flash` was deprecated on 2026-11-20 and is not used. bge-m3 is called with `{"text": [...]}` at `/ai/run/@cf/baai/bge-m3`, plus `cf-aig-gateway-id`.

Authenticated gateways use `cf-aig-authorization: Bearer $CF_AIG_TOKEN`. BYOK omits `Authorization` so the stored provider key is used. Generation sends `cf-aig-skip-cache: true` because the gateway cache key includes the whole body, and repeated persona samples are often identical. The tool never sends `cf-aig-cache-key`. Embedding requests may use the gateway cache. Decisions is not available on z.ai and is not documented on the gateway. Point the decide role at direct OpenAI, or leave decisions off.

Changing the embedding model changes the SSR scale. Rerun validation for each embedding provider. Prices for unknown models are left empty and called out in the report. They are not treated as zero. List prices checked on 2026-11-23: GLM-5.3 1.4/4.4, GLM-5.3-Flash 0.15/0.50, GLM-5.3-FlashX 0.37/1.25, GLM-4.6V 0.30/0.90, GLM-4.6V-FlashX 0.04/0.40 USD per 1M tokens.

## Validation

`ssr-abtest validate` scores a seeded, week-stratified sample of headline tests from the Upworthy Research Archive and compares stated click intent with observed click rate. The archive is CC BY 4.0. Cite Matias, Munger, Aubin Le Quere, and Ebersole, Scientific Data (2021), <https://doi.org/10.1038/s41597-021-00934-7>, and the OSF deposit <https://doi.org/10.17605/OSF.IO/JD64P>. Fetch and checksum details are in [benchmarks/upworthy/README.md](benchmarks/upworthy/README.md). The CSV is not in this repository.

The report gives pairwise agreement against the click-rate winner, only when a two-proportion z-test separates the pair, plus a bootstrap interval. It also reports the mean within-test Spearman correlation between predicted score and click rate, calibration bins, a random baseline, and a longer-headline baseline. Direct 1–5 ratings are always run beside SSR. `--decisions` adds the Decisions score. `--dry-run` prints the cost and writes fixture scores. A live run requires `--budget` and does not start when the estimate is over the cap.

```bash
python benchmarks/upworthy/fetch_archive.py --output data/upworthy
ssr-abtest validate data/upworthy/upworthy-archive-exploratory-packages-03.12.2020.csv --dry-run --n 20
```

## Tests

```bash
ruff check src tests
pytest
```

GitHub Actions runs that pair on Python 3.11 and 3.12, then dry-runs the example with an empty `OPENAI_API_KEY`.

## Limits

Details live in [docs/limitations.md](docs/limitations.md) and [docs/method.md](docs/method.md). Short version: stated intent only, anchors are uncalibrated, demographic slices are exploratory, and a public-archive calibration is a follow-up ([benchmarks/upworthy/README.md](benchmarks/upworthy/README.md)).

Further follow-ups: Batch API pricing, joint population draws, and fitting ε and T to an external outcome file.

## License

Apache-2.0. Copyright 2026 Riku. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

日本語の説明は [README.ja.md](README.ja.md) にあります。
