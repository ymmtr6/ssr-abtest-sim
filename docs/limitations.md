# Limitations

Read this before treating a report as evidence.

* The measured quantity is stated intent: relevance, clarity, satisfaction,
  click intent, purchase intent, and similar attitudes. It is not a
  click-through rate, a conversion rate, or time on page.
* A higher mean is a reason to prefer a variant in a later live experiment.
  It is not a claim that the variant would win that experiment.
* Direct numeric ratings from language models pile up in the middle of the
  scale. SSR exists to avoid that. The optional Decisions score is closer
  to a direct rating and can still bunch in the middle. Keep SSR as the
  primary endpoint.
* Anchor wording moves the distribution. Average several sets, and treat a
  large spread across sets as a warning that the anchors are unstable.
* Personas need attributes. Sampling here draws each attribute independently
  from the weights in the YAML file. It does not reproduce correlations in
  a real population, and it does not use microdata.
* The paper reproduced age and income patterns more readily than gender,
  region, or other demographic splits. Those splits are labelled
  exploratory in the report.
* The model, the embedding model, and the seed are part of the result.
  Record them, and expect movement when any of them change.
* Screenshot capture is optional and local. Captured images are not sent to
  the model in this version. The prompt uses extracted visible text.
* Placeholder persona weights in the example are not official statistics.
  Replace them with aggregate tables you are allowed to ship.
* There is no bundled calibration against a public A/B archive. Agreement
  with live outcomes has to be measured on data you supply. See
  `benchmarks/upworthy/README.md`.
