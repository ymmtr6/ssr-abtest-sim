# Upworthy Research Archive

Matias, Munger, Aubin Le Quere, and Ebersole, *The Upworthy Research Archive,
a time series of 32,487 experiments in U.S. media*, Scientific Data (2021),
<https://doi.org/10.1038/s41597-021-00934-7>.
A correction was published in 2024: <https://doi.org/10.1038/s41597-024-03600-w>.

The archive is CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>).
The Open Science Framework project is <https://doi.org/10.17605/OSF.IO/JD64P>.

This repository does not contain the CSV. The exploratory file
(`upworthy-archive-exploratory-packages-03.12.2020.csv`, 14,260,949 bytes)
is downloaded from <https://osf.io/download/3vqmp/>. OSF publishes SHA-256
`8368313b060f4015a0c6fb34e6d788163cee29554144aba9390b14922eb9d8ce`
(checked 2026-11-23).

```bash
python benchmarks/upworthy/fetch_archive.py --output data/upworthy
ssr-abtest validate data/upworthy/upworthy-archive-exploratory-packages-03.12.2020.csv --dry-run --n 20
ssr-abtest validate data/upworthy/upworthy-archive-exploratory-packages-03.12.2020.csv --n 30 --budget 5
```

A live run requires `--budget` in USD and stops before any API call when the
estimate is above that cap. The comparison uses click rate
(`clicks / impressions`). It does not use the archive's `winner`,
`first_place`, or `significance` fields. Tests created from 25 June 2013
through 10 January 2014 are left out unless you pass
`--include-imbalance-window`. The CSV does not include the images that ran
with the headlines.
