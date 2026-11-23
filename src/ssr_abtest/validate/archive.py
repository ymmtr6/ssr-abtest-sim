# SPDX-License-Identifier: Apache-2.0
"""Read Upworthy packages and draw a seeded, week-stratified sample.

Ground truth is the click rate ``clicks / impressions``. The archive's
``winner``, ``first_place``, and ``significance`` columns are editorial or
legacy fields and are not used.

The public exploratory file has no imbalance flag. Tests whose packages
were created from 25 June 2013 through 10 January 2014 are dropped unless
``include_imbalance_window`` is set. That window is the period the archive
authors asked confirmatory work to avoid, because a web cache may have
served one arm at a time (Scientific Data correction, 2024).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

IMBALANCE_START = datetime(2013, 6, 25)
IMBALANCE_END = datetime(2014, 1, 11)  # exclusive; through 10 January 2014

CITATION = {
    "title": "The Upworthy Research Archive, a time series of 32,487 experiments in U.S. media",
    "authors": "J. Nathan Matias, Kevin Munger, Marianne Aubin Le Quere, and Charles R. Ebersole",
    "year": 2021,
    "venue": "Scientific Data",
    "article_url": "https://doi.org/10.1038/s41597-021-00934-7",
    "correction_url": "https://doi.org/10.1038/s41597-024-03600-w",
    "osf_url": "https://doi.org/10.17605/OSF.IO/JD64P",
    "project_url": "https://osf.io/jd64p/",
    "exploratory_download": "https://osf.io/download/3vqmp/",
    "license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "exploratory_sha256": "8368313b060f4015a0c6fb34e6d788163cee29554144aba9390b14922eb9d8ce",
}


@dataclass(frozen=True)
class Package:
    test_id: str
    package_id: str
    headline: str
    excerpt: str
    eyecatcher_id: str
    impressions: int
    clicks: int
    created_at: datetime
    test_week: str

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions


@dataclass(frozen=True)
class HeadlineTest:
    test_id: str
    week: str
    packages: tuple[Package, ...]


def _parse_time(value: str) -> datetime:
    text = value.strip().replace("Z", "")
    if "." in text:
        head, frac = text.split(".", 1)
        text = head + "." + frac[:6]
        return datetime.fromisoformat(text)
    return datetime.fromisoformat(text)


def _flagged(row: dict[str, str]) -> bool:
    for key in ("problem", "randomization_imbalance_risk"):
        value = (row.get(key) or "").strip().lower()
        if value in {"1", "true", "t", "yes"}:
            return True
    return False


def load_packages(path: Path) -> list[Package]:
    packages: list[Package] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"clickability_test_id", "headline", "impressions", "clicks", "created_at", "test_week"}
        fieldnames = set(reader.fieldnames or [])
        missing = required - fieldnames
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for index, row in enumerate(reader):
            if _flagged(row):
                continue
            headline = (row.get("headline") or "").strip()
            if not headline:
                continue
            try:
                impressions = int(float(row["impressions"]))
                clicks = int(float(row["clicks"]))
            except ValueError:
                continue
            if impressions <= 0 or clicks < 0 or clicks > impressions:
                continue
            packages.append(
                Package(
                    test_id=row["clickability_test_id"].strip(),
                    package_id=f"{row['clickability_test_id'].strip()}:{index}",
                    headline=headline,
                    excerpt=(row.get("excerpt") or "").strip(),
                    eyecatcher_id=(row.get("eyecatcher_id") or "").strip(),
                    impressions=impressions,
                    clicks=clicks,
                    created_at=_parse_time(row["created_at"]),
                    test_week=str(row.get("test_week") or "").strip() or "unknown",
                )
            )
    if not packages:
        raise ValueError(f"{path} has no usable packages")
    return packages


def _in_imbalance_window(packages: list[Package]) -> bool:
    return any(IMBALANCE_START <= package.created_at < IMBALANCE_END for package in packages)


def _same_image_group(packages: list[Package]) -> list[Package] | None:
    groups: dict[str, list[Package]] = defaultdict(list)
    for package in packages:
        groups[package.eyecatcher_id].append(package)
    best = max(groups.values(), key=len)
    if len(best) < 2:
        return None
    return best


def build_tests(
    packages: list[Package],
    *,
    min_impressions: int,
    max_packages: int,
    same_image: bool,
    include_imbalance_window: bool,
) -> list[HeadlineTest]:
    grouped: dict[str, list[Package]] = defaultdict(list)
    for package in packages:
        grouped[package.test_id].append(package)
    tests: list[HeadlineTest] = []
    for test_id, rows in grouped.items():
        if not include_imbalance_window and _in_imbalance_window(rows):
            continue
        chosen = _same_image_group(rows) if same_image else list(rows)
        if not chosen:
            continue
        chosen = [package for package in chosen if package.impressions >= min_impressions]
        if len(chosen) < 2 or len(chosen) > max_packages:
            continue
        week = chosen[0].test_week
        tests.append(HeadlineTest(test_id=test_id, week=week, packages=tuple(chosen)))
    return tests


def sample_tests(tests: list[HeadlineTest], n: int, seed: int) -> list[HeadlineTest]:
    """Week-stratified sample. Returns every remaining test when fewer than ``n`` exist."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if not tests:
        raise ValueError("no tests left after filters")
    if n >= len(tests):
        return list(tests)
    groups: dict[str, list[HeadlineTest]] = defaultdict(list)
    for test in tests:
        groups[test.week].append(test)
    weeks = sorted(groups)
    rng = np.random.default_rng(seed)
    for week in weeks:
        order = np.arange(len(groups[week]))
        rng.shuffle(order)
        groups[week] = [groups[week][int(index)] for index in order]
    sizes = np.array([len(groups[week]) for week in weeks], dtype=float)
    raw = sizes / sizes.sum() * n
    alloc = np.floor(raw).astype(int)
    remainder = int(n - alloc.sum())
    fraction = raw - alloc
    for index in np.argsort(-fraction):
        if remainder <= 0:
            break
        if alloc[index] < sizes[index]:
            alloc[index] += 1
            remainder -= 1
    # If rounding still left seats because of caps, fill from any leftover.
    chosen: list[HeadlineTest] = []
    leftovers: list[HeadlineTest] = []
    for week, count in zip(weeks, alloc):
        chosen.extend(groups[week][: int(count)])
        leftovers.extend(groups[week][int(count) :])
    if len(chosen) < n and leftovers:
        order = np.arange(len(leftovers))
        rng.shuffle(order)
        for index in order:
            if len(chosen) >= n:
                break
            chosen.append(leftovers[int(index)])
    return chosen[:n]
