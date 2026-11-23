# SPDX-License-Identifier: Apache-2.0
"""Load experiment, persona, and anchor YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_EXPERIMENT_KEYS = {
    "name",
    "language",
    "variants",
    "render",
    "personas",
    "scenario",
    "questions",
    "models",
    "samples_per_persona",
    "temperature",
    "max_output_tokens",
    "assumed_output_tokens",
    "max_page_chars",
    "ssr",
    "decisions",
    "bootstrap",
    "exploratory_attributes",
    "input_mode",
    "screenshot",
}
_RENDER_MODES = {"text", "screenshot"}
_INPUT_MODES = {"text", "image", "both"}
_DETAILS = {"low", "high", "auto", "original"}
_LIKERT_POINTS = 5
VIEWPORTS = {
    "desktop": (1280, 800),
    "mobile": (390, 844),
}


@dataclass
class Variant:
    id: str
    label: str
    kind: str
    path: Path | None
    text: str | None
    url: str | None
    image: Path | None


@dataclass
class AnchorSet:
    id: str
    statements: list[str]


@dataclass
class Question:
    id: str
    prompt: str
    anchors_path: Path
    sets: list[AnchorSet]


@dataclass
class PersonaSource:
    path: Path
    spec_id: str
    description: str
    source_note: str
    attributes: list[dict[str, Any]]
    n: int
    seed: int


@dataclass
class Models:
    generate: str
    embed: str
    decide: str
    prices: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class SSRSettings:
    epsilon: float = 0.0
    temperature: float = 1.0


@dataclass
class DecisionSettings:
    enabled: bool = False
    persona_break_threshold: float = 0.8


@dataclass
class BootstrapSettings:
    replicates: int = 2000
    seed: int = 42
    confidence: float = 0.95


@dataclass
class ScreenshotSettings:
    viewport: str = "desktop"
    width: int = 1280
    height: int = 800
    capture: str = "fold"
    detail: str = "high"
    full_page_height_factor: float = 3.0


@dataclass
class ExperimentConfig:
    name: str
    language: str
    variants: dict[str, Variant]
    render: list[str]
    input_mode: str
    screenshot: ScreenshotSettings
    personas: PersonaSource
    scenario: str
    questions: list[Question]
    models: Models
    samples_per_persona: int
    temperature: float
    max_output_tokens: int
    assumed_output_tokens: int
    max_page_chars: int
    ssr: SSRSettings
    decisions: DecisionSettings
    bootstrap: BootstrapSettings
    exploratory_attributes: list[str]
    source_path: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _resolve(raw: str, config_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    from_config = (config_path.parent / path).resolve()
    if from_config.exists():
        return from_config
    raise FileNotFoundError(f"file not found: {raw} (looked in {from_cwd} and {from_config})")


def _require(mapping: dict, key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{where} is missing {key}")
    return mapping[key]


def _unknown(mapping: dict, allowed: set[str], where: str) -> None:
    extra = sorted(set(mapping) - allowed)
    if extra:
        raise ValueError(f"{where} has unknown keys: {', '.join(extra)}")


def _load_anchors(path: Path) -> tuple[list[AnchorSet], str | None, str | None]:
    data = _load_yaml(path)
    _unknown(data, {"question_id", "language", "sets"}, str(path))
    sets = _require(data, "sets", str(path))
    if not isinstance(sets, list) or not sets:
        raise ValueError(f"{path} needs at least one anchor set")
    loaded: list[AnchorSet] = []
    seen: set[str] = set()
    for index, item in enumerate(sets):
        if not isinstance(item, dict):
            raise ValueError(f"{path} set {index} must be a mapping")
        set_id = str(item.get("id") or f"set_{index + 1}")
        if set_id in seen:
            raise ValueError(f"{path} repeats anchor set id {set_id}")
        seen.add(set_id)
        statements = item.get("statements")
        if not isinstance(statements, list) or len(statements) != _LIKERT_POINTS:
            raise ValueError(f"{path} set {set_id} needs exactly {_LIKERT_POINTS} statements")
        cleaned = [str(statement).strip() for statement in statements]
        if any(not statement for statement in cleaned):
            raise ValueError(f"{path} set {set_id} has an empty statement")
        loaded.append(AnchorSet(id=set_id, statements=cleaned))
    language = data.get("language")
    question_id = data.get("question_id")
    return loaded, (str(language) if language else None), (str(question_id) if question_id else None)


def _load_personas(path: Path, n: int, seed: int) -> PersonaSource:
    data = _load_yaml(path)
    _unknown(data, {"id", "description", "source_note", "attributes"}, str(path))
    attributes = _require(data, "attributes", str(path))
    if not isinstance(attributes, list) or not attributes:
        raise ValueError(f"{path} needs attributes")
    names: set[str] = set()
    for attribute in attributes:
        if not isinstance(attribute, dict):
            raise ValueError(f"{path} attributes must be mappings")
        name = str(_require(attribute, "name", str(path)))
        if name in names:
            raise ValueError(f"{path} repeats attribute {name}")
        names.add(name)
        levels = _require(attribute, "levels", f"{path} attribute {name}")
        if not isinstance(levels, list) or len(levels) < 1:
            raise ValueError(f"{path} attribute {name} needs levels")
        for level in levels:
            if "label" not in level or "weight" not in level:
                raise ValueError(f"{path} attribute {name} levels need label and weight")
            if float(level["weight"]) < 0:
                raise ValueError(f"{path} attribute {name} has a negative weight")
    return PersonaSource(
        path=path,
        spec_id=str(data.get("id") or path.stem),
        description=str(data.get("description") or ""),
        source_note=str(data.get("source_note") or ""),
        attributes=attributes,
        n=n,
        seed=seed,
    )


def _load_screenshot(raw: dict[str, Any]) -> ScreenshotSettings:
    if not isinstance(raw, dict):
        raise ValueError("screenshot must be a mapping")
    _unknown(
        raw,
        {"viewport", "width", "height", "capture", "detail", "full_page_height_factor"},
        "screenshot",
    )
    viewport = str(raw.get("viewport") or "desktop")
    if viewport not in VIEWPORTS and ("width" not in raw or "height" not in raw):
        raise ValueError("screenshot.viewport must be desktop or mobile, or set width and height")
    default_width, default_height = VIEWPORTS.get(viewport, (1280, 800))
    width = int(raw.get("width") or default_width)
    height = int(raw.get("height") or default_height)
    if width < 1 or height < 1:
        raise ValueError("screenshot width and height must be positive")
    capture = str(raw.get("capture") or "fold")
    if capture not in {"full_page", "fold"}:
        raise ValueError("screenshot.capture must be full_page or fold")
    detail = str(raw.get("detail") or "high")
    if detail not in _DETAILS:
        raise ValueError("screenshot.detail must be low, high, auto, or original")
    factor = float(raw.get("full_page_height_factor") or 3)
    if factor < 1:
        raise ValueError("screenshot.full_page_height_factor must be >= 1")
    return ScreenshotSettings(
        viewport=viewport,
        width=width,
        height=height,
        capture=capture,
        detail=detail,
        full_page_height_factor=factor,
    )


def load_experiment(path: Path) -> ExperimentConfig:
    path = Path(path)
    data = _load_yaml(path)
    _unknown(data, _EXPERIMENT_KEYS, str(path))
    name = str(_require(data, "name", str(path))).strip()
    if not name or any(char in name for char in "/\\"):
        raise ValueError("experiment name must be a single path segment")
    language = str(_require(data, "language", str(path)))
    raw_variants = _require(data, "variants", str(path))
    if not isinstance(raw_variants, dict) or len(raw_variants) < 2:
        raise ValueError("define at least two variants")
    variants: dict[str, Variant] = {}
    for variant_id, spec in raw_variants.items():
        if not isinstance(spec, dict):
            raise ValueError(f"variant {variant_id} must be a mapping")
        _unknown(spec, {"path", "text", "url", "image", "label"}, f"variant {variant_id}")
        sources = [key for key in ("path", "text", "url", "image") if spec.get(key) not in (None, "")]
        if len(sources) != 1:
            raise ValueError(
                f"variant {variant_id} needs exactly one of path, text, url, or image"
            )
        source = sources[0]
        variant_path = None
        variant_text = None
        variant_url = None
        variant_image = None
        if source == "text":
            kind = "text"
            variant_text = str(spec["text"])
        elif source == "url":
            kind = "url"
            variant_url = str(spec["url"]).strip()
            if not variant_url.startswith(("http://", "https://")):
                raise ValueError(f"variant {variant_id} url must start with http:// or https://")
        elif source == "image":
            kind = "image"
            variant_image = _resolve(str(spec["image"]), path)
            if variant_image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                raise ValueError(f"variant {variant_id} image must be PNG, JPEG, WebP, or GIF")
        else:
            variant_path = _resolve(str(spec["path"]), path)
            suffix = variant_path.suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                raise ValueError(
                    f"variant {variant_id} looks like an image. Set image: instead of path:."
                )
            kind = "html" if suffix in {".html", ".htm"} else "text"
        variants[str(variant_id)] = Variant(
            id=str(variant_id),
            label=str(spec.get("label") or variant_id),
            kind=kind,
            path=variant_path,
            text=variant_text,
            url=variant_url,
            image=variant_image,
        )
    render = list(data.get("render") or ["text"])
    if any(mode not in _RENDER_MODES for mode in render):
        raise ValueError("render must list text, screenshot, or both")
    input_mode = str(data.get("input_mode") or "").strip()
    if not input_mode:
        input_mode = "both" if "screenshot" in render else "text"
    if input_mode not in _INPUT_MODES:
        raise ValueError("input_mode must be text, image, or both")
    for variant in variants.values():
        if input_mode == "text" and variant.kind in {"url", "image"}:
            raise ValueError(
                f"variant {variant.id} is a {variant.kind}, which has no text transcript. "
                "Use input_mode image or both."
            )
    screenshot = _load_screenshot(data.get("screenshot") or {})
    persona_block = _require(data, "personas", str(path))
    _unknown(persona_block, {"config", "n", "seed"}, "personas")
    n_personas = int(_require(persona_block, "n", "personas"))
    seed = int(_require(persona_block, "seed", "personas"))
    personas = _load_personas(_resolve(str(_require(persona_block, "config", "personas")), path), n_personas, seed)
    if n_personas < len(variants):
        raise ValueError("personas.n must be at least the number of variants")
    questions_raw = _require(data, "questions", str(path))
    if not isinstance(questions_raw, list) or not questions_raw:
        raise ValueError("at least one question is required")
    questions: list[Question] = []
    seen_questions: set[str] = set()
    for item in questions_raw:
        _unknown(item, {"id", "prompt", "anchors"}, "question")
        question_id = str(_require(item, "id", "question"))
        if question_id in seen_questions:
            raise ValueError(f"duplicate question id {question_id}")
        seen_questions.add(question_id)
        anchors_path = _resolve(str(_require(item, "anchors", question_id)), path)
        sets, anchor_language, anchor_question = _load_anchors(anchors_path)
        if anchor_question is not None and anchor_question != question_id:
            raise ValueError(
                f"question {question_id} points at {anchors_path}, "
                f"whose question_id is {anchor_question}"
            )
        if anchor_language is not None and anchor_language != language:
            raise ValueError(
                f"question {question_id} uses {anchor_language} anchors "
                f"in a {language} experiment"
            )
        questions.append(
            Question(
                id=question_id,
                prompt=str(_require(item, "prompt", question_id)).strip(),
                anchors_path=anchors_path,
                sets=sets,
            )
        )
    models_raw = _require(data, "models", str(path))
    _unknown(models_raw, {"generate", "embed", "decide", "prices"}, "models")
    prices = models_raw.get("prices") or {}
    if not isinstance(prices, dict):
        raise ValueError("models.prices must be a mapping")
    normalized_prices: dict[str, dict[str, float]] = {}
    for key, value in prices.items():
        if not isinstance(value, dict) or "input" not in value:
            raise ValueError(f"models.prices.{key} needs an input rate in USD per 1,000,000 tokens")
        normalized_prices[str(key)] = {str(rate): float(amount) for rate, amount in value.items()}
    models = Models(
        generate=str(_require(models_raw, "generate", "models")),
        embed=str(_require(models_raw, "embed", "models")),
        decide=str(models_raw.get("decide") or "gpt-6-luna"),
        prices=normalized_prices,
    )
    ssr_raw = data.get("ssr") or {}
    _unknown(ssr_raw, {"epsilon", "temperature"}, "ssr")
    ssr = SSRSettings(
        epsilon=float(ssr_raw.get("epsilon", 0.0)),
        temperature=float(ssr_raw.get("temperature", 1.0)),
    )
    if ssr.epsilon < 0:
        raise ValueError("ssr.epsilon must be >= 0")
    if ssr.temperature < 0:
        raise ValueError("ssr.temperature must be >= 0")
    decisions_raw = data.get("decisions") or {}
    _unknown(decisions_raw, {"enabled", "persona_break_threshold"}, "decisions")
    decisions = DecisionSettings(
        enabled=bool(decisions_raw.get("enabled", False)),
        persona_break_threshold=float(decisions_raw.get("persona_break_threshold", 0.8)),
    )
    if not 0.0 <= decisions.persona_break_threshold <= 1.0:
        raise ValueError("decisions.persona_break_threshold must be between 0 and 1")
    boot_raw = data.get("bootstrap") or {}
    _unknown(boot_raw, {"replicates", "seed", "confidence"}, "bootstrap")
    bootstrap = BootstrapSettings(
        replicates=int(boot_raw.get("replicates", 2000)),
        seed=int(boot_raw.get("seed", seed)),
        confidence=float(boot_raw.get("confidence", 0.95)),
    )
    if bootstrap.replicates < 1:
        raise ValueError("bootstrap.replicates must be >= 1")
    if not 0.0 < bootstrap.confidence < 1.0:
        raise ValueError("bootstrap.confidence must be between 0 and 1")
    samples = int(data.get("samples_per_persona", 2))
    if samples < 1:
        raise ValueError("samples_per_persona must be >= 1")
    exploratory = list(data.get("exploratory_attributes") or ["gender", "region"])
    return ExperimentConfig(
        name=name,
        language=language,
        variants=variants,
        render=render,
        input_mode=input_mode,
        screenshot=screenshot,
        personas=personas,
        scenario=str(_require(data, "scenario", str(path))).strip(),
        questions=questions,
        models=models,
        samples_per_persona=samples,
        temperature=float(data.get("temperature", 0.5)),
        max_output_tokens=int(data.get("max_output_tokens", 400)),
        assumed_output_tokens=int(data.get("assumed_output_tokens", 120)),
        max_page_chars=int(data.get("max_page_chars", 8000)),
        ssr=ssr,
        decisions=decisions,
        bootstrap=bootstrap,
        exploratory_attributes=[str(item) for item in exploratory],
        source_path=path.resolve(),
    )
