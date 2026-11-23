# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from ssr_abtest.cache import DiskCache
from ssr_abtest.providers.decisions import build_decision_questions
from ssr_abtest.providers.openai_provider import OpenAIProvider


class _Responses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="I would look at one more detail before deciding.",
            usage=SimpleNamespace(input_tokens=15, output_tokens=9, total_tokens=24),
        )


class _Embeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        # Reverse the payload so the provider must honor the index field.
        data = [
            SimpleNamespace(index=index, embedding=[float(index), 0.5])
            for index in range(len(kwargs["input"]) - 1, -1, -1)
        ]
        return SimpleNamespace(
            data=data,
            usage=SimpleNamespace(prompt_tokens=7, total_tokens=7),
        )


class _Decisions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            answers=[
                SimpleNamespace(
                    type="predicate",
                    name="damaged",
                    probability=0.95,
                )
            ],
            usage=SimpleNamespace(
                input_tokens=42,
                output_tokens=0,
                total_tokens=42,
                input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
            ),
        )


class _Client:
    def __init__(self):
        self.responses = _Responses()
        self.embeddings = _Embeddings()
        self.decisions = _Decisions()


def test_generate_embed_and_decide_use_the_documented_calls(tmp_path):
    client = _Client()
    provider = OpenAIProvider(client=client, cache=DiskCache(tmp_path))
    first = provider.generate(
        model="gpt-5.4-mini",
        instructions="Stay in character.",
        user_input="Page text",
        temperature=0.5,
        max_output_tokens=400,
        sample_index=0,
    )
    assert first.text.startswith("I would look")
    assert client.responses.calls[0]["model"] == "gpt-5.4-mini"
    assert client.responses.calls[0]["temperature"] == 0.5
    assert client.responses.calls[0]["text"] == {"format": {"type": "text"}}
    cached = provider.generate(
        model="gpt-5.4-mini",
        instructions="Stay in character.",
        user_input="Page text",
        temperature=0.5,
        max_output_tokens=400,
        sample_index=0,
    )
    assert cached.text == first.text
    assert len(client.responses.calls) == 1

    embedded = provider.embed(["alpha", "beta"], model="text-embedding-3-small")
    assert client.embeddings.calls[0]["model"] == "text-embedding-3-small"
    assert embedded.vectors[0] == [0.0, 0.5]
    assert embedded.vectors[1] == [1.0, 0.5]
    assert embedded.input_tokens == 7
    provider.embed(["alpha"], model="text-embedding-3-small")
    assert len(client.embeddings.calls) == 1

    questions = build_decision_questions("Would you continue?")
    decided = provider.decide(model="gpt-6-luna", input_text="The answer.", questions=questions)
    sent = client.decisions.calls[0]
    assert sent["model"] == "gpt-6-luna"
    assert sent["input"] == "The answer."
    assert sent["questions"][0]["type"] == "score"
    assert sent["questions"][0]["levels"][0]["label"] == "1"
    assert decided.payload["answers"][0]["probability"] == pytest.approx(0.95)
    assert decided.output_tokens == 0
    provider.decide(model="gpt-6-luna", input_text="The answer.", questions=questions)
    assert len(client.decisions.calls) == 1


def test_temperature_is_retried_only_when_the_model_rejects_it():
    class _RejectTemperature:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if "temperature" in kwargs:
                raise RuntimeError("unsupported parameter: temperature")
            return SimpleNamespace(
                output_text="Still in character.",
                usage=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5),
            )

    client = _Client()
    client.responses = _RejectTemperature()
    provider = OpenAIProvider(client=client)
    generated = provider.generate(
        model="gpt-5.4-mini",
        instructions="Stay in character.",
        user_input="Page text",
        temperature=0.5,
        max_output_tokens=400,
        sample_index=1,
    )
    assert generated.text == "Still in character."
    assert "temperature" in client.responses.calls[0]
    assert "temperature" not in client.responses.calls[1]


def test_generate_sends_an_image_part(tmp_path):
    from ssr_abtest.render.images import write_placeholder_png

    image = tmp_path / "page.png"
    write_placeholder_png(image)
    client = _Client()
    provider = OpenAIProvider(client=client, cache=DiskCache(tmp_path / "cache"))
    provider.generate(
        model="gpt-5.4-mini",
        instructions="Stay in character.",
        user_input="What do you think of this page?",
        temperature=0.5,
        max_output_tokens=400,
        sample_index=0,
        image_path=str(image),
        image_detail="high",
    )
    content = client.responses.calls[0]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "What do you think of this page?"}
    assert content[1]["type"] == "input_image"
    assert content[1]["detail"] == "high"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    provider.generate(
        model="gpt-5.4-mini",
        instructions="Stay in character.",
        user_input="What do you think of this page?",
        temperature=0.5,
        max_output_tokens=400,
        sample_index=0,
        image_path=str(image),
        image_detail="high",
    )
    assert len(client.responses.calls) == 1


def test_missing_decisions_attribute_is_a_clear_error():
    client = _Client()
    del client.decisions
    provider = OpenAIProvider(client=client)
    with pytest.raises(RuntimeError, match="decisions"):
        provider.decide(model="gpt-6-luna", input_text="x", questions=[{"type": "predicate"}])


def test_constructor_refuses_to_read_a_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIProvider()
