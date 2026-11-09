"""The `openai` Python SDK against joshua's OpenAI / vLLM routes."""

import math

import openai
import pytest


@pytest.fixture(scope="module")
def client(url):
    return openai.OpenAI(base_url=f"{url}/v1", api_key="unused", max_retries=0)


def test_models(client, model):
    ids = [m.id for m in client.models.list()]
    assert model in ids
    card = client.models.retrieve(model)
    assert card.id == model
    with pytest.raises(openai.NotFoundError):
        client.models.retrieve("no-such-model")


def test_chat(client, model):
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=8
    )
    choice = r.choices[0]
    assert choice.message.role == "assistant"
    assert isinstance(choice.message.content, str)
    assert choice.finish_reason in ("stop", "length")
    assert r.usage.prompt_tokens > 0
    assert r.usage.completion_tokens <= 8
    assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens


def test_chat_without_max_tokens_runs_to_a_stop(client, model):
    # No max_tokens: generation ends at end of sequence or of the context,
    # never at a hidden default cap.
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}])
    assert r.choices[0].finish_reason in ("stop", "length")
    assert r.usage.completion_tokens > 0


def test_chat_stream_with_usage(client, model):
    chunks = list(
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert chunks
    finishes = [c.choices[0].finish_reason for c in chunks if c.choices]
    assert finishes[-1] in ("stop", "length")
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens <= 8
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert isinstance(text, str)


def test_chat_logprobs(client, model):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4,
        logprobs=True,
        top_logprobs=2,
    )
    content = r.choices[0].logprobs.content
    assert len(content) == r.usage.completion_tokens
    for token in content:
        assert token.logprob <= 0
        assert len(token.top_logprobs) == 2


def test_chat_with_tools_is_accepted(client, model):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "weather in Paris?"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=8,
    )
    assert r.choices[0].finish_reason in ("stop", "length", "tool_calls")


def test_vllm_sampling_parameters(client, model):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4,
        temperature=0.7,
        top_p=0.9,
        seed=7,
        extra_body={"top_k": 20, "min_p": 0.05, "repetition_penalty": 1.1},
    )
    assert r.choices[0].finish_reason in ("stop", "length")


def test_bad_request_is_a_400(client, model):
    with pytest.raises(openai.BadRequestError):
        # OpenAI caps top_logprobs at 20; asking for more is a request error.
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4,
            logprobs=True,
            top_logprobs=1000,
        )


def test_completions(client, model):
    r = client.completions.create(model=model, prompt="Once upon", max_tokens=6, echo=True)
    assert r.choices[0].text.startswith("Once upon")
    chunks = list(client.completions.create(model=model, prompt="Once upon", max_tokens=6, stream=True))
    assert chunks and chunks[-1].choices[0].finish_reason in ("stop", "length")


def test_embeddings_float_and_base64_agree(client, model):
    # The SDK asks for base64 unless told otherwise, and decodes it.
    b64 = client.embeddings.create(model=model, input=["hello", "world"])
    flt = client.embeddings.create(model=model, input=["hello", "world"], encoding_format="float")
    assert len(b64.data) == 2
    for a, b in zip(b64.data, flt.data):
        assert len(a.embedding) == len(b.embedding) > 0
        assert all(math.isfinite(x) for x in a.embedding)
        assert all(abs(x - y) < 1e-6 for x, y in zip(a.embedding, b.embedding))


def test_responses(client, model):
    r = client.responses.create(model=model, input="hi", max_output_tokens=8)
    assert r.status in ("completed", "incomplete")
    assert isinstance(r.output_text, str)
    again = client.responses.retrieve(r.id)
    assert again.id == r.id
    events = [e.type for e in client.responses.create(model=model, input="hi", max_output_tokens=4, stream=True)]
    assert events[0] == "response.created"
    assert events[-1] in ("response.completed", "response.incomplete")
