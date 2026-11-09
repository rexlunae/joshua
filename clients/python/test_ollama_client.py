"""The `ollama` Python library against joshua's Ollama routes."""

import ollama
import pytest

OPTIONS = {"num_predict": 8}


@pytest.fixture(scope="module")
def client(url):
    return ollama.Client(host=url)


def test_version(url):
    # ollama-python has no version call; the CLI's first request is HEAD /,
    # then GET /api/version.
    import httpx

    base = url.rstrip("/")
    assert httpx.head(f"{base}/").status_code == 200
    assert "version" in httpx.get(f"{base}/api/version").json()


def test_list_and_show(client, model):
    names = [m.model for m in client.list().models]
    assert any(n.split(":")[0] == model for n in names), names
    listed = next(m for m in client.list().models if m.model.split(":")[0] == model)
    assert listed.digest and listed.modified_at is not None
    assert listed.details.format == "gguf"
    shown = client.show(model)
    assert shown.modelinfo
    assert "completion" in (shown.capabilities or [])


def test_chat(client, model):
    r = client.chat(model=model, messages=[{"role": "user", "content": "hi"}], options=OPTIONS)
    assert r.message.role == "assistant"
    assert isinstance(r.message.content, str)
    assert r.done and r.done_reason in ("stop", "length")
    assert r.eval_count <= 8 and r.prompt_eval_count > 0


def test_chat_stream(client, model):
    parts = list(client.chat(model=model, messages=[{"role": "user", "content": "hi"}], options=OPTIONS, stream=True))
    assert parts[-1].done
    assert all(not p.done for p in parts[:-1])


def test_chat_format_json(client, model):
    r = client.chat(
        model=model,
        messages=[{"role": "user", "content": "a json object please"}],
        format="json",
        options={"num_predict": 32},
    )
    assert isinstance(r.message.content, str)


def test_generate_with_context(client, model):
    first = client.generate(model=model, prompt="hello", options=OPTIONS)
    assert first.done and first.context
    second = client.generate(model=model, prompt="again", context=first.context, options=OPTIONS)
    assert second.done
    parts = list(client.generate(model=model, prompt="hello", options=OPTIONS, stream=True))
    assert parts[-1].done


def test_embed(client, model):
    r = client.embed(model=model, input=["hello", "world"])
    assert len(r.embeddings) == 2 and len(r.embeddings[0]) > 0
    legacy = client.embeddings(model=model, prompt="hello")
    assert len(legacy.embedding) == len(r.embeddings[0])


def test_ps(client, model):
    running = [m.model.split(":")[0] for m in client.ps().models]
    assert model in running


def test_copy_and_delete(client, model):
    client.copy(model, "tiny-copy")
    assert any(m.model.split(":")[0] == "tiny-copy" for m in client.list().models)
    client.delete("tiny-copy")
    assert not any(m.model.split(":")[0] == "tiny-copy" for m in client.list().models)
    with pytest.raises(ollama.ResponseError) as e:
        client.delete("tiny-copy")
    assert e.value.status_code == 404
