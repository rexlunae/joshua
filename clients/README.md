# Real-client tests

These tests drive a real `joshua serve` with the client libraries people use
against llama.cpp, vLLM and Ollama servers, rather than hand-written HTTP:

- `python/test_openai_client.py`: the `openai` Python SDK (OpenAI and vLLM
  clients), including vLLM's `extra_body` sampling parameters.
- `python/test_ollama_client.py`: the `ollama` Python library.
- `node/test.mjs`: the `openai` and `ollama` Node libraries (openai-node asks
  for base64 embeddings by default).

The model is the test suite's tiny Qwen3 (random weights and a tokenizer of JSON
characters, so `format: json` can be satisfied), so the tests check protocol
shapes, not answers.

Run them all with `scripts/client-tests.sh`, which builds joshua, writes the
model, starts the server and runs both suites; CI runs it on every push.  To
run one suite against a server you started yourself, set `JOSHUA_URL` (and
`JOSHUA_MODEL`, the model's id) and run `pytest clients/python` or
`node --test clients/node/test.mjs` after installing `requirements.txt` /
`npm ci` in `clients/node`.
