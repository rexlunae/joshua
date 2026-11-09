// The `openai` and `ollama` Node libraries against a running joshua.
// JOSHUA_URL (default http://127.0.0.1:18080) and JOSHUA_MODEL (default
// "tiny") name the server and the model id.

import assert from "node:assert/strict";
import { test } from "node:test";
import OpenAI from "openai";
import { Ollama } from "ollama";

const url = (process.env.JOSHUA_URL ?? "http://127.0.0.1:18080").replace(/\/$/, "");
const model = process.env.JOSHUA_MODEL ?? "tiny";
const openai = new OpenAI({ baseURL: `${url}/v1`, apiKey: "unused", maxRetries: 0 });
const ollama = new Ollama({ host: url });
const hi = [{ role: "user", content: "hi" }];

test("openai: models", async () => {
  const ids = [];
  for await (const m of openai.models.list()) ids.push(m.id);
  assert.ok(ids.includes(model), ids.join(","));
});

test("openai: chat", async () => {
  const r = await openai.chat.completions.create({ model, messages: hi, max_tokens: 8 });
  assert.equal(r.choices[0].message.role, "assistant");
  assert.ok(["stop", "length"].includes(r.choices[0].finish_reason));
  assert.equal(r.usage.total_tokens, r.usage.prompt_tokens + r.usage.completion_tokens);
});

test("openai: chat stream with usage", async () => {
  const stream = await openai.chat.completions.create({
    model,
    messages: hi,
    max_tokens: 8,
    stream: true,
    stream_options: { include_usage: true },
  });
  const chunks = [];
  for await (const c of stream) chunks.push(c);
  const finishes = chunks.filter((c) => c.choices.length).map((c) => c.choices[0].finish_reason);
  assert.ok(["stop", "length"].includes(finishes.at(-1)));
  assert.ok(chunks.at(-1).usage.completion_tokens <= 8);
});

test("openai: embeddings default to base64 and decode", async () => {
  // openai-node asks for base64 and decodes it itself; the vector must
  // match the float form.
  const b64 = await openai.embeddings.create({ model, input: "hello" });
  const float = await openai.embeddings.create({ model, input: "hello", encoding_format: "float" });
  const a = Array.from(b64.data[0].embedding);
  const b = float.data[0].embedding;
  assert.equal(a.length, b.length);
  assert.ok(a.length > 0);
  a.forEach((x, i) => assert.ok(Math.abs(x - b[i]) < 1e-6, `dim ${i}: ${x} vs ${b[i]}`));
});

test("openai: a bad request is a 400", async () => {
  await assert.rejects(
    openai.chat.completions.create({ model, messages: hi, max_tokens: 4, logprobs: true, top_logprobs: 1000 }),
    (e) => e instanceof OpenAI.BadRequestError,
  );
});

test("ollama: list and show", async () => {
  const { models } = await ollama.list();
  assert.ok(models.some((m) => m.model.split(":")[0] === model));
  const shown = await ollama.show({ model });
  assert.ok(shown.model_info);
});

test("ollama: chat", async () => {
  const r = await ollama.chat({ model, messages: hi, options: { num_predict: 8 } });
  assert.equal(r.message.role, "assistant");
  assert.ok(r.done);
  assert.ok(r.eval_count <= 8);
});

test("ollama: chat stream", async () => {
  const parts = [];
  for await (const p of await ollama.chat({ model, messages: hi, stream: true, options: { num_predict: 8 } }))
    parts.push(p);
  assert.ok(parts.at(-1).done);
  assert.ok(parts.slice(0, -1).every((p) => !p.done));
});

test("ollama: generate", async () => {
  const r = await ollama.generate({ model, prompt: "hello", options: { num_predict: 6 } });
  assert.ok(r.done);
  assert.equal(typeof r.response, "string");
});

test("ollama: embed", async () => {
  const r = await ollama.embed({ model, input: ["hello", "world"] });
  assert.equal(r.embeddings.length, 2);
  assert.ok(r.embeddings[0].length > 0);
});

test("ollama: ps", async () => {
  const { models } = await ollama.ps();
  assert.ok(models.some((m) => m.model.split(":")[0] === model));
});
