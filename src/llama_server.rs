//! llama-server's native routes: `POST /completion`, `/infill`,
//! `/embedding`, `/apply-template` and `/tokenize`'s pieces, and the
//! generation settings `/props` reports.
//!
//! The prompt forms, infill prompt, options and runners are
//! [`crate::completions`]'s; this module reads llama-server's request fields
//! and frames llama-server's answers.  A request that asks for something
//! the engine does not do (`grammar`, multimodal prompts) is refused rather
//! than ignored.  A request's `lora` scales and `GET`/`POST /lora-adapters`
//! run the adapters [`crate::lora_registry`] holds.

use std::sync::Arc;

use axum::{
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
    Extension, Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::completions::{
    build_jobs, complete_all, infill_prompt, options, prompts, prompts_in, stream_all, Build, Job,
    Progress, PromptForms,
};
use crate::engine::{Completion, Engine, PromptPart, RawPrompt};
use crate::fim::InfillChunk;
use crate::sampling::SamplingProfile;
use crate::job_admission::JobLease;
use crate::server::{
    check_embedding_inputs, embed_prompts, embedding_response, select_model, ApiError, AppState,
    EMBEDDING_RESPONSE_LIMIT,
};
use crate::types::{
    ChatCompletionRequest, ConstraintSpec, GenerationOptions, TokenLogprob, MAX_TOKENS_UNLIMITED,
};

/// The fields of a `/completion` or `/infill` request that are not sampling
/// settings.
#[derive(Debug, Deserialize)]
struct LlamaRequest {
    /// Not llama-server's (it serves one model): picks one of several.
    #[serde(default)]
    model: String,
    #[serde(default)]
    prompt: Option<Value>,
    #[serde(default)]
    stream: Option<bool>,
    /// Put the generated token ids in `tokens`.
    #[serde(default)]
    return_tokens: Option<bool>,
    /// Report each generated token's probability with this many
    /// alternatives, in `completion_probabilities`.
    #[serde(default)]
    n_probs: Option<u32>,
    /// Probabilities after the sampler chain rather than the model's own;
    /// only the model's are reported.
    #[serde(default)]
    post_sampling_probs: Option<bool>,
    #[serde(default)]
    grammar: Option<String>,
    #[serde(default)]
    json_schema: Option<Value>,
    #[serde(default)]
    response_format: Option<Value>,
    // `/infill`'s.
    #[serde(default)]
    input_prefix: Option<String>,
    #[serde(default)]
    input_suffix: Option<String>,
    #[serde(default)]
    input_extra: Option<Vec<InfillChunk>>,
}

impl LlamaRequest {
    fn parse(body: &Value) -> Result<Self, ApiError> {
        let req: Self = serde_json::from_value(body.clone())
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        if let Some(n) = req.n_probs {
            crate::logprobs::count("n_probs", i64::from(n)).map_err(ApiError::bad_request)?;
        }
        if req.post_sampling_probs == Some(true) {
            return Err(ApiError::bad_request(
                "`post_sampling_probs` is not supported; probabilities are the model's own",
            ));
        }
        if req.grammar.as_deref().is_some_and(|g| !g.trim().is_empty()) {
            return Err(ApiError::bad_request(
                "`grammar` (GBNF) is not supported; use `json_schema`",
            ));
        }
        Ok(req)
    }

    /// `json_schema` (llama-server's) or `response_format` (its OpenAI
    /// spelling); `json_schema: {}` is any JSON object, as in llama-server.
    fn constraint(&self) -> Result<Option<ConstraintSpec>, ApiError> {
        match (&self.json_schema, &self.response_format) {
            (Some(_), Some(_)) => Err(ApiError::bad_request(
                "give `json_schema` or `response_format`, not both",
            )),
            (Some(Value::Null), None) | (None, None) => Ok(None),
            (Some(Value::Object(o)), None) if o.is_empty() => Ok(Some(ConstraintSpec::JsonObject)),
            (Some(schema), None) => Ok(Some(ConstraintSpec::JsonSchema {
                schema: schema.clone(),
            })),
            (None, Some(format)) => {
                crate::types::parse_response_format(format).map_err(ApiError::bad_request)
            }
        }
    }
}

/// llama-server's `generation_settings`: the sampler settings a request ran
/// with, under llama-server's names (`n_predict` -1 is no cap, `seed`
/// 4294967295 a random one).
pub(crate) fn generation_settings(options: &GenerationOptions) -> Value {
    let n_predict = match options.max_tokens {
        MAX_TOKENS_UNLIMITED => -1,
        n => i64::from(n),
    };
    json!({
        "n_predict": n_predict,
        "seed": options.seed.unwrap_or(u64::from(u32::MAX)),
        "temperature": options.temperature,
        "dynatemp_range": 0.0,
        "top_k": options.top_k,
        "top_p": options.top_p,
        "min_p": options.min_p,
        "top_n_sigma": options.top_n_sigma,
        "xtc_probability": options.xtc_probability,
        "xtc_threshold": options.xtc_threshold,
        "typical_p": options.typical_p,
        "repeat_last_n": options.repeat_last_n,
        "repeat_penalty": options.repetition_penalty,
        "presence_penalty": options.presence_penalty,
        "frequency_penalty": options.frequency_penalty,
        "dry_multiplier": options.dry.multiplier,
        "dry_base": options.dry.base,
        "dry_allowed_length": options.dry.allowed_length,
        "dry_penalty_last_n": options.dry.penalty_last_n,
        "dry_sequence_breakers": options.dry.sequence_breakers,
        "mirostat": options.mirostat,
        "mirostat_tau": options.mirostat_tau,
        "mirostat_eta": options.mirostat_eta,
        "stop": options.stop_sequences,
        "max_tokens": n_predict,
        "ignore_eos": options.ignore_eos,
        "min_tokens": options.min_tokens,
        "n_probs": options.logprobs.unwrap_or(0),
        "grammar": "",
    })
}

/// How each result of a request is framed.
#[derive(Clone)]
struct Frame {
    model: String,
    settings: Value,
    return_tokens: bool,
    /// The engine that names the tokens of `completion_probabilities`,
    /// when the request set `n_probs`.
    probs: Option<Arc<Engine>>,
}

/// What a result reports about its prompt: the prompt's text and whether
/// `truncate_prompt_tokens` cut it.
struct JobFacts {
    prompt: String,
    truncated: bool,
}

impl JobFacts {
    fn of(job: &Job) -> Self {
        Self {
            prompt: job.echo.clone(),
            truncated: job.truncated,
        }
    }
}

impl Frame {
    /// A piece of streamed text, with its tokens' probabilities.
    fn partial(&self, index: usize, content: &str, entries: &[TokenLogprob]) -> Value {
        let mut value = json!({
            "index": index, "content": content, "tokens": [], "stop": false, "id_slot": -1,
        });
        if let Some(engine) = &self.probs {
            value["completion_probabilities"] = crate::logprobs::llama(engine, entries);
        }
        value
    }

    /// A finished prompt's result (llama-server's final response), with
    /// the probabilities of `entries`: all the prompt's tokens, or those a
    /// stream has not yet sent.
    fn result(
        &self,
        index: usize,
        content: &str,
        job: &JobFacts,
        completion: &Completion,
        entries: &[TokenLogprob],
    ) -> Value {
        let mut value = self.result_fields(index, content, job, completion);
        if let Some(engine) = &self.probs {
            value["completion_probabilities"] = crate::logprobs::llama(engine, entries);
        }
        value
    }

    fn result_fields(
        &self,
        index: usize,
        content: &str,
        job: &JobFacts,
        completion: &Completion,
    ) -> Value {
        json!({
            "index": index,
            "content": content,
            "tokens": if self.return_tokens { completion.token_ids.as_slice() } else { &[] },
            "id_slot": -1,
            "stop": true,
            "model": self.model,
            "tokens_predicted": completion.usage.completion_tokens,
            "tokens_evaluated": completion.usage.prompt_tokens,
            "generation_settings": self.settings,
            "prompt": job.prompt,
            "has_new_line": completion.text.contains('\n'),
            "truncated": job.truncated,
            "stop_type": completion.stop_type.as_str(),
            "stopping_word": completion.stop_sequence.as_deref().unwrap_or(""),
            "tokens_cached": completion.cached_tokens,
            "timings": completion.timings,
        })
    }
}

/// Run a `/completion` or `/infill` request's jobs and answer as
/// llama-server does: one result object for one prompt, an array for
/// several; streamed, a `data:` event per piece of text and the result
/// (with empty `content`) last, with no `[DONE]`.
async fn answer(
    state: &AppState,
    engine: Arc<Engine>,
    jobs: Vec<Job>,
    options: GenerationOptions,
    frame: Frame,
    stream: bool,
) -> Result<Response, ApiError> {
    if stream {
        let facts: Vec<JobFacts> = jobs.iter().map(JobFacts::of).collect();
        let framer = move |progress: Progress<'_>| -> Option<Value> {
            Some(match progress {
                // llama-server reports no prompt log-probabilities.
                Progress::Prompt(..) => return None,
                Progress::Text(index, text, entries) => frame.partial(index, text, entries),
                Progress::Done(index, completion, entries) => {
                    frame.result(index, "", &facts[index], completion, entries)
                }
            })
        };
        let error =
            json!({"error": {"code": 500, "message": "generation failed", "type": "server_error"}});
        return Ok(stream_all(
            engine,
            jobs,
            options,
            Arc::clone(&state.metrics),
            framer,
            error,
            None,
        ));
    }
    let completions = complete_all(&engine, &jobs, options, &state.metrics).await?;
    let mut results: Vec<Value> = completions
        .iter()
        .zip(&jobs)
        .enumerate()
        .map(|(index, (completion, job))| {
            frame.result(
                index,
                &completion.text,
                &JobFacts::of(job),
                completion,
                &completion.logprobs,
            )
        })
        .collect();
    Ok(match results.len() {
        1 => Json(results.pop().unwrap_or_default()).into_response(),
        _ => Json(Value::Array(results)).into_response(),
    })
}

/// llama.cpp's default seed, `4294967295` (`LLAMA_DEFAULT_SEED`, which
/// `generation_settings` report for a random one), means a random seed in
/// llama-server; the shared mapping reads only a negative seed as random
/// (an OpenAI request's `4294967295` stays fixed), so it becomes `-1`.
fn random_seed_spelled_out(body: &Value) -> std::borrow::Cow<'_, Value> {
    if body.get("seed").and_then(Value::as_u64) != Some(u64::from(u32::MAX)) {
        return std::borrow::Cow::Borrowed(body);
    }
    let mut body = body.clone();
    body["seed"] = json!(-1);
    std::borrow::Cow::Owned(body)
}

/// The pieces every generating route shares: the request, its model, its
/// options and how its results are framed.
async fn prepare(
    state: &AppState,
    headers: &HeaderMap,
    body: &Value,
) -> Result<(LlamaRequest, Arc<Engine>, GenerationOptions, Frame), ApiError> {
    let req = LlamaRequest::parse(body)?;
    let constraint = req.constraint()?;
    let (model, engine) = select_model(state, &req.model).await?;
    let mut options = options(
        &engine,
        &random_seed_spelled_out(body),
        SamplingProfile::LlamaCpp,
        constraint,
        headers,
    )?;
    options.logprobs = req.n_probs.filter(|n| *n > 0);
    let frame = Frame {
        model,
        settings: generation_settings(&options),
        return_tokens: req.return_tokens == Some(true),
        probs: options.logprobs.map(|_| Arc::clone(&engine)),
    };
    Ok((req, engine, options, frame))
}

/// `POST /completion` (and `/completions`): complete a raw prompt — a
/// string, token ids, a mix of both, or an array of those for several.
pub(crate) async fn completion(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let (req, engine, options, frame) = prepare(&state, &headers, &body).await?;
    if req.prompt.as_ref().is_some_and(|p| {
        p.is_object()
            || p.as_array()
                .is_some_and(|items| items.iter().any(Value::is_object))
    }) {
        return Err(ApiError::bad_request(
            "multimodal prompts are not supported",
        ));
    }
    let prompts =
        prompts(req.prompt.as_ref(), PromptForms::LlamaCpp).map_err(ApiError::bad_request)?;
    let build = Build {
        suffix: None,
        special: true,
        echo: true,
    };
    let jobs = build_jobs(&engine, build, &options, prompts).await?;
    answer(
        &state,
        engine,
        jobs,
        options,
        frame,
        req.stream == Some(true),
    )
    .await
}

/// `POST /infill`: fill in the middle between `input_prefix` and
/// `input_suffix`, with `input_extra` files as context and `prompt` as the
/// start of the middle, in the model's FIM format.
pub(crate) async fn infill(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let (req, engine, options, frame) = prepare(&state, &headers, &body).await?;
    let prompt = match &req.prompt {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(_) => {
            return Err(ApiError::bad_request(
                "`prompt` of /infill must be a string",
            ))
        }
    };
    let prefix = req.input_prefix.clone().unwrap_or_default();
    let suffix = req.input_suffix.clone().unwrap_or_default();
    let extra = req.input_extra.clone().unwrap_or_default();
    let jobs = {
        let engine = Arc::clone(&engine);
        let options = options.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<Job>, ApiError> {
            let prompt = infill_prompt(&engine, &prefix, &suffix, &extra, &prompt, &options, true)?;
            let truncated = engine
                .check_prompt(&prompt, &options)
                .map_err(ApiError::from)?;
            let ids: Vec<u32> = prompt
                .parts
                .iter()
                .flat_map(|part| match part {
                    PromptPart::Tokens(ids) => ids.as_slice(),
                    PromptPart::Text(_) => &[],
                })
                .copied()
                .collect();
            let echo = engine.detokenize(&ids, false).map_err(ApiError::from)?;
            Ok(vec![Job {
                prompt,
                echo,
                truncated,
            }])
        })
        .await
        .map_err(|e| ApiError::internal(e.to_string()))??
    };
    answer(
        &state,
        engine,
        jobs,
        options,
        frame,
        req.stream == Some(true),
    )
    .await
}

/// `POST /embedding` (and `/embeddings`): llama-server's own embedding
/// shape, `[{"index": i, "embedding": [[...]]}]`, one pooled, L2-normalized
/// vector per input.
pub(crate) async fn embedding(
    State(state): State<AppState>,
    Extension(lease): Extension<Arc<JobLease>>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let field = if body.get("content").is_none() && body.get("input").is_some() {
        "input"
    } else {
        "content"
    };
    let prompts: Vec<RawPrompt> = prompts_in(field, body.get(field), PromptForms::LlamaCpp)
        .map_err(ApiError::bad_request)?
        .into_iter()
        .map(|prompt| prompt.raw(true))
        .collect();
    check_embedding_inputs(prompts.len())?;
    match body.get("embd_normalize") {
        None | Some(Value::Null) => {}
        Some(n) if n.as_i64() == Some(2) => {}
        Some(_) => {
            return Err(ApiError::bad_request(
                "only `embd_normalize: 2` (L2, the default) is supported",
            ))
        }
    }
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let (_, engine) = select_model(&state, model).await?;
    let (vectors, _) = embed_prompts(&state, &engine, prompts, Some(lease)).await?;
    let results: Vec<Value> = vectors
        .into_iter()
        .enumerate()
        .map(|(index, vector)| json!({"index": index, "embedding": [vector]}))
        .collect();
    embedding_response(&results, EMBEDDING_RESPONSE_LIMIT)
}

/// `POST /apply-template`: the prompt a chat request renders to, without
/// running the model: `{"prompt": "..."}`.
pub(crate) async fn apply_template(
    State(state): State<AppState>,
    Json(mut body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    if let Some(fields) = body.as_object_mut() {
        fields.entry("model").or_insert_with(|| json!(""));
    }
    let req: ChatCompletionRequest =
        serde_json::from_value(body).map_err(|e| ApiError::bad_request(e.to_string()))?;
    let (_, engine) = select_model(&state, &req.model).await?;
    let options = req
        .to_generation_options_with(&engine.sampling_defaults(SamplingProfile::LlamaCpp))
        .map_err(ApiError::bad_request)?;
    let tools = req.offered_tools();
    let prompt = tokio::task::spawn_blocking(move || {
        engine.apply_chat_template(&req.messages, tools.as_deref(), &options)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;
    Ok(Json(json!({ "prompt": prompt })))
}

/// `/tokenize`'s `with_pieces` entries: each id with its piece, as text
/// when it is whole UTF-8 and as its bytes when not.
pub(crate) fn pieces(engine: &Engine, ids: &[u32]) -> crate::Result<Vec<Value>> {
    Ok(ids
        .iter()
        .zip(engine.token_pieces(ids)?)
        .map(|(id, bytes)| match String::from_utf8(bytes) {
            Ok(piece) => json!({"id": id, "piece": piece}),
            Err(e) => json!({"id": id, "piece": e.into_bytes()}),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: Value) -> Result<LlamaRequest, ApiError> {
        LlamaRequest::parse(&body)
    }

    #[test]
    fn unsupported_fields_are_refused() {
        assert!(request(json!({"n_probs": 3})).is_ok());
        assert!(request(json!({"n_probs": 21})).is_err());
        assert!(request(json!({"n_probs": 3, "post_sampling_probs": true})).is_err());
        assert!(request(json!({"grammar": "root ::= \"a\""})).is_err());
        assert!(request(json!({"n_probs": 0, "grammar": "", "lora": []})).is_ok());
    }

    #[test]
    fn json_schema_constrains_the_output() {
        // `None` is a refused request.
        let constraint = |body: Value| request(body).ok()?.constraint().ok();
        assert!(matches!(constraint(json!({})), Some(None)));
        assert!(matches!(
            constraint(json!({"json_schema": {}})),
            Some(Some(ConstraintSpec::JsonObject))
        ));
        assert!(matches!(
            constraint(json!({"json_schema": {"type": "array"}})),
            Some(Some(ConstraintSpec::JsonSchema { .. }))
        ));
        assert!(matches!(
            constraint(json!({"response_format": {"type": "json_object"}})),
            Some(Some(ConstraintSpec::JsonObject))
        ));
        assert!(
            constraint(json!({"json_schema": {}, "response_format": {"type": "json_object"}}))
                .is_none()
        );
    }

    #[test]
    fn llama_cpp_default_seed_is_random() {
        let seed = |body: Value| random_seed_spelled_out(&body)["seed"].clone();
        assert_eq!(seed(json!({"seed": 4294967295u64})), -1);
        assert_eq!(seed(json!({"seed": 42})), 42);
        assert_eq!(seed(json!({})), Value::Null);
    }

    #[test]
    fn settings_use_llama_server_names() {
        let options = GenerationOptions::with_defaults(&SamplingProfile::LlamaCpp.defaults());
        let settings = generation_settings(&options);
        assert_eq!(settings["n_predict"], -1);
        assert_eq!(settings["seed"], u64::from(u32::MAX));
        assert_eq!(
            settings["repeat_penalty"],
            json!(options.repetition_penalty)
        );
    }
}
