//! `POST /v1/completions`: OpenAI's and vLLM's raw completion endpoint.
//!
//! The prompt is completed as given, with no chat template: a string, an
//! array of token ids, or an array of either for one choice per prompt.
//! `stream` sends server-sent `text_completion` chunks ending in
//! `data: [DONE]`; `echo` repeats the prompt in front of the text; `suffix`
//! builds a fill-in-the-middle prompt for a model with FIM tokens; and
//! `response_format` constrains the output as on the chat route; `logprobs`
//! reports each generated token's log-probability with that many
//! alternatives, and with `echo` the prompt's too; vLLM's
//! `prompt_logprobs` reports the prompt's in its own shape (not when
//! streaming, as vLLM).  Sampling fields are the shared
//! [`SamplingParams`].  `n > 1` and `best_of > 1` are refused rather than
//! ignored.
//!
//! The prompt forms, the infill prompt, the request's options and the
//! runners that complete a request's prompts are shared with llama-server's
//! native `/completion` and `/infill` ([`crate::llama_server`]); each route
//! only frames the results for its clients.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use axum::{
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::engine::{Completion, Engine, PromptPart, RawPrompt};
use crate::fim::InfillChunk;
use crate::sampling::SamplingProfile;
use crate::server::{select_model, sse_response, ApiError, AppState, CancelOnDrop, ServerMetrics};
use crate::types::{
    unix_time, ConstraintSpec, GenerationOptions, LogprobSink, SamplingParams, TokenLogprob,
    UsageInfo, MAX_TOKENS_UNLIMITED,
};

/// The fields of a completion request that are not sampling settings.
#[derive(Debug, Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: String,
    prompt: Option<Value>,
    #[serde(default)]
    suffix: Option<String>,
    #[serde(default)]
    echo: Option<bool>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    logprobs: Option<Value>,
    /// vLLM's: the prompt's log-probabilities, with this many alternatives.
    #[serde(default)]
    prompt_logprobs: Option<i64>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    best_of: Option<u32>,
    /// vLLM's switch for the tokenizer's BOS on a text prompt (on by
    /// default).
    #[serde(default)]
    add_special_tokens: Option<bool>,
    #[serde(default)]
    response_format: Option<Value>,
}

/// One prompt of a request, as the client gave it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
    /// Text and token ids in one prompt (llama-server only).
    Mixed(Vec<PromptPart>),
}

/// How a request's `prompt` array is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptForms {
    /// OpenAI's and vLLM's: a string, token ids, or an array of either.
    OpenAi,
    /// llama-server's: those, plus an array mixing strings and token ids,
    /// which is one prompt, at the top level or as an element.
    LlamaCpp,
}

impl Prompt {
    /// The prompt for the engine; `special` lets the tokenizer add its
    /// special tokens (BOS) to leading text.
    pub(crate) fn raw(self, special: bool) -> RawPrompt {
        match self {
            Prompt::Text(text) => RawPrompt::text(text).with_special_tokens(special),
            Prompt::Tokens(ids) => RawPrompt::tokens(ids),
            Prompt::Mixed(parts) => RawPrompt::mixed(parts).with_special_tokens(special),
        }
    }
}

/// The request's `prompt` as one or more prompts.
pub(crate) fn prompts(prompt: Option<&Value>, forms: PromptForms) -> Result<Vec<Prompt>, String> {
    prompts_in("prompt", prompt, forms)
}

/// The request field `field`, read the way [`prompts`] reads `prompt`
/// (an embedding request's `input` takes the same forms).
pub(crate) fn prompts_in(
    field: &str,
    prompt: Option<&Value>,
    forms: PromptForms,
) -> Result<Vec<Prompt>, String> {
    let id = |v: &Value| v.as_u64().and_then(|id| u32::try_from(id).ok());
    let ids = |items: &[Value]| -> Option<Vec<u32>> { items.iter().map(id).collect() };
    // llama-server's mixed prompt: any array holding both a string and an
    // integer, every item one or the other.
    let mixed = |items: &[Value]| -> Option<Vec<PromptPart>> {
        if forms != PromptForms::LlamaCpp
            || !items.iter().any(Value::is_string)
            || !items.iter().any(Value::is_number)
        {
            return None;
        }
        let mut parts: Vec<PromptPart> = Vec::new();
        for item in items {
            match (item, parts.last_mut()) {
                (Value::String(s), _) => parts.push(PromptPart::Text(s.clone())),
                (n, Some(PromptPart::Tokens(run))) => run.push(id(n)?),
                (n, _) => parts.push(PromptPart::Tokens(vec![id(n)?])),
            }
        }
        Some(parts)
    };
    let invalid = || match forms {
        PromptForms::OpenAi => {
            format!("`{field}` must be a string, an array of token ids, or an array of either")
        }
        PromptForms::LlamaCpp => format!(
            "`{field}` must be a string, an array of token ids or strings and token ids, \
             or an array of those"
        ),
    };
    let out = match prompt {
        None | Some(Value::Null) => return Err(format!("Missing required field '{field}'")),
        Some(Value::String(s)) => vec![Prompt::Text(s.clone())],
        Some(Value::Array(items)) if items.is_empty() => return Err(invalid()),
        Some(Value::Array(items)) if items.iter().all(Value::is_number) => {
            vec![Prompt::Tokens(ids(items).ok_or_else(invalid)?)]
        }
        Some(Value::Array(items))
            if items.iter().all(|v| v.is_string() || v.is_number())
                && items.iter().any(Value::is_number) =>
        {
            vec![Prompt::Mixed(mixed(items).ok_or_else(invalid)?)]
        }
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::String(s) => Ok(Prompt::Text(s.clone())),
                Value::Array(t) if !t.is_empty() && t.iter().all(Value::is_number) => {
                    ids(t).map(Prompt::Tokens).ok_or_else(invalid)
                }
                Value::Array(t) => mixed(t).map(Prompt::Mixed).ok_or_else(invalid),
                _ => Err(invalid()),
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(invalid()),
    };
    Ok(out)
}

impl CompletionRequest {
    /// Refuse what this endpoint does not do, rather than ignore it, and
    /// read `logprobs` and `prompt_logprobs`: how many alternatives to
    /// report with each token (0 reports the token's own), or `None` for
    /// no log-probabilities.
    fn check(&self) -> Result<(Option<u32>, Option<u32>), String> {
        if self.n.is_some_and(|n| n != 1) {
            return Err("only n = 1 is supported".to_string());
        }
        if self.best_of.is_some_and(|n| n > 1) {
            return Err("best_of > 1 is not supported".to_string());
        }
        let logprobs = match &self.logprobs {
            None | Some(Value::Null) | Some(Value::Bool(false)) => None,
            Some(Value::Number(n)) => Some(crate::logprobs::count(
                "logprobs",
                n.as_i64().ok_or("`logprobs` must be a whole number")?,
            )?),
            Some(_) => return Err("`logprobs` must be a whole number".to_string()),
        };
        let prompt_logprobs = self
            .prompt_logprobs
            .map(|n| crate::logprobs::count("prompt_logprobs", n))
            .transpose()?;
        if prompt_logprobs.is_some() && self.stream == Some(true) {
            return Err("`prompt_logprobs` is not available when streaming".to_string());
        }
        let infill = self.suffix.as_deref().is_some_and(|s| !s.is_empty());
        if infill && logprobs.is_some() && self.echo == Some(true) {
            // The scored prompt is the infill template, not the echoed text.
            return Err("`echo` with `logprobs` is not available with `suffix`".to_string());
        }
        Ok((logprobs, prompt_logprobs))
    }
}

/// A prompt as the engine completes it, plus the text `echo` repeats.
pub(crate) struct Job {
    pub(crate) prompt: RawPrompt,
    pub(crate) echo: String,
    /// `truncate_prompt_tokens` cuts the prompt.
    pub(crate) truncated: bool,
}

/// How a request's prompts are built.
pub(crate) struct Build {
    pub(crate) suffix: Option<String>,
    /// The tokenizer's special tokens (BOS) on a text or infill prompt.
    pub(crate) special: bool,
    /// Keep the prompt's text for `echo`.
    pub(crate) echo: bool,
}

/// A fill-in-the-middle prompt in the model's FIM format (see
/// [`Engine::format_infill`]), or a client error for a model without FIM
/// tokens.
pub(crate) fn infill_prompt(
    engine: &Engine,
    prefix: &str,
    suffix: &str,
    extra: &[InfillChunk],
    prompt: &str,
    options: &GenerationOptions,
    special: bool,
) -> Result<RawPrompt, ApiError> {
    if let Some(missing) = engine.fim_tokens().missing() {
        return Err(ApiError::bad_request(format!(
            "fill-in-the-middle needs a model with FIM tokens: {missing}"
        )));
    }
    let n_predict = match options.max_tokens {
        MAX_TOKENS_UNLIMITED => -1,
        n => i64::from(n),
    };
    engine
        .format_infill(prefix, suffix, extra, prompt, n_predict, false, special)
        .map(RawPrompt::tokens)
        .map_err(ApiError::from)
}

/// Build each prompt's [`Job`]: a text prompt with the tokenizer's special
/// tokens unless `add_special_tokens` is false, token ids verbatim, and with
/// `suffix` an infill prompt in the model's FIM format.  Each is checked as
/// generation will check it (token ids, empty or too long for the window),
/// so a bad prompt is a client error before a stream starts.
pub(crate) fn jobs(
    engine: &Engine,
    build: &Build,
    options: &GenerationOptions,
    prompts: Vec<Prompt>,
) -> Result<Vec<Job>, ApiError> {
    let suffix = build.suffix.as_deref().filter(|s| !s.is_empty());
    let special = build.special;
    let text_of = |ids: &[u32]| engine.detokenize(ids, false).map_err(ApiError::from);
    prompts
        .into_iter()
        .map(|prompt| {
            let echo = match &prompt {
                _ if !build.echo => String::new(),
                Prompt::Text(text) => text.clone(),
                Prompt::Tokens(ids) => text_of(ids)?,
                Prompt::Mixed(parts) => parts
                    .iter()
                    .map(|part| match part {
                        PromptPart::Text(text) => Ok(text.clone()),
                        PromptPart::Tokens(ids) => text_of(ids),
                    })
                    .collect::<Result<String, _>>()?,
            };
            let prompt = match (prompt, suffix) {
                (Prompt::Text(text), Some(suffix)) => {
                    infill_prompt(engine, &text, suffix, &[], "", options, special)?
                }
                (_, Some(_)) => return Err(ApiError::bad_request("`suffix` needs a text prompt")),
                (prompt, None) => prompt.raw(special),
            };
            let truncated = engine
                .check_prompt(&prompt, options)
                .map_err(ApiError::from)?;
            Ok(Job {
                prompt,
                echo,
                truncated,
            })
        })
        .collect()
}

/// A request's generation options: `profile`'s defaults, then its
/// sampling fields (`max_tokens` absent or -1 runs to the end of the
/// sequence) and `constraint`.  A request the model cannot serve is a
/// client error before a stream starts.
pub(crate) fn options(
    engine: &Engine,
    body: &Value,
    profile: SamplingProfile,
    constraint: Option<ConstraintSpec>,
    headers: &HeaderMap,
) -> Result<GenerationOptions, ApiError> {
    let sampling: SamplingParams =
        serde_json::from_value(body.clone()).map_err(|e| ApiError::bad_request(e.to_string()))?;
    let mut options = GenerationOptions::with_defaults(&engine.sampling_defaults(profile));
    sampling
        .apply(&mut options)
        .map_err(ApiError::bad_request)?;
    options.constraint = constraint;
    let key = body.get("prompt_cache_key").and_then(Value::as_str);
    options.cache_scope = Some(crate::affinity::cache_scope(headers, None, key));
    engine.validate_options(&options).map_err(ApiError::from)?;
    Ok(options)
}

/// Build a request's jobs off the async runtime (tokenizing a long prompt
/// is real work).
pub(crate) async fn build_jobs(
    engine: &Arc<Engine>,
    build: Build,
    options: &GenerationOptions,
    prompts: Vec<Prompt>,
) -> Result<Vec<Job>, ApiError> {
    let engine = Arc::clone(engine);
    let options = options.clone();
    tokio::task::spawn_blocking(move || jobs(&engine, &build, &options, prompts))
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
}

/// Complete each job's prompt in turn, until the client goes away.
pub(crate) async fn complete_all(
    engine: &Arc<Engine>,
    jobs: &[Job],
    options: GenerationOptions,
    metrics: &ServerMetrics,
) -> Result<Vec<Completion>, ApiError> {
    let cancel = CancelOnDrop::new();
    let flag = cancel.flag();
    let engine = Arc::clone(engine);
    let prompts: Vec<RawPrompt> = jobs.iter().map(|j| j.prompt.clone()).collect();
    let completions = tokio::task::spawn_blocking(move || {
        prompts
            .iter()
            .map(|p| engine.complete_prompt_cancellable(p, &options, &flag))
            .collect::<crate::Result<Vec<_>>>()
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;
    for completion in &completions {
        metrics.record_generation(&completion.usage);
    }
    Ok(completions)
}

/// `entries` with at most `top` alternatives each.
fn alternatives(entries: &[TokenLogprob], top: u32) -> Vec<TokenLogprob> {
    entries
        .iter()
        .map(|e| {
            let mut e = e.clone();
            e.top.truncate(top as usize);
            e
        })
        .collect()
}

/// A prompt's scored `entries`, with `top` alternatives, placed in the
/// `echo`ed text.  They are placed with special tokens written out, as
/// the echo writes them, and aligned to the echo's end: a prompt
/// truncated at its front starts past the echo's start, and a BOS the
/// tokenizer added (not in the echo) lands at its start.
fn echo_entries(
    engine: &Engine,
    entries: &[TokenLogprob],
    top: u32,
    echo: &str,
) -> Vec<TokenLogprob> {
    let mut entries = alternatives(entries, top);
    let placed = engine.place_prompt(&mut entries, false);
    let echoed = echo.chars().count();
    for entry in &mut entries {
        entry.offset = (entry.offset + echoed).saturating_sub(placed);
    }
    entries
}

/// What a streamed request reports, prompt by prompt.  Each carries the
/// log-probabilities of the tokens decoded since the last report, when the
/// request asked for them.
pub(crate) enum Progress<'a> {
    /// The scored prompt of the prompt at this index, ahead of its text,
    /// when the request asked for the prompt's log-probabilities.
    Prompt(usize, &'a [TokenLogprob]),
    /// New text for the prompt at this index.
    Text(usize, &'a str, &'a [TokenLogprob]),
    /// The prompt at this index is done.
    Done(usize, &'a Completion, &'a [TokenLogprob]),
}

/// Stream each job's completion in turn as server-sent events, each
/// [`Progress`] framed by `frame` (`None` sends nothing).  A failed
/// generation ends the stream with `error`; `last`, when set, ends every
/// stream, a failed one included.  The engine stops when the client goes away.
pub(crate) fn stream_all(
    engine: Arc<Engine>,
    jobs: Vec<Job>,
    mut options: GenerationOptions,
    metrics: Arc<ServerMetrics>,
    mut frame: impl FnMut(Progress<'_>) -> Option<Value> + Send + 'static,
    error: Value,
    last: Option<&'static str>,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let cancel = CancelOnDrop::new();
    let flag = cancel.flag();
    metrics.record_stream();
    tokio::task::spawn_blocking(move || {
        let flag: &AtomicBool = &flag;
        let send = |value: Option<Value>| {
            if let Some(value) = value {
                let _ = tx.send(value.to_string());
            }
        };
        let finish = || {
            if let Some(last) = last {
                let _ = tx.send(last.to_string());
            }
        };
        for (index, job) in jobs.iter().enumerate() {
            // Each prompt's tokens are placed in its own text.
            let wanted = options.logprobs.is_some() || options.prompt_logprobs.is_some();
            let sink = wanted.then(LogprobSink::default);
            options.logprob_sink = sink.clone();
            let take = || sink.as_ref().map(LogprobSink::take).unwrap_or_default();
            // The scored prompt, once, ahead of the first text.
            let prompt = |frame: &mut dyn FnMut(Progress<'_>) -> Option<Value>| {
                let entries = sink
                    .as_ref()
                    .map(LogprobSink::take_prompt)
                    .unwrap_or_default();
                if !entries.is_empty() {
                    send(frame(Progress::Prompt(index, &entries)));
                }
            };
            let result =
                engine.complete_prompt_streaming(&job.prompt, &options, flag, &mut |text| {
                    prompt(&mut frame);
                    send(frame(Progress::Text(index, text, &take())))
                });
            match result {
                Ok(completion) => {
                    metrics.record_generation(&completion.usage);
                    prompt(&mut frame);
                    send(frame(Progress::Done(index, &completion, &take())));
                }
                Err(e) => {
                    tracing::error!("streaming completion failed: {e}");
                    send(Some(error));
                    finish();
                    return;
                }
            }
        }
        finish();
    });
    sse_response(rx, cancel)
}

/// The usage of a request's completions: one completion's as the engine
/// reports it, or every count summed, the token details included (each
/// kept when any completion reports it).
fn total_usage<'a>(usages: impl IntoIterator<Item = &'a UsageInfo>) -> UsageInfo {
    let mut usages = usages.into_iter();
    let Some(first) = usages.next() else {
        return UsageInfo::default();
    };
    usages.fold(first.clone(), |mut total, usage| {
        total.prompt_tokens += usage.prompt_tokens;
        total.completion_tokens += usage.completion_tokens;
        total.total_tokens += usage.total_tokens;
        if let Some(d) = &usage.prompt_tokens_details {
            total
                .prompt_tokens_details
                .get_or_insert_with(Default::default)
                .cached_tokens += d.cached_tokens;
        }
        if let Some(d) = &usage.completion_tokens_details {
            total
                .completion_tokens_details
                .get_or_insert_with(Default::default)
                .reasoning_tokens += d.reasoning_tokens;
        }
        total
    })
}

/// `POST /v1/completions`.
pub(crate) async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let req: CompletionRequest =
        serde_json::from_value(body.clone()).map_err(|e| ApiError::bad_request(e.to_string()))?;
    let (logprobs, prompt_logprobs) = req.check().map_err(ApiError::bad_request)?;
    let prompts =
        prompts(req.prompt.as_ref(), PromptForms::OpenAi).map_err(ApiError::bad_request)?;
    let constraint = match &req.response_format {
        Some(format) => {
            crate::types::parse_response_format(format).map_err(ApiError::bad_request)?
        }
        None => None,
    };
    let (model, engine) = select_model(&state, &req.model).await?;
    let mut options = options(
        &engine,
        &body,
        SamplingProfile::OpenAi,
        constraint,
        &headers,
    )?;
    options.logprobs = logprobs;
    let echo = req.echo == Some(true);
    // The prompts' own log-probabilities, scored as each is completed:
    // `echo` repeats them with `logprobs` alternatives, vLLM's
    // `prompt_logprobs` with its own count.
    let echo_logprobs = logprobs.filter(|_| echo);
    options.prompt_logprobs = echo_logprobs.max(prompt_logprobs);
    let build = Build {
        suffix: req.suffix.clone(),
        special: req.add_special_tokens.unwrap_or(true),
        echo,
    };
    let jobs = build_jobs(&engine, build, &options, prompts).await?;
    let id = format!("cmpl-{}", Uuid::new_v4().simple());
    let created = unix_time();
    // Where each choice's generated text starts.
    let shifts: Vec<usize> = jobs
        .iter()
        .map(|j| if echo { j.echo.chars().count() } else { 0 })
        .collect();

    if req.stream == Some(true) {
        // Each chunk's tokens, offset into the whole choice's text; the
        // echoed prompt's go with it.
        let render = {
            let engine = Arc::clone(&engine);
            move |index: usize, prompt: &[TokenLogprob], entries: &[TokenLogprob]| -> Value {
                if logprobs.is_none() {
                    return Value::Null;
                }
                crate::logprobs::completion(&engine, prompt, entries, shifts[index])
            }
        };
        let chunk = move |index: usize,
                          text: &str,
                          prompt: &[TokenLogprob],
                          entries: &[TokenLogprob],
                          finish: Option<&str>| {
            json!({
                "id": id, "object": "text_completion", "created": created, "model": model,
                "choices": [{
                    "index": index, "text": text,
                    "logprobs": render(index, prompt, entries),
                    "finish_reason": finish,
                }],
            })
        };
        let echoes: Vec<String> = jobs.iter().map(|j| j.echo.clone()).collect();
        let count = jobs.len();
        let mut usages = Vec::with_capacity(count);
        let mut started = vec![false; count];
        // Each echoed prompt's scored tokens, until its echo is sent.
        let mut scored: Vec<Vec<TokenLogprob>> = vec![Vec::new(); count];
        let placer = Arc::clone(&engine);
        let echo_texts = echoes.clone();
        // The echoed prompt, and its tokens, go ahead of a prompt's first
        // text (or its end, when it generated none).
        let mut echo_once = move |index: usize,
                                  scored: &mut Vec<Vec<TokenLogprob>>|
              -> Option<(String, Vec<TokenLogprob>)> {
            (echo && !std::mem::replace(&mut started[index], true))
                .then(|| (echoes[index].clone(), std::mem::take(&mut scored[index])))
                .filter(|(e, _)| !e.is_empty())
        };
        let frame = move |progress: Progress<'_>| -> Option<Value> {
            match progress {
                Progress::Prompt(index, entries) => {
                    if let Some(top) = echo_logprobs {
                        scored[index] = echo_entries(&placer, entries, top, &echo_texts[index]);
                    }
                    None
                }
                Progress::Text(index, text, entries) => {
                    let (text, prompt) = match echo_once(index, &mut scored) {
                        Some((echoed, prompt)) => (echoed + text, prompt),
                        None => (text.to_string(), Vec::new()),
                    };
                    Some(chunk(index, &text, &prompt, entries, None))
                }
                Progress::Done(index, completion, entries) => {
                    usages.push(completion.usage.clone());
                    let (text, prompt) = echo_once(index, &mut scored).unwrap_or_default();
                    let mut last = chunk(
                        index,
                        &text,
                        &prompt,
                        entries,
                        Some(completion.finish_reason()),
                    );
                    if index + 1 == count {
                        last["usage"] =
                            serde_json::to_value(total_usage(&usages)).unwrap_or_default();
                        last["timings"] =
                            serde_json::to_value(completion.timings).unwrap_or_default();
                    }
                    Some(last)
                }
            }
        };
        let error = json!({"error": {"message": "generation failed", "type": "server_error"}});
        return Ok(stream_all(
            engine,
            jobs,
            options,
            Arc::clone(&state.metrics),
            frame,
            error,
            Some("[DONE]"),
        ));
    }

    let completions = complete_all(&engine, &jobs, options, &state.metrics).await?;
    let usage = total_usage(completions.iter().map(|c| &c.usage));
    let choices: Vec<Value> = completions
        .iter()
        .zip(&jobs)
        .enumerate()
        .map(|(index, (completion, job))| {
            let text = if echo {
                format!("{}{}", job.echo, completion.text)
            } else {
                completion.text.clone()
            };
            let echoed = match echo_logprobs {
                Some(top) => echo_entries(&engine, &completion.prompt_logprobs, top, &job.echo),
                None => Vec::new(),
            };
            let logprobs = match logprobs {
                Some(_) => crate::logprobs::completion(
                    &engine,
                    &echoed,
                    &completion.logprobs,
                    shifts[index],
                ),
                None => Value::Null,
            };
            let mut choice = json!({
                "text": text, "index": index, "logprobs": logprobs,
                "finish_reason": completion.finish_reason(),
            });
            if let Some(top) = prompt_logprobs {
                choice["prompt_logprobs"] = crate::logprobs::prompt(
                    &engine,
                    &alternatives(&completion.prompt_logprobs, top),
                );
            }
            choice
        })
        .collect();
    Ok(Json(json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage,
        "timings": completions.last().map(|c| &c.timings),
    }))
    .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CompletionTokensDetails, PromptTokensDetails};

    fn usage(prompt: u32, completion: u32, cached: Option<u32>) -> UsageInfo {
        UsageInfo {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            prompt_tokens_details: cached
                .map(|cached_tokens| PromptTokensDetails { cached_tokens }),
            completion_tokens_details: None,
        }
    }

    #[test]
    fn usage_keeps_and_sums_the_token_details() {
        let one = usage(100, 5, Some(80));
        assert_eq!(
            serde_json::to_value(total_usage([&one])).unwrap(),
            serde_json::to_value(&one).unwrap()
        );
        let mut reasoning = usage(10, 4, None);
        reasoning.completion_tokens_details = Some(CompletionTokensDetails {
            reasoning_tokens: 3,
        });
        let total = total_usage([&one, &reasoning, &usage(7, 1, Some(2))]);
        assert_eq!(
            (
                total.prompt_tokens,
                total.completion_tokens,
                total.total_tokens
            ),
            (117, 10, 127)
        );
        assert_eq!(total.prompt_tokens_details.unwrap().cached_tokens, 82);
        assert_eq!(total.completion_tokens_details.unwrap().reasoning_tokens, 3);
    }

    #[test]
    fn prompts_take_every_openai_form() {
        let openai = |v: Value| prompts(Some(&v), PromptForms::OpenAi);
        assert_eq!(openai(json!("a")).unwrap(), [Prompt::Text("a".into())]);
        assert_eq!(openai(json!([1, 2])).unwrap(), [Prompt::Tokens(vec![1, 2])]);
        assert_eq!(
            openai(json!(["a", [3]])).unwrap(),
            [Prompt::Text("a".into()), Prompt::Tokens(vec![3])]
        );
        for bad in [
            json!([]),
            json!([[]]),
            json!([-1]),
            json!({}),
            json!([1, "a"]),
            json!([["a", 1]]),
        ] {
            assert!(openai(bad.clone()).is_err(), "{bad}");
        }
        assert!(prompts(None, PromptForms::OpenAi).is_err());
    }

    #[test]
    fn llama_prompts_also_mix_text_and_token_ids() {
        let llama = |v: Value| prompts(Some(&v), PromptForms::LlamaCpp);
        let mixed = Prompt::Mixed(vec![
            PromptPart::Tokens(vec![1, 2]),
            PromptPart::Text("a".into()),
            PromptPart::Tokens(vec![3]),
        ]);
        assert_eq!(llama(json!([1, 2, "a", 3])).unwrap(), std::slice::from_ref(&mixed));
        assert_eq!(
            llama(json!(["b", [1, 2, "a", 3], [4]])).unwrap(),
            [Prompt::Text("b".into()), mixed, Prompt::Tokens(vec![4])]
        );
        assert_eq!(
            llama(json!(["a", "b"])).unwrap(),
            [Prompt::Text("a".into()), Prompt::Text("b".into())]
        );
        for bad in [json!([1, "a", -1]), json!([["a", {}]]), json!([[]])] {
            assert!(llama(bad.clone()).is_err(), "{bad}");
        }
    }
}
