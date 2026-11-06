//! `POST /v1/completions`: OpenAI's and vLLM's raw completion endpoint.
//!
//! The prompt is completed as given, with no chat template: a string, an
//! array of token ids, or an array of either for one choice per prompt.
//! `stream` sends server-sent `text_completion` chunks ending in
//! `data: [DONE]`; `echo` repeats the prompt in front of the text; `suffix`
//! builds a fill-in-the-middle prompt for a model with FIM tokens; and
//! `response_format` constrains the output as on the chat route.  Sampling
//! fields are the shared [`SamplingParams`].  `logprobs`, `n > 1` and
//! `best_of > 1` are refused rather than ignored.

use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    extract::State,
    http::HeaderMap,
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::engine::{Completion, Engine, RawPrompt};
use crate::sampling::SamplingProfile;
use crate::server::{select_model, ApiError, AppState, CancelOnDrop};
use crate::types::{unix_time, GenerationOptions, SamplingParams, UsageInfo};

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

/// One prompt of a request: what is completed, and the text `echo` repeats.
#[derive(Debug, Clone, PartialEq)]
enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
}

/// The request's `prompt` as one or more prompts.
fn prompts(prompt: Option<&Value>) -> Result<Vec<Prompt>, String> {
    let ids = |items: &[Value]| -> Option<Vec<u32>> {
        items
            .iter()
            .map(|v| v.as_u64().and_then(|id| u32::try_from(id).ok()))
            .collect()
    };
    let invalid = || {
        "`prompt` must be a string, an array of token ids, or an array of either".to_string()
    };
    let out = match prompt {
        None | Some(Value::Null) => return Err("Missing required field 'prompt'".to_string()),
        Some(Value::String(s)) => vec![Prompt::Text(s.clone())],
        Some(Value::Array(items)) if items.is_empty() => return Err(invalid()),
        Some(Value::Array(items)) if items.iter().all(Value::is_number) => {
            vec![Prompt::Tokens(ids(items).ok_or_else(invalid)?)]
        }
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::String(s) => Ok(Prompt::Text(s.clone())),
                Value::Array(t) if !t.is_empty() => ids(t).map(Prompt::Tokens).ok_or_else(invalid),
                _ => Err(invalid()),
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(invalid()),
    };
    Ok(out)
}

impl CompletionRequest {
    /// Refuse what this endpoint does not do, rather than ignore it.
    fn check(&self) -> Result<(), String> {
        if self.n.is_some_and(|n| n != 1) {
            return Err("only n = 1 is supported".to_string());
        }
        if self.best_of.is_some_and(|n| n > 1) {
            return Err("best_of > 1 is not supported".to_string());
        }
        let wants_logprobs = match &self.logprobs {
            None | Some(Value::Null) | Some(Value::Bool(false)) => false,
            Some(Value::Number(n)) => n.as_u64() != Some(0) || n.as_i64().is_none(),
            Some(_) => true,
        };
        if wants_logprobs {
            return Err("logprobs are not supported".to_string());
        }
        Ok(())
    }
}

/// A prompt as the engine completes it, plus the text `echo` repeats.
struct Job {
    prompt: RawPrompt,
    echo: String,
}

/// How a request's prompts are built.
struct Build {
    suffix: Option<String>,
    /// The tokenizer's special tokens (BOS) on a text or infill prompt.
    special: bool,
    /// Keep the prompt's text for `echo`.
    echo: bool,
}

/// Build each prompt's [`Job`]: a text prompt with the tokenizer's special
/// tokens unless `add_special_tokens` is false, token ids verbatim, and with
/// `suffix` an infill prompt in the model's FIM format.  Each is checked as
/// generation will check it (token ids, empty or too long for the window),
/// so a bad prompt is a client error before a stream starts.
fn jobs(
    engine: &Engine,
    build: &Build,
    options: &GenerationOptions,
    prompts: Vec<Prompt>,
) -> Result<Vec<Job>, ApiError> {
    let suffix = build.suffix.as_deref().filter(|s| !s.is_empty());
    let special = build.special;
    prompts
        .into_iter()
        .map(|prompt| {
            let echo = match &prompt {
                _ if !build.echo => String::new(),
                Prompt::Text(text) => text.clone(),
                Prompt::Tokens(ids) => engine.detokenize(ids, false).map_err(ApiError::from)?,
            };
            let prompt = match (prompt, suffix) {
                (Prompt::Text(text), Some(suffix)) => {
                    if let Some(missing) = engine.fim_tokens().missing() {
                        return Err(ApiError::bad_request(format!(
                            "`suffix` needs a fill-in-the-middle model: {missing}"
                        )));
                    }
                    let ids = engine
                        .format_infill(&text, suffix, &[], "", -1, false, special)
                        .map_err(ApiError::from)?;
                    RawPrompt::tokens(ids)
                }
                (Prompt::Tokens(_), Some(_)) => {
                    return Err(ApiError::bad_request("`suffix` needs a text prompt"))
                }
                (Prompt::Text(text), None) => RawPrompt::text(text).with_special_tokens(special),
                (Prompt::Tokens(ids), None) => RawPrompt::tokens(ids),
            };
            engine.check_prompt(&prompt, options).map_err(ApiError::from)?;
            Ok(Job { prompt, echo })
        })
        .collect()
}

/// The request's generation options: the OpenAI profile's defaults, then
/// its sampling fields (`max_tokens` absent or -1 runs to the end of the
/// sequence) and `response_format`.
fn options(
    engine: &Engine,
    body: &Value,
    req: &CompletionRequest,
    headers: &HeaderMap,
) -> Result<GenerationOptions, ApiError> {
    let sampling: SamplingParams =
        serde_json::from_value(body.clone()).map_err(|e| ApiError::bad_request(e.to_string()))?;
    let mut options =
        GenerationOptions::with_defaults(&engine.sampling_defaults(SamplingProfile::OpenAi));
    sampling.apply(&mut options).map_err(ApiError::bad_request)?;
    if let Some(format) = &req.response_format {
        options.constraint =
            crate::types::parse_response_format(format).map_err(ApiError::bad_request)?;
    }
    options.cache_scope = Some(crate::affinity::cache_scope(headers, None));
    // Answer a request the model cannot serve before a stream starts.
    engine.validate_options(&options).map_err(ApiError::from)?;
    Ok(options)
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
    req.check().map_err(ApiError::bad_request)?;
    let prompts = prompts(req.prompt.as_ref()).map_err(ApiError::bad_request)?;
    let (model, engine) = select_model(&state, &req.model).await?;
    let options = options(&engine, &body, &req, &headers)?;
    let echo = req.echo == Some(true);
    let (jobs, options) = {
        let engine = Arc::clone(&engine);
        let build = Build {
            suffix: req.suffix.clone(),
            special: req.add_special_tokens.unwrap_or(true),
            echo,
        };
        tokio::task::spawn_blocking(move || {
            jobs(&engine, &build, &options, prompts).map(|jobs| (jobs, options))
        })
        .await
        .map_err(|e| ApiError::internal(e.to_string()))??
    };
    let id = format!("cmpl-{}", Uuid::new_v4().simple());
    let created = unix_time();
    let cancel = CancelOnDrop::new();

    if req.stream == Some(true) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let flag = cancel.flag();
        state.metrics.record_stream();
        let metrics = Arc::clone(&state.metrics);
        tokio::task::spawn_blocking(move || {
            let chunk = |index: usize, text: &str, finish: Option<&str>| {
                json!({
                    "id": id, "object": "text_completion", "created": created, "model": model,
                    "choices": [{
                        "index": index, "text": text, "logprobs": null, "finish_reason": finish,
                    }],
                })
            };
            let send = |value: Value| {
                let _ = tx.send(value.to_string());
            };
            let mut usages = Vec::with_capacity(jobs.len());
            let count = jobs.len();
            for (index, job) in jobs.into_iter().enumerate() {
                if echo && !job.echo.is_empty() {
                    send(chunk(index, &job.echo, None));
                }
                let result = engine.complete_prompt_streaming(&job.prompt, &options, &flag, &mut |text| {
                    send(chunk(index, text, None))
                });
                match result {
                    Ok(completion) => {
                        metrics.record_generation(&completion.usage);
                        usages.push(completion.usage.clone());
                        let mut last = chunk(index, "", Some(completion.finish_reason()));
                        if index + 1 == count {
                            last["usage"] =
                                serde_json::to_value(total_usage(&usages)).unwrap_or_default();
                            last["timings"] =
                                serde_json::to_value(&completion.timings).unwrap_or_default();
                        }
                        send(last);
                    }
                    Err(error) => {
                        tracing::error!("streaming completion failed: {error}");
                        send(json!({
                            "error": {"message": "generation failed", "type": "server_error"},
                        }));
                        break;
                    }
                }
            }
            let _ = tx.send("[DONE]".to_string());
        });
        let events = futures_util::stream::unfold((rx, cancel), |(mut rx, cancel)| async move {
            rx.recv()
                .await
                .map(|data| (Ok::<Event, Infallible>(Event::default().data(data)), (rx, cancel)))
        });
        return Ok(Sse::new(events).into_response());
    }

    let flag = cancel.flag();
    let completions: Vec<Completion> = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let prompts: Vec<RawPrompt> = jobs.iter().map(|j| j.prompt.clone()).collect();
        move || {
            prompts
                .iter()
                .map(|p| engine.complete_prompt_cancellable(p, &options, &flag))
                .collect::<crate::Result<Vec<_>>>()
        }
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;

    let usage = total_usage(completions.iter().map(|c| &c.usage));
    let choices: Vec<Value> = completions
        .iter()
        .zip(&jobs)
        .enumerate()
        .map(|(index, (completion, job))| {
            state.metrics.record_generation(&completion.usage);
            let text = if echo {
                format!("{}{}", job.echo, completion.text)
            } else {
                completion.text.clone()
            };
            json!({
                "text": text, "index": index, "logprobs": null,
                "finish_reason": completion.finish_reason(),
            })
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
            prompt_tokens_details: cached.map(|cached_tokens| PromptTokensDetails { cached_tokens }),
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
        reasoning.completion_tokens_details = Some(CompletionTokensDetails { reasoning_tokens: 3 });
        let total = total_usage([&one, &reasoning, &usage(7, 1, Some(2))]);
        assert_eq!((total.prompt_tokens, total.completion_tokens, total.total_tokens), (117, 10, 127));
        assert_eq!(total.prompt_tokens_details.unwrap().cached_tokens, 82);
        assert_eq!(total.completion_tokens_details.unwrap().reasoning_tokens, 3);
    }

    #[test]
    fn prompts_take_every_openai_form() {
        assert_eq!(prompts(Some(&json!("a"))).unwrap(), [Prompt::Text("a".into())]);
        assert_eq!(prompts(Some(&json!([1, 2]))).unwrap(), [Prompt::Tokens(vec![1, 2])]);
        assert_eq!(
            prompts(Some(&json!(["a", [3]]))).unwrap(),
            [Prompt::Text("a".into()), Prompt::Tokens(vec![3])]
        );
        for bad in [json!([]), json!([[]]), json!([-1]), json!({}), json!([1, "a"])] {
            assert!(prompts(Some(&bad)).is_err(), "{bad}");
        }
        assert!(prompts(None).is_err());
    }
}
