//! OpenAI's Responses API: `POST /v1/responses`, `GET`/`DELETE
//! /v1/responses/{id}`, `GET /v1/responses/{id}/input_items` and `POST
//! /v1/responses/{id}/cancel`.
//!
//! A request is translated into a chat completion: `instructions` become the
//! system message, the `input` items (and, with `previous_response_id`, the
//! earlier turns' items and outputs) become the conversation, and the
//! function `tools`, `tool_choice`, `text.format`, `reasoning.effort` and
//! sampling fields map onto their chat counterparts.  The answer goes through
//! the same reasoning splitter and tool-call reader as the chat routes and
//! comes back as `reasoning`, `message` and `function_call` output items;
//! streamed, as the Responses API's typed events.
//!
//! Responses are kept, unless a request says `store: false`, in a bounded
//! in-memory store ([`ResponseStore`]), so a later request can continue from
//! one with `previous_response_id` and a client can fetch or delete it.  The
//! store forgets the oldest responses past [`STORE_ENTRIES`] responses or
//! [`STORE_BYTES`] bytes, and does not outlive the server.
//!
//! Refused rather than ignored: `background`, `conversation`, hosted tools
//! (anything but `function`), `item_reference` and file inputs,
//! `truncation: "auto"`, `top_logprobs` and `max_tool_calls`.

use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures_util::stream;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use uuid::Uuid;

use crate::engine::{Completion, StopType};
use crate::reasoning::ReasoningOutput;
use crate::sampling::SamplingProfile;
use crate::server::{
    chat_answer, select_model, stream_chat_output, ApiError, AppState, CancelOnDrop, ChatPiece,
    ServerMetrics,
};
use crate::tools::ParsedToolCall;
use crate::types::{unix_time, ChatCompletionRequest, UsageInfo};

/// Most responses the store keeps.
pub const STORE_ENTRIES: usize = 1024;
/// Most bytes (of the responses and their conversations, as JSON) the store
/// keeps.
pub const STORE_BYTES: usize = 64 << 20;

/// The prefix of the call ids minted for function calls the model gives
/// none (OpenAI's).
const CALL_IDS: &str = "call_";

// ─── Store ─────────────────────────────────────────────────────────────────

/// Stored responses, oldest first out.
#[derive(Default)]
pub struct ResponseStore {
    inner: Mutex<StoreInner>,
}

#[derive(Default)]
struct StoreInner {
    entries: HashMap<String, Stored>,
    /// Ids, oldest first.
    order: VecDeque<String>,
    bytes: usize,
}

/// One stored response.
#[derive(Clone)]
struct Stored {
    /// The response object, as returned.
    response: Value,
    /// Its conversation before its output: the earlier turns' input items
    /// and outputs, then its own input items, each with an `id`.
    history: Vec<Value>,
    bytes: usize,
}

impl ResponseStore {
    fn insert(&self, id: String, response: Value, history: Vec<Value>) {
        let bytes = response.to_string().len()
            + history
                .iter()
                .map(|item| item.to_string().len())
                .sum::<usize>();
        if bytes > STORE_BYTES {
            // Past the whole budget on its own: kept, it would hold the
            // store over its bound.  It cannot be fetched or continued.
            tracing::warn!(id, bytes, "response too large for the store; not kept");
            return;
        }
        let mut inner = self.lock();
        if let Some(old) = inner.entries.remove(&id) {
            inner.bytes -= old.bytes;
            inner.order.retain(|kept| kept != &id);
        }
        inner.bytes += bytes;
        inner.order.push_back(id.clone());
        inner.entries.insert(
            id,
            Stored {
                response,
                history,
                bytes,
            },
        );
        while inner.order.len() > STORE_ENTRIES
            || (inner.bytes > STORE_BYTES && inner.order.len() > 1)
        {
            let Some(oldest) = inner.order.pop_front() else {
                break;
            };
            if let Some(gone) = inner.entries.remove(&oldest) {
                inner.bytes -= gone.bytes;
            }
        }
    }

    fn get(&self, id: &str) -> Option<Stored> {
        self.lock().entries.get(id).cloned()
    }

    fn remove(&self, id: &str) -> bool {
        let mut inner = self.lock();
        match inner.entries.remove(id) {
            Some(gone) => {
                inner.bytes -= gone.bytes;
                inner.order.retain(|kept| kept != id);
                true
            }
            None => false,
        }
    }

    /// How many responses are stored.
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// Whether no response is stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, StoreInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

fn not_found(id: &str) -> ApiError {
    ApiError::new(
        StatusCode::NOT_FOUND,
        format!("Response with id '{id}' not found."),
        "invalid_request_error",
    )
}

// ─── Request ───────────────────────────────────────────────────────────────

/// A `POST /v1/responses` body.  Fields the endpoint does not read
/// (`include`, `service_tier`, `stream_options`, ...) are ignored.
#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    input: Option<Value>,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
    #[serde(default)]
    reasoning: Option<Value>,
    #[serde(default)]
    text: Option<Value>,
    #[serde(default)]
    metadata: Option<Value>,
    #[serde(default)]
    truncation: Option<String>,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    prompt_cache_key: Option<String>,
    #[serde(default)]
    background: Option<bool>,
    #[serde(default)]
    conversation: Option<Value>,
    #[serde(default)]
    top_logprobs: Option<u32>,
    #[serde(default)]
    max_tool_calls: Option<u32>,
}

impl ResponsesRequest {
    /// Refuse what this endpoint does not do.
    fn check(&self) -> Result<(), String> {
        if self.background == Some(true) {
            return Err("`background` responses are not supported".to_string());
        }
        if self.conversation.is_some() {
            return Err(
                "`conversation` is not supported; continue with `previous_response_id`".to_string(),
            );
        }
        if self.truncation.as_deref().is_some_and(|t| t != "disabled") {
            return Err("only `truncation: \"disabled\"` is supported".to_string());
        }
        if self.top_logprobs.is_some_and(|n| n > 0) {
            return Err("`top_logprobs` is not supported".to_string());
        }
        if self.max_tool_calls.is_some() {
            return Err("`max_tool_calls` is not supported".to_string());
        }
        if self.max_output_tokens == Some(0) {
            return Err("`max_output_tokens` must be at least 1".to_string());
        }
        Ok(())
    }

    /// The input as items: a string is one user message.
    fn input_items(&self) -> Result<Vec<Value>, String> {
        match &self.input {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(Value::String(text)) => Ok(vec![json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            })]),
            Some(Value::Array(items)) => items
                .iter()
                .map(|item| {
                    if item.is_object() {
                        Ok(item.clone())
                    } else {
                        Err("each `input` item must be an object".to_string())
                    }
                })
                .collect(),
            Some(_) => Err("`input` must be a string or an array of items".to_string()),
        }
    }

    /// The chat completion this request runs: `messages` are its
    /// conversation.
    fn chat_request(&self, messages: Vec<Value>) -> Result<ChatCompletionRequest, String> {
        let mut body = Map::new();
        body.insert("model".into(), json!(self.model));
        body.insert("messages".into(), Value::Array(messages));
        if let Some(n) = self.max_output_tokens {
            body.insert("max_tokens".into(), json!(n));
        }
        if let Some(t) = self.temperature {
            body.insert("temperature".into(), json!(t));
        }
        if let Some(p) = self.top_p {
            body.insert("top_p".into(), json!(p));
        }
        if let Some(tools) = self.tools.as_ref().filter(|tools| !tools.is_empty()) {
            body.insert("tools".into(), Value::Array(chat_tools(tools)?));
        }
        if let Some(choice) = &self.tool_choice {
            body.insert("tool_choice".into(), chat_tool_choice(choice)?);
        }
        if let Some(format) = self.text.as_ref().and_then(|t| t.get("format")) {
            if let Some(format) = response_format(format)? {
                body.insert("response_format".into(), format);
            }
        }
        if let Some(effort) = self
            .reasoning
            .as_ref()
            .and_then(|r| r.get("effort"))
            .filter(|e| !e.is_null())
        {
            body.insert("reasoning_effort".into(), effort.clone());
        }
        serde_json::from_value(Value::Object(body)).map_err(|e| e.to_string())
    }
}

/// Function tools in the chat shape.
fn chat_tools(tools: &[Value]) -> Result<Vec<Value>, String> {
    tools
        .iter()
        .map(|tool| {
            match tool.get("type").and_then(Value::as_str) {
                Some("function") => {}
                Some(other) => {
                    return Err(format!(
                        "tool type '{other}' is not supported; only `function` tools are"
                    ))
                }
                None => return Err("each tool needs a `type`".to_string()),
            }
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .ok_or("a function tool needs a `name`")?;
            let mut function = json!({"name": name});
            if let Some(description) = tool.get("description").filter(|d| !d.is_null()) {
                function["description"] = description.clone();
            }
            if let Some(parameters) = tool.get("parameters").filter(|p| !p.is_null()) {
                function["parameters"] = parameters.clone();
            }
            Ok(json!({"type": "function", "function": function}))
        })
        .collect()
}

/// `tool_choice` in the chat shape.
fn chat_tool_choice(choice: &Value) -> Result<Value, String> {
    match choice {
        Value::String(mode) if matches!(mode.as_str(), "auto" | "none" | "required") => {
            Ok(choice.clone())
        }
        Value::Object(object) if object.get("type").and_then(Value::as_str) == Some("function") => {
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .ok_or("a function `tool_choice` needs a `name`")?;
            Ok(json!({"type": "function", "function": {"name": name}}))
        }
        _ => Err(format!("unsupported tool_choice: {choice}")),
    }
}

/// `text.format` as a chat `response_format` (`None` for plain text).
fn response_format(format: &Value) -> Result<Option<Value>, String> {
    match format.get("type").and_then(Value::as_str) {
        None | Some("text") => Ok(None),
        Some("json_object") => Ok(Some(json!({"type": "json_object"}))),
        Some("json_schema") => {
            let body = format
                .get("schema")
                .filter(|s| s.is_object())
                .ok_or("a `json_schema` text.format needs a `schema` object")?;
            let mut schema = json!({
                "name": format.get("name").cloned().unwrap_or(json!("response")),
                "schema": body,
            });
            if let Some(strict) = format.get("strict").filter(|s| !s.is_null()) {
                schema["strict"] = strict.clone();
            }
            Ok(Some(json!({"type": "json_schema", "json_schema": schema})))
        }
        Some(other) => Err(format!("unsupported text.format type '{other}'")),
    }
}

// ─── Items and messages ─────────────────────────────────────────────────────

/// `item` with an `id`, minting one when it has none.
fn with_id(mut item: Value) -> Value {
    if item.get("id").and_then(Value::as_str).is_none() {
        let prefix = match item.get("type").and_then(Value::as_str) {
            Some("function_call") => "fc",
            Some("function_call_output") => "fco",
            Some("reasoning") => "rs",
            _ => "msg",
        };
        item["id"] = json!(new_id(prefix));
    }
    item
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::new_v4().simple())
}

/// The text of a content array's text parts (or of a string), joined.
fn parts_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// A message item's content in the chat shape: text parts and images.
fn message_content(content: &Value) -> Result<Value, String> {
    let parts = match content {
        Value::String(_) | Value::Null => return Ok(content.clone()),
        Value::Array(parts) => parts,
        _ => return Err("message `content` must be a string or an array of parts".to_string()),
    };
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("input_text" | "output_text" | "text" | "summary_text") => {
                let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
                out.push(json!({"type": "text", "text": text}));
            }
            Some("refusal") => {
                let text = part
                    .get("refusal")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                out.push(json!({"type": "text", "text": text}));
            }
            Some("input_image") => {
                let url = match part.get("image_url") {
                    Some(Value::String(url)) => url.clone(),
                    Some(Value::Object(image)) => image
                        .get("url")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    _ => {
                        return Err(
                            "`input_image` needs an `image_url`; `file_id` is not supported"
                                .to_string(),
                        )
                    }
                };
                out.push(json!({"type": "image_url", "image_url": {"url": url}}));
            }
            Some(other) => return Err(format!("content part type '{other}' is not supported")),
            None => return Err("each content part needs a `type`".to_string()),
        }
    }
    Ok(Value::Array(out))
}

/// The chat conversation of `items`, input and output items alike:
/// messages (`developer` reads as `system`), function calls (one assistant
/// turn holds the calls written together) and their outputs, with
/// reasoning passed back on the assistant turn after it.
fn chat_messages(items: &[Value]) -> Result<Vec<Value>, String> {
    let mut messages: Vec<Value> = Vec::new();
    let mut reasoning: Option<String> = None;
    // Whether the last message is an assistant turn calls can join.
    let mut open_turn = false;
    for item in items {
        let kind = item.get("type").and_then(Value::as_str);
        let kind = match kind {
            None if item.get("role").is_some() => "message",
            None => return Err("each `input` item needs a `type` or a `role`".to_string()),
            Some(kind) => kind,
        };
        match kind {
            "message" => {
                let role = item
                    .get("role")
                    .and_then(Value::as_str)
                    .ok_or("a message item needs a `role`")?;
                let role = match role {
                    "developer" | "system" => "system",
                    "user" | "assistant" => role,
                    other => return Err(format!("unsupported message role '{other}'")),
                };
                let content = item.get("content").cloned().unwrap_or(Value::Null);
                let mut message = if role == "assistant" {
                    json!({"role": role, "content": parts_text(&content)})
                } else {
                    json!({"role": role, "content": message_content(&content)?})
                };
                if role == "assistant" {
                    if let Some(text) = reasoning.take() {
                        message["reasoning_content"] = json!(text);
                    }
                }
                open_turn = role == "assistant";
                messages.push(message);
            }
            "function_call" => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .ok_or("a `function_call` item needs a `call_id`")?;
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("a `function_call` item needs a `name`")?;
                let arguments = match item.get("arguments") {
                    Some(Value::String(arguments)) => arguments.clone(),
                    Some(other) if !other.is_null() => other.to_string(),
                    _ => "{}".to_string(),
                };
                let call = json!({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                });
                if !open_turn {
                    let mut turn = json!({"role": "assistant", "content": ""});
                    if let Some(text) = reasoning.take() {
                        turn["reasoning_content"] = json!(text);
                    }
                    messages.push(turn);
                    open_turn = true;
                }
                let turn = messages.last_mut().expect("an assistant turn is open");
                match turn.get_mut("tool_calls").and_then(Value::as_array_mut) {
                    Some(calls) => calls.push(call),
                    None => turn["tool_calls"] = json!([call]),
                }
            }
            "function_call_output" => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .ok_or("a `function_call_output` item needs a `call_id`")?;
                let output = item.get("output").map(parts_text).unwrap_or_default();
                messages.push(json!({"role": "tool", "tool_call_id": call_id, "content": output}));
                open_turn = false;
            }
            "reasoning" => {
                let text = match item.get("content").filter(|c| !c.is_null()) {
                    Some(content) => parts_text(content),
                    None => item.get("summary").map(parts_text).unwrap_or_default(),
                };
                if !text.is_empty() {
                    reasoning = Some(text);
                }
                open_turn = false;
            }
            other => return Err(format!("input item type '{other}' is not supported")),
        }
    }
    Ok(messages)
}

// ─── Response objects ───────────────────────────────────────────────────────

/// What a response object repeats of its request.
#[derive(Clone)]
struct Meta {
    id: String,
    created_at: u64,
    model: String,
    instructions: Option<String>,
    max_output_tokens: Option<u32>,
    previous_response_id: Option<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    tools: Vec<Value>,
    tool_choice: Value,
    parallel_tool_calls: bool,
    reasoning_effort: Value,
    store: bool,
    text: Value,
    user: Option<String>,
    metadata: Value,
}

impl Meta {
    fn of(req: &ResponsesRequest, model: String) -> Self {
        Self {
            id: new_id("resp"),
            created_at: unix_time(),
            model,
            instructions: req.instructions.clone(),
            max_output_tokens: req.max_output_tokens,
            previous_response_id: req.previous_response_id.clone(),
            temperature: req.temperature,
            top_p: req.top_p,
            tools: req.tools.clone().unwrap_or_default(),
            tool_choice: req.tool_choice.clone().unwrap_or(json!("auto")),
            parallel_tool_calls: req.parallel_tool_calls.unwrap_or(true),
            reasoning_effort: req
                .reasoning
                .as_ref()
                .and_then(|r| r.get("effort"))
                .cloned()
                .unwrap_or(Value::Null),
            store: req.store.unwrap_or(true),
            text: req
                .text
                .clone()
                .unwrap_or_else(|| json!({"format": {"type": "text"}})),
            user: req.user.clone(),
            metadata: req.metadata.clone().unwrap_or_else(|| json!({})),
        }
    }

    /// How many function calls a turn may make.
    fn max_calls(&self) -> usize {
        if self.parallel_tool_calls {
            usize::MAX
        } else {
            1
        }
    }

    /// The response object with `status` and `output`.
    fn response(&self, status: &str, output: Vec<Value>) -> Value {
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "background": false,
            "error": null,
            "incomplete_details": null,
            "instructions": self.instructions,
            "max_output_tokens": self.max_output_tokens,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": self.parallel_tool_calls,
            "previous_response_id": self.previous_response_id,
            "reasoning": {"effort": self.reasoning_effort, "summary": null},
            "store": self.store,
            "temperature": self.temperature,
            "text": self.text,
            "tool_choice": self.tool_choice,
            "tools": self.tools,
            "top_p": self.top_p,
            "truncation": "disabled",
            "usage": null,
            "user": self.user,
            "metadata": self.metadata,
        })
    }

    /// The finished response: `completed`, or `incomplete` when a token cap
    /// cut it off, with its usage.
    fn finished(&self, completion: &Completion, output: Vec<Value>) -> Value {
        let capped = matches!(completion.stop_type, StopType::Limit | StopType::Cancelled);
        let mut response = self.response(if capped { "incomplete" } else { "completed" }, output);
        if capped {
            response["incomplete_details"] = json!({"reason": "max_output_tokens"});
        }
        response["usage"] = usage(&completion.usage);
        response
    }
}

fn usage(usage: &UsageInfo) -> Value {
    json!({
        "input_tokens": usage.prompt_tokens,
        "input_tokens_details": {
            "cached_tokens": usage.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens),
        },
        "output_tokens": usage.completion_tokens,
        "output_tokens_details": {
            "reasoning_tokens": usage
                .completion_tokens_details
                .as_ref()
                .map_or(0, |d| d.reasoning_tokens),
        },
        "total_tokens": usage.total_tokens,
    })
}

fn reasoning_item(id: &str, text: &str) -> Value {
    json!({
        "type": "reasoning",
        "id": id,
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    })
}

fn text_part(text: &str) -> Value {
    json!({"type": "output_text", "text": text, "annotations": [], "logprobs": []})
}

fn message_item(id: &str, status: &str, content: Vec<Value>) -> Value {
    json!({
        "type": "message",
        "id": id,
        "status": status,
        "role": "assistant",
        "content": content,
    })
}

fn call_item(id: &str, call_id: &str, name: &str, arguments: &str, status: &str) -> Value {
    json!({
        "type": "function_call",
        "id": id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    })
}

// ─── Handlers ───────────────────────────────────────────────────────────────

/// `POST /v1/responses`.
pub(crate) async fn create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ResponsesRequest>,
) -> Result<Response, ApiError> {
    req.check().map_err(ApiError::bad_request)?;
    let input = req.input_items().map_err(ApiError::bad_request)?;
    // The conversation so far: the earlier turns, then this input.
    let mut history = match &req.previous_response_id {
        Some(id) => {
            let previous = state.responses.get(id).ok_or_else(|| {
                ApiError::bad_request(format!("Previous response with id '{id}' not found."))
            })?;
            let mut history = previous.history;
            if let Some(output) = previous.response.get("output").and_then(Value::as_array) {
                history.extend(output.iter().cloned());
            }
            history
        }
        None => Vec::new(),
    };
    history.extend(input.into_iter().map(with_id));
    if history.is_empty() {
        return Err(ApiError::bad_request("`input` is required"));
    }
    let mut messages = Vec::new();
    if let Some(instructions) = req.instructions.as_ref().filter(|i| !i.is_empty()) {
        messages.push(json!({"role": "system", "content": instructions}));
    }
    messages.extend(chat_messages(&history).map_err(ApiError::bad_request)?);
    let chat = req.chat_request(messages).map_err(ApiError::bad_request)?;

    let (resolved_model, engine) = select_model(&state, &req.model).await?;
    let defaults = engine.sampling_defaults(SamplingProfile::OpenAi);
    let mut options = chat
        .to_generation_options_with(&defaults)
        .map_err(ApiError::bad_request)?;
    engine.validate_options(&options).map_err(ApiError::from)?;
    options.cache_scope = Some(crate::affinity::cache_scope(
        &headers,
        options.context_window,
        req.prompt_cache_key.as_deref(),
    ));
    let messages = chat.messages.clone();
    let tools = chat.offered_tools();
    let model = if req.model.is_empty() {
        resolved_model
    } else {
        req.model.clone()
    };
    let meta = Meta::of(&req, model);
    let show_reasoning = state.reasoning_format != ReasoningOutput::Hidden;

    if req.stream == Some(true) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<(&'static str, String)>();
        let cancel = CancelOnDrop::new();
        let flag = cancel.flag();
        state.metrics.record_stream();
        let split = state.reasoning_format.splits(true);
        let store = Arc::clone(&state.responses);
        let metrics = Arc::clone(&state.metrics);
        tokio::task::spawn_blocking(move || {
            let run = StreamRun {
                engine: &engine,
                messages: &messages,
                tools: tools.as_deref(),
                options: &options,
                cancel: &flag,
                split,
                show_reasoning,
                metrics: &metrics,
            };
            run.stream(Emitter::new(tx, meta), &store, history);
        });
        let events = stream::unfold((rx, cancel), |(mut rx, cancel)| async move {
            rx.recv().await.map(|(event, data)| {
                (
                    Ok::<Event, Infallible>(Event::default().event(event).data(data)),
                    (rx, cancel),
                )
            })
        });
        return Ok(Sse::new(events)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response());
    }

    let tool_call_format = if tools.is_some() {
        engine
            .tool_call_format_for_request(&messages, tools.as_deref(), &options)
            .map_err(ApiError::from)?
    } else {
        engine.tool_call_format()
    };
    let cancel = CancelOnDrop::new();
    let completion = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let tools = tools.clone();
        let flag = cancel.flag();
        move || {
            engine.complete_chat_cancellable_detailed(&messages, tools.as_deref(), &options, &flag)
        }
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;
    state.metrics.record_generation(&completion.usage);

    let answer = chat_answer(
        &engine,
        &completion,
        state.reasoning_format.splits(false),
        tools.as_deref(),
        tool_call_format,
        CALL_IDS,
    );
    let mut output = Vec::new();
    if let Some(reasoning) = answer.reasoning.as_deref().filter(|r| !r.is_empty()) {
        if show_reasoning {
            output.push(reasoning_item(&new_id("rs"), reasoning));
        }
    }
    if !answer.prose.is_empty() || answer.calls.is_empty() {
        output.push(message_item(
            &new_id("msg"),
            "completed",
            vec![text_part(&answer.prose)],
        ));
    }
    // `parallel_tool_calls: false`: the first call only.
    for call in answer.calls.iter().take(meta.max_calls()) {
        output.push(finished_call(call));
    }
    let response = meta.finished(&completion, output);
    if meta.store {
        state
            .responses
            .insert(meta.id.clone(), response.clone(), history);
    }
    Ok(Json(response).into_response())
}

fn finished_call(call: &ParsedToolCall) -> Value {
    call_item(
        &new_id("fc"),
        call.id.as_deref().unwrap_or_default(),
        &call.name,
        &call.arguments,
        "completed",
    )
}

/// `GET /v1/responses/{id}`.
pub(crate) async fn retrieve(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    state
        .responses
        .get(&id)
        .map(|stored| Json(stored.response))
        .ok_or_else(|| not_found(&id))
}

/// `DELETE /v1/responses/{id}`.
pub(crate) async fn delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    if state.responses.remove(&id) {
        Ok(Json(
            json!({"id": id, "object": "response.deleted", "deleted": true}),
        ))
    } else {
        Err(not_found(&id))
    }
}

/// `POST /v1/responses/{id}/cancel`: only a background response can be
/// cancelled, and this server runs none.
pub(crate) async fn cancel(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    match state.responses.get(&id) {
        Some(_) => Err(ApiError::bad_request(
            "Only responses created with `background: true` can be cancelled.",
        )),
        None => Err(not_found(&id)),
    }
}

/// `GET /v1/responses/{id}/input_items` query.
#[derive(Debug, Deserialize)]
pub(crate) struct ListQuery {
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    order: Option<String>,
    #[serde(default)]
    after: Option<String>,
}

/// `GET /v1/responses/{id}/input_items`: the conversation the response
/// answered, newest first unless `order=asc`, `limit` (1-100, default 20)
/// at a time after the item `after`.
pub(crate) async fn input_items(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Value>, ApiError> {
    let stored = state.responses.get(&id).ok_or_else(|| not_found(&id))?;
    let limit = query.limit.unwrap_or(20);
    if !(1..=100).contains(&limit) {
        return Err(ApiError::bad_request("`limit` must be between 1 and 100"));
    }
    let mut items = stored.history;
    match query.order.as_deref() {
        None | Some("desc") => items.reverse(),
        Some("asc") => {}
        Some(other) => {
            return Err(ApiError::bad_request(format!(
                "`order` must be `asc` or `desc`, not '{other}'"
            )))
        }
    }
    let start = match &query.after {
        Some(after) => {
            items
                .iter()
                .position(|item| item.get("id").and_then(Value::as_str) == Some(after))
                .ok_or_else(|| ApiError::bad_request(format!("no input item '{after}'")))?
                + 1
        }
        None => 0,
    };
    let page: Vec<Value> = items.iter().skip(start).take(limit).cloned().collect();
    let has_more = start + page.len() < items.len();
    let id_of = |item: Option<&Value>| item.and_then(|i| i.get("id")).cloned();
    Ok(Json(json!({
        "object": "list",
        "data": page,
        "first_id": id_of(page.first()),
        "last_id": id_of(page.last()),
        "has_more": has_more,
    })))
}

// ─── Streaming ──────────────────────────────────────────────────────────────

/// One streamed response's generation.
struct StreamRun<'a> {
    engine: &'a crate::engine::Engine,
    messages: &'a [crate::types::ChatMessage],
    tools: Option<&'a [crate::types::Tool]>,
    options: &'a crate::types::GenerationOptions,
    cancel: &'a AtomicBool,
    split: bool,
    show_reasoning: bool,
    metrics: &'a ServerMetrics,
}

impl StreamRun<'_> {
    /// Generate, sending each event; the finished response is stored (when
    /// asked) before `response.completed` goes out, so a client can continue
    /// from it as soon as it reads that.
    fn stream(self, mut emitter: Emitter, store: &ResponseStore, history: Vec<Value>) {
        emitter.start();
        let show_reasoning = self.show_reasoning;
        let result = stream_chat_output(
            self.engine,
            self.messages,
            self.tools,
            self.options,
            self.cancel,
            self.split,
            CALL_IDS,
            &mut |piece| emitter.piece(piece, show_reasoning),
        );
        match result {
            Ok((completion, calls)) => {
                self.metrics.record_generation(&completion.usage);
                let response = emitter.finish(&completion, &calls);
                if emitter.meta.store {
                    store.insert(emitter.meta.id.clone(), response.clone(), history);
                }
                let event = match response["status"].as_str() {
                    Some("incomplete") => "response.incomplete",
                    _ => "response.completed",
                };
                emitter.send(event, json!({"response": response}));
            }
            Err(error) => {
                tracing::error!("streaming response failed: {error}");
                emitter.failed("generation failed");
            }
        }
    }
}

/// The output item a stream has open.
enum Open {
    Reasoning {
        index: usize,
        text: String,
    },
    Text {
        index: usize,
        text: String,
    },
    Call {
        index: usize,
        call: usize,
        arguments: String,
    },
}

/// Writes a response's typed events: each output item is added, its
/// content streamed as deltas, then marked done, in the order the model
/// writes them.
struct Emitter {
    tx: tokio::sync::mpsc::UnboundedSender<(&'static str, String)>,
    meta: Meta,
    sequence: u64,
    /// The output so far: finished items, then the open one as it stands.
    output: Vec<Value>,
    open: Option<Open>,
    /// Whitespace before any answer text, held until text follows it (a
    /// tool call drops it, as the complete answer trims it).
    held: String,
}

impl Emitter {
    fn new(tx: tokio::sync::mpsc::UnboundedSender<(&'static str, String)>, meta: Meta) -> Self {
        Self {
            tx,
            meta,
            sequence: 0,
            output: Vec::new(),
            open: None,
            held: String::new(),
        }
    }

    fn send(&mut self, event: &'static str, mut data: Value) {
        data["type"] = json!(event);
        data["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        let _ = self.tx.send((event, data.to_string()));
    }

    fn start(&mut self) {
        let response = self.meta.response("in_progress", Vec::new());
        self.send("response.created", json!({"response": response}));
        self.send("response.in_progress", json!({"response": response}));
    }

    fn item_id(&self, index: usize) -> Value {
        self.output[index]["id"].clone()
    }

    /// Add an output item, closing the open one.
    fn add(&mut self, item: Value) -> usize {
        self.close();
        let index = self.output.len();
        self.output.push(item.clone());
        self.send(
            "response.output_item.added",
            json!({"output_index": index, "item": item}),
        );
        index
    }

    fn close(&mut self) {
        let Some(open) = self.open.take() else {
            return;
        };
        match open {
            Open::Reasoning { index, text } => {
                let id = self.item_id(index);
                let part = json!({"type": "reasoning_text", "text": text});
                self.send(
                    "response.reasoning_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "text": text}),
                );
                self.send(
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "part": part}),
                );
                self.output[index]["content"] = json!([part]);
            }
            Open::Text { index, text } => {
                let id = self.item_id(index);
                self.send(
                    "response.output_text.done",
                    json!({
                        "item_id": id, "output_index": index, "content_index": 0,
                        "text": text, "logprobs": [],
                    }),
                );
                let part = text_part(&text);
                self.send(
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "part": part}),
                );
                self.output[index]["content"] = json!([part]);
                self.output[index]["status"] = json!("completed");
            }
            Open::Call {
                index, arguments, ..
            } => {
                let id = self.item_id(index);
                let name = self.output[index]["name"].clone();
                self.send(
                    "response.function_call_arguments.done",
                    json!({
                        "item_id": id, "output_index": index,
                        "name": name, "arguments": arguments,
                    }),
                );
                self.output[index]["arguments"] = json!(arguments);
                self.output[index]["status"] = json!("completed");
            }
        }
        let index = self.output.len() - 1;
        let item = self.output[index].clone();
        self.send(
            "response.output_item.done",
            json!({"output_index": index, "item": item}),
        );
    }

    fn reasoning(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        let index = match &self.open {
            Some(Open::Reasoning { index, .. }) => *index,
            _ => {
                let index = self.add(json!({
                    "type": "reasoning", "id": new_id("rs"), "summary": [], "content": [],
                }));
                let id = self.item_id(index);
                self.send(
                    "response.content_part.added",
                    json!({
                        "item_id": id, "output_index": index, "content_index": 0,
                        "part": {"type": "reasoning_text", "text": ""},
                    }),
                );
                self.open = Some(Open::Reasoning {
                    index,
                    text: String::new(),
                });
                index
            }
        };
        if let Some(Open::Reasoning { text, .. }) = &mut self.open {
            text.push_str(delta);
        }
        let id = self.item_id(index);
        self.send(
            "response.reasoning_text.delta",
            json!({"item_id": id, "output_index": index, "content_index": 0, "delta": delta}),
        );
    }

    fn text(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        let index = match &self.open {
            Some(Open::Text { index, .. }) => *index,
            _ if delta.trim().is_empty() => {
                self.held.push_str(delta);
                return;
            }
            _ => self.open_text(),
        };
        let delta = std::mem::take(&mut self.held) + delta;
        if let Some(Open::Text { text, .. }) = &mut self.open {
            text.push_str(&delta);
        }
        let id = self.item_id(index);
        self.send(
            "response.output_text.delta",
            json!({
                "item_id": id, "output_index": index, "content_index": 0,
                "delta": delta, "logprobs": [],
            }),
        );
    }

    /// Add a message item and its (empty) text part.
    fn open_text(&mut self) -> usize {
        let index = self.add(message_item(&new_id("msg"), "in_progress", Vec::new()));
        let id = self.item_id(index);
        self.send(
            "response.content_part.added",
            json!({
                "item_id": id, "output_index": index, "content_index": 0,
                "part": text_part(""),
            }),
        );
        self.open = Some(Open::Text {
            index,
            text: String::new(),
        });
        index
    }

    fn call_start(&mut self, call: usize, call_id: &str, name: &str) {
        self.held.clear();
        let index = self.add(call_item(&new_id("fc"), call_id, name, "", "in_progress"));
        self.open = Some(Open::Call {
            index,
            call,
            arguments: String::new(),
        });
    }

    fn call_args(&mut self, call: usize, delta: &str) {
        let index = match &mut self.open {
            Some(Open::Call {
                index,
                call: open,
                arguments,
            }) if *open == call => {
                arguments.push_str(delta);
                *index
            }
            _ => {
                tracing::debug!("arguments for tool call {call} after its item closed");
                return;
            }
        };
        let id = self.item_id(index);
        self.send(
            "response.function_call_arguments.delta",
            json!({"item_id": id, "output_index": index, "delta": delta}),
        );
    }

    fn piece(&mut self, piece: ChatPiece<'_>, show_reasoning: bool) {
        match piece {
            ChatPiece::Reasoning(text) if show_reasoning => self.reasoning(text),
            ChatPiece::Reasoning(_) => {}
            ChatPiece::Text(text) => self.text(text),
            // `parallel_tool_calls: false`: the first call only.
            ChatPiece::CallStart { index, .. } | ChatPiece::CallArgs { index, .. }
                if index >= self.meta.max_calls() => {}
            ChatPiece::CallStart { index, id, name } => self.call_start(index, id, name),
            ChatPiece::CallArgs { index, delta } => self.call_args(index, delta),
        }
    }

    /// Close the open item and give the finished response, its function
    /// calls carrying the arguments as read from the answer.
    fn finish(&mut self, completion: &Completion, calls: &[ParsedToolCall]) -> Value {
        // An answer of nothing (or only whitespace) and no call is an empty
        // message, as the complete answer, trimmed, gives.
        let answered = self
            .output
            .iter()
            .any(|item| item["type"] == "message" || item["type"] == "function_call");
        if !answered {
            self.held.clear();
            self.open_text();
        }
        self.close();
        let mut calls = calls.iter();
        for item in &mut self.output {
            if item["type"] == "function_call" {
                if let Some(call) = calls.next() {
                    item["arguments"] = json!(call.arguments);
                }
            }
        }
        let output = std::mem::take(&mut self.output);
        self.meta.finished(completion, output)
    }

    fn failed(&mut self, message: &str) {
        let mut response = self
            .meta
            .response("failed", std::mem::take(&mut self.output));
        response["error"] = json!({"code": "server_error", "message": message});
        self.send("response.failed", json!({"response": response}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_become_a_chat_conversation() {
        let items = vec![
            json!({"role": "developer", "content": "be brief"}),
            json!({"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "weather?"},
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
            ]}),
            json!({"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "look it up"}]}),
            json!({"type": "function_call", "call_id": "c1", "name": "w", "arguments": "{}"}),
            json!({"type": "function_call", "call_id": "c2", "name": "w", "arguments": "{\"a\":1}"}),
            json!({"type": "function_call_output", "call_id": "c1", "output": "sunny"}),
            json!({"type": "function_call_output", "call_id": "c2", "output": [{"type": "input_text", "text": "rain"}]}),
            json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "mixed"}]}),
        ];
        let messages = chat_messages(&items).unwrap();
        assert_eq!(messages.len(), 6, "{messages:?}");
        assert_eq!(
            messages[0],
            json!({"role": "system", "content": "be brief"})
        );
        assert_eq!(
            messages[1]["content"][1]["image_url"]["url"],
            "data:image/png;base64,AA=="
        );
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["reasoning_content"], "look it up");
        assert_eq!(messages[2]["tool_calls"].as_array().unwrap().len(), 2);
        assert_eq!(
            messages[3],
            json!({"role": "tool", "tool_call_id": "c1", "content": "sunny"})
        );
        assert_eq!(messages[4]["content"], "rain");
        assert_eq!(
            messages[5],
            json!({"role": "assistant", "content": "mixed"})
        );
    }

    #[test]
    fn unsupported_items_and_tools_are_refused() {
        assert!(chat_messages(&[json!({"type": "item_reference", "id": "x"})]).is_err());
        assert!(chat_messages(&[
            json!({"role": "user", "content": [{"type": "input_file", "file_id": "f"}]})
        ])
        .is_err());
        assert!(chat_tools(&[json!({"type": "web_search"})]).is_err());
        assert!(chat_tool_choice(&json!({"type": "web_search"})).is_err());
        assert_eq!(
            chat_tool_choice(&json!({"type": "function", "name": "f"})).unwrap(),
            json!({"type": "function", "function": {"name": "f"}})
        );
    }

    #[test]
    fn the_store_forgets_the_oldest_past_its_cap() {
        let store = ResponseStore::default();
        for i in 0..STORE_ENTRIES + 2 {
            store.insert(format!("r{i}"), json!({}), Vec::new());
        }
        assert_eq!(store.len(), STORE_ENTRIES);
        assert!(store.get("r0").is_none() && store.get("r1").is_none());
        assert!(store.get("r2").is_some());
        assert!(store.remove("r2") && !store.remove("r2"));
        assert_eq!(store.len(), STORE_ENTRIES - 1);
        // One response past the whole budget is not kept at all.
        let huge = json!("x".repeat(STORE_BYTES + 1));
        store.insert("huge".into(), huge, Vec::new());
        assert!(store.get("huge").is_none());
        assert_eq!(store.len(), STORE_ENTRIES - 1);
    }
}
