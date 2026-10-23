//! Anthropic Messages API adaptation.
//!
//! The engine and the rest of the server speak Joshua's native
//! [`ChatMessage`] / [`GenerationOptions`] types.  This module only translates
//! the Anthropic wire format into those types and back, so `POST
//! /v1/messages` has the same generation behaviour as the OpenAI endpoint.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::tools::ParsedToolCall;
use crate::types::{ChatMessage, FunctionDef, GenerationOptions, Tool, UsageInfo};

/// `POST /v1/messages` request body.
#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    /// Model id.
    pub model: String,
    /// Required output-token cap.
    pub max_tokens: u32,
    /// Conversation turns.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// System prompt, as text or text blocks.
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k sampling (0 disables).
    #[serde(default)]
    pub top_k: Option<i32>,
    /// Stop sequences.
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    /// Emit Anthropic SSE events.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Tools offered to the model.
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    /// `auto`, `any`, `tool`, or `none`.
    #[serde(default)]
    pub tool_choice: Option<Value>,
}

/// Anthropic message.
#[derive(Debug, Deserialize)]
pub struct Message {
    /// `user` or `assistant`.
    pub role: String,
    /// Text or content blocks.
    pub content: Content,
}

/// Message content: a bare string or a block array.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// Plain text.
    Text(String),
    /// Structured blocks.
    Blocks(Vec<Block>),
}

/// System prompt: a bare string or text blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    /// Plain text.
    Text(String),
    /// Text blocks.
    Blocks(Vec<Block>),
}

/// One Anthropic content block.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    /// Text.
    Text {
        /// The text.
        text: String,
    },
    /// Base64 image.
    Image {
        /// Image source.
        source: ImageSource,
    },
    /// A tool invocation emitted by the assistant.
    ToolUse {
        /// Call id.
        id: String,
        /// Function name.
        name: String,
        /// Parsed input object.
        input: Value,
    },
    /// The result of a previous tool call.
    ToolResult {
        /// Id of the assistant's tool_use block.
        tool_use_id: String,
        /// String or array of content blocks.
        content: Value,
        /// Whether the tool reported an error.
        #[serde(default)]
        is_error: Option<bool>,
    },
}

/// Anthropic image source.
#[derive(Debug, Deserialize)]
pub struct ImageSource {
    /// `base64`.
    #[serde(rename = "type")]
    pub source_type: String,
    /// MIME type, e.g. `image/png`.
    pub media_type: String,
    /// Base64 payload.
    pub data: String,
}

/// Anthropic tool definition.
#[derive(Debug, Deserialize)]
pub struct AnthropicTool {
    /// Function name.
    pub name: String,
    /// Description.
    #[serde(default)]
    pub description: Option<String>,
    /// JSON schema.
    #[serde(default)]
    pub input_schema: Option<Value>,
}

/// Convert an Anthropic request into the engine's chat messages, tools, and
/// sampling options.
pub fn to_chat_request(
    req: &MessagesRequest,
) -> Result<(Vec<ChatMessage>, Option<Vec<Tool>>, GenerationOptions), String> {
    if req.max_tokens == 0 {
        return Err("max_tokens must be at least 1".to_string());
    }
    let mut messages = Vec::new();
    if let Some(system) = &req.system {
        let text = match system {
            SystemPrompt::Text(text) => text.clone(),
            SystemPrompt::Blocks(blocks) => blocks_text(blocks),
        };
        if !text.is_empty() {
            messages.push(ChatMessage::text("system", text));
        }
    }

    for message in &req.messages {
        match message.role.as_str() {
            "assistant" => messages.push(assistant_message(message)?),
            "user" => user_messages(message, &mut messages)?,
            other => return Err(format!("unsupported message role '{other}'")),
        }
    }

    let tools = req
        .tools
        .as_ref()
        .filter(|tools| !tools.is_empty())
        .map(|tools| {
            tools
                .iter()
                .map(|tool| Tool {
                    tool_type: "function".to_string(),
                    function: FunctionDef {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: tool.input_schema.clone(),
                    },
                })
                .collect::<Vec<_>>()
        });

    let mut options = GenerationOptions::default();
    options.max_tokens = req.max_tokens;
    // Anthropic's defaults are deliberately different from Joshua's
    // OpenAI-oriented defaults: temperature 1.0, no nucleus/top-k truncation,
    // and no repetition penalty unless the caller asks for one.
    options.temperature = req.temperature.unwrap_or(1.0);
    options.top_p = req.top_p.unwrap_or(1.0);
    options.top_k = req.top_k.unwrap_or(0);
    options.min_p = 0.0;
    options.repetition_penalty = 1.0;
    options.stop_sequences = req.stop_sequences.clone().unwrap_or_default();

    Ok((messages, tools, options))
}

fn blocks_text(blocks: &[Block]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            Block::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn user_messages(message: &Message, out: &mut Vec<ChatMessage>) -> Result<(), String> {
    match &message.content {
        Content::Text(text) => out.push(ChatMessage::text("user", text.clone())),
        Content::Blocks(blocks) => {
            let mut text = String::new();
            let mut images = Vec::new();
            for block in blocks {
                match block {
                    Block::Text { text: part } => {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(part);
                    }
                    Block::Image { source } => {
                        images.push(format!("data:{};base64,{}", source.media_type, source.data));
                    }
                    Block::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        // A tool result is its own turn in the OpenAI-shaped
                        // chat template.  Flush any preceding user text first
                        // so the result follows the assistant tool call.
                        if !text.is_empty() || !images.is_empty() {
                            out.push(ChatMessage {
                                role: "user".to_string(),
                                content: std::mem::take(&mut text),
                                images: (!images.is_empty()).then(|| std::mem::take(&mut images)),
                                name: None,
                                tool_calls: None,
                                tool_call_id: None,
                            });
                        }
                        let (mut result, images) = content_to_message(content);
                        if is_error == &Some(true) {
                            result = format!("Error: {result}");
                        }
                        out.push(ChatMessage {
                            role: "tool".to_string(),
                            content: result,
                            images: (!images.is_empty()).then_some(images),
                            name: None,
                            tool_calls: None,
                            tool_call_id: Some(tool_use_id.clone()),
                        });
                    }
                    Block::ToolUse { .. } => {
                        return Err(
                            "tool_use blocks are only valid in assistant messages".to_string()
                        )
                    }
                }
            }
            if !text.is_empty() || !images.is_empty() {
                out.push(ChatMessage {
                    role: "user".to_string(),
                    content: text,
                    images: (!images.is_empty()).then_some(images),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }
    }
    Ok(())
}

fn assistant_message(message: &Message) -> Result<ChatMessage, String> {
    match &message.content {
        Content::Text(text) => Ok(ChatMessage::text("assistant", text.clone())),
        Content::Blocks(blocks) => {
            let mut text = String::new();
            let mut calls = Vec::new();
            for block in blocks {
                match block {
                    Block::Text { text: part } => {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(part);
                    }
                    Block::ToolUse { id, name, input } => calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serde_json::to_string(input)
                                .map_err(|e| e.to_string())?,
                        },
                    })),
                    Block::ToolResult { .. } => {
                        return Err("tool_result blocks are only valid in user messages".to_string())
                    }
                    Block::Image { .. } => {
                        return Err("assistant image blocks are not supported".to_string())
                    }
                }
            }
            Ok(ChatMessage {
                role: "assistant".to_string(),
                content: text,
                images: None,
                name: None,
                tool_calls: (!calls.is_empty()).then_some(Value::Array(calls)),
                tool_call_id: None,
            })
        }
    }
}

/// Normalise a parsed tool call's arguments to the `Value` used by the
/// Anthropic non-streaming `tool_use.input` field.
///
/// The model normally emits a JSON object, but a malformed call can carry a
/// bare string.  Treating that string as the input value keeps the response
/// valid JSON; the streaming emitter uses [`tool_arguments_json`] so both
/// paths describe the same input.
pub fn tool_input(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_string()))
}

/// Serialise [`tool_input`] for Anthropic's `input_json_delta.partial_json`.
pub fn tool_arguments_json(arguments: &str) -> String {
    tool_input(arguments).to_string()
}

/// Extract text and supported image blocks from an Anthropic content value.
///
/// Tool results may carry either a plain string or an array of blocks.  Text
/// blocks are joined as prose; base64 image blocks become `data:` URLs so
/// the engine's normal media path can attach them to the tool message.
fn content_to_message(content: &Value) -> (String, Vec<String>) {
    match content {
        Value::String(text) => (text.clone(), Vec::new()),
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            let mut images = Vec::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("image") => {
                        if let Some(source) = block.get("source") {
                            let is_base64 =
                                source.get("type").and_then(Value::as_str) == Some("base64");
                            if let (true, Some(media_type), Some(data)) = (
                                is_base64,
                                source.get("media_type").and_then(Value::as_str),
                                source.get("data").and_then(Value::as_str),
                            ) {
                                images.push(format!("data:{media_type};base64,{data}"));
                            }
                        }
                    }
                    _ => {
                        if let Some(text) = block.get("text").and_then(Value::as_str) {
                            parts.push(text);
                        }
                    }
                }
            }
            (parts.join("\n"), images)
        }
        Value::Null => (String::new(), Vec::new()),
        other => (other.to_string(), Vec::new()),
    }
}

/// An Anthropic response content block.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseBlock {
    /// Assistant text.
    Text {
        /// The text.
        text: String,
    },
    /// A tool invocation.
    ToolUse {
        /// Call id.
        id: String,
        /// Function name.
        name: String,
        /// Parsed input object.
        input: Value,
    },
}

/// Anthropic usage envelope.
#[derive(Debug, Serialize)]
pub struct MessagesUsage {
    /// Prompt tokens.
    pub input_tokens: u32,
    /// Generated tokens.
    pub output_tokens: u32,
}

/// `POST /v1/messages` response.
#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    /// Message id.
    pub id: String,
    /// Always `message`.
    #[serde(rename = "type")]
    pub message_type: String,
    /// Always `assistant`.
    pub role: String,
    /// Model id.
    pub model: String,
    /// Text and tool-use blocks.
    pub content: Vec<ResponseBlock>,
    /// `end_turn`, `max_tokens`, or `tool_use`.
    pub stop_reason: String,
    /// Matched stop sequence, if any.
    pub stop_sequence: Option<String>,
    /// Token usage.
    pub usage: MessagesUsage,
}

/// Build a non-streaming response from the model's text.
pub fn response(
    id: String,
    model: String,
    text: &str,
    usage: &UsageInfo,
    tools_offered: bool,
    max_tokens: u32,
    stop_sequence: Option<&str>,
) -> (MessagesResponse, Vec<ParsedToolCall>) {
    let (prose, calls) = if tools_offered {
        crate::tools::parse_tool_calls(text)
    } else {
        (text.to_string(), Vec::new())
    };
    let mut content = Vec::new();
    if !prose.is_empty() {
        content.push(ResponseBlock::Text { text: prose });
    }
    for call in &calls {
        content.push(ResponseBlock::ToolUse {
            id: format!("toolu_{}", uuid::Uuid::new_v4().simple()),
            name: call.name.clone(),
            input: tool_input(&call.arguments),
        });
    }
    let stop_reason = if !calls.is_empty() {
        "tool_use"
    } else if stop_sequence.is_some() {
        "stop_sequence"
    } else if usage.completion_tokens >= max_tokens {
        "max_tokens"
    } else {
        "end_turn"
    };
    let response = MessagesResponse {
        id,
        message_type: "message".to_string(),
        role: "assistant".to_string(),
        model,
        content,
        stop_reason: stop_reason.to_string(),
        stop_sequence: stop_sequence.map(str::to_string),
        usage: MessagesUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        },
    };
    (response, calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_use_and_tool_result_round_trip_through_chat_messages() {
        let value = json!({
            "model": "m",
            "max_tokens": 8,
            "system": "be brief",
            "tools": [{
                "name": "add",
                "description": "add two numbers",
                "input_schema": {"type": "object"}
            }],
            "messages": [
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me add."},
                    {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 2, "b": 2}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "4"}
                ]}
            ]
        });
        let request: MessagesRequest = serde_json::from_value(value).unwrap();
        let (messages, tools, options) = to_chat_request(&request).unwrap();
        assert_eq!(options.max_tokens, 8);
        assert_eq!(tools.unwrap()[0].function.name, "add");
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].content, "2+2?");
        assert_eq!(messages[2].role, "assistant");
        assert!(messages[2].tool_calls.is_some());
        assert_eq!(messages[3].role, "tool");
        assert_eq!(messages[3].tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(messages[3].content, "4");
    }

    #[test]
    fn response_maps_tool_markup_to_tool_use_blocks() {
        let usage = UsageInfo {
            prompt_tokens: 5,
            completion_tokens: 3,
            total_tokens: 8,
        };
        let (response, calls) = response(
            "msg_1".to_string(),
            "m".to_string(),
            "<tool_call>{\"name\":\"add\",\"arguments\":{\"a\":2,\"b\":2}}</tool_call>",
            &usage,
            true,
            16,
            None,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(response.stop_reason, "tool_use");
        assert!(matches!(
            &response.content[0],
            ResponseBlock::ToolUse { .. }
        ));
    }

    #[test]
    fn tool_result_images_are_attached_to_the_tool_message() {
        let value = json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "what is this?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "screenshot", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                        {"type": "text", "text": "captured"},
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "AQID"
                        }}
                    ]}
                ]}
            ]
        });
        let request: MessagesRequest = serde_json::from_value(value).unwrap();
        let (messages, _, _) = to_chat_request(&request).unwrap();
        assert_eq!(messages[2].role, "tool");
        assert_eq!(messages[2].content, "captured");
        assert_eq!(
            messages[2].images.as_deref(),
            Some(&["data:image/png;base64,AQID".to_string()][..])
        );
    }

    #[test]
    fn malformed_tool_arguments_are_normalised_for_streaming_and_json() {
        assert_eq!(
            tool_input("not-json"),
            Value::String("not-json".to_string())
        );
        assert_eq!(tool_input("{\"a\":1}"), json!({"a": 1}));

        let streamed = tool_arguments_json("not-json");
        assert_eq!(streamed, "\"not-json\"");
        assert!(
            serde_json::from_str::<Value>(&streamed).is_ok(),
            "partial_json must be valid JSON: {streamed}"
        );
    }

    #[test]
    fn response_reports_the_matched_stop_sequence() {
        let usage = UsageInfo {
            prompt_tokens: 5,
            completion_tokens: 3,
            total_tokens: 8,
        };
        let (response, calls) = response(
            "msg_1".to_string(),
            "m".to_string(),
            "hello ",
            &usage,
            false,
            16,
            Some("END"),
        );
        assert!(calls.is_empty());
        assert_eq!(response.stop_reason, "stop_sequence");
        assert_eq!(response.stop_sequence.as_deref(), Some("END"));
    }
}
