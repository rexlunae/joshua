//! Tool-call parsing for model output.
//!
//! When a request supplies tool definitions, the chat template instructs the
//! model how to emit calls.  Different model families emit different markup;
//! this module recognises the common conventions and normalises them to
//! OpenAI-style `(name, arguments-JSON)` pairs:
//!
//! - **Hermes / Qwen / LFM2**: `<tool_call>{"name": …, "arguments": …}</tool_call>`
//!   (one tag per call, possibly surrounded by prose).
//! - **Mistral**: `[TOOL_CALLS][{"name": …, "arguments": …}, …]`.
//! - **Llama 3.x / bare JSON**: the whole response is a single JSON object
//!   `{"name": …, "parameters": …}` (or `"arguments"`).

use serde_json::Value;

/// A tool call extracted from raw model output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    /// Function name.
    pub name: String,
    /// JSON-encoded argument object.
    pub arguments: String,
}

/// Extract tool calls from raw model output.
///
/// Returns the leftover prose (text outside any tool-call markup, trimmed)
/// and the calls in emission order.  An empty `Vec` means the output is a
/// plain text response.
pub fn parse_tool_calls(text: &str) -> (String, Vec<ParsedToolCall>) {
    // ── Hermes-style <tool_call> … </tool_call> tags ─────────────────────────
    if text.contains("<tool_call>") {
        let mut calls = Vec::new();
        let mut prose = String::new();
        let mut rest = text;
        while let Some(start) = rest.find("<tool_call>") {
            prose.push_str(&rest[..start]);
            let after = &rest[start + "<tool_call>".len()..];
            let Some(end) = after.find("</tool_call>") else {
                // Unterminated tag (generation was cut off) — try the payload
                // anyway, then stop scanning.
                if let Some(call) = call_from_json_str(after) {
                    calls.push(call);
                }
                rest = "";
                break;
            };
            if let Some(call) = call_from_json_str(&after[..end]) {
                calls.push(call);
            }
            rest = &after[end + "</tool_call>".len()..];
        }
        prose.push_str(rest);
        if !calls.is_empty() {
            return (prose.trim().to_string(), calls);
        }
    }

    // ── Mistral-style [TOOL_CALLS][…] ────────────────────────────────────────
    if let Some(idx) = text.find("[TOOL_CALLS]") {
        let payload = &text[idx + "[TOOL_CALLS]".len()..];
        if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(payload.trim()) {
            let calls: Vec<ParsedToolCall> =
                items.iter().filter_map(call_from_json_value).collect();
            if !calls.is_empty() {
                return (text[..idx].trim().to_string(), calls);
            }
        }
    }

    // ── Bare JSON object (Llama 3.x convention) ──────────────────────────────
    let trimmed = text.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        if let Some(call) = call_from_json_str(trimmed) {
            return (String::new(), vec![call]);
        }
    }

    (trimmed.to_string(), Vec::new())
}

/// Parse a JSON string as a single tool call.
fn call_from_json_str(s: &str) -> Option<ParsedToolCall> {
    let value: Value = serde_json::from_str(s.trim()).ok()?;
    call_from_json_value(&value)
}

/// Interpret a JSON value as `{"name": …, "arguments"|"parameters": …}`.
fn call_from_json_value(value: &Value) -> Option<ParsedToolCall> {
    let obj = value.as_object()?;
    let name = obj.get("name")?.as_str()?.to_string();
    let args = obj
        .get("arguments")
        .or_else(|| obj.get("parameters"))
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));
    // Arguments may arrive as an object or a pre-encoded string; normalise to
    // a JSON string, matching OpenAI's `function.arguments` field.
    let arguments = match args {
        Value::String(s) => s,
        other => other.to_string(),
    };
    Some(ParsedToolCall { name, arguments })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_has_no_tool_calls() {
        let (content, calls) = parse_tool_calls("The weather is sunny today.");
        assert_eq!(content, "The weather is sunny today.");
        assert!(calls.is_empty());
    }

    #[test]
    fn hermes_single_call() {
        let (content, calls) = parse_tool_calls(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].arguments).unwrap()["city"],
            "Paris"
        );
    }

    #[test]
    fn hermes_multiple_calls_with_prose() {
        let (content, calls) = parse_tool_calls(
            "Let me check both.\n\
             <tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>\n\
             <tool_call>{\"name\": \"b\", \"arguments\": {\"x\": 1}}</tool_call>",
        );
        assert_eq!(content, "Let me check both.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn hermes_unterminated_tag_still_parses() {
        let (_, calls) =
            parse_tool_calls("<tool_call>{\"name\": \"f\", \"arguments\": {\"k\": \"v\"}}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
    }

    #[test]
    fn mistral_tool_calls_array() {
        let (content, calls) = parse_tool_calls(
            "[TOOL_CALLS][{\"name\": \"get_time\", \"arguments\": {\"tz\": \"UTC\"}}]",
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_time");
    }

    #[test]
    fn llama3_bare_json_with_parameters_key() {
        let (content, calls) =
            parse_tool_calls("{\"name\": \"lookup\", \"parameters\": {\"q\": \"rust\"}}");
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].arguments).unwrap()["q"],
            "rust"
        );
    }

    #[test]
    fn bare_json_without_name_is_plain_content() {
        let input = "{\"answer\": 42}";
        let (content, calls) = parse_tool_calls(input);
        assert_eq!(content, input);
        assert!(calls.is_empty());
    }

    #[test]
    fn string_encoded_arguments_pass_through() {
        let (_, calls) = parse_tool_calls(
            "<tool_call>{\"name\": \"f\", \"arguments\": \"{\\\"a\\\": 1}\"}</tool_call>",
        );
        assert_eq!(calls[0].arguments, "{\"a\": 1}");
    }
}
