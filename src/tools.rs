//! Tool-call parsing for model output.
//!
//! When a request supplies tool definitions, the chat template instructs the
//! model how to emit calls.  Different model families emit different markup;
//! this module recognises the common conventions and normalises them to
//! OpenAI-style `(name, arguments-JSON)` pairs:
//!
//! - **Hermes / Qwen / LFM2**: `<tool_call>{"name": …, "arguments": …}</tool_call>`
//!   (one tag per call, possibly surrounded by prose).
//! - **Qwen3-Coder**: XML-style calls —
//!   `<tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>`
//!   — either wrapped in `<tool_call>` tags or emitted bare.
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
                } else {
                    calls.extend(calls_from_xml(after));
                }
                rest = "";
                break;
            };
            if let Some(call) = call_from_json_str(&after[..end]) {
                calls.push(call);
            } else {
                calls.extend(calls_from_xml(&after[..end]));
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

    // ── Qwen3-Coder XML-style calls without a <tool_call> wrapper ───────────
    if trimmed.contains("<function=") {
        let mut calls = Vec::new();
        let mut prose = String::new();
        let mut rest = trimmed;
        while let Some(start) = rest.find("<function=") {
            prose.push_str(&rest[..start]);
            let after = &rest[start..];
            match after.find("</function>") {
                Some(end) => {
                    calls.extend(calls_from_xml(&after[..end + "</function>".len()]));
                    rest = &after[end + "</function>".len()..];
                }
                None => {
                    // Cut off mid-call — parse what is there and stop.
                    calls.extend(calls_from_xml(after));
                    rest = "";
                    break;
                }
            }
        }
        prose.push_str(rest);
        if !calls.is_empty() {
            return (prose.trim().to_string(), calls);
        }
    }

    (trimmed.to_string(), Vec::new())
}

/// Parse Qwen3-Coder-style XML function calls out of `text`.
///
/// Recognised shape (whitespace between tags is tolerated; parameter values
/// are raw text, trimmed, and kept as JSON strings unless they parse as some
/// other JSON type):
///
/// ```text
/// <function=get_weather>
/// <parameter=city>
/// Tokyo
/// </parameter>
/// </function>
/// ```
fn calls_from_xml(text: &str) -> Vec<ParsedToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(fn_start) = rest.find("<function=") {
        // Function name runs to the closing '>' of the opening tag.
        let name_start = fn_start + "<function=".len();
        let Some(name_end_rel) = rest[name_start..].find('>') else {
            break;
        };
        let name = rest[name_start..name_start + name_end_rel].trim().to_string();
        if name.is_empty() {
            break;
        }
        let body = &rest[name_start + name_end_rel + 1..];
        // The call ends at `</function>` or at the end of the text when the
        // generation was cut off.
        let body = match body.find("</function>") {
            Some(end) => {
                rest = &body[end + "</function>".len()..];
                &body[..end]
            }
            None => {
                rest = "";
                body
            }
        };

        let mut arguments = serde_json::Map::new();
        let mut params = body;
        while let Some(p_start) = params.find("<parameter=") {
            let key_start = p_start + "<parameter=".len();
            let Some(key_end_rel) = params[key_start..].find('>') else {
                break;
            };
            let key = params[key_start..key_start + key_end_rel].trim().to_string();
            let value_start = key_start + key_end_rel + 1;
            let value_body = &params[value_start..];
            let value_raw = match value_body.find("</parameter>") {
                Some(end) => {
                    params = &value_body[end + "</parameter>".len()..];
                    &value_body[..end]
                }
                None => {
                    params = "";
                    value_body
                }
            };
            if !key.is_empty() {
                let trimmed_value = value_raw.trim();
                let json_value = serde_json::from_str::<Value>(trimmed_value)
                    .unwrap_or_else(|_| Value::String(trimmed_value.to_string()));
                arguments.insert(key, json_value);
            }
        }

        calls.push(ParsedToolCall {
            name,
            arguments: Value::Object(arguments).to_string(),
        });
    }
    calls
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

    #[test]
    fn qwen3_coder_xml_call_inside_tool_call_tags() {
        // The exact shape Qwen3-Coder-30B-A3B emits with its GGUF template.
        let (content, calls) = parse_tool_calls(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["city"], "Tokyo");
    }

    #[test]
    fn qwen3_coder_xml_bare_multiple_params_with_types() {
        let (content, calls) = parse_tool_calls(
            "Calling now.\n<function=search>\n<parameter=query>\nrust async\n</parameter>\n<parameter=limit>5</parameter>\n<parameter=exact>true</parameter>\n</function>",
        );
        assert_eq!(content, "Calling now.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["query"], "rust async");
        assert_eq!(args["limit"], 5);
        assert_eq!(args["exact"], true);
    }

    #[test]
    fn qwen3_coder_xml_cut_off_still_parses_params() {
        let (_, calls) = parse_tool_calls(
            "<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo</parameter>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["city"], "Tokyo");
    }
}
