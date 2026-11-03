//! Kimi's tool calls.
//!
//! - **Kimi K2** (and Kimi-Linear): a section of calls whose header is the
//!   model's own call id, `functions.<name>:<index>`:
//!
//!   ```text
//!   <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|><|tool_calls_section_end|>
//!   ```
//!
//! - **Kimi K3**: XTML tags built from `<|open|>` / `<|close|>` / `<|sep|>`,
//!   one tag per argument with its JSON type, inside the message's `tools`
//!   channel after its `response` channel:
//!
//!   ```text
//!   <|open|>response<|sep|>On it.<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="get_weather" index="1"<|sep|><|open|>argument key="city" type="string"<|sep|>Paris<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>
//!   ```
//!
//!   The `response` and `message` tags around the answer are dropped from
//!   the content, as vLLM's `kimi_k3` parser does.

use super::args::{self, Header, TaggedParams};
use super::generic::close_after;
use super::json;
use super::{header_pending, section, CallRead, Cx, Grammar, Marker, PartialCall, Scan};

// ─── Kimi K2 ────────────────────────────────────────────────────────────────

const K2_SECTION: &str = "<|tool_calls_section_begin|>";
const K2_SECTION_END: &str = "<|tool_calls_section_end|>";
const K2_CALL: &str = "<|tool_call_begin|>";
const K2_ARGS: &str = "<|tool_call_argument_begin|>";
const K2_CALL_END: &str = "<|tool_call_end|>";

/// Kimi K2 and Kimi-Linear.
pub(super) struct KimiK2;

pub(super) static KIMI_K2: KimiK2 = KimiK2;

impl Grammar for KimiK2 {
    fn markers(&self) -> &'static [Marker] {
        const MARKERS: &[Marker] = &[Marker::anywhere(K2_SECTION), Marker::anywhere(K2_CALL)];
        MARKERS
    }

    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan {
        section(
            region,
            &[K2_SECTION],
            &[K2_CALL],
            &[K2_SECTION_END],
            cx,
            k2_call,
        )
    }
}

/// One call after `<|tool_call_begin|>`.  The name is the id's last dotted
/// component before `:<index>` (vLLM's `_extract_tool_id_and_name`); the
/// whole id is reported back, since the chat template expects it again in
/// the history.
fn k2_call(s: &str, cx: &Cx<'_>) -> CallRead {
    let Some(header_len) = s.find(K2_ARGS) else {
        return header_pending(s, K2_ARGS, cx);
    };
    let header = s[..header_len].trim();
    let (id, name) = match header.rsplit_once(':') {
        Some((head, index)) if !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()) => {
            (Some(header.to_string()), head)
        }
        _ => (None, header),
    };
    let name = name.strip_prefix("functions.").unwrap_or(name);
    if name.is_empty() || name.contains(char::is_whitespace) {
        return Err(());
    }
    let args_at = header_len + K2_ARGS.len();
    // The section's end also closes a call whose own end is missing.
    let raw = json::raw_args(&s[args_at..], &[K2_CALL_END, K2_SECTION_END], cx.finished);
    let end = raw.end.map(|end| {
        let end = args_at + end;
        end - if s[..end].ends_with(K2_SECTION_END) {
            K2_SECTION_END.len()
        } else {
            0
        }
    });
    let call = PartialCall {
        id,
        name: name.to_string(),
        args: raw.stream,
        done: raw.done,
    };
    Ok(Some((call, end)))
}

// ─── Kimi K3 ────────────────────────────────────────────────────────────────

const SEP: &str = "<|sep|>";
const K3_TOOLS: &str = "<|open|>tools<|sep|>";
const K3_TOOLS_END: &str = "<|close|>tools<|sep|>";
const K3_CALL: &str = "<|open|>call";
const K3_CALL_END: &str = "<|close|>call<|sep|>";
const K3_JSON: &str = "<|open|>json";
const K3_JSON_END: &str = "<|close|>json<|sep|>";

/// The tags around the answer, dropped from the content.
const K3_FRAMING: &[&str] = &[
    "<|open|>response<|sep|>",
    "<|close|>response<|sep|>",
    "<|close|>message<|sep|>",
    "<|end_of_msg|>",
];

/// `<|open|>argument key="k" type="t"<|sep|>value<|close|>argument<|sep|>`:
/// a `string` value is the raw text and any other type is JSON (vLLM's
/// `_decode_call`); without a type the tool's schema decides.
const K3_PARAMS: TaggedParams = TaggedParams {
    call_end: &[K3_CALL_END],
    next: &["<|open|>argument", K3_CALL_END],
    value_end: &["<|close|>argument<|sep|>", "<|open|>argument", K3_CALL_END],
    value_close: &["<|close|>argument<|sep|>"],
    header: |s| {
        let Some(len) = s.find(SEP) else {
            return Header::Partial;
        };
        let attrs = &s[..len];
        match args::attribute(attrs, "key") {
            Some(key) => Header::Done {
                key,
                string: args::attribute(attrs, "type").map(|kind| kind == "string"),
                len: len + SEP.len(),
            },
            None => Header::Bad,
        }
    },
    trim: |value| value,
};

/// Kimi K3.
pub(super) struct KimiK3;

pub(super) static KIMI_K3: KimiK3 = KimiK3;

impl Grammar for KimiK3 {
    fn markers(&self) -> &'static [Marker] {
        const MARKERS: &[Marker] = &[
            Marker::anywhere(K3_TOOLS),
            Marker::anywhere(K3_FRAMING[0]),
            Marker::anywhere(K3_FRAMING[1]),
            Marker::anywhere(K3_FRAMING[2]),
            Marker::anywhere(K3_FRAMING[3]),
        ];
        MARKERS
    }

    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan {
        if let Some(tag) = K3_FRAMING.iter().find(|tag| region.starts_with(**tag)) {
            return Scan::Calls {
                calls: Vec::new(),
                end: Some(tag.len()),
            };
        }
        section(
            region,
            &[K3_TOOLS],
            &[K3_CALL],
            &[K3_TOOLS_END],
            cx,
            k3_call,
        )
    }
}

/// One ` tool="name" index="n"<|sep|>…<|close|>call<|sep|>` call after its
/// `<|open|>call` opener.  The arguments are tags, or a JSON object in a
/// `json` tag (how the template writes string-encoded arguments back).
fn k3_call(s: &str, cx: &Cx<'_>) -> CallRead {
    let Some(attrs_len) = s.find(SEP) else {
        return header_pending(s, SEP, cx);
    };
    let Some(name) = args::attribute(&s[..attrs_len], "tool") else {
        return Err(());
    };
    if name.is_empty() {
        return Err(());
    }
    let body_at = attrs_len + SEP.len();
    let body = &s[body_at..];
    if body.starts_with(K3_JSON) {
        let Some(json_attrs) = body.find(SEP) else {
            return Ok(Some((
                PartialCall {
                    name: name.into_owned(),
                    ..PartialCall::default()
                },
                None,
            )));
        };
        let json_at = body_at + json_attrs + SEP.len();
        let raw = json::raw_args(&s[json_at..], &[K3_JSON_END], cx.finished);
        let end = raw
            .end
            .and_then(|end| close_after(s, json_at + end, K3_CALL_END, cx.finished));
        let call = PartialCall {
            id: None,
            name: name.into_owned(),
            args: raw.stream,
            done: raw.done,
        };
        return Ok(Some((call, end)));
    }
    let params = K3_PARAMS.read(body, &name, cx);
    Ok(Some((
        params.call(&name, cx),
        params.end.map(|end| body_at + end),
    )))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{args, assert_streams_like_parse, parse};
    use super::super::{parse_tool_calls, ToolCallFormat};
    use crate::types::{FunctionDef, Tool};
    use serde_json::json;

    const K2: ToolCallFormat = ToolCallFormat::KimiK2;
    const K3: ToolCallFormat = ToolCallFormat::KimiK3;

    /// llama.cpp's Kimi-K2-Thinking cases: the model's id comes back.
    #[test]
    fn k2_calls_keep_the_model_ids() {
        let text = "<|tool_calls_section_begin|>\
                    <|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|>\
                    <|tool_call_begin|>functions.special_function_with_opt:1<|tool_call_argument_begin|>{\"arg1\": 1, \"arg2\": 2}<|tool_call_end|>\
                    <|tool_calls_section_end|>";
        let (content, calls) = parse(K2, text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "special_function");
        assert_eq!(calls[0].id.as_deref(), Some("functions.special_function:0"));
        assert_eq!(calls[1].name, "special_function_with_opt");
        assert_eq!(
            calls[1].id.as_deref(),
            Some("functions.special_function_with_opt:1")
        );
        assert_eq!(args(&calls[1]), json!({"arg1": 1, "arg2": 2}));
        assert_streams_like_parse(K2, &[], text);
    }

    #[test]
    fn k2_content_before_a_call_and_empty_arguments() {
        let text = "Hello, world!\nWhat's up?\
                    <|tool_calls_section_begin|><|tool_call_begin|>functions.empty_args:0<|tool_call_argument_begin|>\
                    {}<|tool_call_end|><|tool_calls_section_end|>";
        let (content, calls) = parse(K2, text);
        assert_eq!(content, "Hello, world!\nWhat's up?");
        assert_eq!(calls[0].name, "empty_args");
        assert_eq!(calls[0].arguments, "{}");
        assert_streams_like_parse(K2, &[], text);
    }

    /// A call cut off mid-arguments keeps what was generated (llama.cpp's
    /// partial case), and a section closed without the call's own end.
    #[test]
    fn k2_partial_and_unclosed_calls() {
        let (_, calls) = parse(
            K2,
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": ",
        );
        assert_eq!(calls[0].name, "special_function");
        assert_eq!(calls[0].arguments, "{\"arg1\":");
        let text = "<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{\"a\": 1}<|tool_calls_section_end|>";
        let (content, calls) = parse(K2, text);
        assert_eq!((content.as_str(), calls.len()), ("", 1));
        assert_streams_like_parse(K2, &[], text);
    }

    fn python_tool() -> Tool {
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "python".into(),
                description: None,
                parameters: Some(
                    json!({"type": "object", "properties": {"code": {"type": "string"}}}),
                ),
            },
        }
    }

    /// llama.cpp's Kimi-K3 cases (after the reasoning split).
    #[test]
    fn k3_calls_and_response_channel() {
        let text = "<|open|>response<|sep|>On it.<|close|>response<|sep|>\
                    <|open|>tools<|sep|>\
                    <|open|>call tool=\"special_function\" index=\"1\"<|sep|>\
                    <|open|>argument key=\"arg1\" type=\"number\"<|sep|>1<|close|>argument<|sep|>\
                    <|close|>call<|sep|>\
                    <|open|>call tool=\"special_function_with_opt\" index=\"2\"<|sep|>\
                    <|open|>argument key=\"arg1\" type=\"number\"<|sep|>1<|close|>argument<|sep|>\
                    <|open|>argument key=\"arg2\" type=\"number\"<|sep|>2<|close|>argument<|sep|>\
                    <|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>";
        let (content, calls) = parse(K3, text);
        assert_eq!(content, "On it.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "special_function");
        assert_eq!(calls[0].arguments, r#"{"arg1":1}"#);
        assert_eq!(calls[1].arguments, r#"{"arg1":1,"arg2":2}"#);
        assert_streams_like_parse(K3, &[], text);

        let (content, calls) = parse(
            K3,
            "<|open|>response<|sep|>Use the response tag, then message the handler.\
             <|close|>response<|sep|><|close|>message<|sep|>",
        );
        assert_eq!(content, "Use the response tag, then message the handler.");
        assert!(calls.is_empty());
    }

    /// A `string` argument keeps its text verbatim, escaped attributes are
    /// read back, and a string-typed value streams before it closes.
    #[test]
    fn k3_string_arguments_and_escaped_attributes() {
        let tools = [python_tool()];
        let text = "<|open|>response<|sep|><|close|>response<|sep|>\
                    <|open|>tools<|sep|>\
                    <|open|>call tool=\"python\" index=\"1\"<|sep|>\
                    <|open|>argument key=\"code\" type=\"string\"<|sep|>print('hey')\n42\
                    <|close|>argument<|sep|>\
                    <|open|>argument key=\"say &quot;hi&quot; &amp; go\" type=\"boolean\"<|sep|>true<|close|>argument<|sep|>\
                    <|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>";
        let (content, calls) = parse_tool_calls(K3, &tools, text);
        assert_eq!(content, "");
        assert_eq!(
            args(&calls[0]),
            json!({"code": "print('hey')\n42", "say \"hi\" & go": true})
        );
        assert_streams_like_parse(K3, &tools, text);
    }

    #[test]
    fn k3_json_block_arguments() {
        let text = "<|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>\
                    <|open|>json type=\"object\"<|sep|>{\"a\": [1, 2]}<|close|>json<|sep|>\
                    <|close|>call<|sep|><|close|>tools<|sep|>";
        let (_, calls) = parse(K3, text);
        assert_eq!(calls[0].arguments, r#"{"a":[1,2]}"#);
        assert_streams_like_parse(K3, &[], text);
    }
}
