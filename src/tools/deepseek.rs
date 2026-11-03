//! DeepSeek's tool calls.
//!
//! - **V3 / R1** (fenced JSON) and **V3.1** (bare JSON), in a
//!   `<｜tool▁calls▁begin｜>` section:
//!
//!   ```text
//!   <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
//!   ```json
//!   {"city": "Paris"}
//!   ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
//!   <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
//!   ```
//!
//! - **V3.2 / V4 / V4.1** DSML, one tag per parameter.  V3.2 names the block
//!   `function_calls`, V4 `tool_calls`, and V4.1 writes a space after the
//!   `｜DSML｜` prefix and names it `calls`:
//!
//!   ```text
//!   <｜DSML｜tool_calls>
//!   <｜DSML｜invoke name="get_weather">
//!   <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
//!   <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
//!   </｜DSML｜invoke>
//!   </｜DSML｜tool_calls>
//!   ```

use super::args::{self, Header, TaggedParams};
use super::generic::close_after;
use super::json;
use super::{header_pending, section, CallRead, Cx, Grammar, Marker, PartialCall, Scan};

// ─── V3 / R1 / V3.1 ─────────────────────────────────────────────────────────

/// The section opener, with the spellings R1 distills write instead of the
/// special token (llama.cpp accepts the same ones).
const SECTION: &[&str] = &[
    "<｜tool▁calls▁begin｜>",
    "<｜tool_calls_begin｜>",
    "<｜tool calls begin｜>",
    "<｜tool\\_calls\\_begin｜>",
];
const SECTION_END: &str = "<｜tool▁calls▁end｜>";
const CALL: &str = "<｜tool▁call▁begin｜>";
const CALL_END: &str = "<｜tool▁call▁end｜>";
const SEP: &str = "<｜tool▁sep｜>";
const FENCE: &str = "```json";
const FENCE_END: &str = "```";

/// DeepSeek V3, R1 and V3.1.
pub(super) struct DeepSeekV3;

pub(super) static DEEPSEEK_V3: DeepSeekV3 = DeepSeekV3;

impl Grammar for DeepSeekV3 {
    fn markers(&self) -> &'static [Marker] {
        const MARKERS: &[Marker] = &[
            Marker::anywhere(SECTION[0]),
            Marker::anywhere(SECTION[1]),
            Marker::anywhere(SECTION[2]),
            Marker::anywhere(SECTION[3]),
            Marker::anywhere(CALL),
        ];
        MARKERS
    }

    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan {
        section(region, SECTION, &[CALL], &[SECTION_END], cx, v3_call)
    }
}

/// One call after `<｜tool▁call▁begin｜>`.  V3 writes the call type, the
/// separator, then the name on its own line and fenced JSON; V3.1 writes
/// the name, the separator and the bare JSON.
fn v3_call(s: &str, cx: &Cx<'_>) -> CallRead {
    let Some(sep) = s.find(SEP) else {
        return header_pending(s, SEP, cx);
    };
    let head = s[..sep].trim();
    let after_sep = sep + SEP.len();
    let after = &s[after_sep..];
    let fenced = match after.find('\n') {
        _ if after.trim_start().starts_with(['{', '[']) => None,
        Some(line) if !after[..line].contains(CALL_END) => {
            let fence = &after[line + 1..];
            if fence.starts_with(FENCE) {
                Some(line)
            } else if !cx.finished && FENCE.starts_with(fence) {
                return Ok(None);
            } else {
                None
            }
        }
        None if !cx.finished && !after.contains(CALL_END) => return Ok(None),
        _ => None,
    };
    let (name, args_at, ends): (&str, usize, &[&str]) = match fenced {
        Some(line) => {
            let fence = after_sep + line + 1 + FENCE.len();
            let body = fence + usize::from(s[fence..].starts_with('\n'));
            (after[..line].trim(), body, &[FENCE_END])
        }
        None => (head, after_sep, &[CALL_END]),
    };
    if name.is_empty() || name.contains(char::is_whitespace) {
        return Err(());
    }
    let raw = json::raw_args(&s[args_at..], ends, cx.finished);
    let end = match (fenced, raw.end) {
        (_, None) => None,
        (None, Some(end)) => Some(args_at + end),
        // The fence closes the arguments; the call's own end follows.
        (Some(_), Some(end)) => close_after(s, args_at + end, CALL_END, cx.finished),
    };
    let call = PartialCall {
        id: None,
        name: name.to_string(),
        args: raw.stream,
        done: raw.done,
    };
    Ok(Some((call, end)))
}

// ─── V3.2 / V4 / V4.1 DSML ──────────────────────────────────────────────────

/// The call blocks: V3.2's, V4's (and the misspellings vLLM has seen from
/// it), and V4.1's spaced one.
const BLOCKS: &[&str] = &[
    "<｜DSML｜function_calls>",
    "<｜DSML｜tool_calls>",
    "<｜DSML｜toolcalls>",
    "<｜DSML｜tool>",
    "<｜DSML｜ calls>",
];
const BLOCK_ENDS: &[&str] = &[
    "</｜DSML｜function_calls>",
    "</｜DSML｜tool_calls>",
    "</｜DSML｜ calls>",
];
const INVOKE: &[&str] = &["<｜DSML｜invoke", "<｜DSML｜ invoke"];

/// `<｜DSML｜parameter name="key" string="true">value</｜DSML｜parameter>`.
/// `string="true"` keeps the value as text and `string="false"` reads it as
/// JSON; without the attribute the tool's schema decides.  A value also
/// ends at the next parameter or the call's end when its closing tag is
/// missing (vLLM's `_PARAM_RE`).
const DSML_PARAMS: TaggedParams = TaggedParams {
    call_end: &[
        "</｜DSML｜invoke>",
        "</｜DSML｜ invoke>",
        "</｜DSML｜function_calls>",
        "</｜DSML｜tool_calls>",
        "</｜DSML｜ calls>",
    ],
    next: &[
        "<｜DSML｜parameter",
        "<｜DSML｜ parameter",
        "</｜DSML｜invoke>",
        "</｜DSML｜ invoke>",
        "</｜DSML｜function_calls>",
        "</｜DSML｜tool_calls>",
        "</｜DSML｜ calls>",
    ],
    value_end: &[
        "</｜DSML｜parameter>",
        "</｜DSML｜ parameter>",
        "<｜DSML｜parameter",
        "<｜DSML｜ parameter",
        "</｜DSML｜invoke>",
        "</｜DSML｜ invoke>",
    ],
    value_close: &["</｜DSML｜parameter>", "</｜DSML｜ parameter>"],
    header: |s| {
        let Some(len) = s.find('>') else {
            return Header::Partial;
        };
        let attrs = &s[..len];
        match args::attribute(attrs, "name") {
            Some(key) => Header::Done {
                key,
                string: args::attribute(attrs, "string").map(|string| string == "true"),
                len: len + 1,
            },
            None => Header::Bad,
        }
    },
    trim: |value| value,
};

/// DeepSeek V3.2, V4 and V4.1.
pub(super) struct Dsml;

pub(super) static DSML: Dsml = Dsml;

impl Grammar for Dsml {
    fn markers(&self) -> &'static [Marker] {
        const MARKERS: &[Marker] = &[
            Marker::anywhere(BLOCKS[0]),
            Marker::anywhere(BLOCKS[1]),
            Marker::anywhere(BLOCKS[2]),
            Marker::anywhere(BLOCKS[3]),
            Marker::anywhere(BLOCKS[4]),
            Marker::anywhere(INVOKE[0]),
            Marker::anywhere(INVOKE[1]),
        ];
        MARKERS
    }

    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan {
        section(region, BLOCKS, INVOKE, BLOCK_ENDS, cx, invoke)
    }
}

/// One ` name="…">` invoke after its `<｜DSML｜invoke` opener.
fn invoke(s: &str, cx: &Cx<'_>) -> CallRead {
    const NAME: &str = " name=\"";
    let Some(rest) = s.strip_prefix(NAME) else {
        return if !cx.finished && NAME.starts_with(s) {
            Ok(None)
        } else {
            Err(())
        };
    };
    let Some(name_len) = rest.find('"') else {
        return if cx.finished || rest.contains(['\n', '<']) {
            Err(())
        } else {
            Ok(None)
        };
    };
    let name = &rest[..name_len];
    let after_name = NAME.len() + name_len + 1;
    let header_end = match s[after_name..].find('>') {
        Some(at) if s[after_name..after_name + at].trim().is_empty() => after_name + at + 1,
        Some(_) => return Err(()),
        None if cx.finished => s.len(),
        None => return Ok(None),
    };
    if name.is_empty() {
        return Err(());
    }
    let params = DSML_PARAMS.read(&s[header_end..], name, cx);
    let end = params.end.map(|end| {
        let end = header_end + end;
        // A block end that also ended the call is left for the section.
        match BLOCK_ENDS.iter().find(|block| s[..end].ends_with(**block)) {
            Some(block) => end - block.len(),
            None => end,
        }
    });
    Ok(Some((params.call(name, cx), end)))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{args, assert_streams_like_parse, parse};
    use super::super::{parse_tool_calls, ToolCallFormat};
    use crate::types::{FunctionDef, Tool};
    use serde_json::json;

    const V3: ToolCallFormat = ToolCallFormat::DeepSeekV3;
    const DSML: ToolCallFormat = ToolCallFormat::DeepSeekDsml;

    fn tool(name: &str, parameters: serde_json::Value) -> Tool {
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: name.into(),
                description: None,
                parameters: Some(parameters),
            },
        }
    }

    /// DeepSeek V3 / R1's fenced calls (vLLM's `deepseek_v3` and
    /// llama.cpp's R1 cases).
    #[test]
    fn v3_fenced_calls() {
        let text = "Let me check.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n\
                    ```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜>\n\
                    <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n\
                    ```json\n{\"tz\": \"CET\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        let (content, calls) = parse(V3, text);
        assert_eq!(content, "Let me check.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
        assert_eq!(calls[1].name, "get_time");
        assert_eq!(args(&calls[1]), json!({"tz": "CET"}));
        assert_streams_like_parse(V3, &[], text);
    }

    /// llama.cpp's DeepSeek-V3.1 cases (after the reasoning split).
    #[test]
    fn v31_bare_calls() {
        let text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \
                    \"XYZCITY\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        let (content, calls) = parse(V3, text);
        assert_eq!(content, "");
        assert_eq!(calls[0].name, "get_time");
        assert_eq!(calls[0].arguments, r#"{"city":"XYZCITY"}"#);
        assert_streams_like_parse(V3, &[], text);

        let text = "CONTENT<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \
                    \"Paris\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \
                    \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        let (content, calls) = parse(V3, text);
        assert_eq!(content, "CONTENT");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].name, "get_weather");
        assert_streams_like_parse(V3, &[], text);
    }

    #[test]
    fn v3_cut_off_call_keeps_its_name_and_arguments() {
        let (_, calls) = parse(
            V3,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Pa",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_time");
        assert_eq!(calls[0].arguments, "{\"city\": \"Pa");
    }

    #[test]
    fn v3_markers_in_prose_without_a_call_stay_text() {
        let text = "The section opener is <｜tool▁calls▁begin｜> in the docs.";
        let (content, calls) = parse(V3, text);
        assert_eq!(content, text);
        assert!(calls.is_empty());
    }

    /// llama.cpp's DeepSeek V3.2 cases: `string="false"` reads JSON.
    #[test]
    fn dsml_v32_calls() {
        let text = "<｜DSML｜function_calls>\n\
                    <｜DSML｜invoke name=\"special_function\">\n\
                    <｜DSML｜parameter name=\"arg1\" string=\"false\">1</｜DSML｜parameter>\n\
                    </｜DSML｜invoke>\n\
                    </｜DSML｜function_calls>";
        let (content, calls) = parse(DSML, text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "special_function");
        assert_eq!(calls[0].arguments, r#"{"arg1":1}"#);
        assert_streams_like_parse(DSML, &[], text);

        let text = "Let me call the function.\n\n\
                    <｜DSML｜function_calls>\n\
                    <｜DSML｜invoke name=\"get_time\">\n\
                    <｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n\
                    </｜DSML｜invoke>\n\
                    <｜DSML｜invoke name=\"get_weather\">\n\
                    <｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n\
                    </｜DSML｜invoke>\n\
                    </｜DSML｜function_calls>";
        let (content, calls) = parse(DSML, text);
        assert_eq!(content, "Let me call the function.");
        assert_eq!(calls.len(), 2);
        assert_eq!(args(&calls[1]), json!({"city": "Paris"}));
        assert_streams_like_parse(DSML, &[], text);
    }

    /// V4's `tool_calls` block, with a string parameter that spells a
    /// number and a multi-line one that streams before it closes.
    #[test]
    fn dsml_v4_string_parameters_stay_text_and_stream() {
        let tools = [tool(
            "write_file",
            json!({"type": "object", "properties": {"path": {"type": "string"}, "body": {"type": "string"}}}),
        )];
        let text = "<｜DSML｜tool_calls>\n\
                    <｜DSML｜invoke name=\"write_file\">\n\
                    <｜DSML｜parameter name=\"path\" string=\"true\">007</｜DSML｜parameter>\n\
                    <｜DSML｜parameter name=\"body\" string=\"true\">line one\n  line two\n</｜DSML｜parameter>\n\
                    </｜DSML｜invoke>\n\
                    </｜DSML｜tool_calls>";
        let (_, calls) = parse_tool_calls(DSML, &tools, text);
        assert_eq!(
            args(&calls[0]),
            json!({"path": "007", "body": "line one\n  line two\n"})
        );
        assert_streams_like_parse(DSML, &tools, text);

        let mut stream = super::super::ToolCallStream::new(DSML, Some(&tools));
        let cut = text.find("two").unwrap();
        let streamed: String = stream
            .push(&text[..cut])
            .into_iter()
            .filter_map(|event| match event {
                super::super::ToolEvent::CallArgs { delta, .. } => Some(delta),
                _ => None,
            })
            .collect();
        assert!(
            streamed.ends_with(r#""body":"line one\n  line "#),
            "{streamed}"
        );
    }

    /// V4.1's spaced tags, a parameter without a `string` attribute (the
    /// schema decides), and one whose closing tag is missing.
    #[test]
    fn dsml_v41_spaced_tags() {
        let tools = [tool(
            "search",
            json!({"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}),
        )];
        let text = "<｜DSML｜ calls>\n\
                    <｜DSML｜ invoke name=\"search\">\n\
                    <｜DSML｜ parameter name=\"query\">42</｜DSML｜ parameter>\n\
                    <｜DSML｜ parameter name=\"limit\">5\n\
                    </｜DSML｜ invoke>\n\
                    </｜DSML｜ calls>";
        let (content, calls) = parse_tool_calls(DSML, &tools, text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(args(&calls[0]), json!({"query": "42", "limit": 5}));
        assert_streams_like_parse(DSML, &tools, text);
    }

    #[test]
    fn dsml_invoke_without_a_block_and_prose_after() {
        let text = "<｜DSML｜invoke name=\"f\">\n</｜DSML｜invoke> Done.";
        let (content, calls) = parse(DSML, text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments, "{}");
        assert_eq!(content, "Done.");
    }
}
