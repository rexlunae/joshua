//! GLM-4.5 and later (4.6, 4.7, GLM-5): the function name, then one
//! `<arg_key>` / `<arg_value>` pair per argument, inside `<tool_call>`.
//! GLM-4.5 and 4.6 put newlines between the parts; 4.7 does not:
//!
//! ```text
//! <tool_call>get_weather
//! <arg_key>city</arg_key>
//! <arg_value>Paris</arg_value>
//! </tool_call>
//! <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>
//! ```
//!
//! A value the tool's schema types as a string is kept as text; any other
//! value is read as JSON when it is JSON (the template writes non-string
//! values with `tojson`).

use super::args::{Header, TaggedParams};
use super::generic::hermes;
use super::json::skip_ws;
use super::{find_first, Cx, Grammar, Marker, Scan};

const TOOL_CALL: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const ARG_KEY: &str = "<arg_key>";
const ARG_KEY_END: &str = "</arg_key>";
const ARG_VALUE: &str = "<arg_value>";
const ARG_VALUE_END: &str = "</arg_value>";

const GLM_PARAMS: TaggedParams = TaggedParams {
    call_end: &[TOOL_CALL_END],
    next: &[ARG_KEY, TOOL_CALL_END],
    value_end: &[ARG_VALUE_END, TOOL_CALL_END],
    value_close: &[ARG_VALUE_END],
    header: |s| {
        let Some(key_len) = s.find(ARG_KEY_END) else {
            return Header::Partial;
        };
        let value = skip_ws(s, key_len + ARG_KEY_END.len());
        let rest = &s[value..];
        if rest.starts_with(ARG_VALUE) {
            Header::Done {
                key: s[..key_len].trim().into(),
                string: None,
                len: value + ARG_VALUE.len(),
            }
        } else if ARG_VALUE.starts_with(rest) {
            Header::Partial
        } else {
            Header::Bad
        }
    },
    trim: |value| value,
};

/// GLM-4.5 / 4.6 / 4.7 / 5.
pub(super) struct Glm45;

pub(super) static GLM45: Glm45 = Glm45;

impl Grammar for Glm45 {
    fn markers(&self) -> &'static [Marker] {
        const MARKERS: &[Marker] = &[Marker::anywhere(TOOL_CALL)];
        MARKERS
    }

    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan {
        let payload = &region[TOOL_CALL.len()..];
        if payload.trim_start().starts_with('{') {
            // A Hermes JSON call, which GLM sometimes writes instead.
            return hermes(region, cx);
        }
        let start = skip_ws(payload, 0);
        let name_end = find_first(&payload[start..], &[ARG_KEY, TOOL_CALL_END, "\n"])
            .map(|(at, _)| start + at);
        let name = payload[start..name_end.unwrap_or(payload.len())].trim();
        if name.contains(char::is_whitespace) {
            return Scan::Text(TOOL_CALL.len());
        }
        match name_end {
            None if !cx.finished => return Scan::Pending,
            _ if name.is_empty() => return Scan::Text(TOOL_CALL.len()),
            _ => {}
        }
        let body_at = TOOL_CALL.len() + name_end.unwrap_or(payload.len());
        let params = GLM_PARAMS.read(&region[body_at..], name, cx);
        Scan::Calls {
            calls: vec![params.call(name, cx)],
            end: params.end.map(|end| body_at + end),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{args, assert_streams_like_parse, parse};
    use super::super::{parse_tool_calls, ToolCallFormat};
    use crate::types::{FunctionDef, Tool};
    use serde_json::json;

    const GLM: ToolCallFormat = ToolCallFormat::Glm45;

    /// llama.cpp's GLM-4.6 case (newlines between the parts).
    #[test]
    fn glm46_call_with_newlines() {
        let text = "<tool_call>special_function\n\
                    <arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n\
                    </tool_call>";
        let (content, calls) = parse(GLM, text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "special_function");
        assert_eq!(calls[0].arguments, r#"{"arg1":1}"#);
        assert_streams_like_parse(GLM, &[], text);
    }

    /// llama.cpp's GLM-4.7-Flash and Spark cases: compact calls, several in
    /// a row, typed values, and a call without arguments.
    #[test]
    fn glm47_compact_calls_with_typed_values() {
        let text = "Preparing updates.\
                    <tool_call>magic_int<arg_key>ref</arg_key><arg_value>42</arg_value>\
                    <arg_key>name</arg_key><arg_value>上海</arg_value></tool_call>\
                    <tool_call>toggle<arg_key>enabled</arg_key><arg_value>true</arg_value></tool_call>\
                    <tool_call>set_config<arg_key>config</arg_key>\
                    <arg_value>{\"source\": \"spark\", \"options\": {\"strict\": true}}</arg_value></tool_call>\
                    <tool_call>empty_args</tool_call>";
        let (content, calls) = parse(GLM, text);
        assert_eq!(content, "Preparing updates.");
        assert_eq!(calls.len(), 4);
        assert_eq!(args(&calls[0]), json!({"ref": 42, "name": "上海"}));
        assert_eq!(args(&calls[1]), json!({"enabled": true}));
        assert_eq!(
            args(&calls[2]),
            json!({"config": {"source": "spark", "options": {"strict": true}}})
        );
        assert_eq!(calls[3].name, "empty_args");
        assert_eq!(calls[3].arguments, "{}");
        assert_streams_like_parse(GLM, &[], text);
    }

    /// With the schema, a string value stays text even when it spells JSON,
    /// and streams while it is written.
    #[test]
    fn glm_string_values_follow_the_schema() {
        let tools = [Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "lookup".into(),
                description: None,
                parameters: Some(json!({"type": "object", "properties": {
                    "id": {"type": "string"}, "count": {"type": "integer"}
                }})),
            },
        }];
        let text = "<tool_call>lookup<arg_key>id</arg_key><arg_value>0042</arg_value>\
                    <arg_key>count</arg_key><arg_value>3</arg_value></tool_call>";
        let (_, calls) = parse_tool_calls(GLM, &tools, text);
        assert_eq!(args(&calls[0]), json!({"id": "0042", "count": 3}));
        assert_streams_like_parse(GLM, &tools, text);
    }

    #[test]
    fn glm_tool_call_tags_in_prose_stay_text() {
        let text = "Wrap calls in <tool_call> tags like this.";
        let (content, calls) = parse(GLM, text);
        assert_eq!(content, text);
        assert!(calls.is_empty());
    }

    #[test]
    fn glm_accepts_hermes_json_calls() {
        let (_, calls) = parse(
            GLM,
            "<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": 1}}</tool_call>",
        );
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments, r#"{"a":1}"#);
    }
}
