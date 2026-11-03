//! Arguments of the formats that write one parameter per tag (Qwen3-Coder,
//! GLM, DeepSeek DSML, Kimi K3) rather than a JSON object.

use std::borrow::Cow;
use std::collections::BTreeSet;

use serde_json::Value;

use super::{before_partial, find_first, Cx, PartialCall};
use crate::types::Tool;

/// A call's arguments object, built one parameter at a time so the JSON text
/// only ever grows while the call streams.
#[derive(Debug, Clone)]
pub(crate) struct ArgsJson {
    text: String,
}

impl ArgsJson {
    pub fn new() -> Self {
        Self {
            text: "{".to_string(),
        }
    }

    fn key(text: &mut String, key: &str) {
        if text.len() > 1 {
            text.push(',');
        }
        text.push_str(&Value::from(key).to_string());
        text.push(':');
    }

    /// Append a finished parameter.
    pub fn push(&mut self, key: &str, value: &Value) {
        Self::key(&mut self.text, key);
        self.text.push_str(&value.to_string());
    }

    /// The text to stream: the finished parameters, plus `open` (a string
    /// parameter still being written) without its closing quote.
    pub fn streaming(&self, open: Option<(&str, &str)>) -> String {
        let mut text = self.text.clone();
        if let Some((key, partial)) = open {
            Self::key(&mut text, key);
            let quoted = Value::from(partial).to_string();
            text.push_str(&quoted[..quoted.len() - 1]);
        }
        text
    }

    /// The finished object.
    pub fn finish(&self) -> String {
        format!("{}}}", self.text)
    }
}

/// Whether the tool's schema types parameter `key` of `function` as a string
/// and nothing else (`type`, `enum`, `const`, `anyOf` / `oneOf` / `allOf`).
/// Such a value is kept verbatim, and can stream before it is finished.
pub(crate) fn string_param(tools: &[Tool], function: &str, key: &str) -> bool {
    let Some(schema) = tools
        .iter()
        .find(|tool| tool.function.name == function)
        .and_then(|tool| tool.function.parameters.as_ref())
        .and_then(|parameters| parameters.get("properties"))
        .and_then(|properties| properties.get(key))
    else {
        return false;
    };
    let mut types = BTreeSet::new();
    schema_types(schema, &mut types);
    !types.is_empty() && types.iter().all(|t| *t == "string")
}

fn schema_types<'a>(schema: &'a Value, types: &mut BTreeSet<&'a str>) {
    match schema.get("type") {
        Some(Value::String(t)) => {
            types.insert(t);
        }
        Some(Value::Array(list)) => types.extend(list.iter().filter_map(Value::as_str)),
        _ => {}
    }
    let literals = schema
        .get("enum")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .chain(schema.get("const"));
    for literal in literals {
        types.insert(match literal {
            Value::String(_) => "string",
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        });
    }
    for choice in ["anyOf", "oneOf", "allOf"] {
        for option in schema
            .get(choice)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            schema_types(option, types);
        }
    }
}

/// The value of a parameter the model wrote as text: the text itself when
/// the schema types it as a string, otherwise the JSON it spells, falling
/// back to the text.
pub(crate) fn convert(text: &str, string: bool) -> Value {
    if string {
        return Value::String(text.to_string());
    }
    serde_json::from_str(text.trim()).unwrap_or_else(|_| Value::String(text.to_string()))
}

/// The rest of a parameter's opening tag, after the string that opens it.
pub(crate) enum Header<'a> {
    /// The tag is still arriving.
    Partial,
    /// Not a parameter tag.
    Bad,
    /// A complete tag `len` bytes long.  `string` is what the tag says about
    /// the value's type (`None`: ask the tool's schema).
    Done {
        key: Cow<'a, str>,
        string: Option<bool>,
        len: usize,
    },
}

/// How a family writes one parameter per tag.
pub(crate) struct TaggedParams {
    /// Strings that end the call.
    pub call_end: &'static [&'static str],
    /// What may come next between values: the strings that open a
    /// parameter, and `call_end`.
    pub next: &'static [&'static str],
    /// Strings that end a value.
    pub value_end: &'static [&'static str],
    /// The members of `value_end` that close the value and belong to it;
    /// the others (the next parameter, the call's end) are read again.
    pub value_close: &'static [&'static str],
    /// Reads the rest of a parameter's tag after its opening string.
    pub header: fn(&str) -> Header<'_>,
    /// The part of a value's text that is the value.
    pub trim: fn(&str) -> &str,
}

/// The parameters of one call read so far.
pub(crate) struct Params<'a> {
    pub args: ArgsJson,
    /// A string parameter still being written: its key and text so far.
    pub open: Option<(Cow<'a, str>, &'a str)>,
    /// Bytes of the body up to and including the call's end, once read.
    pub end: Option<usize>,
}

impl TaggedParams {
    /// Read the parameters of `function` from `body`, the text after the
    /// call's opening tag.
    pub fn read<'a>(&self, body: &'a str, function: &str, cx: &Cx<'_>) -> Params<'a> {
        let mut params = Params {
            args: ArgsJson::new(),
            open: None,
            end: None,
        };
        let mut i = 0;
        while let Some((at, tag)) = find_first(&body[i..], self.next) {
            let tag_end = i + at + tag.len();
            if self.call_end.contains(&tag) {
                params.end = Some(tag_end);
                break;
            }
            let (key, string, value_start) = match (self.header)(&body[tag_end..]) {
                Header::Partial => break,
                Header::Bad => {
                    i = tag_end;
                    continue;
                }
                Header::Done { key, string, len } => (key, string, tag_end + len),
            };
            let string = string.unwrap_or_else(|| string_param(cx.tools, function, &key));
            let value = &body[value_start..];
            match find_first(value, self.value_end) {
                Some((len, end)) => {
                    if !key.is_empty() {
                        params
                            .args
                            .push(&key, &convert((self.trim)(&value[..len]), string));
                    }
                    i = value_start + len;
                    if self.value_close.contains(&end) {
                        i += end.len();
                    }
                }
                None if cx.finished => {
                    if !key.is_empty() {
                        params.args.push(&key, &convert((self.trim)(value), string));
                    }
                    break;
                }
                None => {
                    if string && !key.is_empty() {
                        let text = (self.trim)(before_partial(value, self.value_end));
                        params.open = Some((key, text));
                    }
                    break;
                }
            }
        }
        params
    }
}

impl Params<'_> {
    /// The call as far as it has been read: final once its end has been
    /// read or decoding has finished.
    pub fn call(&self, name: &str, cx: &Cx<'_>) -> PartialCall {
        let open = self.open.as_ref().map(|(key, text)| (key.as_ref(), *text));
        tagged_call(name, &self.args, open, self.end.is_some() || cx.finished)
    }
}

/// A call whose arguments were written one tagged parameter at a time.
pub(crate) fn tagged_call(
    name: &str,
    args: &ArgsJson,
    open: Option<(&str, &str)>,
    done: bool,
) -> PartialCall {
    if done {
        let text = args.finish();
        PartialCall {
            id: None,
            name: name.to_string(),
            done: Some(super::json::canonical(&text)),
            args: text,
        }
    } else {
        PartialCall {
            id: None,
            name: name.to_string(),
            args: args.streaming(open),
            done: None,
        }
    }
}

/// The value of an XML-style attribute `name="…"` in `attrs`, with the
/// `&quot;` / `&amp;` escapes Kimi K3's template writes undone.
pub(crate) fn attribute<'a>(attrs: &'a str, name: &str) -> Option<Cow<'a, str>> {
    let mut rest = attrs;
    loop {
        let at = rest.find(name)?;
        let before_ok = at == 0 || rest[..at].ends_with(char::is_whitespace);
        let after = &rest[at + name.len()..];
        if before_ok {
            if let Some(value) = after.strip_prefix("=\"") {
                let end = value.find('"')?;
                let value = &value[..end];
                return Some(if value.contains('&') {
                    Cow::Owned(value.replace("&quot;", "\"").replace("&amp;", "&"))
                } else {
                    Cow::Borrowed(value)
                });
            }
        }
        rest = after;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FunctionDef;

    fn tool(parameters: Value) -> Tool {
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "f".into(),
                description: None,
                parameters: Some(parameters),
            },
        }
    }

    #[test]
    fn streaming_text_is_a_prefix_of_the_finished_object() {
        let mut args = ArgsJson::new();
        args.push("n", &Value::from(5));
        let open = args.streaming(Some(("s", "line\none \"q")));
        args.push("s", &Value::from("line\none \"quoted\""));
        let done = args.finish();
        assert!(done.starts_with(&open), "{open} vs {done}");
        assert_eq!(done, r#"{"n":5,"s":"line\none \"quoted\""}"#);
        assert_eq!(ArgsJson::new().finish(), "{}");
    }

    #[test]
    fn schema_types_decide_string_parameters() {
        let tools = [tool(serde_json::json!({
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "count": {"type": "integer"},
                "mode": {"enum": ["a", "b"]},
                "either": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "untyped": {"description": "anything"}
            }
        }))];
        assert!(string_param(&tools, "f", "code"));
        assert!(string_param(&tools, "f", "mode"));
        assert!(!string_param(&tools, "f", "count"));
        assert!(!string_param(&tools, "f", "either"));
        assert!(!string_param(&tools, "f", "untyped"));
        assert!(!string_param(&tools, "f", "missing"));
        assert!(!string_param(&tools, "g", "code"));
        assert_eq!(convert("007", true), Value::from("007"));
        assert_eq!(convert(" 7 ", false), Value::from(7));
        assert_eq!(convert("rust async", false), Value::from("rust async"));
    }
}
