//! FunctionGemma (`google/functiongemma-270m-it`, a Gemma 3 fine-tune that
//! runs on the `gemma3` architecture): each call between
//! `<start_function_call>` and `<end_function_call>`, its arguments written
//! the chat template's way, strings between `<escape>` tokens and every
//! other value bare:
//!
//! ```text
//! <start_function_call>call:set_alarm{days:[<escape>mon<escape>,<escape>tue<escape>],enabled:true,hour:7,label:<escape>Wake up<escape>,snooze:{minutes:5}}<end_function_call>
//! ```
//!
//! Values are read as vLLM's `functiongemma` parser reads them: an escaped
//! value is the JSON it spells, or its text (always its text when the
//! tool's schema types it as a string); a bare one is `true` / `false` /
//! `null` / `None`, JSON, or its text.

use serde_json::{Map, Value};

use super::args::{self, tagged_call, ArgsJson};
use super::generic::{close_after, undecided};
use super::{Cx, Grammar, Marker, Scan, Stop, MAX_NESTING};

const START: &str = "<start_function_call>";
const END: &str = "<end_function_call>";
const CALL: &str = "call:";
const ESCAPE: &str = "<escape>";

/// FunctionGemma.
pub(super) struct FunctionGemma;

pub(super) static FUNCTION_GEMMA: FunctionGemma = FunctionGemma;

impl Grammar for FunctionGemma {
    fn markers(&self) -> &'static [Marker] {
        const MARKERS: &[Marker] = &[Marker::anywhere(START).closed_by(&[END])];
        MARKERS
    }

    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan {
        let head = &region[START.len()..];
        let Some(call) = head.strip_prefix(CALL) else {
            return if CALL.starts_with(head) {
                undecided(cx, START)
            } else {
                Scan::Text(START.len())
            };
        };
        // `call:name{`: the name is a run of word characters (vLLM's `\w+`).
        let name_len = call
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(call.len());
        let name = &call[..name_len];
        match call[name_len..].chars().next() {
            None => return undecided(cx, START),
            Some('{') if !name.is_empty() => {}
            Some(_) => return Scan::Text(START.len()),
        }
        let body_at = START.len() + CALL.len() + name_len + 1;
        let mut r = Reader {
            s: &region[body_at..],
            i: 0,
            finished: cx.finished,
            depth: 0,
        };
        let Ok((params, open, end)) = arguments(&mut r, name, cx) else {
            return Scan::Text(START.len());
        };
        let open = open
            .as_ref()
            .map(|(key, text)| (key.as_str(), text.as_str()));
        Scan::Calls {
            calls: vec![tagged_call(
                name,
                &params,
                open,
                end.is_some() || cx.finished,
            )],
            end: end.and_then(|end| close_after(region, body_at + end, END, &[START], cx.finished)),
        }
    }
}

/// A value as written: an escaped string, whose reading depends on the
/// schema at the top level, or any other value, already read.
enum Read {
    Escaped(String),
    Value(Value),
}

/// The arguments read so far, a string argument still being written (its
/// key and text, when the schema types it as a string) and, once the
/// closing `}` has arrived, the length of the body up to and including it.
type Arguments = (ArgsJson, Option<(String, String)>, Option<usize>);

/// The `key:value,…}` after a call's `{`.  `Err` when it is not an argument
/// list.
fn arguments(r: &mut Reader<'_>, name: &str, cx: &Cx<'_>) -> Result<Arguments, ()> {
    let mut params = ArgsJson::new();
    loop {
        r.skip(|b| matches!(b, b' ' | b'\n' | b','));
        match r.peek() {
            None => return Ok((params, None, None)),
            Some(b'}') => return Ok((params, None, Some(r.i + 1))),
            _ => {}
        }
        let key = match r.key() {
            Ok(key) => key,
            Err(Stop::Partial) => return Ok((params, None, None)),
            Err(Stop::Bad) => return Err(()),
        };
        let string = args::string_param(cx.tools, name, &key);
        let at = r.i;
        match r.value() {
            Ok(Read::Escaped(text)) => params.push(&key, &args::convert(&text, string)),
            Ok(Read::Value(value)) => params.push(&key, &value),
            Err(Stop::Partial) => {
                // A string argument streams while the model writes it.
                let open = r
                    .open_escape(at)
                    .filter(|_| string)
                    .map(|text| (key, text.to_string()));
                return Ok((params, open, None));
            }
            Err(Stop::Bad) => return Err(()),
        }
    }
}

/// A cursor over the arguments' text.
struct Reader<'a> {
    s: &'a str,
    i: usize,
    /// Decoding has ended: a bare value at the end of the text is complete.
    finished: bool,
    /// The objects and arrays the cursor is inside.
    depth: usize,
}

impl<'a> Reader<'a> {
    fn peek(&self) -> Option<u8> {
        self.s.as_bytes().get(self.i).copied()
    }

    fn skip(&mut self, skip: impl Fn(u8) -> bool) {
        while self.peek().is_some_and(&skip) {
            self.i += 1;
        }
    }

    /// A key and its `:`: text without the template's delimiters.
    fn key(&mut self) -> Result<String, Stop> {
        let rest = &self.s[self.i..];
        let Some(len) = rest.find([':', '{', '}', '[', ']', ',', '<']) else {
            return Err(Stop::Partial);
        };
        let key = rest[..len].trim();
        if !rest[len..].starts_with(':') || key.is_empty() {
            return Err(Stop::Bad);
        }
        self.i += len + 1;
        Ok(key.to_string())
    }

    fn value(&mut self) -> Result<Read, Stop> {
        self.skip(|b| matches!(b, b' ' | b'\n'));
        let rest = &self.s[self.i..];
        if let Some(text) = rest.strip_prefix(ESCAPE) {
            let len = text.find(ESCAPE).ok_or(Stop::Partial)?;
            self.i += ESCAPE.len() + len + ESCAPE.len();
            return Ok(Read::Escaped(text[..len].to_string()));
        }
        if ESCAPE.starts_with(rest) && !rest.is_empty() && !self.finished {
            return Err(Stop::Partial);
        }
        match self.peek() {
            None => Err(Stop::Partial),
            Some(b'{') => {
                self.i += 1;
                self.deeper(Self::object).map(Read::Value)
            }
            Some(b'[') => {
                self.i += 1;
                self.deeper(Self::array).map(Read::Value)
            }
            Some(b'<') => Err(Stop::Bad),
            Some(_) => {
                // A bare value runs to the next delimiter (or markup).
                let len = rest.find([',', '}', ']', '<']).unwrap_or(rest.len());
                if len == rest.len() && !self.finished {
                    // More digits or letters may follow.
                    return Err(Stop::Partial);
                }
                let raw = rest[..len].trim();
                if raw.is_empty() {
                    return Err(Stop::Bad);
                }
                self.i += len;
                Ok(Read::Value(match raw {
                    "None" => Value::Null,
                    _ => serde_json::from_str(raw).unwrap_or_else(|_| Value::from(raw)),
                }))
            }
        }
    }

    /// `read` an object or array one level deeper: not a value past
    /// [`MAX_NESTING`].
    fn deeper(
        &mut self,
        read: impl FnOnce(&mut Self) -> Result<Value, Stop>,
    ) -> Result<Value, Stop> {
        if self.depth == MAX_NESTING {
            return Err(Stop::Bad);
        }
        self.depth += 1;
        let value = read(self);
        self.depth -= 1;
        value
    }

    /// A nested value: an escaped string is the JSON it spells, or text.
    fn nested(&mut self) -> Result<Value, Stop> {
        Ok(match self.value()? {
            Read::Escaped(text) => args::convert(&text, false),
            Read::Value(value) => value,
        })
    }

    /// `{key:value,…}` after its `{`, with bare keys.
    fn object(&mut self) -> Result<Value, Stop> {
        let mut map = Map::new();
        loop {
            self.skip(|b| matches!(b, b' ' | b'\n' | b','));
            match self.peek() {
                None => return Err(Stop::Partial),
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Value::Object(map));
                }
                _ => {}
            }
            let key = self.key()?;
            let value = self.nested()?;
            map.insert(key, value);
        }
    }

    /// `[value,…]` after its `[`.
    fn array(&mut self) -> Result<Value, Stop> {
        let mut items = Vec::new();
        loop {
            self.skip(|b| matches!(b, b' ' | b'\n' | b','));
            match self.peek() {
                None => return Err(Stop::Partial),
                Some(b']') => {
                    self.i += 1;
                    return Ok(Value::Array(items));
                }
                Some(b'}') => return Err(Stop::Bad),
                _ => items.push(self.nested()?),
            }
        }
    }

    /// The text so far of the escaped string at `at` that the text ends
    /// inside, short of a partial closing `<escape>`.
    fn open_escape(&self, at: usize) -> Option<&'a str> {
        let rest = self.s[at..].trim_start_matches([' ', '\n']);
        let text = rest.strip_prefix(ESCAPE)?;
        Some(super::before_partial(text, &[ESCAPE]))
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{args, assert_streams_like_parse, parse};
    use super::super::{parse_tool_calls, ToolCallFormat};
    use crate::types::{FunctionDef, Tool};
    use serde_json::json;

    const FG: ToolCallFormat = ToolCallFormat::FunctionGemma;

    /// vLLM's `test_functiongemma_tool_parser.py` cases.
    #[test]
    fn calls_and_text_before_them() {
        let (content, calls) = parse(FG, "Hello, how can I help you today?");
        assert_eq!(content, "Hello, how can I help you today?");
        assert!(calls.is_empty());

        let text = "Let me check the weather for you. \
                    <start_function_call>call:get_weather{location:<escape>London<escape>}\
                    <end_function_call>\
                    <start_function_call>call:get_time{timezone:<escape>UTC<escape>}\
                    <end_function_call>";
        let (content, calls) = parse(FG, text);
        assert_eq!(content, "Let me check the weather for you.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(args(&calls[0]), json!({"location": "London"}));
        assert_eq!(calls[1].name, "get_time");
        assert_streams_like_parse(FG, &[], text);

        let text = "<start_function_call>call:get_weather{location:<escape>San Francisco<escape>,\
                    unit:<escape>celsius<escape>}<end_function_call>";
        let (_, calls) = parse(FG, text);
        assert_eq!(
            args(&calls[0]),
            json!({"location": "San Francisco", "unit": "celsius"})
        );
        assert_streams_like_parse(FG, &[], text);
    }

    /// The template's own notation: escaped strings, bare booleans,
    /// numbers, lists and objects with bare keys (vLLM's #63983 cases).
    #[test]
    fn every_argument_type_the_template_writes() {
        let text = "<start_function_call>call:set_alarm{\
                    days:[<escape>mon<escape>,<escape>tue<escape>],enabled:true,hour:7,\
                    label:<escape>Wake up<escape>,snooze:{minutes:5}}<end_function_call>";
        let (content, calls) = parse(FG, text);
        assert_eq!(content, "");
        assert_eq!(
            args(&calls[0]),
            json!({
                "days": ["mon", "tue"],
                "enabled": true,
                "hour": 7,
                "label": "Wake up",
                "snooze": {"minutes": 5}
            })
        );
        assert_streams_like_parse(FG, &[], text);
        // Escaped values that spell JSON are JSON; `None` is null.
        let (_, calls) = parse(
            FG,
            "<start_function_call>call:f{count:<escape>42<escape>,on:<escape>true<escape>,x:None}\
             <end_function_call>",
        );
        assert_eq!(args(&calls[0]), json!({"count": 42, "on": true, "x": null}));
    }

    /// With the tool's schema, a string parameter stays a string and streams
    /// while the model writes it.
    #[test]
    fn string_parameters_follow_the_schema_and_stream() {
        let tools = [Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "note".into(),
                description: None,
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "text": {"type": "string"}}
                })),
            },
        }];
        let text = "<start_function_call>call:note{id:<escape>007<escape>,\
                    text:<escape>Buy milk, then eggs.<escape>}<end_function_call>";
        let (_, calls) = parse_tool_calls(FG, &tools, text);
        assert_eq!(
            args(&calls[0]),
            json!({"id": "007", "text": "Buy milk, then eggs."})
        );
        assert_streams_like_parse(FG, &tools, text);
        let mut stream = super::super::ToolCallStream::new(FG, Some(&tools));
        let cut = text.find("eggs").unwrap();
        let streamed: String = stream
            .push(&text[..cut])
            .into_iter()
            .filter_map(|event| match event {
                super::super::ToolEvent::CallArgs { delta, .. } => Some(delta),
                _ => None,
            })
            .collect();
        assert!(
            streamed.ends_with(r#""text":"Buy milk, then "#),
            "{streamed}"
        );
    }

    #[test]
    fn markers_in_prose_without_a_call_stay_text() {
        for text in [
            "Calls start with <start_function_call> in this format.",
            "<start_function_call>call:bad name{x:1}<end_function_call>",
        ] {
            assert_eq!(parse(FG, text), (text.to_string(), Vec::new()), "{text}");
            assert_streams_like_parse(FG, &[], text);
        }
    }
}
