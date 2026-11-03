//! Calls written as a Python list of keyword-argument calls:
//!
//! - **Pythonic** (Llama 3.2 / Llama 4, vLLM's `pythonic` and
//!   `llama4_pythonic`): the whole response is the list, optionally between
//!   Llama 4's `<|python_start|>` and `<|python_end|>`:
//!   `[get_weather(city="Paris"), get_time()]`.
//! - **LFM2**: the list between `<|tool_call_start|>` and
//!   `<|tool_call_end|>`, anywhere in the response.
//!
//! Values are Python literals (strings, numbers, `True` / `False` / `None`,
//! lists, tuples, dicts), plus the JSON spellings `true` / `false` / `null`
//! that LFM2 writes; they become the JSON of `ast.literal_eval` +
//! `json.dumps`, as in vLLM.

use serde_json::{Map, Number, Value};

use super::args::{tagged_call, ArgsJson};
use super::generic::{close_after, undecided};
use super::json::skip_ws;
use super::{Cx, Grammar, Marker, PartialCall, Scan};

const PYTHON_START: &str = "<|python_start|>";
const PYTHON_END: &str = "<|python_end|>";
const LFM2_START: &str = "<|tool_call_start|>";
const LFM2_END: &str = "<|tool_call_end|>";

/// Llama 3.2 / Llama 4 pythonic calls.
pub(super) struct Pythonic;

pub(super) static PYTHONIC: Pythonic = Pythonic;

impl Grammar for Pythonic {
    fn markers(&self) -> &'static [Marker] {
        const MARKERS: &[Marker] = &[Marker::anywhere(PYTHON_START), Marker::leading("[")];
        MARKERS
    }

    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan {
        if region.starts_with(PYTHON_START) {
            return marked_list(region, PYTHON_START, PYTHON_END, cx);
        }
        // A response that opens with `[`: a call list or ordinary text.
        let start = skip_ws(region, 0);
        match call_list(&region[start..], cx) {
            Ok((calls, end)) if !calls.is_empty() => Scan::Calls {
                calls,
                end: end.map(|end| start + end),
            },
            Ok(_) if !cx.finished => Scan::Pending,
            _ => Scan::Text(region.len()),
        }
    }
}

/// LFM2's tagged pythonic calls.
pub(super) struct Lfm2;

pub(super) static LFM2: Lfm2 = Lfm2;

impl Grammar for Lfm2 {
    fn markers(&self) -> &'static [Marker] {
        const MARKERS: &[Marker] = &[Marker::anywhere(LFM2_START)];
        MARKERS
    }

    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan {
        marked_list(region, LFM2_START, LFM2_END, cx)
    }
}

/// A call list between `open` and `close`.
fn marked_list(region: &str, open: &str, close: &str, cx: &Cx<'_>) -> Scan {
    let start = skip_ws(region, open.len());
    if start >= region.len() {
        return undecided(cx, open);
    }
    match call_list(&region[start..], cx) {
        Err(()) => Scan::Text(open.len()),
        Ok((calls, end)) => Scan::Calls {
            calls,
            end: end.and_then(|end| close_after(region, start + end, close, cx.finished)),
        },
    }
}

/// Why a value could not be read.
enum Stop {
    /// The text ends inside it.
    Partial,
    /// It is not a literal.
    Bad,
}

type Read<T> = Result<T, Stop>;

/// A call list at the start of `s`, as far as it has been generated: the
/// calls with a complete name, and the list's length once it has closed.
/// `Err` when the text is not a call list.
fn call_list(s: &str, cx: &Cx<'_>) -> Result<(Vec<PartialCall>, Option<usize>), ()> {
    let mut r = Reader { s, i: 0 };
    match r.peek() {
        None => return Ok((Vec::new(), None)),
        Some(b'[') => r.i += 1,
        Some(_) => return Err(()),
    }
    let mut calls = Vec::new();
    loop {
        r.ws();
        match r.peek() {
            None => return Ok((calls, None)),
            Some(b']') if !calls.is_empty() => return Ok((calls, Some(r.i + 1))),
            Some(b',') if !calls.is_empty() => {
                r.i += 1;
                r.ws();
            }
            _ if !calls.is_empty() => return Err(()),
            _ => {}
        }
        let name_start = r.i;
        while r
            .peek()
            .is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.')
        {
            r.i += 1;
        }
        let name = &s[name_start..r.i];
        let Some(next) = r.peek() else {
            return Ok((calls, None));
        };
        if next != b'(' || !is_dotted_name(name) {
            return Err(());
        }
        r.i += 1;
        match read_arguments(&mut r, name, cx) {
            Ok(call) => calls.push(call),
            Err(Stop::Partial) => return Ok((calls, None)),
            Err(Stop::Bad) => return Err(()),
        }
        // A partial call ends the list read.
        if calls.last().is_some_and(|call| call.done.is_none()) {
            return Ok((calls, None));
        }
    }
}

/// `a=1, b="x")` after a call's `(`.  A call cut off by the end of the text
/// is returned unfinished, with a string argument still being written.
fn read_arguments(r: &mut Reader<'_>, name: &str, cx: &Cx<'_>) -> Read<PartialCall> {
    let mut args = ArgsJson::new();
    let mut first = true;
    let partial =
        |args: &ArgsJson, open: Option<(&str, &str)>| tagged_call(name, args, open, cx.finished);
    loop {
        r.ws();
        match r.peek() {
            None => return Ok(partial(&args, None)),
            Some(b')') => {
                r.i += 1;
                return Ok(tagged_call(name, &args, None, true));
            }
            Some(b',') if !first => {
                r.i += 1;
                r.ws();
                if r.peek() == Some(b')') {
                    continue;
                }
            }
            _ if !first => return Err(Stop::Bad),
            _ => {}
        }
        first = false;
        let key_start = r.i;
        while r
            .peek()
            .is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            r.i += 1;
        }
        let key = &r.s[key_start..r.i];
        r.ws();
        match r.peek() {
            None => return Ok(partial(&args, None)),
            Some(b'=') if is_dotted_name(key) && !key.contains('.') => r.i += 1,
            _ => return Err(Stop::Bad),
        }
        r.ws();
        let value_start = r.i;
        match r.value() {
            Ok(value) => args.push(key, &value),
            Err(Stop::Partial) => {
                let open = string_prefix(&r.s[value_start..]);
                return Ok(partial(&args, open.as_deref().map(|text| (key, text))));
            }
            Err(Stop::Bad) => return Err(Stop::Bad),
        }
    }
}

/// A Python identifier, or several joined by dots (`Calendar.create_event`).
fn is_dotted_name(name: &str) -> bool {
    !name.is_empty()
        && name.split('.').all(|part| {
            part.bytes()
                .next()
                .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        })
}

/// The decoded text so far of a string literal that has not closed.
fn string_prefix(s: &str) -> Option<String> {
    let mut r = Reader { s, i: 0 };
    let quote = r.peek().filter(|b| matches!(b, b'"' | b'\''))?;
    r.i += 1;
    let mut out = String::new();
    while let Some(b) = r.peek() {
        if b == quote {
            return None;
        }
        match r.char_in_string(&mut out) {
            Ok(()) => {}
            Err(_) => break,
        }
    }
    Some(out)
}

struct Reader<'a> {
    s: &'a str,
    i: usize,
}

impl Reader<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.as_bytes().get(self.i).copied()
    }

    fn ws(&mut self) {
        self.i = skip_ws(self.s, self.i);
    }

    /// One Python literal.
    fn value(&mut self) -> Read<Value> {
        match self.peek().ok_or(Stop::Partial)? {
            b'"' | b'\'' => self.string().map(Value::String),
            b'[' => self.sequence(b']').map(Value::Array),
            b'(' => self.sequence(b')').map(Value::Array),
            b'{' => self.dict(),
            b'-' | b'+' | b'0'..=b'9' | b'.' => self.number(),
            _ => self.word(),
        }
    }

    fn string(&mut self) -> Read<String> {
        let quote = self.s.as_bytes()[self.i];
        self.i += 1;
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return Err(Stop::Partial),
                Some(b) if b == quote => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(_) => self.char_in_string(&mut out)?,
            }
        }
    }

    /// One character (or escape) of a string literal into `out`.
    fn char_in_string(&mut self, out: &mut String) -> Read<()> {
        let rest = &self.s[self.i..];
        let mut chars = rest.chars();
        let c = chars.next().ok_or(Stop::Partial)?;
        if c != '\\' {
            out.push(c);
            self.i += c.len_utf8();
            return Ok(());
        }
        let escape = chars.next().ok_or(Stop::Partial)?;
        let (decoded, len) = match escape {
            'n' => ('\n', 2),
            't' => ('\t', 2),
            'r' => ('\r', 2),
            '0' => ('\0', 2),
            '\\' | '\'' | '"' => (escape, 2),
            'x' | 'u' | 'U' => {
                let digits = match escape {
                    'x' => 2,
                    'u' => 4,
                    _ => 8,
                };
                let hex = rest.get(2..2 + digits).ok_or(Stop::Partial)?;
                let code = u32::from_str_radix(hex, 16).map_err(|_| Stop::Bad)?;
                (char::from_u32(code).ok_or(Stop::Bad)?, 2 + digits)
            }
            // Python keeps an unknown escape as written.
            other => {
                out.push('\\');
                (other, 1 + other.len_utf8())
            }
        };
        out.push(decoded);
        self.i += len;
        Ok(())
    }

    /// The items of a list or tuple, up to `close`.
    fn sequence(&mut self, close: u8) -> Read<Vec<Value>> {
        self.i += 1;
        let mut items = Vec::new();
        loop {
            self.ws();
            match self.peek().ok_or(Stop::Partial)? {
                b if b == close => {
                    self.i += 1;
                    return Ok(items);
                }
                b',' if !items.is_empty() => {
                    self.i += 1;
                    continue;
                }
                _ => {}
            }
            items.push(self.value()?);
            self.ws();
            match self.peek().ok_or(Stop::Partial)? {
                b',' => self.i += 1,
                b if b == close => {}
                _ => return Err(Stop::Bad),
            }
        }
    }

    fn dict(&mut self) -> Read<Value> {
        self.i += 1;
        let mut map = Map::new();
        loop {
            self.ws();
            match self.peek().ok_or(Stop::Partial)? {
                b'}' => {
                    self.i += 1;
                    return Ok(Value::Object(map));
                }
                b',' if !map.is_empty() => {
                    self.i += 1;
                    continue;
                }
                _ => {}
            }
            let key = match self.value()? {
                Value::String(key) => key,
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => if b { "True" } else { "False" }.to_string(),
                _ => return Err(Stop::Bad),
            };
            self.ws();
            if self.peek().ok_or(Stop::Partial)? != b':' {
                return Err(Stop::Bad);
            }
            self.i += 1;
            self.ws();
            let value = self.value()?;
            map.insert(key, value);
            self.ws();
            match self.peek().ok_or(Stop::Partial)? {
                b',' => self.i += 1,
                b'}' => {}
                _ => return Err(Stop::Bad),
            }
        }
    }

    fn number(&mut self) -> Read<Value> {
        let start = self.i;
        let b = self.s.as_bytes();
        let mut end = start;
        if matches!(b.get(end), Some(b'-' | b'+')) {
            end += 1;
        }
        loop {
            match b.get(end) {
                Some(c) if c.is_ascii_digit() || matches!(c, b'.' | b'_') => end += 1,
                Some(b'e' | b'E') => {
                    end += 1;
                    if matches!(b.get(end), Some(b'-' | b'+')) {
                        end += 1;
                    }
                }
                _ => break,
            }
        }
        if end >= b.len() {
            // More digits may follow.
            return Err(Stop::Partial);
        }
        let text = self.s[start..end].replace('_', "");
        let text = text.strip_prefix('+').unwrap_or(&text);
        let value = if let Ok(n) = text.parse::<i64>() {
            Value::Number(n.into())
        } else {
            let f: f64 = text.parse().map_err(|_| Stop::Bad)?;
            Value::Number(Number::from_f64(f).ok_or(Stop::Bad)?)
        };
        self.i = end;
        Ok(value)
    }

    /// `True` / `False` / `None` and their JSON spellings.
    fn word(&mut self) -> Read<Value> {
        let rest = &self.s[self.i..];
        let len = rest
            .bytes()
            .position(|b| !b.is_ascii_alphanumeric() && b != b'_')
            .unwrap_or(rest.len());
        let word = &rest[..len];
        let value = match word {
            "True" | "true" => Value::Bool(true),
            "False" | "false" => Value::Bool(false),
            "None" | "null" => Value::Null,
            _ if len == rest.len()
                && ["True", "False", "None", "true", "false", "null"]
                    .iter()
                    .any(|w| w.starts_with(word)) =>
            {
                return Err(Stop::Partial);
            }
            _ => return Err(Stop::Bad),
        };
        if len == rest.len() {
            // `None` could still become `Nonexistent`.
            return Err(Stop::Partial);
        }
        self.i += len;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{args, assert_streams_like_parse, parse};
    use super::super::{ToolCallFormat, ToolCallStream, ToolEvent};
    use serde_json::json;

    const PY: ToolCallFormat = ToolCallFormat::Pythonic;
    const LFM2: ToolCallFormat = ToolCallFormat::Lfm2;

    /// vLLM's pythonic parser cases.
    #[test]
    fn pythonic_calls() {
        let text = "[get_weather(city='San Francisco', metric='celsius'), \
                    register_user(name='John Doe', age=37, address={'city': 'San Francisco', 'state': 'CA'}, \
                    role=None, passed_test=True, aliases=['John', 'Johnny'], scores=(1.5, -2))]";
        let (content, calls) = parse(PY, text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            args(&calls[0]),
            json!({"city": "San Francisco", "metric": "celsius"})
        );
        assert_eq!(
            args(&calls[1]),
            json!({"name": "John Doe", "age": 37, "address": {"city": "San Francisco", "state": "CA"},
                   "role": null, "passed_test": true, "aliases": ["John", "Johnny"], "scores": [1.5, -2]})
        );
        assert_streams_like_parse(PY, &[], text);
        let (_, calls) = parse(PY, "[do_something_cool()]");
        assert_eq!(calls[0].arguments, "{}");
    }

    /// Llama 4 wraps the list in `<|python_start|>` / `<|python_end|>`.
    #[test]
    fn llama4_python_tags() {
        let text = "<|python_start|>[get_weather(city=\"LA\", metric=\"F\")]<|python_end|>";
        let (content, calls) = parse(PY, text);
        assert_eq!(content, "");
        assert_eq!(args(&calls[0]), json!({"city": "LA", "metric": "F"}));
        assert_streams_like_parse(PY, &[], text);
    }

    #[test]
    fn pythonic_prose_is_not_a_call() {
        for text in [
            "Use this format: [link text](url).",
            "[1] First point.",
            "[Wikipedia](https://www.wikipedia.org) says so.",
            "No calls here.",
        ] {
            let (content, calls) = parse(PY, text);
            assert_eq!(content, text);
            assert!(calls.is_empty(), "{text}");
        }
    }

    /// llama.cpp's LFM2 cases.
    #[test]
    fn lfm2_calls() {
        for (text, name, expected) in [
            ("<|tool_call_start|>[special_function(arg1=1)]<|tool_call_end|>", "special_function", json!({"arg1": 1})),
            ("<|tool_call_start|>[toggle(enabled=True)]<|tool_call_end|>", "toggle", json!({"enabled": true})),
            ("<|tool_call_start|>[set_nullable(value=None)]<|tool_call_end|>", "set_nullable", json!({"value": null})),
            (
                "<|tool_call_start|>[set_config(config={\"enabled\": true, \"note\": null})]<|tool_call_end|>",
                "set_config",
                json!({"config": {"enabled": true, "note": null}}),
            ),
            (
                "<|tool_call_start|>[Calendar.create_event(title=\"demo\", participants=[\"Alice\", \"Bob\"], \
                 metadata={\"priority\": \"high\", \"reminder\": true})]<|tool_call_end|>",
                "Calendar.create_event",
                json!({"title": "demo", "participants": ["Alice", "Bob"],
                       "metadata": {"priority": "high", "reminder": true}}),
            ),
            (
                "<|tool_call_start|>[python(code=\"def hello():\\n    print('hey')\")]<|tool_call_end|>",
                "python",
                json!({"code": "def hello():\n    print('hey')"}),
            ),
            (
                "<|tool_call_start|>[python(code=\"print(\\\"hi\\\")\")]<|tool_call_end|>",
                "python",
                json!({"code": "print(\"hi\")"}),
            ),
        ] {
            let (content, calls) = parse(LFM2, text);
            assert_eq!(content, "", "{text}");
            assert_eq!(calls.len(), 1, "{text}");
            assert_eq!(calls[0].name, name);
            assert_eq!(args(&calls[0]), expected, "{text}");
            assert_streams_like_parse(LFM2, &[], text);
        }
        let text = "Let me check the time.<|tool_call_start|>[get_time(city=\"Paris\"), \
                    get_weather(city=\"Paris\")]<|tool_call_end|>";
        let (content, calls) = parse(LFM2, text);
        assert_eq!(content, "Let me check the time.");
        assert_eq!(calls.len(), 2);
        assert_streams_like_parse(LFM2, &[], text);
        let text = "Use this format: [link text](url).";
        assert_eq!(parse(LFM2, text).0, text);
    }

    /// A long string argument streams as it is written, decoded.
    #[test]
    fn string_arguments_stream() {
        let mut stream = ToolCallStream::new(LFM2, Some(&[]));
        let mut streamed = String::new();
        for piece in [
            "<|tool_call_start|>[python(",
            "code=\"line one\\n",
            "line two",
        ] {
            for event in stream.push(piece) {
                if let ToolEvent::CallArgs { delta, .. } = event {
                    streamed.push_str(&delta);
                }
            }
        }
        assert_eq!(streamed, r#"{"code":"line one\nline two"#);
    }
}
