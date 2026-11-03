//! Incremental JSON scanning for tool calls that are still being generated.
//!
//! A streamed call is read many times as it grows, so these scanners never
//! allocate a parsed tree: they report how far a value runs, whether it is
//! complete, or that the text cannot be JSON at all.  Complete values are
//! handed to `serde_json` for the final result, which keeps the strictness
//! (and the output) of a plain `serde_json::from_str`.

use serde_json::Value;

use super::PartialCall;

/// How far one JSON value at the start of a text runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Json {
    /// The value ends after this many bytes.
    Done(usize),
    /// The text ends inside the value.
    Partial,
    /// The text cannot start a JSON value.
    Bad,
}

/// Index of the first non-whitespace byte at or after `i`.
pub(crate) fn skip_ws(s: &str, mut i: usize) -> usize {
    let b = s.as_bytes();
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Want {
    Value,
    ValueOrClose,
    KeyOrClose,
    Key,
    Colon,
    After,
}

/// Scan the JSON value that starts exactly at the beginning of `s`.
///
/// A number running to the end of the text is `Partial`: more digits may
/// follow.  Raw control characters inside strings are `Bad`, as they are to
/// `serde_json`.
pub(crate) fn scan_value(s: &str) -> Json {
    let b = s.as_bytes();
    let mut stack: Vec<u8> = Vec::new();
    let mut want = Want::Value;
    let mut i = 0;
    loop {
        if want != Want::Value || !stack.is_empty() {
            i = skip_ws(s, i);
        }
        if i >= b.len() {
            return Json::Partial;
        }
        match want {
            Want::Value | Want::ValueOrClose => {
                if want == Want::ValueOrClose && b[i] == b']' {
                    stack.pop();
                    i += 1;
                    want = Want::After;
                } else {
                    match b[i] {
                        b'{' => {
                            stack.push(b'{');
                            i += 1;
                            want = Want::KeyOrClose;
                            continue;
                        }
                        b'[' => {
                            stack.push(b'[');
                            i += 1;
                            want = Want::ValueOrClose;
                            continue;
                        }
                        b'"' => match scan_string(b, i) {
                            Json::Done(end) => i = end,
                            other => return other,
                        },
                        b'-' | b'0'..=b'9' => match scan_number(b, i) {
                            Json::Done(end) => i = end,
                            other => return other,
                        },
                        b't' | b'f' | b'n' => {
                            let word: &[u8] = match b[i] {
                                b't' => b"true",
                                b'f' => b"false",
                                _ => b"null",
                            };
                            let have = &b[i..b.len().min(i + word.len())];
                            if !word.starts_with(have) {
                                return Json::Bad;
                            }
                            if have.len() < word.len() {
                                return Json::Partial;
                            }
                            i += word.len();
                        }
                        _ => return Json::Bad,
                    }
                    want = Want::After;
                }
            }
            Want::KeyOrClose | Want::Key => {
                if want == Want::KeyOrClose && b[i] == b'}' {
                    stack.pop();
                    i += 1;
                    want = Want::After;
                } else if b[i] == b'"' {
                    match scan_string(b, i) {
                        Json::Done(end) => i = end,
                        other => return other,
                    }
                    want = Want::Colon;
                    continue;
                } else {
                    return Json::Bad;
                }
            }
            Want::Colon => {
                if b[i] != b':' {
                    return Json::Bad;
                }
                i += 1;
                want = Want::Value;
                continue;
            }
            Want::After => match (stack.last(), b[i]) {
                (Some(b'{'), b',') => {
                    i += 1;
                    want = Want::Key;
                    continue;
                }
                (Some(b'['), b',') => {
                    i += 1;
                    want = Want::Value;
                    continue;
                }
                (Some(b'{'), b'}') | (Some(b'['), b']') => {
                    stack.pop();
                    i += 1;
                }
                _ => return Json::Bad,
            },
        }
        if want == Want::After && stack.is_empty() {
            return Json::Done(i);
        }
    }
}

/// Scan a string literal whose opening quote is at `start`.
fn scan_string(b: &[u8], start: usize) -> Json {
    let mut i = start + 1;
    while i < b.len() {
        match b[i] {
            b'"' => return Json::Done(i + 1),
            b'\\' => {
                let Some(&escape) = b.get(i + 1) else {
                    return Json::Partial;
                };
                match escape {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => i += 2,
                    b'u' => {
                        let hex = &b[i + 2..b.len().min(i + 6)];
                        if !hex.iter().all(u8::is_ascii_hexdigit) {
                            return Json::Bad;
                        }
                        if hex.len() < 4 {
                            return Json::Partial;
                        }
                        i += 6;
                    }
                    _ => return Json::Bad,
                }
            }
            0..=0x1f => return Json::Bad,
            _ => i += 1,
        }
    }
    Json::Partial
}

/// Scan a number starting at `start`: `-? (0 | [1-9][0-9]*) (.[0-9]+)? ([eE][+-]?[0-9]+)?`.
fn scan_number(b: &[u8], start: usize) -> Json {
    let digits = |mut i: usize| {
        let from = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        (i, i - from)
    };
    let mut i = start;
    if b[i] == b'-' {
        i += 1;
    }
    if i >= b.len() {
        return Json::Partial;
    }
    match b[i] {
        b'0' => i += 1,
        b'1'..=b'9' => i = digits(i).0,
        _ => return Json::Bad,
    }
    if i < b.len() && b[i] == b'.' {
        let (end, n) = digits(i + 1);
        if n == 0 {
            return if end >= b.len() {
                Json::Partial
            } else {
                Json::Bad
            };
        }
        i = end;
    }
    if i < b.len() && matches!(b[i], b'e' | b'E') {
        i += 1;
        if i < b.len() && matches!(b[i], b'+' | b'-') {
            i += 1;
        }
        let (end, n) = digits(i);
        if n == 0 {
            return if end >= b.len() {
                Json::Partial
            } else {
                Json::Bad
            };
        }
        i = end;
    }
    if i >= b.len() {
        // More digits may still arrive.
        return Json::Partial;
    }
    Json::Done(i)
}

/// A `{"name": …, "arguments": …}` object as far as it has been generated.
#[derive(Debug, Default)]
pub(crate) struct CallObject<'a> {
    /// The function name, once its string is complete.
    pub name: Option<String>,
    /// The raw arguments value and whether it is complete.
    pub args: Option<(&'a str, bool)>,
    /// Bytes up to and including the closing brace, once it has arrived.
    pub end: Option<usize>,
}

impl CallObject<'_> {
    /// The arguments to stream so far: an object or array as the model is
    /// writing it, any other value only once it is complete (a string must
    /// be decoded first).
    pub fn stream_args(&self) -> String {
        match self.args {
            Some((raw, _)) if raw.starts_with(['{', '[']) => raw.to_string(),
            Some((_, true)) => self.final_args(),
            None if self.end.is_some() => self.final_args(),
            _ => String::new(),
        }
    }

    /// The finished arguments: OpenAI's `function.arguments` string.  An
    /// object is re-serialised compactly; a string-encoded object is passed
    /// through decoded; no arguments at all is `{}`.
    pub fn final_args(&self) -> String {
        match self.args {
            Some((raw, true)) => match serde_json::from_str::<Value>(raw) {
                Ok(Value::String(s)) => s,
                Ok(other) => other.to_string(),
                Err(_) => raw.to_string(),
            },
            Some((raw, false)) => raw.to_string(),
            None => "{}".to_string(),
        }
    }

    /// The call for the stream: final once the object has closed or
    /// decoding has `finished`.
    pub fn partial(&self, finished: bool) -> PartialCall {
        PartialCall {
            id: None,
            name: self.name.clone().unwrap_or_default(),
            args: self.stream_args(),
            done: (self.end.is_some() || finished).then(|| self.final_args()),
        }
    }
}

/// Read a call object at the start of `s` (leading whitespace allowed).
///
/// `Err` when the text cannot be such an object (it is not a JSON object, or
/// it closed without a string `name`).  The name may come from any of
/// `name_keys` and the arguments from any of `arg_keys`, earlier keys
/// winning (`arguments` over `parameters`).
pub(crate) fn call_object<'a>(
    s: &'a str,
    name_keys: &[&str],
    arg_keys: &[&str],
) -> Result<CallObject<'a>, ()> {
    let b = s.as_bytes();
    let mut out = CallObject::default();
    let mut args_rank = usize::MAX;
    let mut i = skip_ws(s, 0);
    if i >= b.len() {
        return Ok(out);
    }
    if b[i] != b'{' {
        return Err(());
    }
    i += 1;
    let mut first = true;
    loop {
        i = skip_ws(s, i);
        if i >= b.len() {
            return Ok(out);
        }
        if b[i] == b'}' && first {
            // `{}`: no name.
            return Err(());
        }
        if !first {
            match b[i] {
                b',' => i = skip_ws(s, i + 1),
                b'}' => {
                    if out.name.is_none() {
                        return Err(());
                    }
                    out.end = Some(i + 1);
                    return Ok(out);
                }
                _ => return Err(()),
            }
            if i >= b.len() {
                return Ok(out);
            }
        }
        first = false;
        if b[i] != b'"' {
            return Err(());
        }
        let key_end = match scan_string(b, i) {
            Json::Done(end) => end,
            Json::Partial => return Ok(out),
            Json::Bad => return Err(()),
        };
        let key: String = serde_json::from_str(&s[i..key_end]).map_err(|_| ())?;
        i = skip_ws(s, key_end);
        if i >= b.len() {
            return Ok(out);
        }
        if b[i] != b':' {
            return Err(());
        }
        i = skip_ws(s, i + 1);
        if i >= b.len() {
            return Ok(out);
        }
        let value = &s[i..];
        let (len, done) = match scan_value(value) {
            Json::Done(len) => (len, true),
            Json::Partial => (value.len(), false),
            Json::Bad => return Err(()),
        };
        if name_keys.contains(&key.as_str()) {
            if done {
                match serde_json::from_str::<Value>(&value[..len]) {
                    Ok(Value::String(name)) => out.name = Some(name),
                    _ => return Err(()),
                }
            } else if !value.starts_with('"') {
                return Err(());
            }
        } else if let Some(rank) = arg_keys.iter().position(|k| *k == key) {
            if rank <= args_rank {
                args_rank = rank;
                out.args = Some((&value[..len], done));
            }
        }
        if !done {
            return Ok(out);
        }
        i += len;
    }
}

/// A JSON array of call objects (`[TOOL_CALLS][…]`, `<TOOLCALL>[…]`) as far
/// as it has been generated.  Complete elements without a name are skipped,
/// as they always were.  Returns the calls and, once the array has closed,
/// the bytes it spans.
pub(crate) fn call_list<'a>(
    s: &'a str,
    name_keys: &[&str],
    arg_keys: &[&str],
) -> Result<(Vec<CallObject<'a>>, Option<usize>), ()> {
    let b = s.as_bytes();
    let mut calls = Vec::new();
    let mut i = skip_ws(s, 0);
    if i >= b.len() {
        return Ok((calls, None));
    }
    if b[i] != b'[' {
        return Err(());
    }
    i += 1;
    let mut first = true;
    loop {
        i = skip_ws(s, i);
        if i >= b.len() {
            return Ok((calls, None));
        }
        if b[i] == b']' {
            return Ok((calls, Some(i + 1)));
        }
        if !first {
            if b[i] != b',' {
                return Err(());
            }
            i = skip_ws(s, i + 1);
            if i >= b.len() {
                return Ok((calls, None));
            }
        }
        first = false;
        let element = &s[i..];
        match call_object(element, name_keys, arg_keys) {
            Ok(call) => match call.end {
                Some(end) => {
                    calls.push(call);
                    i += end;
                }
                None => {
                    calls.push(call);
                    return Ok((calls, None));
                }
            },
            // A complete element that is not a call: skip it if it is still
            // valid JSON, as the array parse always did.
            Err(()) => match scan_value(element) {
                Json::Done(len) => i += len,
                Json::Partial => return Ok((calls, None)),
                Json::Bad => return Err(()),
            },
        }
    }
}

/// Arguments written as raw JSON and closed by one of a family's end
/// markers (DeepSeek V3.1, Kimi K2, harmony).
#[derive(Debug, Default)]
pub(crate) struct RawArgs {
    /// The text to stream.
    pub stream: String,
    /// The final arguments, once closed.
    pub done: Option<String>,
    /// Bytes up to the end of the closing marker.
    pub end: Option<usize>,
}

/// Read raw JSON arguments at the start of `s`, closed by one of `ends`.
///
/// Valid JSON streams as it arrives and its end is found by scanning it, so
/// a marker spelled inside a string value does not cut it short.  Text that
/// is not JSON is passed through verbatim once its end marker arrives, and
/// empty arguments are `{}`.  At the end of decoding an unclosed value is
/// final as it stands, up to an end marker if one follows.
pub(crate) fn raw_args(s: &str, ends: &[&str], finished: bool) -> RawArgs {
    let start = skip_ws(s, 0);
    let body = &s[start..];
    let closed = |raw: &str, end: usize| RawArgs {
        stream: if raw.is_empty() {
            "{}".to_string()
        } else {
            raw.to_string()
        },
        done: Some(if raw.is_empty() {
            "{}".to_string()
        } else {
            canonical(raw)
        }),
        end: Some(end),
    };
    match scan_value(body) {
        Json::Done(len) => {
            let json = &body[..len];
            let after = skip_ws(s, start + len);
            let rest = &s[after..];
            if let Some(end) = ends.iter().find(|end| rest.starts_with(**end)) {
                return closed(json, after + end.len());
            }
            if rest.is_empty() || ends.iter().any(|end| end.starts_with(rest)) {
                return RawArgs {
                    stream: json.to_string(),
                    done: finished.then(|| canonical(json)),
                    end: None,
                };
            }
            // Something follows the JSON before the marker: drop it.
            match super::find_first(rest, ends) {
                Some((at, end)) => closed(json, after + at + end.len()),
                None => RawArgs {
                    stream: json.to_string(),
                    done: finished.then(|| canonical(json)),
                    end: None,
                },
            }
        }
        scan => match super::find_first(s, ends) {
            // Not JSON, or never finished: the first end marker closes it.
            Some((at, end)) if scan == Json::Bad || finished => {
                closed(s[..at].trim(), at + end.len())
            }
            // An unfinished value streams up to an end marker (one inside
            // a string that never closes ends it after all) and its
            // trailing whitespace (a value cut off there ends without it).
            Some((at, _)) => RawArgs {
                stream: s[..at].trim().to_string(),
                done: None,
                end: None,
            },
            None => RawArgs {
                stream: match scan {
                    Json::Bad => String::new(),
                    _ => super::before_partial(body, ends).trim_end().to_string(),
                },
                done: finished.then(|| body.trim_end().to_string()),
                end: None,
            },
        },
    }
}

/// Compact re-serialisation of a complete JSON text, or the text itself when
/// it is not valid JSON.
pub(crate) fn canonical(raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .map(|value| value.to_string())
        .unwrap_or_else(|_| raw.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_value_reports_complete_partial_and_bad() {
        assert_eq!(
            scan_value(r#"{"a": [1, 2.5e3, "x\"y"]} tail"#),
            Json::Done(25)
        );
        assert_eq!(scan_value(r#"{"a": [1, 2"#), Json::Partial);
        assert_eq!(scan_value(r#"{"a": tr"#), Json::Partial);
        assert_eq!(scan_value(r#"{"a": "\u00"#), Json::Partial);
        assert_eq!(scan_value("12"), Json::Partial, "more digits may follow");
        assert_eq!(scan_value("12,"), Json::Done(2));
        assert_eq!(scan_value("{a: 1}"), Json::Bad);
        assert_eq!(scan_value(r#"{"a": tru }"#), Json::Bad);
        assert_eq!(scan_value("\"a\nb\""), Json::Bad, "raw newline in a string");
        assert_eq!(scan_value("[1,]"), Json::Bad);
        assert_eq!(scan_value("01"), Json::Done(1));
    }

    #[test]
    fn call_object_streams_arguments_before_the_object_closes() {
        let call = call_object(
            r#"{"name": "f", "arguments": {"city": "Par"#,
            &["name"],
            &["arguments"],
        )
        .unwrap();
        assert_eq!(call.name.as_deref(), Some("f"));
        assert_eq!(call.stream_args(), r#"{"city": "Par"#);
        assert_eq!(call.end, None);
        let call = call_object(
            r#" {"name": "f", "arguments": {"city": "Paris"}} rest"#,
            &["name"],
            &["arguments"],
        )
        .unwrap();
        assert_eq!(call.end, Some(46));
        assert_eq!(call.final_args(), r#"{"city":"Paris"}"#);
    }

    #[test]
    fn call_object_rejects_objects_without_a_name() {
        assert!(call_object(r#"{"answer": 42}"#, &["name"], &["arguments"]).is_err());
        assert!(call_object(r#"{"name": 3}"#, &["name"], &["arguments"]).is_err());
        assert!(call_object("[1]", &["name"], &["arguments"]).is_err());
        // Undecided while the name is still arriving.
        let call = call_object(r#"{"name": "lo"#, &["name"], &["arguments"]).unwrap();
        assert!(call.name.is_none());
    }

    #[test]
    fn call_list_reads_each_element() {
        let (calls, end) = call_list(
            r#"[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}] after"#,
            &["name"],
            &["arguments"],
        )
        .unwrap();
        assert_eq!(end, Some(70));
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].final_args(), r#"{"x":1}"#);
        let (calls, end) = call_list(
            r#"[{"name": "a", "arguments": {"#,
            &["name"],
            &["arguments"],
        )
        .unwrap();
        assert_eq!((calls.len(), end), (1, None));
    }
}
