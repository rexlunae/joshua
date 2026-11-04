//! JSON-family formats: the generic Hermes / Qwen3-Coder / Mistral / bare
//! JSON grammar, Llama 3.x JSON, and Nemotron Nano v2's `<TOOLCALL>` list.

use super::args::{Header, TaggedParams};
use super::json::{self, Json, skip_ws};
use super::{Cx, Grammar, Marker, PartialCall, Scan, find_first};

const TOOL_CALL: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const FUNCTION: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";
const MISTRAL: &str = "[TOOL_CALLS]";
const MISTRAL_ARGS: &str = "[ARGS]";
const MISTRAL_CALL_ID: &str = "[CALL_ID]";

const NAME_KEYS: &[&str] = &["name"];
const ARG_KEYS: &[&str] = &["arguments", "parameters"];

fn broken_call_end(
    region: &str,
    after: usize,
    open: &str,
    close: &str,
    finished: bool,
) -> Option<usize> {
    let close_at = region[after..].find(close).map(|offset| after + offset);
    let next_at = region[after..].find(open).map(|offset| after + offset);
    match (next_at, close_at) {
        (Some(next), Some(close)) if next < close => Some(next),
        (Some(next), None) => Some(next),
        (_, Some(at)) => Some(at + close.len()),
        (None, None) if finished => Some(region.len()),
        (None, None) => None,
    }
}

fn next_json_call(s: &str, after: usize) -> Option<usize> {
    s[after..]
        .match_indices('{')
        .filter_map(|(offset, _)| {
            let at = after + offset;
            let previous = s[..at].chars().next_back()?;
            if !previous.is_whitespace() && !matches!(previous, '}' | ';') {
                return None;
            }
            let call = json::call_object(&s[at..], NAME_KEYS, ARG_KEYS).ok()?;
            call.name.map(|_| at)
        })
        .next()
}

fn json_calls_after(s: &str, mut at: usize) -> (Vec<PartialCall>, usize) {
    let mut calls = Vec::new();
    loop {
        let start = skip_ws(s, at);
        let Ok(call) = json::call_object(&s[start..], NAME_KEYS, ARG_KEYS) else {
            return (calls, at);
        };
        if call.name.is_none() {
            return (calls, at);
        }
        let end = call.end.map_or(s.len() - start, |end| end);
        calls.push(call.partial(true));
        at = start + end;
        if at == s.len() {
            return (calls, at);
        }
    }
}

/// Where a region closed by `close` ends, given that its content ends at
/// `at`: after the closing tag (whitespace before it allowed); after a
/// closing tag further on, dropping what comes between as broken markup
/// (a stray brace, a second object), unless one of `stops` (a new region
/// or call) comes first; still open while the tag may yet arrive; or at
/// `at` when none does, what follows being prose.
pub(super) fn close_after(
    region: &str,
    at: usize,
    close: &str,
    stops: &[&str],
    finished: bool,
) -> Option<usize> {
    let tag = skip_ws(region, at);
    let rest = &region[tag..];
    if rest.is_empty() {
        return None;
    }
    if rest.starts_with(close) {
        return Some(tag + close.len());
    }
    if close.starts_with(rest) {
        return finished.then_some(region.len());
    }
    match (rest.find(close), find_first(rest, stops)) {
        (Some(end), stop) if stop.is_none_or(|(stop, _)| end < stop) => {
            tracing::debug!("dropping text after a tool call, before its closing tag");
            Some(tag + end + close.len())
        }
        (None, None) if !finished => None,
        _ => Some(at),
    }
}

/// Undecided while the output may still grow, ordinary text once it cannot.
pub(super) fn undecided(cx: &Cx<'_>, marker: &str) -> Scan {
    if cx.finished {
        Scan::Text(marker.len())
    } else {
        Scan::Pending
    }
}

// ─── Generic ────────────────────────────────────────────────────────────────

/// Hermes / Qwen `<tool_call>` JSON, Qwen3-Coder XML (wrapped or bare),
/// Mistral `[TOOL_CALLS]`, and a response that is a single JSON call.
pub(super) struct Generic;

pub(super) static GENERIC: Generic = Generic;

impl Grammar for Generic {
    fn markers(&self) -> &'static [Marker] {
        const MARKERS: &[Marker] = &[
            Marker::anywhere(TOOL_CALL).closed_by(&[TOOL_CALL_END]),
            Marker::anywhere(FUNCTION).closed_by(&[FUNCTION_END]),
            Marker::anywhere(MISTRAL),
            Marker::leading("{"),
        ];
        MARKERS
    }

    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan {
        if region.starts_with(TOOL_CALL) {
            hermes(region, cx)
        } else if region.starts_with(FUNCTION) {
            match xml_functions(region, 0, cx) {
                Err(()) => Scan::Text(FUNCTION.len()),
                Ok(None) => undecided(cx, FUNCTION),
                Ok(Some((calls, end))) => Scan::Calls { calls, end },
            }
        } else if region.starts_with(MISTRAL) {
            mistral(region, cx)
        } else {
            bare_json(region, cx)
        }
    }
}

/// `<tool_call>` holding a JSON call or Qwen3-Coder `<function=…>` blocks.
pub(super) fn hermes(region: &str, cx: &Cx<'_>) -> Scan {
    let start = skip_ws(region, TOOL_CALL.len());
    let payload = &region[start..];
    if payload.is_empty() {
        return undecided(cx, TOOL_CALL);
    }
    if payload.starts_with('{') {
        let call = match json::call_object(payload, NAME_KEYS, ARG_KEYS) {
            Ok(call) => call,
            Err(()) => {
                let Some((call, prefix_len)) =
                    json::call_object_name_prefix(payload, NAME_KEYS, ARG_KEYS)
                else {
                    return Scan::Text(TOOL_CALL.len());
                };
                let end = broken_call_end(
                    region,
                    start + prefix_len,
                    TOOL_CALL,
                    TOOL_CALL_END,
                    cx.finished,
                );
                return Scan::Calls {
                    calls: vec![call],
                    end,
                };
            }
        };
        if call.name.is_none() {
            return undecided(cx, TOOL_CALL);
        }
        let end = call.end.and_then(|end| {
            close_after(
                region,
                start + end,
                TOOL_CALL_END,
                &[TOOL_CALL],
                cx.finished,
            )
        });
        return Scan::Calls {
            calls: vec![call.partial(cx.finished)],
            end,
        };
    }
    if !payload.starts_with(FUNCTION) {
        return if FUNCTION.starts_with(payload) {
            undecided(cx, TOOL_CALL)
        } else {
            Scan::Text(TOOL_CALL.len())
        };
    }
    match xml_functions(region, start, cx) {
        Err(()) => Scan::Text(TOOL_CALL.len()),
        Ok(None) => undecided(cx, TOOL_CALL),
        Ok(Some((calls, Some(end)))) => Scan::Calls {
            calls,
            end: close_after(region, end, TOOL_CALL_END, &[TOOL_CALL], cx.finished),
        },
        Ok(Some((calls, None))) => Scan::Calls { calls, end: None },
    }
}

/// Calls read so far and, once the last has closed, where the region ends.
pub(super) type Calls = (Vec<PartialCall>, Option<usize>);

/// Consecutive Qwen3-Coder `<function=name>` blocks starting at `start`.
/// `Ok(None)` while the first name is still arriving; `Err` when the first
/// block is not a call.  A later `<function=` that is not a call ends the
/// run before it, as it does when it arrives after the run has ended.
fn xml_functions(region: &str, start: usize, cx: &Cx<'_>) -> Result<Option<Calls>, ()> {
    let mut calls = Vec::new();
    let mut at = start;
    let mut last_end = start;
    loop {
        let (call, end) = match xml_function(&region[at..], cx) {
            Ok(Some(call)) => call,
            Err(()) if calls.is_empty() => return Err(()),
            Ok(None) if calls.is_empty() => return Ok(None),
            Err(()) => return Ok(Some((calls, Some(last_end)))),
            Ok(None) => return Ok(Some((calls, None))),
        };
        calls.push(call);
        let Some(end) = end else {
            return Ok(Some((calls, None)));
        };
        at += end;
        last_end = at;
        let next = skip_ws(region, at);
        let rest = &region[next..];
        if rest.starts_with(FUNCTION) {
            at = next;
            continue;
        }
        if !cx.finished && !rest.is_empty() && FUNCTION.starts_with(rest) {
            return Ok(Some((calls, None)));
        }
        return Ok(Some((calls, Some(at))));
    }
}

/// Qwen3-Coder's `<parameter=key>value</parameter>`.  A value also ends at
/// the next parameter or the function's end when its closing tag is
/// missing.
const QWEN3_PARAMS: TaggedParams = TaggedParams {
    call_end: &[FUNCTION_END],
    next: &[PARAMETER, FUNCTION_END],
    value_end: &[PARAMETER_END, PARAMETER, FUNCTION_END],
    value_close: &[PARAMETER_END],
    header: |s| match s.find('>') {
        Some(len) => Header::Done {
            key: s[..len].trim().into(),
            string: None,
            len: len + 1,
        },
        None => Header::Partial,
    },
    trim: trim_wrapping_newlines,
};

/// One `<function=name>…</function>` block at the start of `s`: `Ok(None)`
/// while the name is still arriving, `Err` when it is not a function name.
/// The body is Qwen3-Coder's tagged parameters, or a JSON object
/// (Functionary v3.1's `<function=name>{…}</function>`).
fn xml_function(s: &str, cx: &Cx<'_>) -> Result<Option<(PartialCall, Option<usize>)>, ()> {
    let name_start = FUNCTION.len();
    let Some(name_len) = s[name_start..].find('>') else {
        // Whitespace around the name is allowed, as it is once `>` comes.
        let partial = s[name_start..].trim();
        return if cx.finished || partial.contains(char::is_whitespace) {
            Err(())
        } else {
            Ok(None)
        };
    };
    let name = s[name_start..name_start + name_len].trim();
    if name.is_empty() || name.contains(char::is_whitespace) {
        return Err(());
    }
    let body_start = name_start + name_len + 1;
    let body = &s[body_start..];
    if body.trim_start().starts_with('{') {
        let args = json::raw_args(body, &[FUNCTION_END], cx.finished);
        let call = PartialCall {
            id: None,
            name: name.to_string(),
            args: args.stream,
            done: args.done,
        };
        return Ok(Some((call, args.end.map(|end| body_start + end))));
    }
    let params = QWEN3_PARAMS.read(body, name, cx);
    Ok(Some((
        params.call(name, cx),
        params.end.map(|end| body_start + end),
    )))
}

/// Qwen3-Coder writes a newline after `<parameter=…>` and before the
/// closing tag; strip one of each (vLLM's `_trim_wrapping_newlines`).
fn trim_wrapping_newlines(value: &str) -> &str {
    let value = value.strip_prefix('\n').unwrap_or(value);
    value.strip_suffix('\n').unwrap_or(value)
}

/// Mistral: `[TOOL_CALLS][{…}, …]`, or (v11 and later)
/// `[TOOL_CALLS]name[ARGS]{…}` with an optional `[CALL_ID]id` before
/// `[ARGS]`.
fn mistral(region: &str, cx: &Cx<'_>) -> Scan {
    let start = skip_ws(region, MISTRAL.len());
    let payload = &region[start..];
    if payload.is_empty() {
        return undecided(cx, MISTRAL);
    }
    if payload.starts_with('[') {
        return marked_list(region, MISTRAL, None, cx, |s| json_list(s, cx));
    }
    let Some((name_len, tag)) = find_first(payload, &[MISTRAL_ARGS, MISTRAL_CALL_ID]) else {
        // Still reading the name, unless it has stopped looking like one.
        return if payload.contains(char::is_whitespace) {
            Scan::Text(MISTRAL.len())
        } else {
            undecided(cx, MISTRAL)
        };
    };
    let name = &payload[..name_len];
    if name.is_empty() || name.contains(char::is_whitespace) {
        return Scan::Text(MISTRAL.len());
    }
    let mut at = name_len + tag.len();
    let mut id = None;
    if tag == MISTRAL_CALL_ID {
        let Some(id_len) = payload[at..].find(MISTRAL_ARGS) else {
            return undecided(cx, MISTRAL);
        };
        id = Some(payload[at..at + id_len].trim().to_string());
        at += id_len + MISTRAL_ARGS.len();
    }
    let args_at = skip_ws(payload, at);
    let raw = &payload[args_at..];
    let (args, end) = match json::scan_value(raw) {
        Json::Done(len) => (raw[..len].to_string(), Some(start + args_at + len)),
        _ if cx.finished => (json::partial_json(raw), Some(region.len())),
        _ => (json::partial_json(raw), None),
    };
    let done = end.map(|_| json::canonical(&args));
    let streamable = args.starts_with(['{', '[']) || done.is_some();
    Scan::Calls {
        calls: vec![PartialCall {
            id,
            name: name.to_string(),
            args: if streamable { args } else { String::new() },
            done,
        }],
        end,
    }
}

/// A response that is a single JSON call.  It cannot be told apart from a
/// JSON answer until the whole response is in, so it is held to the end.
fn bare_json(region: &str, cx: &Cx<'_>) -> Scan {
    if !cx.finished {
        return Scan::Pending;
    }
    let body = region.trim();
    match json::call_object(body, NAME_KEYS, ARG_KEYS) {
        Ok(call) if call.name.is_some() && call.end == Some(body.len()) => Scan::Calls {
            calls: vec![call.partial(true)],
            end: Some(region.len()),
        },
        Err(()) => match json::call_object_name_prefix(body, NAME_KEYS, ARG_KEYS) {
            Some((call, prefix_len)) => {
                let leading = region.len() - region.trim_start().len();
                if let Some(next) = next_json_call(body, prefix_len) {
                    let (following, end) = json_calls_after(body, next);
                    let mut calls = vec![call];
                    calls.extend(following);
                    Scan::Calls {
                        calls,
                        end: Some(leading + end),
                    }
                } else {
                    Scan::Calls {
                        calls: vec![call],
                        end: Some(region.len()),
                    }
                }
            }
            None => Scan::Text(region.len()),
        },
        _ => Scan::Text(region.len()),
    }
}

// ─── Llama 3.x JSON ─────────────────────────────────────────────────────────

const PYTHON_TAG: &str = "<|python_tag|>";
const EOM: &str = "<|eom_id|>";

/// Llama 3.x's JSON calls, `;`-separated: `{"name": …, "parameters": …}`,
/// at the start of the response or after `<|python_tag|>` (vLLM's
/// `llama3_json`).  Unlike the generic bare-JSON case, a response that
/// opens with `{"name": …` is taken as a call and streamed.
pub(super) struct LlamaJson;

pub(super) static LLAMA_JSON: LlamaJson = LlamaJson;

impl Grammar for LlamaJson {
    fn markers(&self) -> &'static [Marker] {
        const MARKERS: &[Marker] = &[
            Marker::anywhere(PYTHON_TAG).closed_by(&[EOM]),
            Marker::leading("{"),
        ];
        MARKERS
    }

    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan {
        let first = if region.starts_with(PYTHON_TAG) {
            PYTHON_TAG.len()
        } else {
            0
        };
        llama_json_calls(region, first, cx)
    }
}

/// `;`-separated JSON calls from `first`, and the end-of-message token
/// Llama writes after a tagged call.  What follows a `;` is another call:
/// when it is not one, the region is broken markup like a call that breaks
/// off, so the stream keeps the calls read before it and drops the rest.
/// A region that is no call gives back only its opening as text, so a
/// marker later in it is still found.
fn llama_json_calls(region: &str, first: usize, cx: &Cx<'_>) -> Scan {
    let mut at = first;
    let mut calls = Vec::new();
    loop {
        let start = skip_ws(region, at);
        if start >= region.len() {
            return Scan::Calls { calls, end: None };
        }
        let call = match json::call_object(&region[start..], NAME_KEYS, ARG_KEYS) {
            Ok(call) if call.name.is_some() => call,
            Ok(_) if !cx.finished => {
                return if calls.is_empty() {
                    Scan::Pending
                } else {
                    Scan::Calls { calls, end: None }
                };
            }
            Err(()) => {
                if let Some((call, _)) =
                    json::call_object_name_prefix(&region[start..], NAME_KEYS, ARG_KEYS)
                {
                    calls.push(call);
                    let end = region[start..]
                        .find(EOM)
                        .map(|offset| start + offset + EOM.len())
                        .or_else(|| cx.finished.then_some(region.len()));
                    return Scan::Calls { calls, end };
                }
                return Scan::Text(first);
            }
            _ => return Scan::Text(first),
        };
        calls.push(call.partial(cx.finished));
        let Some(end) = call.end else {
            return Scan::Calls { calls, end: None };
        };
        at = start + end;
        let next = skip_ws(region, at);
        let rest = &region[next..];
        if rest.starts_with(';') {
            at = next + 1;
        } else if rest.starts_with(EOM) {
            return Scan::Calls {
                calls,
                end: Some(next + EOM.len()),
            };
        } else if !cx.finished && EOM.starts_with(rest) {
            // More calls, or the end-of-message token, may follow.
            return Scan::Calls { calls, end: None };
        } else if !rest.is_empty() {
            return Scan::Calls {
                calls,
                end: Some(at),
            };
        }
    }
}

// ─── Nemotron Nano v2 ───────────────────────────────────────────────────────

const NEMOTRON: &str = "<TOOLCALL>";
const NEMOTRON_END: &str = "</TOOLCALL>";

/// Nemotron Nano v2: `<TOOLCALL>[{"name": …, "arguments": …}]</TOOLCALL>`.
pub(super) struct NemotronV2;

pub(super) static NEMOTRON_V2: NemotronV2 = NemotronV2;

impl Grammar for NemotronV2 {
    fn markers(&self) -> &'static [Marker] {
        const MARKERS: &[Marker] = &[Marker::anywhere(NEMOTRON).closed_by(&[NEMOTRON_END])];
        MARKERS
    }

    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan {
        marked_list(region, NEMOTRON, Some(NEMOTRON_END), cx, |s| {
            json_list(s, cx)
        })
    }
}

// ─── Phi-4-mini / Firefunction v2 ───────────────────────────────────────────

const FUNCTOOLS: &str = "functools";

/// Phi-4-mini and Firefunction v2: `functools[{"name": …, "arguments": …}]`
/// (vLLM's `phi4_mini_json`).
pub(super) struct Functools;

pub(super) static FUNCTOOLS_LIST: Functools = Functools;

impl Grammar for Functools {
    fn markers(&self) -> &'static [Marker] {
        const MARKERS: &[Marker] = &[Marker::anywhere("functools[")];
        MARKERS
    }

    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan {
        marked_list(region, FUNCTOOLS, None, cx, |s| json_list(s, cx))
    }
}

/// A list of calls after `marker`, closed by `close` when the family writes
/// one: Mistral's `[TOOL_CALLS][…]`, Nemotron's `<TOOLCALL>[…]</TOOLCALL>`,
/// `functools[…]`, and the pythonic lists of Llama 4 and LFM2.  `list`
/// reads the calls from the text after the marker, with the list's length
/// once it has closed; `Err` when it is not a list.
pub(super) fn marked_list(
    region: &str,
    marker: &str,
    close: Option<&str>,
    cx: &Cx<'_>,
    list: impl FnOnce(&str) -> Result<Calls, ()>,
) -> Scan {
    let start = skip_ws(region, marker.len());
    if start >= region.len() {
        return undecided(cx, marker);
    }
    match list(&region[start..]) {
        Err(()) => Scan::Text(marker.len()),
        Ok((calls, end)) => Scan::Calls {
            calls,
            end: end.and_then(|end| match close {
                Some(close) => close_after(region, start + end, close, &[marker], cx.finished),
                None => Some(start + end),
            }),
        },
    }
}

/// A JSON list of `{"name": …, "arguments": …}` calls, for [`marked_list`].
fn json_list(s: &str, cx: &Cx<'_>) -> Result<Calls, ()> {
    let (calls, end) = json::call_list(s, NAME_KEYS, ARG_KEYS)?;
    let calls = calls
        .iter()
        .map(|call| call.partial(cx.finished || end.is_some()))
        .collect();
    Ok((calls, end))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{args, assert_streams_like_parse, parse};
    use super::super::{ToolCallFormat, parse_tool_calls};
    use crate::types::{FunctionDef, Tool};

    const G: ToolCallFormat = ToolCallFormat::Generic;

    #[test]
    fn hermes_single_call() {
        let text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        let (content, calls) = parse(G, text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(args(&calls[0])["city"], "Paris");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
        assert_streams_like_parse(G, &[], text);
    }

    #[test]
    fn hermes_multiple_calls_with_prose() {
        let text = "Let me check both.\n\
             <tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>\n\
             <tool_call>{\"name\": \"b\", \"arguments\": {\"x\": 1}}</tool_call>";
        let (content, calls) = parse(G, text);
        assert_eq!(content, "Let me check both.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert_streams_like_parse(G, &[], text);
    }

    /// Markup that breaks off after the call's name was read (Python
    /// literals, a trailing comma): a stream has sent the call by then, so
    /// it stands as far as it was read, streamed or not, and the rest of
    /// its tag never reaches the content.
    #[test]
    fn hermes_call_broken_after_its_name_stays_a_call() {
        for (call, name, arguments) in [
            (
                "{\"name\": \"get_weather\", \"arguments\": {'city': 'Paris'}}",
                "get_weather",
                "{",
            ),
            (
                "{\"name\": \"set\", \"arguments\": {\"on\": True}}",
                "set",
                "{\"on\": ",
            ),
            (
                "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\",}}",
                "get_weather",
                "{\"city\": \"Paris\",",
            ),
        ] {
            let text = format!("Checking.<tool_call>\n{call}\n</tool_call> Done.");
            let (content, calls) = parse(G, &text);
            assert_eq!(content, "Checking. Done.", "{text}");
            assert_eq!(calls.len(), 1, "{text}");
            assert_eq!(calls[0].name, name);
            assert_eq!(calls[0].arguments, arguments);
            assert_streams_like_parse(G, &[], &text);
            // Cut off before its closing tag, the rest of the output is
            // the broken call's.
            let text = format!("<tool_call>\n{call} Done.");
            let (content, calls) = parse(G, &text);
            assert_eq!((content.as_str(), calls.len()), ("", 1), "{text}");
            assert_streams_like_parse(G, &[], &text);
        }
        // A name that never completes is no call.
        let text = "<tool_call>{\"name\": 'f'}</tool_call>";
        assert_eq!(parse(G, text), (text.to_string(), Vec::new()));
        assert_streams_like_parse(G, &[], text);
    }

    /// Text after a complete call but inside its tag (a stray brace, a
    /// second object, which vLLM's reader returns as content) is dropped
    /// with the tag; without a closing tag, what follows is prose.
    #[test]
    fn hermes_text_after_the_call_inside_its_tag_is_dropped() {
        for text in [
            "<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": 1}}}}</tool_call> Done.",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n\
             {\"name\": \"g\", \"arguments\": {}}\n</tool_call> Done.",
        ] {
            let (content, calls) = parse(G, text);
            assert_eq!(content, "Done.", "{text}");
            assert_eq!(calls.len(), 1, "{text}");
            assert_eq!(calls[0].arguments, r#"{"a":1}"#);
            assert_streams_like_parse(G, &[], text);
        }
        let text = "<tool_call>{\"name\": \"f\", \"arguments\": {}} Done.";
        let (content, calls) = parse(G, text);
        assert_eq!((content.as_str(), calls.len()), ("Done.", 1));
        assert_streams_like_parse(G, &[], text);
        // A second call's tag ends the first, unclosed one.
        let text = "<tool_call>{\"name\": \"f\", \"arguments\": {}}\n\
                    <tool_call>{\"name\": \"g\", \"arguments\": {}}</tool_call>";
        assert_eq!(parse(G, text).1.len(), 2);
        assert_streams_like_parse(G, &[], text);
    }

    #[test]
    fn hermes_unterminated_tag_still_parses() {
        let (_, calls) = parse(
            G,
            "<tool_call>{\"name\": \"f\", \"arguments\": {\"k\": \"v\"}}",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments, r#"{"k":"v"}"#);
    }

    #[test]
    fn mistral_tool_calls_array() {
        let text = "[TOOL_CALLS][{\"name\": \"get_time\", \"arguments\": {\"tz\": \"UTC\"}}]";
        let (content, calls) = parse(G, text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_time");
        assert_streams_like_parse(G, &[], text);
    }

    /// Mistral Small 3.2 / Devstral / Magistral: `name[ARGS]{…}`, with the
    /// call id the model writes after `[CALL_ID]` (llama.cpp's
    /// `test-chat.cpp` cases).
    #[test]
    fn mistral_v11_calls_with_and_without_ids() {
        let text = "[TOOL_CALLS]special_function[CALL_ID]000000001[ARGS]{\"arg1\": 1}\
                    [TOOL_CALLS]special_function_with_opt[CALL_ID]000000002[ARGS]{\"arg1\": 1, \"arg2\": 2}";
        let (content, calls) = parse(G, text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id.as_deref(), Some("000000001"));
        assert_eq!(calls[1].name, "special_function_with_opt");
        assert_eq!(calls[1].arguments, r#"{"arg1":1,"arg2":2}"#);
        assert_streams_like_parse(G, &[], text);

        let text = "Hello, world!\nWhat's up?[TOOL_CALLS]special_function[ARGS]{\"arg1\": 1}";
        let (content, calls) = parse(G, text);
        assert_eq!(content, "Hello, world!\nWhat's up?");
        assert_eq!(calls[0].name, "special_function");
        assert_eq!(calls[0].id, None);
        assert_streams_like_parse(G, &[], text);
    }

    #[test]
    fn llama3_bare_json_with_parameters_key() {
        let text = "{\"name\": \"lookup\", \"parameters\": {\"q\": \"rust\"}}";
        let (content, calls) = parse(G, text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
        assert_eq!(args(&calls[0])["q"], "rust");
        assert_streams_like_parse(G, &[], text);
    }

    #[test]
    fn bare_json_without_name_is_plain_content() {
        let input = "{\"answer\": 42}";
        let (content, calls) = parse(G, input);
        assert_eq!(content, input);
        assert!(calls.is_empty());
    }

    #[test]
    fn string_encoded_arguments_pass_through() {
        let (_, calls) = parse(
            G,
            "<tool_call>{\"name\": \"f\", \"arguments\": \"{\\\"a\\\": 1}\"}</tool_call>",
        );
        assert_eq!(calls[0].arguments, "{\"a\": 1}");
    }

    #[test]
    fn qwen3_coder_xml_call_inside_tool_call_tags() {
        // The exact shape Qwen3-Coder-30B-A3B emits with its GGUF template.
        let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = parse(G, text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(args(&calls[0])["city"], "Tokyo");
        assert_streams_like_parse(G, &[], text);
    }

    #[test]
    fn qwen3_coder_xml_bare_multiple_params_with_types() {
        let text = "Calling now.\n<function=search>\n<parameter=query>\nrust async\n</parameter>\n<parameter=limit>5</parameter>\n<parameter=exact>true</parameter>\n</function>";
        let (content, calls) = parse(G, text);
        assert_eq!(content, "Calling now.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        let a = args(&calls[0]);
        assert_eq!(a["query"], "rust async");
        assert_eq!(a["limit"], 5);
        assert_eq!(a["exact"], true);
        assert_eq!(
            calls[0].arguments,
            r#"{"query":"rust async","limit":5,"exact":true}"#
        );
        assert_streams_like_parse(G, &[], text);
    }

    #[test]
    fn qwen3_coder_xml_cut_off_still_parses_params() {
        let (_, calls) = parse(
            G,
            "<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo</parameter>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(args(&calls[0])["city"], "Tokyo");
    }

    /// With the tool's schema, a string parameter stays a string even when
    /// it spells a number, keeps its indentation, and streams while it is
    /// still being written.
    #[test]
    fn qwen3_coder_string_parameters_follow_the_schema_and_stream() {
        let tools = [Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "write_file".into(),
                description: None,
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "mode": {"type": "integer"}
                    }
                })),
            },
        }];
        let text = "<tool_call>\n<function=write_file>\n<parameter=path>\n007\n</parameter>\n\
                    <parameter=content>\n    indented();\n    next();\n</parameter>\n\
                    <parameter=mode>\n644\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(G, &tools, text);
        let a = args(&calls[0]);
        assert_eq!(a["path"], "007");
        assert_eq!(a["content"], "    indented();\n    next();");
        assert_eq!(a["mode"], 644);
        assert_streams_like_parse(G, &tools, text);

        // The content parameter streams before its closing tag arrives.
        let mut stream = super::super::ToolCallStream::new(G, Some(&tools));
        let cut = text.find("next").unwrap();
        let mut streamed = String::new();
        for event in stream.push(&text[..cut]) {
            if let super::super::ToolEvent::CallArgs { delta, .. } = event {
                streamed.push_str(&delta);
            }
        }
        assert!(
            streamed.ends_with(r#""content":"    indented();\n    "#),
            "{streamed}"
        );
    }

    #[test]
    fn llama_json_streams_and_splits_on_semicolons() {
        let f = ToolCallFormat::LlamaJson;
        let text = "{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Tokyo\"}}; \
                    {\"name\": \"add\", \"parameters\": {\"x\": 1, \"y\": 2}}";
        let (content, calls) = parse(f, text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].arguments, r#"{"x":1,"y":2}"#);
        assert_streams_like_parse(f, &[], text);
        let (content, calls) = parse(f, "Hello, world!\nWhat's up?");
        assert_eq!(content, "Hello, world!\nWhat's up?");
        assert!(calls.is_empty());
        let (content, calls) = parse(f, "{\"answer\": 42}");
        assert_eq!(content, "{\"answer\": 42}");
        assert!(calls.is_empty());
    }

    /// After a `;` comes another call.  One that breaks off after its name
    /// stands as far as it was read, like any call whose markup breaks
    /// off, and the rest of the region (through `<|eom_id|>` after
    /// `<|python_tag|>`, else the rest of the output) never reaches the
    /// content, however the text is split.
    #[test]
    fn llama_json_call_broken_after_a_semicolon_stays_a_call() {
        let f = ToolCallFormat::LlamaJson;
        let paris = "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}";
        for (text, prose, broken) in [
            (
                format!("{paris}; {{\"name\": \"get_weather\", \"parameters\": {{\"city\": x}}}}"),
                "",
                "{\"city\": ",
            ),
            (
                format!(
                    "Let me look.<|python_tag|>{paris}; {{\"name\": \"get_weather\", \
                     \"parameters\": {{'city': 'Rome'}}}}<|eom_id|> Done."
                ),
                "Let me look. Done.",
                "{",
            ),
        ] {
            let (content, calls) = parse(f, &text);
            assert_eq!(content, prose, "{text}");
            assert_eq!(calls.len(), 2, "{text}");
            assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
            assert_eq!(calls[1].arguments, broken, "{text}");
            assert_streams_like_parse(f, &[], &text);
        }
        // Text after a `;` that never starts a call is dropped too.
        let text = format!("{paris}; and that is all.");
        assert_eq!(parse(f, &text).0, "");
        assert_eq!(parse(f, &text).1.len(), 1);
        assert_streams_like_parse(f, &[], &text);
        // Without a `;`, what follows the call is prose.
        let text = format!("{paris} and that is all.");
        assert_eq!(parse(f, &text).0, "and that is all.");
        assert_streams_like_parse(f, &[], &text);
    }

    /// A `<function=` after Qwen3-Coder / Functionary calls that is not a
    /// call ends their run: bare, what follows is prose; inside
    /// `<tool_call>`, it is dropped through the closing tag.
    #[test]
    fn xml_function_that_is_not_a_call_ends_the_run() {
        let call = "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>";
        for (text, prose) in [
            (
                format!("{call}\n<function=get weather> and the rest of my answer is here."),
                "<function=get weather> and the rest of my answer is here.",
            ),
            (
                format!("<tool_call>\n{call}\n<function=get weather> oops\n</tool_call> Done."),
                "Done.",
            ),
        ] {
            let (content, calls) = parse(G, &text);
            assert_eq!(content, prose, "{text}");
            assert_eq!(calls.len(), 1, "{text}");
            assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
            assert_streams_like_parse(G, &[], &text);
        }
    }

    /// llama.cpp's NVIDIA-Nemotron-Nano-v2 cases (after the reasoning).
    #[test]
    fn nemotron_v2_toolcall_list() {
        let f = ToolCallFormat::NemotronV2;
        let text = "\n<TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>\n";
        let (content, calls) = parse(f, text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "special_function");
        assert_eq!(args(&calls[0]), serde_json::json!({"arg1": 1}));
        assert_streams_like_parse(f, &[], text);
        let (content, _) = parse(
            f,
            "Sure.<TOOLCALL>[{\"name\": \"a\", \"arguments\": {}}]</TOOLCALL> Done.",
        );
        assert_eq!(content, "Sure. Done.");
    }

    #[test]
    fn generic_markup_inside_prose_without_a_call_is_kept() {
        let text = "Use <tool_call> tags like this: <tool_call>not json</tool_call>.";
        let (content, calls) = parse(G, text);
        assert_eq!(content, text);
        assert!(calls.is_empty());
        let (content, calls) = parse(G, "See [TOOL_CALLS] in the docs.");
        assert_eq!(content, "See [TOOL_CALLS] in the docs.");
        assert!(calls.is_empty());
        let text = "Call it as <function=name with spaces> in prose.";
        let (content, calls) = parse(G, text);
        assert_eq!(content, text);
        assert!(calls.is_empty());
    }

    /// Functionary v3.1 writes a JSON object inside `<function=…>`.
    #[test]
    fn functionary_json_function_body() {
        let text = "<function=get_weather>{\"location\": \"Paris\"}</function>";
        let (content, calls) = parse(G, text);
        assert_eq!(content, "");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"location":"Paris"}"#);
        assert_streams_like_parse(G, &[], text);
    }

    /// Llama 3.1's `<|python_tag|>` before its JSON calls, with the
    /// end-of-message token it writes after them.
    #[test]
    fn llama_json_after_python_tag() {
        let f = ToolCallFormat::LlamaJson;
        let text = "Let me look.<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}<|eom_id|>";
        let (content, calls) = parse(f, text);
        assert_eq!(content, "Let me look.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
        assert_streams_like_parse(f, &[], text);
        // A built-in tool call is not JSON: it stays text.
        let text = "<|python_tag|>brave_search.call(query=\"rust\")";
        assert_eq!(parse(f, text), (text.to_string(), Vec::new()));
    }

    /// Phi-4-mini / Firefunction v2 (vLLM's `phi4_mini_json` cases).
    #[test]
    fn functools_list() {
        let f = ToolCallFormat::Functools;
        let text = "functools[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, \
                    {\"name\": \"get_time\", \"arguments\": {}}]";
        let (content, calls) = parse(f, text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].name, "get_time");
        assert_streams_like_parse(f, &[], text);
        let text = "Python's functools module is useful.";
        assert_eq!(parse(f, text).0, text);
    }
}
