//! The streaming tool-call state machine every chat API drives.
//!
//! [`ToolCallStream`] takes decoded text as it arrives and turns it into
//! [`ToolEvent`]s: prose as soon as no tool-call marker can begin in it, a
//! call's start once its name is known, then its arguments as they grow.
//! Partial markup never reaches the prose: a possible marker prefix at the
//! end of the text is held back, and a region that turns out not to be a
//! call is released as text.  A call counts from the moment its name is
//! read, however the text is split into pieces: if its markup then breaks
//! off, the call stands as far as it was read and the rest of its region is
//! dropped, so a complete output parses to what its stream sent.

use super::{Cx, Grammar, Marker, ParsedToolCall, PartialCall, Scan, ToolCallFormat};
use crate::types::Tool;

/// One step of a streamed response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolEvent {
    /// Prose outside any tool call.
    Text(String),
    /// A tool call begins.  `id` is the model's own call id, when it wrote
    /// one; otherwise the API assigns one.
    CallStart {
        index: usize,
        id: Option<String>,
        name: String,
    },
    /// More of call `index`'s JSON arguments.
    CallArgs { index: usize, delta: String },
}

/// What has been sent for one call.
#[derive(Debug)]
struct Sent {
    id: Option<String>,
    name: String,
    args: String,
    done: Option<String>,
}

/// Incremental tool-call parser for one generation.
pub struct ToolCallStream<'a> {
    /// `None`: the request offered no tools, so all text is prose.
    grammar: Option<&'static dyn Grammar>,
    format: Option<ToolCallFormat>,
    tools: &'a [Tool],
    text: String,
    /// Bytes of `text` already emitted or consumed.
    pos: usize,
    /// Whether this is still the beginning of the response, independent of
    /// buffer compaction.
    at_start: bool,
    /// Start of the tool-call region being read.
    region: Option<usize>,
    /// Index in `calls` of the open region's first call.
    region_first: usize,
    /// Where the open region's markup broke off after a call in it had
    /// begun: the region is skipped from there through its closing string.
    broken: Option<usize>,
    calls: Vec<Sent>,
    /// Whether whitespace-only prose is held back: at the start of the
    /// output and after a call, until other prose follows.  Held
    /// whitespace is dropped if a call (or, after a call, the end) comes
    /// first, so none is sent around or between calls.
    hold_whitespace: bool,
    held_whitespace: String,
}

impl<'a> ToolCallStream<'a> {
    /// A stream for a model emitting `format`.  Without `tools`, markup is
    /// ordinary text and everything passes straight through.
    pub fn new(format: ToolCallFormat, tools: Option<&'a [Tool]>) -> Self {
        let mut stream =
            Self::with_grammar(tools.map(|_| format.grammar()), tools.unwrap_or_default());
        stream.format = tools.map(|_| format);
        stream
    }

    fn with_grammar(grammar: Option<&'static dyn Grammar>, tools: &'a [Tool]) -> Self {
        Self {
            grammar,
            format: None,
            tools,
            text: String::new(),
            pos: 0,
            at_start: true,
            region: None,
            region_first: 0,
            broken: None,
            calls: Vec::new(),
            hold_whitespace: true,
            held_whitespace: String::new(),
        }
    }

    /// Feed newly decoded text; returns what can be sent now.
    pub fn push(&mut self, delta: &str) -> Vec<ToolEvent> {
        if self.grammar.is_none() {
            return if delta.is_empty() {
                Vec::new()
            } else {
                vec![ToolEvent::Text(delta.to_string())]
            };
        }
        self.text.push_str(delta);
        let mut out = Vec::new();
        self.drain(false, &mut out);
        self.compact();
        out
    }

    /// Flush everything held back once decoding has ended.
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        let mut out = Vec::new();
        if self.grammar.is_none() {
            return out;
        }
        self.drain(true, &mut out);
        self.compact();
        // An output of nothing but whitespace is still the answer.
        let held = std::mem::take(&mut self.held_whitespace);
        if self.calls.is_empty() {
            push_text(&mut out, &held);
        }
        out
    }

    /// The calls found so far, with their final arguments.
    pub fn calls(&self) -> Vec<ParsedToolCall> {
        self.calls
            .iter()
            .map(|call| ParsedToolCall {
                name: call.name.clone(),
                arguments: call.done.clone().unwrap_or_else(|| call.args.clone()),
                id: call.id.clone(),
            })
            .collect()
    }

    /// Whether any tool call has started.
    pub fn has_calls(&self) -> bool {
        !self.calls.is_empty()
    }

    fn drain(&mut self, finished: bool, out: &mut Vec<ToolEvent>) {
        let Some(grammar) = self.grammar else {
            return;
        };
        loop {
            let Some(start) = self.region else {
                let rest = &self.text[self.pos..];
                match self.find_marker(grammar, rest, finished) {
                    Found::At(at) => {
                        let text = rest[..at].to_string();
                        self.prose(&text, out);
                        self.at_start = false;
                        self.pos += at;
                        self.region = Some(self.pos);
                        self.region_first = self.calls.len();
                        continue;
                    }
                    Found::Hold(keep) => {
                        let ready = rest.len() - keep;
                        let text = rest[..ready].to_string();
                        self.prose(&text, out);
                        self.pos += ready;
                        return;
                    }
                }
            };
            if let Some(from) = self.broken {
                match broken_end(grammar, &self.text[start..], from, self.tools, finished) {
                    Some(end) => {
                        self.pos = start + end;
                        self.region = None;
                        self.broken = None;
                        continue;
                    }
                    None => return,
                }
            }
            let cx = Cx {
                tools: self.tools,
                finished,
            };
            let scan = grammar.scan(&self.text[start..], &cx);
            match scan {
                Scan::Pending if !finished => return,
                // Not a call, or no longer one.  A call that had begun in the
                // region stands as far as it was read, as a stream fed in
                // smaller pieces would have sent it, and the rest of the
                // region is broken markup, kept out of the prose.  A grammar
                // that cannot decide at the end has no call.
                Scan::Pending | Scan::Text(_) => {
                    let (calls, good) =
                        calls_before_break(grammar, &self.text[start..], self.tools);
                    self.sync(calls, out);
                    if self.calls.len() > self.region_first {
                        self.break_off(good);
                        continue;
                    }
                    let n = match scan {
                        Scan::Text(n) => n,
                        _ => self.text.len() - start,
                    };
                    self.release(start, n, out);
                }
                Scan::Calls { calls, end } => {
                    // A call that began in the region and that the scan no
                    // longer returns broke off too.
                    let lost = calls.len() < self.calls.len() - self.region_first;
                    self.sync(calls, out);
                    if lost {
                        let read = end.filter(|&n| n > 0);
                        self.break_off(read.unwrap_or(self.text.len() - start));
                        continue;
                    }
                    match end {
                        Some(n) if n > 0 => {
                            self.pos = start + n;
                            self.region = None;
                        }
                        _ => {
                            if finished {
                                self.pos = self.text.len();
                                self.region = None;
                            }
                            return;
                        }
                    }
                }
            }
        }
    }

    fn compact(&mut self) {
        let consumed = self.region.unwrap_or(self.pos);
        if consumed == 0 {
            return;
        }
        if consumed == self.text.len() {
            self.text = String::new();
        } else if consumed >= 4096 && consumed >= self.text.len() / 2 {
            self.text = self.text[consumed..].to_string();
        } else {
            self.text.drain(..consumed);
        }
        self.pos -= consumed;
        if let Some(start) = &mut self.region {
            *start -= consumed;
        }
    }

    /// The open region's markup broke off `from` bytes in, after a call in
    /// it had begun: the calls sent stand, and the rest of the region is
    /// dropped.
    fn break_off(&mut self, from: usize) {
        tracing::debug!("tool-call markup broke off after a call began; dropping the rest");
        self.broken = Some(from);
    }

    /// Release the first `n` bytes of the region at `start` as prose.
    fn release(&mut self, start: usize, n: usize, out: &mut Vec<ToolEvent>) {
        // Always make progress past at least one character.
        let region = &self.text[start..];
        let first = region.chars().next().map_or(0, char::len_utf8);
        let mut n = n.max(first).min(region.len());
        while !region.is_char_boundary(n) {
            n += 1;
        }
        let text = region[..n].to_string();
        self.prose(&text, out);
        self.pos = start + n;
        self.region = None;
    }

    fn prose(&mut self, text: &str, out: &mut Vec<ToolEvent>) {
        if text.is_empty() {
            return;
        }
        if self.hold_whitespace {
            if text.trim().is_empty() {
                self.held_whitespace.push_str(text);
                return;
            }
            self.hold_whitespace = false;
            let held = std::mem::take(&mut self.held_whitespace);
            push_text(out, &held);
        }
        self.at_start = false;
        push_text(out, text);
    }

    /// Emit what changed in the open region's calls.
    fn sync(&mut self, calls: Vec<PartialCall>, out: &mut Vec<ToolEvent>) {
        for (offset, call) in calls.into_iter().enumerate() {
            let index = self.region_first + offset;
            if index == self.calls.len() {
                if call.name.is_empty() {
                    // Later calls cannot start before this one.
                    return;
                }
                // Formats that may add a late model id and Mistral's JSON
                // list cannot start until that id is known.
                let region_has_mistral_ids = self.region.is_some_and(|start| {
                    self.text[start..]
                        .trim_start()
                        .strip_prefix("[TOOL_CALLS]")
                        .is_some_and(|rest| rest.trim_start().starts_with('['))
                });
                let format_can_supply_ids = self.format.is_some_and(|format| {
                    format.may_supply_late_id()
                });
                if call.id.is_none()
                    && call.done.is_none()
                    && (region_has_mistral_ids || format_can_supply_ids)
                {
                    return;
                }
                self.held_whitespace.clear();
                self.hold_whitespace = true;
                out.push(ToolEvent::CallStart {
                    index,
                    id: call.id.clone(),
                    name: call.name.clone(),
                });
                self.calls.push(Sent {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    args: String::new(),
                    done: None,
                });
            }
            let sent = &mut self.calls[index];
            // Generic JSON can put an optional id after arguments have
            // started streaming; retain it in the final parsed call.
            if sent.id.is_none() && call.id.is_some() {
                sent.id = call.id.clone();
            }
            // Once the call is complete, final arguments that extend what
            // was streamed (arguments only readable whole: a string-encoded
            // object, text that is not JSON) are streamed too.
            let args = match &call.done {
                Some(done) if done.len() > call.args.len() && done.starts_with(&sent.args) => {
                    done.as_str()
                }
                _ => call.args.as_str(),
            };
            if args.len() > sent.args.len() && args.starts_with(&sent.args) {
                out.push(ToolEvent::CallArgs {
                    index,
                    delta: args[sent.args.len()..].to_string(),
                });
                sent.args = args.to_string();
            } else if !args.starts_with(&sent.args) {
                tracing::debug!(
                    "tool-call arguments for {} changed after streaming; keeping the streamed text",
                    sent.name
                );
            }
            if call.done.is_some() {
                sent.done = call.done;
            }
        }
    }

    fn find_marker(&self, grammar: &dyn Grammar, rest: &str, finished: bool) -> Found {
        let markers = grammar.markers();
        // A leading marker can only open the output itself.
        if self.at_start {
            let body = rest.trim_start();
            let leading = || markers.iter().filter(|m| m.leading);
            if leading().any(|marker| body.starts_with(marker.text)) {
                return Found::At(0);
            }
            if !finished && leading().any(|marker| marker.text.starts_with(body)) {
                return Found::Hold(rest.len());
            }
        }
        let anywhere: Vec<&str> = markers
            .iter()
            .filter(|m| !m.leading)
            .map(|m| m.text)
            .collect();
        if let Some((at, _)) = super::find_first(rest, &anywhere) {
            return Found::At(at);
        }
        if finished {
            return Found::Hold(0);
        }
        Found::Hold(super::partial_suffix(rest, anywhere.iter().copied()))
    }
}

enum Found {
    /// A marker starts at this offset.
    At(usize),
    /// No marker; the last this many bytes may begin one.
    Hold(usize),
}

/// The marker a region starts with, and the bytes up to its end (a leading
/// marker may follow whitespace).
fn opening(grammar: &dyn Grammar, region: &str) -> Option<(Marker, usize)> {
    let indent = region.len() - region.trim_start().len();
    grammar
        .markers()
        .iter()
        .filter_map(|marker| {
            let at = if marker.leading { indent } else { 0 };
            region[at..]
                .starts_with(marker.text)
                .then_some((*marker, at + marker.text.len()))
        })
        .max_by_key(|&(_, end)| end)
}

/// The calls in the longest prefix of `region` that, read while decoding is
/// still going, scans as calls, and that prefix's length: what a stream fed
/// one character at a time would have sent before the markup broke off.
/// `region` itself no longer scans as calls.  Prefixes scan as pending, then
/// as calls, then as text, so the break is found by bisection.
fn calls_before_break(
    grammar: &dyn Grammar,
    region: &str,
    tools: &[Tool],
) -> (Vec<PartialCall>, usize) {
    let cx = Cx {
        tools,
        finished: false,
    };
    let scan = |len: usize| grammar.scan(&region[..len], &cx);
    match scan(region.len()) {
        // Broken only because decoding ended.
        Scan::Calls { calls, .. } => return (calls, region.len()),
        Scan::Pending => return (Vec::new(), region.len()),
        Scan::Text(_) => {}
    }
    // `good` scans as something other than text, `bad` as text.
    let Some((_, mut good)) = opening(grammar, region) else {
        return (Vec::new(), region.len());
    };
    let mut bad = region.len();
    loop {
        let mut mid = good + (bad - good) / 2;
        while mid > good && !region.is_char_boundary(mid) {
            mid -= 1;
        }
        if mid == good {
            mid = (good + 1..bad)
                .find(|&at| region.is_char_boundary(at))
                .unwrap_or(bad);
        }
        if mid >= bad {
            break;
        }
        match scan(mid) {
            Scan::Text(_) => bad = mid,
            _ => good = mid,
        }
    }
    match scan(good) {
        Scan::Calls { calls, .. } => (calls, good),
        _ => (Vec::new(), good),
    }
}

/// Where a region whose markup broke off at `from` ends: after the first
/// closing string of the marker that opened it, or at the end of the output
/// once decoding has ended without one.  `None` while it may still arrive.
fn broken_end(
    grammar: &dyn Grammar,
    region: &str,
    from: usize,
    tools: &[Tool],
    finished: bool,
) -> Option<usize> {
    let close = opening(grammar, region).map_or(&[][..], |(marker, _)| marker.close);
    let close_at =
        super::find_first(&region[from..], close).map(|(at, marker)| (from + at, marker.len()));
    let cx = Cx {
        tools,
        finished: true,
    };
    let next_at = grammar
        .markers()
        .iter()
        .filter(|marker| !marker.close.is_empty())
        .flat_map(|marker| {
            region[from..]
                .match_indices(marker.text)
                .map(move |(offset, _)| (marker, from + offset))
        })
        .filter_map(|(marker, at)| {
            let previous = region[..at].chars().next_back();
            if marker.leading
                && previous.is_some_and(|c| !c.is_whitespace() && !matches!(c, '}' | ';'))
            {
                return None;
            }
            match grammar.scan(&region[at..], &cx) {
                Scan::Calls { calls, .. } if !calls.is_empty() => Some(at),
                _ => None,
            }
        })
        .min();
    match (next_at, close_at) {
        (Some(next), Some((close, _))) if next < close => Some(next),
        (Some(next), None) => Some(next),
        (_, Some((close, len))) => Some(close + len),
        (None, None) if finished => Some(region.len()),
        (None, None) => None,
    }
}

/// Append prose, merging with a preceding text event.
fn push_text(out: &mut Vec<ToolEvent>, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Some(ToolEvent::Text(last)) = out.last_mut() {
        last.push_str(text);
    } else {
        out.push(ToolEvent::Text(text.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(stream: &mut ToolCallStream<'_>, pieces: &[&str]) -> Vec<ToolEvent> {
        let mut events = Vec::new();
        for piece in pieces {
            events.extend(stream.push(piece));
        }
        events.extend(stream.finish());
        events
    }

    fn prose(events: &[ToolEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                ToolEvent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn hides_a_tool_marker_split_across_chunks() {
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let events = collect(
            &mut stream,
            &[
                "I will call ",
                "<tool",
                "_call>{\"name\":",
                "\"f\",\"arguments\":{}}</tool_call>",
            ],
        );
        assert_eq!(prose(&events), "I will call ");
        let calls = stream.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn forwards_everything_without_tools() {
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, None);
        let events = collect(&mut stream, &["a ", "<tool_call>", "c"]);
        assert_eq!(prose(&events), "a <tool_call>c");
        assert!(!stream.has_calls());
    }

    #[test]
    fn discards_consumed_prose_before_the_next_stream_chunk() {
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let text = "a".repeat(8192);
        assert_eq!(stream.push(&text), vec![ToolEvent::Text(text)]);
        assert!(stream.text.is_empty());
        assert!(stream.text.capacity() < 8192);

        let events = collect(
            &mut stream,
            &[r#"<tool_call>{"name":"f","arguments":{}}</tool_call>"#],
        );
        assert_eq!(stream.calls().len(), 1);
        assert_eq!(prose(&events), "");
    }

    #[test]
    fn resumes_prose_after_a_tool_call() {
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let events = collect(
            &mut stream,
            &[
                "Checking ",
                "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>",
                " Done",
            ],
        );
        assert_eq!(prose(&events), "Checking  Done");
        assert_eq!(stream.calls().len(), 1);
    }

    #[test]
    fn does_not_duplicate_a_bare_json_tool_call() {
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let events = collect(
            &mut stream,
            &["{\"name\":\"lookup\",", "\"arguments\":{\"q\":\"rust\"}}"],
        );
        assert_eq!(prose(&events), "");
        let calls = stream.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
    }

    #[test]
    fn releases_plain_json_instead_of_dropping_it() {
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let events = collect(&mut stream, &["{\"answer\":", "42}"]);
        assert_eq!(prose(&events), "{\"answer\":42}");
        assert!(!stream.has_calls());
    }

    #[test]
    fn does_not_swallow_prose_after_an_invalid_marker() {
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let events = collect(
            &mut stream,
            &["before ", "<tool_call>not a call</tool_call>", " after"],
        );
        assert_eq!(
            prose(&events),
            "before <tool_call>not a call</tool_call> after"
        );
        assert!(!stream.has_calls());
    }

    /// OpenAI-style deltas: the call starts once its name is known, then its
    /// arguments arrive in pieces as they are generated, before the call has
    /// closed.
    #[test]
    fn streams_arguments_as_they_are_generated() {
        let mut stream = ToolCallStream::new(ToolCallFormat::Mistral, Some(&[]));
        let mut events = Vec::new();
        for piece in [
            "Let me look.",
            "[TOOL_CALLS]get_",
            "weather[ARGS]{\"ci",
            "ty\": \"Par",
            "is\"}",
        ] {
            events.push(stream.push(piece));
        }
        events.push(stream.finish());
        assert_eq!(events[0], vec![ToolEvent::Text("Let me look.".into())]);
        assert!(events[1].is_empty(), "{:?}", events[1]);
        assert_eq!(
            events[2],
            vec![
                ToolEvent::CallStart {
                    index: 0,
                    id: None,
                    name: "get_weather".into()
                },
                ToolEvent::CallArgs {
                    index: 0,
                    delta: "{\"ci".into()
                },
            ]
        );
        assert_eq!(
            events[3],
            vec![ToolEvent::CallArgs {
                index: 0,
                delta: "ty\": \"Par".into()
            }]
        );
        assert_eq!(
            events[4],
            vec![ToolEvent::CallArgs {
                index: 0,
                delta: "is\"}".into()
            }]
        );
        assert_eq!(stream.calls()[0].arguments, "{\"city\":\"Paris\"}");
    }

    #[test]
    fn mistral_stream_waits_for_a_late_id_but_ignores_split_non_string_ids() {
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let first = stream.push(r#"[TOOL_CALLS][{"name":"weather","arguments":{"city":"Paris"},"#);
        assert!(first.is_empty());
        let last = stream.push(r#""id":"a1b2c3d4e"}]"#);
        assert_eq!(
            last,
            vec![
                ToolEvent::CallStart {
                    index: 0,
                    id: Some("a1b2c3d4e".into()),
                    name: "weather".into(),
                },
                ToolEvent::CallArgs {
                    index: 0,
                    delta: r#"{"city":"Paris"}"#.into(),
                },
            ]
        );

        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        assert!(
            stream
                .push(r#"[TOOL_CALLS][{"name":"weather","arguments":{"city":"Paris"},"id":1"#)
                .is_empty()
        );
        let events = stream.push("2}]");
        assert!(matches!(
            events.first(),
            Some(ToolEvent::CallStart { id: None, name, .. }) if name == "weather"
        ));
        assert_eq!(stream.calls().len(), 1);
        assert_eq!(stream.calls()[0].arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn generic_json_streams_arguments_before_a_late_optional_id() {
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let first = stream.push(
            r#"<tool_call>{"name":"weather","arguments":{"city":"Pa"#,
        );
        assert!(first.iter().any(|event| matches!(
            event,
            ToolEvent::CallStart {
                id: None,
                name,
                ..
            } if name == "weather"
        )));
        assert!(first.iter().any(|event| matches!(
            event,
            ToolEvent::CallArgs { delta, .. } if delta == r#"{"city":"Pa"#
        )));

        stream.push(r#"ris"},"id":"model-123"}</tool_call>"#);
        assert_eq!(stream.calls()[0].id.as_deref(), Some("model-123"));
        assert_eq!(stream.calls()[0].arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn llama_json_waits_for_a_late_model_id() {
        let mut stream = ToolCallStream::new(ToolCallFormat::LlamaJson, Some(&[]));
        assert!(
            stream
                .push(r#"{"name":"weather","parameters":{"city":"Paris"},"#)
                .is_empty()
        );
        assert!(stream
            .push(r#""id":"model-456"}"#)
            .iter()
            .any(|event| matches!(
                event,
                ToolEvent::CallStart { id: Some(id), .. } if id == "model-456"
            )));
        assert_eq!(stream.calls()[0].id.as_deref(), Some("model-456"));
    }

    #[test]
    fn mistral_inline_arguments_preserve_spaces_across_chunks() {
        let mut stream = ToolCallStream::new(ToolCallFormat::Mistral, Some(&[]));
        let mut streamed = String::new();
        for piece in [r#"[TOOL_CALLS]write_file[ARGS]{"text":"hello "#, "world\"}"] {
            for event in stream.push(piece) {
                if let ToolEvent::CallArgs { delta, .. } = event {
                    streamed.push_str(&delta);
                }
            }
        }
        stream.finish();
        assert_eq!(streamed, stream.calls()[0].arguments);
        assert_eq!(streamed, r#"{"text":"hello world"}"#);
    }

    #[test]
    fn malformed_call_does_not_consume_the_next_call() {
        for text in [
            r#"<tool_call>{"name":"f","arguments":{'x':1}}<tool_call>{"name":"g","arguments":{}}</tool_call>"#,
            r#"<tool_call>{"name":"f","arguments":{'x':1}<tool_call>{"name":"g","arguments":{}}</tool_call>"#,
        ] {
            let (_, parsed) = super::super::parse_tool_calls(ToolCallFormat::Generic, &[], text);
            assert_eq!(parsed.len(), 2, "{text}: {parsed:?}");
            assert_eq!(parsed[0].name, "f");
            assert_eq!(parsed[1].name, "g");

            for one_byte_at_a_time in [false, true] {
                let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
                if one_byte_at_a_time {
                    for ch in text.chars() {
                        stream.push(&ch.to_string());
                    }
                } else {
                    stream.push(text);
                }
                stream.finish();
                let calls = stream.calls();
                assert_eq!(calls.len(), 2, "{text}");
                assert_eq!(calls[0].name, "f");
                assert_eq!(calls[1].name, "g");
            }
        }
    }

    #[test]
    fn drops_whitespace_before_a_call_but_keeps_it_in_plain_text() {
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let events = collect(
            &mut stream,
            &["\n\n", "<tool_call>{\"name\":\"f\"}</tool_call>"],
        );
        assert_eq!(prose(&events), "");
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let events = collect(&mut stream, &["\n\n", "Hi"]);
        assert_eq!(prose(&events), "\n\nHi");
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let events = collect(&mut stream, &[" \n"]);
        assert_eq!(prose(&events), " \n");
    }

    /// A grammar whose `<x>` region reads a call `f` while it is `<x>f`
    /// and no call once more follows: a call the scan stops returning
    /// after the stream has sent it.
    struct Forgetful;

    impl Grammar for Forgetful {
        fn markers(&self) -> &'static [Marker] {
            const MARKERS: &[Marker] = &[Marker::anywhere("<x>").closed_by(&["</x>"])];
            MARKERS
        }

        fn scan(&self, region: &str, _: &Cx<'_>) -> Scan {
            let calls = if region == "<x>f" || region.starts_with("<x>f</x>") {
                vec![PartialCall {
                    name: "f".into(),
                    ..PartialCall::default()
                }]
            } else {
                Vec::new()
            };
            Scan::Calls {
                calls,
                end: if region.starts_with("<x>f</x>") {
                    Some("<x>f</x>".len())
                } else {
                    (region.len() > "<x>f".len()).then_some("<x>".len())
                },
            }
        }
    }

    /// The call sent stands and the rest of its region is dropped, as
    /// when a region stops scanning as calls.
    #[test]
    fn a_call_the_scan_loses_stands_and_its_region_is_dropped() {
        static FORGETFUL: Forgetful = Forgetful;
        let mut stream = ToolCallStream::with_grammar(Some(&FORGETFUL), &[]);
        let events = collect(&mut stream, &["<x>f", "(broken</x> after"]);
        assert_eq!(prose(&events), " after");
        assert_eq!(stream.calls().len(), 1);
        assert_eq!(stream.calls()[0].name, "f");

        let mut stream = ToolCallStream::with_grammar(Some(&FORGETFUL), &[]);
        let events = collect(&mut stream, &["<x>f", "(broken<x>f</x>"]);
        assert_eq!(prose(&events), "");
        assert_eq!(stream.calls().len(), 2);
    }

    /// No whitespace-only prose is sent between or after calls; prose
    /// after a call keeps the whitespace before it.
    #[test]
    fn drops_whitespace_between_and_after_calls() {
        let call = |name: &str| format!("<tool_call>{{\"name\":\"{name}\"}}</tool_call>");
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let events = collect(
            &mut stream,
            &["Both.\n", &call("a"), "\n", &call("b"), "\n"],
        );
        assert_eq!(prose(&events), "Both.\n");
        assert_eq!(stream.calls().len(), 2);
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let events = collect(&mut stream, &[&call("a"), "\n\n", "Done."]);
        assert_eq!(prose(&events), "\n\nDone.");
    }
}
