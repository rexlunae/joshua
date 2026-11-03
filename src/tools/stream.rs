//! The streaming tool-call state machine every chat API drives.
//!
//! [`ToolCallStream`] takes decoded text as it arrives and turns it into
//! [`ToolEvent`]s: prose as soon as no tool-call marker can begin in it, a
//! call's start once its name is known, then its arguments as they grow.
//! Partial markup never reaches the prose: a possible marker prefix at the
//! end of the text is held back, and a region that turns out not to be a
//! call is released as text.

use super::{Cx, Grammar, ParsedToolCall, PartialCall, Scan, ToolCallFormat};
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
    tools: &'a [Tool],
    text: String,
    /// Bytes of `text` already emitted or consumed.
    pos: usize,
    /// Start of the tool-call region being read.
    region: Option<usize>,
    /// Index in `calls` of the open region's first call.
    region_first: usize,
    calls: Vec<Sent>,
    /// Whether any prose has been sent.  Whitespace before the first prose
    /// is held: it is dropped if a tool call comes first.
    prose_started: bool,
    held_whitespace: String,
}

impl<'a> ToolCallStream<'a> {
    /// A stream for a model emitting `format`.  Without `tools`, markup is
    /// ordinary text and everything passes straight through.
    pub fn new(format: ToolCallFormat, tools: Option<&'a [Tool]>) -> Self {
        Self {
            grammar: tools.map(|_| format.grammar()),
            tools: tools.unwrap_or_default(),
            text: String::new(),
            pos: 0,
            region: None,
            region_first: 0,
            calls: Vec::new(),
            prose_started: false,
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
        out
    }

    /// Flush everything held back once decoding has ended.
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        let mut out = Vec::new();
        if self.grammar.is_none() {
            return out;
        }
        self.drain(true, &mut out);
        if self.calls.is_empty() && !self.prose_started {
            let held = std::mem::take(&mut self.held_whitespace);
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
            let cx = Cx {
                tools: self.tools,
                finished,
            };
            let scan = grammar.scan(&self.text[start..], &cx);
            match scan {
                // The markup broke off after a call from it had started:
                // keep the call as streamed and drop the rest of the markup
                // read so far rather than repeat it as prose.
                Scan::Pending | Scan::Text(_) if self.calls.len() > self.region_first => {
                    tracing::debug!("malformed tool-call markup after a call started; dropping it");
                    self.pos = self.text.len();
                    self.region = None;
                    if !finished {
                        return;
                    }
                }
                Scan::Pending if !finished => return,
                // A grammar that cannot decide at the end has no call.
                Scan::Pending => self.release(start, self.text.len() - start, out),
                Scan::Text(n) => self.release(start, n, out),
                Scan::Calls { calls, end } => {
                    self.sync(calls, out);
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
        if !self.prose_started {
            if text.trim().is_empty() {
                self.held_whitespace.push_str(text);
                return;
            }
            self.prose_started = true;
            let held = std::mem::take(&mut self.held_whitespace);
            push_text(out, &held);
        }
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
                self.held_whitespace.clear();
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
        if self.pos == 0 {
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
        let mut stream = ToolCallStream::new(ToolCallFormat::Generic, Some(&[]));
        let mut events = Vec::new();
        for piece in [
            "Let me look.",
            "<tool_call>\n{\"name\": \"get_",
            "weather\", \"arguments\": {\"ci",
            "ty\": \"Par",
            "is\"}}\n</tool_call>",
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
}
