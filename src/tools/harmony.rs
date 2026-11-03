//! gpt-oss harmony: a call is a message addressed to `functions.<name>`,
//! with the recipient in the role header or in the channel header:
//!
//! ```text
//! <|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"city": "Paris"}<|call|>
//! <|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city": "Paris"}<|call|>
//! ```
//!
//! Reasoning (the `analysis` channel) is split off before tool calls are
//! parsed (see [`crate::reasoning`]); what reaches this grammar is the
//! answer, where it reads every message header: a `functions.*` recipient
//! is a call, another recipient (a built-in tool such as `python`) is
//! dropped as llama.cpp drops it, and the `final` and `commentary` headers
//! around plain text are removed so only the text remains.  An `analysis`
//! message left in the answer (reasoning shown inline) stays as written.

use super::json;
use super::{Cx, Grammar, Marker, PartialCall, Scan};

const START: &str = "<|start|>";
const CHANNEL: &str = "<|channel|>";
const MESSAGE: &str = "<|message|>";
const ENDS: &[&str] = &["<|call|>", "<|end|>", "<|return|>"];
const FUNCTIONS: &str = "functions.";
/// Longer than any message header: text this long without `<|message|>`
/// is not one.
const MAX_HEADER: usize = 256;

/// gpt-oss.
pub(super) struct Harmony;

pub(super) static HARMONY: Harmony = Harmony;

impl Grammar for Harmony {
    fn markers(&self) -> &'static [Marker] {
        const MARKERS: &[Marker] = &[
            Marker::anywhere(START),
            Marker::anywhere(CHANNEL),
            Marker::anywhere(ENDS[0]),
            Marker::anywhere(ENDS[1]),
            Marker::anywhere(ENDS[2]),
            // The prompt ends with `<|start|>assistant`, so a call addressed
            // in the role header starts the output with ` to=functions.…`.
            Marker::leading("to="),
        ];
        MARKERS
    }

    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan {
        if let Some(end) = ENDS.iter().find(|end| region.starts_with(**end)) {
            // A message's end marker: not part of the text.
            return Scan::Calls {
                calls: Vec::new(),
                end: Some(end.len()),
            };
        }
        let marker = [START, CHANNEL]
            .into_iter()
            .find(|marker| region.starts_with(marker))
            .map_or_else(|| json::skip_ws(region, 0) + "to=".len(), str::len);
        let Some(header_len) = region.find(MESSAGE) else {
            let framed = region.starts_with(START) || region.starts_with(CHANNEL);
            return if region.contains('\n') || region.len() > MAX_HEADER {
                Scan::Text(marker)
            } else if !cx.finished {
                Scan::Pending
            } else if framed {
                // An unfinished header is not a message: drop it.
                Scan::Calls {
                    calls: Vec::new(),
                    end: Some(region.len()),
                }
            } else {
                Scan::Text(marker)
            };
        };
        let header = &region[..header_len];
        if header.contains('\n') {
            return Scan::Text(marker);
        }
        let body_at = header_len + MESSAGE.len();
        let body = &region[body_at..];
        match recipient(header) {
            Some(to) if to.starts_with(FUNCTIONS) && to.len() > FUNCTIONS.len() => {
                let raw = json::raw_args(body, ENDS, cx.finished);
                let call = PartialCall {
                    id: None,
                    name: to[FUNCTIONS.len()..].to_string(),
                    args: raw.stream,
                    done: raw.done,
                };
                Scan::Calls {
                    calls: vec![call],
                    end: raw.end.map(|end| body_at + end),
                }
            }
            // A built-in tool's call: nothing for the client.
            Some(to) if to != "assistant" => Scan::Calls {
                calls: Vec::new(),
                end: message_end(body, cx).map(|end| body_at + end),
            },
            _ if channel(header) == Some("analysis") => match message_end(body, cx) {
                Some(end) => Scan::Text(body_at + end),
                None => Scan::Pending,
            },
            // A `final` or `commentary` message: keep its text.
            _ => Scan::Calls {
                calls: Vec::new(),
                end: Some(body_at),
            },
        }
    }
}

/// Where a message body ends: after its end marker, or at the end of the
/// output once decoding has finished.
fn message_end(body: &str, cx: &Cx<'_>) -> Option<usize> {
    match super::find_first(body, ENDS) {
        Some((at, end)) => Some(at + end.len()),
        None => cx.finished.then_some(body.len()),
    }
}

/// The header's `to=` recipient.
fn recipient(header: &str) -> Option<&str> {
    let mut rest = header;
    while let Some(at) = rest.find("to=") {
        let before = &rest[..at];
        let after = &rest[at + "to=".len()..];
        if before.is_empty() || before.ends_with(char::is_whitespace) || before.ends_with('>') {
            let end = after
                .find(|c: char| c.is_whitespace() || c == '<')
                .unwrap_or(after.len());
            return Some(&after[..end]);
        }
        rest = after;
    }
    None
}

/// The header's channel name.
fn channel(header: &str) -> Option<&str> {
    let at = header.rfind(CHANNEL)?;
    let rest = &header[at + CHANNEL.len()..];
    rest.split(|c: char| c.is_whitespace() || c == '<').next()
}

#[cfg(test)]
mod tests {
    use super::super::tests::{args, assert_streams_like_parse, parse};
    use super::super::ToolCallFormat;
    use serde_json::json;

    const H: ToolCallFormat = ToolCallFormat::Harmony;

    /// llama.cpp's gpt-oss cases, with the analysis already split off.
    #[test]
    fn calls_addressed_in_either_header() {
        for text in [
            " to=functions.special_function<|channel|>analysis<|message|>{\"arg1\": 1}",
            "<|channel|>analysis to=functions.special_function<|message|>{\"arg1\": 1}",
            " to=functions.special_function<|channel|>analysis <|constrain|>json<|message|>{\"arg1\": 1}",
            "<|channel|>commentary to=functions.special_function<|message|>{\"arg1\": 1}",
            "<|start|>assistant to=functions.special_function<|channel|>commentary json<|message|>{\"arg1\": 1}<|call|>",
        ] {
            let (content, calls) = parse(H, text);
            assert_eq!(content, "", "{text}");
            assert_eq!(calls.len(), 1, "{text}");
            assert_eq!(calls[0].name, "special_function");
            assert_eq!(calls[0].arguments, r#"{"arg1":1}"#);
            assert_streams_like_parse(H, &[], text);
        }
    }

    #[test]
    fn edit_call_after_reasoning() {
        let text = "<|start|>assistant<|channel|>commentary to=functions.edit <|constrain|>json\
                    <|message|>{\"oldString\": \"if (part < railCount - 1) {\", \"newString\": \"if (part < 4) {\", \"replaceAll\": false}";
        let (_, calls) = parse(H, text);
        assert_eq!(calls[0].name, "edit");
        assert_eq!(
            args(&calls[0]),
            json!({"oldString": "if (part < railCount - 1) {", "newString": "if (part < 4) {", "replaceAll": false})
        );
        assert_streams_like_parse(H, &[], text);
    }

    #[test]
    fn plain_messages_keep_only_their_text() {
        let (content, calls) = parse(H, "<|channel|>final<|message|>Hello, world!\nWhat's up?");
        assert_eq!(content, "Hello, world!\nWhat's up?");
        assert!(calls.is_empty());
        let text = "<|channel|>commentary<|message|>Checking.<|end|>\
                    <|start|>assistant<|channel|>commentary to=functions.f <|constrain|>json<|message|>{}<|call|>";
        let (content, calls) = parse(H, text);
        assert_eq!(content, "Checking.");
        assert_eq!(calls.len(), 1);
        assert_streams_like_parse(H, &[], text);
        // Reasoning shown inline stays as written.
        let (content, _) = parse(
            H,
            "<|channel|>analysis<|message|>I'm\nthinking<|end|><|start|>assistant<|channel|>final<|message|>Hi",
        );
        assert_eq!(
            content,
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>Hi"
        );
    }

    /// Calls to built-in tools are dropped (llama.cpp's "unsolicited" cases).
    #[test]
    fn builtin_tool_calls_are_dropped() {
        for text in [
            "<|start|>assistant to=container.exec<|channel|>commentary<|message|>python3 -c 'print(\"hello\")'",
            "<|start|>assistant<|channel|>commentary to=python <|constrain|>code<|message|>print(\"hello\")",
        ] {
            let (content, calls) = parse(H, text);
            assert_eq!(content, "", "{text}");
            assert!(calls.is_empty());
        }
    }

    #[test]
    fn prose_that_starts_like_a_recipient_stays_text() {
        let (content, calls) = parse(H, "to=do list: milk, eggs.\nThanks.");
        assert_eq!(content, "to=do list: milk, eggs.\nThanks.");
        assert!(calls.is_empty());
        let (content, _) = parse(H, "Today is fine.");
        assert_eq!(content, "Today is fine.");
    }
}
