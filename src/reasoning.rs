//! Splitting a reasoning model's output into its reasoning and its answer.
//!
//! Thinking models write their chain of thought before the answer, marked
//! up the way their chat template expects:
//!
//! * **Tags** — `<think>…</think>` (Qwen3, QwQ, DeepSeek-R1 and V3.1+,
//!   GLM-4.5+, Kimi-K2-Thinking, MiniMax-M2, Hunyuan), `<seed:think>…`
//!   (Seed-OSS), `[THINK]…[/THINK]` (Magistral) or
//!   `<|START_THINKING|>…<|END_THINKING|>` (Command-R7B).  Some templates
//!   open the block in the generation prompt (DeepSeek-R1's
//!   `<｜Assistant｜><think>\n`), so the output starts inside it; see
//!   [`ReasoningFormat::prompt_opens_reasoning`].
//! * **Harmony** — gpt-oss channels: `<|channel|>analysis<|message|>…<|end|>`
//!   is reasoning and `<|channel|>final<|message|>…` the answer.  Other
//!   channels (`commentary`, which carries tool calls), and any message
//!   addressed to a tool (`to=functions.f`, even on `analysis`), are passed
//!   through verbatim as answer text for the tool-call parser.
//!
//! [`ReasoningSplitter`] works on text as it is decoded: it holds back only
//! a possible partial marker, so each piece of reasoning or answer is
//! forwarded as soon as it is unambiguous.  [`split`] is the one-shot form
//! for a complete output; streaming and one-shot results are identical.
//!
//! Reasoning keeps its own text, minus the whitespace between it and its
//! tags; the answer after a reasoning block starts at its first
//! non-whitespace character, as llama-server and Qwen's own parser report
//! it.

use crate::tools::harmony::{
    self, CHANNEL as H_CHANNEL, ENDS as H_ENDS, MESSAGE as H_MESSAGE, START as H_START,
};
use crate::tools::{find_first, partial_suffix};

/// How a model marks its reasoning in generated text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningFormat {
    /// No reasoning markup: everything is answer text.
    None,
    /// Reasoning between a start and an end tag.
    Tags {
        /// The opening tag, e.g. `<think>`.
        start: String,
        /// The closing tag, e.g. `</think>`.
        end: String,
    },
    /// gpt-oss harmony channels.
    Harmony,
}

/// The tag pairs [`ReasoningFormat::detect`] knows, most specific first.
const TAG_PAIRS: [(&str, &str); 4] = [
    ("<seed:think>", "</seed:think>"),
    ("<|START_THINKING|>", "<|END_THINKING|>"),
    ("[THINK]", "[/THINK]"),
    ("<think>", "</think>"),
];

impl ReasoningFormat {
    /// Work out a model's format from its chat template source and its
    /// vocabulary (`is_token` reports whether a string is a single token).
    ///
    /// Harmony needs a template that writes `<|channel|>`; a tag pair needs
    /// its closing tag to be a vocabulary token or to appear in the
    /// template, so plain models whose output merely mentions `<think>` are
    /// left alone.
    pub fn detect(template: Option<&str>, is_token: &dyn Fn(&str) -> bool) -> Self {
        let template = template.unwrap_or("");
        if template.contains(H_CHANNEL) && is_token(H_CHANNEL) {
            return Self::Harmony;
        }
        for (start, end) in TAG_PAIRS {
            if is_token(end) || template.contains(end) {
                return Self::Tags {
                    start: start.to_string(),
                    end: end.to_string(),
                };
            }
        }
        Self::None
    }

    /// Whether a rendered prompt leaves a reasoning block open, so the
    /// model's output starts inside it: the last opening tag comes after
    /// the last closing one (DeepSeek-R1 and V3.1's `<think>` generation
    /// prompt, GLM's forced thinking), or a harmony prompt ends inside an
    /// `analysis` message.
    pub fn prompt_opens_reasoning(&self, prompt: &str) -> bool {
        match self {
            Self::None => false,
            Self::Tags { start, end } => {
                match (prompt.rfind(start.as_str()), prompt.rfind(end.as_str())) {
                    (Some(s), Some(e)) => s > e,
                    (Some(_), None) => true,
                    _ => false,
                }
            }
            Self::Harmony => {
                let tail = prompt.trim_end();
                tail.ends_with(&format!("{H_CHANNEL}analysis{H_MESSAGE}"))
            }
        }
    }

    /// Whether the format has reasoning at all.
    pub fn is_some(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// How a response reports the model's reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningOutput {
    /// Separate from the answer: `reasoning_content` / `reasoning` on
    /// OpenAI routes, `thinking` blocks on Anthropic's (llama.cpp's
    /// `deepseek` / `auto`, vLLM's behaviour).
    #[default]
    Separate,
    /// Left inline in the answer text (llama.cpp's `none`).
    Inline,
    /// Separate in a complete response, inline while streaming (llama.cpp's
    /// `deepseek-legacy`).
    Legacy,
    /// Dropped from the response (vLLM's `include_reasoning: false`).
    Hidden,
}

impl ReasoningOutput {
    /// Whether output is split at all for a streamed (`streaming`) or
    /// complete response.
    pub fn splits(self, streaming: bool) -> bool {
        match self {
            Self::Separate | Self::Hidden => true,
            Self::Inline => false,
            Self::Legacy => !streaming,
        }
    }
}

/// Reasoning and answer text produced by one step of a
/// [`ReasoningSplitter`].  Within a step, the reasoning comes first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Split {
    /// Reasoning text.
    pub reasoning: String,
    /// Answer text.
    pub content: String,
}

impl Split {
    /// Whether the step produced nothing.
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_empty() && self.content.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Nothing decided yet: the output may open a reasoning block.
    Start,
    /// Inside a reasoning block.
    Reasoning,
    /// In the answer.  `after_reasoning` strips the whitespace a reasoning
    /// block leaves before the answer.
    Content { after_reasoning: bool },
    /// Harmony: reading a message header up to `<|message|>`.
    Header,
    /// Harmony: inside a message body.
    Body(Channel),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    Analysis,
    Final,
    /// Any other channel, forwarded verbatim with its markers.
    Raw,
}

/// Streaming reasoning/answer splitter for one generation.
#[derive(Debug, Clone)]
pub struct ReasoningSplitter {
    format: ReasoningFormat,
    state: State,
    /// Text received but not yet classified.
    buf: String,
    /// Reasoning whitespace held back until more reasoning follows (it is
    /// dropped if the block closes instead).
    pending_ws: String,
    /// Whether any reasoning text has been emitted (leading whitespace of a
    /// block is dropped).
    reasoning_started: bool,
}

impl ReasoningSplitter {
    /// A splitter for `format`; `in_reasoning` when the prompt already
    /// opened the reasoning block (see
    /// [`ReasoningFormat::prompt_opens_reasoning`]).
    pub fn new(format: ReasoningFormat, in_reasoning: bool) -> Self {
        let state = match (&format, in_reasoning) {
            (ReasoningFormat::None, _) => State::Content {
                after_reasoning: false,
            },
            (ReasoningFormat::Tags { .. }, true) => State::Reasoning,
            (ReasoningFormat::Tags { .. }, false) => State::Start,
            (ReasoningFormat::Harmony, true) => State::Body(Channel::Analysis),
            (ReasoningFormat::Harmony, false) => State::Header,
        };
        Self {
            format,
            state,
            buf: String::new(),
            pending_ws: String::new(),
            reasoning_started: false,
        }
    }

    /// A splitter that passes everything through as answer text.
    pub fn disabled() -> Self {
        Self::new(ReasoningFormat::None, false)
    }

    /// Whether the output is still inside reasoning (or undecided).
    pub fn in_reasoning(&self) -> bool {
        matches!(
            self.state,
            State::Reasoning | State::Body(Channel::Analysis)
        )
    }

    /// Feed newly decoded text; returns what can be classified now.
    pub fn push(&mut self, text: &str) -> Split {
        let mut out = Split::default();
        if let State::Content {
            after_reasoning: false,
        } = self.state
        {
            // The common case once decided: no buffering at all.
            out.content.push_str(text);
            return out;
        }
        self.buf.push_str(text);
        self.drain(&mut out, false);
        out
    }

    /// Flush everything still held back once decoding has ended.
    pub fn finish(&mut self) -> Split {
        let mut out = Split::default();
        self.drain(&mut out, true);
        let rest = std::mem::take(&mut self.buf);
        match self.state {
            State::Start => out.content.push_str(&rest),
            State::Reasoning | State::Body(Channel::Analysis) => {
                self.emit_reasoning(&mut out, &rest, true)
            }
            State::Content { after_reasoning } => {
                if after_reasoning {
                    out.content.push_str(rest.trim_start());
                } else {
                    out.content.push_str(&rest);
                }
            }
            // An unfinished header is not a message; a raw body is kept.
            State::Header => {
                if !looks_like_header(&rest) {
                    out.content.push_str(&rest);
                }
            }
            State::Body(_) => out.content.push_str(&rest),
        }
        out
    }

    /// Emit reasoning text, holding back its trailing whitespace until more
    /// reasoning follows; `closing` (the block ended) drops it.
    fn emit_reasoning(&mut self, out: &mut Split, text: &str, closing: bool) {
        let text = if self.reasoning_started {
            text
        } else {
            text.trim_start()
        };
        let body = text.trim_end();
        if !body.is_empty() {
            out.reasoning
                .push_str(&std::mem::take(&mut self.pending_ws));
            out.reasoning.push_str(body);
            self.reasoning_started = true;
        }
        if closing {
            self.pending_ws.clear();
        } else {
            self.pending_ws.push_str(&text[body.len()..]);
        }
    }

    fn emit_content(&mut self, out: &mut Split, text: &str) {
        match self.state {
            State::Content {
                after_reasoning: true,
            } => {
                let trimmed = text.trim_start();
                if !trimmed.is_empty() {
                    out.content.push_str(trimmed);
                    self.state = State::Content {
                        after_reasoning: false,
                    };
                }
            }
            _ => out.content.push_str(text),
        }
    }

    fn drain(&mut self, out: &mut Split, flushing: bool) {
        loop {
            match (&self.format, self.state) {
                (_, State::Content { .. }) => {
                    let rest = std::mem::take(&mut self.buf);
                    self.emit_content(out, &rest);
                    return;
                }
                (ReasoningFormat::Tags { start, .. }, State::Start) => {
                    let trimmed = self.buf.trim_start();
                    if let Some(after) = trimmed.strip_prefix(start.as_str()) {
                        self.buf = after.to_string();
                        self.state = State::Reasoning;
                        continue;
                    }
                    if !flushing && start.starts_with(trimmed) {
                        // Possibly the start of the tag: wait.
                        return;
                    }
                    self.state = State::Content {
                        after_reasoning: false,
                    };
                }
                (ReasoningFormat::Tags { end, .. }, State::Reasoning) => {
                    let end = end.clone();
                    if let Some(at) = self.buf.find(&end) {
                        let body = self.buf[..at].to_string();
                        self.buf.drain(..at + end.len());
                        self.emit_reasoning(out, &body, true);
                        self.state = State::Content {
                            after_reasoning: true,
                        };
                        continue;
                    }
                    if flushing {
                        return;
                    }
                    let keep = partial_suffix(&self.buf, [end.as_str()]);
                    let ready = self.buf[..self.buf.len() - keep].to_string();
                    self.buf.drain(..self.buf.len() - keep);
                    self.emit_reasoning(out, &ready, false);
                    return;
                }
                (ReasoningFormat::Harmony, State::Header) => {
                    if let Some(at) = self.buf.find(H_MESSAGE) {
                        let header = self.buf[..at].to_string();
                        self.buf.drain(..at + H_MESSAGE.len());
                        // A message addressed to a tool (`to=functions.f`,
                        // in either header) is a call on whatever channel:
                        // it goes to the tool-call parser as written.
                        let channel = match harmony::channel(&header) {
                            _ if harmony::recipient(&header).is_some() => Channel::Raw,
                            Some("analysis") => Channel::Analysis,
                            Some("final") => Channel::Final,
                            _ => Channel::Raw,
                        };
                        if channel == Channel::Raw {
                            out.content.push_str(&header);
                            out.content.push_str(H_MESSAGE);
                        }
                        self.state = State::Body(channel);
                        continue;
                    }
                    if !looks_like_header(&self.buf) {
                        // Plain text without harmony framing: an answer.
                        self.state = State::Content {
                            after_reasoning: false,
                        };
                        continue;
                    }
                    return;
                }
                (ReasoningFormat::Harmony, State::Body(channel)) => {
                    if let Some((at, marker)) = find_first(&self.buf, H_ENDS) {
                        let body = self.buf[..at].to_string();
                        self.buf.drain(..at + marker.len());
                        match channel {
                            Channel::Analysis => self.emit_reasoning(out, &body, true),
                            Channel::Final => out.content.push_str(&body),
                            Channel::Raw => {
                                out.content.push_str(&body);
                                out.content.push_str(marker);
                            }
                        }
                        self.state = State::Header;
                        continue;
                    }
                    if flushing {
                        return;
                    }
                    let keep = partial_suffix(&self.buf, H_ENDS.iter().copied());
                    let ready = self.buf[..self.buf.len() - keep].to_string();
                    self.buf.drain(..self.buf.len() - keep);
                    match channel {
                        Channel::Analysis => self.emit_reasoning(out, &ready, false),
                        Channel::Final | Channel::Raw => out.content.push_str(&ready),
                    }
                    return;
                }
                // A tag state under another format cannot happen; treat the
                // rest as answer text.
                _ => {
                    self.state = State::Content {
                        after_reasoning: false,
                    };
                }
            }
        }
    }
}

/// Whether `text` is, or may become, a harmony message header.
fn looks_like_header(text: &str) -> bool {
    let t = text.trim_start();
    t.is_empty()
        || t.starts_with(H_START)
        || t.starts_with(H_CHANNEL)
        || H_START.starts_with(t)
        || H_CHANNEL.starts_with(t)
}

/// Split a complete output (see [`ReasoningSplitter`]).  Returns the
/// reasoning (`None` when there was none) and the answer.
pub fn split(format: &ReasoningFormat, in_reasoning: bool, text: &str) -> (Option<String>, String) {
    let mut splitter = ReasoningSplitter::new(format.clone(), in_reasoning);
    let mut all = splitter.push(text);
    let tail = splitter.finish();
    all.reasoning.push_str(&tail.reasoning);
    all.content.push_str(&tail.content);
    let reasoning = (!all.reasoning.is_empty()).then_some(all.reasoning);
    (reasoning, all.content)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn think() -> ReasoningFormat {
        ReasoningFormat::Tags {
            start: "<think>".into(),
            end: "</think>".into(),
        }
    }

    /// Feed `text` in every possible two-piece split and in single
    /// characters; every way must give the one-shot result.
    fn assert_streaming_matches(
        format: &ReasoningFormat,
        open: bool,
        text: &str,
    ) -> (Option<String>, String) {
        let whole = split(format, open, text);
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut cuts: Vec<Vec<usize>> = chars.iter().map(|&(i, _)| vec![i]).collect();
        cuts.push(chars.iter().map(|&(i, _)| i).collect());
        for cut in cuts {
            let mut s = ReasoningSplitter::new(format.clone(), open);
            let mut got = Split::default();
            let mut prev = 0;
            for &c in cut.iter().chain(std::iter::once(&text.len())) {
                let step = s.push(&text[prev..c]);
                got.reasoning.push_str(&step.reasoning);
                got.content.push_str(&step.content);
                prev = c;
            }
            let tail = s.finish();
            got.reasoning.push_str(&tail.reasoning);
            got.content.push_str(&tail.content);
            let got = (
                (!got.reasoning.is_empty()).then_some(got.reasoning),
                got.content,
            );
            assert_eq!(got, whole, "cuts {cut:?} of {text:?}");
        }
        whole
    }

    #[test]
    fn think_tags_split_reasoning_from_the_answer() {
        let (r, c) = assert_streaming_matches(
            &think(),
            false,
            "<think>\nThe user greets me.\nI greet back.\n</think>\n\nHello!",
        );
        assert_eq!(r.as_deref(), Some("The user greets me.\nI greet back."));
        assert_eq!(c, "Hello!");
    }

    #[test]
    fn a_prompt_opened_block_starts_in_reasoning() {
        // DeepSeek-R1: the template wrote `<think>\n`.
        let (r, c) =
            assert_streaming_matches(&think(), true, "Let me see.</think>The answer is 4.");
        assert_eq!(r.as_deref(), Some("Let me see."));
        assert_eq!(c, "The answer is 4.");
        // Cut off by the length limit inside the reasoning: all reasoning.
        let (r, c) = assert_streaming_matches(&think(), true, "Still thinking about <thi");
        assert_eq!(r.as_deref(), Some("Still thinking about <thi"));
        assert_eq!(c, "");
    }

    #[test]
    fn output_without_tags_is_all_answer() {
        let (r, c) = assert_streaming_matches(
            &think(),
            false,
            "  Plain <b>answer</b> with </think> inside",
        );
        assert_eq!(r, None);
        assert_eq!(c, "  Plain <b>answer</b> with </think> inside");
        // A lone `<` at the end that never becomes the tag.
        let (r, c) = assert_streaming_matches(&think(), false, "<");
        assert_eq!(r, None);
        assert_eq!(c, "<");
        let (r, c) = assert_streaming_matches(&ReasoningFormat::None, false, "<think>x</think>y");
        assert_eq!(r, None);
        assert_eq!(c, "<think>x</think>y");
    }

    #[test]
    fn empty_reasoning_blocks_report_no_reasoning() {
        // Qwen3 with thinking off writes an empty block itself sometimes.
        let (r, c) = assert_streaming_matches(&think(), false, "<think>\n\n</think>\n\nHi");
        assert_eq!(r, None);
        assert_eq!(c, "Hi");
    }

    #[test]
    fn multibyte_text_survives_any_split() {
        let (r, c) =
            assert_streaming_matches(&think(), false, "<think>日本語で考える</think>答え：はい");
        assert_eq!(r.as_deref(), Some("日本語で考える"));
        assert_eq!(c, "答え：はい");
    }

    #[test]
    fn other_tag_families() {
        let seed = ReasoningFormat::Tags {
            start: "<seed:think>".into(),
            end: "</seed:think>".into(),
        };
        let (r, c) = assert_streaming_matches(&seed, false, "<seed:think>hmm</seed:think>ok");
        assert_eq!((r.as_deref(), c.as_str()), (Some("hmm"), "ok"));
        let magistral = ReasoningFormat::Tags {
            start: "[THINK]".into(),
            end: "[/THINK]".into(),
        };
        let (r, c) = assert_streaming_matches(&magistral, false, "[THINK]a b[/THINK] c");
        assert_eq!((r.as_deref(), c.as_str()), (Some("a b"), "c"));
    }

    #[test]
    fn harmony_channels() {
        let text = "<|channel|>analysis<|message|>User says hi. Respond.<|end|>\
                    <|start|>assistant<|channel|>final<|message|>Hello there!";
        let (r, c) = assert_streaming_matches(&ReasoningFormat::Harmony, false, text);
        assert_eq!(r.as_deref(), Some("User says hi. Respond."));
        assert_eq!(c, "Hello there!");
        // A tool call on the commentary channel stays verbatim for the
        // tool-call parser.
        let text = "<|channel|>analysis<|message|>Need weather.<|end|>\
                    <|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>";
        let (r, c) = assert_streaming_matches(&ReasoningFormat::Harmony, false, text);
        assert_eq!(r.as_deref(), Some("Need weather."));
        assert_eq!(
            c,
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>"
        );
        // So does a call addressed on the analysis channel (llama.cpp's
        // gpt-oss cases), which is not reasoning.
        let text = "<|channel|>analysis<|message|>Need weather.<|end|>\
                    <|start|>assistant<|channel|>analysis to=functions.get_weather<|message|>{\"city\":\"Paris\"}<|call|>";
        let (r, c) = assert_streaming_matches(&ReasoningFormat::Harmony, false, text);
        assert_eq!(r.as_deref(), Some("Need weather."));
        let (prose, calls) =
            crate::tools::parse_tool_calls(crate::tools::ToolCallFormat::Harmony, &[], &c);
        assert_eq!(prose, "");
        assert_eq!(calls.len(), 1, "{c}");
        assert_eq!(calls[0].name, "get_weather");
        // Unframed text is an answer.
        let (r, c) = assert_streaming_matches(&ReasoningFormat::Harmony, false, "just text");
        assert_eq!((r, c.as_str()), (None, "just text"));
    }

    #[test]
    fn detection_and_prompt_state() {
        let qwen_vocab = |t: &str| t == "<think>" || t == "</think>";
        let none = |_: &str| false;
        assert_eq!(
            ReasoningFormat::detect(Some("{{ x }}"), &qwen_vocab),
            think()
        );
        assert_eq!(
            ReasoningFormat::detect(Some("{{ x }}"), &none),
            ReasoningFormat::None
        );
        assert_eq!(ReasoningFormat::detect(None, &none), ReasoningFormat::None);
        let harmony_vocab = |t: &str| t == "<|channel|>";
        assert_eq!(
            ReasoningFormat::detect(Some("<|start|>assistant<|channel|>final"), &harmony_vocab),
            ReasoningFormat::Harmony
        );
        // A template mention is enough even when the tag is not one token.
        assert_eq!(
            ReasoningFormat::detect(Some("... '</think>' ..."), &none),
            think()
        );

        let f = think();
        assert!(f.prompt_opens_reasoning("<｜User｜>hi<｜Assistant｜><think>\n"));
        assert!(!f.prompt_opens_reasoning("<|im_start|>assistant\n"));
        assert!(!f.prompt_opens_reasoning("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert!(!f.prompt_opens_reasoning("<think>old</think> new turn"));
        assert!(ReasoningFormat::Harmony
            .prompt_opens_reasoning("<|start|>assistant<|channel|>analysis<|message|>"));
        assert!(!ReasoningFormat::Harmony.prompt_opens_reasoning("<|start|>assistant"));
    }

    #[test]
    fn output_modes() {
        assert!(ReasoningOutput::Separate.splits(true));
        assert!(ReasoningOutput::Hidden.splits(false));
        assert!(!ReasoningOutput::Inline.splits(false));
        assert!(ReasoningOutput::Legacy.splits(false));
        assert!(!ReasoningOutput::Legacy.splits(true));
    }
}
