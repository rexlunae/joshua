//! Tool-call parsing for model output.
//!
//! When a request supplies tool definitions, the chat template instructs the
//! model how to emit calls.  Each model family has its own markup;
//! [`ToolCallFormat`] names the one a model uses (chosen from its chat
//! template, then its architecture, or set with `--tool-call-parser`) and
//! this module normalises its calls to OpenAI-style `(name, arguments-JSON)`
//! pairs:
//!
//! - **Generic** (Hermes / Qwen / Qwen3-Coder / Mistral / bare JSON):
//!   `<tool_call>{"name": …, "arguments": …}</tool_call>`, Qwen3-Coder's
//!   `<function=name><parameter=key>value</parameter></function>` (wrapped
//!   in `<tool_call>` or bare; Functionary v3.1 puts a JSON object in it),
//!   Mistral's `[TOOL_CALLS][{…}]` and `[TOOL_CALLS]name[ARGS]{…}`, or a
//!   response that is one JSON object.
//! - **Mistral**: the generic reader, for models whose templates take back
//!   only 9-character call ids, so the ids minted for their calls are those.
//! - **Llama 3.x JSON**: `{"name": …, "parameters": …}`, `;`-separated,
//!   optionally after `<|python_tag|>`.
//! - **Pythonic** (Llama 3.2 / 4): `[get_weather(city="Paris")]`, and
//!   **LFM2**'s `<|tool_call_start|>[…]<|tool_call_end|>`.
//! - **Nemotron Nano v2**: `<TOOLCALL>[{…}]</TOOLCALL>`.
//! - **Phi-4-mini / Firefunction v2**: `functools[{…}]`.
//! - **DeepSeek V3 / R1 / V3.1**: `<｜tool▁calls▁begin｜>` sections.
//! - **DeepSeek V3.2 / V4 / V4.1**: DSML `<｜DSML｜invoke name="…">` blocks.
//! - **Kimi K2** (`<|tool_calls_section_begin|>`) and **Kimi K3**
//!   (`<|open|>tools<|sep|>`).
//! - **GLM-4.5 and later**: `<tool_call>name<arg_key>…</arg_key><arg_value>…`.
//! - **gpt-oss harmony**: a message addressed `to=functions.name`.
//! - **FunctionGemma**: `<start_function_call>call:name{key:<escape>…<escape>}`.
//!
//! Every format is a [`Grammar`]: the strings that open a tool-call region
//! and a scan of that region.  [`ToolCallStream`] drives a grammar over text
//! as it is decoded, forwarding prose at once and tool calls as OpenAI-style
//! deltas; [`parse_tool_calls`] runs the same stream over a complete output,
//! so streamed and complete responses agree.

mod args;
mod deepseek;
mod gemma;
mod generic;
mod glm;
mod harmony;
mod json;
mod kimi;
mod pythonic;
mod stream;

use std::fmt;
use std::str::FromStr;

use crate::model::Architecture;
use crate::types::Tool;

pub use stream::{ToolCallStream, ToolEvent};

/// The `to=` recipient of a gpt-oss harmony message header, which makes the
/// message a tool call (see [`crate::reasoning`]).
pub(crate) use harmony::recipient as harmony_recipient;

/// A tool call extracted from raw model output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    /// Function name.
    pub name: String,
    /// JSON-encoded argument object.
    pub arguments: String,
    /// The call id the model wrote itself (Kimi's `functions.name:0`,
    /// Mistral's `[CALL_ID]`).  Its chat template expects the same id back
    /// in the history, so the API reports it instead of minting one.
    pub id: Option<String>,
}

/// Extract tool calls from a complete model output in `format`.
///
/// Returns the leftover prose (text outside any tool-call markup, trimmed)
/// and the calls in emission order.  An empty `Vec` means the output is a
/// plain text response.  `tools` are the request's definitions; their
/// schemas decide which tagged parameters stay strings.
pub fn parse_tool_calls(
    format: ToolCallFormat,
    tools: &[Tool],
    text: &str,
) -> (String, Vec<ParsedToolCall>) {
    let (prose, calls) = read_tool_calls(format, tools, text);
    (prose.trim().to_string(), calls)
}

/// [`parse_tool_calls`] with the prose untrimmed: exactly the text a
/// [`ToolCallStream`] sends, which is the output itself when it holds no
/// calls or framing.
pub fn read_tool_calls(
    format: ToolCallFormat,
    tools: &[Tool],
    text: &str,
) -> (String, Vec<ParsedToolCall>) {
    let mut stream = ToolCallStream::new(format, Some(tools));
    let mut prose = String::new();
    let mut events = stream.push(text);
    events.extend(stream.finish());
    for event in events {
        if let ToolEvent::Text(text) = event {
            prose.push_str(&text);
        }
    }
    (prose, stream.calls())
}

// ─── Formats ────────────────────────────────────────────────────────────────

/// The tool-call markup a model family emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolCallFormat {
    /// Hermes / Qwen `<tool_call>` JSON, Qwen3-Coder XML, Mistral
    /// `[TOOL_CALLS]` and a bare JSON object, all recognised.
    #[default]
    Generic,
    /// Mistral's `[TOOL_CALLS]`, read as [`ToolCallFormat::Generic`] is;
    /// its templates take back only 9-alphanumeric call ids.
    Mistral,
    /// Llama 3.x JSON calls (`{"name": …, "parameters": …}`).
    LlamaJson,
    /// Python-call lists (`[f(a=1)]`): Llama 3.2 / Llama 4 pythonic.
    Pythonic,
    /// LFM2: a pythonic list between `<|tool_call_start|>` and
    /// `<|tool_call_end|>`.
    Lfm2,
    /// Nemotron Nano v2: a JSON call list between `<TOOLCALL>` tags.
    NemotronV2,
    /// Phi-4-mini and Firefunction v2: `functools[{…}]`.
    Functools,
    /// DeepSeek V3 / R1 (fenced JSON) and V3.1 (bare JSON) sections.
    DeepSeekV3,
    /// DeepSeek V3.2 / V4 / V4.1 DSML.
    DeepSeekDsml,
    /// Kimi K2 / Kimi-Linear.
    KimiK2,
    /// Kimi K3 XTML channels.
    KimiK3,
    /// GLM-4.5 / 4.6 / 4.7 / 5 `<arg_key>` / `<arg_value>` calls.
    Glm45,
    /// gpt-oss harmony messages addressed `to=functions.name`.
    Harmony,
    /// FunctionGemma's `<start_function_call>call:name{…}`.
    FunctionGemma,
}

/// `--tool-call-parser` names (vLLM's spelling where vLLM has the parser)
/// and the format each selects.  The first name of a format is its own.
const NAMES: &[(&str, ToolCallFormat)] = &[
    ("hermes", ToolCallFormat::Generic),
    ("qwen3_json", ToolCallFormat::Generic),
    ("qwen3_coder", ToolCallFormat::Generic),
    ("qwen3_xml", ToolCallFormat::Generic),
    ("mistral", ToolCallFormat::Mistral),
    ("llama3_json", ToolCallFormat::LlamaJson),
    ("llama4_json", ToolCallFormat::LlamaJson),
    ("pythonic", ToolCallFormat::Pythonic),
    ("llama4_pythonic", ToolCallFormat::Pythonic),
    ("lfm2", ToolCallFormat::Lfm2),
    ("nemotron_v2", ToolCallFormat::NemotronV2),
    ("phi4_mini_json", ToolCallFormat::Functools),
    ("deepseek_v3", ToolCallFormat::DeepSeekV3),
    ("deepseek_v31", ToolCallFormat::DeepSeekV3),
    ("deepseek_v4", ToolCallFormat::DeepSeekDsml),
    ("deepseek_v32", ToolCallFormat::DeepSeekDsml),
    ("deepseek_v41", ToolCallFormat::DeepSeekDsml),
    ("kimi_k2", ToolCallFormat::KimiK2),
    ("kimi_k3", ToolCallFormat::KimiK3),
    ("glm45", ToolCallFormat::Glm45),
    ("glm47", ToolCallFormat::Glm45),
    ("openai", ToolCallFormat::Harmony),
    ("functiongemma", ToolCallFormat::FunctionGemma),
];

/// Whether a chat template's source belongs to a family.
type Fingerprint = fn(&str) -> bool;

/// Chat-template fingerprints, most specific first (llama.cpp's
/// `common_chat_template_detect_params_init` order): a string only that
/// family's templates contain.
const TEMPLATE_FINGERPRINTS: &[(ToolCallFormat, Fingerprint)] = &[
    (ToolCallFormat::Harmony, |t| t.contains("<|channel|>")),
    (ToolCallFormat::KimiK2, |t| {
        t.contains("<|tool_calls_section_begin|>") && t.contains("<|tool_call_begin|>")
    }),
    (ToolCallFormat::KimiK3, |t| {
        t.contains("<|open|>") && t.contains("<|close|>") && t.contains("<|end_of_msg|>")
    }),
    (ToolCallFormat::DeepSeekDsml, |t| {
        (t.contains("dsml_token") && t.contains("DSML"))
            || t.contains("｜DSML｜ calls>")
            || t.contains("<｜DSML｜function_calls>")
    }),
    (ToolCallFormat::DeepSeekV3, |t| {
        t.contains("<｜tool▁calls▁begin｜>")
    }),
    (ToolCallFormat::Glm45, |t| t.contains("<arg_key>")),
    (ToolCallFormat::FunctionGemma, |t| {
        t.contains("<start_function_call>")
    }),
    (ToolCallFormat::Lfm2, |t| {
        t.contains("<|tool_list_start|>")
            || t.contains("<|tool_call_start|>")
            || t.contains("List of tools: [")
    }),
    (ToolCallFormat::NemotronV2, |t| t.contains("<TOOLCALL>")),
    (ToolCallFormat::Functools, |t| t.contains("functools[")),
    (ToolCallFormat::Mistral, |t| t.contains("[TOOL_CALLS]")),
    (ToolCallFormat::Generic, |t| {
        t.contains("<tool_call>") || t.contains("<function=")
    }),
    (ToolCallFormat::Pythonic, |t| t.contains("[func_name1(")),
    (ToolCallFormat::LlamaJson, |t| {
        t.contains("\"parameters\": dictionary of argument name and its value")
    }),
];

impl ToolCallFormat {
    /// The format a chat template teaches, or `None` when the template
    /// matches no specific family.
    pub fn from_template(source: &str) -> Option<Self> {
        TEMPLATE_FINGERPRINTS
            .iter()
            .find(|(_, matches)| matches(source))
            .map(|&(format, _)| format)
    }

    /// The format of an architecture whose models all share one, for a
    /// GGUF whose template does not tell.
    pub fn for_architecture(arch: Architecture) -> Option<Self> {
        match arch {
            Architecture::DeepSeek4 | Architecture::DeepSeek41 => Some(Self::DeepSeekDsml),
            Architecture::KimiK3 => Some(Self::KimiK3),
            Architecture::KimiLinear => Some(Self::KimiK2),
            Architecture::Glm4Moe | Architecture::GlmDsa | Architecture::Glm5Next => {
                Some(Self::Glm45)
            }
            Architecture::Lfm2 => Some(Self::Lfm2),
            _ => None,
        }
    }

    /// The format to parse a model's output in: the operator's choice
    /// (`--tool-call-parser`), then the one its template teaches, then its
    /// architecture's, then [`ToolCallFormat::Generic`].
    pub fn select(
        parser: Option<Self>,
        template: Option<Self>,
        arch: Option<Architecture>,
    ) -> Self {
        parser
            .or(template)
            .or_else(|| arch.and_then(Self::for_architecture))
            .unwrap_or_default()
    }

    /// Parse a `--tool-call-parser` value: `auto` (`None`) detects the
    /// format from the model.
    pub fn parse_override(s: &str) -> Result<Option<Self>, String> {
        if s.trim().eq_ignore_ascii_case("auto") {
            return Ok(None);
        }
        s.parse().map(Some)
    }

    /// The canonical `--tool-call-parser` name.
    pub fn name(self) -> &'static str {
        NAMES
            .iter()
            .find(|(_, format)| *format == self)
            .map_or("hermes", |&(name, _)| name)
    }

    /// Every accepted `--tool-call-parser` name besides `auto`.
    pub fn names() -> impl Iterator<Item = &'static str> {
        NAMES.iter().map(|&(name, _)| name)
    }

    /// A fresh id for a call the model wrote none for: `prefix` (`call_`
    /// for OpenAI, `toolu_` for Anthropic) and a UUID, or for Mistral the 9
    /// alphanumerics its templates require of the ids sent back to them
    /// (vLLM's Mistral parser mints the same).
    pub fn call_id(self, prefix: &str) -> String {
        let uuid = uuid::Uuid::new_v4().simple().to_string();
        match self {
            Self::Mistral => uuid[..9].to_string(),
            _ => format!("{prefix}{uuid}"),
        }
    }

    fn grammar(self) -> &'static dyn Grammar {
        match self {
            Self::Generic | Self::Mistral => &generic::GENERIC,
            Self::LlamaJson => &generic::LLAMA_JSON,
            Self::NemotronV2 => &generic::NEMOTRON_V2,
            Self::Functools => &generic::FUNCTOOLS_LIST,
            Self::Pythonic => &pythonic::PYTHONIC,
            Self::Lfm2 => &pythonic::LFM2,
            Self::DeepSeekV3 => &deepseek::DEEPSEEK_V3,
            Self::DeepSeekDsml => &deepseek::DSML,
            Self::KimiK2 => &kimi::KIMI_K2,
            Self::KimiK3 => &kimi::KIMI_K3,
            Self::Glm45 => &glm::GLM45,
            Self::Harmony => &harmony::HARMONY,
            Self::FunctionGemma => &gemma::FUNCTION_GEMMA,
        }
    }
}

impl FromStr for ToolCallFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let wanted = s.trim().to_ascii_lowercase().replace('-', "_");
        NAMES
            .iter()
            .find(|(name, _)| *name == wanted)
            .map(|&(_, format)| format)
            .ok_or_else(|| {
                format!(
                    "unknown tool-call parser `{s}` (one of: {})",
                    Self::names().collect::<Vec<_>>().join(", ")
                )
            })
    }
}

impl fmt::Display for ToolCallFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ─── Grammar interface ──────────────────────────────────────────────────────

/// One family's markup: what opens a tool-call region and how to read it.
/// [`ToolCallStream`] does everything else (prose, holdback, deltas).
pub(crate) trait Grammar: Sync {
    /// Strings that can open a tool-call region.
    fn markers(&self) -> &'static [Marker];

    /// Read a region that starts with one of [`Grammar::markers`].  `region`
    /// runs to the end of the text generated so far, and is read again from
    /// its start as it grows.
    fn scan(&self, region: &str, cx: &Cx<'_>) -> Scan;
}

/// A string that opens a tool-call region.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Marker {
    pub text: &'static str,
    /// Only at the very start of the output (after whitespace), as for a
    /// response that is nothing but a JSON object or a call list.
    pub leading: bool,
    /// The strings that close a region this marker opens, when the family
    /// writes one; otherwise the region may run to the end of the output.
    /// A call whose markup breaks off is dropped through the first of them.
    pub close: &'static [&'static str],
}

impl Marker {
    pub const fn anywhere(text: &'static str) -> Self {
        Self {
            text,
            leading: false,
            close: &[],
        }
    }

    pub const fn leading(text: &'static str) -> Self {
        Self {
            text,
            leading: true,
            close: &[],
        }
    }

    /// The marker, with the strings that close its region.
    pub const fn closed_by(self, close: &'static [&'static str]) -> Self {
        Self { close, ..self }
    }
}

/// What a scan may consult.
pub(crate) struct Cx<'a> {
    /// The request's tool definitions.
    pub tools: &'a [Tool],
    /// Decoding has ended: no more text will arrive, so a scan must decide.
    pub finished: bool,
}

/// The result of reading a region.
#[derive(Debug)]
pub(crate) enum Scan {
    /// Cannot tell yet; never returned once decoding has finished.
    Pending,
    /// Not a tool call: the first `n` bytes (at least the marker) are
    /// ordinary text.
    Text(usize),
    /// Tool calls (possibly none yet).  `end` is the region's length once it
    /// has closed; an open region runs to the end of the output.
    Calls {
        calls: Vec<PartialCall>,
        end: Option<usize>,
    },
}

/// A tool call as far as it has been generated.
#[derive(Debug, Clone, Default)]
pub(crate) struct PartialCall {
    /// An id the model wrote itself.
    pub id: Option<String>,
    /// The function name; empty until it is complete.
    pub name: String,
    /// The arguments text safe to stream: every later read extends it.
    pub args: String,
    /// The final arguments once the call is complete.  They may spell the
    /// same JSON differently from the streamed text (compact, decoded).
    pub done: Option<String>,
}

/// Why a value the pythonic or FunctionGemma reader reads by hand could not
/// be read.
pub(crate) enum Stop {
    /// The text ends inside it.
    Partial,
    /// It is not a value.
    Bad,
}

/// How deep those readers let a value nest (`serde_json`'s limit).  They
/// recurse once per level, so deeper text is not a value: a model stuck
/// writing `[[[[…` cannot overflow the decoding thread's stack.
pub(crate) const MAX_NESTING: usize = 128;

/// Length of the longest suffix of `text` that is a proper prefix of one of
/// `patterns`: text that may still turn into one of them.
pub(crate) fn partial_suffix<'p>(text: &str, patterns: impl IntoIterator<Item = &'p str>) -> usize {
    let mut keep = 0;
    for pattern in patterns {
        let mut n = pattern.len().saturating_sub(1).min(text.len());
        while n > keep {
            if pattern.is_char_boundary(n) && text.ends_with(&pattern[..n]) {
                keep = n;
                break;
            }
            n -= 1;
        }
    }
    keep
}

/// The earliest occurrence of any of `patterns` in `text`.
pub(crate) fn find_first<'p>(text: &str, patterns: &[&'p str]) -> Option<(usize, &'p str)> {
    patterns
        .iter()
        .filter_map(|pattern| text.find(pattern).map(|at| (at, *pattern)))
        .min_by_key(|&(at, pattern)| (at, std::cmp::Reverse(pattern.len())))
}

/// `text` up to the start of a trailing partial match of `patterns`: the
/// part of an unfinished value that is safe to stream.
pub(crate) fn before_partial<'t>(text: &'t str, patterns: &[&str]) -> &'t str {
    &text[..text.len() - partial_suffix(text, patterns.iter().copied())]
}

/// One call read by a [`section`] reader from just after its opening
/// string: `Ok(None)` while it cannot tell yet, `Err` when it is not a call,
/// otherwise the call and, once it has ended, the bytes it spans.
pub(crate) type CallRead = Result<Option<(PartialCall, Option<usize>)>, ()>;

/// A call header `s` that has not reached its closing `end` tag yet: still
/// arriving while it stays on one line without other markup (a partial
/// `end` at its tail aside), not a call otherwise or once decoding ends.
pub(crate) fn header_pending(s: &str, end: &str, cx: &Cx<'_>) -> CallRead {
    if cx.finished || before_partial(s, &[end]).contains(['\n', '<']) {
        Err(())
    } else {
        Ok(None)
    }
}

/// A run of calls each opened by one of `call`, optionally wrapped in a
/// section that one of `opens` starts and one of `closes` ends (DeepSeek's
/// `<｜tool▁calls▁begin｜>`, Kimi's `<|tool_calls_section_begin|>`, DSML's
/// `<｜DSML｜tool_calls>`).  The region starts with an `opens` or a `call`
/// string; `read` reads one call.  Calls stop at anything else, which is
/// prose again, except inside a section that has calls: there it is broken
/// markup, dropped up to the next call or the section's end.
pub(crate) fn section(
    region: &str,
    opens: &[&str],
    call: &[&str],
    closes: &[&str],
    cx: &Cx<'_>,
    read: impl Fn(&str, &Cx<'_>) -> CallRead,
) -> Scan {
    let open = opens.iter().find(|open| region.starts_with(**open));
    let marker = open
        .or_else(|| call.iter().find(|call| region.starts_with(**call)))
        .map_or(1, |marker| marker.len());
    let mut at = open.map_or(0, |open| open.len());
    let mut calls = Vec::new();
    loop {
        let next = json::skip_ws(region, at);
        let rest = &region[next..];
        if let Some(opener) = call.iter().find(|call| rest.starts_with(**call)) {
            let body = next + opener.len();
            match read(&region[body..], cx) {
                Ok(Some((partial, end))) => {
                    calls.push(partial);
                    match end {
                        Some(end) => {
                            at = body + end;
                            continue;
                        }
                        None => return Scan::Calls { calls, end: None },
                    }
                }
                Ok(None) if !cx.finished => {
                    return if calls.is_empty() && open.is_none() {
                        Scan::Pending
                    } else {
                        Scan::Calls { calls, end: None }
                    };
                }
                _ if calls.is_empty() => return Scan::Text(marker),
                // A broken call in a section: skip it below.
                _ if open.is_some() => {
                    at = body;
                    continue;
                }
                _ => {
                    return Scan::Calls {
                        calls,
                        end: Some(at),
                    }
                }
            }
        }
        if let Some(close) = closes.iter().find(|close| rest.starts_with(**close)) {
            return Scan::Calls {
                calls,
                end: Some(next + close.len()),
            };
        }
        let partial = call.iter().chain(closes).any(|tag| tag.starts_with(rest));
        if partial && (open.is_some() || !calls.is_empty()) {
            // Waiting for the next call or the section's end.
            return Scan::Calls {
                calls,
                end: cx.finished.then_some(region.len()),
            };
        }
        if open.is_some() && !calls.is_empty() {
            // Inside a section that has calls, anything else is broken
            // markup: drop it up to the next call or the section's end.
            let skip = [call, closes]
                .into_iter()
                .filter_map(|tags| find_first(rest, tags))
                .map(|(skip, _)| skip)
                .min();
            match skip {
                Some(skip) => {
                    tracing::debug!("dropping text inside a tool-call section");
                    at = next + skip;
                    continue;
                }
                None => {
                    return Scan::Calls {
                        calls,
                        end: cx.finished.then_some(region.len()),
                    }
                }
            }
        }
        return if calls.is_empty() {
            Scan::Text(marker)
        } else {
            Scan::Calls {
                calls,
                end: Some(at),
            }
        };
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::Value;

    /// Parse `text` in `format` with no tool schemas.
    pub(crate) fn parse(format: ToolCallFormat, text: &str) -> (String, Vec<ParsedToolCall>) {
        parse_tool_calls(format, &[], text)
    }

    /// The arguments of `call` as JSON.
    pub(crate) fn args(call: &ParsedToolCall) -> Value {
        serde_json::from_str(&call.arguments)
            .unwrap_or_else(|e| panic!("arguments are not JSON ({e}): {}", call.arguments))
    }

    /// Feed `text` to a stream in `chunk`-byte pieces (on char boundaries)
    /// and return the prose, each call's concatenated argument deltas, and
    /// the final calls.
    pub(crate) fn stream_in_pieces(
        format: ToolCallFormat,
        tools: &[Tool],
        text: &str,
        chunk: usize,
    ) -> (String, Vec<(String, String)>, Vec<ParsedToolCall>) {
        let mut stream = ToolCallStream::new(format, Some(tools));
        let mut events = Vec::new();
        let mut start = 0;
        while start < text.len() {
            let mut end = (start + chunk).min(text.len());
            while !text.is_char_boundary(end) {
                end += 1;
            }
            events.extend(stream.push(&text[start..end]));
            start = end;
        }
        events.extend(stream.finish());
        let mut prose = String::new();
        let mut streamed: Vec<(String, String)> = Vec::new();
        for event in events {
            match event {
                ToolEvent::Text(text) => prose.push_str(&text),
                ToolEvent::CallStart { index, name, .. } => {
                    assert_eq!(index, streamed.len(), "calls start in order");
                    streamed.push((name, String::new()));
                }
                ToolEvent::CallArgs { index, delta } => streamed[index].1.push_str(&delta),
            }
        }
        (prose, streamed, stream.calls())
    }

    /// Every chunking of `text` streams the same calls the one-shot parse
    /// returns, and the argument deltas join up to the same JSON value (or,
    /// for a call whose markup broke off, to the same text).
    pub(crate) fn assert_streams_like_parse(format: ToolCallFormat, tools: &[Tool], text: &str) {
        let (prose, calls) = parse_tool_calls(format, tools, text);
        for chunk in [1, 2, 3, 5, 8, 13, text.len().max(1)] {
            let (streamed_prose, streamed, final_calls) =
                stream_in_pieces(format, tools, text, chunk);
            assert_eq!(final_calls, calls, "chunk {chunk}: final calls");
            assert_eq!(streamed_prose.trim(), prose, "chunk {chunk}: prose");
            assert_eq!(streamed.len(), calls.len(), "chunk {chunk}: call count");
            for ((name, arguments), call) in streamed.iter().zip(&calls) {
                assert_eq!(name, &call.name, "chunk {chunk}");
                if serde_json::from_str::<Value>(&call.arguments).is_err() {
                    assert_eq!(
                        arguments, &call.arguments,
                        "chunk {chunk}: broken arguments"
                    );
                    continue;
                }
                let joined: Value = serde_json::from_str(arguments).unwrap_or_else(|e| {
                    panic!("chunk {chunk}: streamed arguments are not JSON ({e}): {arguments}")
                });
                assert_eq!(joined, args(call), "chunk {chunk}: streamed arguments");
            }
        }
    }

    /// A model stuck nesting brackets inside a call: the readers that
    /// recurse per level stop at [`MAX_NESTING`] instead of overflowing the
    /// stack of the thread decoding runs on (tokio's 2 MiB), and the call
    /// stands as far as it was read, as for any markup that breaks off.
    #[test]
    fn deep_nesting_does_not_overflow_the_stack() {
        for (format, open, level) in [
            (ToolCallFormat::Pythonic, "[f(a=", "["),
            (ToolCallFormat::Lfm2, "<|tool_call_start|>[f(a=", "["),
            (
                ToolCallFormat::FunctionGemma,
                "<start_function_call>call:f{a:",
                "{b:",
            ),
        ] {
            let text = format!("{open}{}", level.repeat(100_000));
            let (prose, calls) = std::thread::Builder::new()
                .stack_size(2 << 20)
                .spawn(move || parse(format, &text))
                .unwrap()
                .join()
                .unwrap();
            assert_eq!(prose, "", "{format}");
            assert_eq!(calls.len(), 1, "{format}");
            assert_eq!(calls[0].name, "f");
            assert_eq!(calls[0].arguments, "{", "{format}");
            let text = format!("{open}{}", level.repeat(MAX_NESTING + 2));
            assert_streams_like_parse(format, &[], &text);
        }
        // Up to the limit, a value nests as deep as it is written.
        let (open, close) = ("[".repeat(MAX_NESTING), "]".repeat(MAX_NESTING));
        let (_, calls) = parse(ToolCallFormat::Pythonic, &format!("[f(a={open}1{close})]"));
        assert_eq!(calls[0].arguments, format!("{{\"a\":{open}1{close}}}"));
    }

    #[test]
    fn plain_text_has_no_tool_calls() {
        let (content, calls) = parse(ToolCallFormat::Generic, "The weather is sunny today.");
        assert_eq!(content, "The weather is sunny today.");
        assert!(calls.is_empty());
    }

    /// The untrimmed prose a complete response returns when it has no calls
    /// is the output itself, unless the format frames its text (Kimi K3's
    /// response tags, harmony's channel headers): then it is the text the
    /// stream sends.
    #[test]
    fn read_tool_calls_returns_the_streamed_prose() {
        let text = "\n  A plain answer that mentions <tool_call> tags.  \n";
        assert_eq!(read_tool_calls(ToolCallFormat::Generic, &[], text).0, text);
        let framed = "<|open|>response<|sep|>Hello there<|close|>response<|sep|>\
                      <|close|>message<|sep|>";
        assert_eq!(
            read_tool_calls(ToolCallFormat::KimiK3, &[], framed).0,
            "Hello there"
        );
        assert_eq!(
            read_tool_calls(
                ToolCallFormat::Harmony,
                &[],
                "<|channel|>final<|message|>Hello"
            )
            .0,
            "Hello"
        );
    }

    #[test]
    fn format_names_round_trip() {
        for name in ToolCallFormat::names() {
            let format: ToolCallFormat = name.parse().unwrap();
            assert_eq!(format.name().parse::<ToolCallFormat>().unwrap(), format);
        }
        assert_eq!("DeepSeek-V4".parse(), Ok(ToolCallFormat::DeepSeekDsml));
        assert!("nope".parse::<ToolCallFormat>().is_err());
        assert_eq!(ToolCallFormat::Generic.name(), "hermes");
        assert_eq!(ToolCallFormat::parse_override("auto"), Ok(None));
        assert_eq!(
            ToolCallFormat::parse_override("kimi_k3"),
            Ok(Some(ToolCallFormat::KimiK3))
        );
        assert!(ToolCallFormat::parse_override("nope").is_err());
    }

    #[test]
    fn architecture_picks_a_format_when_the_template_does_not() {
        use crate::model::Architecture as A;
        let select = |template, arch| ToolCallFormat::select(None, template, arch);
        assert_eq!(
            select(None, Some(A::DeepSeek41)),
            ToolCallFormat::DeepSeekDsml
        );
        assert_eq!(select(None, Some(A::KimiK3)), ToolCallFormat::KimiK3);
        assert_eq!(select(None, Some(A::Glm4Moe)), ToolCallFormat::Glm45);
        assert_eq!(select(None, Some(A::Qwen3)), ToolCallFormat::Generic);
        assert_eq!(select(None, None), ToolCallFormat::Generic);
        // The template wins: a Hermes-tuned LFM2 is not pythonic.
        assert_eq!(
            select(Some(ToolCallFormat::Generic), Some(A::Lfm2)),
            ToolCallFormat::Generic
        );
        // The operator wins over both.
        assert_eq!(
            ToolCallFormat::select(
                Some(ToolCallFormat::Pythonic),
                Some(ToolCallFormat::Generic),
                Some(A::Lfm2)
            ),
            ToolCallFormat::Pythonic
        );
    }

    #[test]
    fn partial_suffix_finds_marker_prefixes() {
        assert_eq!(partial_suffix("abc <tool", ["<tool_call>"]), 5);
        assert_eq!(partial_suffix("abc <tool_call>", ["<tool_call>"]), 0);
        assert_eq!(partial_suffix("abc", ["<tool_call>"]), 0);
        assert_eq!(
            partial_suffix("x<｜DSML", ["<｜DSML｜invoke"]),
            "<｜DSML".len()
        );
        assert_eq!(partial_suffix("x<｜", ["<｜DSML｜invoke"]), "<｜".len());
    }
}
