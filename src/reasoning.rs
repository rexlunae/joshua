//! Split a model's inline thinking trace out of the visible answer.
//!
//! Reasoning templates (Qwen3, GLM, DeepSeek, Kimi, …) open the scratchpad with `<think` right
//! after the assistant header and close it with `</think>`, all of it inline in the same token
//! stream as the answer.  Clients that want to collapse the trace need the two channels separated,
//! so this splitter consumes decoded text and reports how much of it is reasoning and how much is
//! answer.
//!
//! The splitter is streaming-safe: a marker can be split across two decoded tokens, so it holds
//! back only the trailing bytes that could still become a marker and releases them on the next
//! feed (or on [`ReasoningSplitter::finish`]).

/// Opening marker of a thinking trace.
const OPEN: &str = "<think>";
/// Closing marker of a thinking trace.
const CLOSE: &str = "</think>";

/// Separates a model's thinking trace from its answer.
#[derive(Debug, Default)]
pub struct ReasoningSplitter {
    /// True while inside the thinking span.
    in_reasoning: bool,
    /// Text held back because it may be the start of a marker.
    pending: String,
}

impl ReasoningSplitter {
    /// A splitter positioned outside any thinking span.
    pub fn new() -> Self {
        Self::default()
    }

    /// True while the splitter is inside a thinking span.
    pub fn in_reasoning(&self) -> bool {
        self.in_reasoning
    }

    /// Consume decoded text, returning `(content, reasoning)` deltas.
    ///
    /// Either side may be empty.  Text that could be the beginning of a marker is withheld until
    /// the next call, so an emitted delta never has to be retracted.
    pub fn feed(&mut self, text: &str) -> (String, String) {
        self.pending.push_str(text);
        self.drain(false)
    }

    /// Flush everything still held back.  Call once, when generation ends.
    pub fn finish(&mut self) -> (String, String) {
        self.drain(true)
    }

    fn drain(&mut self, final_chunk: bool) -> (String, String) {
        let mut content = String::new();
        let mut reasoning = String::new();
        loop {
            if self.in_reasoning {
                if let Some(at) = self.pending.find(CLOSE) {
                    let head: String = self.pending.drain(..at).collect();
                    reasoning.push_str(&head);
                    self.pending.drain(..CLOSE.len());
                    self.in_reasoning = false;
                    continue;
                }
                let keep = if final_chunk { 0 } else { partial_marker_len(&self.pending, CLOSE) };
                let release = self.pending.len().saturating_sub(keep);
                let head: String = self.pending.drain(..release).collect();
                reasoning.push_str(&head);
                break;
            }

            // Outside a span: the next marker decides.  A stray close marker (templates that
            // disable thinking emit one) is dropped rather than shown as prose.
            let open = self.pending.find(OPEN);
            let close = self.pending.find(CLOSE);
            match (open, close) {
                (Some(o), Some(c)) if c < o => {
                    let head: String = self.pending.drain(..c).collect();
                    content.push_str(&head);
                    self.pending.drain(..CLOSE.len());
                }
                (Some(o), _) => {
                    let head: String = self.pending.drain(..o).collect();
                    content.push_str(&head);
                    self.pending.drain(..OPEN.len());
                    self.in_reasoning = true;
                }
                (None, Some(c)) => {
                    let head: String = self.pending.drain(..c).collect();
                    content.push_str(&head);
                    self.pending.drain(..CLOSE.len());
                }
                (None, None) => {
                    let keep = if final_chunk {
                        0
                    } else {
                        partial_marker_len(&self.pending, OPEN)
                            .max(partial_marker_len(&self.pending, CLOSE))
                    };
                    let release = self.pending.len().saturating_sub(keep);
                    let head: String = self.pending.drain(..release).collect();
                    content.push_str(&head);
                    break;
                }
            }
        }
        (content, reasoning)
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of `marker`.
fn partial_marker_len(text: &str, marker: &str) -> usize {
    let max = marker.len().min(text.len());
    for len in (1..=max).rev() {
        if text.as_bytes()[text.len() - len..] == marker.as_bytes()[..len] {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_all(chunks: &[String]) -> (String, String) {
        let mut splitter = ReasoningSplitter::new();
        let mut content = String::new();
        let mut reasoning = String::new();
        for chunk in chunks {
            let (c, r) = splitter.feed(chunk);
            content.push_str(&c);
            reasoning.push_str(&r);
        }
        let (c, r) = splitter.finish();
        content.push_str(&c);
        reasoning.push_str(&r);
        (content, reasoning)
    }

    #[test]
    fn separates_a_complete_think_span() {
        let input = format!("{OPEN}\n2+2 is 4\n{CLOSE}\n\nThe answer is 4.");
        let (content, reasoning) = feed_all(&[input]);
        assert_eq!(reasoning, "\n2+2 is 4\n");
        assert_eq!(content, "\n\nThe answer is 4.");
    }

    #[test]
    fn a_marker_split_across_tokens_is_not_leaked_as_prose() {
        // The markers arrive in pieces, as decoded tokens can.
        let pieces: Vec<String> = [
            OPEN[..3].to_string(),
            format!("{}hidden", &OPEN[3..]),
            CLOSE[..3].to_string(),
            format!("{}shown", &CLOSE[3..]),
        ]
        .to_vec();
        let (content, reasoning) = feed_all(&pieces);
        assert_eq!(reasoning, "hidden");
        assert_eq!(content, "shown");
    }

    #[test]
    fn text_without_markers_is_passed_through_unchanged() {
        let (content, reasoning) = feed_all(&["Hello, ".to_string(), "world".to_string(), "!".to_string()]);
        assert_eq!(content, "Hello, world!");
        assert_eq!(reasoning, "");
    }

    #[test]
    fn a_trailing_partial_marker_is_held_until_it_resolves() {
        let mut splitter = ReasoningSplitter::new();
        let head = format!("answer {}", &OPEN[..OPEN.len() - 1]);
        assert_eq!(splitter.feed(&head), ("answer ".to_string(), String::new()));
        let tail = format!("{}why", &OPEN[OPEN.len() - 1..]);
        assert_eq!(splitter.feed(&tail), (String::new(), "why".to_string()));
    }

    #[test]
    fn opening_a_span_without_closing_it_still_yields_the_reasoning() {
        let (content, reasoning) = feed_all(&[format!("{OPEN}unfinished")]);
        assert_eq!(content, "");
        assert_eq!(reasoning, "unfinished");
    }

    #[test]
    fn a_close_marker_without_an_open_one_is_dropped() {
        // Templates that disable thinking emit the close marker as the first token.
        let (content, reasoning) = feed_all(&[format!("{CLOSE}\n\nAnswer")]);
        assert_eq!(content, "\n\nAnswer");
        assert_eq!(reasoning, "");
    }
}
