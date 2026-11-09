//! Token log-probabilities on the wire.
//!
//! The engine reports each generated token's [`TokenLogprob`] (see
//! [`crate::types::GenerationOptions::logprobs`]); this module writes them
//! in each API's shape: OpenAI chat's `logprobs.content`, the legacy
//! completions API's parallel arrays, and llama-server's
//! `completion_probabilities`.

use serde_json::{json, Map, Value};

use crate::engine::Engine;
use crate::types::{TokenLogprob, MAX_TOP_LOGPROBS};

/// What OpenAI writes for a probability of zero, which JSON cannot spell
/// as `-Infinity`.
const NEG_INFINITY: f32 = -9999.0;

/// A log-probability JSON can carry.
fn finite(logprob: f32) -> f32 {
    if logprob.is_finite() {
        logprob
    } else {
        NEG_INFINITY
    }
}

/// A token's text and bytes, as an entry names it.
fn piece(engine: &Engine, id: u32) -> (String, Vec<u8>) {
    let bytes = engine.token_piece(id);
    (String::from_utf8_lossy(&bytes).into_owned(), bytes)
}

/// A request's count of alternatives, checked against
/// [`MAX_TOP_LOGPROBS`]; `field` names it in the error.
pub(crate) fn count(field: &str, n: i64) -> Result<u32, String> {
    u32::try_from(n)
        .ok()
        .filter(|n| *n <= MAX_TOP_LOGPROBS)
        .ok_or_else(|| format!("`{field}` must be between 0 and {MAX_TOP_LOGPROBS}"))
}

/// OpenAI chat's `logprobs` object: `{"content": [...]}`, one entry per
/// token with its `token`, `logprob`, `bytes` and `top_logprobs`.
pub(crate) fn chat(engine: &Engine, entries: &[TokenLogprob]) -> Value {
    let item = |id: u32, logprob: f32| {
        let (token, bytes) = piece(engine, id);
        json!({"token": token, "logprob": finite(logprob), "bytes": bytes})
    };
    let content: Vec<Value> = entries
        .iter()
        .map(|entry| {
            let mut value = item(entry.token, entry.logprob);
            value["top_logprobs"] = entry.top.iter().map(|&(id, lp)| item(id, lp)).collect();
            value
        })
        .collect();
    json!({ "content": content })
}

/// The legacy completions API's `logprobs` object: parallel `tokens`,
/// `token_logprobs`, `top_logprobs` (each a token-to-logprob map holding
/// the alternatives and the token itself, as vLLM's does) and
/// `text_offset` arrays.  Offsets count characters from `offset`, where
/// this run of tokens starts in the choice's text.
pub(crate) fn completion(engine: &Engine, entries: &[TokenLogprob], offset: usize) -> Value {
    let mut tokens = Vec::with_capacity(entries.len());
    let mut token_logprobs = Vec::with_capacity(entries.len());
    let mut top_logprobs = Vec::with_capacity(entries.len());
    let mut text_offset = Vec::with_capacity(entries.len());
    let mut at = offset;
    for entry in entries {
        let (token, _) = piece(engine, entry.token);
        let mut top = Map::new();
        for &(id, lp) in &entry.top {
            top.insert(piece(engine, id).0, json!(finite(lp)));
        }
        top.insert(token.clone(), json!(finite(entry.logprob)));
        text_offset.push(at);
        at += token.chars().count();
        tokens.push(token);
        token_logprobs.push(finite(entry.logprob));
        top_logprobs.push(Value::Object(top));
    }
    json!({
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    })
}

/// The characters [`completion`] advances its offsets by for `entries`.
pub(crate) fn text_len(engine: &Engine, entries: &[TokenLogprob]) -> usize {
    entries
        .iter()
        .map(|entry| piece(engine, entry.token).0.chars().count())
        .sum()
}

/// llama-server's `completion_probabilities`: one entry per token with its
/// `id`, `token`, `bytes`, `logprob` and `top_logprobs`.
pub(crate) fn llama(engine: &Engine, entries: &[TokenLogprob]) -> Value {
    let item = |id: u32, logprob: f32| {
        let (token, bytes) = piece(engine, id);
        json!({"id": id, "token": token, "bytes": bytes, "logprob": finite(logprob)})
    };
    entries
        .iter()
        .map(|entry| {
            let mut value = item(entry.token, entry.logprob);
            value["top_logprobs"] = entry.top.iter().map(|&(id, lp)| item(id, lp)).collect();
            value
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_are_bounded_and_infinities_are_spelled_as_openai_does() {
        assert_eq!(count("top_logprobs", 0), Ok(0));
        assert_eq!(count("top_logprobs", 20), Ok(20));
        assert!(count("top_logprobs", 21).is_err());
        assert!(count("top_logprobs", -1).is_err());
        assert_eq!(finite(f32::NEG_INFINITY), NEG_INFINITY);
        assert_eq!(finite(-0.5), -0.5);
    }
}
