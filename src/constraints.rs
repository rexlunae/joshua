//! Structured-output constraints for generated tokens.
//!
//! `response_format: {"type": "json_object"}` and `json_schema` compile to an
//! [llguidance] grammar.  Sampling asks the grammar for the set of legal next
//! tokens and masks everything else out before top-k/top-p, so the output is
//! valid JSON by construction and EOS is selectable only once the document is
//! complete.
//!
//! Grammar compilation is keyed by the schema text and cached in a bounded
//! LRU.  A parser instance is cheap to clone and carries no request state, so
//! the cache stores the initial matcher and hands each request a clone.
//!
//! [llguidance]: https://github.com/guidance-ai/llguidance

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory};
use serde_json::json;

use crate::types::ConstraintSpec;

/// Re-exported so the sampler can name the mask type without depending on
/// llguidance directly.
pub(crate) use llguidance::toktrie::SimpleVob;

/// How many compiled schemas one engine keeps.  A schema is a few hundred
/// bytes to a few kilobytes, and each matcher is a few hundred more; the
/// bound is a DoS guard, not a memory optimization.
const CACHE_CAPACITY: usize = 32;

/// Wall-clock budget for one compilation.  The parser's own fuel limits bound
/// the work; this catches a pathological schema whose compilation is slow
/// without a usable error.
const COMPILE_BUDGET: Duration = Duration::from_secs(2);

/// Compiled grammar environment plus a bounded cache of initial matchers.
pub(crate) struct ConstraintEngine {
    factory: ParserFactory,
    cache: Mutex<Cache>,
}

struct Cache {
    entries: VecDeque<(String, Matcher)>,
}

impl ConstraintEngine {
    /// Build the tokenizer trie and parser factory from `tokenizer.json`.
    ///
    /// `eos_ids` come from the model metadata; they are the only tokens the
    /// grammar can use to finish a document.
    pub(crate) fn from_path(path: &Path, eos_ids: &[u32]) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("parse {}: {e}", path.display()))?;
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("load {}: {e}", path.display()))?;

        // llguidance's byte-level parser understands the two decoders that
        // split UTF-8 across tokens (ByteLevel and ByteFallback).  WordLevel
        // and other decoder-less tokenizers are handled by decoding each id,
        // which is exactly what this engine does when it emits tokens.
        let mut token_bytes = match llguidance::token_bytes_from_tokenizer_json(&value) {
            Ok(bytes) => bytes,
            Err(e) => {
                tracing::debug!(
                    "llguidance could not derive token bytes from {} ({e}); \
                     falling back to per-token decode",
                    path.display()
                );
                token_bytes_from_tokenizer(&tokenizer)?
            }
        };
        let vocab_size = tokenizer.get_vocab_size(true);
        if vocab_size == 0 {
            return Err("tokenizer has an empty vocabulary".to_string());
        }
        if token_bytes.len() < vocab_size {
            token_bytes.resize(vocab_size, Vec::new());
        }
        token_bytes.truncate(vocab_size);

        let eos: Vec<u32> = eos_ids
            .iter()
            .copied()
            .filter(|&id| (id as usize) < vocab_size)
            .collect();
        let first_eos = eos.first().copied().unwrap_or(0);
        if first_eos as usize >= vocab_size {
            return Err("EOS token is outside the vocabulary".to_string());
        }
        let mut trie = TokTrie::from(
            &TokRxInfo::new(vocab_size as u32, first_eos),
            &token_bytes,
        );
        if !eos.is_empty() {
            trie = trie.with_eos_tokens(&eos);
        }
        let env: TokEnv = std::sync::Arc::new(ApproximateTokEnv::new(trie));
        let mut factory =
            ParserFactory::new_simple(&env).map_err(|e| format!("parser factory: {e}"))?;
        factory.quiet();
        {
            // Bound the schema and lexer work even before the wall-clock
            // check below.  The defaults are generous (500k symbols); these
            // keep a hostile request from pinning a worker for seconds.
            let limits = factory.limits_mut();
            limits.max_grammar_size = limits.max_grammar_size.min(200_000);
            limits.initial_lexer_fuel = limits.initial_lexer_fuel.min(500_000);
            limits.step_lexer_fuel = limits.step_lexer_fuel.min(200_000);
        }
        Ok(Self {
            factory,
            cache: Mutex::new(Cache {
                entries: VecDeque::new(),
            }),
        })
    }

    /// Return a fresh matcher for `spec`.
    ///
    /// The initial mask is computed before the matcher enters the cache, so a
    /// schema that cannot produce any token is rejected here (an invalid
    /// request) rather than after the model has already prefilled.
    pub(crate) fn matcher(&self, spec: &ConstraintSpec) -> Result<Matcher, String> {
        let key = cache_key(spec);
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(i) = cache.entries.iter().position(|(k, _)| *k == key) {
                let entry = cache.entries.remove(i).expect("position was just checked");
                let matcher = entry.1.clone();
                cache.entries.push_back(entry);
                return Ok(matcher);
            }
        }

        let start = Instant::now();
        let grammar = grammar(spec);
        let parser = self
            .factory
            .create_parser(grammar)
            .map_err(|e| format!("invalid JSON schema: {e}"))?;
        let mut matcher = Matcher::new(Ok(parser));
        let mask = matcher
            .compute_mask()
            .map_err(|e| format!("invalid JSON schema: {e}"))?;
        if mask.num_set() == 0 {
            return Err("the JSON schema accepts no output".to_string());
        }
        if start.elapsed() > COMPILE_BUDGET {
            return Err(format!(
                "JSON schema compilation exceeded {} ms",
                COMPILE_BUDGET.as_millis()
            ));
        }

        if let Ok(mut cache) = self.cache.lock() {
            cache.entries.push_back((key, matcher.clone()));
            while cache.entries.len() > CACHE_CAPACITY {
                cache.entries.pop_front();
            }
        }
        Ok(matcher)
    }

    #[cfg(test)]
    fn cached_len(&self) -> usize {
        self.cache.lock().map(|c| c.entries.len()).unwrap_or(0)
    }
}

fn cache_key(spec: &ConstraintSpec) -> String {
    match spec {
        ConstraintSpec::JsonObject => "object".to_string(),
        ConstraintSpec::JsonSchema { schema } => {
            format!("schema:{}", serde_json::to_string(schema).unwrap_or_default())
        }
    }
}

fn grammar(spec: &ConstraintSpec) -> TopLevelGrammar {
    match spec {
        ConstraintSpec::JsonObject => TopLevelGrammar::from_json_schema(json!({"type": "object"})),
        ConstraintSpec::JsonSchema { schema } => {
            TopLevelGrammar::from_json_schema(schema.clone())
        }
    }
}

/// Decode every token on its own, matching how the engine emits word-level
/// and other decoder-less vocabularies.
fn token_bytes_from_tokenizer(tokenizer: &tokenizers::Tokenizer) -> Result<Vec<Vec<u8>>, String> {
    let n = tokenizer.get_vocab_size(true);
    let mut out = Vec::with_capacity(n);
    for id in 0..n as u32 {
        let piece = tokenizer
            .decode(&[id], false)
            .map_err(|e| format!("decode token {id}: {e}"))?;
        out.push(piece.into_bytes());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A WordLevel vocabulary whose ids are the characters of `{"a":1}` plus
    /// a few fillers.  Id 3 is EOS, matching the tiny GGUF fixture.
    const JSON_TOKENIZER: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 3, "content": "</s>", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"<unk>": 0, "{": 1, "\"": 2, "</s>": 3,
                      "a": 4, ":": 5, "1": 6, "}": 7, ",": 8, "[": 9,
                      "]": 10, "b": 11, "2": 12, "x": 13, "true": 14, "null": 15},
            "unk_token": "<unk>"
        }
    }"#;

    fn tokenizer_path() -> PathBuf {
        // One file for the whole test binary: writing it per test races with
        // a concurrent reader and can expose a truncated file.
        static PATH: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
        PATH.get_or_init(|| {
            let dir = std::env::temp_dir().join(format!(
                "joshua-constraints-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("tokenizer.json");
            std::fs::write(&path, JSON_TOKENIZER).unwrap();
            path
        })
        .clone()
    }

    fn engine() -> ConstraintEngine {
        ConstraintEngine::from_path(&tokenizer_path(), &[3]).expect("constraint engine")
    }

    fn forced_schema() -> ConstraintSpec {
        ConstraintSpec::JsonSchema {
            schema: json!({
                "type": "object",
                "properties": {"a": {"const": 1}},
                "required": ["a"],
                "additionalProperties": false
            }),
        }
    }

    /// Walk the forced schema using the grammar masks and collect the bytes.
    /// Every step must have exactly one legal token; the result must be the
    /// canonical `{"a":1}`.
    #[test]
    fn a_forced_schema_admits_exactly_the_expected_document() {
        let engine = engine();
        let mut matcher = engine.matcher(&forced_schema()).expect("valid schema");
        let chars = ["{", "\"", "a", "\"", ":", "1", "}"];
        for (i, ch) in chars.iter().enumerate() {
            let mask = matcher.compute_mask().expect("mask");
            let id = match *ch {
                "{" => 1,
                "\"" => 2,
                "a" => 4,
                ":" => 5,
                "1" => 6,
                "}" => 7,
                other => panic!("unexpected character {other}"),
            };
            assert!(
                mask.is_allowed(id),
                "step {i} ({ch}) should be legal; mask allows {} tokens",
                mask.num_set()
            );
            assert_eq!(
                mask.num_set(),
                1,
                "step {i} ({ch}) is forced, but the mask has alternatives"
            );
            matcher.consume_token(id).expect("consume");
        }
        // Consuming the closing brace can move the parser into its stop
        // state, after which only EOS is legal (and `compute_mask` refuses
        // to run); `compute_mask_or_eos` is the sampling-loop call.
        let mask = matcher.compute_mask_or_eos().expect("final mask");
        assert!(mask.is_allowed(3), "EOS must be legal after a complete document");
        assert_eq!(mask.num_set(), 1, "only EOS may follow the closing brace");
    }

    /// A schema with only a `type` keyword accepts many documents, but the
    /// mask must never allow a token that cannot start an object.
    #[test]
    fn json_object_starts_with_an_object_brace() {
        let engine = engine();
        let mut matcher = engine
            .matcher(&ConstraintSpec::JsonObject)
            .expect("json_object");
        let mask = matcher.compute_mask().expect("mask");
        assert!(mask.is_allowed(1), "`{{` must be legal");
        assert!(!mask.is_allowed(4), "`a` cannot start an object");
        assert!(!mask.is_allowed(6), "`1` cannot start an object");
    }

    #[test]
    fn invalid_schema_is_rejected_during_compilation() {
        let engine = engine();
        let bad = ConstraintSpec::JsonSchema {
            schema: json!({"type": "definitely-not-a-json-type"}),
        };
        let err = match engine.matcher(&bad) {
            Ok(_) => panic!("schema must be rejected"),
            Err(e) => e,
        };
        assert!(err.contains("invalid JSON schema"), "got: {err}");
    }

    #[test]
    fn cache_is_bounded_and_reuses_compiled_schemas() {
        let engine = engine();
        for i in 0..CACHE_CAPACITY + 3 {
            let spec = ConstraintSpec::JsonSchema {
                schema: json!({"type": "object", "title": format!("s{i}")}),
            };
            engine.matcher(&spec).expect("schema");
        }
        assert_eq!(engine.cached_len(), CACHE_CAPACITY);
        // The first entries were evicted, but an old schema can be compiled
        // again and the cache stays bounded.
        let spec = ConstraintSpec::JsonSchema {
            schema: json!({"type": "object", "title": "s0"}),
        };
        engine.matcher(&spec).expect("recompile");
        assert_eq!(engine.cached_len(), CACHE_CAPACITY);
    }
}
