//! A separate draft model for speculative decoding.
//!
//! A small model that shares the target's tokenizer (Qwen3-0.6B for a
//! Qwen3-30B-A3B, say) proposes the next few tokens one cheap step at a time;
//! the target scores them all in one verification pass (see
//! [`crate::speculative`]).  Under sampling, each draft token is drawn from
//! the draft model's own distribution `q` and verified with the full
//! speculative-sampling rule `min(1, p/q)`, so the output is distributed as
//! plain decoding's; at temperature 0 the draft is its argmax and the output
//! is token-for-token identical.
//!
//! The draft model keeps its own KV cache.  A [`DraftSession`] remembers the
//! tokens its cache holds, and before each draft it is brought up to the
//! decode context by cutting the cache back to the longest prefix both agree
//! on and feeding the rest — the tokens the target accepted or substituted
//! since the last draft, usually one or two.  That needs an arbitrary KV
//! truncate, so only attention-only models of Joshua's own loaders (qwen3,
//! qwen3moe, glm4, deepseek2, …) can draft; the load refuses the rest.
//!
//! Weights load once from their own mapping and every session shares them,
//! adding only a KV cache.  Sessions are pooled with the tokens they hold, so
//! a follow-up turn re-reads only its new tokens.

use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use memmap2::Mmap;

use crate::error::{JoshuaError, Result};
use crate::model::QuantizedModel;

/// Default maximum number of draft-model tokens verified per step.
pub const DEFAULT_DRAFT_MODEL_MAX: usize = 4;
/// Idle draft sessions kept for later requests.
const MAX_IDLE_SESSIONS: usize = 4;
/// Longest run fed to the draft model in one forward pass while it catches
/// up on a prompt, bounding its activation memory.
const CATCH_UP_CHUNK: usize = 512;

/// A loaded draft model: the shared weights and a pool of idle sessions.
pub struct DraftModel {
    path: PathBuf,
    /// The weights every session borrows; never used for a forward pass.
    template: QuantizedModel,
    device: Device,
    idle: Mutex<Vec<DraftSession>>,
    /// Logit rows the draft produces (its embedding vocabulary).
    vocab: usize,
}

/// One draft model session and the tokens its KV cache holds.
pub struct DraftSession {
    model: QuantizedModel,
    tokens: Vec<u32>,
}

impl DraftModel {
    /// Load the draft model at `path` (a `.gguf`, or a directory holding
    /// one) onto `device`.
    ///
    /// `target_tokens` is the target GGUF's `tokenizer.ggml.tokens` list,
    /// when it has one: the two vocabularies must agree on every id they
    /// share, or the draft would propose different text under the same ids.
    pub fn load(
        path: &Path,
        device: &Device,
        n_ctx: usize,
        target_tokens: Option<&[String]>,
    ) -> Result<Self> {
        let path = if path.is_dir() {
            crate::find_gguf_in_dir(path)?
        } else {
            path.to_path_buf()
        };
        let file = File::open(&path)
            .map_err(|e| JoshuaError::ModelLoad(format!("draft model {}: {e}", path.display())))?;
        let file_len = file.metadata()?.len();
        // SAFETY: the mapping is read-only; as for the target model, a file
        // changed underneath a running engine is outside its contract.
        let mmap = Arc::new(unsafe { Mmap::map(&file) }?);
        crate::gguf_ext::read_header(&mut Cursor::new(&mmap[..]))
            .and_then(|raw| raw.validate_layout(file_len))
            .map_err(|e| JoshuaError::ModelLoad(format!("draft model GGUF check failed: {e}")))?;
        let gguf = crate::engine::read_gguf_header(&mmap)
            .map_err(|e| JoshuaError::ModelLoad(format!("draft model GGUF read failed: {e}")))?;

        if let (Some(target), Some(draft)) = (target_tokens, gguf_tokens(&gguf.metadata)) {
            check_vocab(target, &draft)?;
        } else {
            tracing::warn!(
                "draft model: cannot compare vocabularies (no tokenizer.ggml.tokens list); \
                 make sure it shares the target's tokenizer"
            );
        }

        let mut cursor = Cursor::new(&mmap[..]);
        let template = QuantizedModel::from_gguf_mmap_placed(
            gguf,
            &mut cursor,
            device,
            device,
            Some(Arc::clone(&mmap)),
            None,
            n_ctx,
            None,
        )
        .map_err(|e| JoshuaError::ModelLoad(format!("draft model init failed: {e}")))?;
        if !template.supports_shared_weights() || !template.supports_kv_truncate() {
            return Err(JoshuaError::ModelLoad(format!(
                "draft model {}: its loader cannot cut its KV cache back to an arbitrary \
                 prefix, which drafting needs; use an attention-only model Joshua loads \
                 natively (qwen3, qwen3moe, glm4, deepseek2, …)",
                path.display()
            )));
        }
        let mut model = Self {
            path,
            template,
            device: device.clone(),
            idle: Mutex::new(Vec::new()),
            vocab: 0,
        };
        // Probe the vocabulary once with a one-token pass on a scratch
        // session, so callers know how long a q row is.
        let mut probe = model.new_session()?;
        model.vocab = probe.forward(&[0], 0, &model.device)?.len();
        tracing::info!(
            "draft model {} loaded ({} vocabulary rows)",
            model.path.display(),
            model.vocab
        );
        Ok(model)
    }

    /// Path of the loaded draft GGUF.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A session for a request whose context starts with `context`: the idle
    /// one sharing the longest prefix with it, else a fresh one.
    pub fn acquire(&self, context: &[u32]) -> Result<DraftSession> {
        let reused = {
            let mut idle = self.idle.lock().unwrap_or_else(|p| p.into_inner());
            let best = idle
                .iter()
                .enumerate()
                .max_by_key(|(_, s)| crate::prefix_cache::common_token_prefix(&s.tokens, context))
                .map(|(i, _)| i);
            best.map(|i| idle.swap_remove(i))
        };
        match reused {
            Some(s) => Ok(s),
            None => self.new_session(),
        }
    }

    /// Return a session to the pool.
    pub fn release(&self, session: DraftSession) {
        let mut idle = self.idle.lock().unwrap_or_else(|p| p.into_inner());
        idle.push(session);
        if idle.len() > MAX_IDLE_SESSIONS {
            idle.remove(0);
        }
    }

    /// The draft model's device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    fn new_session(&self) -> Result<DraftSession> {
        let model = self
            .template
            .new_session()
            .ok_or_else(|| JoshuaError::Inference("draft model cannot derive a session".into()))?;
        Ok(DraftSession {
            model,
            tokens: Vec::new(),
        })
    }
}

impl DraftSession {
    /// Bring the KV cache to hold exactly `context` and return the logits
    /// after its last token.  Tokens the cache already holds (the longest
    /// common prefix) are kept; a cache holding all of `context` re-feeds
    /// its last token, whose logits are wanted.
    pub fn sync(&mut self, context: &[u32], device: &Device) -> Result<Vec<f32>> {
        if context.is_empty() {
            return Err(JoshuaError::Inference("draft model: empty context".into()));
        }
        let mut keep =
            crate::prefix_cache::common_token_prefix(&self.tokens, context).min(context.len() - 1);
        if keep < self.tokens.len() && !self.truncate(keep) {
            keep = 0;
        }
        let mut logits = Vec::new();
        for chunk in context[keep..].chunks(CATCH_UP_CHUNK) {
            logits = self.forward(chunk, keep, device)?;
            self.tokens.extend_from_slice(chunk);
            keep += chunk.len();
        }
        Ok(logits)
    }

    /// Feed one drafted token after the synced context, returning the next
    /// logits.
    pub fn push(&mut self, token: u32, device: &Device) -> Result<Vec<f32>> {
        let logits = self.forward(&[token], self.tokens.len(), device)?;
        self.tokens.push(token);
        Ok(logits)
    }

    /// Tokens the KV cache holds.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn truncate(&mut self, keep: usize) -> bool {
        match self.model.truncate_kv_cache(keep) {
            Ok(true) => {
                self.tokens.truncate(keep);
                true
            }
            _ => {
                // Unknown state: start over from an empty cache.
                self.model.clear_kv_cache();
                self.tokens.clear();
                false
            }
        }
    }

    fn forward(&mut self, tokens: &[u32], pos: usize, device: &Device) -> Result<Vec<f32>> {
        let input = Tensor::new(tokens, device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| JoshuaError::Inference(e.to_string()))?;
        self.model
            .forward(&input, pos)
            .and_then(|l| l.flatten_all())
            .and_then(|l| l.to_vec1::<f32>())
            .map_err(|e| JoshuaError::Inference(format!("draft model: {e}")))
    }
}

/// A GGUF's `tokenizer.ggml.tokens` list.
pub fn gguf_tokens(
    metadata: &std::collections::HashMap<String, gguf_file::Value>,
) -> Option<Vec<String>> {
    match metadata.get("tokenizer.ggml.tokens")? {
        gguf_file::Value::Array(items) => items
            .iter()
            .map(|v| match v {
                gguf_file::Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

/// Refuse a draft vocabulary that disagrees with the target's on a shared id.
/// Lengths may differ (added or padding tokens at the end).
fn check_vocab(target: &[String], draft: &[String]) -> Result<()> {
    if let Some(i) = target.iter().zip(draft).position(|(a, b)| a != b) {
        return Err(JoshuaError::ModelLoad(format!(
            "draft model's tokenizer differs from the target's at id {i} ({:?} vs {:?}); \
             a draft model must share the target's vocabulary",
            draft[i], target[i]
        )));
    }
    if target.len() != draft.len() {
        tracing::warn!(
            "draft model vocabulary has {} tokens, the target {}; ids past the shorter \
             list are never accepted",
            draft.len(),
            target.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn vocab_check_accepts_shared_prefix_and_refuses_mismatch() {
        assert!(check_vocab(&v(&["a", "b", "c"]), &v(&["a", "b", "c"])).is_ok());
        assert!(check_vocab(&v(&["a", "b", "c"]), &v(&["a", "b"])).is_ok());
        assert!(check_vocab(&v(&["a", "b"]), &v(&["a", "b", "x"])).is_ok());
        let err = check_vocab(&v(&["a", "b", "c"]), &v(&["a", "x", "c"])).unwrap_err();
        assert!(err.to_string().contains("id 1"), "{err}");
    }
}
