//! Shared **layer-streaming prefill** for the joshua-native MoE loaders.
//!
//! # The problem
//!
//! The engine prefills a long prompt in bounded chunks ([`crate::engine::DEFAULT_PREFILL_CHUNK`]
//! tokens unless `--prefill-chunk` says otherwise)
//! so a `[1, n_prompt, hidden]` activation is never materialized in one piece.
//! Each chunk is fed through *the whole model* ([`crate::model::QuantizedModel::forward`]),
//! so for a 4096-token prompt split into 8 chunks, every layer's weights are
//! read from disk / resident-memory **8 times** — once per chunk.  For an
//! mmap-backed model whose weights lazy-fault from disk, that is up to 8× the
//! weight I/O, and on a GPU it is up to 8× the dense-upload traffic.
//!
//! # The fix
//!
//! Process the prompt **layer-by-layer**: keep each chunk's activation resident
//! between layers, sweep `layer 0` over all chunks (reading layer 0's weights
//! once), then `layer 1` over all chunks, and so on.  Each layer's weights are
//! touched exactly once per prefill.  The intermediate activations (one per
//! chunk) live in RAM the whole time — the same total bytes as one full
//! `[1, n_prompt, hidden]` tensor, so the memory bound of today's chunked
//! prefill is preserved; only the *per-layer transient workspace* (attention
//! scores, MoE dispatch buffers) stays chunk-bounded.
//!
//! # The contract (DRY)
//!
//! The ordering and memory bookkeeping — embed every chunk, then the
//! layer-outer / chunk-inner sweep, KV-position tracking, and the final-head
//! pass — are shared here.  Each loader implements [`StreamPrefill`], whose four
//! methods are the only model-specific surface: embed a chunk, apply one layer
//! to one chunk (returning the next layer's activation), report the layer count,
//! and collapse the last activation to logits.
//!
//! # Correctness of the KV reordering
//!
//! In normal chunked prefill, chunk `c` runs through all layers and attention at
//! layer `l` reads the KV of chunks `0..=c` at layer `l` (written during the same
//! pass).  In layer-streaming, when we sweep chunks `0..n` at layer `l`, the KV
//! for layer `l` accumulates in the same order (chunk 0's keys, then chunk 1's),
//! so chunk `c`'s attention at layer `l` reads exactly the KV `0..=c` — identical
//! values, in identical order.  The reordering is therefore numerically
//! identical for the KV; residual connections are per-chunk-local and unaffected.
//! Bit-for-bit parity is asserted by tests.

use candle_core::{Device, Result, Tensor};

/// A bounded prefill chunk: a slice of prompt tokens plus the absolute KV
/// position of its first token.
pub struct Chunk<'a> {
    pub tokens: &'a [u32],
    pub pos: usize,
}

/// Model-independent surface a loader implements to participate in the shared
/// layer-streaming prefill ([`stream_prefill`]).  Implementations are
/// intentionally minimal — the sweep order, activation array, KV positions and
/// memory bookkeeping all live in the shared runner.
pub trait StreamPrefill {
    /// Number of transformer layers (the outer loop bound).
    fn n_layers(&self) -> usize;

    /// Called once before a sweep starts (before any chunk is embedded):
    /// a hook for per-prefill bookkeeping such as the routing trace.
    fn begin_stream(&mut self) {}

    /// Embed `tokens` into one chunk's activation (the layer-`-1` step),
    /// returning whatever activation shape the loader's layers consume.
    fn embed_chunk(&self, tokens: &[u32], device: &Device) -> Result<Tensor>;

    /// Apply layer `l` to one chunk.  `xs` is that chunk's current activation
    /// (produced by `embed_chunk` for `l == 0`, or by the previous layer for
    /// `l > 0`); `pos` is the chunk's absolute KV start; `tokens` is the
    /// chunk's prompt tokens (needed by MoE loaders that route on token ids).
    /// Must append this chunk's keys/values to layer `l`'s KV cache **and
    /// return the next layer's activation** for this chunk.
    ///
    /// Borrowing: implementations borrow `self` mutably (for the layer's KV),
    /// but must not keep any reference to `xs` beyond the call.
    fn apply_layer_chunk(
        &mut self,
        l: usize,
        xs: &Tensor,
        pos: usize,
        tokens: &[u32],
    ) -> Result<Tensor>;

    /// Collapse the final layer's activation to the prefill logits (narrow to
    /// the last token, apply the final norm + output head, return `[1, vocab]`
    /// on the compute device).
    fn final_logits(&self, last: &Tensor) -> Result<Tensor>;
}

/// Run the layer-streaming prefill over `chunks`, returning the prefill logits
/// ([`StreamPrefill::final_logits`]).
///
/// # Memory
///
/// `chunks.len()` activation tensors are held resident for the whole sweep (one
/// per chunk); each is sized `[.., chunk_len, hidden]`, so the total activation
/// bytes are those of the full prompt — the same ceiling as a single un-chunked
/// forward.  The per-layer transient workspace (attention score matrices, MoE
/// dispatch) is bounded to one chunk at a time.
pub fn stream_prefill<M: StreamPrefill + ?Sized>(
    m: &mut M,
    chunks: &[Chunk],
    device: &Device,
) -> Result<Tensor> {
    let Some(last_chunk) = chunks.last() else {
        // Edge: nothing to prefill (fully-reused KV).  The caller handles the
        // logits from decode; return an empty that the loader's final head
        // path treats as `none`.  Use a 1-token sentinel to keep a valid
        // activation shape.
        let t = m.embed_chunk(&[0u32], device)?;
        return m.final_logits(&t);
    };

    m.begin_stream();
    // Embed every chunk once (layer -1), into the persistent activation array.
    let mut acts: Vec<Tensor> = chunks
        .iter()
        .map(|c| m.embed_chunk(c.tokens, device))
        .collect::<Result<Vec<_>>>()?;

    // Layer-outer, chunk-inner: each layer's weights are touched exactly once
    // across all chunks; each chunk's KV at this layer accumulates in order as
    // we sweep, so attention sees the same KV it would have in linear order.
    for l in 0..m.n_layers() {
        for (c, chunk) in chunks.iter().enumerate() {
            acts[c] = m.apply_layer_chunk(l, &acts[c], chunk.pos, chunk.tokens)?;
        }
    }

    // The prefill prediction is the last chunk's last-token logits.
    let _ = last_chunk;
    m.final_logits(acts.last().expect("non-empty chunks"))
}