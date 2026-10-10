//! The per-session half of Joshua's native decoder loaders.
//!
//! A loaded model splits into immutable weights shared by every concurrent
//! conversation and a small amount of per-session state (each layer's KV
//! cache or recurrent state, plus the hot-expert routing record).  The
//! `deepseek2` and Qwen loaders used to each carry their own copy of the
//! session plumbing around that split — the forward loop with its causal
//! mask and hot-expert refresh, session derivation, cache clearing and
//! truncation, and the layer-streaming prefill hooks.  [`Session`] is the
//! one copy; a loader supplies its weights as a [`LayerStack`].

use std::sync::Arc;

use candle_core::{Device, Result, Tensor};

use crate::residency::ExpertResidency;
use crate::token_embedding::TokenEmbedding;

/// What a decoder layer sees besides its input activations.
pub struct LayerInput<'a> {
    /// Additive causal mask `[1, 1, seq, offset + seq]`, or `None` for a
    /// single-token step.
    pub mask: Option<&'a Tensor>,
    /// Absolute position of the first input token.
    pub offset: usize,
    /// Every token fed so far, through this input (`offset + seq` of them)
    /// — only when [`LayerStack::wants_tokens`], else empty.
    pub tokens: &'a [u32],
    /// Whether this input is a speculative **verification** pass (every
    /// position scored in one sweep) rather than an ordinary prefill chunk.
    ///
    /// A verification pass may be rolled back past its rejected tail, so a
    /// layer whose state is running rather than append-only snapshots what it
    /// is about to overwrite (see [`LayerStack::snapshot_state`]).  Prefill
    /// sets `false`: it is never partially undone.
    pub verify: bool,
}

/// A loader's immutable weights, seen as an embedding, a stack of decoder
/// layers, and an output head.
pub trait LayerStack: Send + Sync + 'static {
    /// One layer's per-session state (KV cache, recurrent state, …).
    type State: Clone + Default;

    /// A copy of the part of `State` that a verification pass advances but an
    /// ordinary truncate cannot rewind — see [`Self::snapshot_state`].
    type Snapshot: Clone + Send;

    /// A copy of one layer's running state at a prompt boundary, so a later
    /// edit of the conversation can resume from there — see
    /// [`Self::checkpoint_state`].
    type Checkpoint: Clone + Send;

    fn n_layers(&self) -> usize;
    /// Routed experts per MoE layer (0 for a dense model).
    fn n_expert(&self) -> usize;
    fn device(&self) -> &Device;
    fn tok_embeddings(&self) -> &TokenEmbedding;
    /// The expert residency backend the hot set is pushed to.
    fn residency(&self) -> &Arc<dyn ExpertResidency>;

    /// Whether layers read the token history ([`LayerInput::tokens`]).  The
    /// session then keeps it, at the cost of a host copy of each input.
    fn wants_tokens(&self) -> bool {
        false
    }

    /// Turn the token embedding `[1, seq, n_embd]` into the residual form
    /// the layers carry (identity by default; hyper-connection models widen
    /// it into parallel streams).
    fn lift(&self, emb: Tensor) -> Result<Tensor> {
        Ok(emb)
    }

    /// Run layer `l` over `xs` (`[1, seq, residual]`), updating
    /// `states[l]`.  Returns the layer output and the routed expert ids this
    /// input used (empty for a dense layer).
    ///
    /// Every layer's state is passed so a layer can read what an earlier
    /// layer published for the same input (GLM-5.2's shared sparse-attention
    /// selection).  Under layer-streaming prefill the earlier layer has seen
    /// every chunk before this one sees the first, so anything published
    /// must be keyed by position.
    fn layer(
        &self,
        l: usize,
        states: &mut [Self::State],
        xs: &Tensor,
        input: &LayerInput<'_>,
    ) -> Result<(Tensor, Vec<u32>)>;

    /// Final norm + output head: `[1, n, residual]` → `[1, n, vocab]` (f32).
    fn head(&self, xs: &Tensor) -> Result<Tensor>;

    /// Whether every layer's state can be cut back to an arbitrary prefix
    /// ([`Self::truncate_state`]); recurrent layers cannot.
    fn can_truncate(&self) -> bool {
        true
    }

    /// Keep only the first `keep` positions of one layer's state.
    fn truncate_state(state: &mut Self::State, keep: usize) -> Result<()>;

    /// Arm one layer's state to record what a verification pass is about to
    /// overwrite, before that pass runs.
    ///
    /// The default is a no-op for append-only state (an attention layer's KV
    /// cache rewinds through [`Self::truncate_state`] alone).  A layer with
    /// running state installs an empty capture here; its
    /// [`Self::snapshot_state`] then reports the filled one.
    fn begin_verify(_state: &mut Self::State) {}

    /// Release whatever [`Self::begin_verify`] armed.
    ///
    /// The session keeps its own copy in the checkpoint, so the per-layer slot
    /// can be dropped as soon as that copy is taken; otherwise a pooled session
    /// would sit on a full copy of every recurrent layer's state — the matrices
    /// and the verification inputs — long after decoding finished.
    fn clear_verify(_state: &mut Self::State) {}

    /// Copy whatever [`Self::truncate_state`] cannot rewind on its own — the
    /// running state of a recurrent layer — so a verification pass can be
    /// rolled back.
    ///
    /// Returns `None` when the loader has no such state, or when no
    /// verification pass has been armed ([`Self::begin_verify`]).  A loader
    /// with no restore path keeps [`Self::can_rewind`] `false`: a multi-token
    /// verification pass would otherwise leave the tokens the target model
    /// rejected folded permanently into the recurrence.
    fn snapshot_state(_state: &Self::State) -> Option<Self::Snapshot> {
        None
    }

    /// Roll this layer's state back to what it was `retained` tokens into the
    /// verification pass that produced `snap`, discarding the pass's rejected
    /// tail.
    ///
    /// `retained` is `keep - pass_offset` — the number of the pass's tokens
    /// that survived.  An implementation that snapshots running state must
    /// **replay** those `retained` tokens into the restored state: restoring
    /// the pre-pass snapshot alone would also discard the tokens that were
    /// accepted, leaving the recurrence behind plain decoding.  The snapshot
    /// therefore carries the pass's own recurrence inputs for that replay —
    /// the same reason `deepseek4`'s capture keeps the pass's projections
    /// instead of recomputing them.
    fn restore_state(
        _state: &mut Self::State,
        _snap: Self::Snapshot,
        _retained: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Whether a just-run verification pass can be rolled back to any earlier
    /// position: every layer either rewinds on its own ([`Self::can_truncate`])
    /// or exposes snapshot/restore.
    fn can_rewind(&self) -> bool {
        self.can_truncate()
    }

    /// Copy the running state of one layer — what [`Self::truncate_state`]
    /// cannot rewind — as it stands after a prefill, so the session can later
    /// rewind to exactly this position (a prefix checkpoint).
    ///
    /// `None` for a layer whose state is append-only (it rewinds through
    /// [`Self::truncate_state`] alone) and for a loader without running state.
    fn checkpoint_state(_state: &Self::State) -> Option<Self::Checkpoint> {
        None
    }

    /// Reinstate the running state [`Self::checkpoint_state`] copied.  The
    /// session has already cut the layer's append-only state back to the
    /// checkpoint's position with [`Self::truncate_state`].
    fn restore_checkpoint(_state: &mut Self::State, _checkpoint: &Self::Checkpoint) -> Result<()> {
        Ok(())
    }
}

/// Keep the final token's routed experts out of an input's `routed` ids
/// (token-major, `k` per token — see [`crate::moe::dispatch_prefill_with`]).
fn record_last_routed(slot: &mut Vec<u32>, routed: &[u32], seq_len: usize) {
    slot.clear();
    if seq_len > 0 && !routed.is_empty() && routed.len().is_multiple_of(seq_len) {
        slot.extend_from_slice(&routed[routed.len() - routed.len() / seq_len..]);
    }
}

/// How many prefix checkpoints a session of a recurrent model keeps by
/// default (see [`Session::set_prefix_checkpoints`]).
pub const DEFAULT_PREFIX_CHECKPOINTS: usize = 4;

/// The running state of every layer as it stood after `pos` tokens: where a
/// recurrent model can rewind to although its state keeps no history.
struct PrefixCheckpoint<C> {
    /// Tokens fed when the copy was taken.
    pos: usize,
    /// Per layer, its running state (`None` for an append-only layer).
    layers: Vec<Option<C>>,
}

/// A bounded set of prefix checkpoints, ordered by position.
///
/// A model with recurrent layers cannot cut its state back to an arbitrary
/// prefix, so an edited conversation — an agent harness replacing the last
/// message or an old tool result — used to re-read the whole prompt.  A copy
/// of the running state at the end of every prefill (each turn's prompt
/// boundary) lets the edit resume from the closest boundary before it; the
/// attention caches simply truncate to the same position.
struct PrefixCheckpoints<C> {
    /// Most checkpoints kept; `0` disables them.
    budget: usize,
    points: Vec<PrefixCheckpoint<C>>,
}

impl<C> PrefixCheckpoints<C> {
    fn new(budget: usize) -> Self {
        Self {
            budget,
            points: Vec::new(),
        }
    }

    /// Drop every checkpoint past `pos`: those positions are about to be
    /// rewritten (or were cut away), so a copy taken there no longer
    /// describes this session's history.  A checkpoint at `pos` itself stays.
    fn invalidate_after(&mut self, pos: usize) {
        self.points.retain(|p| p.pos <= pos);
    }

    /// The latest checkpoint at or before `keep`.
    fn best(&self, keep: usize) -> Option<&PrefixCheckpoint<C>> {
        self.points.iter().rev().find(|p| p.pos <= keep)
    }

    /// Add a checkpoint (replacing one at the same position), then thin the
    /// set back to the budget.
    fn insert(&mut self, cp: PrefixCheckpoint<C>) {
        self.points.retain(|p| p.pos != cp.pos);
        let at = self.points.partition_point(|p| p.pos < cp.pos);
        self.points.insert(at, cp);
        while self.points.len() > self.budget {
            self.thin();
        }
    }

    /// Remove the checkpoint whose loss widens the gap to the one before it
    /// the least — so the survivors stay spread over the conversation instead
    /// of crowding its end — never the newest, which serves the most common
    /// edit (the last message replaced).
    fn thin(&mut self) {
        let n = self.points.len();
        if n <= 1 {
            self.points.clear();
            return;
        }
        let victim = (0..n - 1)
            .min_by_key(|&i| {
                let prev = if i == 0 { 0 } else { self.points[i - 1].pos };
                self.points[i + 1].pos - prev
            })
            .expect("n > 1");
        self.points.remove(victim);
    }
}

/// A just-run multi-position (verification) pass that is still pending
/// rollback: the position span it covered and each layer's snapshot.
///
/// The same shape `deepseek4` keeps in its own `VerifyCheckpoint`, so a
/// rejected draft leaves both loader families in the state plain decoding
/// would have produced.
struct VerifyCheckpoint<S> {
    /// Absolute position of the pass's first token.
    offset: usize,
    /// Number of tokens the pass fed.
    seq: usize,
    /// Per layer, the snapshot taken before the pass ran (`None` where the
    /// layer rewinds on its own).  Kept layer-aligned so a rollback walks the
    /// layers in lockstep with the state.
    layers: Vec<Option<S>>,
}

impl<S> VerifyCheckpoint<S> {
    /// How many of the pass's tokens survive a rollback to `keep`, or `None`
    /// when `keep` lies outside the span this pass covered.
    fn retained(&self, keep: usize) -> Option<usize> {
        (self.offset..=self.offset + self.seq)
            .contains(&keep)
            .then_some(keep - self.offset)
    }
}

/// A conversation over shared weights `W`: the weights behind one `Arc` plus
/// this session's per-layer state and routing record.
pub struct Session<W: LayerStack> {
    weights: Arc<W>,
    state: Vec<W::State>,
    /// Every token fed so far, when the weights [want it](LayerStack::wants_tokens).
    tokens: Vec<u32>,
    /// Routing-frequency LRU hot-expert cache (shared bookkeeping; budget set
    /// after load via [`Session::set_pin_hot_experts`], CLI flag
    /// `--pin-hot-experts`).  Records routing, re-selects the hot set every
    /// [`crate::hot_experts::REFRESH_STEPS`] decode steps, and reports newly
    /// hot experts for the residency backend.
    hot_experts: crate::hot_experts::HotExpertCache,
    /// Causal-mask builder on the weights' device: the streamed prefill asks
    /// for the same `(chunk, position)` mask once per layer, so the pattern is
    /// derived on-device from cached position ranges instead of re-filled and
    /// re-uploaded per call (see [`crate::moe::CausalMask`]).
    mask: crate::moe::CausalMask,
    /// The last multi-position pass, pending rollback.  Only a verification
    /// pass arms it; an ordinary prefill leaves it `None` because nothing
    /// partially undoes a prefill.
    verify: Option<VerifyCheckpoint<W::Snapshot>>,
    /// Copies of the running state at prompt boundaries, so a model with
    /// recurrent layers can rewind an edited conversation (see
    /// [`PrefixCheckpoints`]).  Unused when every layer truncates on its own.
    prefix: PrefixCheckpoints<W::Checkpoint>,
    /// Whether decode fires the speculative expert prefetch (see
    /// [`Session::set_speculative_expert_prefetch`]).
    predict_experts: bool,
    /// Per layer, the experts the last input's final token routed to: the
    /// prediction for the next step.  Empty until a layer has routed (and for
    /// dense layers).
    last_routed: Vec<Vec<u32>>,
}

impl<W: LayerStack> Session<W> {
    /// The first session over freshly loaded weights.
    pub fn from_weights(weights: W) -> Self {
        let weights = Arc::new(weights);
        let n = weights.n_layers();
        Self {
            state: vec![W::State::default(); n],
            tokens: Vec::new(),
            hot_experts: crate::hot_experts::HotExpertCache::new(n, weights.n_expert(), 0),
            mask: crate::moe::CausalMask::new(weights.device()),
            verify: None,
            prefix: PrefixCheckpoints::new(DEFAULT_PREFIX_CHECKPOINTS),
            predict_experts: false,
            last_routed: vec![Vec::new(); n],
            weights,
        }
    }

    /// The shared weights.
    pub fn weights(&self) -> &W {
        &self.weights
    }

    /// A fresh session over the same weights: shares every weight tensor
    /// (one `Arc` clone — no read, no upload) and starts with empty state and
    /// a fresh routing record under the same hot-expert budget.
    pub fn new_session(&self) -> Self {
        Self {
            weights: Arc::clone(&self.weights),
            state: vec![W::State::default(); self.state.len()],
            tokens: Vec::new(),
            hot_experts: crate::hot_experts::HotExpertCache::new(
                self.state.len(),
                self.weights.n_expert(),
                self.hot_experts.budget(),
            ),
            mask: crate::moe::CausalMask::new(self.weights.device()),
            verify: None,
            prefix: PrefixCheckpoints::new(self.prefix.budget),
            predict_experts: self.predict_experts,
            last_routed: vec![Vec::new(); self.state.len()],
        }
    }

    /// Number of sessions (including this one) sharing these weights.
    pub fn shared_session_count(&self) -> usize {
        Arc::strong_count(&self.weights)
    }

    /// Whether the token-embedding table is held quantized (diagnostics).
    pub fn embeddings_quantized(&self) -> bool {
        self.weights.tok_embeddings().is_quantized()
    }

    fn embed(&self, ids: &Tensor, seq_len: usize) -> Result<Tensor> {
        let emb = self.weights.tok_embeddings();
        let xs = emb.forward(&ids.flatten_all()?)?.reshape((1, seq_len, emb.hidden()?))?;
        self.weights.lift(xs)
    }

    /// Record `new` as the tokens at `offset..` of the history (when the
    /// weights want it).  Re-feeding a position overwrites what followed.
    fn record_tokens(&mut self, offset: usize, new: &[u32]) -> Result<()> {
        if !self.weights.wants_tokens() {
            return Ok(());
        }
        if self.tokens.len() < offset {
            candle_core::bail!(
                "token history covers {} positions but input starts at {offset}",
                self.tokens.len()
            );
        }
        self.tokens.truncate(offset);
        self.tokens.extend_from_slice(new);
        Ok(())
    }

    /// Forward pass. `input` is `[1, seq_len]`; `offset` is the cache
    /// position of the first input token.  Returns the last position's
    /// logits, `[1, vocab]`.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        self.forward_impl(input, offset, false)
    }

    /// [`Self::forward`], returning the logits of **every** input position,
    /// `[1, seq_len, vocab]` — the speculative-decoding verification pass,
    /// which scores a whole draft in one step (see [`crate::speculative`]).
    pub fn forward_all_logits(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        self.forward_impl(input, offset, true)
    }

    fn forward_impl(&mut self, input: &Tensor, offset: usize, all_logits: bool) -> Result<Tensor> {
        let (_b, seq_len) = input.dims2()?;
        // A verification pass is a (multi-token) decode step, not a prefill:
        // it advances the hot-expert cadence like a single-token step.
        let decode = seq_len == 1 || all_logits;
        let w = Arc::clone(&self.weights);
        if w.wants_tokens() {
            let ids: Vec<u32> = input.flatten_all()?.to_vec1()?;
            self.record_tokens(offset, &ids)?;
        }
        // Positions from `offset` on are rewritten by this pass.
        self.prefix.invalidate_after(offset);
        let mut xs = self.embed(input, seq_len)?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask.mask(seq_len, offset)?)
        };

        // Routing-frequency hot-expert cache: every
        // crate::hot_experts::REFRESH_STEPS decode steps, re-select the
        // most-used experts (recency as the tie-break) and WILLNEED their
        // pages, so the common routing path stays resident instead of
        // faulting from disk each step.  Runs before the layer loop so the
        // prefetch has a full step of compute to stream in behind.
        let step = self.hot_experts.begin_step(decode);
        if self.hot_experts.refresh_due(decode) {
            for (l, e) in self.hot_experts.refresh() {
                // Protect the routing-frequency hot set from LRU eviction in a
                // device slot pool (#62), then make it resident.  No-op on
                // CPU/host residency backends.
                w.residency().mark_hot(l, e);
                w.residency().acquire(l, e);
            }
        }

        // Speculative next-step expert prefetch: routing is temporally local,
        // so the step about to run routes largely to the experts each layer
        // chose last step.  Hinting them all before layer 0 gives layer `l`'s
        // pages `l` layers of compute to stream in behind, instead of faulting
        // one 4 KiB page at a time inside the expert matmul.  A wrong guess
        // costs only the advice call (`deepseek4` does the same).
        if decode && self.predict_experts {
            for (l, ids) in self.last_routed.iter().enumerate() {
                for &e in ids {
                    w.residency().prefetch_hint(l as u32, e);
                }
            }
        }

        let layer_input = LayerInput {
            mask: mask.as_ref(),
            offset,
            tokens: if w.wants_tokens() { &self.tokens[..offset + seq_len] } else { &[] },
            verify: all_logits,
        };
        // A verification pass may be rolled back past its rejected tail, so arm
        // every layer to record what it is about to overwrite before the loop
        // starts.  A prefill is never partially undone, so it arms nothing and
        // drops any checkpoint left by an earlier pass.
        self.verify = None;
        if all_logits {
            for s in self.state.iter_mut() {
                W::begin_verify(s);
            }
        }
        for l in 0..self.state.len() {
            let (out, routed) = w.layer(l, &mut self.state, &xs, &layer_input)?;
            self.hot_experts.record(l, &routed, step);
            if self.predict_experts {
                record_last_routed(&mut self.last_routed[l], &routed, seq_len);
            }
            xs = out;
        }
        if all_logits {
            // Collect what the pass recorded, layer-aligned, and remember the
            // span it covered so `truncate_kv_cache` can roll back inside it.
            let layers = self.state.iter().map(W::snapshot_state).collect();
            self.verify = Some(VerifyCheckpoint {
                offset,
                seq: seq_len,
                layers,
            });
            // The checkpoint holds its own copy, so the per-layer slots are dead
            // weight from here; dropping them keeps a pooled session from
            // sitting on a second copy of every recurrent layer's state.
            for s in self.state.iter_mut() {
                W::clear_verify(s);
            }
        } else if seq_len > 1 {
            // A prefill chunk: its end may be where an edited conversation
            // later diverges.
            self.take_prefix_checkpoint(offset + seq_len);
        }

        if all_logits {
            w.head(&xs)
        } else {
            w.head(&xs.narrow(1, seq_len - 1, 1)?)?.squeeze(1)
        }
    }

    /// Set the routing-frequency hot-expert cache budget (experts kept
    /// resident).  Call once after load, before serving; routing is recorded
    /// from the first forward pass and the pinned set is re-selected every
    /// [`crate::hot_experts::REFRESH_STEPS`] decode steps.
    pub fn set_pin_hot_experts(&mut self, n: usize) {
        self.hot_experts.set_budget(n);
    }

    /// Set how many prefix checkpoints this session (and every session
    /// derived from it) keeps; `0` turns them off.  Only a model with
    /// recurrent layers takes any: each holds one copy of every recurrent
    /// layer's running state.  Default [`DEFAULT_PREFIX_CHECKPOINTS`].
    pub fn set_prefix_checkpoints(&mut self, n: usize) {
        self.prefix.budget = n;
        while self.prefix.points.len() > n {
            self.prefix.thin();
        }
    }

    /// Fire a `MADV_WILLNEED` for each MoE layer's predicted experts — the
    /// ones its previous step routed to — before every decode step.  For a
    /// model whose experts page in on demand (larger than RAM); on a model
    /// that is already resident it would only add advice calls, so it is
    /// off by default.  Inherited by derived sessions.
    pub fn set_speculative_expert_prefetch(&mut self, on: bool) {
        self.predict_experts = on;
    }

    /// Per layer, the experts the last input's final token routed to — the
    /// speculative prefetch's prediction (diagnostics; empty unless
    /// [`Self::set_speculative_expert_prefetch`] is on).
    pub fn last_routed_experts(&self) -> &[Vec<u32>] {
        &self.last_routed
    }

    /// Positions of the prefix checkpoints this session holds (diagnostics).
    pub fn prefix_checkpoint_positions(&self) -> Vec<usize> {
        self.prefix.points.iter().map(|p| p.pos).collect()
    }

    /// The longest prefix of at most `keep` tokens that
    /// [`Self::truncate_kv_cache`] can rewind this session to: `keep` itself
    /// when every layer truncates on its own, else the latest prefix
    /// checkpoint at or before it (`0` when there is none).
    pub fn rewind_target(&self, keep: usize) -> usize {
        if self.weights.can_truncate() {
            keep
        } else {
            self.prefix.best(keep).map_or(0, |p| p.pos)
        }
    }

    /// Copy every layer's running state as it stands after `pos` tokens.
    fn take_prefix_checkpoint(&mut self, pos: usize) {
        if self.prefix.budget == 0 || self.weights.can_truncate() {
            return;
        }
        let layers: Vec<_> = self.state.iter().map(W::checkpoint_state).collect();
        if layers.iter().all(Option::is_none) {
            return;
        }
        self.prefix.insert(PrefixCheckpoint { pos, layers });
    }

    /// Number of experts the residency backend can hold resident
    /// (informational on CPU; a phase-5 auto-sizing input on devices).
    pub fn expert_residency_capacity(&self) -> usize {
        self.weights.residency().capacity()
    }

    /// Reset every layer's state so this instance can serve an unrelated
    /// prompt.
    pub fn clear_kv_cache(&mut self) {
        for s in self.state.iter_mut() {
            *s = W::State::default();
        }
        self.tokens.clear();
        // A pending checkpoint refers into the state just replaced; drop it.
        self.verify = None;
        self.prefix.points.clear();
        for ids in self.last_routed.iter_mut() {
            ids.clear();
        }
    }

    /// Whether this instance's state can be cut back to an *arbitrary* prefix.
    ///
    /// True for the append-only attention loaders.  False for a model with
    /// recurrent layers, whose running state reaches back over nothing — that
    /// one can still roll back a just-run verification pass, which is what
    /// [`Self::supports_rewind`] answers.
    pub fn supports_truncate(&self) -> bool {
        self.weights.can_truncate()
    }

    /// Whether a speculative verification pass can be rolled back — the stronger
    /// property speculative decoding needs.
    pub fn supports_rewind(&self) -> bool {
        self.weights.can_rewind()
    }

    /// Whether a verification pass is still pending rollback.
    pub fn has_pending_verify(&self) -> bool {
        self.verify.is_some()
    }

    /// Drop a pending verification checkpoint.
    ///
    /// A pass whose drafts were all accepted needs no rollback, but its
    /// snapshots — a full copy of every recurrent layer's state — would stay
    /// resident until the next forward pass replaced them, including while the
    /// session sat in the engine's warm pool.  The decode loop calls this once
    /// it is done with the step.
    pub fn discard_verify(&mut self) {
        self.verify = None;
        for s in self.state.iter_mut() {
            W::clear_verify(s);
        }
    }

    /// Keep only the first `keep` fed tokens of every layer's state.
    ///
    /// Two shapes are exact:
    ///
    /// * **Inside a pending verification pass** — `keep` lands in the span
    ///   [`Self::forward_all_logits`] last covered.  Each layer rewinds its
    ///   append-only state through [`LayerStack::truncate_state`], and a layer
    ///   with running state additionally restores its pre-pass snapshot and
    ///   replays the accepted prefix, so the result matches plain decoding
    ///   that never saw the rejected tokens.
    /// * **An arbitrary earlier prefix** — only when every layer rewinds on its
    ///   own ([`Self::supports_truncate`]); this is the engine's edited-context
    ///   reuse, which rewinds a whole conversation.  A recurrent model refuses:
    ///   its state has no history to reach back into, and pretending otherwise
    ///   would silently desynchronise it from the KV cache.
    ///
    /// Used by the engine's edited-context prefix reuse and by speculative
    /// decoding's rollback.
    pub fn truncate_kv_cache(&mut self, keep: usize) -> Result<()> {
        match self.verify.take() {
            Some(mut checkpoint) => {
                let retained = checkpoint.retained(keep).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "cannot roll back to {keep} tokens: the pending verification pass covered \
                         {}..={}",
                        checkpoint.offset,
                        checkpoint.offset + checkpoint.seq
                    ))
                })?;
                for (i, s) in self.state.iter_mut().enumerate() {
                    W::truncate_state(s, keep)?;
                    if let Some(snap) = checkpoint.layers.get_mut(i).and_then(Option::take) {
                        W::restore_state(s, snap, retained)?;
                    }
                }
            }
            None if self.weights.can_truncate() => {
                for s in self.state.iter_mut() {
                    W::truncate_state(s, keep)?;
                }
            }
            None => {
                // Running state has no history to reach back into; only a
                // prefix checkpoint taken at exactly `keep` can serve it.
                let Some(checkpoint) = self.prefix.points.iter().find(|p| p.pos == keep) else {
                    candle_core::bail!(
                        "this model's recurrent layer state cannot be truncated to {keep} \
                         tokens: no prefix checkpoint there (see `rewind_target`), and only a \
                         speculative verification pass can be rolled back otherwise"
                    );
                };
                for (s, cp) in self.state.iter_mut().zip(&checkpoint.layers) {
                    W::clear_verify(s);
                    W::truncate_state(s, keep)?;
                    if let Some(cp) = cp {
                        W::restore_checkpoint(s, cp)?;
                    }
                }
            }
        }
        self.prefix.invalidate_after(keep);
        self.tokens.truncate(keep);
        Ok(())
    }
}

// ─── Layer-streaming prefill (shared framework) ──────────────────────────────
impl<W: LayerStack> crate::stream_prefill::StreamPrefill for Session<W> {
    fn n_layers(&self) -> usize {
        self.state.len()
    }

    fn embed_chunk(&self, tokens: &[u32], device: &Device) -> Result<Tensor> {
        self.embed(
            &Tensor::new(tokens, device)?.unsqueeze(0)?,
            tokens.len(),
        )
    }

    fn apply_layer_chunk(
        &mut self,
        l: usize,
        xs: &Tensor,
        pos: usize,
        tokens: &[u32],
    ) -> Result<Tensor> {
        // The sweep is layer-outer: layer 0 sees each chunk first, in order,
        // so that is when its tokens join the history.
        if l == 0 {
            self.record_tokens(pos, tokens)?;
            self.prefix.invalidate_after(pos);
        }
        // Chunk-local causal mask: this chunk's tokens attend to all prior
        // positions ([chunk_len, chunk_len + pos]) — the same causal mask the
        // chunked prefill builds for the same `pos`.
        let seq = xs.dim(1)?;
        let mask = self.mask.mask(seq, pos)?;
        let w = Arc::clone(&self.weights);
        let tokens = if w.wants_tokens() { &self.tokens[..pos + seq] } else { &[][..] };
        // A prefill is never partially undone, so nothing here arms a rewind.
        let input = LayerInput {
            mask: Some(&mask),
            offset: pos,
            tokens,
            verify: false,
        };
        let (out, routed) = w.layer(l, &mut self.state, xs, &input)?;
        let step = self.hot_experts.begin_step(false);
        self.hot_experts.record(l, &routed, step);
        // Chunks reach a layer in order, so the last one leaves the prompt's
        // final token as the prediction for the first decode step.
        if self.predict_experts {
            record_last_routed(&mut self.last_routed[l], &routed, seq);
        }
        Ok(out)
    }

    fn end_stream(&mut self, end: usize) {
        // Every layer has seen every chunk: the state now stands at the
        // prompt's end, the boundary an edit of the next turn resumes from.
        self.take_prefix_checkpoint(end);
    }

    fn final_logits(&self, last: &Tensor) -> Result<Tensor> {
        let seq_len = last.dim(1)?;
        self.weights
            .head(&last.narrow(1, seq_len - 1, 1)?)?
            .squeeze(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoints(budget: usize, at: &[usize]) -> PrefixCheckpoints<()> {
        let mut c = PrefixCheckpoints::new(budget);
        for &pos in at {
            c.insert(PrefixCheckpoint { pos, layers: vec![Some(())] });
        }
        c
    }

    fn positions(c: &PrefixCheckpoints<()>) -> Vec<usize> {
        c.points.iter().map(|p| p.pos).collect()
    }

    /// Thinning drops the checkpoint whose loss opens the smallest gap and
    /// never the newest, so the survivors stay spread over the conversation.
    #[test]
    fn prefix_checkpoints_thin_to_a_spread() {
        let c = checkpoints(4, &[100, 200, 300, 10_000, 10_100]);
        assert_eq!(positions(&c), [200, 300, 10_000, 10_100]);
        // Out-of-order and repeated inserts stay sorted and unique.
        let c = checkpoints(3, &[50, 10, 50, 30]);
        assert_eq!(positions(&c), [10, 30, 50]);
        let c = checkpoints(1, &[10, 20, 30]);
        assert_eq!(positions(&c), [30]);
        assert!(checkpoints(0, &[10]).points.is_empty());
    }

    #[test]
    fn prefix_checkpoints_lookup_and_invalidation() {
        let mut c = checkpoints(4, &[5, 11, 20]);
        assert_eq!(c.best(4).map(|p| p.pos), None);
        assert_eq!(c.best(11).map(|p| p.pos), Some(11));
        assert_eq!(c.best(19).map(|p| p.pos), Some(11));
        c.invalidate_after(11);
        assert_eq!(positions(&c), [5, 11]);
    }

    /// Only the final token's `k` ids are kept as the prediction.
    #[test]
    fn last_routed_keeps_the_final_token() {
        let mut slot = vec![9];
        record_last_routed(&mut slot, &[1, 2, 3, 4, 5, 6], 3);
        assert_eq!(slot, [5, 6]);
        record_last_routed(&mut slot, &[7, 8], 1);
        assert_eq!(slot, [7, 8]);
        // A dense layer (no ids) clears the prediction.
        record_last_routed(&mut slot, &[], 1);
        assert!(slot.is_empty());
    }
}
