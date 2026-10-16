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

    /// A single-token decode step through a fused device kernel (see
    /// [`crate::megakernel`]): every layer plus the head over the embedded
    /// token `xs` (`[1, 1, residual]`) at `offset`, updating `states`.
    /// Returns the logits `[1, vocab]`, or `None` (the default) to take the
    /// layer-by-layer path, in which case `states` must be untouched.
    fn fused_decode(&self, _states: &mut [Self::State], _xs: &Tensor, _offset: usize) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Whether the weights carry a NextN / MTP draft block (see
    /// [`Session::mtp_draft`]).  The three methods below serve only such
    /// weights.
    fn has_mtp(&self) -> bool {
        false
    }

    /// The final norm alone: `[1, n, residual]` → the normed hidden
    /// `[1, n, hidden]` the head reads ([`Self::head`] is
    /// `head_logits(head_norm(xs))`).
    fn head_norm(&self, _xs: &Tensor) -> Result<Tensor> {
        candle_core::bail!("these weights have no MTP block")
    }

    /// The output head over normed rows: `[1, n, hidden]` → `[1, n, vocab]`
    /// (f32).
    fn head_logits(&self, _normed: &Tensor) -> Result<Tensor> {
        candle_core::bail!("these weights have no MTP block")
    }

    /// The MTP block over `n` rows at `input.offset..`: `emb` is the rows'
    /// token embeddings and `seeds` the normed trunk hidden of each row's
    /// previous position, both `[1, n, hidden]`.  Appends the block's KV to
    /// `state` and returns each row's normed output `[1, n, hidden]` (the
    /// head's input, and a chained draft's seed).
    fn mtp_forward(
        &self,
        _state: &mut Self::State,
        _emb: &Tensor,
        _seeds: &Tensor,
        _input: &LayerInput<'_>,
    ) -> Result<Tensor> {
        candle_core::bail!("these weights have no MTP block")
    }

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

    /// A copy of one layer's state holding its first `keep` positions on
    /// `device`, sharing nothing that a later write to either copy could
    /// disturb: the KV cache is written in place, so its rows are copied into
    /// fresh buffers (see [`crate::attention::fork_kv`]).  Running state is
    /// copied as it stands; [`Session::snapshot`] reinstates a prefix
    /// checkpoint first when `keep` lies behind it.  A pending verification
    /// capture is not carried over.
    fn fork_state(state: &Self::State, keep: usize, device: &Device) -> Result<Self::State>;

    /// Bytes one layer's state holds, for the engine's prefix-cache budget.
    fn state_bytes(state: &Self::State) -> u64;
}

/// `t` on `device`, sharing its storage when it is already there.  For the
/// state a forward replaces rather than writes in place (recurrent tensors,
/// indexer keys), sharing is a safe copy.
pub fn tensor_on(t: &Tensor, device: &Device) -> Result<Tensor> {
    if t.device().same_device(device) {
        Ok(t.clone())
    } else {
        t.to_device(device)
    }
}

/// A session's MTP block state: the block's own KV, kept in step with the
/// trunk so a draft can start at the next position.  Every trunk pass also
/// runs its rows through the block, seeded with the trunk's normed hidden of
/// each row's previous position — what the block was trained on — so the
/// block's KV holds exactly what the model's authors' decoder would.
struct MtpState<S> {
    state: S,
    /// Positions the block's KV covers.
    len: usize,
    /// The trunk's normed hidden at position `len - 1`: the seed of the row
    /// at `len`.  `None` when unknown (after a rewind to a checkpoint or a
    /// restored snapshot cut short), which seeds that row with zeros — a
    /// worse draft, never a wrong output, since drafts are verified.
    seed: Option<Tensor>,
    /// A pending verification pass's normed trunk rows `(offset, rows)`, so
    /// its rollback can recover the seed at the position it keeps.
    verify_rows: Option<(usize, Tensor)>,
}

impl<S: Default> MtpState<S> {
    fn new() -> Self {
        Self { state: S::default(), len: 0, seed: None, verify_rows: None }
    }
}

/// Run the trunk rows `offset..offset + n` through the MTP block: `emb` the
/// rows' embeddings, `normed` the trunk's normed hidden of each row.  Skipped
/// (the block falls behind and stops drafting until the session is cleared)
/// when the block missed earlier positions.
fn mtp_advance<W: LayerStack>(
    w: &W,
    m: &mut MtpState<W::State>,
    emb: &Tensor,
    normed: &Tensor,
    offset: usize,
    mask: Option<&Tensor>,
    verify: bool,
) -> Result<()> {
    if m.len < offset {
        return Ok(());
    }
    if m.len > offset {
        W::truncate_state(&mut m.state, offset)?;
        m.len = offset;
        m.seed = None;
    }
    let n = normed.dim(1)?;
    let first = match &m.seed {
        Some(s) => s.clone(),
        None => normed.narrow(1, 0, 1)?.zeros_like()?,
    };
    let seeds = if n > 1 {
        Tensor::cat(&[&first, &normed.narrow(1, 0, n - 1)?], 1)?
    } else {
        first
    };
    let input = LayerInput { mask, offset, tokens: &[], verify: false };
    w.mtp_forward(&mut m.state, emb, &seeds, &input)?;
    m.len = offset + n;
    // A copy, so the seed does not hold a whole prefill's rows alive.
    m.seed = Some(normed.narrow(1, n - 1, 1)?.copy()?);
    m.verify_rows = verify.then(|| (offset, normed.clone()));
    Ok(())
}

/// Token `idx`'s routed experts in an input's `routed` ids (token-major, `k`
/// per token — see [`crate::moe::dispatch_prefill_with`]); empty when the
/// layer is dense or `idx` is out of range.
fn routed_at(routed: &[u32], seq_len: usize, idx: usize) -> &[u32] {
    if idx < seq_len && !routed.is_empty() && routed.len().is_multiple_of(seq_len) {
        let k = routed.len() / seq_len;
        &routed[idx * k..(idx + 1) * k]
    } else {
        &[]
    }
}

/// Keep the final token's routed experts out of an input's `routed` ids.
fn record_last_routed(slot: &mut Vec<u32>, routed: &[u32], seq_len: usize) {
    slot.clear();
    slot.extend_from_slice(routed_at(routed, seq_len, seq_len.wrapping_sub(1)));
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

/// A conversation's state after `pos` tokens, detached from any session: what
/// the engine's prefix cache keeps between requests ([`Session::snapshot`],
/// [`Session::restore`]).  It shares no buffer a live session writes to.
pub struct SessionSnapshot<W: LayerStack> {
    /// The weights the state was computed with; a restore into a session over
    /// other weights is refused.
    weights: Arc<W>,
    pos: usize,
    state: Vec<W::State>,
    /// The token history, when the weights [want it](LayerStack::wants_tokens).
    tokens: Vec<u32>,
    /// The MTP block's KV and seed, when the session kept one up to `pos`.
    mtp: Option<(W::State, Option<Tensor>)>,
    device: Device,
    bytes: u64,
}

/// A copy of an MTP block's state at `keep` positions on `device`, with its
/// seed when that is the position the seed belongs to.
fn fork_mtp<W: LayerStack>(
    state: &W::State,
    seed: Option<&Tensor>,
    seed_at: bool,
    keep: usize,
    device: &Device,
) -> Result<(W::State, Option<Tensor>)> {
    let seed = match seed.filter(|_| seed_at) {
        Some(t) => Some(t.to_device(device)?.copy()?),
        None => None,
    };
    Ok((W::fork_state(state, keep, device)?, seed))
}

fn mtp_bytes<W: LayerStack>(mtp: &Option<(W::State, Option<Tensor>)>) -> u64 {
    mtp.as_ref().map_or(0, |(s, seed)| {
        W::state_bytes(s) + seed.as_ref().map_or(0, crate::attention::tensor_bytes)
    })
}

impl<W: LayerStack> SessionSnapshot<W> {
    /// Tokens the state covers.
    pub fn len(&self) -> usize {
        self.pos
    }

    /// Whether the snapshot covers no tokens (never true for one
    /// [`Session::snapshot`] returned).
    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }

    /// Bytes the state holds.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// An independent copy on `device`, for a snapshot the cache keeps while
    /// a session continues from it.
    pub fn duplicate(&self, device: &Device) -> Result<Self> {
        let state = self
            .state
            .iter()
            .map(|s| W::fork_state(s, self.pos, device))
            .collect::<Result<Vec<_>>>()?;
        let mtp = match &self.mtp {
            Some((s, seed)) => Some(fork_mtp::<W>(s, seed.as_ref(), true, self.pos, device)?),
            None => None,
        };
        Ok(Self {
            weights: Arc::clone(&self.weights),
            pos: self.pos,
            bytes: state.iter().map(W::state_bytes).sum::<u64>() + mtp_bytes::<W>(&mtp),
            state,
            tokens: self.tokens.clone(),
            mtp,
            device: device.clone(),
        })
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
    /// Per layer, every token's routed experts from a pending verification
    /// pass, so a rollback can predict from the token it keeps rather than
    /// a rejected draft.
    verify_routed: Vec<Vec<u32>>,
    /// The MTP block's state while MTP drafting is on (see
    /// [`Session::set_mtp`]).
    mtp: Option<MtpState<W::State>>,
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
            verify_routed: vec![Vec::new(); n],
            mtp: None,
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
            verify_routed: vec![Vec::new(); self.state.len()],
            mtp: self.mtp.as_ref().map(|_| MtpState::new()),
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
        // The MTP block reads each row's embedding (the trunk carries a
        // plain residual whenever it has the block).
        let emb = self.mtp.is_some().then(|| xs.clone());
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
        if seq_len == 1 && !all_logits && self.mtp.is_none() {
            if let Some(logits) = w.fused_decode(&mut self.state, &xs, offset)? {
                return Ok(logits);
            }
        }
        for l in 0..self.state.len() {
            let (out, routed) = w.layer(l, &mut self.state, &xs, &layer_input)?;
            self.hot_experts.record(l, &routed, step);
            if self.predict_experts {
                record_last_routed(&mut self.last_routed[l], &routed, seq_len);
                if all_logits {
                    self.verify_routed[l].clear();
                    self.verify_routed[l].extend_from_slice(&routed);
                }
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

        if let (Some(m), Some(emb)) = (self.mtp.as_mut(), emb) {
            // The trunk's normed rows seed the MTP block; the head reads the
            // same rows, so the logits are those of `head`.
            let normed = w.head_norm(&xs)?;
            mtp_advance(&*w, m, &emb, &normed, offset, mask.as_ref(), all_logits)?;
            return if all_logits {
                w.head_logits(&normed)
            } else {
                w.head_logits(&normed.narrow(1, seq_len - 1, 1)?)?.squeeze(1)
            };
        }
        if all_logits {
            w.head(&xs)
        } else {
            w.head(&xs.narrow(1, seq_len - 1, 1)?)?.squeeze(1)
        }
    }

    /// Turn MTP drafting on or off for this session and every session derived
    /// from it.  On, every forward pass also runs its rows through the
    /// weights' NextN block (one more layer per row) so
    /// [`Self::mtp_draft`] can draft from the next position.  Takes effect
    /// from an empty session; returns whether the weights have the block.
    pub fn set_mtp(&mut self, on: bool) -> bool {
        let has = self.weights.has_mtp();
        self.mtp = (on && has).then(MtpState::new);
        has
    }

    /// Whether MTP drafting is on (see [`Self::set_mtp`]).
    pub fn mtp_enabled(&self) -> bool {
        self.mtp.is_some()
    }

    /// Draft up to `n` tokens after `next`, the token about to be fed at
    /// `pos`, with the MTP block: the row `(next, h_{pos-1})` predicts the
    /// first draft, and each further row chains the block's own output as
    /// its seed.  Greedy picks.  Leaves the session as it was (the draft
    /// rows' KV is cut back off); empty when MTP is off or the block is not
    /// at `pos`.
    pub fn mtp_draft(&mut self, next: u32, pos: usize, n: usize) -> Result<Vec<u32>> {
        let w = Arc::clone(&self.weights);
        let Some(m) = self.mtp.as_mut() else {
            return Ok(Vec::new());
        };
        if n == 0 || m.len != pos {
            return Ok(Vec::new());
        }
        let emb_table = w.tok_embeddings();
        let hidden = emb_table.hidden()?;
        // Each pick stays on the device and feeds the next row's embedding
        // directly; the draft is read back once, at the end.
        let mut ids = Tensor::new(&[next], w.device())?;
        let mut picks = Vec::with_capacity(n);
        let mut seed = m.seed.clone();
        for d in 0..n {
            let emb = emb_table.forward(&ids)?.reshape((1, 1, hidden))?;
            let s = match seed.take() {
                Some(s) => s,
                None => emb.zeros_like()?,
            };
            let input = LayerInput { mask: None, offset: pos + d, tokens: &[], verify: false };
            let out = w.mtp_forward(&mut m.state, &emb, &s, &input)?;
            ids = w.head_logits(&out)?.flatten_all()?.argmax_keepdim(0)?;
            picks.push(ids.clone());
            seed = Some(out);
        }
        W::truncate_state(&mut m.state, pos)?;
        Tensor::cat(&picks, 0)?.to_vec1::<u32>()
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

    /// Take a prefix checkpoint at `pos`, the number of tokens this session
    /// has been fed, unless one is already there.  The engine calls it before
    /// prefilling a follow-up turn, so the boundary before the new message —
    /// where an agent that replaces its last message diverges — can be
    /// resumed from (Strata's message-boundary checkpoint).  A no-op for a
    /// model whose layers all truncate on their own.
    pub fn checkpoint_prefix(&mut self, pos: usize) {
        if pos > 0 && self.verify.is_none() && !self.prefix.points.iter().any(|p| p.pos == pos) {
            self.take_prefix_checkpoint(pos);
        }
    }

    /// A detached copy of this session's state as it stood after `keep` of
    /// the `len` tokens it has been fed, on `device` (the host, to park it
    /// off an accelerator).
    ///
    /// `None` when the state cannot be had at `keep`: a model with recurrent
    /// layers needs `keep == len` or a prefix checkpoint at exactly `keep`,
    /// and a pending verification pass leaves the state mid-rollback.
    pub fn snapshot(
        &self,
        keep: usize,
        len: usize,
        device: &Device,
    ) -> Result<Option<SessionSnapshot<W>>> {
        if keep == 0 || keep > len || self.verify.is_some() {
            return Ok(None);
        }
        let checkpoint = if keep == len || self.weights.can_truncate() {
            None
        } else {
            match self.prefix.points.iter().find(|p| p.pos == keep) {
                Some(cp) => Some(cp),
                None => return Ok(None),
            }
        };
        let mut state = Vec::with_capacity(self.state.len());
        for (i, s) in self.state.iter().enumerate() {
            let running = checkpoint.and_then(|c| c.layers.get(i)).and_then(Option::as_ref);
            state.push(match running {
                Some(cp) => {
                    // Cut a throwaway clone back (it only moves its own
                    // length) and give it the checkpoint's running state.
                    let mut at = s.clone();
                    W::truncate_state(&mut at, keep)?;
                    W::restore_checkpoint(&mut at, cp)?;
                    W::fork_state(&at, keep, device)?
                }
                None => W::fork_state(s, keep, device)?,
            });
        }
        let tokens = if self.weights.wants_tokens() {
            self.tokens[..keep.min(self.tokens.len())].to_vec()
        } else {
            Vec::new()
        };
        let mtp = match self.mtp.as_ref().filter(|m| m.len >= keep) {
            Some(m) => Some(fork_mtp::<W>(&m.state, m.seed.as_ref(), m.len == keep, keep, device)?),
            None => None,
        };
        Ok(Some(SessionSnapshot {
            weights: Arc::clone(&self.weights),
            pos: keep,
            bytes: state.iter().map(W::state_bytes).sum::<u64>() + mtp_bytes::<W>(&mtp),
            state,
            tokens,
            mtp,
            device: device.clone(),
        }))
    }

    /// Bytes this session's state holds: what a [`Self::snapshot`] of it
    /// copies at most.
    pub fn state_bytes(&self) -> u64 {
        let mtp = self.mtp.as_ref().map_or(0, |m| W::state_bytes(&m.state));
        self.state.iter().map(W::state_bytes).sum::<u64>() + mtp
    }

    /// Continue from `snap`: this session's state becomes the snapshot's
    /// (moved back onto the weights' device when it was parked elsewhere).
    /// Refused (`Ok(false)`, state untouched) for a snapshot taken over other
    /// weights.
    pub fn restore(&mut self, snap: SessionSnapshot<W>) -> Result<bool> {
        if !Arc::ptr_eq(&self.weights, &snap.weights) || snap.state.len() != self.state.len() {
            return Ok(false);
        }
        let device = self.weights.device().clone();
        let state = if snap.device.same_device(&device) {
            snap.state
        } else {
            snap.state
                .iter()
                .map(|s| W::fork_state(s, snap.pos, &device))
                .collect::<Result<Vec<_>>>()?
        };
        self.state = state;
        self.tokens = snap.tokens;
        if let Some(m) = self.mtp.as_mut() {
            // Without the block's state the conversation runs undrafted
            // (the block stays behind) until the session is cleared.
            *m = MtpState::new();
            if let Some((st, seed)) = snap.mtp {
                let (state, seed) = if snap.device.same_device(&device) {
                    (st, seed)
                } else {
                    fork_mtp::<W>(&st, seed.as_ref(), true, snap.pos, &device)?
                };
                *m = MtpState { state, len: snap.pos, seed, verify_rows: None };
            }
        }
        self.verify = None;
        self.prefix.points.clear();
        self.forget_last_routed();
        // A recurrent model can come back to the restored position only
        // through a checkpoint there.
        self.take_prefix_checkpoint(snap.pos);
        Ok(true)
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
        self.forget_last_routed();
        if let Some(m) = self.mtp.as_mut() {
            *m = MtpState::new();
        }
    }

    /// Drop the next-step expert prediction: the token it was taken from is
    /// gone, and the one now last was routed in a pass not kept.
    fn forget_last_routed(&mut self) {
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
        if let Some(m) = self.mtp.as_mut() {
            m.verify_rows = None;
        }
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
                // Predict from the last token kept, not the rejected tail.
                if self.predict_experts && retained < checkpoint.seq {
                    for (slot, routed) in self.last_routed.iter_mut().zip(&self.verify_routed) {
                        slot.clear();
                        if let Some(idx) = retained.checked_sub(1) {
                            slot.extend_from_slice(routed_at(routed, checkpoint.seq, idx));
                        }
                    }
                }
            }
            None if self.weights.can_truncate() => {
                for s in self.state.iter_mut() {
                    W::truncate_state(s, keep)?;
                }
                self.forget_last_routed();
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
                self.forget_last_routed();
            }
        }
        self.prefix.invalidate_after(keep);
        self.tokens.truncate(keep);
        if let Some(m) = self.mtp.as_mut() {
            let rows = m.verify_rows.take();
            if m.len > keep {
                W::truncate_state(&mut m.state, keep)?;
                m.len = keep;
                // Inside the verification pass just rolled back, its rows
                // hold the trunk's hidden at the position kept.
                m.seed = match rows {
                    Some((off, rows)) if keep > off && keep <= off + rows.dim(1)? => {
                        Some(rows.narrow(1, keep - off - 1, 1)?.copy()?)
                    }
                    _ => None,
                };
            }
        }
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

    fn finish_chunks(
        &mut self,
        chunks: &[crate::stream_prefill::Chunk],
        acts: &[Tensor],
    ) -> Result<()> {
        if self.mtp.is_none() {
            return Ok(());
        }
        let w = Arc::clone(&self.weights);
        for (chunk, act) in chunks.iter().zip(acts) {
            let n = chunk.tokens.len();
            let emb = self.embed(&Tensor::new(chunk.tokens, w.device())?.unsqueeze(0)?, n)?;
            let normed = w.head_norm(act)?;
            let mask = if n == 1 { None } else { Some(self.mask.mask(n, chunk.pos)?) };
            let m = self.mtp.as_mut().expect("checked above");
            mtp_advance(&*w, m, &emb, &normed, chunk.pos, mask.as_ref(), false)?;
        }
        Ok(())
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
