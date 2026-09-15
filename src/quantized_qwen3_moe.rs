//! Pure-Rust quantized loader for the `qwen3moe` GGUF architecture.
//!
//! Covers Qwen3-30B-A3B and Qwen3-Coder-30B-A3B — every model llama.cpp labels
//! `general.architecture = "qwen3moe"`.  candle ships a `qwen3_moe` model but
//! no *quantized* GGUF path that runs anywhere except CUDA (`moe_gemm_gguf` is
//! CUDA-only), so on CPU / Metal this module is the only way to run these
//! models in Joshua.
//!
//! The architecture is a standard decoder stack (pre-norm, GQA) with two
//! Qwen3-family twists plus a fine-grained MoE:
//!
//! * **Per-head Q/K RMSNorm.**  `attn_q_norm`/`attn_k_norm` are `[head_dim]`
//!   weight vectors applied to each query/key head independently — not to the
//!   whole hidden state.  Unlike Qwen3 dense (which may fuse them), the GGUF
//!   always ships them as separate tensors.
//!
//! * **Half-split RoPE over the full head dim.**  llama.cpp uses
//!   `LLAMA_ROPE_TYPE_HALF` for Qwen3/Qwen3MoE: frequencies pair
//!   `(i, i + head_dim/2)`, exactly candle's `rotary_emb::rope`.  The
//!   `head_dim` comes from `attention.key_length` and is *not* derived from
//!   `embedding_length / head_count` (Qwen3-Coder: 128 vs 64).
//!
//! * **Fine-grained MoE.**  Every layer routes each token to
//!   `expert_used_count` of `expert_count` experts (8 of 128 for 30B-A3B) via a
//!   softmax router with `norm_topk_prob` weight normalisation.  There are no
//!   shared experts and no gating bias.  Experts stay **quantized**: the 3-D
//!   expert tensor is sliced into per-expert [`QMatMul`]s straight from its
//!   quantized bytes (borrowed in place from the mmap when possible), so the
//!   model keeps its on-disk footprint instead of exploding to f32 in RAM.
//!
//! Activations run in f32 for CPU accuracy, mirroring the other Joshua
//! quantized loaders (`glm4`, `deepseek2`).

use std::borrow::Cow;
use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, GgmlDType, QMatMul, QStorage, QTensor};
use crate::paged_weights::{PagedWeight, WeightCache};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::ops::{silu, softmax_last_dim};
use candle_transformers::quantized_nn::RmsNorm;

use crate::token_embedding::TokenEmbedding;
use crate::zero_copy_metal::{ZcContext, ZcWeight};

// ─── Opt-in phase profiler (JOSHUA_PROFILE=1) ────────────────────────────────
//
// Diagnostic for the CPU decode path: accumulates nanoseconds per phase
// across forward passes and prints per-step averages every 16 steps on the
// calling thread.  Off by default; zero cost when disabled.
mod prof {
    use std::cell::Cell;
    use std::sync::OnceLock;
    use std::time::Instant;

    type Slot = &'static std::thread::LocalKey<Cell<u128>>;

    pub fn enabled() -> bool {
        static E: OnceLock<bool> = OnceLock::new();
        *E.get_or_init(|| {
            std::env::var("JOSHUA_PROFILE").map(|v| !v.is_empty()).unwrap_or(false)
        })
    }

    thread_local! {
        pub static ATT: Cell<u128> = Cell::new(0);
        pub static MOE: Cell<u128> = Cell::new(0);
        pub static EXPERTS: Cell<u128> = Cell::new(0);
        pub static HEAD: Cell<u128> = Cell::new(0);
        static STEPS: Cell<u64> = Cell::new(0);
    }

    pub struct Phase(Option<(Instant, Slot)>);

    impl Phase {
        pub fn start(slot: Slot) -> Self {
            Self(if enabled() { Some((Instant::now(), slot)) } else { None })
        }
    }

    impl Drop for Phase {
        fn drop(&mut self) {
            if let Some((t0, slot)) = self.0.take() {
                let ns = t0.elapsed().as_nanos();
                slot.with(|c| c.set(c.get() + ns));
            }
        }
    }

    /// Report averages over the steps accumulated so far and reset.
    pub fn report() {
        if !enabled() {
            return;
        }
        STEPS.with(|s| {
            let n = s.get() + 1;
            s.set(n);
            if n % 16 == 0 {
                let a = ATT.with(|c| c.get());
                let m = MOE.with(|c| c.get());
                let e = EXPERTS.with(|c| c.get());
                let h = HEAD.with(|c| c.get());
                eprintln!(
                    "[profile] avg ms/step over last 16: attention {:.1}, moe {:.1} \
                     (expert matmuls {:.1}), lm_head {:.1}",
                    a as f64 / 16e6,
                    m as f64 / 16e6,
                    e as f64 / 16e6,
                    h as f64 / 16e6,
                );
                ATT.with(|c| c.set(0));
                MOE.with(|c| c.set(0));
                EXPERTS.with(|c| c.set(0));
                HEAD.with(|c| c.set(0));
            }
        });
    }
}

// ─── Configuration ───────────────────────────────────────────────────────────

struct Config {
    n_layer: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rms_eps: f64,
    // RoPE.
    rope_theta: f32,
    context_length: usize,
    // MoE.
    n_expert: usize,
    n_expert_used: usize,
    expert_weights_norm: bool,
}

// ─── Metadata helpers ───────────────────────────────────────────────────────

struct Meta<'a>(&'a std::collections::HashMap<String, gguf_file::Value>);

impl Meta<'_> {
    fn u32(&self, key: &str) -> Result<u32> {
        match self.0.get(key) {
            Some(v) => v.to_u32(),
            None => candle_core::bail!("qwen3moe: missing GGUF metadata key `{key}`"),
        }
    }
    fn u32_or(&self, key: &str, default: u32) -> u32 {
        self.0.get(key).and_then(|v| v.to_u32().ok()).unwrap_or(default)
    }
    fn f32(&self, key: &str) -> Result<f32> {
        match self.0.get(key) {
            Some(v) => v.to_f32(),
            None => candle_core::bail!("qwen3moe: missing GGUF metadata key `{key}`"),
        }
    }
    fn f32_or(&self, key: &str, default: f32) -> f32 {
        self.0.get(key).and_then(|v| v.to_f32().ok()).unwrap_or(default)
    }
    fn bool_or(&self, key: &str, default: bool) -> bool {
        self.0.get(key).and_then(|v| v.to_bool().ok()).unwrap_or(default)
    }
}

impl Config {
    fn from_metadata(md: &std::collections::HashMap<String, gguf_file::Value>) -> Result<Self> {
        let m = Meta(md);
        let a = "qwen3moe";
        let n_head = m.u32(&format!("{a}.attention.head_count"))? as usize;
        let n_kv_head = m.u32(&format!("{a}.attention.head_count_kv"))? as usize;
        let n_layer = m.u32(&format!("{a}.block_count"))? as usize;
        let n_embd = m.u32(&format!("{a}.embedding_length"))? as usize;
        let context_length = m.u32(&format!("{a}.context_length"))? as usize;
        let rms_eps = m.f32(&format!("{a}.attention.layer_norm_rms_epsilon"))? as f64;

        // Qwen3-Coder decouples head_dim from the embedding width (2048/32 = 64
        // would be wrong; the real value is 128), so `attention.key_length`
        // must win when present.  Fall back to n_embd/n_head only for GGUFs
        // that omit it.
        let head_dim = m.u32_or(&format!("{a}.attention.key_length"), (n_embd / n_head) as u32)
            as usize;
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            candle_core::bail!("qwen3moe: invalid head_dim {head_dim} (must be a positive even number)");
        }
        if n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
            candle_core::bail!(
                "qwen3moe: head_count {n_head} must be a multiple of head_count_kv {n_kv_head}"
            );
        }

        let rope_theta = m.f32_or(&format!("{a}.rope.freq_base"), 10_000.0);

        let n_expert = m.u32(&format!("{a}.expert_count"))? as usize;
        let n_expert_used = m.u32(&format!("{a}.expert_used_count"))? as usize;
        let expert_ff = m.u32(&format!("{a}.expert_feed_forward_length"))? as usize;
        if n_expert_used == 0 || n_expert_used > n_expert {
            candle_core::bail!(
                "qwen3moe: expert_used_count {n_expert_used} must be in 1..=expert_count {n_expert}"
            );
        }
        if expert_ff == 0 {
            candle_core::bail!("qwen3moe: expert_feed_forward_length must be positive");
        }
        // Qwen3MoE always normalises the top-k routing weights; the metadata
        // key is present in well-formed conversions, but default to the
        // architecture's true behaviour when a conversion omits it.
        let expert_weights_norm = m.bool_or(&format!("{a}.attention.norm_topk_prob"), true);

        Ok(Self {
            n_layer,
            n_head,
            n_kv_head,
            head_dim,
            rms_eps,
            rope_theta,
            context_length,
            n_expert,
            n_expert_used,
            expert_weights_norm,
        })
    }
}

// ─── Quantized GGUF reader (with optional mmap borrowing) ───────────────────

/// Small GGUF reader over the (possibly memory-mapped) file.
struct Reader<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
    /// Device the routed-expert tensors are built on.  Same as `device` for
    /// a CPU model or when the experts are uploaded; the CPU when the model
    /// runs on an accelerator that cannot hold the expert pool (see
    /// [`crate::placement::ExpertPlacement`]) — each expert is then borrowed
    /// from the mapping and [`Moe::dispatch`] hops the activations across.
    expert_device: Device,
    /// When present, tensors are borrowed in place from this mapping instead
    /// of being copied onto the heap (see [`crate::mmap_tensor`]).
    mmap: Option<Arc<memmap2::Mmap>>,
    /// When present, quantized weights are bound straight into the mapping
    /// via a no-copy Metal buffer (see [`crate::zero_copy_metal`]) instead of
    /// being uploaded.  `None` on CPU, without a mapping, or when the
    /// no-copy buffer could not be created (the loader then copies).
    zc: Option<Arc<ZcContext>>,
    /// Opt-in bounded GPU weight cache (see `JOSHUA_GPU_WEIGHT_CACHE`).  When
    /// present, quantized experts are kept compressed in the mmap and uploaded
    /// to the device on demand instead of being copied up front.
    gpu_cache: Option<Arc<WeightCache>>,
}

impl<R: Read + Seek> Reader<R> {
    fn qtensor(&mut self, name: &str) -> Result<QTensor> {
        if let Some(mmap) = &self.mmap {
            return crate::mmap_tensor::qtensor_from_mmap(
                &self.ct,
                mmap,
                &mut self.reader,
                name,
                &self.device,
            );
        }
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
    fn qmatmul(&mut self, name: &str) -> Result<Weight> {
        if let Some(zc) = &self.zc {
            if let Some(w) = zc.weight(&self.ct, name)? {
                return Ok(Weight::Zc(Arc::new(w)));
            }
        }
        Ok(Weight::Candle(QMatMul::from_qtensor(self.qtensor(name)?)?))
    }
    fn qmatmul_opt(&mut self, name: &str) -> Option<Weight> {
        if self.has(name) {
            self.qmatmul(name).ok()
        } else {
            None
        }
    }
    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        RmsNorm::from_qtensor(self.qtensor(name)?, eps)
    }
    fn f32_tensor(&mut self, name: &str) -> Result<Tensor> {
        self.qtensor(name)?.dequantize(&self.device)?.to_dtype(DType::F32)
    }
    fn has(&self, name: &str) -> bool {
        self.ct.tensor_infos.contains_key(name)
    }
}

// ─── Weight carrier (candle upload vs zero-copy mmap binding) ───────────────

/// A quantized linear weight in one of two homes:
///
/// * [`Weight::Candle`] — candle's own `QMatMul` (copied onto the device).
///   Used on CPU, without a mapping, and for the tensors the zero-copy path
///   does not serve.
/// * [`Weight::Zc`] — a [`ZcWeight`] bound at its file offset inside a
///   no-copy Metal buffer.  The GPU reads the mapped pages directly; nothing
///   is copied or uploaded.
#[derive(Clone)]
enum Weight {
    Candle(QMatMul),
    Zc(Arc<ZcWeight>),
    /// Quantized weights kept compressed in mmap and uploaded to the device on
    /// demand through a bounded [`WeightCache`] (opt-in; see
    /// `JOSHUA_GPU_WEIGHT_CACHE`).  Each expert owns one [`PagedWeight`].
    Paged(Arc<PagedWeight>),
}

impl Weight {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Weight::Candle(q) => {
                // Joshua's SIMD path first: fused dequant+dot kernels
                // (Q8_0/Q2_K/Q4_K) or dequant-row + NEON/AVX2 dot (other
                // k-quants), parallelised across rows — instead of candle's
                // scalar single-threaded kernel.  Anything without a fast
                // path here falls through to candle unchanged.
                if let QMatMul::QTensor(qt) = q {
                    if let Some(res) = crate::quant_matmul::try_fast_cpu_qmatmul(qt, xs) {
                        return res;
                    }
                }
                q.forward(xs)
            }
            Weight::Zc(z) => z.forward(xs),
            Weight::Paged(p) => p.forward(xs),
        }
    }
}

// ─── RoPE (half-split, matching llama.cpp LLAMA_ROPE_TYPE_HALF) ─────────────

struct RotaryEmbedding {
    sin: Tensor, // [max_seq, head_dim/2]
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq: usize, theta: f32, dev: &Device) -> Result<Self> {
        let max_seq = max_seq.max(1);
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / dim as f32))
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), dev)?;
        let t = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        Ok(Self { sin, cos })
    }

    /// Apply half-split RoPE to `q`/`k`, each shaped
    /// `[b, heads, seq_len, head_dim]` (cos/sin narrowed to `[seq_len, d/2]`).
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

/// Repeat KV heads `n_rep` times (GQA). No-op when `n_rep == 1`.
fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let b_sz = x.dim(0)?;
    let n_kv_head = x.dim(1)?;
    let seq_len = x.dim(2)?;
    let head_dim = x.dim(3)?;
    x.reshape((b_sz, n_kv_head, 1, seq_len, head_dim))?
        .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
        .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
}

// ─── Attention (GQA + per-head Q/K RMSNorm) ─────────────────────────────────

struct Attention {
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: RmsNorm, // [head_dim] per-head query norm
    k_norm: RmsNorm, // [head_dim] per-head key norm
    rotary: Arc<RotaryEmbedding>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    softmax_scale: f64,
}

/// One layer's KV cache: `(k, v)`, each `[b, n_kv_head, seq, head_dim]`,
/// appended along dim 2.  Owned by the session, not the (shared) weights.
type KvCache = Option<(Tensor, Tensor)>;

impl Attention {
    fn new<R: Read + Seek>(
        rd: &mut Reader<R>,
        p: &str,
        cfg: &Config,
        rotary: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        Ok(Self {
            q: rd.qmatmul(&format!("{p}.attn_q.weight"))?,
            k: rd.qmatmul(&format!("{p}.attn_k.weight"))?,
            v: rd.qmatmul(&format!("{p}.attn_v.weight"))?,
            o: rd.qmatmul(&format!("{p}.attn_output.weight"))?,
            q_norm: rd.rms_norm(&format!("{p}.attn_q_norm.weight"), cfg.rms_eps)?,
            k_norm: rd.rms_norm(&format!("{p}.attn_k_norm.weight"), cfg.rms_eps)?,
            rotary,
            n_head: cfg.n_head,
            n_kv_head: cfg.n_kv_head,
            head_dim: cfg.head_dim,
            softmax_scale: 1.0 / (cfg.head_dim as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        kv_cache: &mut KvCache,
        xs: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let _p = prof::Phase::start(&prof::ATT);
        let (b, seq_len, _) = xs.dims3()?;

        let q = self
            .q
            .forward(xs)?
            .reshape((b, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(xs)?
            .reshape((b, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(xs)?
            .reshape((b, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Per-head Q/K RMSNorm: flatten heads into the batch dim, normalise
        // each `[seq_len, head_dim]` slice with the shared `[head_dim]`
        // weight, then restore.
        let q = self
            .q_norm
            .forward(&q.flatten(0, 2)?)?
            .reshape((b, self.n_head, seq_len, self.head_dim))?;
        let k = self
            .k_norm
            .forward(&k.flatten(0, 2)?)?
            .reshape((b, self.n_kv_head, seq_len, self.head_dim))?;

        let (q, k) = self.rotary.apply(&q, &k, offset)?;

        // KV cache (append along the sequence dim).
        let (k, v) = match &*kv_cache {
            None => (k, v),
            Some((pk, pv)) => (
                Tensor::cat(&[pk, &k], 2)?.contiguous()?,
                Tensor::cat(&[pv, &v], 2)?.contiguous()?,
            ),
        };
        *kv_cache = Some((k.clone(), v.clone()));

        // Scaled dot-product attention (GQA: repeat K/V heads).
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?.contiguous()?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?.contiguous()?;
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.softmax_scale)?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // [b, n_head, seq, head_dim]
        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq_len, self.n_head * self.head_dim))?;
        self.o.forward(&ctx)
    }
}

/// Keep only the first `keep` sequence positions of a layer's KV cache.
///
/// Edited-context prefix reuse: positions `[0..keep)` were produced by
/// exactly the tokens both conversations share, so they remain valid;
/// everything from `keep` on is dropped and recomputed by the next
/// prefill (which continues at absolute position `keep`).  The prefix is
/// materialised (`contiguous`) so the full-length buffer is actually
/// freed instead of lingering behind a view until the next append.
fn truncate_kv(kv_cache: &mut KvCache, keep: usize) -> Result<()> {
    match kv_cache {
        None => Ok(()),
        Some((_k, _v)) if keep == 0 => {
            *kv_cache = None;
            Ok(())
        }
        Some((k, v)) => {
            // Sequence length lives on dim 2 ([b, n_kv_head, seq, dim]).
            let len = k.dim(2)?;
            if keep >= len {
                return Ok(());
            }
            let k = k.narrow(2, 0, keep)?.contiguous()?;
            let v = v.narrow(2, 0, keep)?.contiguous()?;
            *kv_cache = Some((k, v));
            Ok(())
        }
    }
}

// ─── Mixture of experts (Qwen3MoE routing) ──────────────────────────────────

#[derive(Clone)]
struct Mlp {
    gate: Weight,
    up: Weight,
    down: Weight,
    /// Per-tensor byte-range handles for best-effort page prefetch, present
    /// only when the weights are borrowed from the model mapping (absent for
    /// zero-copy Metal buffers and streamed loads).
    prefetch: Option<crate::residency::ExpertHandles>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.gate.forward(xs)?;
        let w3 = self.up.forward(xs)?;
        self.down.forward(&(silu(&w1)? * w3)?)
    }

    /// Ask the kernel to prefetch this expert's weight pages (best effort;
    /// no-op when the weights are not mmap-backed).
    fn prefetch(&self) {
        if let Some(p) = &self.prefetch {
            p.gate.prefetch();
            p.up.prefetch();
            p.down.prefetch();
        }
    }
}

/// The device-resident form of one routed expert: the same gate/up/down
/// projections, uploaded to the expert device, wrapped in an opaque
/// [`crate::residency::DeviceExpertSlot`] payload so the
/// [`crate::residency::DeviceResidency`] pool can hold it and dispatch can run
/// it on the GPU.  Forwarding mirrors [`Mlp::forward`] (silu(gate·x)⊗up then
/// down), operating on the device tensors.
struct DeviceExpert {
    gate: Weight,
    up: Weight,
    down: Weight,
    bytes: u64,
}

impl crate::residency::DeviceExpertSlot for DeviceExpert {
    fn device_bytes(&self) -> u64 {
        self.bytes
    }
}

impl DeviceExpert {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.gate.forward(xs)?;
        let w3 = self.up.forward(xs)?;
        self.down.forward(&(silu(&w1)? * w3)?)
    }
}

/// Monotonic per-layer expert index mix: layer + a per-layer monotonic clock
/// so the residency LRU's recency is comparable across layers.
type ExpertKey = (u32, u32);

struct Moe {
    gate_t: Tensor, // router weight transposed to [n_embd, n_expert], contiguous (cached)
    experts: Vec<Mlp>,
    n_expert_used: usize,
    weights_norm: bool,
    /// This MoE block's layer index (for the residency key).
    layer: u32,
    /// Device the routed-expert weights live on.  The model device when the
    /// experts were uploaded; the CPU when the pool stays in host RAM on an
    /// accelerator (see [`crate::placement::ExpertPlacement`]).  `dispatch`
    /// moves the block's input across once and its result back once.
    expert_device: Device,
    /// Bounded VRAM expert cache (#62): `Some` when a device-resident subset of
    /// the routed experts is enabled.  `dispatch`/`dispatch_decode` call
    /// [`crate::residency::DeviceResidency::lookup`] per expert and run the
    /// resident device form on `expert_device`, falling back to the host
    /// `experts[e]` on a miss.  `None` (the default) keeps today's all-host or
    /// all-device path byte-for-byte.
    residency: Option<std::sync::Arc<crate::residency::DeviceResidency<DeviceExpert>>>,
}

impl Moe {
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Vec<u32>)> {
        let _p = prof::Phase::start(&prof::MOE);
        let (b, seq_len, h) = xs.dims3()?;
        let n_tokens = b * seq_len;
        let x2 = xs.reshape((n_tokens, h))?;
        let (topk_idx, weights) = self.route(&x2)?;
        // dispatch already drains the routed ids to the host for its own
        // bucketing; reuse them for the hot-expert cache instead of a second
        // device-to-host sync (which would cost a synchronization per layer
        // even when the cache is disabled).
        let (routed, ids) = self.dispatch(&x2, &topk_idx, &weights, n_tokens)?;
        Ok((routed.reshape((b, seq_len, h))?, ids))
    }

    /// Router logits → softmax probs → top-k ids and gathered weights, with
    /// llama.cpp's `norm_topk_prob` normalisation (min-clamp to the f16
    /// epsilon so a degenerate routing can never divide by zero).
    fn route(&self, x2: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = x2.matmul(&self.gate_t)?; // [n_tokens, n_expert]
        let probs = softmax_last_dim(&logits)?;
        let topk_idx = topk_indices(&probs, self.n_expert_used)?; // [n_tokens, k]
        let mut weights = probs.gather(&topk_idx, D::Minus1)?; // [n_tokens, k]
        if self.weights_norm {
            let denom = weights
                .sum_keepdim(D::Minus1)?
                .clamp(6.103_515_6e-5, f32::INFINITY)?;
            weights = weights.broadcast_div(&denom)?;
        }
        Ok((topk_idx, weights))
    }

    /// Run each selected expert over its routed tokens and accumulate the
    /// weighted outputs. Experts stay quantized.
    ///
    /// Prefill (`n_tokens > 1`) buckets tokens per expert and runs one batched
    /// matmul per expert.  Decode (`n_tokens == 1`) takes the separate path
    /// below that avoids every per-expert host round-trip.
    fn dispatch(
        &self,
        x2: &Tensor,
        topk_idx: &Tensor,
        weights: &Tensor,
        n_tokens: usize,
    ) -> Result<(Tensor, Vec<u32>)> {
        if n_tokens == 1 {
            return self.dispatch_decode(x2, topk_idx, weights);
        }
        let k = self.n_expert_used;
        let h = x2.dim(1)?;
        let ids: Vec<u32> = topk_idx.flatten_all()?.to_vec1()?;
        let wts: Vec<f32> = weights.flatten_all()?.to_vec1()?;

        // The routed experts may live on a different device from the
        // activations (host-resident expert pool on an accelerator).  Move
        // the whole block's input across once and its result back once, so
        // the boundary costs two transfers per layer instead of one per
        // expert matmul.
        let out_device = x2.device().clone();
        let same_device = self.expert_device.same_device(&out_device);
        let x2_hopped;
        let x2 = if same_device {
            x2
        } else {
            x2_hopped = x2.to_device(&self.expert_device)?;
            &x2_hopped
        };

        // Bucket (token, weight) pairs by expert.
        let mut per_expert: Vec<Vec<(u32, f32)>> = vec![Vec::new(); self.experts.len()];
        for t in 0..n_tokens {
            for s in 0..k {
                let e = ids[t * k + s] as usize;
                if e >= self.experts.len() {
                    // Defensive: a corrupt router output must fail loudly with
                    // context instead of panicking on the slice index below.
                    let start = t * k;
                    let row: Vec<u32> = ids[start..start + k].to_vec();
                    eprintln!(
                        "ROUTER OOB: token {t} slot {s} expert id {e} ({} experts); ids row {row:?}",
                        self.experts.len()
                    );
                    candle_core::bail!(
                        "qwen3moe: router selected expert {e} out of {} (token {t}, slot {s})",
                        self.experts.len()
                    );
                }
                per_expert[e].push((t as u32, wts[t * k + s]));
            }
        }

        let dev = &self.expert_device;
        let mut y = Tensor::zeros((n_tokens, h), DType::F32, dev)?;
        let _p = prof::Phase::start(&prof::EXPERTS);
        for (e, bucket) in per_expert.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let token_idx: Vec<u32> = bucket.iter().map(|(t, _)| *t).collect();
            let w: Vec<f32> = bucket.iter().map(|(_, w)| *w).collect();
            let count = token_idx.len();
            let idx = Tensor::from_vec(token_idx, count, dev)?;
            let x_sel = x2.index_select(&idx, 0)?; // [count, h]
            let out = self.expert_forward(e, &x_sel)?; // [count, h]
            let w = Tensor::from_vec(w, (count, 1), dev)?;
            y = y.index_add(&idx, &out.broadcast_mul(&w)?, 0)?;
        }
        drop(_p);
        let y = if same_device { y } else { y.to_device(&out_device)? };
        Ok((y, ids))
    }

    /// Single-token decode dispatch.  With one token every selected expert
    /// consumes the same input row, so there is nothing to gather: run the
    /// `k` expert MLPs over `x2`, stack their outputs, scale by the routing
    /// weights (still on the device — no copy back) and sum.
    fn dispatch_decode(
        &self,
        x2: &Tensor,
        topk_idx: &Tensor,
        weights: &Tensor,
    ) -> Result<(Tensor, Vec<u32>)> {
        let k = self.n_expert_used;
        let ids: Vec<u32> = topk_idx.flatten_all()?.to_vec1()?; // [k] — the one host sync
        // Host-resident experts on an accelerator: one hop in, one hop out
        // (see `dispatch`).
        let out_device = x2.device().clone();
        let same_device = self.expert_device.same_device(&out_device);
        let x2_hopped;
        let x2 = if same_device {
            x2
        } else {
            x2_hopped = x2.to_device(&self.expert_device)?;
            &x2_hopped
        };
        // NB: deliberately serial.  Candle's own CPU kernels already fan out
        // through its global rayon pool per op, and an outer parallel loop
        // over experts measurably *regresses* decode (~20%) through nested-
        // pool contention; the memory streams of k=8 experts overlap fine
        // within that pool.
        let mut outs = Vec::with_capacity(k);
        let _p = prof::Phase::start(&prof::EXPERTS);
        for &e in ids.iter() {
            outs.push(self.expert_forward(e as usize, x2)?); // [1, h] each
        }
        drop(_p);
        let out = Tensor::stack(&outs, 0)?; // [k, 1, h]
        let out = if same_device { out } else { out.to_device(&out_device)? };
        let w = weights.reshape((k, 1, 1))?; // [k, 1, 1]
        let y = out.broadcast_mul(&w)?.sum(0)?; // [1, h]
        Ok((y, ids))
    }

    /// Run one routed expert over `x`, preferring the device-resident form
    /// (a `DeviceResidency` hit) and falling back to the host `Mlp` on a miss
    /// (#62 dispatch partition).  With no residency configured this is exactly
    /// `self.experts[e].forward(x)` — byte-for-byte the pre-cache path.
    fn expert_forward(&self, e: usize, x: &Tensor) -> Result<Tensor> {
        if let Some(res) = &self.residency {
            if let Some(dev) = res.lookup(self.layer, e as u32) {
                return dev.forward(x);
            }
        }
        self.experts[e].forward(x)
    }
}

/// Indices of the top-`k` values along the last dim (descending), as u32.
fn topk_indices(t: &Tensor, k: usize) -> Result<Tensor> {
    t.arg_sort_last_dim(false)?
        .narrow(D::Minus1, 0, k)?
        .contiguous()
}

// ─── Layer + model ───────────────────────────────────────────────────────────

struct Layer {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    ffn: Moe,
}

/// The immutable half of a loaded model: every weight, the expert residency
/// backend and the device.  Sessions share one `Arc` of it (see
/// [`GGUFQWenMoE::new_session`]) and own only their KV caches, so a second
/// concurrent conversation costs its KV cache rather than a second copy of
/// the weights — on an accelerator, a second upload of the whole model.
struct Shared {
    tok_embeddings: TokenEmbedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: Weight,
    device: Device,
    /// Executes residency for the hot set (CPU madvise today; a device slot
    /// cache later).  Built once at load from the per-expert handles.
    residency: std::sync::Arc<dyn crate::residency::ExpertResidency>,
    n_expert: usize,
}

/// A quantized Qwen3-MoE model loaded from GGUF.
pub struct GGUFQWenMoE {
    shared: Arc<Shared>,
    /// Per-layer KV cache — the only per-session tensor state.
    kv: Vec<KvCache>,
    /// Routing-frequency LRU hot-expert cache (shared bookkeeping; budget set
    /// after load via [`GGUFQWenMoE::set_pin_hot_experts`], CLI flag
    /// `--pin-hot-experts`).  Records routing, re-selects the hot set every
    /// [`crate::hot_experts::REFRESH_STEPS`] decode steps, and reports newly
    /// hot experts for the residency backend.
    hot_experts: crate::hot_experts::HotExpertCache,
}

/// Slice a 3-D expert tensor `[n_expert, out, in]` into per-expert [`Weight`]s
/// plus an optional prefetch handle for the expert's bytes in the mapping.
///
/// Preferred paths point each expert at its own bytes **without reading or
/// copying anything**: on Metal via a no-copy buffer at the expert's file
/// offset, on CPU via a borrowed slice of the mapping.  Either way an expert
/// costs a pointer and a length until tokens actually route to it.  The
/// fallback (no mapping, or a tensor that cannot be borrowed) reads the whole
/// tensor and copies per-expert slices onto the device.
fn split_experts<R: Read + Seek>(
    rd: &mut Reader<R>,
    name: &str,
    n_expert: usize,
) -> Result<Vec<(Weight, Option<Arc<dyn crate::mmap_tensor::MmapPrefetch>>)>> {
    let (dims, dtype, tensor_offset) = {
        let Some(info) = rd.ct.tensor_infos.get(name) else {
            candle_core::bail!("qwen3moe: missing tensor `{name}`");
        };
        (info.shape.dims().to_vec(), info.ggml_dtype, info.offset)
    };
    if dims.len() != 3 || dims[0] != n_expert {
        candle_core::bail!(
            "qwen3moe: expected expert tensor `{name}` shaped [n_expert, out, in], got {dims:?}"
        );
    }
    let (out, inn) = (dims[1], dims[2]);
    let per_elems = out * inn;
    let block_size = dtype.block_size();
    let whole_blocks = block_size > 0 && per_elems.is_multiple_of(block_size);
    let base = rd.ct.tensor_data_offset.saturating_add(tensor_offset) as usize;
    let host_experts = rd.expert_device.is_cpu();

    // Host-resident experts with a mapping: borrow each expert's slice of the
    // mapping.  Building all of them reads nothing — an expert is a pointer
    // and a length — so a layer with 128 experts costs almost no memory
    // until tokens actually route to them.  Borrowed storage is always
    // CPU-resident, hence the `host_experts` gate; on an accelerator this is
    // the `ExpertPlacement::Host` layout (dense set on the device, experts in
    // host RAM).
    if host_experts && whole_blocks {
        if let Some(mmap) = rd.mmap.clone() {
            let per_bytes = per_elems / block_size * dtype.type_size();
            let mut borrowed = Vec::with_capacity(n_expert);
            for e in 0..n_expert {
                match crate::mmap_tensor::borrowed_range(
                    &mmap,
                    dtype,
                    base + e * per_bytes,
                    (out, inn).into(),
                )? {
                    Some(t) => borrowed.push((
                        Weight::Candle(QMatMul::from_qtensor(t)?),
                        crate::mmap_tensor::prefetch_handle(
                            &mmap,
                            dtype,
                            base + e * per_bytes,
                            per_elems / block_size,
                        ),
                    )),
                    // Any expert that cannot be borrowed (misalignment,
                    // truncated file) drops the whole layer to the copying
                    // path rather than mixing the two.
                    None => {
                        borrowed.clear();
                        break;
                    }
                }
            }
            if borrowed.len() == n_expert {
                return Ok(borrowed);
            }
        }
    }

    // Zero-copy Metal path: each expert is a byte range inside the shared
    // no-copy buffer.  No reads, no uploads.
    if !host_experts && whole_blocks {
        if let (Some(zc), Some(info)) = (&rd.zc, rd.ct.tensor_infos.get(name)) {
            let per_bytes = per_elems / block_size * dtype.type_size();
            let mut experts = Vec::with_capacity(n_expert);
            for e in 0..n_expert {
                experts.push((
                    Weight::Zc(Arc::new(ZcWeight::expert(
                        zc,
                        info,
                        rd.ct.tensor_data_offset,
                        [out, inn],
                        e * per_bytes,
                    )?)),
                    None, // Metal no-copy buffers have no pages to advise.
                ));
            }
            return Ok(experts);
        }
    }

    // GPU paging path: keep each expert's quantized block compressed in the
    // mmap and upload it to the device on demand through the bounded cache,
    // instead of copying every expert up front.  Only for block-capable
    // devices (Metal/CUDA); OpenCL cannot host quantized blocks yet
    // (OpenClStorage is f32-dense), so it keeps the eager-copy path below.
    if !host_experts && whole_blocks && !rd.expert_device.is_opencl() {
        if let Some(cache) = rd.gpu_cache.clone() {
            let per_bytes = per_elems / block_size * dtype.type_size();
            let mut paged = Vec::with_capacity(n_expert);
            for e in 0..n_expert {
                match cache.weight(base + e * per_bytes, out, inn, dtype) {
                    Ok(p) => paged.push((Weight::Paged(Arc::new(p)), None)),
                    Err(_) => {
                        paged.clear();
                        break;
                    }
                }
            }
            if paged.len() == n_expert {
                return Ok(paged);
            }
        }
    }

    // Device-resident experts with a mapping: upload each expert straight
    // from its bytes in the mapping.  This used to read the whole stacked
    // tensor into a host `Vec`, upload it as one device tensor, download it
    // again (`QTensor::data`) and re-upload it per expert — three bus
    // crossings and a transient whole-tensor device buffer per layer, which
    // on a 24 GiB card was the difference between a model that loads and one
    // that does not.  Now each expert is one copy from the page cache.
    if !host_experts {
        if let Some(mmap) = rd.mmap.as_ref() {
            if let Some(slices) = crate::mmap_tensor::expert_slices(mmap, dtype, base, n_expert, per_elems) {
                let mut experts = Vec::with_capacity(n_expert);
                for slice in slices {
                    let storage = QStorage::from_data(Cow::Borrowed(slice), &rd.expert_device, dtype)?;
                    let qt = QTensor::new(storage, (out, inn))?;
                    experts.push((Weight::Candle(QMatMul::from_qtensor(qt)?), None));
                }
                return Ok(experts);
            }
        }
    }

    // No mapping (streamed load) or a tensor that cannot be sliced: read the
    // stacked tensor once onto the host and copy per-expert slices to the
    // expert device.  The host copy is the only staging buffer.
    let qt = rd.ct.tensor(&mut rd.reader, name, &Device::Cpu)?;
    let bytes = qt.data()?;
    if bytes.len() % n_expert != 0 {
        candle_core::bail!(
            "qwen3moe: expert tensor `{name}` byte length {} not divisible by n_expert {n_expert}",
            bytes.len()
        );
    }
    let per = bytes.len() / n_expert;
    let mut experts = Vec::with_capacity(n_expert);
    for e in 0..n_expert {
        let slice = &bytes[e * per..(e + 1) * per];
        let storage = QStorage::from_data(Cow::Borrowed(slice), &rd.expert_device, dtype)?;
        let qt = QTensor::new(storage, (out, inn))?;
        experts.push((Weight::Candle(QMatMul::from_qtensor(qt)?), None));
    }
    Ok(experts)
}

fn load_moe<R: Read + Seek>(rd: &mut Reader<R>, p: &str, cfg: &Config) -> Result<Moe> {
    let gate_t = rd
        .f32_tensor(&format!("{p}.ffn_gate_inp.weight"))? // [n_expert, n_embd]
        .t()?
        .contiguous()?; // [n_embd, n_expert]

    let gate_exps = split_experts(rd, &format!("{p}.ffn_gate_exps.weight"), cfg.n_expert)?;
    let up_exps = split_experts(rd, &format!("{p}.ffn_up_exps.weight"), cfg.n_expert)?;
    let down_exps = split_experts(rd, &format!("{p}.ffn_down_exps.weight"), cfg.n_expert)?;
    let experts = gate_exps
        .into_iter()
        .zip(up_exps)
        .zip(down_exps)
        .map(|((gate, up), down)| Mlp {
            gate: gate.0,
            up: up.0,
            down: down.0,
            prefetch: match (gate.1, up.1, down.1) {
                (Some(gate), Some(up), Some(down)) => {
                    Some(crate::residency::ExpertHandles { gate, up, down })
                }
                _ => None,
            },
        })
        .collect();

    let layer = p
        .rsplit('.')
        .next()
        .and_then(|id| id.parse::<u32>().ok())
        .unwrap_or(0);
    Ok(Moe {
        gate_t,
        experts,
        n_expert_used: cfg.n_expert_used,
        weights_norm: cfg.expert_weights_norm,
        layer,
        expert_device: rd.expert_device.clone(),
        residency: None,
    })
}

impl GGUFQWenMoE {
    /// Load a `qwen3moe` GGUF (Qwen3-30B-A3B, Qwen3-Coder-30B-A3B).
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_mmap(ct, reader, device, None)
    }

    /// Load with weights borrowed in place from `mmap` where possible.
    ///
    /// Without a mapping every tensor is copied onto the heap and the whole
    /// model must fit in RAM.  With one, weights are referenced directly in
    /// the page cache and fault in on demand.  The routed experts go to
    /// `device` too; see [`GGUFQWenMoE::from_gguf_mmap_placed`] to keep them
    /// in host RAM on an accelerator.
    pub fn from_gguf_mmap<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        mmap: Option<Arc<memmap2::Mmap>>,
    ) -> Result<Self> {
        Self::from_gguf_mmap_placed(ct, reader, device, device, mmap)
    }

    /// [`GGUFQWenMoE::from_gguf_mmap`] with an explicit device for the routed
    /// experts.
    ///
    /// The dense set always goes to `device`.  With `expert_device` the CPU
    /// on an accelerator model, the routed-expert pool stays in host RAM —
    /// borrowed in place from `mmap` (or read onto the heap without one) and
    /// run through the CPU expert kernels, with the MoE block's activations
    /// hopping across the bus once per layer.  That is the layout for a
    /// model larger than the device's memory (see
    /// [`crate::placement::ExpertPlacement`]).  Any other `expert_device`
    /// must equal `device`.
    pub fn from_gguf_mmap_placed<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        expert_device: &Device,
        mmap: Option<Arc<memmap2::Mmap>>,
    ) -> Result<Self> {
        if !expert_device.is_cpu() && !expert_device.same_device(device) {
            candle_core::bail!(
                "qwen3moe: routed experts must live on the model device or the CPU, not {expert_device:?}"
            );
        }
        let cfg = Config::from_metadata(&ct.metadata)?;
        // Zero-copy Metal: bind the mapped weights into no-copy GPU buffers
        // so quantized weights are never uploaded.  Best effort — if the
        // mapping is not page-aligned or the device refuses the buffers, fall
        // back to candle's copying path.  Chunk boundaries follow tensor
        // boundaries so no weight straddles two buffers.
        let zc = match (&device, &mmap) {
            (Device::Metal(md), Some(mmap)) if !expert_device.is_cpu() => {
                match ZcContext::new_for_tensors(md, mmap.clone(), &ct.tensor_infos, ct.tensor_data_offset)
                {
                    Ok(zc) => {
                        tracing::info!(
                            "zero-copy Metal: binding {} bytes of weights into {} no-copy GPU buffers",
                            zc.len(),
                            zc.num_chunks()
                        );
                        Some(Arc::new(zc))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "zero-copy Metal unavailable ({}); copying weights onto the GPU",
                            e
                        );
                        None
                    }
                }
            }
            _ => None,
        };
        // Opt-in bounded GPU weight cache.  Heed the env budget (MiB); 0/unset
        // disables.  Only meaningful with an mmap-backed model on a device that
        // can hold quantized blocks; building it is best-effort.
        let gpu_cache = match std::env::var("JOSHUA_GPU_WEIGHT_CACHE") {
            Ok(v) if !v.is_empty() && v != "0" => {
                let MiB = v.parse::<usize>().unwrap_or(0);
                if MiB == 0 {
                    None
                } else if let Some(m) = mmap.clone() {
                    crate::paged_weights::WeightCache::new(
                        m.clone(),
                        device.clone(),
                        MiB * 1024 * 1024,
                    ).ok()
                } else {
                    tracing::warn!("JOSHUA_GPU_WEIGHT_CACHE set but no mmap-backed model; disabled");
                    None
                }
            }
            _ => None,
        };
        // The Content owns metadata; move it into our reader together with the
        // underlying file handle (borrowed for the lifetime of the load).
        let mut rd = Reader {
            ct,
            reader,
            device: device.clone(),
            expert_device: expert_device.clone(),
            mmap,
            zc,
            gpu_cache,
        };

        // Kept quantized: dequantizing the table to f32 costs vocab × hidden
        // × 4 bytes of anonymous memory per instance (1.2 GiB on 30B-A3B).
        let tok_embeddings = TokenEmbedding::load(rd.qtensor("token_embd.weight")?, device)?;
        let norm = rd.rms_norm("output_norm.weight", cfg.rms_eps)?;
        let output = match rd.qmatmul_opt("output.weight") {
            Some(q) => q,
            // tie_word_embeddings conversions ship no output head.
            None => rd.qmatmul("token_embd.weight")?,
        };

        let rotary = Arc::new(RotaryEmbedding::new(
            cfg.head_dim,
            cfg.context_length,
            cfg.rope_theta,
            device,
        )?);

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for layer_idx in 0..cfg.n_layer {
            let p = format!("blk.{layer_idx}");
            let attn_norm = rd.rms_norm(&format!("{p}.attn_norm.weight"), cfg.rms_eps)?;
            let ffn_norm = rd.rms_norm(&format!("{p}.ffn_norm.weight"), cfg.rms_eps)?;
            let attn = Attention::new(&mut rd, &p, &cfg, rotary.clone())?;
            let ffn = load_moe(&mut rd, &p, &cfg)?;
            layers.push(Layer {
                attn_norm,
                attn,
                ffn_norm,
                ffn,
            });
        }

        let n_layers = layers.len();
        let residency: std::sync::Arc<dyn crate::residency::ExpertResidency> =
            std::sync::Arc::new(crate::residency::CpuResidency::new(
                layers
                    .iter()
                    .map(|layer| layer.ffn.experts.iter().map(|m| m.prefetch.clone()).collect())
                    .collect(),
            ));
        Ok(Self {
            shared: Arc::new(Shared {
                tok_embeddings,
                layers,
                norm,
                output,
                device: device.clone(),
                residency,
                n_expert: cfg.n_expert,
            }),
            kv: vec![None; n_layers],
            hot_experts: crate::hot_experts::HotExpertCache::new(
                n_layers,
                cfg.n_expert,
                0,
            ),
        })
    }

    /// A fresh session over the same weights: shares every weight tensor
    /// (one `Arc` clone — no read, no upload) and starts with an empty KV
    /// cache and a fresh routing record under the same hot-expert budget.
    pub fn new_session(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            kv: vec![None; self.kv.len()],
            hot_experts: crate::hot_experts::HotExpertCache::new(
                self.kv.len(),
                self.shared.n_expert,
                self.hot_experts.budget(),
            ),
        }
    }

    /// Number of sessions (including this one) sharing these weights.
    pub fn shared_session_count(&self) -> usize {
        Arc::strong_count(&self.shared)
    }

    /// Whether the token-embedding table is held quantized (diagnostics).
    pub fn embeddings_quantized(&self) -> bool {
        self.shared.tok_embeddings.is_quantized()
    }

    fn causal_mask(&self, seq_len: usize, offset: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len + offset).map(move |j| {
                    if j > i + offset {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (1, 1, seq_len, seq_len + offset), &self.shared.device)
    }

    /// Forward pass. `input` is `[1, seq_len]`; `offset` is the KV-cache
    /// position of the first input token.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, seq_len) = input.dims2()?;
        let sh = Arc::clone(&self.shared);
        let mut xs = sh
            .tok_embeddings
            .forward(&input.flatten_all()?)?
            .reshape((1, seq_len, sh.tok_embeddings.hidden()?))?;

        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.causal_mask(seq_len, offset)?)
        };

        // Routing-frequency hot-expert cache: every
        // crate::hot_experts::REFRESH_STEPS decode steps, re-select the
        // most-used experts (recency as the tie-break) and WILLNEED their
        // pages, so the common routing path stays resident instead of
        // faulting from disk each step.  Runs before the layer loop so the
        // prefetch has a full step of compute to stream in behind.
        let step = self.hot_experts.begin_step(seq_len == 1);
        if self.hot_experts.refresh_due(seq_len == 1) {
            for (l, e) in self.hot_experts.refresh() {
                sh.residency.acquire(l, e);
            }
        }

        for (i, layer) in sh.layers.iter().enumerate() {
            let residual = &xs;
            let h = layer.attn_norm.forward(&xs)?;
            let h = layer.attn.forward(&mut self.kv[i], &h, mask.as_ref(), offset)?;
            let xs2 = (residual + h)?;

            let residual = &xs2;
            let h = layer.ffn_norm.forward(&xs2)?;
            let (h, routed) = layer.ffn.forward(&h)?;
            self.hot_experts.record(i, &routed, step);
            xs = (residual + h)?;
        }

        let xs = xs.narrow(1, seq_len - 1, 1)?;
        let xs = sh.norm.forward(&xs)?;
        let _p = prof::Phase::start(&prof::HEAD);
        let out = sh.output.forward(&xs)?.to_dtype(DType::F32)?.squeeze(1);
        drop(_p);
        prof::report();
        out
    }

    /// Set the routing-frequency hot-expert cache budget (experts kept
    /// resident).  Call once after load, before serving; routing is recorded
    /// from the first forward pass and the pinned set is re-selected every
    /// [`crate::hot_experts::REFRESH_STEPS`] decode steps.
    pub fn set_pin_hot_experts(&mut self, n: usize) {
        self.hot_experts.set_budget(n);
    }

    /// Number of experts the residency backend can hold resident
    /// (informational on CPU; a phase-5 auto-sizing input on devices).
    pub fn expert_residency_capacity(&self) -> usize {
        self.shared.residency.capacity()
    }

    /// Reset the KV cache so this instance can serve an unrelated prompt.
    pub fn clear_kv_cache(&mut self) {
        for kv in self.kv.iter_mut() {
            *kv = None;
        }
    }

    /// Keep only the first `keep` fed tokens of every layer's KV cache.
    ///
    /// See [`Attention::truncate_kv_cache`] for the semantics; used by the
    /// engine's edited-context prefix reuse (a follow-up prompt that shares
    /// a prefix with the cached history after an agent harness truncated or
    /// replaced middle blocks).
    pub fn truncate_kv_cache(&mut self, keep: usize) -> Result<()> {
        for kv in self.kv.iter_mut() {
            truncate_kv(kv, keep)?;
        }
        Ok(())
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lin(_rows: usize, _cols: usize, t: &Tensor) -> Weight {
        Weight::Candle(QMatMul::Tensor(t.clone()))
    }

    fn rms_from_f32(weights: &[f32], dim: usize, dev: &Device) -> Result<RmsNorm> {
        let bytes: Vec<u8> = weights.iter().flat_map(|x| x.to_le_bytes()).collect();
        let storage = QStorage::from_data(Cow::Owned(bytes), dev, GgmlDType::F32)?;
        RmsNorm::from_qtensor(QTensor::new(storage, dim)?, 1e-5)
    }

    fn tiny_moe(dev: &Device, weights_norm: bool) -> Result<Moe> {
        let (h, ne, nfe) = (8usize, 4usize, 16usize);
        let gate_t = Tensor::randn(0f32, 1f32, (h, ne), dev)?.contiguous()?; // [n_embd, n_expert]
        let mut experts = Vec::with_capacity(ne);
        for _ in 0..ne {
            experts.push(Mlp {
                gate: lin(nfe, h, &Tensor::randn(0f32, 1f32, (nfe, h), dev)?),
                up: lin(nfe, h, &Tensor::randn(0f32, 1f32, (nfe, h), dev)?),
                down: lin(h, nfe, &Tensor::randn(0f32, 1f32, (h, nfe), dev)?),
                prefetch: None,
            });
        }
        Ok(Moe {
            gate_t,
            experts,
            n_expert_used: 2,
            weights_norm,
            layer: 0,
            expert_device: dev.clone(),
            residency: None,
        })
    }

    /// A mixed-residency `Moe` (some experts resident in a `DeviceResidency`,
    /// the rest host) must produce **same logits** as the all-host `Moe` from
    /// the same weights.  The fake "device" form wraps the same host weights,
    /// so the assertion exercises the dispatch *partition* (per-expert lookup /
    /// hit-vs-miss routing), not real device math — the actual VRAM copy and
    /// GPU matmul still need a real accelerator to validate (issue #62).
    #[test]
    fn mixed_residency_dispatch_matches_all_host() -> Result<()> {
        let dev = Device::Cpu;
        let moe = tiny_moe(&dev, false)?; // all-host reference

        // A cache whose slots wrap the **same** expert weights (as "device
        // form"), so a hit runs numerically-identical math on the CPU.
        let experts: Vec<Mlp> = moe.experts.clone();
        let upload: std::sync::Arc<
            dyn Fn(u32, u32) -> Option<std::sync::Arc<DeviceExpert>> + Send + Sync,
        > = std::sync::Arc::new(move |l, e| {
            experts
                .get(e as usize)
                .map(|m| {
                    std::sync::Arc::new(DeviceExpert {
                        gate: m.gate.clone(),
                        up: m.up.clone(),
                        down: m.down.clone(),
                        bytes: 1024,
                    })
                })
                .filter(|_| l == 0)
        });
        let res = std::sync::Arc::new(crate::residency::DeviceResidency::<DeviceExpert>::new(
            2048, 1024, upload,
        ));
        // Experts 1,2 resident; 0,3 are host misses -> a genuine mixed layout.
        res.mark_hot(0, 1);
        res.mark_hot(0, 2);
        res.acquire(0, 1);
        res.acquire(0, 2);

        // mixed shares moe's exact weights (same clones), differing only in the
        // residency: hits run the (numerically identical) device form, misses run
        // the host form.
        let mixed = Moe {
            gate_t: moe.gate_t.clone(),
            experts: moe.experts.clone(),
            n_expert_used: moe.n_expert_used,
            weights_norm: moe.weights_norm,
            layer: moe.layer,
            expert_device: moe.expert_device.clone(),
            residency: Some(std::sync::Arc::clone(&res)),
        };
        let _ = &res;

        let xs = Tensor::randn(0f32, 1f32, (1, 4, 8), &dev)?; // [b, seq, h]
        let (a, _) = moe.forward(&xs)?;
        let (b, _) = mixed.forward(&xs)?;
        let av = a.flatten_all()?.to_vec1::<f32>()?;
        let bv = b.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(av.len(), bv.len());
        for (i, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-5,
                "mixed/host logit {i} diverges: {x} vs {y}"
            );
        }
        Ok(())
    }

    /// Half-split RoPE must pair `(i, i + d/2)` with the `(d/2)`-long
    /// frequency table — the exact convention llama.cpp uses for qwen3moe.
    #[test]
    fn rope_half_split_matches_manual() -> Result<()> {
        let dev = Device::Cpu;
        let dim = 8usize;
        let theta = 10_000f32;
        let rope = RotaryEmbedding::new(dim, 64, theta, &dev)?;
        let x = Tensor::randn(0f32, 1f32, (1, 1, 1, dim), &dev)?; // [b, h, seq, d]
        let (rotated, _) = rope.apply(&x, &x, 5)?;

        let xv: Vec<f32> = x.flatten_all()?.to_vec1()?;
        let rv: Vec<f32> = rotated.flatten_all()?.to_vec1()?;
        let half = dim / 2;
        let mut expected = vec![0f32; dim];
        for i_d in 0..half {
            let freq = 1f32 / theta.powf((2 * i_d) as f32 / dim as f32);
            let c = (5f32 * freq).cos();
            let s = (5f32 * freq).sin();
            expected[i_d] = xv[i_d] * c - xv[i_d + half] * s;
            expected[i_d + half] = xv[i_d] * s + xv[i_d + half] * c;
        }
        for i in 0..dim {
            assert!(
                (rv[i] - expected[i]).abs() < 1e-5,
                "rope element {i}: got {}, expected {}",
                rv[i],
                expected[i]
            );
        }
        Ok(())
    }

    /// With `norm_topk_prob`, the gathered routing weights must sum to 1 for
    /// every token (llama.cpp `build_moe_ffn` normalisation).
    #[test]
    fn moe_routing_normalizes_weights() -> Result<()> {
        let dev = Device::Cpu;
        let moe = tiny_moe(&dev, true)?;
        let xs = Tensor::randn(0f32, 1f32, (1, 3, 8), &dev)?;
        let (_, weights) = moe.route(&xs.reshape((3, 8))?)?;
        let sums: Vec<f32> = weights.sum_keepdim(D::Minus1)?.flatten_all()?.to_vec1()?;
        for s in sums {
            assert!((s - 1.0).abs() < 1e-4, "routing weights must sum to 1, got {s}");
        }
        Ok(())
    }

    /// The decode dispatch path must agree with the prefill bucketing path for
    /// the same token (they compute the same weighted expert sum).
    #[test]
    fn moe_prefill_matches_decode_path() -> Result<()> {
        let dev = Device::Cpu;
        let moe = tiny_moe(&dev, true)?;
        let row = Tensor::randn(0f32, 1f32, (1, 1, 8), &dev)?;
        let two = Tensor::cat(&[&row, &row], 1)?; // [1, 2, 8], identical rows
        let (out2, _) = moe.forward(&two)?; // prefill path (n_tokens = 2)
        let (out1, _) = moe.forward(&row)?; // decode path (n_tokens = 1)
        let a: Vec<f32> = out2.narrow(1, 1, 1)?.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = out1.flatten_all()?.to_vec1()?;
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-4, "prefill vs decode diverge: {x} vs {y}");
        }
        Ok(())
    }

    /// Per-head Q/K norms reshape correctly and the KV cache appends along the
    /// sequence dim with the right per-head layout.
    #[test]
    fn attention_qk_norm_and_kv_cache_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let (h, nh, nkv, hd) = (8usize, 2usize, 1usize, 4usize);
        let mk = |r: usize, c: usize| Tensor::randn(0f32, 1f32, (r, c), &dev).unwrap();
        let attn = Attention {
            q: lin(nh * hd, h, &mk(nh * hd, h)),
            k: lin(nkv * hd, h, &mk(nkv * hd, h)),
            v: lin(nkv * hd, h, &mk(nkv * hd, h)),
            o: lin(h, nh * hd, &mk(h, nh * hd)),
            q_norm: rms_from_f32(&vec![1.0; hd], hd, &dev)?,
            k_norm: rms_from_f32(&vec![1.0; hd], hd, &dev)?,
            rotary: Arc::new(RotaryEmbedding::new(hd, 64, 10_000.0, &dev)?),
            n_head: nh,
            n_kv_head: nkv,
            head_dim: hd,
            softmax_scale: 1.0 / (hd as f64).sqrt(),
        };
        let mut kv_cache: KvCache = None;

        let xs = Tensor::randn(0f32, 1f32, (1, 3, h), &dev)?;
        let out = attn.forward(&mut kv_cache, &xs, None, 0)?;
        assert_eq!(out.dims(), &[1, 3, h]);
        let (k, v) = kv_cache.as_ref().expect("cache after prefill");
        assert_eq!(k.dims(), &[1, nkv, 3, hd], "k cache must be [b, n_kv_head, seq, head_dim]");
        assert_eq!(v.dims(), &[1, nkv, 3, hd]);

        // Decode appends one more position.
        let _ = attn.forward(&mut kv_cache, &Tensor::randn(0f32, 1f32, (1, 1, h), &dev)?, None, 3)?;
        let (k, v) = kv_cache.as_ref().expect("cache after decode");
        assert_eq!(k.dims(), &[1, nkv, 4, hd]);
        assert_eq!(v.dims(), &[1, nkv, 4, hd]);
        Ok(())
    }

    /// `attention.key_length` must win over `embedding_length / head_count`
    /// (Qwen3-Coder: 128 vs 2048/32 = 64), and `norm_topk_prob` defaults to
    /// true when a conversion omits it.
    #[test]
    fn config_prefers_key_length_and_defaults_topk_norm() -> Result<()> {
        use std::collections::HashMap;
        let mut md = HashMap::new();
        let u = |v: u32| gguf_file::Value::U32(v);
        md.insert("qwen3moe.attention.head_count".into(), u(32));
        md.insert("qwen3moe.attention.head_count_kv".into(), u(4));
        md.insert("qwen3moe.block_count".into(), u(48));
        md.insert("qwen3moe.embedding_length".into(), u(2048));
        md.insert("qwen3moe.context_length".into(), u(262_144));
        md.insert(
            "qwen3moe.attention.layer_norm_rms_epsilon".into(),
            gguf_file::Value::F32(1e-6),
        );
        md.insert("qwen3moe.expert_count".into(), u(128));
        md.insert("qwen3moe.expert_used_count".into(), u(8));
        md.insert("qwen3moe.expert_feed_forward_length".into(), u(768));

        // No key_length, no norm_topk_prob: fallbacks apply.
        let cfg = Config::from_metadata(&md)?;
        assert_eq!(cfg.head_dim, 2048 / 32);
        assert!(cfg.expert_weights_norm, "norm_topk_prob must default to true");
        assert_eq!(cfg.n_expert, 128);
        assert_eq!(cfg.n_expert_used, 8);

        // Qwen3-Coder advertises key_length 128 explicitly.
        md.insert("qwen3moe.attention.key_length".into(), u(128));
        let cfg = Config::from_metadata(&md)?;
        assert_eq!(cfg.head_dim, 128);
        Ok(())
    }
}
