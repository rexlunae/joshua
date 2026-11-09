//! LoRA adapters over a native loader's quantized projections.
//!
//! An adapter is never merged into the base weights.  Those stay quantized,
//! borrowed from the mapping and shared by every session; each targeted
//! projection instead adds a low-rank delta after the base matmul and before
//! the bias, which is llama.cpp's order (`build_lora_mm`, with the bias added
//! by the caller):
//!
//! ```text
//! y = W·x + s₁·B₁(A₁·x) + s₂·B₂(A₂·x) + …   (+ b)
//! ```
//!
//! `A` is `[r, in]` and `B` is `[out, r]`, and `s` is the effective scale
//! `user_scale · alpha / r`, or the user scale alone when the adapter's
//! alpha is 0 (`llama_adapter_lora_weight::get_scale`).
//!
//! This module holds three pieces:
//!
//! * [`LoraAdapter`]: a GGUF adapter in llama.cpp's format (what
//!   `convert_lora_to_gguf.py` writes), parsed and validated against the base
//!   model's [`SlotTable`] the way `llama-adapter.cpp` validates it.  Targets
//!   this engine cannot apply a delta to (routed experts, routers, Gated
//!   DeltaNet projections, the token embedding, the output head) fail the
//!   load instead of being skipped, and activated LoRA (aLoRA) is refused.
//! * [`ActiveSet`]: one or more adapters with their scales, laid out per
//!   layer and projection for the forward pass, named by an [`AdapterKey`]
//!   that every reuse of computed state (pooled sessions, snapshots) must
//!   match.
//! * The delta kernels behind [`SlotDelta::apply`].  On the CPU a fused kernel
//!   adds into the base output in place and computes each output from its
//!   own input row alone, so a one-row decode step and a many-row prefill or
//!   verification pass add bit-identical deltas.  The scale multiplies `A·x`
//!   rather than being folded into `B`, so a per-request scale costs no copy
//!   of `B`.  On an accelerator the delta is two f32 matmuls on the weights'
//!   device.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::QMatMul;
use candle_core::{CpuStorage, DType, Device, InplaceOp2, Layout, Module, Tensor};
use sha2::{Digest, Sha256};

use crate::{JoshuaError, Result};

/// A projection a LoRA delta can target, named as in the GGUF
/// (`blk.{layer}.{name}.weight`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Proj {
    AttnQ,
    AttnK,
    AttnV,
    /// The fused `[q ‖ k ‖ v]` projection (Qwen1, ChatGLM).
    AttnQkv,
    AttnOutput,
    FfnGate,
    /// The up projection, or the fused `[gate ‖ up]` one where the file has
    /// no `ffn_gate` (ChatGLM, GLM-4).
    FfnUp,
    FfnDown,
    FfnGateShexp,
    FfnUpShexp,
    FfnDownShexp,
}

/// Number of [`Proj`] variants.
pub const N_PROJ: usize = 11;

impl Proj {
    pub const ALL: [Proj; N_PROJ] = [
        Proj::AttnQ,
        Proj::AttnK,
        Proj::AttnV,
        Proj::AttnQkv,
        Proj::AttnOutput,
        Proj::FfnGate,
        Proj::FfnUp,
        Proj::FfnDown,
        Proj::FfnGateShexp,
        Proj::FfnUpShexp,
        Proj::FfnDownShexp,
    ];

    /// The tensor name's projection part (`attn_q`, `ffn_up_shexp`, …).
    pub fn gguf_name(self) -> &'static str {
        match self {
            Proj::AttnQ => "attn_q",
            Proj::AttnK => "attn_k",
            Proj::AttnV => "attn_v",
            Proj::AttnQkv => "attn_qkv",
            Proj::AttnOutput => "attn_output",
            Proj::FfnGate => "ffn_gate",
            Proj::FfnUp => "ffn_up",
            Proj::FfnDown => "ffn_down",
            Proj::FfnGateShexp => "ffn_gate_shexp",
            Proj::FfnUpShexp => "ffn_up_shexp",
            Proj::FfnDownShexp => "ffn_down_shexp",
        }
    }

    pub fn from_gguf_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.gguf_name() == name)
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// One projection of one decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SlotKey {
    pub layer: usize,
    pub proj: Proj,
}

impl SlotKey {
    pub fn new(layer: usize, proj: Proj) -> Self {
        Self { layer, proj }
    }

    /// The base weight's GGUF tensor name, `blk.{layer}.{proj}.weight`.
    pub fn tensor_name(&self) -> String {
        format!("blk.{}.{}.weight", self.layer, self.proj.gguf_name())
    }
}

/// The projections of a loaded model that can take a LoRA delta, with each
/// one's `[out, in]` dims, plus what adapter validation needs to name the
/// targets it refuses.  A loader that supports LoRA builds one from its
/// loaded weights; [`LoraAdapter`] is validated against it.
#[derive(Debug, Clone)]
pub struct SlotTable {
    arch: String,
    n_layer: usize,
    device: Device,
    slots: BTreeMap<SlotKey, (usize, usize)>,
    /// Layers whose token mixer is a Gated DeltaNet.
    recurrent: Vec<bool>,
    /// `(n_head, n_kv_head)` when the GGUF stores `attn_q` / `attn_k` rows
    /// permuted for interleaved RoPE (llama.cpp's `llama` arch), which a
    /// Hugging Face (PEFT) adapter's `B` rows must be permuted to match.
    rope_permute: Option<(usize, usize)>,
}

impl SlotTable {
    /// An empty table for an `arch` model of `n_layer` decoder layers whose
    /// dense weights live on `device`.
    pub fn new(arch: impl Into<String>, n_layer: usize, device: Device) -> Self {
        Self {
            arch: arch.into(),
            n_layer,
            device,
            slots: BTreeMap::new(),
            recurrent: vec![false; n_layer],
            rope_permute: None,
        }
    }

    /// Record that `attn_q` / `attn_k` rows are permuted for interleaved
    /// RoPE with these head counts (see [`Self::rope_permute`]).
    pub fn set_rope_permute(&mut self, n_head: usize, n_kv_head: usize) {
        self.rope_permute = Some((n_head, n_kv_head));
    }

    /// `(n_head, n_kv_head)` when the model's Q/K rows are permuted for
    /// interleaved RoPE, as `convert_hf_to_gguf.py` writes llama-arch models.
    pub fn rope_permute(&self) -> Option<(usize, usize)> {
        self.rope_permute
    }

    /// Record a projection that takes a delta, `[n_out, n_in]`.
    pub fn insert(&mut self, key: SlotKey, n_out: usize, n_in: usize) {
        self.slots.insert(key, (n_out, n_in));
    }

    /// Record that `layer`'s mixer is a Gated DeltaNet, so an adapter
    /// targeting its projections is refused by name.
    pub fn mark_recurrent(&mut self, layer: usize) {
        if let Some(r) = self.recurrent.get_mut(layer) {
            *r = true;
        }
    }

    /// `general.architecture` of the base model.
    pub fn arch(&self) -> &str {
        &self.arch
    }

    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    /// The device adapter weights are placed on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `[out, in]` of a projection that takes a delta.
    pub fn dims(&self, key: SlotKey) -> Option<(usize, usize)> {
        self.slots.get(&key).copied()
    }

    /// Every projection that takes a delta, in `(layer, proj)` order, with
    /// its `[out, in]` dims.
    pub fn iter(&self) -> impl Iterator<Item = (SlotKey, usize, usize)> + '_ {
        self.slots.iter().map(|(&k, &(o, i))| (k, o, i))
    }

    /// SHA-256 of the architecture, layer count and every slot's dims: two
    /// loaded models with the same fingerprint accept adapters of the same
    /// shapes.  The device is not part of it: an adapter's weights live on
    /// the device of the table it was loaded against (see
    /// [`LoraAdapter::device`]), and a session checks that separately.
    pub fn fingerprint(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"joshua/lora-slots/v1");
        hash.update((self.arch.len() as u64).to_le_bytes());
        hash.update(self.arch.as_bytes());
        hash.update((self.n_layer as u64).to_le_bytes());
        for (k, &(o, i)) in &self.slots {
            for v in [k.layer, k.proj.index(), o, i] {
                hash.update((v as u64).to_le_bytes());
            }
        }
        for &r in &self.recurrent {
            hash.update([u8::from(r)]);
        }
        // A PEFT adapter's Q/K rows are permuted for these head counts.
        let (n_head, n_kv_head) = self.rope_permute.unwrap_or((0, 0));
        hash.update([u8::from(self.rope_permute.is_some())]);
        hash.update((n_head as u64).to_le_bytes());
        hash.update((n_kv_head as u64).to_le_bytes());
        hash.finalize().into()
    }

    /// Where an adapter tensor pair for base tensor `base` goes, or why it
    /// cannot go anywhere.
    pub(crate) fn target(&self, base: &str) -> std::result::Result<(SlotKey, usize, usize), String> {
        const NOT_YET: &str = "which takes no LoRA delta in this version";
        match base {
            "token_embd.weight" => return Err(format!("targets the token embedding, {NOT_YET}")),
            "output.weight" => return Err(format!("targets the output head, {NOT_YET}")),
            _ => {}
        }
        let parsed = base
            .strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
            .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest.strip_suffix(".weight")?)));
        let Some((layer, proj)) = parsed else {
            return Err("is not a projection this engine applies LoRA deltas to".into());
        };
        if layer >= self.n_layer {
            return Err(format!(
                "targets block {layer}, past the model's {} decoder layers (a NextN / MTP draft \
                 block takes no LoRA delta)",
                self.n_layer
            ));
        }
        if proj.ends_with("_exps") {
            return Err(format!("targets routed experts, {NOT_YET}"));
        }
        if proj.starts_with("ffn_gate_inp") {
            return Err(format!("targets an MoE router gate, {NOT_YET}"));
        }
        if self.recurrent[layer]
            && (proj.starts_with("ssm_") || proj == "attn_qkv" || proj == "attn_gate")
        {
            return Err(format!("targets a Gated DeltaNet projection, {NOT_YET}"));
        }
        let Some(proj) = Proj::from_gguf_name(proj) else {
            return Err("is not a projection this engine applies LoRA deltas to".into());
        };
        let key = SlotKey::new(layer, proj);
        match self.dims(key) {
            Some((o, i)) => Ok((key, o, i)),
            None => Err("does not exist in base model (hint: maybe wrong base model?)".into()),
        }
    }
}

/// Caps on what an adapter load and an [`ActiveSet`] may hold.
#[derive(Debug, Clone, Copy)]
pub struct LoraLimits {
    /// Largest rank of any tensor pair.
    pub max_rank: usize,
    /// Largest adapter, in bytes of the file and of the f32 weights it
    /// decodes to.
    pub max_bytes: u64,
    /// Most adapters one [`ActiveSet`] combines.
    pub max_adapters: usize,
}

impl Default for LoraLimits {
    fn default() -> Self {
        Self {
            max_rank: 1024,
            max_bytes: 4 << 30,
            max_adapters: 16,
        }
    }
}

/// One adapter's `A` `[rank, n_in]` and `B` `[n_out, rank]` for one
/// projection, in f32.
pub struct LoraPair {
    rank: usize,
    n_in: usize,
    n_out: usize,
    data: PairData,
}

enum PairData {
    /// Host copies for the CPU kernel: `A` `[rank, n_in]` and `B`
    /// transposed, `Bᵀ` `[rank, n_out]`, so both stages sweep contiguous
    /// rows.
    Host { a: Vec<f32>, bt: Vec<f32> },
    /// `A` `[rank, n_in]` and `B` `[n_out, rank]` on an accelerator, as f32
    /// `QMatMul`s: each product runs the way an f32 weight runs on every
    /// backend.
    Device { a: QMatMul, b: QMatMul },
}

impl LoraPair {
    /// A pair from row-major `a` (`[rank, n_in]`) and `b` (`[n_out, rank]`),
    /// held on `device`.
    pub fn new(
        a: Vec<f32>,
        b: Vec<f32>,
        rank: usize,
        n_in: usize,
        n_out: usize,
        device: &Device,
    ) -> candle_core::Result<Self> {
        if rank == 0 || a.len() != rank * n_in || b.len() != n_out * rank {
            candle_core::bail!(
                "LoRA pair: {} / {} values for rank {rank}, [{n_out}, {n_in}]",
                a.len(),
                b.len()
            );
        }
        let data = if device.is_cpu() {
            let mut bt = vec![0f32; rank * n_out];
            for (o, row) in b.chunks_exact(rank).enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    bt[j * n_out + o] = v;
                }
            }
            PairData::Host { a, bt }
        } else {
            PairData::Device {
                a: QMatMul::Tensor(Tensor::from_vec(a, (rank, n_in), device)?),
                b: QMatMul::Tensor(Tensor::from_vec(b, (n_out, rank), device)?),
            }
        };
        Ok(Self {
            rank,
            n_in,
            n_out,
            data,
        })
    }

    /// [`Self::new`] held as tensors even on the CPU, so tests can run the
    /// accelerator path against the CPU kernel.
    #[cfg(test)]
    fn new_tensors(
        a: Vec<f32>,
        b: Vec<f32>,
        rank: usize,
        n_in: usize,
        n_out: usize,
    ) -> candle_core::Result<Self> {
        let pair = Self::new(a.clone(), b.clone(), rank, n_in, n_out, &Device::Cpu)?;
        Ok(Self {
            data: PairData::Device {
                a: QMatMul::Tensor(Tensor::from_vec(a, (rank, n_in), &Device::Cpu)?),
                b: QMatMul::Tensor(Tensor::from_vec(b, (n_out, rank), &Device::Cpu)?),
            },
            ..pair
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    /// `[out, in]` of the projection this pair adds to.
    pub fn dims(&self) -> (usize, usize) {
        (self.n_out, self.n_in)
    }
}

/// A GGUF LoRA adapter in llama.cpp's format, validated against one base
/// model's [`SlotTable`] and decoded to f32 on that model's device.
pub struct LoraAdapter {
    digest: [u8; 32],
    base: [u8; 32],
    device: Device,
    alpha: f32,
    task_name: Option<String>,
    prompt_prefix: Option<String>,
    pairs: Vec<(SlotKey, Arc<LoraPair>)>,
    bytes: u64,
}

impl fmt::Debug for LoraAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoraAdapter")
            .field("digest", &Hex(&self.digest))
            .field("device", &self.device.location())
            .field("alpha", &self.alpha)
            .field("pairs", &self.pairs.len())
            .field("bytes", &self.bytes)
            .finish()
    }
}

pub(crate) fn invalid(msg: impl fmt::Display) -> JoshuaError {
    JoshuaError::ModelLoad(format!("LoRA adapter: {msg}"))
}

/// Tensor data types an adapter's `A` / `B` may be stored in: what
/// `convert_lora_to_gguf.py --outtype` writes (F32, F16, BF16, Q8_0) and
/// anything else [`crate::quant_matmul::decode_raw_to_f32`] decodes.  F32 and
/// BF16 are decoded here: that function leaves F32 to candle's reader by
/// design, and teaching it BF16 would move the streamed loaders' BF16 base
/// tensors off candle's path.
fn decode_f32(name: &str, dtype: u32, bytes: &[u8], elems: usize) -> Result<Vec<f32>> {
    const F32: u32 = 0;
    const BF16: u32 = 30;
    let values = match dtype {
        F32 if bytes.len() == elems * 4 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        BF16 if bytes.len() == elems * 2 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        F32 | BF16 => return Err(invalid(format!("tensor '{name}': truncated data"))),
        _ => crate::quant_matmul::decode_raw_to_f32(dtype, bytes, elems)
            .map_err(|e| invalid(format!("tensor '{name}': {e}")))?
            .ok_or_else(|| {
                invalid(format!(
                    "tensor '{name}' has GGUF type {dtype}, which LoRA weights cannot use"
                ))
            })?,
    };
    if values.iter().any(|v: &f32| !v.is_finite()) {
        return Err(invalid(format!("tensor '{name}' holds non-finite values")));
    }
    Ok(values)
}

impl LoraAdapter {
    /// Read and validate the adapter at `path` against `table`.
    ///
    /// The file is mapped, not read: while the pairs are decoded the file's
    /// bytes stay in reclaimable page cache instead of a heap copy, so peak
    /// memory is the decoded pairs plus one pair in flight.
    pub fn load(path: &Path, table: &SlotTable, limits: &LoraLimits) -> Result<Self> {
        if let Some(peft) = crate::lora_peft::locate(path) {
            return crate::lora_peft::load(&peft, table, limits);
        }
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        if len > limits.max_bytes {
            return Err(invalid(format!(
                "{} is {len} bytes, over the {}-byte limit",
                path.display(),
                limits.max_bytes
            )));
        }
        // SAFETY: the mapping is read-only and dropped before returning; as
        // for the model itself, a file changed underneath the engine is
        // outside its contract.
        let mmap = unsafe { memmap2::Mmap::map(&file) }?;
        Self::from_bytes(&mmap, table, limits)
    }

    /// Parse and validate an adapter file's bytes against `table`, mirroring
    /// `llama_adapter_lora_init_impl`: `general.type` must be `adapter`,
    /// `general.architecture` the base's, `adapter.type` `lora`; tensors pair
    /// up as `<base>.lora_a` / `<base>.lora_b` (a `*_norm.weight` is skipped,
    /// any other name is an error); each pair's base tensor must exist with
    /// `A` `[r, in]` and `B` `[out, r]` matching it.
    pub fn from_bytes(bytes: &[u8], table: &SlotTable, limits: &LoraLimits) -> Result<Self> {
        if bytes.len() as u64 > limits.max_bytes {
            return Err(invalid(format!(
                "{} bytes, over the {}-byte limit",
                bytes.len(),
                limits.max_bytes
            )));
        }
        let digest: [u8; 32] = Sha256::digest(bytes).into();
        let mut reader = std::io::Cursor::new(bytes);
        let header = crate::gguf_ext::read_header(&mut reader).map_err(invalid)?;
        header
            .validate_layout(bytes.len() as u64)
            .map_err(invalid)?;

        let md = &header.metadata;
        let string = |key: &str| match md.get(key) {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        };
        let general_type = string("general.type").unwrap_or_default();
        if general_type != "adapter" {
            return Err(invalid(format!(
                "expect general.type to be 'adapter', but got: {general_type}"
            )));
        }
        let arch = string("general.architecture").unwrap_or_default();
        if arch != table.arch() {
            return Err(invalid(format!(
                "model arch and LoRA arch mismatch (adapter '{arch}', model '{}')",
                table.arch()
            )));
        }
        let adapter_type = string("adapter.type").unwrap_or_default();
        if adapter_type != "lora" {
            return Err(invalid(format!(
                "expect adapter.type to be 'lora', but got: {adapter_type}"
            )));
        }
        let alpha = match md.get("adapter.lora.alpha") {
            None => 0.0,
            Some(Value::F32(a)) if a.is_finite() => *a,
            Some(other) => {
                return Err(invalid(format!(
                    "adapter.lora.alpha must be a finite float32, got {other:?}"
                )))
            }
        };
        if md.keys().any(|k| k.starts_with("adapter.alora.")) {
            return Err(invalid(
                "activated LoRA (aLoRA, adapter.alora.invocation_tokens) is not supported",
            ));
        }

        // Pair `<base>.lora_a` with `<base>.lora_b`, in name order so the
        // first error reported does not depend on hash order.
        let mut names: Vec<&String> = header.tensors.keys().collect();
        names.sort();
        let mut halves: BTreeMap<&str, (Option<&str>, Option<&str>)> = BTreeMap::new();
        let mut skipped_norms = 0usize;
        for name in names {
            if let Some(base) = name.strip_suffix(".lora_a") {
                halves.entry(base).or_default().0 = Some(name);
            } else if let Some(base) = name.strip_suffix(".lora_b") {
                halves.entry(base).or_default().1 = Some(name);
            } else if name.ends_with("_norm.weight") {
                // llama.cpp ignores norm vectors too; most adapters work
                // without them.
                skipped_norms += 1;
            } else {
                return Err(invalid(format!(
                    "LoRA tensor '{name}' has unexpected suffix"
                )));
            }
        }
        if skipped_norms > 0 {
            tracing::warn!(
                "LoRA adapter: ignoring {skipped_norms} norm tensor(s), as llama.cpp does"
            );
        }
        if halves.is_empty() {
            return Err(invalid("the file holds no LoRA tensor pairs"));
        }

        let mut planned = Vec::with_capacity(halves.len());
        let mut total = 0u64;
        for (base, ab) in &halves {
            let (Some(a), Some(b)) = *ab else {
                return Err(invalid(format!(
                    "LoRA tensor pair for '{base}' is missing one component"
                )));
            };
            let (key, n_out, n_in) = table
                .target(base)
                .map_err(|why| invalid(format!("LoRA tensor '{base}' {why}")))?;
            let (ai, bi) = (&header.tensors[a], &header.tensors[b]);
            let (&[a_rows, a_cols], &[b_rows, b_cols]) = (&ai.dims[..], &bi.dims[..]) else {
                return Err(invalid(format!(
                    "tensor '{base}' has incorrect shape (hint: maybe wrong base model?)"
                )));
            };
            if a_cols != n_in || b_rows != n_out {
                return Err(invalid(format!(
                    "tensor '{base}' has incorrect shape (hint: maybe wrong base model?)"
                )));
            }
            if a_rows != b_cols {
                return Err(invalid(
                    "lora_a tensor is not transposed (hint: adapter from \"finetune\" example is \
                     no longer supported)",
                ));
            }
            let rank = b_cols;
            if rank == 0 || rank > limits.max_rank {
                return Err(invalid(format!(
                    "tensor '{base}' has rank {rank}, outside 1..={}",
                    limits.max_rank
                )));
            }
            total += (4 * rank * (n_in + n_out)) as u64;
            if total > limits.max_bytes {
                return Err(invalid(format!(
                    "the weights decode to more than the {}-byte limit",
                    limits.max_bytes
                )));
            }
            planned.push((key, a, b, rank, n_in, n_out));
        }

        let mut pairs = Vec::with_capacity(planned.len());
        for (key, a, b, rank, n_in, n_out) in planned {
            let mut half = |name: &str| -> Result<Vec<f32>> {
                let info = &header.tensors[name];
                let raw = header
                    .read_tensor_bytes(&mut reader, name)
                    .map_err(|e| invalid(format!("tensor '{name}': {e}")))?;
                decode_f32(name, info.dtype, &raw, info.elem_count())
            };
            let (a, b) = (half(a)?, half(b)?);
            let pair = LoraPair::new(a, b, rank, n_in, n_out, table.device()).map_err(invalid)?;
            pairs.push((key, Arc::new(pair)));
        }
        pairs.sort_by_key(|(k, _)| *k);
        Ok(Self {
            digest,
            base: table.fingerprint(),
            device: table.device().clone(),
            alpha,
            task_name: string("adapter.lora.task_name"),
            prompt_prefix: string("adapter.lora.prompt_prefix"),
            pairs,
            bytes: total,
        })
    }

    /// An adapter from pairs already decoded and placed against `table`
    /// (the PEFT reader, [`crate::lora_peft`]).  `alpha` 0 runs each pair at
    /// the user's scale alone.
    pub(crate) fn from_pairs(
        digest: [u8; 32],
        table: &SlotTable,
        alpha: f32,
        mut pairs: Vec<(SlotKey, Arc<LoraPair>)>,
        bytes: u64,
    ) -> Self {
        pairs.sort_by_key(|(k, _)| *k);
        Self {
            digest,
            base: table.fingerprint(),
            device: table.device().clone(),
            alpha,
            task_name: None,
            prompt_prefix: None,
            pairs,
            bytes,
        }
    }

    /// SHA-256 of the adapter file.
    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// The [`SlotTable::fingerprint`] the adapter was validated against.
    pub fn base(&self) -> &[u8; 32] {
        &self.base
    }

    /// The device the adapter's weights live on: that table's.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `adapter.lora.alpha` (0 when absent).
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// `adapter.lora.task_name`, when the file has one.
    pub fn task_name(&self) -> Option<&str> {
        self.task_name.as_deref()
    }

    /// `adapter.lora.prompt_prefix`, when the file has one.
    pub fn prompt_prefix(&self) -> Option<&str> {
        self.prompt_prefix.as_deref()
    }

    /// Bytes the decoded f32 weights hold.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The projections the adapter targets, in `(layer, proj)` order.
    pub fn targets(&self) -> impl Iterator<Item = SlotKey> + '_ {
        self.pairs.iter().map(|(k, _)| *k)
    }

    /// The effective scale of a pair of rank `rank` at `user_scale`:
    /// `user_scale · alpha / rank`, or `user_scale` when alpha is 0
    /// (llama.cpp's `get_scale`).
    pub fn scale(&self, user_scale: f32, rank: usize) -> f32 {
        if self.alpha != 0.0 {
            user_scale * self.alpha / rank as f32
        } else {
            user_scale
        }
    }
}

struct Hex<'a>(&'a [u8]);

impl fmt::Debug for Hex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// The identity of an adapter selection: SHA-256 of
/// `joshua/lora-set/v1` followed by each enabled `(content digest, scale)`
/// in canonical order.  State computed under one key is never reused under
/// another; the base model (no adapters, or every scale 0) has no key.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AdapterKey(pub [u8; 32]);

impl fmt::Debug for AdapterKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Hex(&self.0).fmt(f)
    }
}

/// Drop the zero scales and sort by `(digest, scale bits)`: the order the key
/// hashes and the deltas are summed in, so equal keys mean identical sums.
fn canonical(entries: &mut Vec<([u8; 32], f32, usize)>) {
    entries.retain(|(_, s, _)| *s != 0.0);
    entries.sort_by(|a, b| (a.0, a.1.to_bits()).cmp(&(b.0, b.1.to_bits())));
}

impl AdapterKey {
    /// The key of a selection of `(adapter digest, user scale)`, or `None`
    /// when no entry has a nonzero scale.  Independent of the input order.
    pub fn of(selection: &[([u8; 32], f32)]) -> Option<Self> {
        let mut entries: Vec<_> = selection.iter().map(|&(d, s)| (d, s, 0)).collect();
        canonical(&mut entries);
        Self::of_canonical(&entries)
    }

    fn of_canonical(entries: &[([u8; 32], f32, usize)]) -> Option<Self> {
        if entries.is_empty() {
            return None;
        }
        let mut hash = Sha256::new();
        hash.update(b"joshua/lora-set/v1");
        hash.update((entries.len() as u64).to_le_bytes());
        for (digest, scale, _) in entries {
            hash.update(digest);
            hash.update(scale.to_bits().to_le_bytes());
        }
        Some(Self(hash.finalize().into()))
    }
}

/// One or more adapters at their scales, laid out for the forward pass: per
/// layer, per projection, the pairs to add.  Built once per selection and
/// shared by every session that runs it.
pub struct ActiveSet {
    key: AdapterKey,
    base: [u8; 32],
    device: Device,
    adapters: Vec<(Arc<LoraAdapter>, f32)>,
    layers: Vec<Option<Box<LayerLora>>>,
}

impl fmt::Debug for ActiveSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ActiveSet")
            .field("key", &self.key)
            .field("adapters", &self.adapters.len())
            .finish()
    }
}

impl ActiveSet {
    /// The set for `selection` (`(adapter, user scale)`), or `None` when
    /// every scale is 0.  Entries are applied in canonical order (see
    /// [`AdapterKey`]), so the same selection in any order computes the same
    /// sums.  Every adapter must have been validated against the same base
    /// and placed on the same device.
    pub fn new(selection: &[(Arc<LoraAdapter>, f32)], limits: &LoraLimits) -> Result<Option<Self>> {
        if let Some((_, s)) = selection.iter().find(|(_, s)| !s.is_finite()) {
            return Err(JoshuaError::InvalidRequest(format!(
                "LoRA scale {s} is not finite"
            )));
        }
        let mut entries: Vec<_> = selection
            .iter()
            .enumerate()
            .map(|(i, (a, s))| (*a.digest(), *s, i))
            .collect();
        canonical(&mut entries);
        if entries.len() > limits.max_adapters {
            return Err(JoshuaError::InvalidRequest(format!(
                "{} LoRA adapters selected, over the limit of {}",
                entries.len(),
                limits.max_adapters
            )));
        }
        let Some(key) = AdapterKey::of_canonical(&entries) else {
            return Ok(None);
        };
        let adapters: Vec<_> = entries
            .iter()
            .map(|&(_, s, i)| (Arc::clone(&selection[i].0), s))
            .collect();
        let base = *adapters[0].0.base();
        if adapters.iter().any(|(a, _)| a.base() != &base) {
            return Err(JoshuaError::InvalidRequest(
                "the selected LoRA adapters were loaded for different base models".into(),
            ));
        }
        let device = adapters[0].0.device().clone();
        if adapters
            .iter()
            .any(|(a, _)| !a.device().same_device(&device))
        {
            return Err(JoshuaError::InvalidRequest(
                "the selected LoRA adapters were loaded onto different devices".into(),
            ));
        }
        let n_layers = adapters
            .iter()
            .flat_map(|(a, _)| a.targets())
            .map(|k| k.layer + 1)
            .max()
            .unwrap_or(0);
        let mut layers: Vec<Option<Box<LayerLora>>> = (0..n_layers).map(|_| None).collect();
        for (adapter, user_scale) in &adapters {
            for (key, pair) in &adapter.pairs {
                let layer = layers[key.layer].get_or_insert_with(|| {
                    Box::new(LayerLora {
                        slots: std::array::from_fn(|_| None),
                    })
                });
                let scale = adapter.scale(*user_scale, pair.rank);
                layer.slots[key.proj.index()]
                    .get_or_insert_with(|| SlotDelta { terms: Vec::new() })
                    .terms
                    .push((Arc::clone(pair), scale));
            }
        }
        Ok(Some(Self {
            key,
            base,
            device,
            adapters,
            layers,
        }))
    }

    pub fn key(&self) -> AdapterKey {
        self.key
    }

    /// The [`SlotTable::fingerprint`] the adapters were validated against.
    pub fn base(&self) -> &[u8; 32] {
        &self.base
    }

    /// The device the adapters' weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The adapters and user scales, in the order their deltas are summed.
    pub fn adapters(&self) -> &[(Arc<LoraAdapter>, f32)] {
        &self.adapters
    }

    /// Layer `l`'s deltas, or `None` when no adapter targets it.
    pub fn layer(&self, l: usize) -> Option<&LayerLora> {
        self.layers.get(l)?.as_deref()
    }
}

/// One layer's deltas, by projection.
pub struct LayerLora {
    slots: [Option<SlotDelta>; N_PROJ],
}

impl LayerLora {
    pub fn get(&self, proj: Proj) -> Option<&SlotDelta> {
        self.slots[proj.index()].as_ref()
    }
}

/// `layer`'s delta for `proj`, when there is one.
pub fn slot(layer: Option<&LayerLora>, proj: Proj) -> Option<&SlotDelta> {
    layer.and_then(|l| l.get(proj))
}

/// The deltas one projection adds: each pair with its effective scale, in
/// summation order.
pub struct SlotDelta {
    terms: Vec<(Arc<LoraPair>, f32)>,
}

impl SlotDelta {
    /// A delta from explicit pairs and effective scales (tests, and loaders
    /// that build a set by hand).
    pub fn from_terms(terms: Vec<(Arc<LoraPair>, f32)>) -> Self {
        Self { terms }
    }

    /// `y + Σ s·B(A·x)`, for `y` the base projection of `xs` (bias not yet
    /// added).  `xs` is `[…, in]` and `y` `[…, out]`.
    ///
    /// On the CPU the deltas are added into `y`'s own buffer, so `y` must be
    /// the projection's fresh output: not a tensor, or a clone of one, that
    /// anything else still reads.
    pub fn apply(&self, xs: &Tensor, y: Tensor) -> candle_core::Result<Tensor> {
        let host = self
            .terms
            .iter()
            .all(|(p, _)| matches!(p.data, PairData::Host { .. }));
        if host && xs.device().is_cpu() && y.device().is_cpu() {
            let (xs, y) = (xs.contiguous()?, y.contiguous()?);
            y.inplace_op2(&xs, &CpuDelta { terms: &self.terms })?;
            return Ok(y);
        }
        let x = if xs.dtype() == DType::F32 {
            xs.clone()
        } else {
            xs.to_dtype(DType::F32)?
        };
        let mut y = y;
        for (pair, scale) in &self.terms {
            let PairData::Device { a, b } = &pair.data else {
                candle_core::bail!(
                    "LoRA delta: adapter weights are on the host, activations on {:?}",
                    xs.device()
                );
            };
            // llama.cpp's order on the device: `B(A·x)`, then the scale.
            let d = b
                .forward(&a.forward(&x)?)?
                .affine(*scale as f64, 0.0)?
                .to_dtype(y.dtype())?;
            y = (y + d)?;
        }
        Ok(y)
    }
}

/// Below this many multiply-adds a kernel stage runs on the calling thread:
/// a decode step's delta is smaller than a rayon dispatch.
const PAR_MIN_WORK: usize = 1 << 16;

/// Run `f` over `0..n` in chunks, in parallel when the stage is big enough
/// (`work` multiply-adds per index) or a test pins a thread count.
fn run_rows(n: usize, work: usize, threads: Option<usize>, f: impl Fn(Range<usize>) + Send + Sync) {
    #[cfg(test)]
    if let Some(threads) = threads {
        return crate::simd::for_each_row_chunks_with_threads(threads, n, f);
    }
    let _ = threads;
    if n.saturating_mul(work) < PAR_MIN_WORK {
        f(0..n)
    } else {
        crate::simd::for_each_row_chunks(n, f)
    }
}

/// The CPU delta over borrowed input rows, added in place into the base
/// output rows.
struct CpuDelta<'a> {
    terms: &'a [(Arc<LoraPair>, f32)],
}

impl InplaceOp2 for CpuDelta<'_> {
    fn name(&self) -> &'static str {
        "lora-delta"
    }

    fn cpu_fwd(
        &self,
        ys: &mut CpuStorage,
        yl: &Layout,
        xs: &CpuStorage,
        xl: &Layout,
    ) -> candle_core::Result<()> {
        let (Some((y0, y1)), Some((x0, x1))) = (yl.contiguous_offsets(), xl.contiguous_offsets())
        else {
            candle_core::bail!("LoRA delta: non-contiguous input");
        };
        let CpuStorage::F32(ys) = ys else {
            candle_core::bail!("LoRA delta: the base output is not f32");
        };
        let x = &xs.as_slice::<f32>()?[x0..x1];
        let n_out = yl.dims().last().copied().unwrap_or(0);
        let n_in = xl.dims().last().copied().unwrap_or(0);
        delta_rows(self.terms, x, &mut ys[y0..y1], n_in, n_out, None)
    }
}

/// `y[m, o] += Σₖ Σⱼ (sₖ · tₖ[m, j]) · Bₖ[o, j]` with
/// `tₖ[m, j] = Aₖ[j, :] · x[m, :]` (a SIMD dot).
///
/// Every element is one fixed sequence of operations over its own input row
/// — its `t` dots, then a multiply and an add per `(k, j)` in order — so the
/// result does not depend on how many rows are passed together or on the
/// thread count.  The second stage sweeps `Bᵀ` rows across a run of output
/// columns, which vectorizes, instead of a rank-length dot per output.
fn delta_rows(
    terms: &[(Arc<LoraPair>, f32)],
    x: &[f32],
    y: &mut [f32],
    n_in: usize,
    n_out: usize,
    threads: Option<usize>,
) -> candle_core::Result<()> {
    if n_in == 0 || n_out == 0 || x.len() % n_in != 0 || y.len() != x.len() / n_in * n_out {
        candle_core::bail!(
            "LoRA delta: {} inputs of width {n_in} for {} outputs of width {n_out}",
            x.len(),
            y.len()
        );
    }
    let m = x.len() / n_in;
    let dot = crate::simd::dot_fn();
    // Per term: `Bᵀ`, its rank, its scale and `t = A·x` for every row.
    let mut stages: Vec<(&[f32], usize, f32, Vec<f32>)> = Vec::with_capacity(terms.len());
    for (pair, scale) in terms {
        let PairData::Host { a, bt } = &pair.data else {
            candle_core::bail!("LoRA delta: adapter weights are not on the host");
        };
        if pair.n_in != n_in || pair.n_out != n_out {
            candle_core::bail!(
                "LoRA delta: pair [{}, {}] on a [{n_out}, {n_in}] projection",
                pair.n_out,
                pair.n_in
            );
        }
        let r = pair.rank;
        let mut t = vec![0f32; m * r];
        let dst = crate::simd::DstPtr::new(&mut t);
        run_rows(m * r, n_in, threads, |rows| {
            for idx in rows {
                let (row, j) = (idx / r, idx % r);
                let v = dot(
                    &a[j * n_in..(j + 1) * n_in],
                    &x[row * n_in..(row + 1) * n_in],
                );
                // SAFETY: `idx < m * r` and each index is visited once.
                unsafe { dst.write(idx, v) };
            }
        });
        stages.push((bt.as_slice(), r, *scale, t));
    }
    let dst = crate::simd::DstPtr::new(y);
    let rank_sum: usize = stages.iter().map(|s| s.1).sum();
    run_rows(n_out, m * rank_sum, threads, |outs| {
        for row in 0..m {
            // SAFETY: the columns `outs` of every row belong to this chunk
            // alone, and the run lies inside row `row`.
            let cols = unsafe { dst.slice_mut(row * n_out + outs.start, outs.len()) };
            for (bt, r, scale, t) in &stages {
                for (j, &tj) in t[row * r..(row + 1) * r].iter().enumerate() {
                    let c = scale * tj;
                    let b = &bt[j * n_out + outs.start..j * n_out + outs.end];
                    for (d, &bv) in cols.iter_mut().zip(b) {
                        *d += c * bv;
                    }
                }
            }
        }
    });
    Ok(())
}

#[cfg(test)]
pub(crate) mod testing {
    //! Deterministic values and an f64 reference for the delta, shared by the
    //! kernel tests here and the loaders' hook tests.
    use super::*;

    /// `n` deterministic values in `[-amp, amp)` (xorshift32).
    pub fn values(n: usize, seed: u32, amp: f32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state % 2000) as f32 / 1000.0 - 1.0) * amp
            })
            .collect()
    }

    /// `y + Σ s·B(A·x)` in f64, row by row, for pairs held on the host.
    pub fn reference(
        terms: &[(Arc<LoraPair>, f32)],
        x: &[f32],
        y: &[f32],
        n_in: usize,
        n_out: usize,
    ) -> Vec<f64> {
        let m = x.len() / n_in;
        let mut out: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        for (pair, s) in terms {
            let PairData::Host { a, bt } = &pair.data else {
                panic!("host pair")
            };
            let r = pair.rank;
            for row in 0..m {
                let t: Vec<f64> = (0..r)
                    .map(|j| {
                        (0..n_in)
                            .map(|i| a[j * n_in + i] as f64 * x[row * n_in + i] as f64)
                            .sum()
                    })
                    .collect();
                for o in 0..n_out {
                    let d: f64 = (0..r).map(|j| bt[j * n_out + o] as f64 * t[j]).sum();
                    out[row * n_out + o] += *s as f64 * d;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::testing::reference;
    use super::*;

    fn values(n: usize, seed: u32) -> Vec<f32> {
        super::testing::values(n, seed, 0.5)
    }

    fn pair(rank: usize, n_in: usize, n_out: usize, seed: u32) -> Arc<LoraPair> {
        Arc::new(
            LoraPair::new(
                values(rank * n_in, seed),
                values(n_out * rank, seed + 1),
                rank,
                n_in,
                n_out,
                &Device::Cpu,
            )
            .unwrap(),
        )
    }

    /// [`delta_rows`] added into a copy of `y`.
    fn added(
        terms: &[(Arc<LoraPair>, f32)],
        x: &[f32],
        y: &[f32],
        n_in: usize,
        n_out: usize,
        threads: Option<usize>,
    ) -> candle_core::Result<Vec<f32>> {
        let mut out = y.to_vec();
        delta_rows(terms, x, &mut out, n_in, n_out, threads)?;
        Ok(out)
    }

    #[test]
    fn delta_matches_f64_reference() {
        let (n_in, n_out, m) = (40, 24, 5);
        let terms = vec![
            (pair(4, n_in, n_out, 3), 0.75f32),
            (pair(8, n_in, n_out, 9), -1.5),
        ];
        let x = values(m * n_in, 21);
        let y = values(m * n_out, 22);
        let got = added(&terms, &x, &y, n_in, n_out, None).unwrap();
        let want = reference(&terms, &x, &y, n_in, n_out);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((*g as f64 - w).abs() < 1e-5, "element {i}: {g} vs {w}");
        }
    }

    /// One row at a time, all rows at once, and any thread count give the
    /// same bits: decode, prefill and verification agree exactly.
    #[test]
    fn delta_rows_are_independent_of_batch_and_threads() {
        let (n_in, n_out, m) = (64, 48, 9);
        let terms = vec![
            (pair(16, n_in, n_out, 5), 0.5f32),
            (pair(2, n_in, n_out, 7), 2.0),
        ];
        let x = values(m * n_in, 31);
        let y = values(m * n_out, 32);
        let all = added(&terms, &x, &y, n_in, n_out, Some(1)).unwrap();
        for threads in [2, 3, 8] {
            let par = added(&terms, &x, &y, n_in, n_out, Some(threads)).unwrap();
            assert_eq!(par, all, "{threads} threads");
        }
        assert_eq!(
            added(&terms, &x, &y, n_in, n_out, None).unwrap(),
            all,
            "global pool"
        );
        for row in 0..m {
            let one = added(
                &terms,
                &x[row * n_in..(row + 1) * n_in],
                &y[row * n_out..(row + 1) * n_out],
                n_in,
                n_out,
                Some(4),
            )
            .unwrap();
            assert_eq!(one, all[row * n_out..(row + 1) * n_out], "row {row}");
        }
    }

    /// The accelerator path (two matmuls, then scale and add) agrees with the
    /// CPU kernel, for 2-D and 3-D activations and for rows narrowed out of
    /// a larger buffer (contiguous at an offset, where the CPU kernel adds
    /// in place).
    #[test]
    fn tensor_path_matches_cpu_kernel() {
        let (n_in, n_out, m) = (32, 20, 3);
        let mk = |tensors: bool| {
            let (a, b) = (values(6 * n_in, 41), values(n_out * 6, 42));
            let p = if tensors {
                LoraPair::new_tensors(a, b, 6, n_in, n_out).unwrap()
            } else {
                LoraPair::new(a, b, 6, n_in, n_out, &Device::Cpu).unwrap()
            };
            SlotDelta::from_terms(vec![(Arc::new(p), 1.25)])
        };
        let (cpu, dev) = (mk(false), mk(true));
        for (shape, skip) in [
            (vec![m, n_in], 0),
            (vec![1, m, n_in], 0),
            (vec![m, n_in], 2),
        ] {
            let d = shape.len() - 2;
            // `skip` leading rows of a larger buffer, narrowed away.
            let rows = |width: usize, seed: u32| {
                let mut full = shape.clone();
                full[d] += skip;
                *full.last_mut().unwrap() = width;
                let n = full.iter().product();
                Tensor::from_vec(values(n, seed), full.as_slice(), &Device::Cpu)
                    .unwrap()
                    .narrow(d, skip, m)
                    .unwrap()
            };
            let x = rows(n_in, 43);
            // Each path gets its own base output: the CPU kernel adds into it.
            let a = cpu.apply(&x, rows(n_out, 44)).unwrap();
            let b = dev.apply(&x, rows(n_out, 44)).unwrap();
            assert_eq!(a.dims(), b.dims());
            let (a, b): (Vec<f32>, Vec<f32>) = (
                a.flatten_all().unwrap().to_vec1().unwrap(),
                b.flatten_all().unwrap().to_vec1().unwrap(),
            );
            for (i, (p, q)) in a.iter().zip(&b).enumerate() {
                assert!((p - q).abs() < 1e-5, "{shape:?} element {i}: {p} vs {q}");
            }
        }
    }

    #[test]
    fn delta_rejects_mismatched_dims() {
        let terms = vec![(pair(2, 8, 4, 1), 1.0f32)];
        assert!(added(&terms, &values(9, 1), &values(4, 2), 9, 4, None).is_err());
        assert!(added(&terms, &values(8, 1), &values(5, 2), 8, 5, None).is_err());
        assert!(added(&terms, &values(7, 1), &values(4, 2), 8, 4, None).is_err());
    }

    #[test]
    fn adapter_key_is_canonical() {
        let (a, b) = ([1u8; 32], [2u8; 32]);
        let k = AdapterKey::of(&[(a, 1.0), (b, 0.5)]).unwrap();
        assert_eq!(
            AdapterKey::of(&[(b, 0.5), (a, 1.0)]),
            Some(k),
            "order-independent"
        );
        assert_eq!(
            AdapterKey::of(&[(b, 0.5), (a, 1.0), (b, 0.0)]),
            Some(k),
            "scale 0 drops"
        );
        assert_ne!(
            AdapterKey::of(&[(a, 1.0), (b, 0.25)]),
            Some(k),
            "scale is part of the key"
        );
        assert_ne!(AdapterKey::of(&[(a, 1.0)]), Some(k));
        assert_eq!(AdapterKey::of(&[]), None);
        assert_eq!(
            AdapterKey::of(&[(a, 0.0), (b, -0.0)]),
            None,
            "all zero is the base model"
        );
    }

    fn table() -> SlotTable {
        let mut t = SlotTable::new("qwen3", 2, Device::Cpu);
        t.insert(SlotKey::new(0, Proj::AttnQ), 16, 8);
        t.insert(SlotKey::new(1, Proj::AttnQ), 16, 8);
        t.insert(SlotKey::new(1, Proj::FfnDownShexp), 8, 4);
        t.mark_recurrent(0);
        t
    }

    #[test]
    fn targets_are_classified() {
        let t = table();
        assert_eq!(
            t.target("blk.1.attn_q.weight").unwrap(),
            (SlotKey::new(1, Proj::AttnQ), 16, 8)
        );
        assert_eq!(
            t.target("blk.1.ffn_down_shexp.weight").unwrap().0,
            SlotKey::new(1, Proj::FfnDownShexp)
        );
        for (name, why) in [
            ("token_embd.weight", "token embedding"),
            ("output.weight", "output head"),
            ("blk.1.ffn_up_exps.weight", "routed experts"),
            ("blk.1.ffn_gate_inp.weight", "router"),
            ("blk.1.ffn_gate_inp_shexp.weight", "router"),
            ("blk.0.attn_qkv.weight", "Gated DeltaNet"),
            ("blk.0.ssm_out.weight", "Gated DeltaNet"),
            ("blk.0.attn_gate.weight", "Gated DeltaNet"),
            ("blk.2.attn_q.weight", "past the model"),
            ("blk.1.attn_k.weight", "does not exist in base model"),
            ("blk.1.indexer.q_proj.weight", "not a projection"),
            ("rope_freqs.weight", "not a projection"),
        ] {
            let err = t.target(name).unwrap_err();
            assert!(err.contains(why), "{name}: {err}");
        }
    }

    #[test]
    fn fingerprint_tracks_slots() {
        let a = table();
        let mut b = table();
        assert_eq!(a.fingerprint(), b.fingerprint());
        b.insert(SlotKey::new(1, Proj::AttnQ), 16, 9);
        assert_ne!(a.fingerprint(), b.fingerprint());
        // The interleaved-RoPE layout a PEFT adapter is permuted for.
        let mut c = table();
        c.set_rope_permute(2, 2);
        assert_ne!(a.fingerprint(), c.fingerprint());
        let mut d = table();
        d.set_rope_permute(4, 2);
        assert_ne!(c.fingerprint(), d.fingerprint());
    }

    #[test]
    fn proj_names_round_trip() {
        for (i, p) in Proj::ALL.into_iter().enumerate() {
            assert_eq!(p.index(), i);
            assert_eq!(Proj::from_gguf_name(p.gguf_name()), Some(p));
        }
        assert_eq!(Proj::from_gguf_name("ffn_gate_exps"), None);
    }
}
