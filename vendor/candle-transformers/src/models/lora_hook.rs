//! A hook through which a caller adds LoRA deltas to the dense projections
//! of the quantized loaders that take one (`quantized_llama`,
//! `quantized_qwen2`, `quantized_gemma3`, `quantized_phi3`).
//!
//! A loader holds an optional [`LinearDelta`] and, after each hookable
//! projection `y = W·x` of decoder layer `layer`, hands `x` and `y` to it
//! under the projection's GGUF name (`attn_q`, `ffn_down`, …), as llama.cpp's
//! `build_lora_mm` adds `B·A·x` after the base matmul.  Without a hook the
//! forward pass is exactly the base one.  [`LoraSlot`]s describe which
//! projections a loaded model has, with their `[out, in]` dims, so the
//! caller can validate an adapter before installing it.

use candle::{Result, Tensor};

/// What a caller adds after a hookable projection (see the module docs).
pub trait LinearDelta: Send + Sync + std::fmt::Debug {
    /// `y` plus this hook's delta for `layer`'s `proj` given its input `xs`.
    /// `y` is the projection's fresh output (bias not yet added).
    fn apply(&self, layer: usize, proj: &'static str, xs: &Tensor, y: Tensor) -> Result<Tensor>;

    /// For callers that need their own type back.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// One hookable projection of a loaded model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoraSlot {
    pub layer: usize,
    /// The GGUF name of the projection (`attn_q`, `ffn_up`, …).
    pub proj: &'static str,
    pub n_out: usize,
    pub n_in: usize,
}

/// A layer's view of the hook: applies it, when there is one, at `layer`.
#[derive(Clone, Copy)]
pub(crate) struct At<'a> {
    pub hook: Option<&'a dyn LinearDelta>,
    pub layer: usize,
}

impl At<'_> {
    pub(crate) fn none() -> Self {
        Self { hook: None, layer: 0 }
    }

    /// `y` plus the hook's delta for `proj`.
    pub(crate) fn mm(&self, proj: &'static str, xs: &Tensor, y: Tensor) -> Result<Tensor> {
        match self.hook {
            Some(h) => h.apply(self.layer, proj, xs, y),
            None => Ok(y),
        }
    }
}

/// A loaded model's hookable projections.
#[derive(Debug, Clone)]
pub struct LoraSlots {
    /// The model's GGUF `general.architecture`.
    pub arch: String,
    pub n_layer: usize,
    /// The device the weights are on (`None` when there are no slots).
    pub device: Option<candle::Device>,
    pub slots: Vec<LoraSlot>,
}

impl LoraSlots {
    pub(crate) fn new(arch: impl Into<String>, n_layer: usize) -> Self {
        Self {
            arch: arch.into(),
            n_layer,
            device: None,
            slots: Vec::new(),
        }
    }

    /// Add `w` as `layer`'s `proj`, when its dims are known.
    pub(crate) fn push(&mut self, layer: usize, proj: &'static str, w: &candle::quantized::QMatMul) {
        use candle::quantized::QMatMul;
        let (dims, device) = match w {
            QMatMul::QTensor(q) => (q.shape().dims2().ok(), q.device()),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => (t.dims2().ok(), t.device().clone()),
        };
        if let Some((n_out, n_in)) = dims {
            self.device.get_or_insert(device);
            self.slots.push(LoraSlot { layer, proj, n_out, n_in });
        }
    }
}
