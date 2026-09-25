//! Where a native loader's tensors come from, shared by the Qwen-family and
//! DeepSeek-2 loaders.
//!
//! A tensor is served, in order of preference, from:
//!
//! 1. the raw GGUF header, when its dtype is a format candle cannot carry
//!    (Bonsai's Q1_0 / Q2_0, MXFP4, …; see [`GgufHeader::load_raw_only`]);
//! 2. the memory mapping, borrowed in place (see [`crate::mmap_tensor`]) —
//!    what keeps a model far larger than RAM loadable;
//! 3. candle's copying reader.

use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use candle_transformers::quantized_nn::RmsNorm;

use crate::gguf_ext::GgufHeader;

pub struct TensorSource<R: Read + Seek> {
    pub ct: gguf_file::Content,
    /// The header with raw dtype ids, when the file holds tensors `ct`
    /// cannot carry.
    pub raw: Option<GgufHeader>,
    pub reader: R,
    /// Device the dense weights are built on.
    pub device: Device,
    pub mmap: Option<Arc<memmap2::Mmap>>,
    /// `general.architecture`, for error context.
    pub arch: &'static str,
}

impl<R: Read + Seek> TensorSource<R> {
    pub fn qtensor(&mut self, name: &str) -> Result<QTensor> {
        if let Some(raw) = &self.raw {
            if let Some(qt) =
                raw.load_raw_only(name, self.mmap.as_ref(), &mut self.reader, &self.device)?
            {
                return Ok(qt);
            }
        }
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

    pub fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        QMatMul::from_qtensor(self.qtensor(name)?)
    }

    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        RmsNorm::from_qtensor(self.qtensor(name)?, eps)
    }

    pub fn rms_norm_opt(&mut self, name: &str, eps: f64) -> Result<Option<RmsNorm>> {
        self.has(name).then(|| self.rms_norm(name, eps)).transpose()
    }

    pub fn f32_tensor(&mut self, name: &str) -> Result<Tensor> {
        self.qtensor(name)?
            .dequantize(&self.device)?
            .to_dtype(DType::F32)
    }

    pub fn f32_opt(&mut self, name: &str) -> Result<Option<Tensor>> {
        self.has(name).then(|| self.f32_tensor(name)).transpose()
    }

    pub fn has(&self, name: &str) -> bool {
        self.ct.tensor_infos.contains_key(name) || self.raw_dtype(name).is_some()
    }

    /// The GGUF dtype id of a tensor only the raw header carries.
    pub fn raw_dtype(&self, name: &str) -> Option<u32> {
        let info = self.raw.as_ref()?.tensors.get(name)?;
        (!crate::gguf_ext::is_candle_supported(info.dtype)).then_some(info.dtype)
    }

    /// Look up a stacked routed-expert tensor for per-expert slicing.  The
    /// expert paths slice candle-typed blocks, so a raw-format expert stack
    /// is refused with its dtype rather than reported missing.
    pub fn expert_tensor(&self, name: &str, n_expert: usize) -> Result<crate::moe::ExpertTensor> {
        if let Some(dtype) = self.raw_dtype(name) {
            candle_core::bail!(
                "{}: routed-expert tensor `{name}` has GGUF dtype {dtype}, which this loader \
                 only supports for dense weights",
                self.arch
            );
        }
        crate::moe::ExpertTensor::lookup(&self.ct, self.arch, name, n_expert)
    }
}
