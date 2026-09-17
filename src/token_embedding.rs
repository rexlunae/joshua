//! The token-embedding table, kept quantized.
//!
//! Every joshua-native loader used to dequantize `token_embd.weight` to f32
//! at load and gather rows with `index_select`.  That table is the single
//! largest *anonymous* allocation a session makes: `vocab × hidden × 4`
//! bytes — 1.2 GiB for Qwen3-30B-A3B, 3.7 GiB for DeepSeek-V3, 4.7 GiB for
//! Kimi-K2 — per model instance, on whichever device the model runs, and
//! never shared through the page cache.  On a GPU it is the difference
//! between a dense set that fits and one that does not; on the CPU it is
//! RAM the routed experts could have used.
//!
//! [`TokenEmbedding::Quantized`] keeps the table in its on-disk quantization
//! (borrowed in place from the mapping on the CPU, uploaded compressed on an
//! accelerator) and gathers rows through candle's quantized embedding kernel
//! — the dequantized rows exist only for the tokens of the current step.
//! The numbers are identical to the dequantize-then-gather path: the same
//! blocks go through the same `to_float`, just per row instead of whole.
//!
//! [`TokenEmbedding::Dense`] is the old f32 table, kept for backends whose
//! quantized storage has no gather kernel (OpenCL holds weights as dense
//! f32 already, so nothing is saved there) and for float dtypes on
//! accelerators (an f32 table is no larger dense, and the GPU quantized
//! embedding kernels are only exercised for block-quantized types).

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Result, Tensor};

/// A `[vocab, hidden]` token-embedding table.
pub enum TokenEmbedding {
    /// Quantized table, gathered per row through `QTensor::embedding`.
    Quantized(QTensor),
    /// Dequantized f32 table, gathered with `index_select`.
    Dense(Tensor),
}

impl TokenEmbedding {
    /// Choose the representation for `table` on `device`.
    ///
    /// Quantized wherever candle's quantized embedding gather serves the
    /// storage: always on the CPU (both the borrowed-mmap and heap storages
    /// implement it for every dtype), on OpenCL and Vulkan (their block
    /// storages gather rows with an on-device dequantizing kernel for every
    /// GGUF dtype), and for block-quantized dtypes on CUDA/Metal.  Dense
    /// only for float dtypes on CUDA/Metal.
    pub fn load(table: QTensor, device: &Device) -> Result<Self> {
        let float_dtype = matches!(
            table.dtype(),
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        );
        let keep_quantized = device.is_cpu() || device.is_opencl() || device.is_vulkan() || !float_dtype;
        if keep_quantized {
            Ok(Self::Quantized(table))
        } else {
            Ok(Self::Dense(table.dequantize(device)?.to_dtype(DType::F32)?))
        }
    }

    /// Hidden size (the row width).
    pub fn hidden(&self) -> Result<usize> {
        match self {
            Self::Quantized(q) => Ok(q.shape().dims2()?.1),
            Self::Dense(t) => t.dim(1),
        }
    }

    /// Gather the rows for `ids` (any shape of `u32`), returning
    /// `ids.shape() + [hidden]` in f32.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized(q) => q.embedding(ids)?.to_dtype(DType::F32),
            Self::Dense(t) => {
                let mut dims = ids.dims().to_vec();
                dims.push(t.dim(1)?);
                t.index_select(&ids.flatten_all()?, 0)?.reshape(dims)
            }
        }
    }

    /// Whether the table is held quantized (diagnostics).
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The quantized gather must reproduce the dequantize-then-index_select
    /// rows exactly, on both heap and borrowed-mmap storage, for a
    /// block-quantized dtype.
    #[test]
    fn quantized_gather_matches_dense_gather() -> Result<()> {
        let (vocab, hidden) = (10usize, 64usize);
        let data: Vec<f32> = (0..vocab * hidden)
            .map(|i| ((i * 7) % 31) as f32 / 3.0 - 5.0)
            .collect();
        let t = Tensor::from_vec(data, (vocab, hidden), &Device::Cpu)?;
        let q = QTensor::quantize(&t, GgmlDType::Q8_0)?;
        let dense = TokenEmbedding::Dense(q.dequantize(&Device::Cpu)?);
        let quant = TokenEmbedding::load(q, &Device::Cpu)?;
        assert!(quant.is_quantized(), "the CPU keeps the table quantized");
        assert_eq!(quant.hidden()?, hidden);

        let ids = Tensor::new(&[[3u32, 9, 0, 3]], &Device::Cpu)?;
        let a: Vec<f32> = quant.forward(&ids)?.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = dense.forward(&ids)?.flatten_all()?.to_vec1()?;
        assert_eq!(quant.forward(&ids)?.dims(), &[1, 4, hidden]);
        assert_eq!(
            a, b,
            "quantized gather must be bit-identical to the dense path"
        );
        Ok(())
    }

    /// Float tables stay quantized on the CPU (the generic gather covers
    /// f32/f16) — nothing is dequantized ahead of use.
    #[test]
    fn float_tables_stay_quantized_on_cpu() -> Result<()> {
        let t = Tensor::arange(0f32, 32f32, &Device::Cpu)?.reshape((4, 8))?;
        let e = TokenEmbedding::load(QTensor::quantize(&t, GgmlDType::F32)?, &Device::Cpu)?;
        assert!(e.is_quantized());
        let row: Vec<f32> = e
            .forward(&Tensor::new(&[2u32], &Device::Cpu)?)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(row, (16..24).map(|v| v as f32).collect::<Vec<_>>());
        Ok(())
    }
}
