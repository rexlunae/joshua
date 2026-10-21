//! INT8 KV-cache prototype.
//!
//! This module is deliberately independent of [`crate::attention::SeqKvCache`].
//! The production cache stores f32 device tensors and feeds them directly to
//! the attention matmuls.  Replacing that storage with int8 without an int8
//! attention kernel would require dequantizing the whole cache on every
//! decode step, which adds a full-context copy and synchronization and is
//! slower than the memory it saves.
//!
//! The prototype keeps the quantized rows and their per-row scales as device
//! tensors so a future loader or kernel can opt in without changing the
//! quantization format.  `dequantize_*` is provided for parity checks and
//! for the slow fallback path.
//!
//! Layout follows the inputs to the generic attention path:
//! `[batch, heads, seq, dim]`.  `SeqKvCache` stores the same rows as
//! `[batch, heads, dim, cap]` so the attention matmuls can consume transposed
//! views; a future integration would either match that layout or transpose
//! once when appending.
//!
//! Quantization is symmetric int8 with one scale per `(batch, head, token)`
//! row; zero rows use a tiny positive scale so the scale itself is always
//! invertible.

use candle_core::{DType, Device, Result, Tensor, D};

/// A KV cache whose rows are stored as unsigned int8 plus f32 scales.
///
/// The unsigned representation uses a zero point of 128, so `-127..=127`
/// maps to `1..=255` and `0.0` maps to 128.
#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    /// Quantized keys, shape `[batch, heads, seq, dim]`, dtype `U8`.
    k: Tensor,
    /// Quantized values, same shape and dtype as `k`.
    v: Tensor,
    /// Scale for each row of `k`, shape `[batch, heads, seq, 1]`.
    k_scale: Tensor,
    /// Scale for each row of `v`, shape `[batch, heads, seq, 1]`.
    v_scale: Tensor,
    /// Number of cached positions.
    len: usize,
}

impl QuantizedKvCache {
    /// Quantize an initial `[batch, heads, seq, dim]` key/value pair.
    pub fn from_tensors(k: &Tensor, v: &Tensor) -> Result<Self> {
        validate_pair(k, v)?;
        let (k, k_scale) = quantize(k)?;
        let (v, v_scale) = quantize(v)?;
        let len = k.dims()[2];
        Ok(Self {
            k,
            v,
            k_scale,
            v_scale,
            len,
        })
    }

    /// Quantize and append new rows along the sequence dimension.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        validate_pair(k, v)?;
        if k.dims()[0] != self.k.dims()[0]
            || k.dims()[1] != self.k.dims()[1]
            || k.dims()[3] != self.k.dims()[3]
        {
            candle_core::bail!(
                "KV append shape {:?} does not match cache {:?}",
                k.dims(),
                self.k.dims()
            );
        }
        let (k, k_scale) = quantize(k)?;
        let (v, v_scale) = quantize(v)?;
        self.k = Tensor::cat(&[&self.k, &k], 2)?;
        self.v = Tensor::cat(&[&self.v, &v], 2)?;
        self.k_scale = Tensor::cat(&[&self.k_scale, &k_scale], 2)?;
        self.v_scale = Tensor::cat(&[&self.v_scale, &v_scale], 2)?;
        self.len += k.dims()[2];
        Ok(())
    }

    /// Cached positions.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no positions are cached.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Dequantize keys to `[batch, heads, seq, dim]` f32.
    pub fn dequantize_k(&self) -> Result<Tensor> {
        dequantize(&self.k, &self.k_scale)
    }

    /// Dequantize values to `[batch, heads, seq, dim]` f32.
    pub fn dequantize_v(&self) -> Result<Tensor> {
        dequantize(&self.v, &self.v_scale)
    }

    /// The quantized key tensor, dtype `U8`.
    pub fn quantized_k(&self) -> &Tensor {
        &self.k
    }

    /// The quantized value tensor, dtype `U8`.
    pub fn quantized_v(&self) -> &Tensor {
        &self.v
    }

    /// Key scales, shape `[batch, heads, seq, 1]`.
    pub fn k_scale(&self) -> &Tensor {
        &self.k_scale
    }

    /// Value scales, shape `[batch, heads, seq, 1]`.
    pub fn v_scale(&self) -> &Tensor {
        &self.v_scale
    }

    /// Bytes used by the quantized rows and scales.
    pub fn memory_bytes(&self) -> u64 {
        self.k.elem_count() as u64
            + self.v.elem_count() as u64
            + (self.k_scale.elem_count() + self.v_scale.elem_count()) as u64 * 4
    }

    /// Bytes an f32 cache with the same shape would use.
    pub fn f32_bytes(&self) -> u64 {
        (self.k.elem_count() + self.v.elem_count()) as u64 * 4
    }

    /// Device holding the cache.
    pub fn device(&self) -> &Device {
        self.k.device()
    }
}

fn validate_pair(k: &Tensor, v: &Tensor) -> Result<()> {
    if k.rank() != 4 || v.rank() != 4 {
        candle_core::bail!(
            "KV tensors must be rank 4, got {:?} and {:?}",
            k.dims(),
            v.dims()
        );
    }
    if k.dims() != v.dims() {
        candle_core::bail!("KV shapes differ: {:?} vs {:?}", k.dims(), v.dims());
    }
    if k.dtype() != DType::F32 || v.dtype() != DType::F32 {
        candle_core::bail!(
            "KV quantization expects f32 tensors, got {:?} and {:?}",
            k.dtype(),
            v.dtype()
        );
    }
    Ok(())
}

/// Symmetric int8 quantization with a per-row scale.
fn quantize(t: &Tensor) -> Result<(Tensor, Tensor)> {
    let max_abs = t.abs()?.max_keepdim(D::Minus1)?;
    // A zero row has no useful scale; a tiny epsilon keeps the division
    // finite while representing every value as exactly zero.
    let scale = (max_abs / 127.0)?.clamp(1e-12, f32::MAX)?;
    // Expand the scale view before the arithmetic: candle's CPU broadcast
    // kernels currently produce a layout that is smaller than the output
    // shape when the right-hand side has a stride-0 dimension.
    let scale_full = scale.broadcast_as(t.shape())?.contiguous()?;
    let q = (t / &scale_full)?.round()?.clamp(-127.0, 127.0)?;
    let q = (q + 128.0)?.to_dtype(DType::U8)?;
    Ok((q, scale))
}

fn dequantize(q: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let centered = (q.to_dtype(DType::F32)? - 128.0)?;
    let scale = scale.broadcast_as(centered.shape())?.contiguous()?;
    centered.mul(&scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn sample(seq: usize) -> Tensor {
        let dim = 8;
        let values: Vec<f32> = (0..2 * seq * dim)
            .map(|i| {
                let x = i as f32 - 7.0;
                x * 0.37 + if i % 3 == 0 { -2.5 } else { 1.25 }
            })
            .collect();
        Tensor::from_vec(values, (1, 2, seq, dim), &Device::Cpu).unwrap()
    }

    fn max_error(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn int8_round_trip_stays_within_half_a_scale() {
        let k = sample(4);
        let v = sample(4);
        let cache = QuantizedKvCache::from_tensors(&k, &v).unwrap();
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.dequantize_k().unwrap().dims(), k.dims());
        assert_eq!(cache.dequantize_v().unwrap().dims(), v.dims());

        // Symmetric rounding to 127 levels gives at most half a step of
        // error per element, plus f32 rounding.
        let scale = cache
            .k_scale()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(max_error(&cache.dequantize_k().unwrap(), &k) <= scale * 0.5001);
        assert!(cache.memory_bytes() < cache.f32_bytes());
    }

    #[test]
    fn append_preserves_previous_rows_and_new_rows() {
        let full = sample(3);
        let mut cache = QuantizedKvCache::from_tensors(
            &full.narrow(2, 0, 1).unwrap(),
            &full.narrow(2, 0, 1).unwrap(),
        )
        .unwrap();
        cache
            .append(
                &full.narrow(2, 1, 2).unwrap(),
                &full.narrow(2, 1, 2).unwrap(),
            )
            .unwrap();
        assert_eq!(cache.len(), 3);
        let max_scale = cache
            .k_scale()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(max_error(&cache.dequantize_k().unwrap(), &full) <= max_scale * 0.5001);
    }

    #[test]
    fn attention_output_tracks_f32_within_quantization_noise() {
        let bounded = |len: usize| {
            Tensor::from_vec(
                (0..len)
                    .map(|i| ((i * 37) % 200) as f32 / 100.0 - 1.0)
                    .collect::<Vec<_>>(),
                (1, 2, len / 16, 8),
                &Device::Cpu,
            )
            .unwrap()
        };
        let q = bounded(48);
        let k = bounded(64);
        let v = bounded(64);
        let cache = QuantizedKvCache::from_tensors(&k, &v).unwrap();
        let dk = cache.dequantize_k().unwrap();
        let dv = cache.dequantize_v().unwrap();

        let reference = {
            let scores = q.matmul(&k.transpose(2, 3).unwrap()).unwrap();
            candle_nn::ops::softmax_last_dim(&scores)
                .unwrap()
                .matmul(&v)
                .unwrap()
        };
        let quantized = {
            let scores = q.matmul(&dk.transpose(2, 3).unwrap()).unwrap();
            candle_nn::ops::softmax_last_dim(&scores)
                .unwrap()
                .matmul(&dv)
                .unwrap()
        };
        let error = max_error(&reference, &quantized);
        assert!(error < 0.02, "int8 attention diverges by {error}");
    }

    #[test]
    fn zero_rows_use_a_finite_scale() {
        let zero = Tensor::zeros((1, 1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let cache = QuantizedKvCache::from_tensors(&zero, &zero).unwrap();
        let round_trip = cache.dequantize_k().unwrap();
        assert!(
            round_trip
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                < 1e-9
        );
    }
}
