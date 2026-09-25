//! Attention building blocks shared by Joshua's native loaders: the RoPE
//! sin/cos table, GQA head repetition, and the cached scaled-dot-product
//! kernel over a transposed KV cache.
//!
//! Every loader used to build its own rotary table from the same inverse
//! frequencies and carry its own `repeat_kv`; the differences that remain
//! (pairing style, partial rotation, M-RoPE sections, YaRN) are parameters
//! of [`Rope`].

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::ops::softmax_last_dim;

/// How rotated dimensions are paired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// llama.cpp `NEOX` (and `MROPE` / `IMROPE`): dimension `i` pairs with
    /// `i + n_rot/2` — candle's `rotary_emb::rope`.
    Neox,
    /// llama.cpp `NORM`: adjacent dimensions `(2i, 2i+1)` pair — candle's
    /// `rotary_emb::rope_i`.
    Interleaved,
}

/// Standard RoPE inverse frequencies `1 / theta^(2i / n_rot)` for an
/// `n_rot`-wide rotary slice (`n_rot / 2` entries).
pub fn inv_freq(n_rot: usize, theta: f32) -> Vec<f32> {
    (0..n_rot)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f32 / n_rot as f32))
        .collect()
}

/// Zero the inverse frequencies that llama.cpp's multi-section RoPE
/// (`MROPE` for Qwen2-VL, interleaved `IMROPE` for Qwen3-VL / Qwen3.5) drives
/// from the 4th ("extra") position component.
///
/// For text tokens llama.cpp feeds positions `[p, p, p, 0]`: the temporal,
/// height and width sections all see `p`, so they rotate exactly like 1-D
/// RoPE, while any frequency assigned to the extra section sees position 0
/// and is not rotated at all.  Zeroing those frequencies reproduces that
/// with an ordinary table.  Sections index frequency pairs (`n_rot / 2` of
/// them) and wrap every `sum(sections)` pairs, as in `ggml_mrope_cache_init`.
pub fn mrope_text_mask(inv_freq: &mut [f32], sections: [usize; 4], interleaved: bool) {
    let total: usize = sections.iter().sum();
    if total == 0 {
        return;
    }
    let [t, h, w, _] = sections;
    for (i, f) in inv_freq.iter_mut().enumerate() {
        let sector = i % total;
        let extra = if interleaved {
            // Sectors cycle t, h, w; each section claims its first
            // `3 * len` sectors of that residue.
            let len = match sector % 3 {
                0 => t,
                1 => h,
                _ => w,
            };
            sector >= 3 * len
        } else {
            sector >= t + h + w
        };
        if extra {
            *f = 0.0;
        }
    }
}

/// A precomputed RoPE table over positions `[0, max_seq)` for a rotary
/// slice of `2 * inv_freq.len()` dimensions.
pub struct Rope {
    sin: Tensor, // [max_seq, n_rot / 2]
    cos: Tensor,
    style: RopeStyle,
}

impl Rope {
    /// Build the table from inverse frequencies; `mscale` scales both sin
    /// and cos (YaRN attention temperature, 1 otherwise).
    pub fn from_inv_freq(
        inv_freq: Vec<f32>,
        max_seq: usize,
        mscale: f32,
        style: RopeStyle,
        dev: &Device,
    ) -> Result<Self> {
        let max_seq = max_seq.max(1);
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), dev)?;
        let t = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let (sin, cos) = if mscale == 1.0 {
            (freqs.sin()?, freqs.cos()?)
        } else {
            (
                (freqs.sin()? * mscale as f64)?,
                (freqs.cos()? * mscale as f64)?,
            )
        };
        Ok(Self { sin, cos, style })
    }

    /// Plain (unscaled) RoPE over an `n_rot`-wide slice.
    pub fn new(
        n_rot: usize,
        theta: f32,
        max_seq: usize,
        style: RopeStyle,
        dev: &Device,
    ) -> Result<Self> {
        Self::from_inv_freq(inv_freq(n_rot, theta), max_seq, 1.0, style, dev)
    }

    /// Width of the rotated slice.
    pub fn n_rot(&self) -> Result<usize> {
        Ok(self.sin.dim(1)? * 2)
    }

    fn rotate(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = x.contiguous()?;
        match self.style {
            RopeStyle::Neox => candle_nn::rotary_emb::rope(&x, cos, sin),
            RopeStyle::Interleaved => candle_nn::rotary_emb::rope_i(&x, cos, sin),
        }
    }

    /// Rotate `x` (`[b, heads, seq, n_rot]`) for positions starting at
    /// `offset`.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let seq = x.dim(2)?;
        let sin = self.sin.narrow(0, offset, seq)?;
        let cos = self.cos.narrow(0, offset, seq)?;
        self.rotate(x, &cos, &sin)
    }

    /// The inverse rotation of [`Self::apply`] (`ggml_rope_ext_back`).
    pub fn apply_inverse(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let seq = x.dim(2)?;
        let sin = (self.sin.narrow(0, offset, seq)? * -1.0)?;
        let cos = self.cos.narrow(0, offset, seq)?;
        self.rotate(x, &cos, &sin)
    }

    /// Rotate `q` and `k` (possibly with different head counts) together.
    pub fn apply_pair(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        Ok((self.apply(q, offset)?, self.apply(k, offset)?))
    }

    /// Rotate only the *leading* `n_rot` dims of `x`'s last axis, passing the
    /// rest through (llama.cpp partial rotary: `n_rot < head_dim`).
    pub fn apply_leading(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let d = x.dim(D::Minus1)?;
        let n_rot = self.n_rot()?;
        if n_rot >= d {
            return self.apply(x, offset);
        }
        let rot = self.apply(&x.narrow(D::Minus1, 0, n_rot)?, offset)?;
        let pass = x.narrow(D::Minus1, n_rot, d - n_rot)?.contiguous()?;
        Tensor::cat(&[&rot, &pass], D::Minus1)
    }

    /// Rotate rows of `x` (`[n, d]`, `d >= n_rot`) at explicit per-row
    /// `positions` (`[n]` u32); like [`Self::apply_leading`], only the
    /// leading `n_rot` dims turn.
    pub fn apply_at(&self, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let (n, d) = x.dims2()?;
        let n_rot = self.n_rot()?;
        let rot = x.narrow(1, 0, n_rot)?.contiguous()?.reshape((n, 1, 1, n_rot))?;
        // rope / rope_i accept 3-D cos/sin as [b, t, d], one row per batch item.
        let cos = self.cos.index_select(positions, 0)?.unsqueeze(1)?;
        let sin = self.sin.index_select(positions, 0)?.unsqueeze(1)?;
        let rot = self.rotate(&rot, &cos, &sin)?.reshape((n, n_rot))?;
        if n_rot >= d {
            return Ok(rot);
        }
        Tensor::cat(&[&rot, &x.narrow(1, n_rot, d - n_rot)?.contiguous()?], 1)
    }
}

/// Repeat KV heads `n_rep` times along the head axis (GQA): `[b, n_kv, s, d]`
/// → `[b, n_kv * n_rep, s, d]`, query head `h` reading KV head `h / n_rep`.
pub fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv, seq, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, seq, hd))?
        .reshape((b, n_kv * n_rep, seq, hd))
}

/// One layer's KV cache, kept transposed — `(k [b, n_kv, dk, seq], v [b,
/// n_kv, dv, seq])` — and appended along dim 3 ([`KV_SEQ_DIM`]).
pub type KvCache = crate::moe::KvCache;

/// The sequence axis of a [`KvCache`] (for [`crate::moe::truncate_kv`]).
pub const KV_SEQ_DIM: usize = 3;

/// Append this step's keys/values to `kv_cache` and attend `q` over the whole
/// cached context.
///
/// * `q`: `[b, n_head, seq, dk]`
/// * `k_new`: `[b, n_kv, dk, seq]`, `v_new`: `[b, n_kv, dv, seq]` (transposed,
///   matching the cache layout)
///
/// `n_head` must be a multiple of `n_kv`; query head `h` reads KV head
/// `h / (n_head / n_kv)`.  Returns the attention context
/// `[b, seq, n_head * dv]`, ready for the output projection.
///
/// The cache lives transposed so the matmuls run directly on it: `q · kᵀ`
/// becomes `q · k_cache` and `probs · v` becomes `v_cache · probsᵀ`.  A decode
/// step then copies only the two append cats (the accumulated context), never
/// an extra transpose of it.  Grouped query heads are folded into the row
/// dimension of their KV head instead of repeating K/V per query head.
pub fn cached_attention(
    kv_cache: &mut KvCache,
    q: &Tensor,
    k_new: Tensor,
    v_new: Tensor,
    mask: Option<&Tensor>,
    softmax_scale: f64,
) -> Result<Tensor> {
    let (b, n_head, seq_len, dk) = q.dims4()?;
    let n_kv = k_new.dim(1)?;
    let dv = v_new.dim(2)?;
    let rep = n_head / n_kv;

    let (k_cache, v_cache) = match &*kv_cache {
        None => (k_new, v_new),
        Some((kc, vc)) => (
            Tensor::cat(&[kc, &k_new], KV_SEQ_DIM)?.contiguous()?,
            Tensor::cat(&[vc, &v_new], KV_SEQ_DIM)?.contiguous()?,
        ),
    };
    *kv_cache = Some((k_cache.clone(), v_cache.clone()));
    let seq_total = k_cache.dim(KV_SEQ_DIM)?;

    // Scaled dot-product attention over the whole cached context.
    let q = q.contiguous()?.reshape((b, n_kv, rep * seq_len, dk))?;
    let scores = (q.matmul(&k_cache)? * softmax_scale)?.reshape((b, n_head, seq_len, seq_total))?;
    let scores = match mask {
        Some(m) => scores.broadcast_add(m)?,
        None => scores,
    };
    let probs = softmax_last_dim(&scores)?.reshape((b, n_kv, rep * seq_len, seq_total))?;
    let ctx = v_cache.matmul(&probs.transpose(2, 3)?.contiguous()?)?; // [b, n_kv, dv, rep*seq]
    ctx.reshape((b, n_kv, dv, rep, seq_len))?
        .permute((0, 4, 1, 3, 2))? // [b, seq, n_kv, rep, dv]
        .contiguous()?
        .reshape((b, seq_len, n_head * dv))
}

/// [`cached_attention`] for keys/values in the usual `[b, n_kv, seq, d]`
/// head layout: transposes them into the cache layout first.
pub fn cached_attention_heads(
    kv_cache: &mut KvCache,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    softmax_scale: f64,
) -> Result<Tensor> {
    cached_attention(
        kv_cache,
        q,
        k.transpose(2, 3)?.contiguous()?,
        v.transpose(2, 3)?.contiguous()?,
        mask,
        softmax_scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grouped attention must equal the textbook form: K/V repeated per query
    /// head, softmax(q·kᵀ·scale + mask)·v — over both a prefill and a decode
    /// step appended to the cache.
    #[test]
    fn gqa_cached_attention_matches_repeated_kv_reference() -> Result<()> {
        let dev = Device::Cpu;
        let (b, n_head, n_kv, d, dv) = (1usize, 4usize, 2usize, 6usize, 5usize);
        let rep = n_head / n_kv;
        let scale = 0.37;

        let reference = |q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor| -> Result<Tensor> {
            let (k, v) = (repeat_kv(k.clone(), rep)?, repeat_kv(v.clone(), rep)?);
            let scores = (q.matmul(&k.t()?)? * scale)?.broadcast_add(mask)?;
            let ctx = softmax_last_dim(&scores)?.matmul(&v)?; // [b, h, s, dv]
            let s = ctx.dim(2)?;
            ctx.transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, n_head * dv))
        };

        let total = 4usize;
        let q = Tensor::randn(0f32, 1f32, (b, n_head, total, d), &dev)?;
        let k = Tensor::randn(0f32, 1f32, (b, n_kv, total, d), &dev)?;
        let v = Tensor::randn(0f32, 1f32, (b, n_kv, total, dv), &dev)?;

        let mut kv: KvCache = None;
        // Prefill 3 tokens, then decode the 4th.
        let mask = crate::moe::causal_mask(3, 0, &dev)?;
        let pre = cached_attention_heads(
            &mut kv,
            &q.narrow(2, 0, 3)?,
            &k.narrow(2, 0, 3)?,
            &v.narrow(2, 0, 3)?,
            Some(&mask),
            scale,
        )?;
        let dec = cached_attention_heads(
            &mut kv,
            &q.narrow(2, 3, 1)?,
            &k.narrow(2, 3, 1)?,
            &v.narrow(2, 3, 1)?,
            None,
            scale,
        )?;
        let got = Tensor::cat(&[&pre, &dec], 1)?;
        let want = reference(&q, &k, &v, &crate::moe::causal_mask(total, 0, &dev)?)?;

        let max = (got - want)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            max < 1e-5,
            "grouped attention diverges from reference: {max}"
        );
        let (kc, vc) = kv.as_ref().unwrap();
        assert_eq!(
            kc.dims(),
            &[b, n_kv, d, total],
            "cache keeps the un-repeated KV heads"
        );
        assert_eq!(vc.dims(), &[b, n_kv, dv, total]);
        Ok(())
    }

    /// Qwen3-VL sections `[24, 20, 20, 0]` over 64 frequency pairs: the
    /// interleaved layout leaves pairs 61 and 62 to the (zero) extra
    /// position, and nothing else.
    #[test]
    fn imrope_text_positions_only_zero_the_extra_sector() {
        let mut f = vec![1.0f32; 64];
        mrope_text_mask(&mut f, [24, 20, 20, 0], true);
        let zeroed: Vec<usize> = (0..64).filter(|&i| f[i] == 0.0).collect();
        assert_eq!(zeroed, vec![61, 62]);

        // Plain M-RoPE covering every pair is exactly 1-D RoPE.
        let mut g = vec![1.0f32; 64];
        mrope_text_mask(&mut g, [16, 24, 24, 0], false);
        assert!(g.iter().all(|&x| x == 1.0));
        // …and a short section list leaves the tail to the extra position.
        let mut g = vec![1.0f32; 8];
        mrope_text_mask(&mut g, [2, 1, 1, 0], false);
        assert_eq!(g, vec![1.0; 8], "sections wrap every sum(sections) pairs");
        let mut g = vec![1.0f32; 8];
        mrope_text_mask(&mut g, [2, 1, 1, 2], false);
        assert_eq!(g, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    }

    /// Partial rotation leaves the trailing dims untouched and rotates the
    /// leading slice exactly like a full-width table over it.
    #[test]
    fn partial_rope_rotates_only_the_leading_slice() -> Result<()> {
        let dev = Device::Cpu;
        let rope = Rope::new(4, 10_000.0, 16, RopeStyle::Neox, &dev)?;
        let x = Tensor::randn(0f32, 1f32, (1, 2, 3, 8), &dev)?;
        let y = rope.apply_leading(&x, 5)?;
        let tail_diff = (y.narrow(3, 4, 4)? - x.narrow(3, 4, 4)?)?
            .abs()?
            .flatten_all()?
            .max(0)?;
        assert_eq!(tail_diff.to_scalar::<f32>()?, 0.0);
        let head = rope.apply(&x.narrow(3, 0, 4)?.contiguous()?, 5)?;
        let head_diff = (y.narrow(3, 0, 4)? - head)?.abs()?.flatten_all()?.max(0)?;
        assert_eq!(head_diff.to_scalar::<f32>()?, 0.0);
        Ok(())
    }
}
