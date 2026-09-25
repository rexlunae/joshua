//! Pieces shared by the n-gram hash embeddings that inject into a
//! hyper-connection stream: qwen4exp's PLE and DeepSeek-V4.1's engram.
//! Both gate an embedding-derived value into every stream copy with a
//! sigmoid of a signed square root of a normalized key/query dot product.

use candle_core::{Result, Tensor, D};
use candle_nn::ops::sigmoid;

/// `x / sqrt(mean(x²) + eps) · w` over the last dim, with a per-stream
/// weight `w` (`[hc, n]`) broadcast over `x`'s leading dims (llama.cpp's
/// grouped `ggml_rms_norm` + `ggml_mul` on `[n_embd, hc]` gammas).
pub fn grouped_rms(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    x.broadcast_div(&rms)?.broadcast_mul(w)
}

/// Per-copy gate `sigmoid(sign(s)·sqrt(max(|s|, 1e-6)))` with
/// `s = key·query / sqrt(n)` over the last dim of the already-normalized
/// `key` and `query` (`[.., hc, n]`); returns `[.., hc]`.
pub fn keyed_gate(key: &Tensor, query: &Tensor) -> Result<Tensor> {
    let n = key.dim(D::Minus1)?;
    let s = ((key * query)?.sum(D::Minus1)? / (n as f64).sqrt())?;
    sigmoid(&(s.sign()? * s.abs()?.maximum(1e-6)?.sqrt()?)?)
}
