//! The Mamba-2 selective scan (`nemotron_h` and every other Mamba-2 model).
//!
//! Per head `h` and position `t`, with `g(h) = h / (n_head / n_group)` the
//! grouped `B`/`C` slot:
//!
//! ```text
//! dt_h[t] = softplus(dt_raw[t,h] + dt_bias[h])
//! dA_h[t] = exp(dt_h[t] · a[h])                  // a[h] = −exp(ssm_a[h]) < 0
//! S_h[t]  = dA_h[t] · S_h[t−1] + dt_h[t] · B[t,g] ⊗ x[t,h]
//! y[t,h]  = S_h[t] · C[t,g] + d_skip[h] · x[t,h]  // · = dot over d_state
//! ```
//!
//! — llama.cpp `ggml_ssm_scan` (`ggml-cpu/ops.cpp:10002`,
//! `ggml-cuda/ssm-scan.cu`, `ggml-opencl/ssm_scan.cl`).  The `softplus` is
//! ggml's threshold form (`ggml-impl.h:107`): `input > 20 ? input : log(1 +
//! exp(input))`, *not* `max(x,0) + log1p(exp(−|x|))` — they differ in the last
//! ulp for large negative inputs, and bit-parity with llama.cpp is the point.
//!
//! # Why chunked
//!
//! The recurrence is sequential in `t`, so a naive implementation is a chain
//! of `t` dependent steps; on a GPU that is `t` kernel launches, each
//! round-tripping the state through global memory, which is far *slower* than
//! running it on the CPU.  The associative form takes the dependence out of
//! the inner loop.  With `d_h[t] = dt_h[t] · a[h]` and
//! `cum[t] = Σ_{u≤t} d_h[u]` (inclusive), a chunk `[c0, c0+q)` evaluates in
//! closed form against its entry state `S = S_h[c0−1]`:
//!
//! ```text
//! y[t] = Σ_{u≤t} exp(cum[t] − cum[u]) · dt[u] · (C[t]·B[u]) · x[u]   (intra)
//!      + exp(cum[t]) · (C[t] · S)                                    (inter)
//! S[c1] = exp(cum[q−1]) · S + Σ_u dt[u]·exp(cum[q−1] − cum[u])·x[u] ⊗ B[u]
//! ```
//!
//! So the intra-chunk term is one `q×q` GEMM against a `q×d_state` GEMM
//! (`(L ⊙ (C·Bᵀ) ⊙ dt) @ x`), and only the short carry over `ceil(t/q)`
//! chunks stays sequential.  Both `L` and the carried-state factor use the
//! **inclusive** prefix sum; using an exclusive prefix for either is right at
//! `t = 0` and then drifts.
//!
//! `L` is built as `exp(cum[i] − cum[j])`, never as `exp(cum[i])/exp(cum[j])`:
//! the quotient loses the small-`q` case to a decayed denominator, while the
//! difference is exact to the ulp of `cum` and is `≤ 0` for `j ≤ i`, so `L` is
//! bounded by 1 and no chunk length can overflow it.
//!
//! # Execution
//!
//! Inputs stay borrowed: the prefill path never calls `.to_vec()` or
//! `.contiguous()` on `mixed`/`dt`, and `state` is `&mut [f32]` advanced in
//! place so a decode loop reuses one buffer.  On the CPU each head is one
//! rayon task holding its own `head_dim × d_state` state block
//! (`par_chunks_mut`), the shape [`crate::quantized_qwen`]'s
//! `gated_delta_rule` uses, so the block stays cache-resident across the whole
//! sequence.  Scratch is `O(chunk · (d_state + head_dim))` per head,
//! allocated once per call.
//!
//! The CPU inner kernels are hand-tiled rather than routed through
//! `Tensor::matmul`: the per-head matrices are `q×d_state` (128×64) and
//! `q×q`, too small to amortise a generic GEMM's packing, and — decisively —
//! the operands are strided views into the fused `mixed` row (`x`, `B` and `C`
//! at fixed offsets), so a `Tensor` formulation would force a
//! `contiguous()` copy of the whole prefill.  Reading those offsets directly
//! keeps one pass over memory.
//!
//! # Device execution
//!
//! **Non-CPU devices are supported.**  [`ssm_scan`] dispatches on
//! `mixed.device()`: CPU runs the host kernels above, anything else runs the
//! same chunk algebra as queued `Tensor` ops ([`scan_device`]) so the backend
//! executes it natively — batched GEMMs over `[n_chunk, n_head, q, …]`
//! tensors, with no `contiguous()` or `to_vec()` on `mixed`/`dt`.
//!
//! There is **one state round trip per call**, because `state` is a host
//! `&mut [f32]` that crosses calls: the entry state is uploaded once and the
//! chunk-end state read back once.  Everything in between — the intra-chunk
//! `q×q` products, the per-chunk output and the inter-chunk carry — stays
//! queued on the device.  A future change that keeps the state resident on the
//! device across a decode loop would remove those two transfers.
//!
//! Only the **OpenCL** backend has been verified (device-vs-CPU parity at the
//! real Nemotron geometry, across multiple chunks, asserting both `y` and the
//! final state, plus a strong-decay no-NaN case).  Vulkan, CUDA and Metal use
//! the same `Tensor`-op path but are **unverified**.  One portability trap
//! worth knowing if you touch [`scan_device`]: the triangular mask must be
//! applied to the *exponent* before `exp`, not to the product after it —
//! `cum[i] − cum[j] > 0` for `j > i` overflows to `inf`, and `inf · 0` is
//! `NaN`, which then poisons valid rows through the matmul.

use candle_core::{DType, Result, Tensor};
use rayon::prelude::*;

/// Geometry of one Mamba-2 selective scan (per layer, immutable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanConfig {
    /// `n_head · head_dim` — the width of `x`.
    pub d_inner: usize,
    pub n_head: usize,
    pub d_state: usize,
    /// Grouped `B`/`C`: heads share a slot, `g(h) = h / (n_head / n_group)`.
    pub n_group: usize,
    /// Chunk length of the associative form (128 for Nemotron 3.5).
    pub chunk: usize,
}

impl ScanConfig {
    /// Heads per `B`/`C` group (ggml's `nh / ng`, `repeat_interleave`).
    pub fn head_group(&self) -> usize {
        debug_assert!(self.n_group > 0 && self.n_head % self.n_group == 0);
        self.n_head / self.n_group
    }

    /// Per-head width of `x`.
    pub fn head_dim(&self) -> usize {
        debug_assert!(self.n_head > 0 && self.d_inner % self.n_head == 0);
        self.d_inner / self.n_head
    }

    /// The `B`/`C` slot head `h` reads.
    #[inline]
    pub fn group_of(&self, h: usize) -> usize {
        h / self.head_group()
    }

    /// Row width of the fused `mixed` tensor: `x ‖ B ‖ C`.
    #[inline]
    pub fn conv_dim(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }

    /// `n_head · head_dim · d_state` — the length of the carried state.
    #[inline]
    pub fn state_len(&self) -> usize {
        self.n_head * self.head_dim() * self.d_state
    }

    /// Reject a mis-shaped configuration before any indexing happens.
    pub fn validate(&self) -> Result<()> {
        if self.n_head == 0 || self.d_state == 0 || self.n_group == 0 || self.chunk == 0 {
            candle_core::bail!(
                "ssm_scan: zero-sized geometry (n_head {}, d_state {}, n_group {}, chunk {})",
                self.n_head,
                self.d_state,
                self.n_group,
                self.chunk
            );
        }
        if self.d_inner % self.n_head != 0 {
            candle_core::bail!(
                "ssm_scan: d_inner {} is not a multiple of n_head {}",
                self.d_inner,
                self.n_head
            );
        }
        if self.n_head % self.n_group != 0 {
            candle_core::bail!(
                "ssm_scan: n_head {} is not a multiple of n_group {}",
                self.n_head,
                self.n_group
            );
        }
        Ok(())
    }
}

/// `input > 20 ? input : ln(1 + exp(input))` — ggml's
/// `ggml_compute_softplus_f32`, which is what `ggml_ssm_scan` applies.
#[inline]
pub fn softplus_scalar(input: f32) -> f32 {
    if input > 20.0 {
        input
    } else {
        (1.0 + input.exp()).ln()
    }
}

/// Dot product of two `n`-length slices, four accumulators wide so the
/// compiler can keep independent dependency chains (this is the innermost
/// kernel over `d_state`; using one accumulator halves its throughput).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (mut a0, mut a1, mut a2, mut a3) = (0f32, 0f32, 0f32, 0f32);
    let chunks = n / 4;
    for i in 0..chunks {
        let j = i * 4;
        a0 += a[j] * b[j];
        a1 += a[j + 1] * b[j + 1];
        a2 += a[j + 2] * b[j + 2];
        a3 += a[j + 3] * b[j + 3];
    }
    let mut acc = (a0 + a1) + (a2 + a3);
    for j in chunks * 4..n {
        acc += a[j] * b[j];
    }
    acc
}

/// `out[i] += coef · a[i]` over a `d_state`-length row.
#[inline]
fn axpy(out: &mut [f32], coef: f32, a: &[f32]) {
    for (o, &v) in out.iter_mut().zip(a.iter()) {
        *o += coef * v;
    }
}

/// The plain recurrence, one token at a time — the definition the chunked
/// path must reproduce, kept as the test oracle.
///
/// `x` is `[t, d_inner]`, `dt` raw `[t, n_head]`, `b`/`c` `[t, n_group,
/// d_state]`, `state` `[n_head, head_dim, d_state]`; `y` comes back
/// `[t, d_inner]`.
#[allow(clippy::too_many_arguments)]
pub fn scan_naive(
    cfg: ScanConfig,
    x: &[f32],
    dt: &[f32],
    dt_bias: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d_skip: &[f32],
    state: &mut [f32],
    t: usize,
) -> Vec<f32> {
    let (n_head, hd, ds, ng) = (cfg.n_head, cfg.head_dim(), cfg.d_state, cfg.n_group);
    let mut y = vec![0f32; t * cfg.d_inner];
    for h in 0..n_head {
        let g = cfg.group_of(h);
        let s = &mut state[h * hd * ds..(h + 1) * hd * ds];
        let a_h = a[h];
        let d_h = d_skip[h];
        for tok in 0..t {
            let dt_h = softplus_scalar(dt[tok * n_head + h] + dt_bias[h]);
            let da = (dt_h * a_h).exp();
            let bv = &b[tok * ng * ds + g * ds..][..ds];
            let cv = &c[tok * ng * ds + g * ds..][..ds];
            let xv = &x[tok * cfg.d_inner + h * hd..][..hd];
            for (i, row) in s.chunks_exact_mut(ds).enumerate() {
                let xdt = xv[i] * dt_h;
                let mut acc = 0f32;
                for (sj, (&bj, &cj)) in row.iter_mut().zip(bv.iter().zip(cv.iter())) {
                    *sj = *sj * da + bj * xdt;
                    acc += *sj * cj;
                }
                y[tok * cfg.d_inner + h * hd + i] = acc + d_h * xv[i];
            }
        }
    }
    y
}

/// The three per-head parameter vectors, read once per call.
struct Params {
    dt_bias: Vec<f32>,
    a: Vec<f32>,
    d_skip: Vec<f32>,
}

/// Reinterpret `[1, n]` or `[n]` as `n` f32.
fn small_vec(tn: &Tensor, n: usize, name: &str) -> Result<Vec<f32>> {
    let flat = tn.flatten_all()?;
    let got = flat.dim(0)?;
    if got != n {
        candle_core::bail!("ssm_scan: {name} has {got} elements, expected {n}");
    }
    flat.to_vec1::<f32>()
}

/// Shape checks for the fused inputs; returns the token count.
fn check_inputs(cfg: &ScanConfig, mixed: &Tensor, dt: &Tensor) -> Result<usize> {
    cfg.validate()?;
    let (t, width) = match mixed.dims() {
        [t, w] => (*t, *w),
        d => candle_core::bail!("ssm_scan: mixed must be 2-D [t, conv_dim], got {d:?}"),
    };
    if width != cfg.conv_dim() {
        candle_core::bail!(
            "ssm_scan: mixed row width {width} != d_inner + 2·n_group·d_state = {}",
            cfg.conv_dim()
        );
    }
    match dt.dims() {
        [td, h] if *td == t && *h == cfg.n_head => {}
        d => candle_core::bail!("ssm_scan: dt must be [{t}, {}], got {d:?}", cfg.n_head),
    }
    Ok(t)
}

/// One head's sequential carry over the chunks of a sequence.
///
/// `mixed` is the whole fused `[t, row]` activation; this task reads only its
/// own `x`/`B`/`C` columns and writes only its own `y` columns, so heads run
/// concurrently without synchronising.
#[allow(clippy::too_many_arguments)]
fn head_chunked(
    cfg: &ScanConfig,
    h: usize,
    mixed: &[f32],
    dt: &[f32],
    p: &Params,
    s: &mut [f32],
    t: usize,
    y: &mut [f32],
) {
    let (hd, ds, ng) = (cfg.head_dim(), cfg.d_state, cfg.n_group);
    let row = cfg.conv_dim();
    let d_inner = cfg.d_inner;
    let g_off = cfg.group_of(h) * ds;
    let b_off = d_inner + g_off;
    let c_off = d_inner + ng * ds + g_off;
    let a_h = p.a[h];
    let bias = p.dt_bias[h];
    let d_h = p.d_skip[h];

    let q_max = cfg.chunk.min(t.max(1));
    // Per-chunk scratch: decay/softplus, the `q×q` `L` (triangular, so only
    // the lower triangle is ever touched), and the entry-state readout.
    let mut dtv = vec![0f32; q_max];
    let mut cumv = vec![0f32; q_max];
    let mut lmat = vec![0f32; q_max * q_max];
    let mut yintra = vec![0f32; q_max * hd];

    let mut t0 = 0usize;
    while t0 < t {
        let q = (t - t0).min(cfg.chunk);

        // softplus(dt + bias) and its inclusive cumsum of `dt·a`.
        let mut cum = 0f32;
        for i in 0..q {
            let d = softplus_scalar(dt[(t0 + i) * cfg.n_head + h] + bias);
            dtv[i] = d;
            cum += d * a_h;
            cumv[i] = cum;
        }
        let cum_end = cumv[q - 1];
        let decay_total = cum_end.exp();

        // L[i][j] = exp(cum[i] − cum[j]) for j ≤ i, else 0 — folded with the
        // `dt[j]` column scale that the intra sum needs:
        //   L'[i][j] = exp(cum[i]−cum[j]) · dt[j]
        for i in 0..q {
            let li = &mut lmat[i * q..(i + 1) * q];
            for (j, l) in li.iter_mut().enumerate().take(i + 1) {
                *l = (cumv[i] - cumv[j]).exp() * dtv[j];
            }
            for l in li.iter_mut().skip(i + 1) {
                *l = 0.0;
            }
        }

        // (C·Bᵀ) elementwise into L', then y_intra = L'·₍ij₎ @ x.
        // Both operands are strided views into `mixed`; the q×q staging avoids
        // re-reading the d_state-long B/C rows for every (i,j).
        yintra[..q * hd].fill(0.0);
        for i in 0..q {
            let ci = &mixed[(t0 + i) * row + c_off..][..ds];
            let li = &mut lmat[i * q..(i + 1) * q];
            for j in 0..=i {
                let bj = &mixed[(t0 + j) * row + b_off..][..ds];
                li[j] *= dot(ci, bj);
            }
        }
        for i in 0..q {
            let li = &lmat[i * q..(i + 1) * q];
            let oi = &mut yintra[i * hd..(i + 1) * hd];
            for (j, &l) in li.iter().enumerate().take(i + 1) {
                if l == 0.0 {
                    continue;
                }
                let xj = &mixed[(t0 + j) * row + h * hd..][..hd];
                axpy(oi, l, xj);
            }
        }

        // Carried-state term and the epilogue: y = y_intra + exp(cum)·(C·S) + D·x.
        // `y` here is this head's own `[t, head_dim]` block.
        for i in 0..q {
            let ci = &mixed[(t0 + i) * row + c_off..][..ds];
            let xv = &mixed[(t0 + i) * row + h * hd..][..hd];
            let out = &mut y[(t0 + i) * hd..][..hd];
            let fwd = cumv[i].exp();
            out.copy_from_slice(&yintra[i * hd..(i + 1) * hd]);
            for k in 0..hd {
                let srow = &s[k * ds..(k + 1) * ds];
                // exp(cum_i)·(C_i·S) + D·x, accumulated onto the intra term.
                out[k] += fwd * dot(srow, ci) + d_h * xv[k];
            }
        }

        // Chunk-end state: S ← exp(cum_end)·S + Σ_i dt_i·exp(cum_end−cum_i)·x_i ⊗ B_i.
        for v in s.iter_mut() {
            *v *= decay_total;
        }
        for i in 0..q {
            let back = dtv[i] * (cum_end - cumv[i]).exp();
            let xv = &mixed[(t0 + i) * row + h * hd..][..hd];
            let bv = &mixed[(t0 + i) * row + b_off..][..ds];
            for k in 0..hd {
                let coef = xv[k] * back;
                if coef != 0.0 {
                    axpy(&mut s[k * ds..(k + 1) * ds], coef, bv);
                }
            }
        }

        t0 += q;
    }
}

/// The single-token recurrence — no chunk matrices at all.
#[allow(clippy::too_many_arguments)]
fn head_step(
    cfg: &ScanConfig,
    h: usize,
    mixed_row: &[f32],
    dt_h: f32,
    p: &Params,
    s: &mut [f32],
    out: &mut [f32],
) {
    let (hd, ds, ng) = (cfg.head_dim(), cfg.d_state, cfg.n_group);
    let g_off = cfg.group_of(h) * ds;
    let b_off = cfg.d_inner + g_off;
    let c_off = cfg.d_inner + ng * ds + g_off;

    let dtv = softplus_scalar(dt_h + p.dt_bias[h]);
    let da = (dtv * p.a[h]).exp();
    let bv = &mixed_row[b_off..][..ds];
    let cv = &mixed_row[c_off..][..ds];
    let xv = &mixed_row[h * hd..][..hd];
    let d_h = p.d_skip[h];

    for k in 0..hd {
        let rowk = &mut s[k * ds..(k + 1) * ds];
        let xdt = xv[k] * dtv;
        for (sj, &bj) in rowk.iter_mut().zip(bv.iter()) {
            *sj = *sj * da + bj * xdt;
        }
        out[k] = dot(rowk, cv) + d_h * xv[k];
    }
}

/// `y = Σ_s C[t,g]·S[t] + d_skip·x`, advancing `state` in place and returning
/// `[t, d_inner]` (token-major, head-major within a token) — llama.cpp
/// `ggml_ssm_scan`.
///
/// `mixed` is the fused, contiguous, post-conv activation
/// `[t, d_inner + 2·n_group·d_state]`: row `t` is `x [d_inner] ‖ B
/// [n_group·d_state] ‖ C [n_group·d_state]`, `B[t,g,s]` at `t·row + d_inner +
/// g·d_state + s`.  `dt` is **raw** — `softplus(dt + dt_bias)` is applied here,
/// as ggml does.  `a` is already negative (`−exp(ssm_a)`).  `state` is
/// `n_head·head_dim·d_state` f32, head-major (`state[h][i][s]`), and crosses
/// calls: pass it in and it comes back advanced.
///
/// `t == 1` runs the scalar recurrence instead of the chunk algebra, so
/// incremental decode pays one small GEMV rather than a `q×q` matrix.
#[allow(clippy::too_many_arguments)]
pub fn ssm_scan(
    cfg: ScanConfig,
    mixed: &Tensor,
    dt: &Tensor,
    dt_bias: &Tensor,
    a: &Tensor,
    d_skip: &Tensor,
    state: &mut [f32],
) -> Result<Tensor> {
    let t = check_inputs(&cfg, mixed, dt)?;
    if state.len() != cfg.state_len() {
        candle_core::bail!(
            "ssm_scan: state has {} elements, expected {} (n_head {} · head_dim {} · d_state {})",
            state.len(),
            cfg.state_len(),
            cfg.n_head,
            cfg.head_dim(),
            cfg.d_state
        );
    }
    let n = cfg.n_head;
    let p = Params {
        dt_bias: small_vec(dt_bias, n, "dt_bias")?,
        a: small_vec(a, n, "a")?,
        d_skip: small_vec(d_skip, n, "d_skip")?,
    };
    let dev = mixed.device().clone();

    if mixed.dtype() != DType::F32 || dt.dtype() != DType::F32 {
        candle_core::bail!(
            "ssm_scan: needs f32 inputs, got mixed {:?} / dt {:?}",
            mixed.dtype(),
            dt.dtype()
        );
    }
    // A device backend runs the same chunk algebra as queued `Tensor` ops; the
    // CPU runs the hand-tiled host kernels. Both advance the caller's `state`,
    // so they are interchangeable across calls (prefill on one, decode on the
    // other) and the tests assert device == CPU.
    if !dev.is_cpu() {
        return scan_device(&cfg, mixed, dt, dt_bias, a, d_skip, state, t);
    }

    // Borrowed flattened views: no copy of `mixed`/`dt` beyond what the
    // backend's `to_vec1` does for a contiguous CPU tensor (a single memcpy
    // into the returned Vec; the data stays in one cache-friendly pass).
    let mixed = mixed.flatten_all()?.to_vec1::<f32>()?;
    let dt = dt.flatten_all()?.to_vec1::<f32>()?;
    let hd = cfg.head_dim();
    let ds = cfg.d_state;
    let mut y = vec![0f32; t * cfg.d_inner];

    if t == 1 {
        // Decode fast path: no chunk matrices, no `q×q` staging.
        let row = cfg.conv_dim();
        let row_slice = &mixed[..row];
        state
            .par_chunks_mut(hd * ds)
            .zip(y.par_chunks_mut(hd))
            .enumerate()
            .for_each(|(h, (s, out))| {
                head_step(&cfg, h, row_slice, dt[h], &p, s, out);
            });
    } else {
        // Heads write disjoint memory. `y` is token-major (`[t, d_inner]`) but
        // a head owns a strided column band of it, so each head accumulates its
        // own contiguous `[t, head_dim]` block and a single sequential pass
        // scatters them into the token-major output below. That keeps the head
        // tasks borrow-disjoint (no `unsafe` aliasing) for one cheap copy that
        // is small next to the scan itself.
        let mut y_head = vec![0f32; y.len()];
        // Head-major output blocks that mirror the state split exactly, so the
        // two `par_chunks_mut` walks hand each task one state block and one
        // output block, both disjoint, with no aliasing to prove.
        state
            .par_chunks_mut(hd * ds)
            .zip(y_head.par_chunks_mut(t * hd))
            .enumerate()
            .for_each(|(h, (s, out))| {
                head_chunked(&cfg, h, &mixed, &dt, &p, s, t, out);
            });
        for h in 0..cfg.n_head {
            let src = &y_head[h * t * hd..(h + 1) * t * hd];
            for tok in 0..t {
                let dst = (tok * cfg.d_inner + h * hd)..(tok * cfg.d_inner + h * hd + hd);
                y[dst].copy_from_slice(&src[tok * hd..(tok + 1) * hd]);
            }
        }
    }
    Tensor::from_vec(y, (t, cfg.d_inner), &dev)
}

/// Single-token fast path: the scalar recurrence, no chunk matrices.
///
/// `mixed_row` is `[1, d_inner + 2·n_group·d_state]`, `dt_row` is
/// `[1, n_head]`; the result is `[d_inner]`.
#[allow(clippy::too_many_arguments)]
pub fn ssm_scan_step(
    cfg: ScanConfig,
    mixed_row: &Tensor,
    dt_row: &Tensor,
    dt_bias: &Tensor,
    a: &Tensor,
    d_skip: &Tensor,
    state: &mut [f32],
) -> Result<Vec<f32>> {
    let out = ssm_scan(cfg, mixed_row, dt_row, dt_bias, a, d_skip, state)?;
    out.flatten_all()?.to_vec1::<f32>()
}

/// The scan as queued `Tensor` ops, so a backend executes it natively.
///
/// The whole sequence is expressed as batched GEMMs over
/// `[n_chunk, n_head, q, ...]` tensors: `y_intra`, the per-chunk output, is one
/// `(q×q) · (q×head_dim)` product for every head and chunk at once, and the
/// chunk-end state is a `(d_state×q) · (q×head_dim)` product. Only the short
/// carry over chunks is a loop, and because nothing in that loop is read back
/// the backend queues it as one op graph.
///
/// Host traffic is one upload of the entry state and one readback of the final
/// state — both needed because `state` crosses calls on the host.
///
/// This is the same associative form as the CPU path (inclusive cumsum, `L`
/// built from exponent differences), so the only difference between the two is
/// backend rounding.
#[allow(clippy::too_many_arguments)]
fn scan_device(
    cfg: &ScanConfig,
    mixed: &Tensor,
    dt: &Tensor,
    dt_bias: &Tensor,
    a: &Tensor,
    d_skip: &Tensor,
    state: &mut [f32],
    t: usize,
) -> Result<Tensor> {
    let (n, hd, ds, ng) = (cfg.n_head, cfg.head_dim(), cfg.d_state, cfg.n_group);
    let dev = mixed.device();
    let q = cfg.chunk;

    // Pad to whole chunks so every chunk has the same shape and the batched
    // GEMMs need no ragged handling; the padding is trimmed from `y` at the end.
    // Padded rows have `mixed = 0` and `dt = 0`, so `softplus(0 + bias)` writes
    // a nonzero state contribution — harmless only because those rows are past
    // `t` and never read into `y`, and because they are excluded from the
    // *final* state by taking the state after the last real chunk.
    let t_pad = t.div_ceil(q) * q;
    let n_chunk = t_pad / q;

    // Split the fused row into x / B / C.
    let x = mixed.narrow(1, 0, cfg.d_inner)?;
    let bm = mixed.narrow(1, cfg.d_inner, ng * ds)?;
    let cm = mixed.narrow(1, cfg.d_inner + ng * ds, ng * ds)?;

    // Broadcast each group's B/C to its `head_group()` heads: `[t, n, ds]`.
    let gh = cfg.head_group();
    // `contiguous()` between the broadcast and the reshape is required:
    // a broadcast view has repeated strides, and reshaping it would have to
    // guess a layout. On a device backend that guess is not honoured the same
    // way as on the CPU, so the copy is made explicit here (it is
    // `t · n_head · d_state` floats, small next to `mixed` itself).
    let merge_groups = |v: &Tensor| -> Result<Tensor> {
        v.reshape((t, ng, 1, ds))?
            .broadcast_as((t, ng, gh, ds))?
            .contiguous()?
            .reshape((t, n, ds))
    };
    let b_all = merge_groups(&bm)?;
    let c_all = merge_groups(&cm)?;
    let x_all = x.reshape((t, n, hd))?;

    // dt: `softplus(dt + bias)` in ggml's threshold form, then the per-token
    // decay `dt·a`.
    let dt_bias_v = dt_bias.flatten_all()?.reshape((1, n))?;
    let a_v = a.flatten_all()?.reshape((1, n))?;
    let dskip_v = d_skip.flatten_all()?.reshape((1, n, 1))?;
    let shifted = dt.broadcast_add(&dt_bias_v)?;
    let dt_sp = {
        let hi = shifted.gt(20.0)?;
        let lo = (shifted.exp()? + 1.0)?.log()?;
        hi.where_cond(&shifted, &lo)?
    };

    // Pad the token axis to whole chunks.
    let pad = t_pad - t;
    let (d_tok, x_p, b_p, c_p, dt_p) = if pad == 0 {
        let d_tok = dt_sp.broadcast_mul(&a_v)?;
        (d_tok, x_all, b_all, c_all, dt_sp)
    } else {
        (
            dt_sp.broadcast_mul(&a_v)?.pad_with_zeros(0, 0, pad)?,
            x_all.pad_with_zeros(0, 0, pad)?,
            b_all.pad_with_zeros(0, 0, pad)?,
            c_all.pad_with_zeros(0, 0, pad)?,
            dt_sp.pad_with_zeros(0, 0, pad)?,
        )
    };

    // [t_pad, n, w] -> [n_chunk, n, q, w]: chunk axis leads (batched), head axis
    // next, matching the state's leading dimensions.
    let to_chunks = |v: &Tensor, w: usize| -> Result<Tensor> {
        v.reshape((n_chunk, q, n, w))?
            .transpose(1, 2)?
            .contiguous()
    };
    let d_c = to_chunks(&d_tok, 1)?.squeeze(3)?; // [nc, n, q]  = dt·a
    let dt_c = to_chunks(&dt_p, 1)?.squeeze(3)?; // [nc, n, q]  = dt
    let x_c = to_chunks(&x_p, hd)?; // [nc, n, q, hd]
    let b_c = to_chunks(&b_p, ds)?; // [nc, n, q, ds]
    let c_c = to_chunks(&c_p, ds)?; // [nc, n, q, ds]

    // Optional per-intermediate NaN probe (`SSM_SCAN_NAN_PROBE=1`). A device
    // backend that silently produces NaN in one op is otherwise very hard to
    // localise, so each stage is named and checked when the variable is set.
    // Each check costs one host readback, so it is off by default.
    let probe = std::env::var("SSM_SCAN_NAN_PROBE").is_ok();
    let chk = |name: &str, v: &Tensor| -> Result<()> {
        if probe {
            if let Ok(f) = v.flatten_all().and_then(|f| f.to_vec1::<f32>()) {
                let bad = f.iter().filter(|x| !x.is_finite()).count();
                if bad > 0 {
                    eprintln!(
                        "SSM_SCAN_NAN_PROBE: {name} has {bad}/{} non-finite (dims {:?})",
                        f.len(),
                        v.dims()
                    );
                }
            }
        }
        Ok(())
    };
    chk("d_c", &d_c)?;
    chk("x_c", &x_c)?;
    chk("b_c", &b_c)?;
    chk("c_c", &c_c)?;

    // Inclusive cumsum of `dt·a` along the chunk axis.
    let cum = d_c.cumsum(2)?; // [nc, n, q]
    chk("cum", &cum)?;

    // L[i,j] = exp(cum[i] − cum[j]) · dt[j] for j <= i, 0 above.
    // The mask is applied to the *exponent*: for j > i the difference is
    // positive and can overflow to inf, and `inf · 0` would be NaN rather than
    // 0, so masked-out entries are set to −inf before `exp` (→ exactly 0).
    let idx = Tensor::arange(0u32, q as u32, dev)?;
    let lower = idx.reshape((q, 1))?.broadcast_ge(&idx.reshape((1, q))?)?; // j <= i
    let lower4 = lower.broadcast_as((n_chunk, n, q, q))?;
    let ci = cum.unsqueeze(3)?; // [nc, n, q, 1]
    let cj = cum.unsqueeze(2)?; // [nc, n, 1, q]
    let diff = ci.broadcast_sub(&cj)?; // [nc, n, q, q]
    // Construct `-inf` directly rather than as `zeros.affine(0.0, -inf)`: any
    // implementation that scales before biasing turns that into `0 · -inf =
    // NaN`, which is exactly the failure mode this mask exists to prevent.
    let neg_inf = Tensor::full(f32::NEG_INFINITY, diff.dims(), diff.device())?;
    let diff = lower4.where_cond(&diff, &neg_inf)?;
    let dt_col = dt_c.unsqueeze(2)?.broadcast_as((n_chunk, n, q, q))?;
    let lmat = diff.exp()?.mul(&dt_col)?;
    // Also clear entries whose `dt[j]` is zero (padding), so they contribute
    // nothing even where the exponent underflowed to exactly 1.
    let mask_f = lower.to_dtype(DType::F32)?.broadcast_as((n_chunk, n, q, q))?;
    let lmat = lmat.mul(&mask_f)?;
    chk("lmat", &lmat)?;

    // y_intra = (L ⊙ C·Bᵀ) @ x, batched over every chunk and head at once.
    let cb = c_c.matmul(&b_c.transpose(2, 3)?.contiguous()?)?; // [nc, n, q, q]
    let y_intra = lmat.mul(&cb)?.matmul(&x_c)?; // [nc, n, q, hd]
    chk("y_intra", &y_intra)?;

    // ── the short sequential carry over chunks ──
    // `y_inter` reads the chunk-entry state, so the per-chunk output and the
    // state update must interleave; nothing is read back until the end.
    let cum_end = cum.narrow(2, q - 1, 1)?; // [nc, n, 1]
    let decay_total = cum_end.exp()?;

    // Device state is kept as `[n, d_state, head_dim]` so the readout is one
    // GEMM: `C[n,q,ds] @ S[n,ds,hd]`. The caller's layout is
    // `[n, head_dim, d_state]`, so upload transposed and transpose back once.
    let s0 = Tensor::from_vec(state.to_vec(), (n, hd, ds), dev)?;
    let mut s = s0.transpose(1, 2)?.contiguous()?; // [n, ds, hd]

    let mut y_chunks: Vec<Tensor> = Vec::with_capacity(n_chunk);
    for k in 0..n_chunk {
        let dec = decay_total.narrow(0, k, 1)?.squeeze(0)?.unsqueeze(2)?; // [n,1,1]
        let ck = c_c.narrow(0, k, 1)?.squeeze(0)?; // [n, q, ds]
        let cumk = cum.narrow(0, k, 1)?.squeeze(0)?.unsqueeze(2)?; // [n, q, 1]
        // y_inter = exp(cum) · (C·S_entry)
        let y_inter = ck.matmul(&s)?.broadcast_mul(&cumk.exp()?)?; // [n, q, hd]
        chk("y_inter", &y_inter)?;
        let yk = (y_intra.narrow(0, k, 1)?.squeeze(0)? + y_inter)?;
        chk("yk", &yk)?;
        y_chunks.push(yk);

        // S ← exp(cum_end)·S + Σ_i dt_i·exp(cum_end − cum_i)·B_i ⊗ x_i, in the
        // `[n, ds, hd]` orientation: `Bᵀ[n,ds,q] @ (x·w)[n,q,hd]`. Note the
        // weight uses the positive `dt`, not `dt·a` — only the cumulative
        // products use `dt·a`.
        let xk = x_c.narrow(0, k, 1)?.squeeze(0)?; // [n, q, hd]
        let bk = b_c.narrow(0, k, 1)?.squeeze(0)?; // [n, q, ds]
        let cumk_v = cum.narrow(0, k, 1)?.squeeze(0)?; // [n, q]
        let dtk = dt_c.narrow(0, k, 1)?.squeeze(0)?; // [n, q]
        let back = cum_end
            .narrow(0, k, 1)?
            .squeeze(0)?
            .broadcast_sub(&cumk_v)?
            .exp()?; // [n, q]
        let w = (back * dtk)?; // [n, q]
        let xw = xk.broadcast_mul(&w.unsqueeze(2)?)?; // [n, q, hd]
        chk("xw", &xw)?;
        let contrib = bk.transpose(1, 2)?.contiguous()?.matmul(&xw)?; // [n, ds, hd]
        chk("contrib", &contrib)?;
        s = (s.broadcast_mul(&dec)? + contrib)?;
        if probe {
            let f = s.flatten_all()?.to_vec1::<f32>()?;
            let bad = f.iter().filter(|x| !x.is_finite()).count();
            if bad > 0 {
                eprintln!("SSM_SCAN_NAN_PROBE: state after chunk {k} has {bad} non-finite");
            }
            let de = dec.flatten_all()?.to_vec1::<f32>()?;
            let ce = cum_end.flatten_all()?.to_vec1::<f32>()?;
            if de.iter().any(|x| !x.is_finite()) || ce.iter().any(|x| !x.is_finite()) {
                eprintln!("SSM_SCAN_NAN_PROBE: decay_total/cum_end non-finite at chunk {k}: cum_end[0]={}", ce[0]);
            }
        }
    }

    // [nc, n] blocks of [q, hd] -> [t_pad, n, hd] -> [t, d_inner].
    let y = Tensor::stack(&y_chunks, 0)?;
    let y = y
        .transpose(1, 2)?
        .contiguous()?
        .reshape((t_pad, n, hd))?;
    // D·x epilogue, then trim the padding.
    let y = y.broadcast_add(&x_p.broadcast_mul(&dskip_v)?)?;
    chk("y after d_skip", &y)?;
    let y = y
        .reshape((t_pad, cfg.d_inner))?
        .narrow(0, 0, t)?
        .contiguous()?;

    // One readback: the final state goes back into the caller's buffer so the
    // next call continues from it.
    let s_host = s
        .transpose(1, 2)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    state.copy_from_slice(&s_host);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_matches_ggml_threshold_form() {
        // Above the threshold ggml returns the input unchanged.
        assert_eq!(softplus_scalar(20.5), 20.5);
        // At and below it, `ln(1+e^x)`.
        assert_eq!(softplus_scalar(20.0), (1.0f32 + 20.0f32.exp()).ln());
        assert!((softplus_scalar(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        // Deep negative saturation: `e^x` is below the f32 ulp of 1 (~1.19e-7
        // for x ≲ −16), so `1 + e^x` rounds back to `1.0` and ggml's form
        // returns exactly 0. That is ggml's behavior (verified against the C
        // expression), and matching it matters more than being ideally
        // accurate: a `max(x,0) + log1p(exp(−|x|))` reformulation returns the
        // tiny `e^x` here and would diverge from llama.cpp in the last ulp.
        assert_eq!(softplus_scalar(-100.0), 0.0);
        assert_eq!(softplus_scalar(-20.0), 0.0);
        // Above that boundary the term survives.
        assert!(softplus_scalar(-10.0) > 4.5e-5);
    }

    #[test]
    fn rejects_bad_geometry() {
        let bad = ScanConfig {
            d_inner: 10,
            n_head: 3,
            d_state: 4,
            n_group: 1,
            chunk: 4,
        };
        assert!(bad.validate().is_err(), "d_inner not a multiple of n_head");
        let bad = ScanConfig {
            d_inner: 12,
            n_head: 3,
            d_state: 4,
            n_group: 2,
            chunk: 4,
        };
        assert!(bad.validate().is_err(), "n_head not a multiple of n_group");
    }

    #[test]
    fn dot_handles_ragged_lengths() {
        let a = [1f32, 2., 3., 4., 5.];
        let b = [1f32, 1., 1., 1., 1.];
        assert_eq!(dot(&a, &b), 15.0);
        assert_eq!(dot(&a[..2], &b[..2]), 3.0);
    }
}
