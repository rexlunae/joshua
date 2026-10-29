//! Numerical acceptance for the chunked Mamba-2 selective scan
//! ([`joshua::scan`]).
//!
//! The contract is that the associative (chunked) form reproduces the
//! per-token recurrence exactly, not approximately: both evaluate the same
//! sum, so the only admissible difference is f32 round-off in the order of
//! accumulation.  These tests pin that down across every chunk size the
//! implementation can be handed (including `Q = 1`, which degenerates to the
//! recurrence, and `Q = 128`, the Nemotron 3.5 value), every relevant
//! sequence length (including lengths below one chunk and lengths that are
//! not a multiple of it), and the real Nemotron geometry (64 heads,
//! `head_dim` 128, `d_state` 64, `n_group` 8).
//!
//! They also cover the property that actually matters for inference: a
//! prefill of `T` tokens must produce the same logits-relevant outputs as
//! feeding those `T` tokens one at a time through the carried state, because
//! decode is the `t == 1` path and prefill is the chunked path.

use candle_core::{DType, Device, Tensor};
use joshua::scan::{scan_naive, ssm_scan, ssm_scan_step, ScanConfig};

/// Deterministic pseudo-random f32s in `[-scale, scale)` — a tiny xorshift so
/// the fixtures are reproducible without pulling in `rand` (and so the tests
/// do not depend on the rest of the crate's RNG choices).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u32(&mut self) -> u32 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    /// Uniform in `[-scale, scale)`.
    fn f32(&mut self, scale: f32) -> f32 {
        let u = self.next_u32() as f32 / u32::MAX as f32; // [0, 1]
        (u * 2.0 - 1.0) * scale
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32(scale)).collect()
    }
}

fn cfg(d_inner: usize, n_head: usize, d_state: usize, n_group: usize, chunk: usize) -> ScanConfig {
    ScanConfig {
        d_inner,
        n_head,
        d_state,
        n_group,
        chunk,
    }
}

/// The real Nemotron 3.5 Mamba-2 geometry (64 heads × 128 = 8192 d_inner,
/// `d_state` 64, 8 groups, 128-token chunks).
fn nemotron() -> ScanConfig {
    cfg(64 * 128, 64, 64, 8, 128)
}

/// A small geometry with the same *shape* of problem (grouped B/C where
/// several heads share a slot, `d_state` wider than `head_dim`) so the tests
/// stay fast while still exercising every index path.
fn small() -> ScanConfig {
    cfg(4 * 8, 4, 16, 2, 128)
}

/// Everything a scan needs, in both the fused (implementation) and split
/// (reference) layouts, so the two can be compared without conversion bugs
/// hiding a difference.
struct Fixture {
    mixed: Vec<f32>, // [t, conv_dim]
    dt: Vec<f32>,    // [t, n_head] raw
    dt_bias: Vec<f32>,
    a: Vec<f32>, // negative
    b: Vec<f32>, // [t, n_group, d_state]
    c: Vec<f32>,
    d_skip: Vec<f32>,
    x: Vec<f32>, // [t, d_inner], the x block of `mixed`
    state0: Vec<f32>,
    t: usize,
}

fn fixture(c: &ScanConfig, t: usize, seed: u64) -> Fixture {
    let mut rng = Rng::new(seed);
    let (hd, ds, ng, n) = (c.head_dim(), c.d_state, c.n_group, c.n_head);
    let row = c.conv_dim();

    let x = rng.vec(t * c.d_inner, 0.5);
    // B and C are small (they are the SSM's input/output projections); x is
    // the one that carries real energy.
    let b = rng.vec(t * ng * ds, 0.2);
    let c_ = rng.vec(t * ng * ds, 0.2);
    let dt = rng.vec(t * n, 1.0);
    let dt_bias = rng.vec(n, 0.5);
    // A = −exp(ssm_a) is negative and O(0.1..16) in real checkpoints. It must
    // be strictly negative: with `a == 0` the recurrence has no decay and the
    // state grows without bound, which is a property of the model, not of the
    // scan — a fixture that hits it would be testing f32 overflow rather than
    // the algorithm, so keep it well inside the decaying regime.
    let a: Vec<f32> = (0..n).map(|_| -(0.5 + rng.f32(0.5).abs() * 15.0)).collect();
    let d_skip = rng.vec(n, 0.5);
    let state0 = rng.vec(n * hd * ds, 0.1);

    // Assemble the fused `mixed` row exactly as the loader does:
    // x [d_inner] ‖ B [n_group*d_state] ‖ C [n_group*d_state].
    let mut mixed = vec![0f32; t * row];
    for tok in 0..t {
        let dst = &mut mixed[tok * row..(tok + 1) * row];
        dst[..c.d_inner].copy_from_slice(&x[tok * c.d_inner..(tok + 1) * c.d_inner]);
        dst[c.d_inner..c.d_inner + ng * ds]
            .copy_from_slice(&b[tok * ng * ds..(tok + 1) * ng * ds]);
        dst[c.d_inner + ng * ds..].copy_from_slice(&c_[tok * ng * ds..(tok + 1) * ng * ds]);
    }

    Fixture {
        mixed,
        dt,
        dt_bias,
        a,
        b,
        c: c_,
        d_skip,
        x,
        state0,
        t,
    }
}

/// Run the chunked path (`ssm_scan`) and the recurrence (`scan_naive`) on the
/// same fixture; return `(y_chunked, state_chunked, y_naive, state_naive)`.
fn run_both(c: &ScanConfig, f: &Fixture) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let dev = Device::Cpu;
    let row = c.conv_dim();
    let mixed = Tensor::from_vec(f.mixed.clone(), (f.t, row), &dev).unwrap();
    let dt = Tensor::from_vec(f.dt.clone(), (f.t, c.n_head), &dev).unwrap();
    let dt_bias = Tensor::from_vec(f.dt_bias.clone(), c.n_head, &dev).unwrap();
    let a = Tensor::from_vec(f.a.clone(), c.n_head, &dev).unwrap();
    let d_skip = Tensor::from_vec(f.d_skip.clone(), c.n_head, &dev).unwrap();

    let mut st_chunked = f.state0.clone();
    let y_chunked = ssm_scan(*c, &mixed, &dt, &dt_bias, &a, &d_skip, &mut st_chunked)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mut st_naive = f.state0.clone();
    let y_naive = scan_naive(
        *c,
        &f.x,
        &f.dt,
        &f.dt_bias,
        &f.a,
        &f.b,
        &f.c,
        &f.d_skip,
        &mut st_naive,
        f.t,
    );

    (y_chunked, st_chunked, y_naive, st_naive)
}

/// Scale-aware closeness for the device-vs-CPU comparisons.
///
/// Element-wise *relative* error is the wrong metric for the state: an entry
/// that happens to be ~1e-8 in magnitude can differ by 100% of itself while the
/// tensor as a whole agrees to f32 round-off, and keying on that either forces
/// an absurdly loose tolerance or produces false alarms. Compare the max
/// absolute deviation against the largest magnitude in the reference instead,
/// which is what "same answer to f32 precision" means for a dense buffer.
fn assert_close_scaled(name: &str, got: &[f32], want: &[f32], rel_of_scale: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(g.is_finite(), "{name}[{i}] is {g} (not finite); want {w}");
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    let rel = worst / scale;
    assert!(
        rel <= rel_of_scale,
        "{name}: max|Δ| = {worst:e} on scale {scale:e} = {rel:e} > {rel_of_scale:e} (at [{at}]: got {}, want {})",
        got[at],
        want[at]
    );
}

fn assert_close(name: &str, got: &[f32], want: &[f32], rel: f32, abs: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        if !g.is_finite() {
            panic!("{name}[{i}] is {g} (not finite); want {w}");
        }
        let err = (g - w).abs();
        let tol = abs + rel * w.abs();
        if err > tol && err > worst {
            worst = err;
            at = i;
        }
    }
    assert!(
        worst == 0.0,
        "{name}: worst |Δ| = {worst:e} at [{at}] (got {}, want {})",
        got[at],
        want[at]
    );
}

// ─── chunked == naive ───────────────────────────────────────────────────────

/// The core acceptance sweep: `Q ∈ {1, 4, 128}` × `T ∈ {1, 7, 128, 131, 512}`.
///
/// These are *exactly* the same computation in a different association order,
/// so the tolerance is f32 round-off only — not a modelling tolerance.  A
/// mismatch here means a wrong formula, not a rounding difference.
#[test]
fn chunked_matches_naive_across_chunk_and_length() {
    let base = small();
    for &q in &[1usize, 4, 128] {
        for &t in &[1usize, 7, 128, 131, 512] {
            let c = ScanConfig { chunk: q, ..base };
            let f = fixture(&c, t, 0x1234_5678 ^ (q as u64) << 8 ^ t as u64);
            let (yc, sc, yn, sn) = run_both(&c, &f);
            let tag = format!("Q={q} T={t}");
            assert_close(&format!("{tag} y"), &yc, &yn, 1e-5, 1e-6);
            assert_close(&format!("{tag} state"), &sc, &sn, 1e-5, 1e-6);
        }
    }
}

/// The real Nemotron geometry: 64 heads, `head_dim` 128, `d_state` 64,
/// `n_group` 8, chunk 128 — at a length that crosses several chunk
/// boundaries and one that does not.
#[test]
fn chunked_matches_naive_nemotron_geometry() {
    let c = nemotron();
    for &t in &[1usize, 128, 131, 512] {
        let f = fixture(&c, t, 0xfeed_0000 + t as u64);
        let (yc, sc, yn, sn) = run_both(&c, &f);
        let tag = format!("nemotron T={t}");
        assert_close(&format!("{tag} y"), &yc, &yn, 1e-5, 1e-5);
        assert_close(&format!("{tag} state"), &sc, &sn, 1e-5, 1e-5);
    }
}

/// `Q > T` (a chunk longer than the whole sequence), `Q` larger than the
/// Nemotron value, and a single-token sequence with a large chunk — the
/// short-chunk edge cases where the `q < chunk` paths run.
#[test]
fn chunked_matches_naive_when_chunk_exceeds_sequence() {
    let base = small();
    for &(q, t) in &[(256usize, 5usize), (1024, 1), (1024, 33), (3, 100), (7, 100)] {
        let c = ScanConfig { chunk: q, ..base };
        let f = fixture(&c, t, 0xabcd_0000 ^ q as u64 ^ (t as u64) << 16);
        let (yc, sc, yn, sn) = run_both(&c, &f);
        let tag = format!("Q={q} T={t}");
        assert_close(&format!("{tag} y"), &yc, &yn, 1e-5, 1e-6);
        assert_close(&format!("{tag} state"), &sc, &sn, 1e-5, 1e-6);
    }
}

/// One head, one group: the degenerate way to index the grouped `B`/`C`
/// lookup (`g(h)` is always 0).
#[test]
fn chunked_matches_naive_single_head_single_group() {
    let c = cfg(16, 1, 32, 1, 8);
    for &t in &[1usize, 9, 40] {
        let f = fixture(&c, t, 0x5150 + t as u64);
        let (yc, sc, yn, sn) = run_both(&c, &f);
        assert_close(&format!("T={t} y"), &yc, &yn, 1e-5, 1e-6);
        assert_close(&format!("T={t} state"), &sc, &sn, 1e-5, 1e-6);
    }
}

/// `n_group == n_head` — one `B`/`C` slot per head (`head_group()` is 1).
#[test]
fn chunked_matches_naive_ungrouped() {
    let c = cfg(4 * 4, 4, 8, 4, 4);
    let f = fixture(&c, 21, 0x9999);
    let (yc, sc, yn, sn) = run_both(&c, &f);
    assert_close("y", &yc, &yn, 1e-5, 1e-6);
    assert_close("state", &sc, &sn, 1e-5, 1e-6);
}

// ─── prefill vs incremental decode ──────────────────────────────────────────

/// The property inference depends on: `T` tokens at once (chunked) equals `T`
/// calls of one token each (the `t == 1` recurrence) with the state carried
/// between them.
#[test]
fn prefill_matches_incremental_decode() {
    let c = small();
    for &t in &[1usize, 2, 7, 128, 131] {
        let f = fixture(&c, t, 0x77aa_0000 + t as u64);
        let dev = Device::Cpu;
        let row = c.conv_dim();
        let mixed = Tensor::from_vec(f.mixed.clone(), (t, row), &dev).unwrap();
        let dt = Tensor::from_vec(f.dt.clone(), (t, c.n_head), &dev).unwrap();
        let dt_bias = Tensor::from_vec(f.dt_bias.clone(), c.n_head, &dev).unwrap();
        let a = Tensor::from_vec(f.a.clone(), c.n_head, &dev).unwrap();
        let d_skip = Tensor::from_vec(f.d_skip.clone(), c.n_head, &dev).unwrap();

        // One shot.
        let mut st_pre = f.state0.clone();
        let y_pre = ssm_scan(c, &mixed, &dt, &dt_bias, &a, &d_skip, &mut st_pre)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Token by token through the step path.
        let mut st_inc = f.state0.clone();
        let mut y_inc = Vec::with_capacity(t * c.d_inner);
        for tok in 0..t {
            let m = mixed.narrow(0, tok, 1).unwrap();
            let d = dt.narrow(0, tok, 1).unwrap();
            let row_y =
                ssm_scan_step(c, &m, &d, &dt_bias, &a, &d_skip, &mut st_inc).unwrap();
            y_inc.extend_from_slice(&row_y);
        }

        assert_close(&format!("T={t} y"), &y_inc, &y_pre, 1e-6, 1e-6);
        assert_close(&format!("T={t} state"), &st_inc, &st_pre, 1e-6, 1e-6);
    }
}

/// A prefill split across two calls (a chunk boundary in the middle of a
/// prompt, i.e. cache continuation) must match one call, because the state
/// carries across.
#[test]
fn split_prefill_matches_single_prefill() {
    let c = small();
    let t = 97;
    let f = fixture(&c, t, 0x4242);
    let dev = Device::Cpu;
    let row = c.conv_dim();
    let mixed = Tensor::from_vec(f.mixed.clone(), (t, row), &dev).unwrap();
    let dt = Tensor::from_vec(f.dt.clone(), (t, c.n_head), &dev).unwrap();
    let dt_bias = Tensor::from_vec(f.dt_bias.clone(), c.n_head, &dev).unwrap();
    let a = Tensor::from_vec(f.a.clone(), c.n_head, &dev).unwrap();
    let d_skip = Tensor::from_vec(f.d_skip.clone(), c.n_head, &dev).unwrap();

    let mut st_one = f.state0.clone();
    let y_one = ssm_scan(c, &mixed, &dt, &dt_bias, &a, &d_skip, &mut st_one)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mut st_split = f.state0.clone();
    let mut y_split = Vec::new();
    for (start, len) in [(0usize, 40usize), (40, 33), (73, 24)] {
        let m = mixed.narrow(0, start, len).unwrap();
        let d = dt.narrow(0, start, len).unwrap();
        let y = ssm_scan(c, &m, &d, &dt_bias, &a, &d_skip, &mut st_split)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        y_split.extend_from_slice(&y);
    }
    assert_close("y", &y_split, &y_one, 1e-6, 1e-6);
    assert_close("state", &st_split, &st_one, 1e-6, 1e-6);
}

// ─── epilogue / parameter paths ─────────────────────────────────────────────

/// `d_skip = 0` must reduce to the pure `C·h` readout (this is the case where
/// llama.cpp's `y = y + cur*ssm_d` is a no-op, and the one an implementation
/// that forgot the D term would accidentally pass — hence checking both).
#[test]
fn zero_d_skip_drops_the_skip_term() {
    let c = small();
    let mut f = fixture(&c, 33, 0x1111);
    f.d_skip.iter_mut().for_each(|v| *v = 0.0);
    let (yc, _, yn, _) = run_both(&c, &f);
    assert_close("y (D=0)", &yc, &yn, 1e-5, 1e-6);

    // And with a large D the skip term must be genuinely present: flipping D's
    // sign must change y by 2·D·x.
    let mut f2 = fixture(&c, 33, 0x1111);
    f2.d_skip.iter_mut().for_each(|v| *v = 0.5);
    let (y_pos, _, _, _) = run_both(&c, &f2);
    f2.d_skip.iter_mut().for_each(|v| *v = -0.5);
    let (y_neg, _, _, _) = run_both(&c, &f2);
    for i in 0..y_pos.len() {
        let h = (i % c.d_inner) / c.head_dim();
        let want = 2.0 * 0.5 * f2.x[i];
        assert!(
            (y_pos[i] - y_neg[i] - want).abs() <= 1e-4 + 1e-4 * want.abs(),
            "D sensitivity at {i}: {} - {} != {want}",
            y_pos[i],
            y_neg[i]
        );
        let _ = h;
    }
}

/// A strongly-decaying head (`a` very negative) drives the chunk decay to
/// underflow; the result must be finite and still match — the reason `L` is
/// built from `exp(cum[i] − cum[j])` rather than a quotient of two decays.
#[test]
fn strong_decay_underflow_stays_finite_and_exact() {
    let c = small();
    let mut f = fixture(&c, 128, 0x2222);
    // dt ~ 1, |a| up to ~200 ⇒ dt·a ≈ −200 per token, so exp(cum) underflows
    // within a couple of tokens.
    f.a.iter_mut().for_each(|v| *v = -180.0);
    let (yc, sc, yn, sn) = run_both(&c, &f);
    assert!(yc.iter().all(|v| v.is_finite()), "y not finite");
    assert!(sc.iter().all(|v| v.is_finite()), "state not finite");
    assert_close("y (strong decay)", &yc, &yn, 1e-5, 1e-5);
    assert_close("state (strong decay)", &sc, &sn, 1e-5, 1e-5);
}

/// `a = 0` is the no-decay limit (every token contributes equally); it must
/// also be exact, and it is the case most likely to expose a missing `dt`
/// scale in the state write.
#[test]
fn zero_decay_accumulates_every_token() {
    let c = small();
    let mut f = fixture(&c, 40, 0x3333);
    f.a.iter_mut().for_each(|v| *v = 0.0);
    let (yc, sc, yn, sn) = run_both(&c, &f);
    assert_close("y (a=0)", &yc, &yn, 1e-5, 1e-6);
    assert_close("state (a=0)", &sc, &sn, 1e-5, 1e-6);
}

// ─── shapes / errors ────────────────────────────────────────────────────────

/// The returned tensor keeps the documented `[t, d_inner]` f32 shape and
/// device, and the state length is the documented product.
#[test]
fn output_shape_and_state_length() {
    let c = nemotron();
    assert_eq!(c.head_dim(), 128);
    assert_eq!(c.conv_dim(), 64 * 128 + 2 * 8 * 64);
    assert_eq!(c.state_len(), 64 * 128 * 64);

    let f = fixture(&small(), 5, 0x4444);
    let sc = small();
    let dev = Device::Cpu;
    let mixed = Tensor::from_vec(f.mixed.clone(), (5, sc.conv_dim()), &dev).unwrap();
    let dt = Tensor::from_vec(f.dt.clone(), (5, sc.n_head), &dev).unwrap();
    let dt_bias = Tensor::from_vec(f.dt_bias.clone(), sc.n_head, &dev).unwrap();
    let a = Tensor::from_vec(f.a.clone(), sc.n_head, &dev).unwrap();
    let d_skip = Tensor::from_vec(f.d_skip.clone(), sc.n_head, &dev).unwrap();
    let mut st = f.state0.clone();
    let y = ssm_scan(sc, &mixed, &dt, &dt_bias, &a, &d_skip, &mut st).unwrap();
    assert_eq!(y.dims(), &[5, sc.d_inner]);
    assert_eq!(y.dtype(), DType::F32);
}

/// A state buffer of the wrong length is a programming error at the call site,
/// not something to index past.
#[test]
fn wrong_state_length_is_an_error() {
    let sc = small();
    let f = fixture(&sc, 4, 0x5555);
    let dev = Device::Cpu;
    let mixed = Tensor::from_vec(f.mixed.clone(), (4, sc.conv_dim()), &dev).unwrap();
    let dt = Tensor::from_vec(f.dt.clone(), (4, sc.n_head), &dev).unwrap();
    let dt_bias = Tensor::from_vec(f.dt_bias.clone(), sc.n_head, &dev).unwrap();
    let a = Tensor::from_vec(f.a.clone(), sc.n_head, &dev).unwrap();
    let d_skip = Tensor::from_vec(f.d_skip.clone(), sc.n_head, &dev).unwrap();

    let mut st = vec![0f32; sc.state_len() - 1];
    assert!(ssm_scan(sc, &mixed, &dt, &dt_bias, &a, &d_skip, &mut st).is_err());

    let mut st = vec![0f32; sc.state_len() + 1];
    assert!(ssm_scan(sc, &mixed, &dt, &dt_bias, &a, &d_skip, &mut st).is_err());
}

/// A `mixed` whose width is not `d_inner + 2·n_group·d_state` is rejected
/// rather than silently reading the `B` block as `C`.
#[test]
fn wrong_mixed_width_is_an_error() {
    let sc = small();
    let f = fixture(&sc, 3, 0x6666);
    let dev = Device::Cpu;
    let bad = Tensor::from_vec(f.mixed[..3 * (sc.conv_dim() - 1)].to_vec(), (3, sc.conv_dim() - 1), &dev)
        .unwrap();
    let dt = Tensor::from_vec(f.dt.clone(), (3, sc.n_head), &dev).unwrap();
    let dt_bias = Tensor::from_vec(f.dt_bias.clone(), sc.n_head, &dev).unwrap();
    let a = Tensor::from_vec(f.a.clone(), sc.n_head, &dev).unwrap();
    let d_skip = Tensor::from_vec(f.d_skip.clone(), sc.n_head, &dev).unwrap();
    let mut st = f.state0.clone();
    assert!(ssm_scan(sc, &bad, &dt, &dt_bias, &a, &d_skip, &mut st).is_err());
}

/// `dt` must be `[t, n_head]` — a transposed or flattened `dt` would otherwise
/// silently scan the wrong token's decay.
#[test]
fn wrong_dt_shape_is_an_error() {
    let sc = small();
    let f = fixture(&sc, 6, 0x7777);
    let dev = Device::Cpu;
    let mixed = Tensor::from_vec(f.mixed.clone(), (6, sc.conv_dim()), &dev).unwrap();
    let dt_bad = Tensor::from_vec(f.dt.clone(), (sc.n_head, 6), &dev).unwrap();
    let dt_bias = Tensor::from_vec(f.dt_bias.clone(), sc.n_head, &dev).unwrap();
    let a = Tensor::from_vec(f.a.clone(), sc.n_head, &dev).unwrap();
    let d_skip = Tensor::from_vec(f.d_skip.clone(), sc.n_head, &dev).unwrap();
    let mut st = f.state0.clone();
    assert!(ssm_scan(sc, &mixed, &dt_bad, &dt_bias, &a, &d_skip, &mut st).is_err());
}

/// The state is advanced in place and never reallocated, so the caller's
/// buffer identity is what carries the recurrence: calling twice with the
/// same buffer must differ from two independent calls on the original state.
#[test]
fn state_advances_in_place() {
    let sc = small();
    let f = fixture(&sc, 16, 0x8888);
    let dev = Device::Cpu;
    let mixed = Tensor::from_vec(f.mixed.clone(), (16, sc.conv_dim()), &dev).unwrap();
    let dt = Tensor::from_vec(f.dt.clone(), (16, sc.n_head), &dev).unwrap();
    let dt_bias = Tensor::from_vec(f.dt_bias.clone(), sc.n_head, &dev).unwrap();
    let a = Tensor::from_vec(f.a.clone(), sc.n_head, &dev).unwrap();
    let d_skip = Tensor::from_vec(f.d_skip.clone(), sc.n_head, &dev).unwrap();

    let mut st = f.state0.clone();
    ssm_scan(sc, &mixed, &dt, &dt_bias, &a, &d_skip, &mut st).unwrap();
    assert_ne!(st, f.state0, "state must advance in place");

    // Same inputs from the same starting state give the same result — the scan
    // is a pure function of `(inputs, state)` even though it mutates in place.
    let mut st2 = f.state0.clone();
    ssm_scan(sc, &mixed, &dt, &dt_bias, &a, &d_skip, &mut st2).unwrap();
    assert_eq!(st, st2, "same inputs, same state");

    // Re-running from the *advanced* state must differ from the first pass:
    // that is what proves the state, not the inputs, carries the history.
    // (It need not differ from a second identical pass over a converging
    // sequence — with strong decay the recurrence has a fixed point, so
    // "not idempotent" would be the wrong assertion here; instead compare
    // against a zero start, which cannot coincide.)
    // The carried state must matter. Use a *weak* decay for this: with the
    // default fixture's `dt·a ≈ −O(10)` per token, a 16-token block drives
    // `exp(cum_end)` to ~1e-70 and any initial state is erased entirely, so a
    // zero start would legitimately land on the same state. Weak decay keeps
    // the initial state's contribution alive, which is the property under test.
    let mut fw = fixture(&sc, 16, 0x8888);
    fw.a.iter_mut().for_each(|v| *v = -0.01);
    let m = Tensor::from_vec(fw.mixed.clone(), (16, sc.conv_dim()), &dev).unwrap();
    let d = Tensor::from_vec(fw.dt.clone(), (16, sc.n_head), &dev).unwrap();
    let a_weak = Tensor::from_vec(fw.a.clone(), sc.n_head, &dev).unwrap();
    let mut st_a = fw.state0.clone();
    ssm_scan(sc, &m, &d, &dt_bias, &a_weak, &d_skip, &mut st_a).unwrap();
    let mut st_zero = vec![0f32; sc.state_len()];
    ssm_scan(sc, &m, &d, &dt_bias, &a_weak, &d_skip, &mut st_zero).unwrap();
    assert_ne!(st_zero, st_a, "the carried state must matter");
}

/// Device parity at the **real Nemotron geometry**, where `head_dim` (128) and
/// `d_state` (64) differ. This is the case that catches a readout matmul
/// contracting the wrong axis: with equal widths the shapes happen to fit and
/// the error is silent, whereas at 128 vs 64 the matmul cannot even run.
#[cfg(feature = "opencl")]
#[test]
fn opencl_matches_cpu_nemotron_geometry() {
    let dev = match Device::new_opencl(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping opencl nemotron parity: no device ({e})");
            return;
        }
    };
    let c = nemotron();
    assert_eq!(c.head_dim(), 128);
    assert_eq!(c.d_state, 64);
    // Multi-chunk lengths (2, 3 and 8 chunks) plus a ragged tail.
    for &t in &[256usize, 384, 1024, 131] {
        let f = fixture(&c, t, 0x0e0_0000 + t as u64);
        let cpu = Device::Cpu;
        let (row, n) = (c.conv_dim(), c.n_head);
        let mixed = Tensor::from_vec(f.mixed.clone(), (t, row), &cpu).unwrap();
        let dt = Tensor::from_vec(f.dt.clone(), (t, n), &cpu).unwrap();
        let dt_bias = Tensor::from_vec(f.dt_bias.clone(), n, &cpu).unwrap();
        let a = Tensor::from_vec(f.a.clone(), n, &cpu).unwrap();
        let d_skip = Tensor::from_vec(f.d_skip.clone(), n, &cpu).unwrap();

        let mut st_cpu = f.state0.clone();
        let y_cpu = ssm_scan(c, &mixed, &dt, &dt_bias, &a, &d_skip, &mut st_cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let mut st_dev = f.state0.clone();
        let y_dev = ssm_scan(
            c,
            &mixed.to_device(&dev).unwrap(),
            &dt.to_device(&dev).unwrap(),
            &dt_bias.to_device(&dev).unwrap(),
            &a.to_device(&dev).unwrap(),
            &d_skip.to_device(&dev).unwrap(),
            &mut st_dev,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        // Both the output *and* the carried state: a wrong chunk-end decay
        // shows up in the state even when `y` still looks close.
        assert_close_scaled(&format!("nemotron opencl T={t} y"), &y_dev, &y_cpu, 1e-4);
        assert_close_scaled(
            &format!("nemotron opencl T={t} state"),
            &st_dev,
            &st_cpu,
            1e-4,
        );
    }
}

/// Strong decay over a long chunk: `dt·a = −1` on every row drives
/// `cum[i] − cum[j]` far positive for `j > i`, so an unmasked `exp` overflows to
/// `inf` and the triangular mask turns `inf * 0` into NaN. The result must be
/// finite everywhere and still match the CPU path.
#[cfg(feature = "opencl")]
#[test]
fn opencl_strong_decay_is_nan_free() {
    let dev = match Device::new_opencl(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping opencl strong-decay: no device ({e})");
            return;
        }
    };
    // A 128-row chunk (the full Nemotron chunk length) with `dt·a == −1` on
    // every token, which is the worst case for the intra-chunk exponent.
    let c = small();
    let mut f = fixture(&c, 128, 0x5ea1_0000);
    for v in f.a.iter_mut() {
        *v = -1.0;
    }
    // Force `dt == 1` so `dt·a == −1` exactly, and zero the bias, making the
    // cumsum a clean −i ramp. `softplus(x) = ln(1 + e^x) = 1` solves to
    // `x = ln(e − 1) = 0.5413…`.
    f.dt_bias.iter_mut().for_each(|v| *v = 0.0);
    let one: f32 = (std::f32::consts::E - 1.0).ln();
    debug_assert!((one.exp() + 1.0).ln() - 1.0 < 1e-6);
    f.dt.iter_mut().for_each(|v| *v = one);

    let cpu = Device::Cpu;
    let (row, n) = (c.conv_dim(), c.n_head);
    let mixed = Tensor::from_vec(f.mixed.clone(), (128, row), &cpu).unwrap();
    let dt = Tensor::from_vec(f.dt.clone(), (128, n), &cpu).unwrap();
    let dt_bias = Tensor::from_vec(f.dt_bias.clone(), n, &cpu).unwrap();
    let a = Tensor::from_vec(f.a.clone(), n, &cpu).unwrap();
    let d_skip = Tensor::from_vec(f.d_skip.clone(), n, &cpu).unwrap();

    let mut st_dev = f.state0.clone();
    let y_dev = ssm_scan(
        c,
        &mixed.to_device(&dev).unwrap(),
        &dt.to_device(&dev).unwrap(),
        &dt_bias.to_device(&dev).unwrap(),
        &a.to_device(&dev).unwrap(),
        &d_skip.to_device(&dev).unwrap(),
        &mut st_dev,
    )
    .unwrap()
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();

    let bad = y_dev.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(bad, 0, "{bad}/{} device outputs are NaN/inf", y_dev.len());
    let bad_st = st_dev.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(bad_st, 0, "{bad_st} device state entries are NaN/inf");

    let mut st_cpu = f.state0.clone();
    let y_cpu = ssm_scan(c, &mixed, &dt, &dt_bias, &a, &d_skip, &mut st_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_close_scaled("strong decay y (device vs cpu)", &y_dev, &y_cpu, 1e-4);
    assert_close_scaled(
        "strong decay state (device vs cpu)",
        &st_dev,
        &st_cpu,
        1e-4,
    );
}

// ─── f32 vs f64 conditioning ────────────────────────────────────────────────

/// The recurrence evaluated in f64 — the "true" value, so a chunked-vs-naive
/// gap can be attributed either to f32 round-off (a benign, tiny relative
/// error) or to a wrong association (a gap that survives the f64 control).
fn scan_f64(
    c: &ScanConfig,
    f: &Fixture,
) -> (Vec<f64>, Vec<f64>) {
    let (hd, ds, ng, n) = (c.head_dim(), c.d_state, c.n_group, c.n_head);
    let mut st: Vec<f64> = f.state0.iter().map(|&v| v as f64).collect();
    let mut y = vec![0f64; f.t * c.d_inner];
    for h in 0..n {
        let g = c.group_of(h);
        let s = &mut st[h * hd * ds..(h + 1) * hd * ds];
        let a_h = f.a[h] as f64;
        let d_h = f.d_skip[h] as f64;
        for tok in 0..f.t {
            let raw = f.dt[tok * n + h] as f64 + f.dt_bias[h] as f64;
            let dt_h = if raw > 20.0 { raw } else { (1.0 + raw.exp()).ln() };
            let da = (dt_h * a_h).exp();
            let bv = &f.b[tok * ng * ds + g * ds..][..ds];
            let cv = &f.c[tok * ng * ds + g * ds..][..ds];
            let xv = &f.x[tok * c.d_inner + h * hd..][..hd];
            for i in 0..hd {
                let xdt = xv[i] as f64 * dt_h;
                let mut acc = 0f64;
                for sj in 0..ds {
                    let v = s[i * ds + sj] * da + bv[sj] as f64 * xdt;
                    s[i * ds + sj] = v;
                    acc += v * cv[sj] as f64;
                }
                y[tok * c.d_inner + h * hd + i] = acc + d_h * xv[i] as f64;
            }
        }
    }
    (y, st)
}

/// Chunked (f32) must agree with the recurrence evaluated in f64 to within
/// f32 round-off, so any gap that exceeds that is a formula error rather than
/// a precision effect. Reported as relative error on the whole vector.
#[test]
fn chunked_agrees_with_f64_recurrence() {
    let base = small();
    for &q in &[1usize, 4, 128, 256] {
        for &t in &[1usize, 7, 128, 131] {
            let c = ScanConfig { chunk: q, ..base };
            let f = fixture(&c, t, 0xdead_0000 ^ (q as u64) << 8 ^ t as u64);
            let (y32, _, _, _) = run_both(&c, &f);
            let (y64, _) = scan_f64(&c, &f);
            let scale = y64.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-30);
            let mut worst = 0f64;
            for (a, b) in y32.iter().zip(y64.iter()) {
                worst = worst.max((*a as f64 - b).abs());
            }
            let rel = worst / scale;
            assert!(
                rel < 2e-4,
                "Q={q} T={t}: |y32−y64|_inf / |y|_inf = {rel:e} (abs {worst:e}, scale {scale:e})"
            );
        }
    }
}

// ─── device parity (OpenCL, when built with --features opencl) ──────────────

/// The device path must reproduce the CPU path on real hardware: same
/// association, same inclusive-cumsum decay, only backend rounding between
/// them.  Compiled only with `--features opencl`; skipped otherwise so the
/// default suite does not depend on a GPU being present.
#[cfg(feature = "opencl")]
#[test]
fn opencl_matches_cpu() {
    let dev = match Device::new_opencl(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping opencl parity: no device ({e})");
            return;
        }
    };
    let c = small();
    for &t in &[1usize, 128, 131, 512] {
        let f = fixture(&c, t, 0x0c1_0000 + t as u64);
        let row = c.conv_dim();
        let cpu = Device::Cpu;
        let mixed = Tensor::from_vec(f.mixed.clone(), (t, row), &cpu).unwrap();
        let dt = Tensor::from_vec(f.dt.clone(), (t, c.n_head), &cpu).unwrap();
        let dt_bias = Tensor::from_vec(f.dt_bias.clone(), c.n_head, &cpu).unwrap();
        let a = Tensor::from_vec(f.a.clone(), c.n_head, &cpu).unwrap();
        let d_skip = Tensor::from_vec(f.d_skip.clone(), c.n_head, &cpu).unwrap();

        let mut st_cpu = f.state0.clone();
        let y_cpu = ssm_scan(c, &mixed, &dt, &dt_bias, &a, &d_skip, &mut st_cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let mut st_dev = f.state0.clone();
        let y_dev = ssm_scan(
            c,
            &mixed.to_device(&dev).unwrap(),
            &dt.to_device(&dev).unwrap(),
            &dt_bias.to_device(&dev).unwrap(),
            &a.to_device(&dev).unwrap(),
            &d_skip.to_device(&dev).unwrap(),
            &mut st_dev,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        assert_close_scaled(&format!("opencl T={t} y"), &y_dev, &y_cpu, 1e-4);
        assert_close_scaled(&format!("opencl T={t} state"), &st_dev, &st_cpu, 1e-4);
    }
}
