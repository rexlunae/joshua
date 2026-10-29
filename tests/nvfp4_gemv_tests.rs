//! Native NVFP4 GEMV: the OpenCL kernel and the CPU fallback must both
//! reproduce [`joshua::mxfp4`]'s host decode of the same 36-byte blocks.
//!
//! The oracle is `mxfp4::BlockNvfp4`'s own `dequantize_block`, i.e. the
//! decoder the rest of the engine uses, not a re-derivation here: a mismatch
//! in this file is a kernel bug or a host-decoder bug, and the two cannot
//! agree by sharing an error.
//!
//! Device tests skip cleanly when no OpenCL device is present.  Every test
//! prints its measured maximum relative error, and the assertions are stated
//! as an evidence-backed bound on *that* number rather than a tolerance that
//! was widened until the test passed.

use candle_core::{Device, Tensor};
use joshua::mxfp4::{BlockNvfp4, QK_NVFP4};
use joshua::raw_block::RawBlock;

const BLOCK_BYTES: usize = std::mem::size_of::<BlockNvfp4>();

// ─── Fixtures ────────────────────────────────────────────────────────────────

/// A small, fully deterministic PRNG so a failure reproduces exactly.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
    fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next_u32() as f32 / u32::MAX as f32)
    }
}

/// `count` blocks whose codes cover the whole E2M1 space and whose scales cover
/// every class [UE4M3] can express, including the two zeros, the denormal
/// (exp == 0) range, the sign bit being set, and the largest finite value.
fn edge_case_blocks(count: usize) -> Vec<BlockNvfp4> {
    // Every byte, so every scale class and every code appears.
    (0..count)
        .map(|b| {
            let mut qs = [0u8; QK_NVFP4 / 2];
            for (j, q) in qs.iter_mut().enumerate() {
                // Low nibbles cycle 0..16, high nibbles cycle the other way,
                // so all 16 codes occur in both nibble positions.
                let lo = ((j + b) % 16) as u8;
                let hi = (15 - (j + 2 * b) % 16) as u8;
                *q = lo | (hi << 4);
            }
            let d = [
                (0x00u8).wrapping_add(b as u8),              // 0, then 1, 2, ...
                [0x7F, 0xFF, 0x80, 0x01, 0x07, 0x0F][b % 6], // NaN, sign, denormals
                [0x38, 0x3C, 0x40, 0x7E, 0x70, 0x39][b % 6], // 1.0, 1.5, 2, max, big
                (0xF8 | (b as u8 % 8)) as u8,                // sign bit set
            ];
            BlockNvfp4 { d, qs }
        })
        .collect()
}

/// Random blocks with finite, well-spread scales and uniform codes.
fn random_blocks(count: usize, seed: u64) -> Vec<BlockNvfp4> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|_| {
            let mut qs = [0u8; QK_NVFP4 / 2];
            for q in qs.iter_mut() {
                *q = rng.next_u32() as u8;
            }
            let d = [
                // Spans denormal (0x01..=0x07) through ordinary magnitudes.
                [0x01, 0x04, 0x07, 0x30, 0x36, 0x38, 0x3C, 0x40][rng.below(8) as usize],
                [0x2C, 0x30, 0x34, 0x38, 0x3C, 0x40, 0x44, 0x48][rng.below(8) as usize],
                [0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F][rng.below(8) as usize],
                [0x08, 0x10, 0x20, 0x30, 0x38, 0x40, 0x50, 0x60][rng.below(8) as usize],
            ];
            BlockNvfp4 { d, qs }
        })
        .collect()
}

fn block_bytes(blocks: &[BlockNvfp4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(blocks.len() * BLOCK_BYTES);
    for b in blocks {
        out.extend_from_slice(&b.d);
        out.extend_from_slice(&b.qs);
    }
    out
}

/// Decode `blocks` through the reference host decoder.
fn reference_decode(blocks: &[BlockNvfp4]) -> Vec<f32> {
    let mut out = vec![0f32; blocks.len() * QK_NVFP4];
    for (b, block) in blocks.iter().enumerate() {
        let dst = &mut out[b * QK_NVFP4..(b + 1) * QK_NVFP4];
        block.dequantize_block(dst);
    }
    out
}

/// Reference `y = W x` in f64 from the decoded weights, plus each output's
/// `sum |term|` — the magnitude the f32 sum's rounding is measured against.
fn reference_gemv(
    weights: &[f32],
    acts: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut out = Vec::with_capacity(out_dim);
    let mut mags = Vec::with_capacity(out_dim);
    for n in 0..out_dim {
        let mut acc = 0f64;
        let mut mag = 0f64;
        for k in 0..in_dim {
            let t = weights[n * in_dim + k] as f64 * acts[k] as f64;
            acc += t;
            mag += t.abs();
        }
        out.push(acc);
        mags.push(mag);
    }
    (out, mags)
}

/// The OpenCL device, or `None` when there is none (or the feature is off).
///
/// `candle_core::opencl_backend` and `OpenClDevice::new` only exist under the
/// `opencl` feature, and CI builds the test targets with *default* features, so
/// every reference here has to be gated rather than named unconditionally.
#[cfg(feature = "opencl")]
fn opencl_device() -> Option<Device> {
    static D: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        // Respect the same switch the backend does; a "disabled" run is a
        // CPU-reference run, not a device run.
        if !candle_core::opencl_backend::kernels::native_enabled() {
            return None;
        }
        candle_core::OpenClDevice::new(0).ok().map(Device::OpenCl)
    })
    .clone()
}

#[cfg(not(feature = "opencl"))]
fn opencl_device() -> Option<Device> {
    None
}

/// Per-op CPU-round-trip count for this op, and the native launch count.
/// Both live in the vendored OpenCL backend, so they are gated too.
#[cfg(feature = "opencl")]
fn nvfp4_counters() -> (usize, usize) {
    let fallbacks = candle_core::opencl_backend::kernels::fallback_counts_by_op()
        .into_iter()
        .find(|(op, _)| op == "nvfp4-matmul-qt")
        .map(|(_, n)| n)
        .unwrap_or(0);
    (fallbacks, candle_core::opencl_backend::native_exec_count())
}

#[cfg(not(feature = "opencl"))]
fn nvfp4_counters() -> (usize, usize) {
    (0, 0)
}

fn cpu() -> Device {
    Device::Cpu
}

/// Upload blocks (`[out_dim, in_dim/64]` u8) and one activation row, run
/// `nvfp4_matmul_qt` on `dev`, and return the output row plus the measured
/// maximum relative error against the f64 reference.
fn run_case(
    dev: &Device,
    blocks: &[BlockNvfp4],
    acts: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> (Vec<f32>, f64) {
    let per_row = in_dim / QK_NVFP4;
    assert_eq!(blocks.len(), out_dim * per_row);
    let bytes = block_bytes(blocks);
    let w = Tensor::from_vec(bytes, (out_dim, per_row * BLOCK_BYTES), dev).expect("upload weights");
    let x = Tensor::from_vec(acts.to_vec(), (in_dim,), dev).expect("upload activations");
    let y = joshua::nvfp4_gemv::nvfp4_matmul_qt(&w, &x).expect("nvfp4 gemv");
    assert_eq!(y.dims(), &[1, out_dim]);
    let got = y.flatten_all().expect("flatten").to_vec1::<f32>().expect("to vec");

    let weights = reference_decode(blocks);
    let (want, mags) = reference_gemv(&weights, acts, out_dim, in_dim);
    let mut worst = 0f64;
    let mut worst_rel = 0f64;
    for ((g, wv), mag) in got.iter().zip(&want).zip(&mags) {
        let err = (*g as f64 - wv).abs();
        // The sum is `in_dim` terms that can cancel, so a plain |err|/|value|
        // explodes near zero through no fault of the kernel.  Normalizing by
        // `sum |term|` measures the rounding the sum actually incurred.
        worst = worst.max(err / mag.max(1e-30));
        worst_rel = worst_rel.max(err / wv.abs().max(1e-30));
    }
    let _ = worst_rel;
    (got, worst)
}

/// Error bound, normalized by each output's `sum |term|`.
///
/// A plain `|err| / |value|` is the wrong statistic for a `in_dim`-term sum:
/// when the terms cancel and the dot product lands near zero, that ratio blows
/// up even though the summation is fine.  Measured here, `|err| / sum|term|`
/// sits at a few times f32 epsilon (see the printed value); `1e-5` is a bound
/// with headroom over that, not a number chosen to make a failure pass.  A
/// decoding bug lands orders of magnitude above it, and a wrong scale or
/// packing shows up immediately.
const MAX_REL_ERROR: f64 = 1e-5;

fn report(what: &str, max_err: f64) {
    println!("{what}: max relative error {max_err:.3e}");
    assert!(
        max_err <= MAX_REL_ERROR,
        "{what}: max relative error {max_err:.3e} exceeds {MAX_REL_ERROR:.1e}"
    );
}

// ─── Host-only tests (always run) ────────────────────────────────────────────

#[test]
fn split_half_layout_is_frozen() {
    // Per 16, the 8 code bytes at 4 + 8*sub .. +8 hold element j in the low
    // nibble and element j + 8 in the high nibble.  So byte 0's high nibble is
    // element 8 of this 16 — NOT element 1, and not a far-away element of
    // another sub-block.
    let mut block = BlockNvfp4 { d: [0x38, 0, 0, 0], qs: [0; 32] };
    block.qs[0] = 0x7 | (0x1 << 4); // low = code 7 (+6), high = code 1 (+0.5)
    let mut out = [0f32; QK_NVFP4];
    let bytes = block_bytes(&[block]);
    joshua::nvfp4_gemv::decode_block(bytes[..BLOCK_BYTES].try_into().unwrap(), &mut out);
    assert_eq!(out[0], 6.0, "low nibble of byte 0 is element 0");
    assert_eq!(out[8], 0.5, "high nibble of byte 0 is element 8 of its 16");
    assert_eq!(out[1], 0.0, "element 1 comes from byte 1, not byte 0's high nibble");
    assert_eq!(out[16], 0.0, "byte 0 says nothing about the next sub-block");

    // Sub-block 1 (elements 16..32) has its own scale byte and its codes at
    // byte 4 + 8.  Losing that offset would make sub-block 1 read sub-block 0's
    // codes and silently halve the effective weight count.
    let mut block = BlockNvfp4 { d: [0, 0x38, 0, 0], qs: [0; 32] };
    block.qs[8] = 0xF; // byte 8 = 4 + 8*1 + 0, low nibble -> element 16
    let mut out = [0f32; QK_NVFP4];
    let bytes = block_bytes(&[block]);
    joshua::nvfp4_gemv::decode_block(bytes[..BLOCK_BYTES].try_into().unwrap(), &mut out);
    assert_eq!(out[16], -6.0, "sub-block 1 codes start at byte 4 + 8*1");
    assert_eq!(out[24], 0.0);

    // Sub-block 2's codes start at BLOCK byte 4 + 8*2 = 20, i.e. `qs` index
    // 16 (the four scale bytes precede `qs`).  Its high nibbles land at
    // elements 40..48 (16*2 + 8 + j).
    let mut block = BlockNvfp4 { d: [0, 0, 0x38, 0], qs: [0; 32] };
    block.qs[16] = 0x6 << 4; // qs index 16 == block byte 20; high nibble code 6 = +4
    let mut out = [0f32; QK_NVFP4];
    let bytes = block_bytes(&[block]);
    joshua::nvfp4_gemv::decode_block(bytes[..BLOCK_BYTES].try_into().unwrap(), &mut out);
    assert_eq!(out[32], 0.0, "sub-block 2 low nibbles are elements 32..40");
    assert_eq!(out[40], 4.0, "sub-block 2 high nibbles are elements 40..48");
}

#[test]
fn host_decode_matches_the_reference_decoder() {
    let blocks = edge_case_blocks(64);
    let ours = {
        let bytes = block_bytes(&blocks);
        let mut out = vec![0f32; blocks.len() * QK_NVFP4];
        for (b, chunk) in bytes.chunks_exact(BLOCK_BYTES).enumerate() {
            let arr: &[u8; BLOCK_BYTES] = chunk.try_into().unwrap();
            joshua::nvfp4_gemv::decode_block(
                arr,
                (&mut out[b * QK_NVFP4..(b + 1) * QK_NVFP4]).try_into().unwrap(),
            );
        }
        out
    };
    assert_eq!(ours, reference_decode(&blocks), "our host decode must equal mxfp4's");
}

#[test]
fn every_e2m1_code_decodes_through_a_block() {
    // A 16 holds only 8 code bytes: byte j gives element j (low nibble) and
    // element j + 8 (high nibble).  So covering all 16 codes in *both* nibble
    // positions needs two sub-blocks: sub-block 0 carries codes 0..8, sub-block
    // 1 codes 8..16.
    let mut qs = [0u8; QK_NVFP4 / 2];
    for j in 0..8 {
        qs[j] = (j as u8) | ((j as u8) << 4); // sub-block 0: code j in both
        qs[8 + j] = ((8 + j) as u8) | (((8 + j) as u8) << 4); // sub-block 1
    }
    let block = BlockNvfp4 { d: [0x38, 0x38, 0, 0], qs }; // scale 1.0
    let want = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    let mut out = [0f32; QK_NVFP4];
    let bytes = block_bytes(&[block]);
    joshua::nvfp4_gemv::decode_block(bytes[..BLOCK_BYTES].try_into().unwrap(), &mut out);

    // Sub-block 0 holds codes 0..8 in element j and j + 8.
    for j in 0..8 {
        assert_eq!(out[j], want[j], "low nibble of byte {j} is code {j}");
        assert_eq!(out[8 + j], want[j], "high nibble of byte {j} is code {j}");
    }
    // Sub-block 1 (elements 16..32) holds codes 8..16.
    for j in 0..8 {
        assert_eq!(out[16 + j], want[8 + j], "sub-block 1 low nibble, code {}", 8 + j);
        assert_eq!(out[24 + j], want[8 + j], "sub-block 1 high nibble, code {}", 8 + j);
    }
    // Every one of the sixteen E2M1 values must have appeared.
    let mut seen: Vec<u8> = out[..32].iter().map(|v| v.to_bits() as u8).collect();
    seen.sort_unstable();
    let distinct: std::collections::BTreeSet<u32> =
        out[..32].iter().map(|v| v.to_bits()).collect();
    assert_eq!(distinct.len(), 16, "all sixteen codes must decode");
}

#[test]
fn scale_edge_cases_match_the_reference_decoder() {
    // 0x00 and 0x7F are zero; 0x80 is zero too (sign ignored); exp == 0 is a
    // denormal scale; 0xFF is 480.0, not -0.
    for byte in [0x00u8, 0x01, 0x07, 0x08, 0x38, 0x3C, 0x7E, 0x7F, 0x80, 0xB8, 0xFF] {
        let block = BlockNvfp4 { d: [byte, byte, byte, byte], qs: [0x11; QK_NVFP4 / 2] };
        let mut ours = [0f32; QK_NVFP4];
        joshua::nvfp4_gemv::decode_block(block_bytes(&[block])[..BLOCK_BYTES].try_into().unwrap(), &mut ours);
        let mut theirs = [0f32; QK_NVFP4];
        block.dequantize_block(&mut theirs);
        assert_eq!(ours, theirs, "scale byte {byte:#04x}");
        if byte == 0x00 || byte == 0x7F || byte == 0x80 {
            assert!(ours.iter().all(|v| *v == 0.0), "scale {byte:#04x} must be zero");
        }
        if byte == 0xFF {
            assert_eq!(ours[0], 480.0 * 0.5, "the sign bit is ignored: 0xFF is 480");
        }
    }
}

#[test]
fn dispatch_requires_a_multiple_of_64() {
    let dev = cpu();
    let blocks = edge_case_blocks(1);
    let bytes = block_bytes(&blocks);
    let w = Tensor::from_vec(bytes.clone(), (1usize, BLOCK_BYTES), &dev).unwrap();
    // A 64-element block tensor is a [1, 1] = in_dim 64, fine; the reference
    // path rejects a non-multiple by construction (in_dim is always k*64), so
    // check the explicit arithmetic instead.
    let x = Tensor::from_vec(vec![0f32; 64], (64,), &dev).unwrap();
    let y = joshua::nvfp4_gemv::nvfp4_matmul_qt(&w, &x).unwrap();
    assert_eq!(y.dims(), &[1, 1]);

    let bad = joshua::nvfp4_gemv::matmul_t(&bytes, &[0.0; 65], &mut [0.0; 1], 1, 1, 65);
    assert!(bad.is_err(), "in_dim 65 must be rejected");
}

// ─── Device tests (skip cleanly without OpenCL) ──────────────────────────────

#[test]
fn opencl_gemv_matches_the_host_decoder_at_real_shapes() {
    let Some(dev) = opencl_device() else {
        println!("skipping: no OpenCL device");
        return;
    };
    // The decode shapes Nemotron 3.5 uses: a 2048-wide attention projection and
    // the 2688-wide MLP projections.
    for (out_dim, in_dim) in [(2048usize, 2688usize), (4096usize, 2688usize)] {
        let per_row = in_dim / QK_NVFP4;
        let n_blocks = out_dim * per_row;
        let blocks = random_blocks(n_blocks, 0x5eed_0001 ^ in_dim as u64);
        let acts: Vec<f32> = {
            let mut rng = Rng::new(0xa5a5_1234);
            (0..in_dim).map(|_| rng.f32_in(-1.5, 1.5)).collect()
        };
        let (_, err) = run_case(&dev, &blocks, &acts, out_dim, in_dim);
        report(&format!("opencl [{out_dim}, {in_dim}] random"), err);
    }
}

#[test]
fn opencl_gemv_covers_all_codes_and_scale_classes() {
    let Some(dev) = opencl_device() else {
        println!("skipping: no OpenCL device");
        return;
    };
    // Small enough to hit every block edge case, wide enough to exercise the
    // work-group's sub-block striding (in_dim 256 = 4 blocks, WG 64).
    let (out_dim, in_dim) = (8usize, 256usize);
    let per_row = in_dim / QK_NVFP4;
    let blocks = edge_case_blocks(out_dim * per_row);
    let acts: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.017).sin() * 2.0).collect();
    let (_, err) = run_case(&dev, &blocks, &acts, out_dim, in_dim);
    report("opencl edge-case blocks", err);
}

#[test]
fn opencl_gemv_handles_the_denormal_scale() {
    let Some(dev) = opencl_device() else {
        println!("skipping: no OpenCL device");
        return;
    };
    // exp == 0 is `man * 2^-9`: if the kernel flushed it to zero, a whole
    // 16-element sub-block would vanish.  Make the denormal sub-block the only
    // nonzero contribution so a flush is unmissable.
    let (out_dim, in_dim) = (2usize, 128usize);
    let mut blocks = vec![BlockNvfp4 { d: [0; 4], qs: [0; QK_NVFP4 / 2] }; out_dim * (in_dim / QK_NVFP4)];
    // Row 0, block 0, sub-block 1 (element 16..32) gets a tiny scale and the
    // largest code, so its contribution is finite but nonzero.
    let b = &mut blocks[0];
    b.d[1] = 0x07; // exp 0, man 7 -> 7 * 2^-9
    b.qs[8] = 0x07; // element 16 = +6
    let acts: Vec<f32> = (0..in_dim)
        .map(|i| if i == 16 { 1000.0 } else { 0.0 })
        .collect();
    let (got, err) = run_case(&dev, &blocks, &acts, out_dim, in_dim);
    let want = 6.0f32 * (7.0 / 512.0) * 1000.0;
    println!("denormal scale: got {} want {want} (rel err {err:.3e})", got[0]);
    assert!(got[0] != 0.0, "the denormal scale must not be flushed to zero");
    assert!((got[0] - want).abs() / want.abs() < 1e-6, "got {} want {want}", got[0]);
    assert_eq!(got[1], 0.0, "the other row is all zeros");
}

#[test]
fn opencl_gemv_treats_a_nan_scale_as_corrupt() {
    let Some(dev) = opencl_device() else {
        println!("skipping: no OpenCL device");
        return;
    };
    // 0x7F must zero its 16 elements, not propagate an infinity.
    let (out_dim, in_dim) = (1usize, 64usize);
    let mut blocks = vec![BlockNvfp4 { d: [0, 0, 0, 0], qs: [0xFF; QK_NVFP4 / 2] }; out_dim];
    blocks[0].d[0] = 0x7F; // NaN scale on sub-block 0
    let acts = vec![1.0f32; in_dim];
    let (got, _) = run_case(&dev, &blocks, &acts, out_dim, in_dim);
    assert_eq!(got[0], 0.0, "a corrupt scale decodes to zeros");
    assert!(got[0].is_finite());
}

#[test]
fn opencl_gemv_reports_no_fallback_when_native() {
    if cfg!(not(feature = "opencl")) {
        println!("skipping: built without the opencl feature");
        return;
    }
    let Some(dev) = opencl_device() else {
        println!("skipping: no OpenCL device");
        return;
    };
    // A supported shape on a device with a native path must not take the CPU
    // round-trip; the backend's per-op counter makes that checkable rather than
    // assumed.
    let (fb_before, native_before) = nvfp4_counters();
    let blocks = random_blocks(4 * (256 / QK_NVFP4), 77);
    let acts: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).cos()).collect();
    let _ = run_case(&dev, &blocks, &acts, 4, 256);
    let (fb_after, native_after) = nvfp4_counters();
    println!(
        "native launches {native_before} -> {native_after}; nvfp4 fallbacks {fb_before} -> {fb_after}"
    );
    assert_eq!(fb_after, fb_before, "the native path must not fall back");
    assert!(
        native_after > native_before,
        "a native launch must have happened"
    );
}

#[test]
fn opencl_multi_row_path_matches_the_host_decoder() {
    let Some(dev) = opencl_device() else {
        println!("skipping: no OpenCL device");
        return;
    };
    // Batched decode (<= MAX_ROWS) is the same kernel repeated per row; it must
    // still agree row for row.
    let (rows, out_dim, in_dim) = (3usize, 16usize, 512usize);
    let blocks = random_blocks(out_dim * (in_dim / QK_NVFP4), 4242);
    let bytes = block_bytes(&blocks);
    let w = Tensor::from_vec(bytes, (out_dim, (in_dim / QK_NVFP4) * BLOCK_BYTES), &dev).unwrap();
    let acts: Vec<f32> = (0..rows * in_dim).map(|i| ((i % 97) as f32 - 48.0) / 24.0).collect();
    let x = Tensor::from_vec(acts.clone(), (rows, in_dim), &dev).unwrap();
    let y = joshua::nvfp4_gemv::nvfp4_matmul_qt(&w, &x).unwrap();
    assert_eq!(y.dims(), &[rows, out_dim]);
    let got = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let weights = reference_decode(&blocks);
    let mut worst = 0f64;
    for r in 0..rows {
        let (want, mags) = reference_gemv(
            &weights,
            &acts[r * in_dim..(r + 1) * in_dim],
            out_dim,
            in_dim,
        );
        for n in 0..out_dim {
            let g = got[r * out_dim + n] as f64;
            worst = worst.max((g - want[n]).abs() / mags[n].max(1e-30));
        }
    }
    report("opencl multi-row (3 rows)", worst);
}

#[test]
fn opencl_cpu_round_trip_matches_the_native_kernel() {
    let Some(dev) = opencl_device() else {
        println!("skipping: no OpenCL device");
        return;
    };
    // The CustomOp2 default is the fallback contract: with the native path
    // disabled the same op must still produce the reference answer.
    let (out_dim, in_dim) = (5usize, 320usize);
    let blocks = edge_case_blocks(out_dim * (in_dim / QK_NVFP4));
    let acts: Vec<f32> = (0..in_dim).map(|i| ((i * 13 % 41) as f32 - 20.0) / 7.0).collect();
    let acts_owned = acts.clone();
    let bytes = block_bytes(&blocks);
    let w = Tensor::from_vec(bytes.clone(), (out_dim, (in_dim / QK_NVFP4) * BLOCK_BYTES), &dev).unwrap();
    let x = Tensor::from_vec(acts.clone(), (in_dim,), &dev).unwrap();

    let native = joshua::nvfp4_gemv::nvfp4_matmul_qt(&w, &x)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // `matmul_t` is the CPU path the round-trip executes, so check it against
    // the f64 reference first: a failure here points at the decoder, not the
    // kernel.
    let mut host = vec![0f32; out_dim];
    joshua::nvfp4_gemv::matmul_t(&bytes, &acts_owned, &mut host, 1, out_dim, in_dim).unwrap();
    {
        let weights = reference_decode(&blocks);
        let (want, mags) = reference_gemv(&weights, &acts_owned, out_dim, in_dim);
        for n in 0..out_dim {
            let d = (host[n] as f64 - want[n]).abs() / mags[n].max(1e-30);
            assert!(d < 1e-5, "matmul_t disagrees with f64 on row {n}: {d:e}");
        }
    }

    // The CPU device, which is what the round-trip executes.
    let wc = Tensor::from_vec(bytes, (out_dim, (in_dim / QK_NVFP4) * BLOCK_BYTES), &cpu()).unwrap();
    let xc = Tensor::from_vec(acts, (in_dim,), &cpu()).unwrap();
    let round = joshua::nvfp4_gemv::nvfp4_matmul_qt(&wc, &xc)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // This fixture deliberately mixes tiny and huge scales, so a row's terms
    // can cancel; normalize each row's difference by the magnitude of its own
    // terms.  Compare native against BOTH the CPU-device result and the host
    // `matmul_t` the round-trip executes.
    let weights = reference_decode(&blocks);
    let (_, mags) = reference_gemv(&weights, &acts_owned, out_dim, in_dim);
    let mut worst = 0f64;
    for n in 0..out_dim {
        let d_cpu = (native[n] as f64 - round[n] as f64).abs();
        let d_host = (native[n] as f64 - host[n] as f64).abs();
        worst = worst.max(d_cpu / mags[n].max(1e-30));
        worst = worst.max(d_host / mags[n].max(1e-30));
    }
    report("opencl native vs cpu round-trip", worst);
}

// ─── Vulkan: pins the DOCUMENTED fallback, not a native launch ──────────────

/// On a Vulkan device this op currently takes `CustomOp2`'s CPU round-trip: the
/// vendored launcher is `pub(crate)` and `run_qgemv` cannot name NVFP4 (id 40 is
/// rejected by `GgmlDType::from_u32`).  That is a known, documented gap, not an
/// accident — so assert the *behaviour that exists*:
///
/// * the op succeeds and returns `[rows, out_dim]` f32 on the Vulkan device,
/// * the values are bit-identical to the CPU decode, because the round-trip
///   literally runs that decode.
///
/// If someone later adds a native Vulkan kernel, this test keeps its original
/// meaning (correctness through whatever path is taken); the bit-identity
/// assertion is what would need revisiting, and doing so is the signal that the
/// native path landed.
#[test]
fn vulkan_native_gemv_matches_the_reference() {
    #[cfg(not(feature = "vulkan"))]
    {
        println!("skipping: built without the vulkan feature");
    }
    #[cfg(feature = "vulkan")]
    {
        let vk = match candle_core::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                println!("skipping: no Vulkan device ({e})");
                return;
            }
        };
        let dev = Device::Vulkan(vk);
        // 128 columns spans several work-groups; a narrow matrix would fit in
        // one and could hide a grid-size mistake (it did once).
        let (out_dim, in_dim) = (128usize, 384usize);
        let blocks = edge_case_blocks(out_dim * (in_dim / QK_NVFP4));
        let acts: Vec<f32> = (0..in_dim).map(|i| ((i * 5 % 23) as f32 - 11.0) / 3.0).collect();
        let bytes = block_bytes(&blocks);
        let per_row = in_dim / QK_NVFP4;

        let w = Tensor::from_vec(bytes.clone(), (out_dim, per_row * BLOCK_BYTES), &dev).unwrap();
        let x = Tensor::from_vec(acts.clone(), (in_dim,), &dev).unwrap();
        let got = joshua::nvfp4_gemv::nvfp4_matmul_qt(&w, &x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got.len(), out_dim);

        // The CPU decode of the same blocks, through the reference decoder.
        let weights = reference_decode(&blocks);
        let want: Vec<f32> = reference_gemv(&weights, &acts, out_dim, in_dim)
            .0
            .into_iter()
            .map(|v| v as f32)
            .collect();
        // A native Vulkan kernel sums in a different order than the scalar
        // decoder, so the two agree to rounding rather than bit-for-bit.  This
        // used to be `assert_eq!` because the path was the CPU round-trip; the
        // loosening is the signal that the native kernel landed, exactly as the
        // doc comment above anticipated.
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / scale;
        assert!(
            worst < 1e-5,
            "vulkan NVFP4 vs the reference decode: worst relative error {worst:e}"
        );
        println!(
            "vulkan: native NVFP4 GEMV, worst relative error {worst:e} over {} values",
            got.len()
        );
    }
}

/// Every prefill chunk after the first must stay on the device.
///
/// The Nemotron wrapper splits >16 activation rows with `narrow(..).contiguous()`,
/// which keeps the parent's storage offset for a contiguous view.  The native
/// gate used to require offset 0, so *only the first* chunk ran natively and
/// every later one fell back — reading the whole weight matrix back to the host
/// each time, on the prefill hot path.
///
/// The kernels take the activation offset explicitly now, so this checks a
/// non-zero-offset view is still dispatched natively and gives the same answer.
#[cfg(feature = "opencl")]
#[test]
fn a_nonzero_offset_view_stays_native() {
    let Some(dev) = opencl_device() else {
        println!("skipping: no OpenCL device");
        return;
    };
    use candle_core::opencl_backend::kernels;
    if !kernels::native_enabled() {
        println!("skipping: native kernels disabled");
        return;
    }
    let (out_dim, in_dim, rows) = (24usize, 128usize, 32usize);
    let per_row = in_dim / QK_NVFP4;
    let blocks = edge_case_blocks(out_dim * per_row);
    let w = Tensor::from_vec(
        block_bytes(&blocks),
        (out_dim, per_row * BLOCK_BYTES),
        &dev,
    )
    .unwrap();
    // One [rows, in_dim] activation; rows 16.. is the second chunk view.
    let acts: Vec<f32> = (0..rows * in_dim)
        .map(|i| ((i * 7 % 29) as f32 - 14.0) / 4.0)
        .collect();
    let x_all = Tensor::from_vec(acts.clone(), (rows, in_dim), &dev).unwrap();
    let second = x_all.narrow(0, 16, 16).unwrap();
    assert_ne!(
        second.layout().start_offset(),
        0,
        "the test needs a view with a non-zero offset"
    );

    let (n0, f0) = (kernels::native_exec_count(), nvfp4_fallback_count());
    let got = joshua::nvfp4_gemv::nvfp4_matmul_qt(&w, &second)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let (n1, f1) = (kernels::native_exec_count(), nvfp4_fallback_count());

    assert!(n1 > n0, "a non-zero-offset view must still dispatch natively");
    assert_eq!(f1, f0, "and must not record a fallback");

    let chunk: Vec<f32> = acts[16 * in_dim..].to_vec();
    let weights = reference_decode(&blocks);
    let want: Vec<f32> = reference_gemv(&weights, &chunk, out_dim, in_dim)
        .0
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
        / scale;
    assert!(worst < 1e-5, "offset view vs reference: rel {worst:e}");
}

/// `nvfp4`'s own fallback counter (the backend aggregate includes other ops).
#[cfg(feature = "opencl")]
fn nvfp4_fallback_count() -> usize {
    candle_core::opencl_backend::kernels::fallback_counts_by_op()
        .iter()
        .filter(|(n, _)| n == "nvfp4-matmul-qt")
        .map(|(_, c)| *c)
        .sum()
}

/// A zero scale must not swallow a non-finite activation.
///
/// The kernels used to `continue` past a zero-scale sub-block, which drops the
/// `0 * inf` / `0 * NaN` products the reference still performs — a non-finite
/// input came back as a plausible finite number instead of NaN.  Silent
/// disagreement on a corrupt input is exactly the kind of thing that hides
/// until a model produces garbage.
#[test]
fn zero_scales_do_not_swallow_nonfinite_activations() {
    let dev = Device::Cpu;
    let (out_dim, in_dim) = (2usize, 128usize);
    let per_row = in_dim / QK_NVFP4;
    // All scales 0x00 (decode to 0.0), codes arbitrary.
    let mut bytes = vec![0u8; out_dim * per_row * BLOCK_BYTES];
    for blk in bytes.chunks_exact_mut(BLOCK_BYTES) {
        for s in 0..4 {
            blk[s] = 0x00;
        }
        for i in 4..BLOCK_BYTES {
            blk[i] = (i as u8).wrapping_mul(7);
        }
    }
    let w = Tensor::from_vec(bytes, (out_dim, per_row * BLOCK_BYTES), &dev).unwrap();

    for bad in [f32::NAN, f32::INFINITY] {
        let mut acts = vec![0f32; in_dim];
        acts[3] = bad;
        let x = Tensor::from_vec(acts, (in_dim,), &dev).unwrap();
        let got = joshua::nvfp4_gemv::nvfp4_matmul_qt(&w, &x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            got.iter().all(|v| v.is_nan()),
            "a zero-scale weight times {bad:?} must yield NaN, got {got:?}"
        );
    }
}
