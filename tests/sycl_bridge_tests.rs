//! SYCL bridge kernel parity tests (Intel Arc Pro B50 / any SYCL 2020 GPU).
//!
//! Every test drives the bridge directly (`joshua::sycl_backend`): alloc,
//! write the input, launch one kernel, read the output, compare against a CPU
//! reference computed in the test.  These are the correctness gate for the
//! SYCL kernels on real hardware — the same role the OpenCL parity tests play
//! for that backend.
//!
//! Run on the SYCL host:
//!   cargo test --release --features sycl --test sycl_bridge_tests -- --nocapture
//!
//! Library locations: `JOSHUA_SYCL_LIBRARY` (libjoshua_sycl.so) and
//! `JOSHUA_SYCL_RUNTIME` (libsycl.so.9), defaulting to the oneAPI toolchain
//! layout used on the SYCL boxes.
#![cfg(all(feature = "sycl", target_os = "linux"))]

use candle_core::{Device, Tensor};
use candle_core::sycl_backend::SyclDevice;

/// All tests share one device: the DPC++ runtime crashes when contexts are
/// opened and closed concurrently from parallel test threads.

/// All SYCL kernel parity tests run in a single serial test: the DPC++
/// runtime crashes with SIGSEGV when multiple threads submit kernels
/// concurrently, even through the same bridge.  Each sub-test is called
/// sequentially here.
#[test]
fn sycl_kernel_parity() {
    let tests: Vec<(&'static str, fn(&SyclDevice))> = vec![
        ("affine", sycl_affine_matches_cpu),
        ("unary_exp", sycl_unary_exp_matches_cpu),
        ("binary_add", sycl_binary_add_matches_cpu),
        ("softmax", sycl_softmax_matches_cpu),
        ("rmsnorm", sycl_rmsnorm_matches_cpu),
        ("gemm", sycl_gemm_matches_cpu),
        ("qgemv_q8_0", sycl_qgemv_q8_0_matches_cpu),
        ("hembed", sycl_hembed_matches_cpu),
    ];
    let Some(dev) = device() else {
        eprintln!("SKIP: no SYCL device");
        return;
    };
    eprintln!("SYCL device: {} ({} MiB)", dev.name(), dev.memory() / (1024 * 1024));
    for (name, test) in &tests {
        eprintln!("  {name}...");
        test(&dev);
        eprintln!("  {name} OK");
    }
    println!("all SYCL kernel parity tests passed");
}

fn device() -> Option<&'static SyclDevice> {
    static DEV: std::sync::OnceLock<Result<SyclDevice, String>> = std::sync::OnceLock::new();
    let dev = DEV.get_or_init(|| SyclDevice::new(0).map_err(|e| e.to_string()));
    match dev {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("SKIP: no SYCL device/bridge available: {e}");
            None
        }
    }
}

fn assert_close(what: &str, dev: &[f32], cpu: &[f32], tol: f32) {
    assert_eq!(dev.len(), cpu.len(), "{what}: length");
    let mut worst = 0.0f32;
    let mut wi = 0;
    for (i, (a, b)) in dev.iter().zip(cpu).enumerate() {
        // Relative to the magnitude of the expectation with an absolute
        // floor, so exact-zero expectations don't inflate the ratio.
        let rel = (a - b).abs() / b.abs().max(0.5);
        if rel > worst {
            worst = rel;
            wi = i;
        }
    }
    if worst > tol {
        let mut shown = 0;
        for (i, (a, b)) in dev.iter().zip(cpu).enumerate() {
            if (a - b).abs() > tol * b.abs().max(0.5) && shown < 6 {
                eprintln!("  mismatch[{i}]: dev={a} cpu={b}");
                shown += 1;
            }
        }
        assert!(false, "{what}: worst rel diff {worst} at {wi} (dev={} cpu={})", dev[wi], cpu[wi]);
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_ne_bytes()).collect()
}

fn f32_from(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

/// `o = x * mul + add` (kernel `k_affine_c`).
fn sycl_affine_matches_cpu(dev: &SyclDevice) {
    let n = 4096;
    let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.5).collect();
    let xb = dev.alloc(n * 4).unwrap();
    let ob = dev.alloc(n * 4).unwrap();
    dev.write(xb, 0, &f32_bytes(&x)).unwrap();
    dev.run_affine_c(xb, ob, n, 0, 1.5, -0.25).unwrap();
    dev.finish().unwrap();
    let mut out = vec![0u8; n * 4];
    dev.read(ob, 0, &mut out).unwrap();
    let got = f32_from(&out);
    let want: Vec<f32> = x.iter().map(|v| v * 1.5 - 0.25).collect();
    assert_close("affine", &got, &want, 1e-6);
    dev.free(xb).unwrap();
    dev.free(ob).unwrap();
}

/// Unary `exp` (kernel `k_unary_c`, op 0).
fn sycl_unary_exp_matches_cpu(dev: &SyclDevice) {
    let n = 4096;
    let x: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.4).collect();
    let xb = dev.alloc(n * 4).unwrap();
    let ob = dev.alloc(n * 4).unwrap();
    dev.write(xb, 0, &f32_bytes(&x)).unwrap();
    dev.run_unary_c(xb, ob, n, 0, 0).unwrap();
    dev.finish().unwrap();
    let mut out = vec![0u8; n * 4];
    dev.read(ob, 0, &mut out).unwrap();
    let got = f32_from(&out);
    let want: Vec<f32> = x.iter().map(|v| v.exp()).collect();
    assert_close("unary exp", &got, &want, 1e-6);
    dev.free(xb).unwrap();
    dev.free(ob).unwrap();
}

/// Binary add (kernel `k_binary_c`, op 0).
fn sycl_binary_add_matches_cpu(dev: &SyclDevice) {
    let n = 4096;
    let a: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let ab = dev.alloc(n * 4).unwrap();
    let bb = dev.alloc(n * 4).unwrap();
    let ob = dev.alloc(n * 4).unwrap();
    dev.write(ab, 0, &f32_bytes(&a)).unwrap();
    dev.write(bb, 0, &f32_bytes(&b)).unwrap();
    dev.run_binary_c(ab, bb, ob, n, 0, 0, 0).unwrap();
    dev.finish().unwrap();
    let mut out = vec![0u8; n * 4];
    dev.read(ob, 0, &mut out).unwrap();
    let got = f32_from(&out);
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_close("binary add", &got, &want, 1e-6);
    dev.free(ab).unwrap();
    dev.free(bb).unwrap();
    dev.free(ob).unwrap();
}

/// Row softmax over [8, 200] — 200-wide rows exercise the multi-stride
/// reduction path that broke the Vulkan softmax at these lengths (#81).
fn sycl_softmax_matches_cpu(dev: &SyclDevice) {
    let (rows, cols) = (8usize, 200usize);
    let n = rows * cols;
    let x: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.3).collect();
    let xb = dev.alloc(n * 4).unwrap();
    let ob = dev.alloc(n * 4).unwrap();
    dev.write(xb, 0, &f32_bytes(&x)).unwrap();
    dev.run_softmax_last(xb, ob, rows, cols, 0).unwrap();
    dev.finish().unwrap();
    let mut out = vec![0u8; n * 4];
    dev.read(ob, 0, &mut out).unwrap();
    let got = f32_from(&out);
    // CPU reference: softmax per row.
    let mut want = vec![0f32; n];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (v - m).exp()).sum();
        for (c, v) in row.iter().enumerate() {
            want[r * cols + c] = (v - m).exp() / sum;
        }
    }
    assert_close("softmax", &got, &want, 1e-5);
    // Every row must sum to 1 — the #81 failure signature was rows summing
    // to ~56 on the Vulkan path.
    for r in 0..rows {
        let s: f32 = got[r * cols..(r + 1) * cols].iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "row {r} sums to {s}, expected 1");
    }
    dev.free(xb).unwrap();
    dev.free(ob).unwrap();
}

/// Row RMSNorm over [8, 200] with unit alpha.
fn sycl_rmsnorm_matches_cpu(dev: &SyclDevice) {
    let (rows, cols) = (8usize, 200usize);
    let n = rows * cols;
    let x: Vec<f32> = (0..n).map(|i| ((i % 37) as f32 - 18.0) * 0.05).collect();
    let alpha = vec![1.0f32; cols];
    let xb = dev.alloc(n * 4).unwrap();
    let ab = dev.alloc(cols * 4).unwrap();
    let ob = dev.alloc(n * 4).unwrap();
    dev.write(xb, 0, &f32_bytes(&x)).unwrap();
    dev.write(ab, 0, &f32_bytes(&alpha)).unwrap();
    dev.run_rmsnorm(xb, ab, ob, rows, cols, 0, 0, 1e-5).unwrap();
    dev.finish().unwrap();
    let mut out = vec![0u8; n * 4];
    dev.read(ob, 0, &mut out).unwrap();
    let got = f32_from(&out);
    let mut want = vec![0f32; n];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (mean_sq + 1e-5).sqrt();
        for (c, v) in row.iter().enumerate() {
            want[r * cols + c] = v * inv * alpha[c];
        }
    }
    assert_close("rmsnorm", &got, &want, 1e-5);
    dev.free(xb).unwrap();
    dev.free(ab).unwrap();
    dev.free(ob).unwrap();
}

/// Tiled GEMM [64, 96] × [96, 51] — N=51 exercises the tile guard paths.
/// (Was segfault-gated: the real bug was the Bs tile overlapping As in
/// shared memory — see the kernels.hpp fix.)
fn sycl_gemm_matches_cpu(dev: &SyclDevice) {
    let (m, k, n) = (64usize, 96usize, 51usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.25).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
    let ab = dev.alloc(m * k * 4).unwrap();
    let bb = dev.alloc(k * n * 4).unwrap();
    let cb = dev.alloc(m * n * 4).unwrap();
    dev.write(ab, 0, &f32_bytes(&a)).unwrap();
    dev.write(bb, 0, &f32_bytes(&b)).unwrap();
    // Row-major: sam = k (A stride over m), sak = 1; B contiguous [k, n]:
    // sbk = n, sbn = 1. b_kc = 0 (B contiguous along n).
    dev.run_gemm(ab, bb, cb, m, n, k, k, 1, n, 1, 0, 0, 0, 0, 0, 0, 0).unwrap();
    dev.finish().unwrap();
    let mut out = vec![0u8; m * n * 4];
    dev.read(cb, 0, &mut out).unwrap();
    let got = f32_from(&out);
    let mut want = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let sum: f32 = (0..k).map(|kk| a[r * k + kk] * b[kk * n + c]).sum();
            want[r * n + c] = sum;
        }
    }
    assert_close("gemm", &got, &want, 1e-3);
    dev.free(ab).unwrap();
    dev.free(bb).unwrap();
    dev.free(cb).unwrap();
}

/// Quantized GEMV (kernel `k_qgemv`, Q8_0): dequantize-in-kernel over
/// GGUF blocks, the single-token decode path.
fn sycl_qgemv_q8_0_matches_cpu(dev: &SyclDevice) {
    let (n, k) = (24usize, 128usize);
    let (qk, bsz) = (32usize, 34usize);
    // Q8_0 blocks: fp16 scale + 32 i8 values.
    let n_blocks = n * (k / qk);
    let mut w_bytes = vec![0u8; n_blocks * bsz];
    for (bi, block) in w_bytes.chunks_exact_mut(bsz).enumerate() {
        let scale = 0.25f32 + (bi % 7) as f32 * 0.125;
        block[0..2].copy_from_slice(&half_to_bits(scale).to_le_bytes());
        for (i, v) in block[2..].iter_mut().enumerate() {
            *v = ((bi * 31 + i * 7) % 15) as i8 as u8;
        }
    }
    let x: Vec<f32> = (0..k).map(|i| ((i % 9) as f32 - 4.0) * 0.5).collect();

    let wb = dev.alloc(w_bytes.len()).unwrap();
    let xb = dev.alloc(k * 4).unwrap();
    let cb = dev.alloc(n * 4).unwrap();
    dev.write(wb, 0, &w_bytes).unwrap();
    dev.write(xb, 0, &f32_bytes(&x)).unwrap();
    // QT_Q8_0 = 8, qk = 32, bsz = 34, woff = 0, xoff = 0, coff = 0, m = 1.
    dev.run_qgemv(xb, wb, cb, n, k, 8, 32, 34, 0, 0, 0, 1).unwrap();
    dev.finish().unwrap();
    let mut out = vec![0u8; n * 4];
    dev.read(cb, 0, &mut out).unwrap();
    let got = f32_from(&out);

    // CPU reference: dequant each Q8_0 block, dot with x.
    let mut want = vec![0f32; n];
    for row in 0..n {
        let mut acc = 0f32;
        for (bi, block) in w_bytes[row * (k / qk) * bsz..(row + 1) * (k / qk) * bsz]
            .chunks_exact(bsz)
            .enumerate()
        {
            let scale =
                f32::from_bits(u16::from_le_bytes([block[0], block[1]]) as u32) as f32;
            let d = half_from_bits(u16::from_le_bytes([block[0], block[1]]));
            let _ = scale;
            for (i, v) in block[2..].iter().enumerate() {
                acc += x[bi * qk + i] * d * (*v as i8 as f32);
            }
        }
        want[row] = acc;
    }
    assert_close("qgemv q8_0", &got, &want, 1e-3);
    dev.free(wb).unwrap();
    dev.free(xb).unwrap();
    dev.free(cb).unwrap();
}

fn half_to_bits(v: f32) -> u16 {
    // Approximate f32→f16 (enough for test data).
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp32 = ((bits >> 23) & 0xff) as i32;
    let frac32 = bits & 0x7fffff;
    if exp32 == 0 { return sign; }
    let e = (exp32 - 127 + 15) as u16;
    let f = (frac32 >> 13) as u16;
    sign | (e << 10) | f
}

fn half_from_bits(bits: u16) -> f32 {
    // IEEE f16 → f32 conversion (no f16 crate in scope for the test).
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as f32 / 1024.0;
    match exp {
        0 => sign * frac * (1.0 / 16384.0),
        31 => sign * f32::INFINITY,
        e => sign * (1.0 + frac) * 2f32.powi(e - 15),
    }
}

/// F16 embedding gather (kernel `k_hembed`).
fn sycl_hembed_matches_cpu(dev: &SyclDevice) {
    let (vocab, k, n_ids) = (64usize, 32usize, 5usize);
    let table: Vec<f32> = (0..vocab * k).map(|i| ((i % 29) as f32 - 14.0) * 0.125).collect();
    let mut f16_bytes = vec![0u8; vocab * k * 2];
    for (i, v) in table.iter().enumerate() {
        let bits = half_to_bits(*v);
        f16_bytes[2 * i..2 * i + 2].copy_from_slice(&bits.to_le_bytes());
    }
    let ids: Vec<u32> = vec![3, 17, 0, 41, 63];

    let wb = dev.alloc(f16_bytes.len()).unwrap();
    let idb = dev.alloc(ids.len() * 4).unwrap();
    let ob = dev.alloc(n_ids * k * 4).unwrap();
    // The kernel writes a fault flag per slot; zero the checker buffer.
    let faultb = dev.alloc(4).unwrap();
    dev.write(faultb, 0, &0u32.to_ne_bytes()).unwrap();
    dev.write(wb, 0, &f16_bytes).unwrap();
    dev.write(idb, 0, &ids.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>()).unwrap();
    dev.run_hembed(wb, idb, ob, n_ids, k, false, 0, 0, vocab, faultb, 0).unwrap();
    dev.finish().unwrap();
    // Overlap check: the kernel must not have corrupted the weight table.
    let mut w_back = vec![0u8; f16_bytes.len()];
    dev.read(wb, 0, &mut w_back).unwrap();
    if w_back != f16_bytes {
        let diffs = w_back.iter().zip(&f16_bytes).filter(|(a, b)| a != b).count();
        eprintln!("WEIGHT TABLE CORRUPTED by the kernel: {diffs} bytes differ (output buffer overlaps W)");
    }
    let mut out = vec![0u8; n_ids * k * 4];
    dev.read(ob, 0, &mut out).unwrap();
    let got = f32_from(&out);
    let mut want = vec![0f32; n_ids * k];
    for (j, id) in ids.iter().enumerate() {
        for c in 0..k {
            want[j * k + c] = table[*id as usize * k + c];
        }
    }
    assert_close("hembed f16", &got, &want, 1e-6);
    dev.free(wb).unwrap();
    dev.free(idb).unwrap();
    dev.free(ob).unwrap();
}
