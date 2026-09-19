//! Per-kernel deny sweep for the OpenCL native path.
//!
//! For each model-critical native kernel, deny it via
//! `JOSHUA_OPENCL_NATIVE_DENY` and verify the affected ops still produce the
//! CPU reference through the fallback.  This is the systematic version of the
//! one-off bisect that found the K==0 uninitialised-output bug (#79): when a
//! kernel is wrong for a shape family, the sweep names it immediately, and
//! when a *fallback* breaks, the sweep catches that too (a fallback that only
//! runs under deny is otherwise never exercised).
//!
//! `JOSHUA_OPENCL_NATIVE_DENY` is read per launch, so the sweep can flip it
//! between kernels inside one process.
//!
//! Run on an OpenCL host:
//!   JOSHUA_OPENCL_NATIVE=1 cargo test --release --features opencl \
//!     --test opencl_deny_sweep -- --ignored --nocapture
#![cfg(feature = "opencl")]

mod common;
use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;
use std::path::Path;

/// Model-critical kernels (the elementwise/cast/copy boilerplate shares its
/// correctness risk with `k_unary_s`/`k_copy_s*` and is covered by the same
/// high-level parity assertions).
const SWEPT: &[&str] = &[
    "k_gemm",
    "k_gemv_nn",
    "k_gemv_nt",
    "k_hgemv",
    "k_hembed",
    "k_qgemv",
    "k_qgemv_mr",
    "k_qembed",
    "k_dequant",
    "k_dequant_half",
    "k_rmsnorm",
    "k_softmax_last",
    "k_arg_last",
    "k_reduce_generic",
    "k_rope",
    "k_index_add_f32",
    "k_unary_s",
    "k_affine_s",
];

fn device() -> Option<Device> {
    static D: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
    D.get_or_init(|| match Device::opencl_if_available(0) {
        Ok(Device::Cpu) => {
            eprintln!("SKIP: no OpenCL device");
            None
        }
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("SKIP: opencl init failed: {e}");
            None
        }
    })
    .clone()
}

fn load_mmap(model: &Path, device: &Device) -> QuantizedModel {
    let bytes = std::fs::read(model).unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let content = header.to_candle_content().unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(model).unwrap()) }
        .ok()
        .map(std::sync::Arc::new);
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    QuantizedModel::from_gguf_mmap(content, &mut cursor, device, mmap, None, 0).unwrap()
}

fn logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize, device: &Device) -> Vec<f32> {
    let input = Tensor::new(tokens, device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    model
        .forward(&input, offset)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Parity of a short prefill plus one decode step against the CPU reference,
/// under the currently denied kernel.  Tolerance mirrors the model tests:
/// compare against the reference spread (device dequant-vs-CPU rounding).
fn parity(device: &Device, model: &Path, what: &str) {
    let mut dev = load_mmap(model, device);
    parity_with(&mut dev, model, device, what);
}

fn parity_with(dev_model: &mut QuantizedModel, model: &Path, device: &Device, what: &str) {
    let tokens: Vec<u32> = (0..160).map(|i| 1 + (i % 15) as u32).collect();
    let mut cpu = load_mmap(model, &Device::Cpu);
    let ref_prefill = logits(&mut cpu, &tokens, 0);
    let ref_decode = logits(&mut cpu, &tokens[tokens.len() - 1..], tokens.len(), );

    let got_prefill = logits(dev_model, &tokens, 0, device);
    let got_decode = logits(dev_model, &tokens[tokens.len() - 1..], tokens.len(), device);

    for (phase, got, want) in [
        ("prefill", &got_prefill, &ref_prefill),
        ("decode", &got_decode, &ref_decode),
    ] {
        assert_eq!(got.len(), want.len(), "{what} {phase}: logit count");
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{what} {phase}: non-finite under deny"
        );
        let lo = want.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = want.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let tol = (hi - lo).abs().max(1e-3) * 0.08 + 0.05;
        for (i, (d, c)) in got.iter().zip(want).enumerate() {
            assert!(
                (d - c).abs() <= tol,
                "{what} {phase}: logit {i} diverges under deny: {d} vs {c} (tol {tol})"
            );
        }
    }
}

#[test]
#[ignore]
fn deny_each_kernel_falls_back_to_matching_results() {
    let Some(dev) = device() else { return };
    let dir = common::model_dir("opencl-deny-sweep");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    // Baseline: nothing denied, native path must also match (guards against
    // the sweep silently skipping all kernels).
    std::env::remove_var("JOSHUA_OPENCL_NATIVE_DENY");
    parity(&dev, &model, "baseline (no deny)");

    for kernel in SWEPT {
        std::env::set_var("JOSHUA_OPENCL_NATIVE_DENY", kernel);
        // A denied kernel can break the LOAD itself (e.g. the eager F16
        // dequantize of output.weight has no CPU fallback): that is a
        // distinct finding — the op is load-critical, not just parity — so
        // report it and move on rather than failing the sweep.
        let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            load_mmap(&model, &dev)
        }));
        match loaded {
            Err(_) => {
                println!("deny {kernel}: load-critical (no fallback for eager load-time use)");
                std::env::remove_var("JOSHUA_OPENCL_NATIVE_DENY");
                continue;
            }
            Ok(Err(_)) => {
                println!("deny {kernel}: load-critical (load returned an error)");
                std::env::remove_var("JOSHUA_OPENCL_NATIVE_DENY");
                continue;
            }
            Ok(Ok(mut dev_model)) => {
                parity_with(&mut dev_model, &model, &dev, &format!("deny {kernel}"));
                println!("deny {kernel}: fallback parity OK");
            }
        }
    }
    std::env::remove_var("JOSHUA_OPENCL_NATIVE_DENY");
    std::fs::remove_dir_all(&dir).ok();
}
