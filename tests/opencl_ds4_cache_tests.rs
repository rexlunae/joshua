#![cfg(feature = "opencl")]
//! DeepSeek-V4 routed experts through the VRAM expert cache on an OpenCL
//! device: a tiny model whose IQ2_XXS gate/up and Q8_0 down experts are
//! borrowed from the mapping on the host and cached on the device under a
//! budget of three experts, in both placements a discrete card is run with
//! (dense set on the device; dense set on the CPU with only the expert
//! cache on the card).  Prefill and decode logits must match the all-CPU
//! model, the pool must churn (uploads, hits, evictions) within its budget,
//! and the expert matmuls must run on the native kernels.
//!
//! One test in its own binary: the fallback counters are process-global, so
//! nothing else may run kernels while a delta is taken.  Skips when no
//! OpenCL platform is installed (pocl serves in CI).
//!
//!   cargo test --features opencl --test opencl_ds4_cache_tests

mod common;

use candle_core::opencl_backend::kernels;
use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

fn opencl_or_skip() -> Option<Device> {
    match Device::opencl_if_available(0) {
        Ok(Device::Cpu) => {
            eprintln!("SKIP: no OpenCL device available on this host");
            None
        }
        Ok(dev) => Some(dev),
        Err(e) => {
            eprintln!("SKIP: opencl init failed: {e}");
            None
        }
    }
}

/// Load through the placed entry point: the dense set on `device`, the
/// routed experts borrowed on the host with a device pool of `cache_bytes`
/// on `expert_device`.
fn load(
    model: &Path,
    device: &Device,
    expert_device: &Device,
    cache_bytes: Option<u64>,
) -> QuantizedModel {
    let bytes = std::fs::read(model).unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let content = header.to_candle_content().unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(model).unwrap()) }.unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    QuantizedModel::from_gguf_mmap_placed(
        content,
        &mut cursor,
        device,
        expert_device,
        Some(Arc::new(mmap)),
        None,
        0,
        cache_bytes,
    )
    .unwrap()
}

fn logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize, device: &Device) -> Vec<f32> {
    let input = Tensor::new(tokens, device)
        .unwrap()
        .reshape((1, tokens.len()))
        .unwrap();
    model
        .forward(&input, offset)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Same rounding-aware comparison as `opencl_model_tests`: the device
/// multiplies f32 activations against decoded blocks, the CPU quantized
/// down-projection quantizes activations to 8 bits.
fn assert_close(what: &str, dev: &[f32], cpu: &[f32]) {
    assert_eq!(dev.len(), cpu.len(), "{what}: logit count");
    assert!(
        dev.iter().all(|v| v.is_finite()),
        "{what}: device logits not finite: {dev:?}"
    );
    let lo = cpu.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let spread = (hi - lo).max(1e-3);
    let worst = dev
        .iter()
        .zip(cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst <= 0.05 * spread,
        "{what}: device logits diverge from cpu by {worst} (spread {spread}):\n dev {dev:?}\n cpu {cpu:?}"
    );
    let arg = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
    };
    assert_eq!(arg(dev), arg(cpu), "{what}: argmax differs");
}

fn fallback_counts() -> std::collections::HashMap<String, usize> {
    kernels::fallback_counts_by_op().into_iter().collect()
}

/// Ops that went through the CPU round-trip between two snapshots.
fn fallback_delta(
    before: &std::collections::HashMap<String, usize>,
    after: &std::collections::HashMap<String, usize>,
) -> Vec<(String, usize)> {
    let mut d: Vec<(String, usize)> = after
        .iter()
        .filter_map(|(op, n)| {
            let delta = n - before.get(op).copied().unwrap_or(0);
            (delta > 0).then(|| (op.clone(), delta))
        })
        .collect();
    d.sort();
    d
}

/// One tiny expert: IQ2_XXS gate + up (128x256 -> 128 blocks x 66 B each)
/// and a Q8_0 down (256x128 -> 1024 blocks x 34 B).
const SLOT_BYTES: u64 = 2 * 128 * 66 + 1024 * 34;
/// The real-layout expert: IQ2_XXS gate + up (256x256 -> 256 blocks x 66 B)
/// and a Q2_K down (256x256 -> 256 blocks x 84 B).
const SLOT_BYTES_Q2K: u64 = 2 * 256 * 66 + 256 * 84;

/// Run prefill + decode on `cached` (dense on `dense`) against the all-CPU
/// `plain`, checking parity at every step and the pool's counters at the end.
fn exercise(
    what: &str,
    plain: &mut QuantizedModel,
    cached: &mut QuantizedModel,
    dense: &Device,
    expect_hits: bool,
) {
    let report = cached
        .device_expert_cache()
        .expect("a pool was built from the budget");
    assert_eq!(
        report.slots, 3,
        "{what}: exact per-expert byte accounting: {report:?}"
    );

    let tokens = [1u32, 4, 2, 7, 5];
    let before = fallback_counts();
    assert_close(
        &format!("{what}: prefill"),
        &logits(cached, &tokens, 0, dense),
        &logits(plain, &tokens, 0, &Device::Cpu),
    );
    cached.wait_for_expert_uploads();
    let after_prefill = cached.device_expert_cache().unwrap();
    assert_eq!(
        after_prefill.stats.hits, 0,
        "{what}: prefill is lookup-only: {after_prefill:?}"
    );
    assert!(
        after_prefill.resident >= 1,
        "{what}: the last prompt row seeds the pool: {after_prefill:?}"
    );

    for (i, t) in [3u32, 3, 3, 3, 8, 3, 3].iter().enumerate() {
        assert_close(
            &format!("{what}: decode step {i}"),
            &logits(cached, &[*t], tokens.len() + i, dense),
            &logits(plain, &[*t], tokens.len() + i, &Device::Cpu),
        );
        cached.wait_for_expert_uploads();
        let r = cached.device_expert_cache().unwrap();
        assert!(
            r.resident <= 3 && r.resident_bytes <= r.budget_bytes,
            "{what}: budget respected: {r:?}"
        );
    }
    // Exclusive tiers: a moment after an upload settles, the expert's host
    // pages are released (this is a real device, so the policy applies).
    // Parity must survive it: a later host miss re-faults the same bytes.
    std::thread::sleep(joshua::residency::HOST_RELEASE_DELAY + std::time::Duration::from_millis(300));
    cached.wait_for_expert_uploads();
    assert_close(
        &format!("{what}: decode after the host pages were released"),
        &logits(cached, &[3], tokens.len() + 7, dense),
        &logits(plain, &[3], tokens.len() + 7, &Device::Cpu),
    );
    cached.wait_for_expert_uploads();
    let r = cached.device_expert_cache().unwrap();
    eprintln!("{what}: {r:?}");
    assert!(
        r.host_releases > 0,
        "{what}: host pages of uploaded experts were released: {r:?}"
    );
    if expect_hits {
        assert!(
            r.stats.hits > 0,
            "{what}: resident experts were used on the device: {r:?}"
        );
    }
    assert!(
        r.stats.uploads >= 3 && r.stats.evictions > 0,
        "{what}: the pool churned: {r:?}"
    );
    assert_eq!(r.stats.refused, 0, "{what}: {r:?}");
    assert_eq!(r.stats.upload_failures, 0, "{what}: {r:?}");
    assert_eq!(r.upload_drops, 0, "{what}: {r:?}");

    // The device's share of the MoE block ran on the native kernels: no
    // quantized matmul, gather, index_add or GEMM round-tripped through the
    // CPU.  (The router's arg_sort is a custom op with no OpenCL kernel; it
    // is not a counted fallback.)
    let delta = fallback_delta(&before, &fallback_counts());
    eprintln!("{what}: fallbacks {delta:?}");
    if candle_core::opencl_backend::native_enabled() {
        for op in [
            "qmatmul",
            "matmul",
            "index_select",
            "index_add",
            "gather",
            "binary",
            "unary",
        ] {
            assert!(
                !delta.iter().any(|(name, _)| name == op),
                "{what}: `{op}` fell back to the CPU round-trip: {delta:?}"
            );
        }
    }
}

#[test]
fn deepseek4_vram_expert_cache_matches_cpu() {
    let Some(ocl) = opencl_or_skip() else { return };
    let dir = common::model_dir("opencl-ds4-cache");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);
    let cpu = Device::Cpu;

    // Dense set on the device, experts cached on the device.
    {
        let mut plain = load(&model, &cpu, &cpu, None);
        let mut cached = load(&model, &ocl, &ocl, Some(3 * SLOT_BYTES));
        exercise("dense on device", &mut plain, &mut cached, &ocl, true);
    }
    // The discrete-card configuration: dense set on the CPU, only the
    // expert cache on the device (activations hop across per layer).
    {
        let mut plain = load(&model, &cpu, &cpu, None);
        let mut cached = load(&model, &cpu, &ocl, Some(3 * SLOT_BYTES));
        exercise("dense on cpu", &mut plain, &mut cached, &cpu, true);
    }
    // The synchronous-upload measurement mode: every decode miss is uploaded
    // on the calling thread and runs on the device in the same step.
    {
        std::env::set_var("JOSHUA_EXPERT_MISS", "upload");
        let mut plain = load(&model, &cpu, &cpu, None);
        let mut cached = load(&model, &cpu, &ocl, Some(3 * SLOT_BYTES));
        std::env::remove_var("JOSHUA_EXPERT_MISS");
        exercise("sync upload on miss", &mut plain, &mut cached, &cpu, true);
        let r = cached.device_expert_cache().unwrap();
        assert!(
            r.stats.hits >= r.stats.misses,
            "every decode miss became a same-step hit: {r:?}"
        );
    }
    // Budget 0 and no budget are the plain host path: no pool at all.
    {
        let none = load(&model, &ocl, &cpu, Some(0));
        assert!(none.device_expert_cache().is_none());
        let none = load(&model, &ocl, &ocl, None);
        assert!(
            none.device_expert_cache().is_none(),
            "an OpenCL expert home without a budget stays on the host"
        );
        // Loading dequantizes F16 dense weights on the device; dropping the
        // models before those launches complete is fine on a conforming
        // runtime (buffer deletion is deferred) but pocl frees eagerly.
        ocl.synchronize().unwrap();
    }
    std::fs::remove_dir_all(&dir).ok();

    // The real V4-Flash expert layout: IQ2_XXS gate/up with a Q2_K down
    // projection, so the slot's down GEMV is the Q2_K kernel the card runs.
    let dir = common::model_dir("opencl-ds4-cache-q2k");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_q2k_down(&model);
    {
        let mut plain = load(&model, &cpu, &cpu, None);
        let mut cached = load(&model, &cpu, &ocl, Some(3 * SLOT_BYTES_Q2K));
        exercise(
            "q2k down, dense on cpu",
            &mut plain,
            &mut cached,
            &cpu,
            true,
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}
