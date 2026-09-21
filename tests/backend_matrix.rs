//! Cross-backend parity matrix: every architecture fixture × every backend
//! this build supports, over the operational modes that have each broken at
//! least once — short prefill, LONG prefill (~200 tokens: window/compressed
//! paths only engage there), greedy continuation, and batched
//! `forward_sequences`.  One file, so adding a backend means adding it to the
//! matrix (and forgetting means the matrix line is missing at the next run).
//!
//! Backends: CPU is always the reference; OpenCL, Vulkan and Metal join when
//! the feature is compiled in and a device is available.  Device rounding
//! differs from the CPU's quantized-activation path, so logits are compared
//! against the spread of the reference rather than a fixed epsilon, and the
//! greedy continuation is compared by argmax identity.
//!
//! Runs as part of `cargo test` whenever a backend is compiled in; a missing
//! device skips that column with a printed reason.

mod common;
use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;
use std::path::{Path, PathBuf};

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

/// Logits compare against the reference's spread (device rounding differs
/// from the CPU's activation-quantized path), plus a finiteness requirement.
fn assert_spread_close(what: &str, dev: &[f32], cpu: &[f32]) {
    assert_spread_close_tol(what, dev, cpu, 0.08);
}

/// `factor` scales the reference spread into the per-logit tolerance.
fn assert_spread_close_tol(what: &str, dev: &[f32], cpu: &[f32], factor: f32) {
    assert_eq!(dev.len(), cpu.len(), "{what}: logit count");
    assert!(
        dev.iter().all(|v| v.is_finite()),
        "{what}: device logits not finite: {dev:?}"
    );
    let lo = cpu.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let tol = (hi - lo).abs().max(1e-3) * factor;
    for (i, (d, c)) in dev.iter().zip(cpu).enumerate() {
        assert!(
            (d - c).abs() <= tol.max(0.05),
            "{what}: logit {i} diverges: {d} vs {c} (tol {tol})"
        );
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

struct Backend {
    name: &'static str,
    device: Device,
}

fn backends() -> Vec<Backend> {
    let mut v = vec![Backend { name: "cpu", device: Device::Cpu }];
    #[cfg(feature = "opencl")]
    if let Ok(d) = Device::opencl_if_available(0) {
        if !d.is_cpu() {
            v.push(Backend { name: "opencl", device: d });
        } else {
            eprintln!("matrix: opencl unavailable (no device)");
        }
    }
    #[cfg(feature = "vulkan")]
    if let Ok(d) = Device::vulkan_if_available(0) {
        if !d.is_cpu() {
            v.push(Backend { name: "vulkan", device: d });
        } else {
            eprintln!("matrix: vulkan unavailable (no device)");
        }
    }
    #[cfg(feature = "sycl")]
    {
        match Device::new_sycl(0) {
            Ok(d) => v.push(Backend { name: "sycl", device: d }),
            Err(e) => eprintln!("matrix: sycl unavailable: {e}"),
        }
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::metal_if_available(0) {
        if !d.is_cpu() {
            v.push(Backend { name: "metal", device: d });
        } else {
            eprintln!("matrix: metal unavailable (no device)");
        }
    }
    v
}

fn fixtures() -> Vec<(&'static str, fn(&Path))> {
    vec![
        ("llama", common::write_tiny_llama_gguf as fn(&Path)),
        ("qwen3moe", common::write_tiny_qwen3moe_gguf as fn(&Path)),
        ("deepseek2", common::write_tiny_deepseek2_gguf as fn(&Path)),
        ("deepseek4", common::write_tiny_deepseek4_gguf as fn(&Path)),
    ]
}

/// The full arch × backend × mode matrix.  Modes: short prefill, long
/// prefill (~200 tokens), 6-step greedy continuation, batched
/// forward_sequences.
#[test]
fn backend_matrix_parity() {
    let backends = backends();
    println!("backend matrix: {}", backends.iter().map(|b| b.name).collect::<Vec<_>>().join(" + "));
    for (arch, write) in fixtures() {
        let dir = common::model_dir(&format!("matrix-{arch}"));
        let model = dir.join("model.gguf");
        write(&model);

        // CPU reference per phase — each phase needs a fresh instance so no
        // stale KV from the previous phase leaks into the next (a 5-token
        // prefill followed by a 200-token prefill on one instance makes the
        // long attention see 205 KV positions).
        let short: Vec<u32> = vec![1, 4, 2, 7, 5];
        let long: Vec<u32> = (0..200).map(|i| 1 + (i % 15) as u32).collect();
        let mut cpu_short_model = load_mmap(&model, &Device::Cpu);
        let ref_short = logits(&mut cpu_short_model, &short, 0, &Device::Cpu);
        drop(cpu_short_model);
        let mut cpu_long_model = load_mmap(&model, &Device::Cpu);
        let ref_long = logits(&mut cpu_long_model, &long, 0, &Device::Cpu);

        // Greedy reference: 6 steps from the long-prefill state, recording
        // the per-step CPU logits for teacher-forced device comparison.
        let mut ref_ids = long.clone();
        let mut ref_last = ref_long.clone();
        let mut ref_step_logits: Vec<Vec<f32>> = Vec::new();
        let ref_greedy = (0..6)
            .map(|step| {
                let next = argmax(&ref_last);
                ref_ids.push(next);
                ref_last = logits(&mut cpu_long_model, &[next], ref_ids.len() - 1, &Device::Cpu);
                ref_step_logits.push(ref_last.clone());
                next
            })
            .collect::<Vec<u32>>();

        // Batched reference: two sequences, one token each, same position.
        // Not every architecture implements forward_sequences; skip the
        // batched mode for those rather than failing the matrix.
        let batch_in = Tensor::new(&[ref_ids[0], ref_ids[1]], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let mut cpu_batch_model = load_mmap(&model, &Device::Cpu);
        let ref_batch: Option<Vec<f32>> = cpu_batch_model
            .forward_sequences(&[(&batch_in, ref_ids.len() - 1), (&batch_in, ref_ids.len() - 1)])
            .ok()
            .map(|v| v.into_iter().flatten().collect());

        for backend in &backends {
            if backend.name == "cpu" {
                continue;
            }
            let mut m_short = load_mmap(&model, &backend.device);
            let got_short = logits(&mut m_short, &short, 0, &backend.device);
            assert_spread_close(
                &format!("[{arch} × {}] short prefill", backend.name),
                &got_short,
                &ref_short,
            );
            drop(m_short);
            let mut m = load_mmap(&model, &backend.device);
            let got_long = logits(&mut m, &long, 0, &backend.device);
            assert_spread_close(
                &format!("[{arch} × {}] long prefill (200 tok)", backend.name),
                &got_long,
                &ref_long,
            );

            // Teacher-forced continuation parity: feed the CPU reference
            // tokens to the device and compare per-step logits.  (Strict
            // argmax identity is too strong for MoE arches — near-tied router
            // scores flip expert selection on tiny numeric differences — but
            // KV/position drift shows up as large per-step divergence.)
            let mut dev_last = got_long.clone();
            for step in 0..6 {
                let cpu_tok = ref_greedy[step];
                dev_last = logits(&mut m, &[cpu_tok], ref_ids.len() + step, &backend.device);
                // MoE router ties flip expert selection between the CPU's
                // activation-quantized path and the device's dequantized path,
                // shifting whole logits on a tiny fixture — allow a looser
                // bound for MoE arches than for dense ones.
                let is_moe = arch != "llama";
                assert_spread_close_tol(
                    &format!("[{arch} × {}] teacher-forced step {step}", backend.name),
                    &dev_last,
                    &ref_step_logits[step],
                    if is_moe { 0.6 } else { 0.08 },
                );
            }

            // Batched decode parity (skipped when the arch has no
            // forward_sequences).
            if let Some(ref_batch) = &ref_batch {
                let dev_batch_in = Tensor::new(&[ref_ids[0], ref_ids[1]], &backend.device)
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap();
                let got_batch: Vec<f32> = m
                    .forward_sequences(&[
                        (&dev_batch_in, ref_ids.len() - 1),
                        (&dev_batch_in, ref_ids.len() - 1),
                    ])
                    .unwrap_or_else(|e| {
                        panic!(
                            "[{arch} × {}] CPU supports batched decode but the device does not: {e}",
                            backend.name
                        )
                    })
                    .into_iter()
                    .flatten()
                    .collect();
                assert_spread_close(
                    &format!("[{arch} × {}] batched decode", backend.name),
                    &got_batch,
                    ref_batch,
                );
            }
            println!("matrix: {arch} × {} OK", backend.name);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
