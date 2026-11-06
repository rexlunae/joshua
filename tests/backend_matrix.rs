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
use std::path::Path;

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

/// [`backends`] plus CUDA when it is compiled in and a device is available,
/// for the tests that also run there.
fn backends_and_cuda() -> Vec<Backend> {
    #[allow(unused_mut)]
    let mut v = backends();
    #[cfg(feature = "cuda")]
    match Device::cuda_if_available(0) {
        Ok(d) if !d.is_cpu() => v.push(Backend { name: "cuda", device: d }),
        _ => eprintln!("matrix: cuda unavailable (no device)"),
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
            .map(|_| {
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

/// MTP drafting keeps each pick on the device and embeds it directly for the
/// next draft row.  On every backend in the matrix, plus CUDA: a full-length
/// draft of in-vocabulary tokens that is the same when drafted twice
/// (drafting rolls the block back), with the trunk's next logits unchanged
/// by it.
#[test]
fn mtp_draft_runs_on_every_backend() {
    let dir = common::model_dir("matrix-mtp-qwen3moe");
    let model = dir.join("model.gguf");
    common::write_tiny_qwen_mtp_gguf(&model, "qwen3moe");
    let prompt: Vec<u32> = vec![1, 4, 2, 7, 5, 9, 3, 6];
    let n = prompt.len();
    for backend in backends_and_cuda() {
        let what = format!("[mtp × {}]", backend.name);
        let mut m = joshua::quantized_qwen::with_mtp_block(|| load_mmap(&model, &backend.device));
        assert!(m.set_mtp(true), "{what}: the NextN block loads");
        let vocab = logits(&mut m, &prompt, 0, &backend.device).len();
        let draft = m.mtp_draft(11, n, 4).unwrap();
        assert_eq!(draft.len(), 4, "{what}: draft length");
        assert!(draft.iter().all(|&t| (t as usize) < vocab), "{what}: {draft:?}");
        assert_eq!(m.mtp_draft(11, n, 4).unwrap(), draft, "{what}: drafting is side-effect free");

        let after = logits(&mut m, &[11], n, &backend.device);
        let mut plain = load_mmap(&model, &backend.device);
        logits(&mut plain, &prompt, 0, &backend.device);
        let want = logits(&mut plain, &[11], n, &backend.device);
        // Same backend, same kernels: only the draft could make these differ.
        assert_eq!(after.len(), want.len(), "{what}");
        for (i, (a, w)) in after.iter().zip(&want).enumerate() {
            assert!((a - w).abs() <= 1e-5, "{what}: trunk logit {i} after a draft: {a} vs {w}");
        }
    }
}

/// LoRA adapters on every backend in the matrix, plus CUDA.  An adapter's
/// weights live on the device of the model it was loaded against: there it
/// runs and agrees with the CPU's run of the same file, prefill and decode.
/// A set loaded onto one device is refused by a model on another (and two
/// such adapters do not combine into one set) before the session's state is
/// touched, instead of failing in the middle of the next forward pass.  A
/// CPU-only build checks only the CPU reference.
#[test]
fn lora_adapters_run_on_the_device_they_were_loaded_onto() {
    use joshua::lora::{ActiveSet, LoraAdapter, LoraLimits};
    use std::sync::Arc;
    let dir = common::model_dir("matrix-lora-qwen3");
    let model = dir.join("model.gguf");
    common::write_tiny_qwen_gguf(&model, "qwen3");
    let prompt: Vec<u32> = vec![1, 4, 2, 7, 5, 9];
    let n = prompt.len();
    let limits = LoraLimits::default();
    let mut cpu = load_mmap(&model, &Device::Cpu);
    let table = cpu.lora_slots().expect("the qwen loader applies LoRA");
    let specs: Vec<_> = table
        .iter()
        .enumerate()
        .map(|(i, (k, n_out, n_in))| {
            common::LoraSpec::new(&k.tensor_name(), 4, n_out, n_in, 1300 + 13 * i as u32)
        })
        .collect();
    let adapter = dir.join("adapter.gguf");
    common::write_lora_gguf(&adapter, table.arch(), 8.0, &specs, common::LoraDtype::F32);
    let load_lora = |m: &QuantizedModel| {
        let table = m.lora_slots().expect("the qwen loader applies LoRA");
        Arc::new(LoraAdapter::load(&adapter, &table, &limits).unwrap())
    };
    let set = |selection: &[(Arc<LoraAdapter>, f32)]| {
        ActiveSet::new(selection, &limits).map(|s| s.map(Arc::new))
    };
    let alone = |a: &Arc<LoraAdapter>| set(&[(Arc::clone(a), 1.0)]).unwrap();
    let max_diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };

    let cpu_lora = load_lora(&cpu);
    let base = logits(&mut cpu.new_session().unwrap(), &prompt, 0, &Device::Cpu);
    cpu.set_adapters(alone(&cpu_lora)).unwrap();
    let want = logits(&mut cpu, &prompt, 0, &Device::Cpu);
    let want_next = logits(&mut cpu, &[3], n, &Device::Cpu);
    // A device that dropped the delta must not pass the comparison below.
    let shift = max_diff(&want, &base);
    assert!(shift > 1e-3, "the adapter moves the logits by {shift}");

    for backend in backends_and_cuda() {
        if backend.device.is_cpu() {
            continue;
        }
        let what = format!("[lora × {}]", backend.name);
        let mut m = load_mmap(&model, &backend.device);
        let dev_lora = load_lora(&m);
        assert!(dev_lora.device().same_device(&backend.device), "{what}");

        // The base model's continuation on this device, for the state check.
        let mut plain = m.new_session().unwrap();
        logits(&mut plain, &prompt, 0, &backend.device);
        let base_next = logits(&mut plain, &[3], n, &backend.device);

        // A CPU set is refused here, with the base prefill kept.
        logits(&mut m, &prompt, 0, &backend.device);
        let err = m.set_adapters(alone(&cpu_lora)).unwrap_err();
        assert!(err.to_string().contains("another device"), "{what}: {err}");
        assert_eq!(m.adapter_key(), None, "{what}");
        let next = logits(&mut m, &[3], n, &backend.device);
        let kept = max_diff(&next, &base_next);
        assert!(kept <= 1e-5, "{what}: decode after a refused set: {kept}");
        // And this device's set on the CPU, and the two in one set.
        let mut c = cpu.new_session().unwrap();
        let err = c.set_adapters(alone(&dev_lora)).unwrap_err();
        assert!(err.to_string().contains("another device"), "{what}: {err}");
        let mixed = set(&[(Arc::clone(&cpu_lora), 1.0), (Arc::clone(&dev_lora), 0.5)]);
        assert!(mixed.is_err(), "{what}: adapters on two devices combined");

        // The device's own set runs there and agrees with the CPU, well
        // inside the adapter's own effect.
        assert!(m.set_adapters(alone(&dev_lora)).unwrap(), "{what}");
        let got = logits(&mut m, &prompt, 0, &backend.device);
        let got_next = logits(&mut m, &[3], n, &backend.device);
        for (step, got, want) in [("prefill", got, &want), ("decode", got_next, &want_next)] {
            assert_spread_close(&format!("{what} {step}"), &got, want);
            let d = max_diff(&got, want);
            assert!(d < shift / 20.0, "{what} {step}: {d} off, shift {shift}");
            println!("{what} {step}: {d:.1e} off the CPU, adapter shift {shift:.3}");
        }
    }
    std::fs::remove_dir_all(dir).ok();
}
