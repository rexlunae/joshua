//! Metal backend tests: run Joshua's own loaders (the MoE ones especially)
//! on the Metal device and check the logits agree with the CPU path.
//!
//! Compiled only with `cargo test --features metal` and skipped at runtime
//! when no Metal device is present (CI runners without GPUs).

#![cfg(feature = "metal")]

mod common;

use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;
use joshua::{ComputeBackend, Engine, EngineOptions};

fn metal() -> Option<Device> {
    match Device::new_metal(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping: no Metal device: {e}");
            None
        }
    }
}

fn load(model: &std::path::Path, device: &Device) -> QuantizedModel {
    let bytes = std::fs::read(model).unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let content = candle_core::quantized::gguf_file::Content::read(&mut cursor).unwrap();
    QuantizedModel::from_gguf(content, &mut cursor, device).unwrap()
}

fn logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize, device: &Device) -> Vec<f32> {
    let input = Tensor::new(tokens, device).unwrap().unsqueeze(0).unwrap();
    model
        .forward(&input, offset)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Max absolute difference between the CPU and Metal logit vectors.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Tiny models use random weights in [-0.1, 0.1] and 2 layers; CPU (AVX/NEON
/// kernels) and Metal (GPU kernels) accumulate in different orders, so small
/// differences are expected.  The tolerance is generous enough for that but
/// tight enough to catch a wrong kernel, a dropped expert, or a transposed
/// weight.
fn assert_close(cpu: &[f32], metal: &[f32], ctx: &str) {
    assert_eq!(cpu.len(), metal.len(), "{ctx}: vocab length");
    let diff = max_abs_diff(cpu, metal);
    assert!(
        diff < 1e-3,
        "{ctx}: CPU/Metal logits diverged by {diff} (cpu={cpu:?} metal={metal:?})"
    );
}

#[test]
fn deepseek2_moe_metal_matches_cpu() {
    let Some(dev) = metal() else { return };
    let dir = common::model_dir("metal-ds2");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek2_gguf(&model);

    let mut cpu = load(&model, &Device::Cpu);
    let mut gpu = load(&model, &dev);
    // Two forward passes (prefill 4 tokens, then 1 more) exercise MLA +
    // MoE routing + shared experts on both paths.
    for (tokens, offset) in [(&[1, 4, 2, 7][..], 0usize), (&[5][..], 4usize)] {
        let l_cpu = logits(&mut cpu, tokens, offset, &Device::Cpu);
        let l_gpu = logits(&mut gpu, tokens, offset, &dev);
        assert_close(&l_cpu, &l_gpu, "deepseek2 (legacy MLA)");
    }
}

#[test]
fn deepseek2_mla_split_metal_matches_cpu() {
    let Some(dev) = metal() else { return };
    let dir = common::model_dir("metal-ds2-mla");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek2_mla_gguf(&model);

    let mut cpu = load(&model, &Device::Cpu);
    let mut gpu = load(&model, &dev);
    let l_cpu = logits(&mut cpu, &[1, 4, 2, 7, 5], 0, &Device::Cpu);
    let l_gpu = logits(&mut gpu, &[1, 4, 2, 7, 5], 0, &dev);
    assert_close(&l_cpu, &l_gpu, "deepseek2 (MLA-split)");
}

#[test]
fn deepseek2_v2_routing_metal_matches_cpu() {
    let Some(dev) = metal() else { return };
    let dir = common::model_dir("metal-ds2-v2");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek2_v2_gguf(&model);

    let mut cpu = load(&model, &Device::Cpu);
    let mut gpu = load(&model, &dev);
    let l_cpu = logits(&mut cpu, &[1, 4, 2, 7, 5], 0, &Device::Cpu);
    let l_gpu = logits(&mut gpu, &[1, 4, 2, 7, 5], 0, &dev);
    assert_close(&l_cpu, &l_gpu, "deepseek2 (V2 routing)");
}

#[test]
fn llama_metal_matches_cpu() {
    let Some(dev) = metal() else { return };
    let dir = common::model_dir("metal-llama");
    let model = dir.join("model.gguf");
    common::write_tiny_llama_gguf(&model);

    let mut cpu = load(&model, &Device::Cpu);
    let mut gpu = load(&model, &dev);
    let l_cpu = logits(&mut cpu, &[1, 4, 2, 7, 5], 0, &Device::Cpu);
    let l_gpu = logits(&mut gpu, &[1, 4, 2, 7, 5], 0, &dev);
    assert_close(&l_cpu, &l_gpu, "llama");
}

#[test]
fn qwen3_metal_matches_cpu() {
    let Some(dev) = metal() else { return };
    let dir = common::model_dir("metal-qwen3");
    let model = dir.join("model.gguf");
    common::write_tiny_gguf(&model, "qwen3");

    let mut cpu = load(&model, &Device::Cpu);
    let mut gpu = load(&model, &dev);
    let l_cpu = logits(&mut cpu, &[1, 4, 2, 7, 5], 0, &Device::Cpu);
    let l_gpu = logits(&mut gpu, &[1, 4, 2, 7, 5], 0, &dev);
    assert_close(&l_cpu, &l_gpu, "qwen3");
}

/// DeepSeek-V4 cannot run on Metal at all: the real conversion keeps IQ2_XXS
/// routed experts (CPU-only by design — see `deepseek4_iq2xxs_experts_refuse_metal`),
/// and even a hypothetical file with candle-readable dtypes hits ops the Metal
/// backend lacks (gather/scatter/index-select in Hyper-Connections and the
/// Lightning Indexer).  CPU-only is enforced, so no test asserts a Metal path.
#[test]
fn deepseek4_iq2xxs_experts_refuse_metal() {
    let Some(dev) = metal() else { return };
    let dir = common::model_dir("metal-ds4-iq2");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_iq2xxs_output(&model);

    let bytes = std::fs::read(&model).unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let content = header.to_candle_content().unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let err = QuantizedModel::from_gguf_mmap(content, &mut cursor, &dev, None)
        .err()
        .expect("IQ2_XXS load on Metal must fail, not allocate ~16× f32");
    let msg = format!("{err}");
    assert!(
        msg.contains("only supported on the CPU device"),
        "unexpected error: {msg}"
    );
}

/// The full engine pipeline — mmap → dispatch → prefill → decode → KV
/// reuse — on the Metal device must produce the same output as CPU.
///
/// Uses the head_dim-32 tiny model: candle's Metal backend routes
/// single-token decode through its SDPA kernel, which only supports
/// head dims ≥ 32 (every real model qualifies).
#[test]
fn engine_metal_full_pipeline_matches_cpu() {
    let Some(_dev) = metal() else { return };
    let dir = common::model_dir("metal-engine");
    common::write_tiny_llama_gguf_hd32(&dir.join("model.gguf"));

    use joshua::types::GenerationOptions;
    // Seed the KV cache without generating, then generate from an extended
    // prompt; a repeat of the same prompt must take the prefix-reuse path
    // and reproduce the identical greedy output.
    let seed = GenerationOptions {
        max_tokens: 0,
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let greedy = GenerationOptions {
        max_tokens: 8,
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };

    let cpu = Engine::with_options(
        &dir,
        EngineOptions::with_n_ctx(64).backend(ComputeBackend::Cpu),
    )
    .expect("cpu engine should load tiny model");
    let metal = Engine::with_options(
        &dir,
        EngineOptions::with_n_ctx(64).backend(ComputeBackend::Metal),
    )
    .expect("metal engine should load tiny model");

    for engine in [&cpu, &metal] {
        engine.complete_raw("hello a", &seed).unwrap();
        assert_eq!(engine.kv_reuse_count(), 0, "first call must prefill");
        let (_, usage_a, _, _) = engine.complete_raw("hello a b c", &greedy).unwrap();
        assert_eq!(usage_a.prompt_tokens, 4);
        assert!(usage_a.completion_tokens <= 8);
        assert_eq!(
            engine.kv_reuse_count(),
            1,
            "extended prompt must reuse the cached prefix on {:?}",
            engine.device()
        );
    }

    let (cpu_text, cpu_usage, _, _) = cpu.complete_raw("hello a b c", &greedy).unwrap();
    let (metal_text, metal_usage, _, _) = metal.complete_raw("hello a b c", &greedy).unwrap();
    assert_eq!(
        cpu_text, metal_text,
        "CPU and Metal engines must produce identical greedy output"
    );
    assert_eq!(cpu_usage.completion_tokens, metal_usage.completion_tokens);

    std::fs::remove_dir_all(&dir).ok();
}

/// In a `metal` build with a working device, `Auto` must resolve to Metal —
/// the same binary then reports the GPU in `device()` and the CLI logs it.
#[test]
fn auto_resolves_to_metal_in_metal_build() {
    let Some(_dev) = metal() else { return };
    let dir = common::model_dir("metal-auto");
    common::write_tiny_llama_gguf(&dir.join("model.gguf"));

    let engine = Engine::with_options(&dir, EngineOptions::with_n_ctx(64))
        .expect("auto backend must load the tiny model");
    assert!(
        engine.device().is_metal(),
        "Auto on a metal build must pick Metal, got {:?}",
        engine.device()
    );

    std::fs::remove_dir_all(&dir).ok();
}
