#![cfg(all(feature = "sycl", target_os = "linux"))]

mod common;

use joshua::{ComputeBackend, DensePlacement, Engine, EngineOptions, GenerationOptions};

#[test]
fn dense_only_cpu_placement_does_not_leave_a_sycl_expert_device() {
    let Ok(_) = candle_core::Device::new_sycl(0) else {
        eprintln!("SKIP: SYCL bridge/device unavailable");
        return;
    };
    let dir = common::model_dir("dense-cpu-sycl-placement");
    common::write_tiny_qwen_gguf(&dir.join("model.gguf"), "qwen3");
    let cpu = Engine::with_options(
        &dir,
        EngineOptions::with_n_ctx(64).backend(ComputeBackend::Cpu),
    )
    .unwrap();
    let sycl = Engine::with_options(
        &dir,
        EngineOptions::with_n_ctx(64)
            .backend(ComputeBackend::Sycl)
            .dense_placement(DensePlacement::Cpu),
    )
    .unwrap();
    assert!(sycl.device().is_sycl());
    assert!(sycl.dense_device().is_cpu());
    assert!(sycl.expert_device().is_cpu());
    let options = GenerationOptions {
        max_tokens: 4,
        temperature: 0.,
        ..Default::default()
    };
    for prompt in ["hello a", "hello a b c"] {
        let (expected, expected_usage, _, _) = cpu.complete_raw(prompt, &options).unwrap();
        let (actual, actual_usage, _, _) = sycl.complete_raw(prompt, &options).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual_usage.prompt_tokens, expected_usage.prompt_tokens);
        assert_eq!(
            actual_usage.completion_tokens,
            expected_usage.completion_tokens
        );
    }
}
