#![cfg(feature = "cuda")]

mod common;

use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;
use std::io::Cursor;

#[test]
fn cuda_deepseek2_cpu_reference_numerical_parity() {
    let device = Device::new_cuda_with_stream(0).unwrap();
    let dir = common::model_dir("cuda-dense-cpu-reference");
    let path = dir.join("model.gguf");
    common::write_tiny_deepseek2_quantized_gguf(&path);
    let bytes = std::fs::read(&path).unwrap();
    let load = |device: &Device, budget| {
        let mut reader = Cursor::new(&bytes);
        let header = joshua::gguf_ext::read_header(&mut reader).unwrap();
        QuantizedModel::from_gguf_mmap_placed(
            header.to_candle_content().unwrap(),
            &mut reader,
            device,
            &Device::Cpu,
            None,
            None,
            0,
            budget,
            joshua::EngramTier::Auto,
        )
        .unwrap()
    };
    let evaluate = |model: &mut QuantizedModel, device: &Device, tokens: &[u32], offset| {
        let input = Tensor::new(tokens, device)
            .unwrap()
            .reshape((1, tokens.len()))
            .unwrap();
        model
            .forward_all_logits(&input, offset)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    for budget in [Some(0), Some(9 * 256 * 144)] {
        let mut gpu = load(&device, budget);
        gpu.set_pin_hot_experts(3);
        let mut cpu = load(&Device::Cpu, None);
        let mut mismatches = 0;
        let mut max_error = 0f32;
        let mut tokens = vec![1u32, 4, 2];
        let mut offset = 0;
        while offset <= 75 {
            let actual = evaluate(&mut gpu, &device, &tokens, offset);
            let expected = evaluate(&mut cpu, &Device::Cpu, &tokens, offset);
            let spread = expected.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - expected.iter().copied().fold(f32::INFINITY, f32::min);
            for (actual, expected) in actual.iter().zip(&expected) {
                max_error = max_error.max((actual - expected).abs());
                assert!(actual.is_finite() && expected.is_finite());
                assert!((actual - expected).abs() <= 0.05 * spread.max(1e-3));
            }
            for (row, (actual, expected)) in actual
                .chunks_exact(16)
                .zip(expected.chunks_exact(16))
                .enumerate()
            {
                let winner = |values: &[f32]| {
                    values
                        .iter()
                        .enumerate()
                        .max_by(|left, right| left.1.total_cmp(right.1))
                        .unwrap()
                        .0
                };
                if winner(actual) != winner(expected) {
                    mismatches += 1;
                    if mismatches <= 8 {
                        eprintln!("DeepSeek2 budget={budget:?} offset={offset} row={row}: CUDA winner={}, CPU winner={}", winner(actual), winner(expected));
                    }
                }
            }
            let row = &expected[expected.len() - 16..];
            let token = row
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .unwrap()
                .0 as u32;
            offset += tokens.len();
            tokens = if offset == 75 {
                vec![3, 8, 3]
            } else {
                vec![token]
            };
            gpu.wait_for_expert_uploads();
        }
        eprintln!("DeepSeek2 budget={budget:?} CUDA-dense vs all-CPU: greedy_mismatches={mismatches}, max_abs_error={max_error}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}
