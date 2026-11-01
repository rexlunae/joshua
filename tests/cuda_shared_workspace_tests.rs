#![cfg(feature = "cuda")]

mod common;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use std::sync::{Arc, Barrier};

fn shared_workspace_parity(device: Device) {
    let source = Tensor::from_vec(common::weights(512 * 1024, 173), (512, 1024), &device).unwrap();
    let weight = Arc::new(QMatMul::QTensor(Arc::new(
        QTensor::quantize(&source, GgmlDType::Q4K).unwrap(),
    )));
    let mut cases = Vec::with_capacity(9);
    let read = |output: Tensor| {
        output
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let dense = source
            .t()
            .unwrap()
            .contiguous()
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        for rows in [1, 8, 32] {
            let input = Tensor::from_vec(
                common::weights(rows * 1024, rows as u32 + 41),
                (rows, 1024),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let expected_quant = read(weight.forward(&input).unwrap());
            let expected_dense = read(input.matmul(&dense).unwrap());
            cases.push((input, dense.clone(), expected_quant, expected_dense));
        }
    }
    let start = Barrier::new(3);
    std::thread::scope(|scope| {
        for worker in 0..3 {
            let cases = &cases;
            let weight = &weight;
            let start = &start;
            scope.spawn(move || {
                start.wait();
                let mut pending = Vec::with_capacity(192);
                for iteration in 0..96 {
                    let index = (iteration + worker) % cases.len();
                    pending.push((index, "quantized", weight.forward(&cases[index].0).unwrap(), &cases[index].2));
                    pending.push((index, "dense", cases[index].0.matmul(&cases[index].1).unwrap(), &cases[index].3));
                }
                for (index, kind, output, expected) in pending {
                    let actual = read(output);
                    assert_eq!(actual.len(), expected.len());
                    for (position, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                        assert_eq!(actual, expected, "worker={worker}, kind={kind}, dtype={:?}, rows={}, position={position}", cases[index].0.dtype(), cases[index].0.dim(0).unwrap());
                    }
                }
            });
        }
    });
}

#[test]
fn cuda_default_shared_workspace_parity() {
    shared_workspace_parity(Device::new_cuda(0).unwrap());
}

#[test]
fn cuda_owned_stream_shared_workspace_parity() {
    shared_workspace_parity(Device::new_cuda_with_stream(0).unwrap());
}
