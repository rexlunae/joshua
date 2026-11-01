#![cfg(feature = "cuda")]

mod common;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use std::sync::{Arc, Barrier, Mutex};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

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
fn cuda_default_stream_contract() {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    let Device::Cuda(device) = Device::new_cuda(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(device.cuda_stream().cu_stream() as usize, 2);
}

#[test]
fn cuda_owned_stream_shared_workspace_parity() {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    shared_workspace_parity(Device::new_cuda_with_stream(0).unwrap());
}

#[test]
fn cuda_default_owned_transfer_event_interop() {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    use candle_core::cuda_backend::{cudarc::driver::sys, CudaStorage};
    for device in [
        Device::new_cuda(0).unwrap(),
        Device::new_cuda_with_stream(0).unwrap(),
    ] {
        let Device::Cuda(cuda) = &device else {
            unreachable!()
        };
        let compute = cuda.cuda_stream();
        assert!(cuda.is_event_tracking());
        let expected = vec![1f32, 2., 3., 4.];
        for external in [
            compute.context().default_stream(),
            compute.context().per_thread_stream(),
            cuda.cuda_transfer_stream().unwrap(),
        ] {
            let input = external.clone_htod(&expected).unwrap();
            let uploaded = external
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                .unwrap();
            compute.wait(&uploaded).unwrap();
            let tensor = Tensor::from_storage(
                candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(input, cuda.clone())),
                (2, 2),
                candle_core::op::BackpropOp::none(),
                false,
            );
            let output = tensor.affine(2., 1.).unwrap();
            let finished = compute
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                .unwrap();
            external.wait(&finished).unwrap();
            let (storage, _) = output.storage_and_layout();
            let candle_core::Storage::Cuda(storage) = &*storage else {
                unreachable!()
            };
            let actual = external
                .clone_dtoh(storage.as_cuda_slice::<f32>().unwrap())
                .unwrap();
            assert_eq!(actual, vec![3., 5., 7., 9.]);
            external.synchronize().unwrap();
        }
        device.synchronize().unwrap();
    }
}

#[test]
fn cuda_owned_stream_capture_replay() {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    use candle_core::cuda_backend::cudarc::driver::sys;
    let device = Device::new_cuda_with_stream(0).unwrap();
    let Device::Cuda(cuda) = &device else {
        unreachable!()
    };
    let stream = cuda.cuda_stream();
    unsafe { cuda.disable_event_tracking() };
    let _cache = cuda.enable_cuda_graph_htod_cache();
    let input = Tensor::new(&[1f32, 2., 3., 4.], &device).unwrap();
    input.affine(2., 1.).unwrap().to_vec1::<f32>().unwrap();
    device.synchronize().unwrap();
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    let output = input.affine(2., 1.);
    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .unwrap()
        .expect("affine capture must contain GPU work");
    let output = output.unwrap();
    for _ in 0..3 {
        graph.launch().unwrap();
        device.synchronize().unwrap();
        assert_eq!(output.to_vec1::<f32>().unwrap(), vec![3., 5., 7., 9.]);
    }
}

#[test]
fn cuda_engine_auto_and_explicit_owned_stream() {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    let dir = common::model_dir("cuda-engine-owned-stream");
    let path = dir.join("model.gguf");
    common::write_tiny_llama_gguf_hd32(&path);
    std::fs::write(dir.join("tokenizer.json"), common::TOKENIZER_JSON).unwrap();
    let options = joshua::GenerationOptions {
        max_tokens: 4,
        temperature: 0.,
        ..Default::default()
    };
    for backend in [joshua::ComputeBackend::Auto, joshua::ComputeBackend::Cuda] {
        let engine = joshua::Engine::with_options(
            &path,
            joshua::EngineOptions::with_n_ctx(128).backend(backend),
        )
        .unwrap();
        let Device::Cuda(cuda) = engine.dense_device() else {
            panic!("CUDA must be available for {backend:?}")
        };
        assert_ne!(cuda.cuda_stream().cu_stream() as usize, 2);
        assert_ne!(cuda.cuda_stream().cu_stream() as usize, 0);
        let expected = engine.complete_raw("hello world", &options).unwrap().0;
        assert_eq!(
            engine.complete_raw("hello world", &options).unwrap().0,
            expected
        );
        std::thread::scope(|scope| {
            let first = scope.spawn(|| engine.complete_raw("hello world", &options).unwrap().0);
            let second = scope.spawn(|| engine.complete_raw("hello world", &options).unwrap().0);
            assert_eq!(first.join().unwrap(), expected);
            assert_eq!(second.join().unwrap(), expected);
        });
        engine.dense_device().synchronize().unwrap();
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "bounded isolated arithmetic diagnostic, not an acceptance gate"]
fn cuda_q4k_expert_arithmetic_probe() {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    use candle_core::quantized::QStorage;
    let device = Device::new_cuda_with_stream(0).unwrap();
    let read = |output: Tensor| output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for seed in [41, 73, 173, 271] {
        let weight =
            Tensor::from_vec(common::weights(256 * 256, seed), (256, 256), &Device::Cpu).unwrap();
        let quantized = QTensor::quantize(&weight, GgmlDType::Q4K).unwrap();
        let storage =
            QStorage::from_data_transfer(quantized.data().unwrap(), &device, quantized.dtype())
                .unwrap();
        let gpu = QMatMul::from_qtensor(QTensor::new(storage, quantized.shape().clone()).unwrap())
            .unwrap();
        let cpu = QMatMul::from_qtensor(quantized).unwrap();
        let input =
            Tensor::from_vec(common::weights(256, seed + 101), (1, 256), &Device::Cpu).unwrap();
        let expected = read(cpu.forward(&input).unwrap());
        let input_gpu = input.to_device(&device).unwrap();
        let fast = read(gpu.forward(&input_gpu).unwrap());
        device.synchronize().unwrap();
        candle_core::quantized::cuda::set_force_dmmv(true);
        let direct = read(gpu.forward(&input_gpu).unwrap());
        let cpu_quantized = QTensor::quantize(&input, GgmlDType::Q8K)
            .unwrap()
            .dequantize(&Device::Cpu)
            .unwrap();
        let matching_input = cpu_quantized.to_device(&device).unwrap();
        let compatible = read(gpu.forward(&matching_input).unwrap());
        let repeated = read(gpu.forward(&matching_input).unwrap());
        assert_eq!(compatible, repeated);
        device.synchronize().unwrap();
        candle_core::quantized::cuda::set_force_dmmv(false);
        let maximum = |actual: &[f32]| {
            actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0f32, f32::max)
        };
        assert!(maximum(&compatible) < 1e-6);
        eprintln!("Q4_K seed={seed}: CPU-Q8_K vs CUDA-Q8_1 max_abs={}, direct-DMMV max_abs={}, CPU-Q8_K-input+DMMV max_abs={}", maximum(&fast), maximum(&direct), maximum(&compatible));
    }
}
