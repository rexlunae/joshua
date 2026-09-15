#![cfg(feature = "opencl")]
//! OpenCL backend round-trip test (M1).
//!
//! Enabled only when joshua is built with the `opencl` cargo feature (which
//! turns on candle-core's vendored OpenCL backend and links system libOpenCL).
//!
//! Skips gracefully when no OpenCL GPU is present, so normal CI stays green.
//! On an OpenCL machine (the joshua host's Intel UHD 730 iGPU) run with:
//!   cargo test --features opencl --test opencl_roundtrip
//!
//! This is the host<->device f32 buffer round-trip through the public candle
//! API (Tensor::from_vec -> OpenClStorage -> to_vec1) — the same shape joshua
//! will use for its deepseek4 dense set.

use candle_core::{Device, Tensor};

/// Write f32 to an OpenCL device buffer and read it back bit-exactly.
#[test]
fn f32_roundtrip() -> candle_core::Result<()> {
    let device = match Device::opencl_if_available(0) {
        Ok(Device::Cpu) => {
            eprintln!("SKIP: no OpenCL device available on this host");
            return Ok(());
        }
        Ok(dev) => dev,
        Err(e) => {
            eprintln!("SKIP: opencl init failed: {e}");
            return Ok(());
        }
    };
    eprintln!("running OpenCL round-trip on device: {device:?}");

    let src = vec![1.5f32, 2.0, 3.0, -4.5, 0.0, 123.25, -987.125, 42.0];
    let a = Tensor::from_vec(src.clone(), (8,), &device)?;
    assert!(a.device().is_opencl(), "tensor should be on the OpenCL device");

    // Read back to CPU (hits OpenClStorage::to_cpu_storage).
    let cpu = a.to_device(&Device::Cpu)?;
    assert!(cpu.device().is_cpu());
    let v = cpu.to_vec1::<f32>()?;
    assert_eq!(v, src, "OpenCL host<->device f32 round-trip should be exact");

    // Also round-trip back to device (hits storage_from_cpu_storage).
    let back = cpu.to_device(&device)?;
    assert!(back.device().is_opencl());
    let v2 = back.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    assert_eq!(v2, src);

    eprintln!("PASS: OpenCL f32 buffer round-trip is exact");
    Ok(())
}

/// Build an OpenCL device, or return `None` to skip when none is present.
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

/// M2: run a representative set of operators on OpenClStorage and assert the
/// result equals the same computation on CPU. Because the OpenCL backend
/// forwards to candle's CPU kernels (M2 CPU-fallback strategy), the results
/// must match bit-exactly.
#[test]
fn operators_match_cpu() -> candle_core::Result<()> {
    let ocl = match opencl_or_skip() {
        Some(d) => d,
        None => return Ok(()),
    };
    eprintln!("running OpenCL operator parity on: {ocl:?}");

    let cpu = Device::Cpu;

    // 1) affine (RMSNorm-style scale+shift)
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &ocl)?;
    let a_cpu = a.to_device(&cpu)?;
    let o = a.affine(2.0, -1.0)?.to_device(&cpu)?.to_vec2::<f32>()?;
    let c = a_cpu.affine(2.0, -1.0)?.to_vec2::<f32>()?;
    assert_eq!(o, c, "affine parity");

    // 2) unary exp
    let o = a.exp()?.to_device(&cpu)?.to_vec2::<f32>()?;
    let c = a_cpu.exp()?.to_vec2::<f32>()?;
    assert_eq!(o, c, "exp parity");

    // 3) binary add (via broadcast)
    let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], (2, 2), &ocl)?;
    let b_cpu = b.to_device(&cpu)?;
    let o = a.broadcast_add(&b)?.to_device(&cpu)?.to_vec2::<f32>()?;
    let c = a_cpu.broadcast_add(&b_cpu)?.to_vec2::<f32>()?;
    assert_eq!(o, c, "binary add parity");

    // 4) matmul (attention GEMM) — the dominant dense op
    let m = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        (3, 4),
        &ocl,
    )?;
    let m_cpu = m.to_device(&cpu)?;
    let v = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (4,), &ocl)?;
    let v_cpu = v.to_device(&cpu)?;
    let o = m
        .matmul(&v.unsqueeze(1)?)?
        .to_device(&cpu)?
        .squeeze(1)?
        .to_vec1::<f32>()?;
    let c = m_cpu
        .matmul(&v_cpu.unsqueeze(1)?)?
        .squeeze(1)?
        .to_vec1::<f32>()?;
    assert_eq!(o, c, "matmul parity");

    // 5) sum (reduce)
    let o = a.sum_all()?.to_device(&cpu)?.to_scalar::<f32>()?;
    let c = a_cpu.sum_all()?.to_scalar::<f32>()?;
    assert_eq!(o, c, "sum parity");

    // 6) to_dtype
    let o = a.to_dtype(candle_core::DType::F64)?.to_vec2::<f64>()?;
    let c = a_cpu.to_dtype(candle_core::DType::F64)?.to_vec2::<f64>()?;
    assert_eq!(o, c, "to_dtype parity");

    // 7) index_select (embedding-style gather)
    let ids = Tensor::from_vec(vec![1u32, 0u32], (2,), &ocl)?;
    let ids_cpu = ids.to_device(&cpu)?;
    let o = a.index_select(&ids, 0)?.to_device(&cpu)?.to_vec2::<f32>()?;
    let c = a_cpu.index_select(&ids_cpu, 0)?.to_vec2::<f32>()?;
    assert_eq!(o, c, "index_select parity");

    eprintln!("PASS: OpenCL operator parity vs CPU is exact");
    Ok(())
}
