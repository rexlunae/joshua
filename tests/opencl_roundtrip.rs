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
