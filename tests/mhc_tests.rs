//! Cached hyper-connection scales must preserve the tensor-scale API's
//! coefficients, including batched streams and nontrivial Sinkhorn rounds.
use candle_core::quantized::QMatMul;
use candle_core::{Device, Result, Tensor};
use joshua::mhc::{self, HyperConnection};

fn weights(hc: usize, d: usize, dev: &Device) -> Result<(QMatMul, Tensor, Tensor)> {
    let width = (2 + hc) * hc;
    let projection: Vec<f32> = (0..width * hc * d)
        .map(|i| ((i % 29) as f32 - 14.0) / 100.0)
        .collect();
    let base: Vec<f32> = (0..width).map(|i| i as f32 / 20.0 - 0.4).collect();
    Ok((
        QMatMul::Tensor(Tensor::from_vec(projection, (width, hc * d), dev)?),
        Tensor::from_vec(base, width, dev)?,
        // The loader historically accepts flattened scales with >=3 entries.
        Tensor::new(&[[0.3f32, -0.7], [1.2, 99.0]], dev)?,
    ))
}

fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    for (a, e) in actual.into_iter().zip(expected) {
        assert!(a.is_finite() && (a - e).abs() < 1e-6, "{a} != {e}");
    }
    Ok(())
}

fn check_cached_scales(dev: &Device) -> Result<()> {
    for hc in [1, 4] {
        let d = 8;
        let (projection, base, scale) = weights(hc, d, dev)?;
        let (cached_projection, cached_base, cached_scale) = weights(hc, d, dev)?;
        let cached = HyperConnection::new(cached_projection, cached_base, &cached_scale)?;
        for (batch, seq) in [(1, 1), (1, 5), (2, 3)] {
            let values: Vec<f32> = (0..batch * seq * hc * d)
                .map(|i| ((i % 17) as f32 - 8.0) / 10.0)
                .collect();
            let x = Tensor::from_vec(values, (batch, seq, hc, d), dev)?;
            for iters in [0, 1, 3] {
                let reference = mhc::mixes(&x, &projection, &scale, &base, 1e-6, iters, 1e-5)?;
                let (collapsed, mix) = cached.enter(&x, 1e-6, iters, 1e-5)?;
                assert_close(&mix.pre, &reference.pre)?;
                assert_close(&mix.post, &reference.post)?;
                assert_close(&mix.comb, &reference.comb)?;
                assert_close(&collapsed, &mhc::collapse(&x, &reference.pre)?)?;
            }
        }
    }
    Ok(())
}

#[test]
fn cached_scales_match_tensor_scales_cpu() -> Result<()> {
    check_cached_scales(&Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn cached_scales_match_tensor_scales_metal() -> Result<()> {
    let dev = match Device::new_metal(0) {
        Ok(dev) => dev,
        Err(e) => {
            eprintln!("skipping: no Metal device: {e}");
            return Ok(());
        }
    };
    check_cached_scales(&dev)
}

#[test]
fn short_scales_fail_at_construction() -> Result<()> {
    let (projection, base, _) = weights(4, 8, &Device::Cpu)?;
    let scale = Tensor::new(&[1f32, 2.0], &Device::Cpu)?;
    let err = HyperConnection::new(projection, base, &scale)
        .err()
        .expect("short scale");
    assert!(err.to_string().contains("has 2 entries, expected 3"));
    Ok(())
}

/// Run with --release --features metal --test mhc_tests -- --ignored --nocapture.
/// Compare the same mixing kernels with per-call device reads versus cached
/// scalars. Synchronize both timed regions so queued GPU work is included.
#[test]
#[ignore = "manual hyper-connection microbenchmark"]
fn benchmark_cached_scales() -> Result<()> {
    fn bench(dev: &Device, name: &str) -> Result<()> {
        let (projection, base, scale) = weights(4, 128, dev)?;
        let (cached_projection, cached_base, cached_scale) = weights(4, 128, dev)?;
        let cached = HyperConnection::new(cached_projection, cached_base, &cached_scale)?;
        let x = Tensor::ones((1, 1, 4, 128), candle_core::DType::F32, dev)?;
        let mut timings = [Vec::new(), Vec::new()];
        for round in 0..7 {
            // Alternate ordering to reduce warmup and clock bias.
            for kind in [round % 2, 1 - round % 2] {
                dev.synchronize()?;
                let start = std::time::Instant::now();
                for _ in 0..100 {
                    let mix = if kind == 0 {
                        mhc::mixes(&x, &projection, &scale, &base, 1e-6, 3, 1e-5)?
                    } else {
                        cached.mixes(&x, 1e-6, 3, 1e-5)?
                    };
                    std::hint::black_box(mix);
                }
                dev.synchronize()?;
                if round != 0 {
                    timings[kind].push(start.elapsed().as_secs_f64() * 1e4);
                }
            }
        }
        for times in &mut timings {
            times.sort_by(f64::total_cmp);
        }
        eprintln!(
            "{name}: tensor scales {:.1} us/call, cached {:.1} us/call",
            timings[0][3], timings[1][3]
        );
        Ok(())
    }
    bench(&Device::Cpu, "CPU")?;
    #[cfg(feature = "metal")]
    bench(&Device::new_metal(0)?, "Metal")?;
    Ok(())
}
