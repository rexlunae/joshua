//! Microbenchmark for the DeepSeek-V4 grouped-output diagonal select
//! (PR: replace the per-forward index rebuild + gather with narrow + cat).
//!
//! `Attention::forward` extracts y[s, g, r] = oa[s, g, g·o_lora_rank + r]
//! from the grouped output projection.  The old path refilled a host index
//! `Vec` (seq × o_groups × o_lora_rank u32), uploaded it, and gathered;
//! the new path takes each group's block with a zero-copy narrow and
//! concatenates — same rows, same order, no index tensor.
//!
//! **Microbenchmark, not an end-to-end measurement** (AGENTS.md): it times
//! the per-call op sequence at the two shapes the forward actually sees
//! (decode: one token; prefill: one 512-token chunk), with
//! `Device::synchronize` inside every timed region.  Run with
//! `--features metal` for the Metal numbers.

use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor, D};

const GROUPS: usize = 8;
const LORA_RANK: usize = 1024;
const CALLS: usize = 2000;

fn chunks() -> Vec<(usize, usize)> {
    vec![(1, 0), (512, 0)]
}

/// The old per-call sequence: refill the index pattern, upload, gather.
fn call_old(oa: &Tensor, seq: usize, dev: &Device) -> Result<Tensor> {
    let mut idxv = Vec::with_capacity(seq * GROUPS * LORA_RANK);
    for _ in 0..seq {
        for g in 0..GROUPS {
            for r in 0..LORA_RANK {
                idxv.push((g * LORA_RANK + r) as u32);
            }
        }
    }
    let idx = Tensor::from_vec(idxv, (seq, GROUPS, LORA_RANK), dev)?;
    oa.gather(&idx, D::Minus1)?.reshape((seq, GROUPS * LORA_RANK))
}

/// The new per-call sequence: zero-copy narrows + one concatenate.
fn call_new(oa: &Tensor, seq: usize) -> Result<Tensor> {
    let blocks: Vec<Tensor> = (0..GROUPS)
        .map(|g| {
            oa.narrow(1, g, 1)?
                .narrow(D::Minus1, g * LORA_RANK, LORA_RANK)?
                .squeeze(1)
        })
        .collect::<Result<_>>()?;
    Tensor::cat(&blocks, 1)?.contiguous()
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn bench(backend: &str, dev: &Device, passes: usize) -> Result<()> {
    for &(seq, _) in &chunks() {
        let oa = Tensor::randn(0f32, 1f32, (seq, GROUPS, GROUPS * LORA_RANK), dev)?
            .to_dtype(DType::F32)?;
        // Warm + parity: the two sequences must produce identical rows.
        let a = call_old(&oa, seq, dev)?;
        let b = call_new(&oa, seq)?;
        dev.synchronize()?;
        assert_eq!(
            a.flatten_all()?.to_vec1::<f32>()?,
            b.flatten_all()?.to_vec1::<f32>()?,
            "narrow+cat must be bit-identical to the index gather (seq={seq})"
        );
        let mut old_us = Vec::with_capacity(passes);
        let mut new_us = Vec::with_capacity(passes);
        for _ in 0..passes {
            let t = Instant::now();
            for _ in 0..CALLS {
                drop(call_old(&oa, seq, dev)?);
            }
            dev.synchronize()?;
            old_us.push(t.elapsed().as_secs_f64() * 1e6 / CALLS as f64);
            let t = Instant::now();
            for _ in 0..CALLS {
                drop(call_new(&oa, seq)?);
            }
            dev.synchronize()?;
            new_us.push(t.elapsed().as_secs_f64() * 1e6 / CALLS as f64);
        }
        println!(
            "{backend} seq={seq:>4}: index-rebuild+gather {:8.2} us/call   narrow+cat {:8.2} us/call   (median of {passes} x {CALLS} calls)",
            median(&mut old_us),
            median(&mut new_us),
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    bench("cpu  ", &Device::Cpu, 5)?;
    #[cfg(feature = "metal")]
    match Device::new_metal(0) {
        Ok(dev) => bench("metal", &dev, 5)?,
        Err(e) => println!("metal: unavailable ({e})"),
    }
    Ok(())
}
