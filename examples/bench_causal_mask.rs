//! Microbenchmark for the streamed-prefill causal mask: the host fill
//! pattern ([`joshua::moe::causal_mask`]) vs the on-device builder
//! ([`joshua::moe::CausalMask`], PR #129).
//!
//! The streamed prefill sweeps layer-outer / chunk-inner, so the *same*
//! `(chunk, position)` mask is requested once per layer: a 40-layer model
//! over a 4096-token prompt (8 × 512 chunks) builds 320 masks per prefill.
//! The builder derives the pattern from a cached device `arange` compare
//! instead of a scalar host fill + host→device copy of the whole matrix.
//!
//! **Microbenchmark, not an end-to-end measurement** (AGENTS.md): it
//! quantifies the per-call mask-construction cost on each backend, with
//! `Device::synchronize` inside every timed region.  Run with
//! `--features metal` for the Metal numbers; end-to-end prefill latency
//! depends on the model and is not measured here.

use std::time::Instant;

use candle_core::{Device, Result};
use joshua::moe::{causal_mask, CausalMask};

const LAYERS: usize = 40;
const CHUNK: usize = 512;
const PROMPT: usize = 4096;
const PASSES: usize = 7;

fn chunks() -> Vec<(usize, usize)> {
    (0..PROMPT)
        .step_by(CHUNK)
        .map(|pos| (CHUNK.min(PROMPT - pos), pos))
        .collect()
}

/// One full streamed-prefill sweep of mask builds: LAYERS × chunks.
/// Tensors drop normally — allocation and free are part of both paths' cost.
fn sweep_host(masks: &[(usize, usize)], dev: &Device) -> Result<()> {
    for _ in 0..LAYERS {
        for &(seq, pos) in masks {
            drop(causal_mask(seq, pos, dev)?);
        }
    }
    Ok(())
}

fn sweep_builder(builder: &mut CausalMask, masks: &[(usize, usize)]) -> Result<()> {
    for _ in 0..LAYERS {
        for &(seq, pos) in masks {
            drop(builder.mask(seq, pos)?);
        }
    }
    Ok(())
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn bench(backend: &str, dev: &Device, passes: usize) -> Result<()> {
    let masks = chunks();
    let calls = (LAYERS * masks.len()) as f64;

    // Warm kernels, page caches and the builder's position table; sync so the
    // warm-up does not bleed into the first timed region.
    sweep_host(&masks, dev)?;
    let mut builder = CausalMask::new(dev);
    sweep_builder(&mut builder, &masks)?;
    dev.synchronize()?;

    // Alternate old/new runs so drift lands on both; every timed region ends
    // in a synchronize (AGENTS.md: include device synchronization in
    // accelerator timings).
    let (mut host_ms, mut builder_ms) = (Vec::with_capacity(passes), Vec::with_capacity(passes));
    for _ in 0..passes {
        let t = Instant::now();
        sweep_host(&masks, dev)?;
        dev.synchronize()?;
        host_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        sweep_builder(&mut builder, &masks)?;
        dev.synchronize()?;
        builder_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }

    println!("{backend}: {LAYERS} layers x {} chunks = {} mask builds ([seq, pos+seq] f32)", masks.len(), calls as usize);
    println!(
        "  {backend}  host fill + upload : {:8.2} ms/sweep  ({:7.1} us/mask)",
        median(&mut host_ms),
        median(&mut host_ms) * 1e3 / calls,
    );
    println!(
        "  {backend}  CausalMask builder : {:8.2} ms/sweep  ({:7.1} us/mask)",
        median(&mut builder_ms),
        median(&mut builder_ms) * 1e3 / calls,
    );
    Ok(())
}

fn main() -> Result<()> {
    bench("cpu  ", &Device::Cpu, PASSES)?;
    #[cfg(feature = "metal")]
    match Device::new_metal(0) {
        Ok(dev) => bench("metal", &dev, PASSES)?,
        Err(e) => println!("metal: unavailable ({e})"),
    }
    Ok(())
}
