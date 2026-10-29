//! Tier-3 (`--ssd-expert-cache`) dispatch — currently GATED OFF.
//!
//! The staging executor itself (`residency::NvmeResidency`) is unit-tested and
//! sound, but engaging it from the decode path changed model outputs: a staged
//! expert computes slightly different values from the same expert on the host
//! path, so the parity check below drifted ~1e-3 and the two CI feature jobs
//! even disagreed with each other run to run (different experts staged per run).
//!
//! Rather than loosen the parity assertion, engagement is disabled in
//! `resolve_ssd_expert_cache_bytes` until the divergence is localised.  These
//! tests pin that state so a future re-enable cannot silently regress it:
//!
//! * **Nothing stages.**  No `--ssd-expert-cache` setting builds a slab, so the
//!   loader keeps the exact pre-Tier-3 behaviour and `--ssd-expert-cache` is
//!   inert rather than corrupting outputs.
//! * **Outputs are exactly the historical ones.**  Every logit matches the
//!   all-host path bit for bit, because the host path *is* the path.
//!
//! When engagement is restored, these assertions invert back to checking that
//! the slab stages and that its logits are identical to the all-host run.

mod common;

use candle_core::Device;
use joshua::model::QuantizedModel;
use joshua::placement::SsdExpertCache;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

/// Load a mapped model with the given Tier-3 setting (the same entry point the
/// engine uses, so the flag is threaded exactly as it is in production).
fn load(path: &Path, ssd: SsdExpertCache) -> QuantizedModel {
    let bytes = std::fs::read(path).unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let content = header.to_candle_content().unwrap();
    let file = std::fs::File::open(path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    QuantizedModel::from_gguf_mmap_placed_with_ssd(
        content,
        &mut cursor,
        &Device::Cpu,
        &Device::Cpu,
        Some(Arc::new(mmap)),
        Some(Arc::new(file)),
        0,
        None,
        joshua::EngramTier::Auto,
        ssd,
    )
    .unwrap()
}

fn logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    common::logits(model, tokens, offset)
}

/// The tiny fixture's routed-expert pool is a few MiB; give the slab room for
/// all of it so a decode step's routing fits.
const SLAB_BUDGET: u64 = 64 * 1024 * 1024;

#[test]
fn ssd_cache_disabled_leaves_no_slab() {
    let dir = common::model_dir("ssd-cache-off");
    let path = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&path);

    for setting in [
        SsdExpertCache::Unset,
        SsdExpertCache::Disabled,
        SsdExpertCache::Bytes(0),
    ] {
        let model = load(&path, setting);
        assert!(
            model.ssd_expert_cache().is_none(),
            "{setting:?} must not build a staging slab"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// The flag is wired in: with a byte budget the load reports a slab, a decode
/// steps experts through it (resident bytes > 0), reads an expert back out of
/// it as host weights, and produces **the same logits** as the all-host run.
/// Gated: no setting engages the slab, and every setting therefore produces the
/// **exact** historical logits.  This is the regression guard for re-enabling
/// Tier-3 — when engagement comes back, this test must be rewritten to assert
/// staging *and* exact parity, so the change cannot land silently.
#[test]
fn ssd_cache_is_gated_off_and_outputs_are_unchanged() {
    let dir = common::model_dir("ssd-cache-gated");
    let path = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&path);

    let prompt: [u32; 8] = [1, 4, 2, 7, 5, 3, 8, 2];

    // The historical path, with no flag at all.
    let mut baseline = load(&path, SsdExpertCache::Unset);
    let expected = logits(&mut baseline, &prompt, 0);
    let expected_next = logits(&mut baseline, &[9], prompt.len());

    for setting in [
        SsdExpertCache::Unset,
        SsdExpertCache::Disabled,
        SsdExpertCache::Bytes(0),
        SsdExpertCache::Bytes(64 << 20),
        SsdExpertCache::Auto,
    ] {
        let mut model = load(&path, setting);
        assert!(
            model.ssd_expert_cache().is_none(),
            "{setting:?} must not engage the Tier-3 slab while engagement is gated"
        );
        assert_eq!(
            logits(&mut model, &prompt, 0),
            expected,
            "{setting:?} must produce the exact historical prefill logits"
        );
        assert_eq!(
            logits(&mut model, &[9], prompt.len()),
            expected_next,
            "{setting:?} must produce the exact historical decode logits"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// Repeating the gated load must be deterministic: the whole point of gating
/// engagement was that the staged path made decode logits vary run to run.
#[test]
fn gated_loads_are_deterministic_across_repeats() {
    let dir = common::model_dir("ssd-cache-gated-determinism");
    let path = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&path);
    let prompt: [u32; 8] = [1, 4, 2, 7, 5, 3, 8, 2];

    let mut first: Option<(Vec<f32>, Vec<f32>)> = None;
    for attempt in 0..4 {
        let mut model = load(&path, SsdExpertCache::Bytes(64 << 20));
        let got = (
            logits(&mut model, &prompt, 0),
            logits(&mut model, &[9], prompt.len()),
        );
        match &first {
            None => first = Some(got),
            Some(expect) => assert_eq!(&got, expect, "run {attempt} diverged"),
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
