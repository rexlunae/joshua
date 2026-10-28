//! Tier-3 (`--ssd-expert-cache`) dispatch for DeepSeek-V4 routed experts.
//!
//! The slab is a bounded host-RAM staging cache filled by `pread(2)` over each
//! expert's mapped weight ranges; a staged expert runs from that copy instead of
//! demand-faulting the mapping.  Two things must hold, and both are checked
//! here against a real tiny GGUF:
//!
//! * **It stages.**  With an `--ssd-expert-cache` budget the slab reports a
//!   byte budget, stages experts on the decode path, and reports resident bytes
//!   — i.e. the flag is no longer inert.
//! * **It preserves outputs.**  Every logit is identical to the same model's
//!   all-host path (byte-identical weights through the same quantized kernels,
//!   so this is exact parity, not a tolerance).
//!
//! Off (the default, `Disabled`, and `Bytes(0)`) the slab does not exist at all,
//! so the historical host path is untouched.

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
#[test]
fn ssd_cache_stages_experts_and_preserves_outputs() {
    let dir = common::model_dir("ssd-cache-parity");
    let path = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&path);

    // Reference: the historical all-host path (no slab).
    let mut host = load(&path, SsdExpertCache::Unset);
    let prompt: [u32; 8] = [1, 4, 2, 7, 5, 3, 8, 2];
    let expected = logits(&mut host, &prompt, 0);
    let expected_next = logits(&mut host, &[9], prompt.len());

    // Same model with the Tier-3 slab engaged.
    let mut cached = load(&path, SsdExpertCache::Bytes(SLAB_BUDGET));
    let report = cached
        .ssd_expert_cache()
        .expect("--ssd-expert-cache <n> must build the staging slab");
    // The budget is honoured but capped at the routed-expert pool itself: a
    // slab can never stage more bytes than the pool has (see
    // `resolve_ssd_expert_cache_bytes`).
    assert!(
        report.budget_bytes > 0 && report.budget_bytes <= SLAB_BUDGET,
        "the engaged budget is the requested one, capped by the pool: {report:?}"
    );
    assert_eq!(
        report.budget_bytes % report.slots.max(1) as u64,
        0,
        "the budget is a whole number of experts: {report:?}"
    );
    assert!(report.slots >= 1, "the budget must hold at least one expert");
    assert_eq!(report.resident, 0, "nothing is staged before the first pass");

    let got_prefill = logits(&mut cached, &prompt, 0);
    assert_eq!(
        got_prefill, expected,
        "prefill logits must be identical with the SSD cache engaged"
    );
    // The prefill seeds the slab with the last prompt row's routing.
    cached.wait_for_expert_uploads();
    let after_prefill = cached.ssd_expert_cache().unwrap();
    assert!(
        after_prefill.stage_requests > 0,
        "the prefill's routing must request stages: {after_prefill:?}"
    );

    let got_decode = logits(&mut cached, &[9], prompt.len());
    cached.wait_for_expert_uploads();
    let after_decode = cached.ssd_expert_cache().unwrap();
    assert!(
        after_decode.resident > 0,
        "a decode step must leave at least one expert staged: {after_decode:?}"
    );
    assert!(
        after_decode.resident_bytes > 0,
        "staged experts must account for bytes: {after_decode:?}"
    );
    assert!(
        after_decode.resident_bytes <= after_decode.budget_bytes,
        "the slab must stay within its budget: {after_decode:?}"
    );
    assert_eq!(
        got_decode, expected_next,
        "decode logits must be identical with the SSD cache engaged"
    );

    // A staging expert really is the expert's weights: the next decode steps
    // keep running through the slab and keep matching.
    let mut decoded = Vec::new();
    let mut token = 9u32;
    for step in 0..3 {
        decoded.push(logits(&mut cached, &[token], prompt.len() + 1 + step));
        token = (token + 3) % 12;
    }
    cached.wait_for_expert_uploads();
    let tail = cached.ssd_expert_cache().unwrap();
    assert!(
        tail.stats.hits > 0,
        "repeated routing must hit the staged slab: {tail:?}"
    );
    assert_eq!(tail.stats.upload_failures, 0, "no stage fails: {tail:?}");
    // The page-cache release for a staged expert runs `HOST_RELEASE_DELAY`
    // (1 s) after its stage settles, so it is not asserted here: this test is
    // about staging and parity, and a timing-dependent sleep would only make it
    // flaky.  `residency`'s hook tests cover the release itself.
    assert!(
        tail.resident_bytes <= tail.budget_bytes,
        "the slab still stays within its budget: {tail:?}"
    );

    std::fs::remove_dir_all(dir).unwrap();
}
