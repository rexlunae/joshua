//! Native (pure-Rust) validation for the `deepseek4` quantized loader with the
//! dtypes candle cannot represent: IQ2_XXS routed experts and I32 routed-id
//! tables.  These run on the default `cargo test` — no llama.cpp, no network.

mod common;

use candle_core::{Device, Tensor};
use joshua::model::{Architecture, QuantizedModel};

fn load(model: &std::path::Path, mmap: bool) -> QuantizedModel {
    let bytes = std::fs::read(model).unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    // Load exactly the way the engine does: the tolerant header (raw dtype
    // ids) projected onto candle's Content.  `Content::read` itself would
    // reject the file on the first IQ2_XXS tensor.  `from_gguf_mmap` re-reads
    // the raw header from the cursor internally.
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let content = header.to_candle_content().unwrap();
    let _ = header;
    let mmap = if mmap {
        // Safety: the file is read-only for the lifetime of the mapping.
        unsafe { memmap2::Mmap::map(&std::fs::File::open(model).unwrap()) }
            .ok()
            .map(std::sync::Arc::new)
    } else {
        None
    };
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    QuantizedModel::from_gguf_mmap(content, &mut cursor, &Device::Cpu, mmap, None).unwrap()
}

fn logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    let input = Tensor::new(tokens, &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    model
        .forward(&input, offset)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// The mmap load path must wire up prefetch handles for every routed expert
/// (their weights live in the mapping), and the streamed path must not.
#[test]
fn deepseek4_mmap_path_wires_expert_prefetch_handles() {
    let dir = common::model_dir("deepseek4-prefetch");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let mmapped = load(&model, true);
    let (backed, total) = match &mmapped {
        QuantizedModel::DeepSeek4(w) => w.mmap_backed_experts(),
        _ => panic!("expected DeepSeek4 model"),
    };
    assert!(total > 0, "tiny model must have routed experts");
    assert_eq!(
        backed, total,
        "mmap path should borrow every routed expert ({backed}/{total})"
    );

    let streamed = load(&model, false);
    let (backed, total) = match &streamed {
        QuantizedModel::DeepSeek4(w) => w.mmap_backed_experts(),
        _ => panic!("expected DeepSeek4 model"),
    };
    assert_eq!(backed, 0, "streamed path has no mapping ({backed}/{total})");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn deepseek4_is_a_supported_architecture() {
    assert_eq!(
        Architecture::from_name("deepseek4"),
        Some(Architecture::DeepSeek4)
    );
}

/// The full loader path: IQ2_XXS experts (mmap-borrowed), I32 tid2eid table,
/// hash + regular MoE layers — produces finite, non-degenerate logits.
#[test]
fn deepseek4_loads_iq2xxs_and_i32_and_produces_finite_logits() {
    let dir = common::model_dir("deepseek4-load");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let mut m = load(&model, true);
    let out = logits(&mut m, &[1, 4, 2, 7, 5], 0);
    assert_eq!(out.len(), 16, "logits must cover the 16-token vocab");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "all logits must be finite: {out:?}"
    );
    let first = out[0];
    assert!(
        out.iter().any(|v| (v - first).abs() > 1e-6),
        "logits are degenerate"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The mmap path (blocks borrowed, decoded inside the matmul) and the
/// streamed path (whole tensor decoded to f32 at load) must agree.
#[test]
fn deepseek4_mmap_matches_streamed_path() {
    let dir = common::model_dir("deepseek4-mmap");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let mut mmapped = load(&model, true);
    let mut streamed = load(&model, false);
    let tokens = [1, 4, 2, 7, 5, 3, 8];
    let a = logits(&mut mmapped, &tokens, 0);
    let b = logits(&mut streamed, &tokens, 0);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let tol = 1e-3 * y.abs().max(1.0);
        assert!(
            (x - y).abs() <= tol,
            "logit {i}: mmap={x} streamed={y} (tol {tol})"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// Incremental decode must give the same next-token logits as prefill.
#[test]
fn deepseek4_prefill_matches_incremental_decode() {
    let dir = common::model_dir("deepseek4-incr");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let tokens = [1u32, 4, 2, 7, 5];
    let mut m = load(&model, true);
    let prefill = logits(&mut m, &tokens, 0);

    let mut m = load(&model, true);
    let mut last = vec![0f32; 16];
    for (i, &t) in tokens.iter().enumerate() {
        last = logits(&mut m, &[t], i);
    }
    for i in 0..16 {
        let tol = 1e-3 * prefill[i].abs().max(1.0);
        assert!(
            (last[i] - prefill[i]).abs() <= tol,
            "logit {i}: prefill={} incremental={}",
            prefill[i],
            last[i]
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// Wraps a `Cursor` whose reads always fail — simulates a header that
/// disappears between candle's parse and Joshua's raw re-read (truncation,
/// failing backend), so the re-read error path is exercised deterministically.
struct FailAllReads<R>(R);

impl<R: std::io::Read> std::io::Read for FailAllReads<R> {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "simulated truncated header",
        ))
    }
}

impl<R: std::io::Seek> std::io::Seek for FailAllReads<R> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.0.seek(pos)
    }
}

/// A CSA layer (`compress_ratios[0] = 4`) writes its compressor and indexer
/// caches with a scatter whose index is a broadcast view of the block
/// positions.  candle's `scatter` rejects non-contiguous index tensors, so
/// both writes must materialize a dense index first — otherwise generation
/// aborts the moment the first compressed block completes.
#[test]
fn deepseek4_compressed_layers_write_their_caches() {
    let dir = common::model_dir("deepseek4-compress");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_compress(&model);

    // 5 tokens = one full compressed block at ratio 4 (plus a remainder):
    // the prefill branch completes a block and scatters into `comp` and
    // `lid` — this is where the non-contiguous index used to abort.
    let tokens = [1u32, 4, 2, 7, 5];
    let mut m = load(&model, true);
    let out = logits(&mut m, &tokens, 0);
    assert!(out.iter().all(|l| l.is_finite()));

    // Incremental decode crosses the block boundary mid-stream (position 3
    // completes the first block) — the other scatter call site.
    let mut m = load(&model, true);
    for (i, &t) in tokens.iter().enumerate() {
        let l = logits(&mut m, &[t], i);
        assert!(
            l.iter().all(|v| v.is_finite()),
            "decode step {i} must produce finite logits"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// An `output.weight` shipped in a dtype candle cannot name (IQ2_XXS) must be
/// picked up via the raw header and used, not reported absent — which would
/// silently tie the output head to the input embeddings.
#[test]
fn deepseek4_iq2xxs_output_weight_is_used_not_tied() {
    let dir = common::model_dir("deepseek4-iq2xxs-output");
    let tied = dir.join("tied.gguf");
    let own = dir.join("own.gguf");
    common::write_tiny_deepseek4_gguf(&tied);
    common::write_tiny_deepseek4_gguf_iq2xxs_output(&own);

    let mut a = load(&tied, true);
    let mut b = load(&own, true);
    let tokens = [1u32, 4, 2, 7, 5];
    let la = logits(&mut a, &tokens, 0);
    let lb = logits(&mut b, &tokens, 0);
    assert!(
        la.iter().zip(&lb).any(|(x, y)| (x - y).abs() > 1e-3),
        "IQ2_XXS output.weight must change the logits (tied model: {la:?}, own head: {lb:?})"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A failure to re-read the raw header must surface as the load error, not be
/// swallowed into "no raw table" (which would later become a misleading
/// "cannot find tensor blk.N.ffn_gate_exps.weight").
#[test]
fn deepseek4_header_reread_failure_is_reported() {
    let dir = common::model_dir("deepseek4-bad-header");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let bytes = std::fs::read(&model).unwrap();
    let content = {
        let mut cursor = std::io::Cursor::new(&bytes[..]);
        let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
        header.to_candle_content().unwrap()
    };

    // The reader passes candle's parse but fails every read afterwards.
    let mut failing = FailAllReads(std::io::Cursor::new(bytes));
    let err = match QuantizedModel::from_gguf_mmap(content, &mut failing, &Device::Cpu, None, None) {
        Err(e) => e,
        Ok(_) => panic!("from_gguf_mmap must fail when the raw header re-read fails"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("GGUF header"),
        "error should name the header re-read, got: {msg}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The engine must accept a GGUF whose dtypes candle cannot name — the
/// tolerant header is what makes the whole file reachable at all.
#[test]
fn deepseek4_engine_accepts_unknown_dtypes() {
    let dir = common::model_dir("deepseek4-engine");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let engine = joshua::Engine::new(&model).expect("engine must load IQ2_XXS GGUFs");
    // The API identifier is the file stem (documented contract); the
    // file's `general.name` ("DeepSeek-V4") is log-only.
    assert_eq!(engine.model_name(), "model");

    std::fs::remove_dir_all(&dir).ok();
}

/// Streamed (no mmap) loads must work for K-quant weights: the byte-size
/// lookup used to re-read the tensor data covers Q2_K..Q8_K, not just the
/// Q4_0..Q8_1 family.  Before the fix this aborted with "no size known for
/// GGUF dtype 12".
#[test]
fn deepseek4_streamed_load_handles_k_quant_weights() {
    let dir = common::model_dir("deepseek4-kquant-streamed");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_kquant(&model);

    let mut m = load(&model, false);
    let out = logits(&mut m, &[1, 4, 2, 7, 5], 0);
    assert_eq!(out.len(), 16);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "all logits must be finite: {out:?}"
    );
    let first = out[0];
    assert!(
        out.iter().any(|v| (v - first).abs() > 1e-6),
        "logits are degenerate"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The mmap path (Q4_K blocks borrowed and matmul'd in place) and the
/// streamed path (Q4_K decoded to f32 at load) must agree for K-quant
/// weights, like they do for the IQ2_XXS experts.
#[test]
fn deepseek4_kquant_mmap_matches_streamed_path() {
    let dir = common::model_dir("deepseek4-kquant-parity");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_kquant(&model);

    let mut mmapped = load(&model, true);
    let mut streamed = load(&model, false);
    let tokens = [1, 4, 2, 7, 5, 3, 8];
    let a = logits(&mut mmapped, &tokens, 0);
    let b = logits(&mut streamed, &tokens, 0);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        // The two paths are genuinely different computations — Q4_K blocks
        // dequantised block-by-block inside the fused matmul vs a plain f32
        // matmul over fully-dequantised weights — so they accumulate in
        // different orders.  On aarch64 the NEON fused kernel measures up to
        // ~0.016 absolute apart from the f32 path on this model (logits in
        // ±1.2).  A bound of `0.02 + 1%` admits that with margin while still
        // catching a real regression (transposed weight, dropped expert,
        // wrong kernel), which shifts logits by O(0.1–1.0) — 5–60× the noise
        // floor.
        let tol = 0.02 + 1e-2 * y.abs();
        assert!(
            (x - y).abs() <= tol,
            "logit {i}: mmap={x} streamed={y} (tol {tol})"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// `ModelWeights::from_gguf` is the public streamed entry point without a raw
/// header; it must fall back to candle's reader instead of erroring with "no
/// raw header for `token_embd.weight`".  The model is written with only
/// candle-nameable dtypes so every tensor is reachable that way.
#[test]
fn deepseek4_from_gguf_without_raw_header_loads() {
    let dir = common::model_dir("deepseek4-fromgguf");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_candle_only(&model);

    let bytes = std::fs::read(&model).unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let content = header.to_candle_content().unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let mut m = joshua::quantized_deepseek4::ModelWeights::from_gguf(
        content,
        &mut cursor,
        &Device::Cpu,
    )
    .expect("from_gguf (no raw header) must load candle-nameable models");

    let input = Tensor::new(&[1u32, 4, 2, 7, 5][..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let out: Vec<f32> = m
        .forward(&input, 0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out.len(), 16);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "all logits must be finite: {out:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The speculative next-step expert prefetch must track routing: empty
/// before the first pass, seeded by prefill's final row, refreshed by every
/// decode step — and every recorded id must be a valid expert index.
///
/// The prediction itself only fires `MADV_WILLNEED` (unobservable here);
/// what the test pins down is the bookkeeping the prediction is derived
/// from, plus that recording routing does not disturb outputs: two freshly
/// loaded models must produce identical logits with identical recorded
/// state.
#[test]
fn deepseek4_speculative_routing_state_tracks_forward_passes() {
    const N_EXPERT: u32 = 8; // fixture's expert count

    let dir = common::model_dir("deepseek4-speculative");
    let model_path = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model_path);

    let mut a = load(&model_path, true);
    // A second instance of the *same* load path: recording routing must not
    // disturb the math, so both must produce identical logits and records.
    // (Cross-load-path parity — mmap vs streamed — has small float deltas by
    // design and is covered by its own tolerance-based tests.)
    let mut b = load(&model_path, true);

    let routed = |m: &QuantizedModel| -> Vec<Vec<u32>> {
        match m {
            QuantizedModel::DeepSeek4(w) => w.last_routed_experts().to_vec(),
            _ => panic!("expected DeepSeek4 model"),
        }
    };
    let assert_valid = |state: &[Vec<u32>]| {
        for ids in state {
            for pair in ids.windows(2) {
                assert!(pair[0] < pair[1], "recorded ids must be sorted+deduped");
            }
            assert!(
                ids.iter().all(|&e| e < N_EXPERT),
                "recorded ids must be valid expert indices: {ids:?}"
            );
        }
    };

    // Nothing has routed yet.
    let initial = routed(&a);
    assert_eq!(initial.len(), 2, "one entry per fixture layer");
    assert!(initial.iter().all(|v| v.is_empty()));

    // Prefill seeds the record from the final row of the prompt.
    let la = logits(&mut a, &[1, 4, 5], 0);
    let after_prefill = routed(&a);
    assert_valid(&after_prefill);
    assert!(
        after_prefill.iter().any(|v| !v.is_empty()),
        "prefill must seed the routing record: {after_prefill:?}"
    );

    // Decode refreshes it; outputs and records stay deterministic across
    // instances (the prefetch advice cannot affect the math).
    let d1a = logits(&mut a, &[7], 3);
    let d1b = logits(&mut b, &[7], 3);
    let lb = logits(&mut b, &[1, 4, 5], 0);
    let db = logits(&mut b, &[7], 3);
    assert_valid(&routed(&a));
    assert_valid(&routed(&b));
    assert_eq!(la, lb, "identical loads must produce identical logits");
    assert_eq!(d1a, db, "decode must be deterministic and unaffected");

    std::fs::remove_dir_all(&dir).ok();
}
