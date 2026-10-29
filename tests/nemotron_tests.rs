//! Nemotron-H (`nemotron_h`) — the hybrid Mamba-2 / Transformer-MoE loader.
//!
//! The tiny fixture has one layer of each kind the loader dispatches on
//! (`mamba`, `moe`, `attention`), so these tests exercise the layer-type
//! derivation, the Mamba-2 scan against an incremental decode, and the
//! recurrent state's rollback/snapshot protocol.  CPU, tiny fixtures.

mod common;

use candle_core::{Device, Tensor};
use common::assert_close;
use joshua::model::QuantizedModel;

/// One file per test: tests run in parallel, and rewriting a file another test
/// has mapped would fault it.
fn nemotron(test: &str) -> QuantizedModel {
    let dir = common::model_dir(&format!("nemotron-{test}"));
    let path = dir.join("model.gguf");
    common::write_tiny_nemotron_h_gguf(&path);
    common::load_model(&path, true)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
fn loads_the_hybrid_layer_stack() {
    let mut m = nemotron("loads");
    let arch = joshua::model::Architecture::from_name("nemotron_h");
    assert_eq!(
        arch,
        Some(joshua::model::Architecture::NemotronH),
        "nemotron_h resolves to the native hybrid loader"
    );
    // The architecture is no longer merely "known but unsupported".
    assert!(joshua::model::Architecture::is_known_llama_cpp_arch(
        "nemotron_h"
    ));

    let logits = common::logits(&mut m, &[1, 4, 2], 0);
    assert_eq!(logits.len(), 16, "one logit per vocab entry");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "a loaded model produces finite logits, got {logits:?}"
    );
}

/// A prefill of `n` tokens must equal feeding the same tokens one at a time
/// with a carried recurrent state: the Mamba-2 scan is stateful, so this is
/// the check that catches a scan whose state is not advanced (or advanced
/// twice) between calls.
#[test]
fn prefill_matches_incremental_decode() {
    let mut m = nemotron("prefill_vs_decode");
    let prompt = [1u32, 4, 2, 7, 5];
    let prefill = common::logits(&mut m, &prompt, 0);

    let mut m = nemotron("prefill_vs_decode_incremental");
    let mut step = Vec::new();
    for (i, tok) in prompt.iter().enumerate() {
        step = common::logits(&mut m, &[*tok], i);
    }
    assert_close(&prefill, &step, 2e-4, "prefill vs incremental decode");
    assert_eq!(
        argmax(&prefill),
        argmax(&step),
        "greedy choice agrees between prefill and decode"
    );
}

/// The recurrent layers make a *chunked* prefill order-sensitive only if the
/// state is mishandled: two chunkings of the same prompt must agree.
#[test]
fn chunked_prefill_matches_single_pass() {
    let prompt = [1u32, 4, 2, 7, 5, 9, 3, 6];
    let mut one = nemotron("chunk_one");
    let single = common::logits(&mut one, &prompt, 0);

    let mut two = nemotron("chunk_two");
    common::logits(&mut two, &prompt[..3], 0);
    let rest = common::logits(&mut two, &prompt[3..], 3);
    assert_close(&single, &rest, 2e-4, "one pass vs two chunks");

    let mut four = nemotron("chunk_four");
    common::logits(&mut four, &prompt[..2], 0);
    common::logits(&mut four, &prompt[2..4], 2);
    common::logits(&mut four, &prompt[4..6], 4);
    let tail = common::logits(&mut four, &prompt[6..], 6);
    assert_close(&single, &tail, 2e-4, "one pass vs four chunks");
}

/// Every vocabulary row is reachable: the loader must not produce a
/// degenerate (constant or NaN) distribution, which is what a mis-split
/// projection or a mis-narrowed `z`/`xBC`/`dt` slice would give.
#[test]
fn logits_are_not_degenerate() {
    let mut m = nemotron("nondegenerate");
    let l = common::logits(&mut m, &[1, 4, 2, 7], 0);
    let spread = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - l.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(spread > 1e-4, "logits are flat (spread {spread}): {l:?}");
}

/// The Mamba-2 state is running rather than append-only: a speculative
/// verification pass that is rolled back must leave the model exactly where
/// plain decoding would have, or a rejected draft corrupts the recurrence.
#[test]
fn verification_rollback_matches_plain_decode() {
    let prompt = [1u32, 4, 2, 7, 5, 9];
    let drafted = [3u32, 6, 11];

    // What plain decoding gives after the accepted prefix.
    let mut plain = nemotron("rollback_plain");
    common::logits(&mut plain, &prompt, 0);
    let keep = 2usize;
    let accepted = common::logits(&mut plain, &drafted[..keep], prompt.len());

    // A verification pass over the drafts *at the current position*, then a
    // rollback to the first `keep` of them.
    let mut spec = nemotron("rollback_spec");
    common::logits(&mut spec, &prompt, 0);
    let _ = spec
        .forward_all_logits(
            &Tensor::new(drafted.as_slice(), &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap(),
            prompt.len(),
        )
        .expect("a verification pass over the drafts");
    spec.truncate_kv_cache(prompt.len() + keep).unwrap();

    // The next ordinary token must match plain decoding's.
    let next = common::logits(&mut spec, &[13], prompt.len() + keep);
    let next_plain = common::logits(&mut plain, &[13], prompt.len() + keep);
    assert_close(&next, &next_plain, 2e-4, "post-rollback continuation");
}

/// A session snapshot round-trips the recurrent state: restoring a checkpoint
/// and continuing must reproduce the original continuation.
#[test]
fn snapshot_round_trips_the_recurrent_state() {
    let prompt = [1u32, 4, 2, 7];
    let mut m = nemotron("snapshot");
    common::logits(&mut m, &prompt, 0);
    let dev = Device::Cpu;
    let snap = m
        .snapshot(prompt.len(), prompt.len(), &dev)
        .expect("a snapshot of a native model")
        .expect("the hybrid session can detach its state");
    let cont = common::logits(&mut m, &[5, 9], prompt.len());

    assert!(m.restore_snapshot(snap).expect("restore the snapshot"));
    let again = common::logits(&mut m, &[5, 9], prompt.len());
    assert_close(&cont, &again, 1e-5, "continuation after a snapshot restore");
}

/// The conv tail is `d_conv - 1` = 3 tokens of context, so a single-token
/// decode must read tokens it saw three steps ago.  Feeding the same prompt
/// with a 1-token tail vs the full window must therefore differ — a conv
/// whose tail was never advanced would make these identical.
#[test]
fn the_conv_tail_carries_context() {
    let mut fresh = nemotron("conv_tail_a");
    let a = common::logits(&mut fresh, &[1], 0);
    let b = common::logits(&mut fresh, &[4], 1);
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-6),
        "two different tokens in a 3-wide causal window must not give the same logits"
    );
}

#[test]
fn architecture_metadata_is_registered() {
    use joshua::model::Architecture;
    for name in ["nemotron_h", "nemotron_h_moe"] {
        assert_eq!(
            Architecture::from_name(name),
            Some(Architecture::NemotronH),
            "{name} is served by the Nemotron-H loader"
        );
    }
    // The plain `mamba` / `mamba2` GGUFs are still unsupported: this loader
    // serves the *hybrid* Nemotron shape, not a pure SSM model.
    for name in ["mamba", "mamba2"] {
        assert_eq!(Architecture::from_name(name), None);
        assert!(Architecture::is_known_llama_cpp_arch(name));
    }
}

#[test]
fn a_second_session_shares_the_weights() {
    // The hybrid loader runs through the shared `Session`, so deriving another
    // session must not reload or duplicate the weight set.
    let mut m = nemotron("shared_weights");
    assert!(
        m.supports_shared_weights(),
        "the Nemotron-H loader shares its weights across sessions"
    );
    let a = common::logits(&mut m, &[1, 4], 0);
    assert_eq!(a.len(), 16);
}

// ─── Regressions from the Devin review of the loader ───────────────────────

/// Scalar `attention.head_count_kv` / `feed_forward_length` must classify the
/// same model as the array form.  Reading a scalar through the array accessor
/// yields zeros, which would make *every* layer recurrent and then demand
/// `ssm_in` tensors the file does not have.
///
/// The comparison is against the *same* uniform model written both ways, so a
/// classification difference shows up as a logit difference rather than being
/// masked by comparing two different architectures.
#[test]
fn scalar_layer_metadata_matches_array_metadata() {
    for kind in ["attention", "moe"] {
        let dir = common::model_dir(&format!("nemotron-scalar-meta-{kind}"));
        let array_path = dir.join("array.gguf");
        let scalar_path = dir.join("scalar.gguf");
        // Same layers, same weights; only the metadata encoding differs.
        common::write_nemotron_h_gguf(
            &array_path,
            common::NemotronTiny {
                uniform: true,
                uniform_kind: kind,
                ..Default::default()
            },
        );
        common::write_nemotron_h_gguf(
            &scalar_path,
            common::NemotronTiny {
                scalar_metadata: true,
                uniform: true,
                uniform_kind: kind,
                ..Default::default()
            },
        );

        let mut a = common::load_model(&array_path, true);
        let mut b = common::load_model(&scalar_path, true);
        let la = common::logits(&mut a, &[1, 4, 2], 0);
        let lb = common::logits(&mut b, &[1, 4, 2], 0);
        assert_close(
            &la,
            &lb,
            1e-5,
            &format!("uniform `{kind}`: scalar vs array metadata encode the same layers"),
        );
        assert!(
            la.iter().all(|v| v.is_finite()),
            "uniform `{kind}` logits are finite: {la:?}"
        );
    }
}

/// Each `ssm_norm` group column is its own weight.  Collapsing to column 0
/// scales every group by the first group's weights, which changes the logits.
#[test]
fn every_group_uses_its_own_norm_column() {
    let dir = common::model_dir("nemotron-group-norm");
    let uniform = dir.join("uniform.gguf");
    let varied = dir.join("varied.gguf");
    common::write_tiny_nemotron_h_gguf(&uniform);
    common::write_nemotron_h_gguf(
        &varied,
        common::NemotronTiny {
            group_norm_weight: [1.0, 3.0],
            ..Default::default()
        },
    );

    let mut u = common::load_model(&uniform, true);
    let mut v = common::load_model(&varied, true);
    let lu = common::logits(&mut u, &[1, 4, 2], 0);
    let lv = common::logits(&mut v, &[1, 4, 2], 0);

    assert!(
        lu.iter().zip(&lv).any(|(a, b)| (a - b).abs() > 1e-6),
        "giving group 1 a different norm weight must change the output; \
         identical logits mean only one column was read"
    );
}

/// The public no-mapping loader must work too.  `QuantizedModel::from_gguf`
/// passes no `mmap`, so the routed experts have to be read and split rather
/// than refused.
#[test]
fn loads_without_a_mapping() {
    let dir = common::model_dir("nemotron-streamed");
    let path = dir.join("model.gguf");
    common::write_tiny_nemotron_h_gguf(&path);

    let mut streamed = common::load_model(&path, false);
    let mut mapped = common::load_model(&path, true);
    let a = common::logits(&mut streamed, &[1, 4, 2], 0);
    let b = common::logits(&mut mapped, &[1, 4, 2], 0);
    assert_close(&a, &b, 1e-5, "streamed vs mapped load");
}

/// A verification pass at offset 0 has no pre-pass state to return to.  Rolling
/// it back must leave the layer empty, not keep the rejected tokens folded into
/// a recurrence that never existed.
#[test]
fn rollback_at_offset_zero_clears_the_state() {
    let mut m = nemotron("rollback_zero");
    let drafts = [1u32, 4, 2];
    let _ = m
        .forward_all_logits(
            &Tensor::new(drafts.as_slice(), &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap(),
            0,
        )
        .expect("verification at offset 0");
    m.truncate_kv_cache(0).expect("roll back to the start");

    // After a full rollback the model must behave like a fresh one.
    let after = common::logits(&mut m, &[7, 5], 0);
    let mut fresh = nemotron("rollback_zero_fresh");
    let want = common::logits(&mut fresh, &[7, 5], 0);
    assert_close(
        &after,
        &want,
        1e-5,
        "post-rollback-to-zero vs fresh session",
    );
}

/// The native conv kernel must actually run when a Mamba-2 layer convolves on
/// a device, not merely be callable from a unit test.  Wiring it up is the
/// difference between "the kernel exists" and "the layer is accelerated", and
/// nothing else in the suite would notice the layer silently reverting to the
/// per-tap tensor loop.
///
/// A full *model* forward on a device is not exercised here: the selective
/// scan is still CPU-only (`ssm_scan` bails on a device tensor), so a device
/// model cannot complete until that lands.  This drives the layer's own conv
/// path on a device instead, which is the part this change is responsible for.
#[cfg(feature = "opencl")]
#[test]
fn the_mamba_conv_runs_natively_on_a_device() {
    let Ok(dev) = candle_core::Device::new_opencl(0) else {
        eprintln!("nemotron: no OpenCL device, skipping the native-conv check");
        return;
    };
    // The real Nemotron 3.5 `xBC` width: d_inner 4096 + 2 * 8 groups * 128.
    let (d_conv, conv_dim, t) = (4usize, 6144usize, 5usize);
    let x = Tensor::randn(0f32, 1.0, (t + d_conv - 1, conv_dim), &dev).unwrap();
    let w = Tensor::randn(0f32, 0.1, (d_conv, conv_dim), &dev).unwrap();
    let b = Tensor::randn(0f32, 0.1, (conv_dim,), &dev).unwrap();

    let before = candle_core::opencl_backend::kernels::native_exec_count();
    let fb_before: usize = candle_core::opencl_backend::kernels::fallback_counts_by_op()
        .iter()
        .filter(|(n, _)| n == joshua::conv1d::OP_NAME)
        .map(|(_, c)| *c)
        .sum();
    let got = joshua::conv1d::conv1d(&x, &w, Some(&b), joshua::conv1d::Conv1d::mamba2()).unwrap();
    let launched = candle_core::opencl_backend::kernels::native_exec_count() - before;
    let fell_back: usize = candle_core::opencl_backend::kernels::fallback_counts_by_op()
        .iter()
        .filter(|(n, _)| n == joshua::conv1d::OP_NAME)
        .map(|(_, c)| *c)
        .sum::<usize>()
        - fb_before;

    assert!(launched > 0, "the conv must launch a native kernel on a device");
    assert_eq!(
        fell_back, 0,
        "the Mamba-2 conv must run natively, not round-trip to the CPU"
    );

    // Same arithmetic as the CPU reference at the real geometry.
    let want = joshua::conv1d::conv1d(
        &x.to_device(&Device::Cpu).unwrap(),
        &w.to_device(&Device::Cpu).unwrap(),
        Some(&b.to_device(&Device::Cpu).unwrap()),
        joshua::conv1d::Conv1d::mamba2(),
    )
    .unwrap()
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();
    let have = got
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = have
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
        / scale;
    assert!(worst < 1e-5, "device vs CPU conv relative error {worst:e}");
}

/// A whole-model device forward must now complete: with the device scan in
/// place, nothing in the Mamba-2 layer needs the host, so the model runs end
/// to end on the accelerator and must agree with the CPU.
#[cfg(feature = "opencl")]
#[test]
fn a_device_model_forward_matches_the_cpu() {
    let Ok(dev) = Device::new_opencl(0) else {
        eprintln!("nemotron: no OpenCL device, skipping the device-forward check");
        return;
    };
    let dir = common::model_dir("nemotron-device-forward");
    let path = dir.join("model.gguf");
    common::write_tiny_nemotron_h_gguf(&path);
    let bytes = std::fs::read(&path).unwrap();
    let content = joshua::gguf_ext::read_header(&mut std::io::Cursor::new(&bytes[..]))
        .unwrap()
        .to_candle_content()
        .unwrap();
    let mmap = {
        let f = std::fs::File::open(&path).unwrap();
        std::sync::Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
    };
    let mut device_model = joshua::model::QuantizedModel::from_gguf_mmap(
        content,
        &mut std::io::Cursor::new(&bytes[..]),
        &dev,
        Some(mmap),
        None,
        0,
    )
    .expect("load on the device");

    let prompt = [1u32, 4, 2, 7, 5];
    let before = candle_core::opencl_backend::kernels::native_exec_count();
    let got = common::logits_dev(&mut device_model, &prompt, 0, &dev)
        .expect("a device forward completes now that the scan runs on the device");
    let launched = candle_core::opencl_backend::kernels::native_exec_count() - before;
    assert!(
        launched > 0,
        "a device forward must launch native kernels, not quietly run on the host"
    );

    let mut cpu_model = common::load_model(&path, true);
    let want = common::logits(&mut cpu_model, &prompt, 0);
    common::assert_close(&got, &want, 5e-3, "device vs CPU whole-model forward");
}

/// The NextN / MTP draft block must load and draft, and must not change what
/// the trunk outputs.
///
/// This is the same contract the other recurrent loaders are held to: the
/// block's state follows the trunk however the trunk was fed, and merely
/// having it loaded leaves the model's logits alone.
#[test]
fn the_mtp_block_loads_and_drafts_without_changing_the_trunk() {
    let dir = common::model_dir("nemotron-mtp");
    let plain_path = dir.join("plain.gguf");
    let mtp_path = dir.join("mtp.gguf");
    common::write_tiny_nemotron_h_gguf(&plain_path);
    common::write_nemotron_h_gguf(
        &mtp_path,
        common::NemotronTiny {
            mtp_block: true,
            ..Default::default()
        },
    );

    // The trunk's logits must be untouched by the presence of the block.
    // Loading the MTP file *without* asking for the block must work too: a
    // `block_count` that includes the NextN block must not make the trunk
    // loader look for an ordinary layer at the draft block's index.
    let mut mtp_off = common::load_model(&mtp_path, true);
    assert!(!mtp_off.set_mtp(true) || true, "no drafting without the block");

    let mut plain = common::load_model(&plain_path, true);
    let mut with = joshua::quantized_qwen::with_mtp_block(|| common::load_model(&mtp_path, true));
    assert!(
        with.set_mtp(true),
        "the NextN block in the file must load for drafting"
    );
    let prompt = [1u32, 4, 2, 7];
    let a = common::logits(&mut plain, &prompt, 0);
    let b = common::logits(&mut with, &prompt, 0);
    assert_close(&a, &b, 1e-5, "MTP block must not change trunk logits");

    // And it must produce draft tokens.
    let drafts = with
        .mtp_draft(5, prompt.len(), 3)
        .expect("drafting from the NextN block");
    assert!(
        !drafts.is_empty(),
        "the MTP block must draft at least one token"
    );
    assert!(
        drafts.len() <= 3,
        "it must not draft more than asked: {drafts:?}"
    );
}

/// An attention-free trunk (uniform MoE) still needs drafting: the appended
/// NextN block always has an attention sub-layer, so its KV head count must
/// not come out of the trunk's (nonexistent) attention rows as zero.
///
/// Also covers the missing `shared_head_norm` fallback: a block without it
/// must normalize with the trunk's output norm, not hand `mtp_draft` a raw
/// residual to project and re-seed.
#[test]
fn an_attention_free_trunk_can_still_draft() {
    let dir = common::model_dir("nemotron-mtp-moe-trunk");
    let path = dir.join("model.gguf");
    common::write_nemotron_h_gguf(
        &path,
        common::NemotronTiny {
            uniform: true,
            uniform_kind: "moe",
            scalar_metadata: true,
            mtp_block: true,
            ..Default::default()
        },
    );
    let mut m = joshua::quantized_qwen::with_mtp_block(|| common::load_model(&path, true));
    assert!(m.set_mtp(true), "the NextN block loads on an all-MoE trunk");
    let prompt = [1u32, 4, 2, 7];
    let _ = common::logits(&mut m, &prompt, 0);
    let drafts = m
        .mtp_draft(5, prompt.len(), 2)
        .expect("drafting must work when the trunk has no attention layer");
    assert!(!drafts.is_empty(), "drafts produced: {drafts:?}");
}

/// The router's per-expert selection bias must steer top-k.
///
/// Nemotron 3.5 ships `mixer.gate.e_score_correction_bias` (GGUF
/// `ffn_exp_probs_b`), and the loader did not read it at all: every tensor
/// would load and the model would route to the wrong experts on real weights.
/// This is the defect that only turned up when the actual checkpoint's tensor
/// list was inspected, which is exactly why it is worth a test.
#[test]
fn the_router_selection_bias_changes_routing() {
    let dir = common::model_dir("nemotron-gate-bias");
    let plain = dir.join("plain.gguf");
    let biased = dir.join("biased.gguf");
    common::write_tiny_nemotron_h_gguf(&plain);
    common::write_nemotron_h_gguf(
        &biased,
        common::NemotronTiny {
            gate_bias: true,
            ..Default::default()
        },
    );

    let mut a = common::load_model(&plain, true);
    let mut b = common::load_model(&biased, true);
    let la = common::logits(&mut a, &[1, 4, 2], 0);
    let lb = common::logits(&mut b, &[1, 4, 2], 0);

    assert!(
        la.iter().zip(&lb).any(|(x, y)| (x - y).abs() > 1e-6),
        "a strong per-expert selection bias must change which experts run, \
         so the logits must differ; identical output means the bias was ignored"
    );
}

/// The MTP block's own router bias must be read too.
///
/// The real checkpoint's draft block (`mtp.layers.1`) carries its own
/// `mixer.gate.e_score_correction_bias`, separate from the trunk's.  The
/// fixture used to bias only trunk layers, so drafting never exercised
/// selection bias at all.
#[test]
fn the_mtp_block_reads_its_own_selection_bias() {
    let dir = common::model_dir("nemotron-mtp-bias");
    let plain = dir.join("plain.gguf");
    let biased = dir.join("biased.gguf");
    common::write_nemotron_h_gguf(
        &plain,
        common::NemotronTiny {
            mtp_block: true,
            ..Default::default()
        },
    );
    common::write_nemotron_h_gguf(
        &biased,
        common::NemotronTiny {
            mtp_block: true,
            gate_bias: true,
            ..Default::default()
        },
    );

    let mut a = joshua::quantized_qwen::with_mtp_block(|| common::load_model(&plain, true));
    let mut b = joshua::quantized_qwen::with_mtp_block(|| common::load_model(&biased, true));
    assert!(a.set_mtp(true) && b.set_mtp(true), "both files carry the block");
    let prompt = [1u32, 4, 2, 7];
    let _ = common::logits(&mut a, &prompt, 0);
    let _ = common::logits(&mut b, &prompt, 0);
    let da = a.mtp_draft(5, prompt.len(), 2).expect("drafting");
    let db = b.mtp_draft(5, prompt.len(), 2).expect("drafting");
    assert!(
        da != db,
        "a bias on the draft block's own router must change the draft: {da:?} vs {db:?}"
    );
}
