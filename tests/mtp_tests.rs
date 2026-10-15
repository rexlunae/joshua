//! MTP drafting from a GGUF's own NextN block (`blk.{n_layer}.nextn.*`):
//! the block's state follows the trunk however the trunk was fed, rolls back
//! with a verification pass, survives a snapshot, and never changes what the
//! trunk outputs.  CPU, tiny fixtures.

mod common;

use candle_core::{Device, Tensor};
use common::{assert_close, logits};
use joshua::model::QuantizedModel;
use joshua::stream_prefill::Chunk;

/// Fixtures with a NextN block: the Qwen ones written with one, and the GLM
/// ones, which carry one without `shared_head_norm`.
const ARCHES: &[&str] = &["qwen3moe", "qwen35moe", "qwen3next", "glm4", "glm4moe"];

fn mtp_model(arch: &str) -> QuantizedModel {
    let dir = common::model_dir(&format!("mtp-{arch}"));
    let path = dir.join("model.gguf");
    match arch {
        "glm4" | "glm4moe" => common::write_tiny_qwen_gguf(&path, arch),
        _ => common::write_tiny_qwen_mtp_gguf(&path, arch),
    }
    let mut m = common::load_model(&path, true);
    assert!(m.set_mtp(true), "{arch}: the NextN block loads");
    m
}

const PROMPT: [u32; 8] = [1, 4, 2, 7, 5, 9, 3, 6];
const NEXT: u32 = 11;

fn draft(m: &mut QuantizedModel, pos: usize) -> Vec<u32> {
    m.mtp_draft(NEXT, pos, 4).unwrap()
}

#[test]
fn mtp_block_follows_the_trunk_however_it_is_fed() {
    for &arch in ARCHES {
        let model = mtp_model(arch);
        let n = PROMPT.len();

        let mut whole = model.new_session().unwrap();
        logits(&mut whole, &PROMPT, 0);
        let want = draft(&mut whole, n);
        assert_eq!(want.len(), 4, "{arch}");
        // Drafting leaves the session as it was: the same draft again.
        assert_eq!(
            draft(&mut whole, n),
            want,
            "{arch}: drafting is side-effect free"
        );
        // The block is at `n`; any other position drafts nothing.
        assert!(draft(&mut whole, n + 1).is_empty(), "{arch}");

        let mut steps = model.new_session().unwrap();
        for (i, &t) in PROMPT.iter().enumerate() {
            logits(&mut steps, &[t], i);
        }
        assert_eq!(draft(&mut steps, n), want, "{arch}: token by token");

        let mut chunked = model.new_session().unwrap();
        logits(&mut chunked, &PROMPT[..3], 0);
        logits(&mut chunked, &PROMPT[3..], 3);
        assert_eq!(draft(&mut chunked, n), want, "{arch}: two chunks");

        let mut streamed = model.new_session().unwrap();
        streamed
            .prefill_streamed(
                &[
                    Chunk {
                        tokens: &PROMPT[..5],
                        pos: 0,
                    },
                    Chunk {
                        tokens: &PROMPT[5..],
                        pos: 5,
                    },
                ],
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(
            draft(&mut streamed, n),
            want,
            "{arch}: layer-streamed prefill"
        );

        // A cleared session starts over.
        whole.clear_kv_cache();
        logits(&mut whole, &PROMPT, 0);
        assert_eq!(draft(&mut whole, n), want, "{arch}: after a clear");
    }
}

#[test]
fn mtp_block_rolls_back_with_a_verification_pass() {
    for &arch in ARCHES {
        let model = mtp_model(arch);
        let n = PROMPT.len();
        // Verify [x, d1, d2, d3] and keep x and d1.
        let block = [NEXT, 8, 10, 12];
        let mut m = model.new_session().unwrap();
        logits(&mut m, &PROMPT, 0);
        let input = Tensor::new(&block[..], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        m.forward_all_logits(&input, n).unwrap();
        assert!(m.truncate_kv_cache(n + 2).unwrap(), "{arch}");

        let mut plain = model.new_session().unwrap();
        logits(&mut plain, &PROMPT, 0);
        logits(&mut plain, &block[..2], n);
        assert_eq!(
            draft(&mut m, n + 2),
            draft(&mut plain, n + 2),
            "{arch}: rolled back"
        );

        // Accepting the whole pass needs no rollback.
        let mut all = model.new_session().unwrap();
        logits(&mut all, &PROMPT, 0);
        all.forward_all_logits(&input, n).unwrap();
        all.discard_verify();
        let mut plain = model.new_session().unwrap();
        logits(&mut plain, &PROMPT, 0);
        logits(&mut plain, &block, n);
        assert_eq!(
            draft(&mut all, n + 4),
            draft(&mut plain, n + 4),
            "{arch}: all accepted"
        );
    }
}

#[test]
fn mtp_block_state_travels_with_a_snapshot() {
    for &arch in ARCHES {
        let model = mtp_model(arch);
        let n = PROMPT.len();
        let mut m = model.new_session().unwrap();
        logits(&mut m, &PROMPT, 0);
        let want = draft(&mut m, n);
        let snap = m.snapshot(n, n, &Device::Cpu).unwrap().unwrap();
        let copy = snap.duplicate(&Device::Cpu).unwrap();
        for snap in [snap, copy] {
            let mut r = model.new_session().unwrap();
            assert!(r.restore_snapshot(snap).unwrap(), "{arch}");
            assert_eq!(draft(&mut r, n), want, "{arch}: restored");
        }
    }
}

#[test]
fn mtp_leaves_the_trunk_output_unchanged() {
    for &arch in ARCHES {
        let on = mtp_model(arch);
        let mut off = on.new_session().unwrap();
        off.set_mtp(false);
        let mut on = on.new_session().unwrap();
        assert!(on.mtp_enabled() && !off.mtp_enabled(), "{arch}");
        let (a, b) = (logits(&mut on, &PROMPT, 0), logits(&mut off, &PROMPT, 0));
        assert_close(&a, &b, 1e-6, &format!("{arch}: prefill logits with MTP on"));
        let (a, b) = (logits(&mut on, &[NEXT], 8), logits(&mut off, &[NEXT], 8));
        assert_close(&a, &b, 1e-6, &format!("{arch}: decode logits with MTP on"));
        let input = Tensor::new(&[3u32, 5, 7][..], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let a = on
            .forward_all_logits(&input, 9)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = off
            .forward_all_logits(&input, 9)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_close(&a, &b, 1e-6, &format!("{arch}: all-row logits with MTP on"));
    }
}

#[test]
fn models_without_a_nextn_block_do_not_draft() {
    let dir = common::model_dir("mtp-none-qwen3moe");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen_gguf(&path, "qwen3moe");
    let mut m = common::load_model(&path, true);
    assert!(!m.set_mtp(true));
    assert!(!m.mtp_enabled());
    logits(&mut m, &PROMPT, 0);
    assert!(m.mtp_draft(NEXT, PROMPT.len(), 3).unwrap().is_empty());
}
