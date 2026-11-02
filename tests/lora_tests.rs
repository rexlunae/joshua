//! LoRA adapters on the native Qwen loader ([`joshua::lora`]): GGUF adapters
//! in llama.cpp's format, parsed and validated against the loaded model; the
//! per-projection delta `y = W·x + s·B(A·x) (+ b)`; and the session state
//! around it.  CPU, tiny fixtures from `tests/common`.
//!
//! The reference for every adapter run is a *merged* model: the same GGUF
//! with each targeted weight replaced by f32 `dequant(W) + s·B·A`, computed
//! in f64 and loaded by the unmodified loader.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::quantized::gguf_file::Value;
use candle_core::{Device, Tensor};
use common::{assert_close, logits, LoraDtype, LoraSpec, RawTensor};
use joshua::lora::{ActiveSet, AdapterKey, LoraAdapter, LoraLimits, SlotTable};
use joshua::model::QuantizedModel;

const PROMPT: [u32; 6] = [1, 4, 2, 7, 5, 9];

fn load(path: &Path) -> QuantizedModel {
    common::load_model(path, false)
}

/// A base fixture in its own directory (tests run in parallel).
fn fixture(test: &str, name: &str, write: impl FnOnce(&Path)) -> PathBuf {
    let dir = common::model_dir(&format!("lora-{test}-{name}"));
    let path = dir.join("model.gguf");
    write(&path);
    path
}

/// The fixtures the merged-model checks run on: name, writer, and the
/// tolerance against the merged model (a quantized base runs its fused
/// kernel where the merged model runs an f32 matmul).
type Writer = fn(&Path);
const FIXTURES: &[(&str, Writer, f32)] = &[
    (
        "qwen3-f32",
        |p| common::write_tiny_qwen_gguf(p, "qwen3"),
        1e-4,
    ),
    (
        "qwen3-q8_0",
        |p| common::write_tiny_qwen3_quantized_layers(p, 2),
        1e-3,
    ),
    (
        "qwen3-q4_k",
        |p| common::write_tiny_qwen3_q4k_layers(p, 2),
        1e-3,
    ),
    // Fused `attn_qkv` with a bias.
    ("qwen", |p| common::write_tiny_qwen_gguf(p, "qwen"), 1e-4),
    // Fused `attn_qkv` and fused `[gate ‖ up]`.
    (
        "chatglm",
        |p| common::write_tiny_qwen_gguf(p, "chatglm"),
        1e-4,
    ),
    // Sandwich norms, fused `[gate ‖ up]`, Q/K/V biases, a NextN block.
    ("glm4", |p| common::write_tiny_qwen_gguf(p, "glm4"), 1e-4),
    // Q/K/V biases and a gated shared expert.
    (
        "qwen2moe",
        |p| common::write_tiny_qwen_gguf(p, "qwen2moe"),
        1e-4,
    ),
    // Attention only: the FFN is all routed experts.
    (
        "qwen3moe",
        |p| common::write_tiny_qwen_gguf(p, "qwen3moe"),
        1e-4,
    ),
    // A DeltaNet layer (FFN only) then gated attention (doubled Q rows).
    (
        "qwen35",
        |p| common::write_tiny_qwen_gguf(p, "qwen35"),
        1e-4,
    ),
    // Hyper-connections, QSA attention and a shared expert.
    (
        "qwen4exp",
        |p| common::write_tiny_qwen_gguf(p, "qwen4exp"),
        1e-4,
    ),
];

/// One spec per slot of `table` whose tensor name `keep` accepts.
fn specs(table: &SlotTable, r: usize, seed: u32, keep: impl Fn(&str) -> bool) -> Vec<LoraSpec> {
    table
        .iter()
        .map(|(k, n_out, n_in)| (k.tensor_name(), n_out, n_in))
        .filter(|(name, _, _)| keep(name))
        .enumerate()
        .map(|(i, (name, n_out, n_in))| LoraSpec::new(&name, r, n_out, n_in, seed + 13 * i as u32))
        .collect()
}

fn write_adapter(
    dir: &Path,
    file: &str,
    table: &SlotTable,
    alpha: f32,
    specs: &[LoraSpec],
) -> PathBuf {
    let path = dir.join(file);
    common::write_lora_gguf(&path, table.arch(), alpha, specs, LoraDtype::F32);
    path
}

fn adapter(model: &QuantizedModel, path: &Path) -> Arc<LoraAdapter> {
    let table = model.lora_slots().expect("the qwen loader applies LoRA");
    Arc::new(LoraAdapter::load(path, &table, &LoraLimits::default()).unwrap())
}

fn active(selection: &[(Arc<LoraAdapter>, f32)]) -> Option<Arc<ActiveSet>> {
    ActiveSet::new(selection, &LoraLimits::default())
        .unwrap()
        .map(Arc::new)
}

/// llama.cpp's effective scale: `user · alpha / r`, or `user` when alpha is 0.
fn eff(user: f32, alpha: f32, r: usize) -> f32 {
    if alpha != 0.0 {
        user * alpha / r as f32
    } else {
        user
    }
}

/// The prompt's logits, then two decode steps'.
fn run(model: &mut QuantizedModel) -> Vec<Vec<f32>> {
    let n = PROMPT.len();
    vec![
        logits(model, &PROMPT, 0),
        logits(model, &[3], n),
        logits(model, &[9], n + 1),
    ]
}

fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn assert_runs_close(got: &[Vec<f32>], want: &[Vec<f32>], tol: f32, what: &str) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_close(g, w, tol, &format!("{what}, step {i}"));
    }
}

/// An adapter on every projection the table lists matches the merged model,
/// and visibly moves the logits away from the base model's.
#[test]
fn adapter_matches_merged_weights() {
    let (alpha, r) = (8.0, 4);
    for &(name, write, tol) in FIXTURES {
        // qwen4exp's hyper-connection mixers scale each layer's output well
        // down, so its delta needs a larger scale to show in the logits.
        let user = if name == "qwen4exp" { 50.0 } else { 0.8 };
        let path = fixture("merged", name, write);
        let dir = path.parent().unwrap();
        let mut model = load(&path);
        let table = model.lora_slots().unwrap();
        let specs = specs(&table, r, 100, |_| true);
        assert!(!specs.is_empty(), "{name}: no LoRA slots");
        let lora = adapter(
            &model,
            &write_adapter(dir, "adapter.gguf", &table, alpha, &specs),
        );
        assert_eq!(lora.targets().count(), specs.len(), "{name}");

        let base = run(&mut model.new_session().unwrap());
        assert!(
            model.set_adapters(active(&[(lora, user)])).unwrap(),
            "{name}"
        );
        let got = run(&mut model);

        let merged = dir.join("merged.gguf");
        let deltas: Vec<_> = specs.iter().map(|s| (s, eff(user, alpha, s.r))).collect();
        common::write_merged_gguf(&path, &merged, &deltas);
        let want = run(&mut load(&merged));

        assert_runs_close(&got, &want, tol, &format!("{name}: adapter vs merged"));
        let moved = max_diff(&got, &base);
        assert!(
            moved > 20.0 * tol,
            "{name}: the adapter barely moves the logits ({moved})"
        );
        std::fs::remove_dir_all(dir).ok();
    }
}

/// Two adapters at once (overlapping targets, one with alpha 0, one with a
/// negative scale) match the model with both deltas merged; the selection
/// order changes neither the key nor a single bit of the output.
#[test]
fn multiple_adapters_match_merged_weights() {
    let path = fixture("multi", "qwen3", |p| {
        common::write_tiny_qwen_gguf(p, "qwen3")
    });
    let dir = path.parent().unwrap();
    let model = load(&path);
    let table = model.lora_slots().unwrap();
    let x_specs = specs(&table, 4, 200, |n| n.contains(".attn_"));
    let y_specs = specs(&table, 2, 300, |n| {
        n.contains(".ffn_") || n.contains(".attn_q.")
    });
    let x = adapter(
        &model,
        &write_adapter(dir, "x.gguf", &table, 16.0, &x_specs),
    );
    let y = adapter(&model, &write_adapter(dir, "y.gguf", &table, 0.0, &y_specs));
    assert_ne!(x.digest(), y.digest());

    let (sx, sy) = (0.7, -0.4);
    let forward = active(&[(Arc::clone(&x), sx), (Arc::clone(&y), sy)]).unwrap();
    let reverse = active(&[(Arc::clone(&y), sy), (Arc::clone(&x), sx)]).unwrap();
    assert_eq!(forward.key(), reverse.key());
    assert_eq!(
        AdapterKey::of(&[(*y.digest(), sy), (*x.digest(), sx)]),
        Some(forward.key())
    );

    let mut a = model.new_session().unwrap();
    a.set_adapters(Some(forward)).unwrap();
    let got = run(&mut a);
    let mut b = model.new_session().unwrap();
    b.set_adapters(Some(reverse)).unwrap();
    assert_eq!(run(&mut b), got, "selection order");

    let mut deltas: Vec<_> = x_specs.iter().map(|s| (s, eff(sx, 16.0, s.r))).collect();
    deltas.extend(y_specs.iter().map(|s| (s, eff(sy, 0.0, s.r))));
    let merged = dir.join("merged.gguf");
    common::write_merged_gguf(&path, &merged, &deltas);
    assert_runs_close(
        &got,
        &run(&mut load(&merged)),
        1e-4,
        "two adapters vs merged",
    );
    std::fs::remove_dir_all(dir).ok();
}

/// Without an adapter the model computes exactly what it did before: a fresh
/// load, a session that ran an adapter and went back to the base model, a
/// sibling session over the same weights, and a scale-0 selection all give
/// the same bits.
#[test]
fn base_model_is_bit_identical_without_adapters() {
    for &(name, write, _) in FIXTURES {
        let path = fixture("bitwise", name, write);
        let dir = path.parent().unwrap();
        let want = run(&mut load(&path));

        let mut model = load(&path);
        let table = model.lora_slots().unwrap();
        let lora = adapter(
            &model,
            &write_adapter(
                dir,
                "adapter.gguf",
                &table,
                8.0,
                &specs(&table, 4, 400, |_| true),
            ),
        );
        assert!(
            ActiveSet::new(&[(Arc::clone(&lora), 0.0)], &LoraLimits::default())
                .unwrap()
                .is_none()
        );
        assert!(
            !model
                .set_adapters(active(&[(Arc::clone(&lora), 0.0)]))
                .unwrap(),
            "{name}: scale 0"
        );

        let mut m = model.new_session().unwrap();
        assert!(
            m.set_adapters(active(&[(Arc::clone(&lora), 1.0)])).unwrap(),
            "{name}"
        );
        let with = run(&mut m);
        assert_ne!(with, want, "{name}");
        let mut sibling = model.new_session().unwrap();
        assert_eq!(
            run(&mut sibling),
            want,
            "{name}: sibling of an adapter session"
        );
        assert!(
            m.set_adapters(None).unwrap(),
            "{name}: leaving the adapter clears the state"
        );
        assert_eq!(run(&mut m), want, "{name}: after an adapter request");
        assert_eq!(run(&mut model), want, "{name}: the template session");
        std::fs::remove_dir_all(dir).ok();
    }
}

/// Under an adapter, a whole-prompt prefill, token-by-token decode, chunked
/// prefill, a verification pass's every-position logits and the
/// layer-streaming prefill agree, as they do for the base model.
#[test]
fn adapter_prefill_decode_verify_and_streaming_agree() {
    use joshua::stream_prefill::Chunk;
    for name in ["qwen3-f32", "qwen3-q8_0", "qwen35", "qwen2moe"] {
        let &(_, write, _) = FIXTURES.iter().find(|f| f.0 == name).unwrap();
        let path = fixture("paths", name, write);
        let dir = path.parent().unwrap();
        let model = load(&path);
        let table = model.lora_slots().unwrap();
        let set = active(&[(
            adapter(
                &model,
                &write_adapter(
                    dir,
                    "adapter.gguf",
                    &table,
                    8.0,
                    &specs(&table, 4, 500, |_| true),
                ),
            ),
            1.0,
        )]);
        let session = || {
            let mut m = model.new_session().unwrap();
            m.set_adapters(set.clone()).unwrap();
            m
        };
        let n = PROMPT.len();
        let whole = logits(&mut session(), &PROMPT, 0);

        let mut steps = session();
        let decoded: Vec<Vec<f32>> = PROMPT
            .iter()
            .enumerate()
            .map(|(i, &t)| logits(&mut steps, &[t], i))
            .collect();
        assert_close(
            &whole,
            &decoded[n - 1],
            1e-4,
            &format!("{name}: prefill vs decode"),
        );

        let mut chunked = session();
        logits(&mut chunked, &PROMPT[..3], 0);
        let tail = logits(&mut chunked, &PROMPT[3..], 3);
        assert_close(&whole, &tail, 1e-4, &format!("{name}: chunked prefill"));

        let mut verify = session();
        let input = Tensor::new(&PROMPT[..], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let rows = verify
            .forward_all_logits(&input, 0)
            .unwrap()
            .squeeze(0)
            .unwrap();
        for (i, want) in decoded.iter().enumerate() {
            let row: Vec<f32> = rows.get(i).unwrap().to_vec1().unwrap();
            assert_close(&row, want, 1e-4, &format!("{name}: all-position row {i}"));
        }

        let mut streamed = session();
        let chunks = [
            Chunk {
                tokens: &PROMPT[..2],
                pos: 0,
            },
            Chunk {
                tokens: &PROMPT[2..],
                pos: 2,
            },
        ];
        let got: Vec<f32> = streamed
            .prefill_streamed(&chunks, &Device::Cpu)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_close(&whole, &got, 1e-4, &format!("{name}: streamed prefill"));
        let a = logits(&mut steps, &[3], n);
        let b = logits(&mut streamed, &[3], n);
        assert_close(
            &a,
            &b,
            1e-4,
            &format!("{name}: decode after streamed prefill"),
        );
        std::fs::remove_dir_all(dir).ok();
    }
}

/// A snapshot taken under an adapter set restores and continues exactly
/// under the same set, and is refused under any other (or none); a base
/// snapshot is refused under an adapter.
#[test]
fn snapshots_carry_the_adapter_key() {
    for name in ["qwen3-f32", "qwen35"] {
        let &(_, write, _) = FIXTURES.iter().find(|f| f.0 == name).unwrap();
        let path = fixture("snapshot", name, write);
        let dir = path.parent().unwrap();
        let model = load(&path);
        let table = model.lora_slots().unwrap();
        let x_path = write_adapter(dir, "x.gguf", &table, 8.0, &specs(&table, 4, 600, |_| true));
        let x = adapter(&model, &x_path);
        let y = adapter(
            &model,
            &write_adapter(dir, "y.gguf", &table, 8.0, &specs(&table, 4, 700, |_| true)),
        );
        let xs = active(&[(Arc::clone(&x), 1.0)]);

        common::check_snapshot_restore_with(&model, &format!("{name} + lora"), &|m| {
            assert!(m.set_adapters(xs.clone()).unwrap());
        });

        let n = PROMPT.len();
        let mut m = model.new_session().unwrap();
        m.set_adapters(xs.clone()).unwrap();
        logits(&mut m, &PROMPT, 0);
        let snap = m.snapshot(n, n, &Device::Cpu).unwrap().unwrap();
        assert_eq!(snap.adapter_key(), xs.as_ref().map(|s| s.key()), "{name}");
        let copy = || snap.duplicate(&Device::Cpu).unwrap();

        let mut base = model.new_session().unwrap();
        assert!(
            !base.restore_snapshot(copy()).unwrap(),
            "{name}: into the base model"
        );
        let mut other = model.new_session().unwrap();
        other
            .set_adapters(active(&[(Arc::clone(&y), 1.0)]))
            .unwrap();
        assert!(
            !other.restore_snapshot(copy()).unwrap(),
            "{name}: into another adapter"
        );
        let mut scaled = model.new_session().unwrap();
        scaled
            .set_adapters(active(&[(Arc::clone(&x), 0.5)]))
            .unwrap();
        assert!(
            !scaled.restore_snapshot(copy()).unwrap(),
            "{name}: at another scale"
        );

        // The same file loaded again is the same set.
        let mut same = model.new_session().unwrap();
        same.set_adapters(active(&[(adapter(&model, &x_path), 1.0)]))
            .unwrap();
        assert!(
            same.restore_snapshot(copy()).unwrap(),
            "{name}: same adapter, new load"
        );
        assert_eq!(
            logits(&mut same, &[3], n),
            logits(&mut m, &[3], n),
            "{name}: continues exactly"
        );

        let mut b = model.new_session().unwrap();
        logits(&mut b, &PROMPT, 0);
        let base_snap = b.snapshot(n, n, &Device::Cpu).unwrap().unwrap();
        assert_eq!(base_snap.adapter_key(), None);
        let mut under = model.new_session().unwrap();
        under.set_adapters(xs.clone()).unwrap();
        assert!(
            !under.restore_snapshot(base_snap).unwrap(),
            "{name}: base snapshot under an adapter"
        );
        std::fs::remove_dir_all(dir).ok();
    }
}

/// Changing the adapter set clears the session's state (and says so);
/// setting the same set again, even rebuilt from the same files, keeps it.
#[test]
fn set_adapters_clears_state_only_when_the_set_changes() {
    let path = fixture("clear", "qwen3", |p| {
        common::write_tiny_qwen_gguf(p, "qwen3")
    });
    let dir = path.parent().unwrap();
    let model = load(&path);
    let table = model.lora_slots().unwrap();
    let lora_path = write_adapter(
        dir,
        "adapter.gguf",
        &table,
        8.0,
        &specs(&table, 4, 800, |_| true),
    );
    let lora = adapter(&model, &lora_path);
    let set = active(&[(Arc::clone(&lora), 1.0)]);
    let n = PROMPT.len();
    let reference = |steps: &[(&[u32], usize)]| {
        let mut r = model.new_session().unwrap();
        r.set_adapters(set.clone()).unwrap();
        steps
            .iter()
            .map(|&(t, at)| logits(&mut r, t, at))
            .last()
            .unwrap()
    };

    let mut m = model.new_session().unwrap();
    assert_eq!(m.adapter_key(), None);
    logits(&mut m, &PROMPT, 0);
    assert!(m.set_adapters(set.clone()).unwrap(), "base → adapter");
    assert_eq!(m.adapter_key(), set.as_ref().map(|s| s.key()));
    // The base model's cache is gone: the prompt is read again from 0.
    assert_eq!(logits(&mut m, &PROMPT, 0), reference(&[(&PROMPT, 0)]));
    assert!(!m.set_adapters(set.clone()).unwrap(), "same set");
    assert!(
        !m.set_adapters(active(&[(adapter(&model, &lora_path), 1.0)]))
            .unwrap(),
        "same files"
    );
    assert_eq!(
        logits(&mut m, &[3], n),
        reference(&[(&PROMPT, 0), (&[3], n)]),
        "state kept"
    );
    assert!(
        m.set_adapters(active(&[(Arc::clone(&lora), 2.0)])).unwrap(),
        "new scale"
    );
    assert!(m.set_adapters(None).unwrap(), "adapter → base");
    assert!(!m.set_adapters(None).unwrap());
    std::fs::remove_dir_all(dir).ok();
}

/// The MTP block drafts without the adapter while the trunk runs it.
#[test]
fn mtp_drafting_runs_beside_an_adapter() {
    let path = fixture("mtp", "qwen35moe", |p| {
        common::write_tiny_qwen_mtp_gguf(p, "qwen35moe")
    });
    let dir = path.parent().unwrap();
    let mut model = joshua::quantized_qwen::with_mtp_block(|| load(&path));
    assert!(model.set_mtp(true));
    let table = model.lora_slots().unwrap();
    assert_eq!(table.n_layer(), 2, "the NextN block is no LoRA target");
    let set = active(&[(
        adapter(
            &model,
            &write_adapter(
                dir,
                "adapter.gguf",
                &table,
                8.0,
                &specs(&table, 4, 900, |_| true),
            ),
        ),
        1.0,
    )]);
    let mut m = model.new_session().unwrap();
    let base = logits(&mut m, &PROMPT, 0);
    m.set_adapters(set).unwrap();
    let with = logits(&mut m, &PROMPT, 0);
    assert_ne!(base, with);
    assert_eq!(m.mtp_draft(11, PROMPT.len(), 3).unwrap().len(), 3);
    std::fs::remove_dir_all(dir).ok();
}

/// F16, BF16 and Q8_0 adapter tensors decode to exactly the values an F32
/// adapter holding them gives.
#[test]
fn adapter_tensor_types_decode_exactly() {
    let path = fixture("dtypes", "qwen3-q8_0", |p| {
        common::write_tiny_qwen3_quantized_layers(p, 2)
    });
    let dir = path.parent().unwrap();
    let model = load(&path);
    let table = model.lora_slots().unwrap();
    // Rank 32 keeps every row a whole number of Q8_0 blocks.
    let specs = specs(&table, 32, 1000, |n| n.starts_with("blk.1."));
    let logits_with = |path: &Path| {
        let mut m = model.new_session().unwrap();
        m.set_adapters(active(&[(adapter(&model, path), 0.5)]))
            .unwrap();
        run(&mut m)
    };
    for dtype in [LoraDtype::F16, LoraDtype::Bf16, LoraDtype::Q8_0] {
        let typed = dir.join(format!("{dtype:?}.gguf"));
        common::write_lora_gguf(&typed, "qwen3", 32.0, &specs, dtype);
        let stored: Vec<_> = specs.iter().map(|s| s.stored(dtype)).collect();
        let exact = write_adapter(dir, &format!("{dtype:?}-f32.gguf"), &table, 32.0, &stored);
        assert_eq!(logits_with(&typed), logits_with(&exact), "{dtype:?}");
    }
    std::fs::remove_dir_all(dir).ok();
}

fn load_err(
    table: &SlotTable,
    md: Vec<(String, Value)>,
    tensors: &[RawTensor],
    limits: &LoraLimits,
    dir: &Path,
) -> String {
    let path = dir.join("bad.gguf");
    common::write_raw_gguf(&path, &md, tensors);
    match LoraAdapter::load(&path, table, limits) {
        Ok(_) => panic!("loaded"),
        Err(e) => e.to_string(),
    }
}

/// Every refusal of `llama-adapter.cpp`, plus the targets this version takes
/// no delta on, fails the load with a message naming the problem.
#[test]
fn invalid_adapters_are_refused() {
    let path = fixture("invalid", "qwen3", |p| {
        common::write_tiny_qwen_gguf(p, "qwen3")
    });
    let dir = path.parent().unwrap();
    let table = load(&path).lora_slots().unwrap();
    let limits = LoraLimits::default();
    let good = LoraSpec::new("blk.0.attn_q.weight", 4, 16, 8, 1);
    let tensors = |s: &LoraSpec| common::lora_tensors(std::slice::from_ref(s), LoraDtype::F32);
    let md = || common::lora_metadata("qwen3", 8.0);
    let with = |key: &str, v: Value| {
        let mut m: Vec<_> = md().into_iter().filter(|(k, _)| k != key).collect();
        m.push((key.into(), v));
        m
    };
    let s = |v: &str| Value::String(v.into());

    // The well-formed adapter loads, with a norm vector skipped as llama.cpp
    // skips it.
    let mut ok = tensors(&good);
    ok.push(RawTensor::f32("blk.0.attn_norm.weight", vec![1.0; 8], &[8]));
    let ok_path = dir.join("ok.gguf");
    common::write_raw_gguf(&ok_path, &md(), &ok);
    let lora = LoraAdapter::load(&ok_path, &table, &limits).unwrap();
    assert_eq!(lora.alpha(), 8.0);
    assert_eq!(lora.scale(1.0, 4), 2.0);
    assert_eq!(lora.bytes(), 4 * 4 * (16 + 8));

    let bad_shape = LoraSpec::new("blk.0.attn_q.weight", 4, 16, 9, 1);
    let mut untransposed = tensors(&good);
    untransposed[1] = RawTensor::f32("blk.0.attn_q.weight.lora_b", vec![0.01; 16 * 5], &[16, 5]);
    let mut nan = good.clone();
    nan.a[3] = f32::NAN;
    let target = |base: &str, n_out, n_in| tensors(&LoraSpec::new(base, 2, n_out, n_in, 3));
    // (what, metadata, tensors, expected error text)
    type Case<'a> = (&'a str, Vec<(String, Value)>, Vec<RawTensor>, &'a str);
    let cases: Vec<Case> = vec![
        (
            "general.type",
            with("general.type", s("model")),
            tensors(&good),
            "expect general.type to be 'adapter'",
        ),
        (
            "architecture",
            with("general.architecture", s("llama")),
            tensors(&good),
            "arch mismatch",
        ),
        (
            "adapter.type",
            with("adapter.type", s("control_vector")),
            tensors(&good),
            "expect adapter.type to be 'lora'",
        ),
        (
            "alpha type",
            with("adapter.lora.alpha", Value::U32(8)),
            tensors(&good),
            "adapter.lora.alpha",
        ),
        (
            "aLoRA",
            with(
                "adapter.alora.invocation_tokens",
                Value::Array(vec![Value::U32(5), Value::U32(6)]),
            ),
            tensors(&good),
            "aLoRA",
        ),
        (
            "missing half",
            md(),
            tensors(&good).into_iter().take(1).collect(),
            "missing one component",
        ),
        (
            "suffix",
            md(),
            vec![RawTensor::f32(
                "blk.0.attn_q.weight",
                vec![0.0; 128],
                &[16, 8],
            )],
            "unexpected suffix",
        ),
        ("shape", md(), tensors(&bad_shape), "incorrect shape"),
        ("transposed", md(), untransposed, "not transposed"),
        ("non-finite", md(), tensors(&nan), "non-finite"),
        (
            "no pairs",
            md(),
            vec![RawTensor::f32("blk.0.attn_norm.weight", vec![1.0; 8], &[8])],
            "no LoRA tensor pairs",
        ),
        (
            "missing base",
            md(),
            target("blk.0.attn_qkv.weight", 32, 8),
            "does not exist in base model",
        ),
        (
            "embedding",
            md(),
            target("token_embd.weight", 16, 8),
            "token embedding",
        ),
        (
            "output",
            md(),
            target("output.weight", 16, 8),
            "output head",
        ),
        (
            "past the trunk",
            md(),
            target("blk.2.attn_q.weight", 16, 8),
            "past the model",
        ),
        (
            "not a projection",
            md(),
            target("blk.0.attn_q_norm.weight", 8, 8),
            "not a projection",
        ),
    ];
    for (what, md, tensors, want) in cases {
        let err = load_err(&table, md, &tensors, &limits, dir);
        assert!(err.contains(want), "{what}: {err}");
    }

    let rank = LoraLimits {
        max_rank: 2,
        ..limits
    };
    assert!(load_err(&table, md(), &tensors(&good), &rank, dir).contains("rank 4"));
    let bytes = LoraLimits {
        max_bytes: 300,
        ..limits
    };
    assert!(load_err(&table, md(), &tensors(&good), &bytes, dir).contains("limit"));
    let count = LoraLimits {
        max_adapters: 1,
        ..limits
    };
    let lora = Arc::new(lora);
    let two = [(Arc::clone(&lora), 1.0), (Arc::clone(&lora), 0.5)];
    assert!(ActiveSet::new(&two, &count).is_err());
    assert!(ActiveSet::new(&[(Arc::clone(&lora), f32::NAN)], &limits).is_err());
    std::fs::remove_dir_all(dir).ok();
}

/// A loaded model's slot table lists exactly the projections a delta
/// reaches: attention and dense or shared-expert FFN weights, never routed
/// experts, routers or DeltaNet projections.
#[test]
fn slot_tables_list_what_the_loader_applies() {
    let attn = ["attn_q", "attn_k", "attn_v", "attn_output"];
    let dense = ["ffn_gate", "ffn_up", "ffn_down"];
    let shexp = ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"];
    let blk = |l: usize, names: &[&str]| -> Vec<String> {
        names
            .iter()
            .map(|n| format!("blk.{l}.{n}.weight"))
            .collect()
    };
    let cat = |parts: Vec<Vec<String>>| -> Vec<String> { parts.concat() };
    for (arch, want) in [
        (
            "qwen3",
            cat(vec![
                blk(0, &attn),
                blk(0, &dense),
                blk(1, &attn),
                blk(1, &dense),
            ]),
        ),
        ("qwen3moe", blk(0, &attn)),
        (
            "qwen2moe",
            cat(vec![
                blk(0, &attn),
                blk(0, &shexp),
                blk(1, &attn),
                blk(1, &shexp),
            ]),
        ),
        (
            "chatglm",
            cat(vec![
                blk(0, &["attn_qkv", "attn_output", "ffn_up", "ffn_down"]),
                blk(1, &["attn_qkv", "attn_output", "ffn_up", "ffn_down"]),
            ]),
        ),
        (
            "qwen35",
            cat(vec![blk(0, &dense), blk(1, &attn), blk(1, &dense)]),
        ),
    ] {
        let path = fixture("slots", arch, |p| common::write_tiny_qwen_gguf(p, arch));
        let table = load(&path).lora_slots().unwrap();
        assert_eq!(table.arch(), arch);
        let mut got: Vec<String> = table.iter().map(|(k, _, _)| k.tensor_name()).collect();
        let mut want = want;
        got.sort();
        want.sort();
        assert_eq!(got, want, "{arch}");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}

/// Routed experts, routers, Gated DeltaNet projections and a NextN block are
/// refused by name; attention on a MoE model and attention next to a DeltaNet
/// layer load.
#[test]
fn unsupported_targets_are_refused_by_name() {
    let limits = LoraLimits::default();
    for (arch, base, (n_out, n_in), want) in [
        (
            "qwen3moe",
            "blk.0.ffn_gate_exps.weight",
            (8, 8),
            Some("routed experts"),
        ),
        (
            "qwen3moe",
            "blk.0.ffn_gate_inp.weight",
            (4, 8),
            Some("router"),
        ),
        ("qwen3moe", "blk.0.attn_q.weight", (8, 8), None),
        (
            "qwen2moe",
            "blk.0.ffn_gate_inp_shexp.weight",
            (1, 8),
            Some("router"),
        ),
        ("qwen2moe", "blk.0.ffn_up_shexp.weight", (8, 8), None),
        (
            "qwen35",
            "blk.0.attn_qkv.weight",
            (32, 8),
            Some("Gated DeltaNet"),
        ),
        (
            "qwen35",
            "blk.0.attn_gate.weight",
            (16, 8),
            Some("Gated DeltaNet"),
        ),
        (
            "qwen35",
            "blk.0.ssm_out.weight",
            (8, 16),
            Some("Gated DeltaNet"),
        ),
        ("qwen35", "blk.0.ffn_up.weight", (16, 8), None),
        ("qwen35", "blk.1.attn_q.weight", (32, 8), None),
        ("glm4", "blk.2.attn_q.weight", (16, 8), Some("NextN")),
    ] {
        let path = fixture("targets", arch, |p| common::write_tiny_qwen_gguf(p, arch));
        let dir = path.parent().unwrap();
        let table = load(&path).lora_slots().unwrap();
        let lora = dir.join("adapter.gguf");
        common::write_lora_gguf(
            &lora,
            arch,
            0.0,
            &[LoraSpec::new(base, 2, n_out, n_in, 5)],
            LoraDtype::F32,
        );
        match (LoraAdapter::load(&lora, &table, &limits), want) {
            (Ok(_), None) => {}
            (Err(e), Some(want)) => assert!(e.to_string().contains(want), "{arch} {base}: {e}"),
            (Ok(_), Some(want)) => panic!("{arch} {base}: loaded, expected `{want}`"),
            (Err(e), None) => panic!("{arch} {base}: {e}"),
        }
        std::fs::remove_dir_all(dir).ok();
    }
}

/// Loaders that apply no deltas refuse an adapter set instead of running the
/// base model, and a set loaded for one model is refused by another.
#[test]
fn adapters_are_refused_where_they_cannot_apply() {
    let path = fixture("refuse", "qwen3", |p| {
        common::write_tiny_qwen_gguf(p, "qwen3")
    });
    let dir = path.parent().unwrap();
    let model = load(&path);
    assert!(model.supports_lora());
    let table = model.lora_slots().unwrap();
    let set = active(&[(
        adapter(
            &model,
            &write_adapter(
                dir,
                "adapter.gguf",
                &table,
                8.0,
                &specs(&table, 4, 1100, |_| true),
            ),
        ),
        1.0,
    )]);

    let llama = dir.join("llama.gguf");
    common::write_tiny_llama_gguf(&llama);
    let deepseek = dir.join("deepseek2.gguf");
    common::write_tiny_deepseek2_gguf(&deepseek);
    for (what, other) in [("llama", llama), ("deepseek2", deepseek)] {
        let mut m = load(&other);
        assert!(!m.supports_lora(), "{what}");
        assert!(m.lora_slots().is_none(), "{what}");
        assert!(m.set_adapters(set.clone()).is_err(), "{what}");
        assert!(!m.set_adapters(None).unwrap(), "{what}");
        assert_eq!(m.adapter_key(), None, "{what}");
    }

    let q8 = dir.join("q8.gguf");
    common::write_tiny_qwen3_quantized_layers(&q8, 2);
    let mut m = load(&q8);
    let err = m.set_adapters(set).unwrap_err().to_string();
    assert!(err.contains("different base model"), "{err}");
    assert_eq!(m.adapter_key(), None);
    std::fs::remove_dir_all(dir).ok();
}
