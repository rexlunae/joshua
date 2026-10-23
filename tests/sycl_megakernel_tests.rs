//! SYCL decode megakernel parity (`JOSHUA_MEGAKERNEL=1`, see
//! `src/megakernel.rs`): the same model decoded op-at-a-time and through the
//! fused kernel must give the same logits at every step.
//!
//! Both runs feed the same tokens (the reference run's greedy picks), mix a
//! multi-token chunk into the middle of the decode so op-at-a-time steps read
//! keys and values the megakernel wrote and the other way round, and roll
//! back with `truncate_kv_cache` before decoding on.
//!
//! Run on the SYCL host, with a bridge built from this tree:
//!   JOSHUA_SYCL_LIBRARY=/path/to/libjoshua_sycl.so \
//!     cargo test --release --features sycl --test sycl_megakernel_tests -- --nocapture
#![cfg(all(feature = "sycl", target_os = "linux"))]

mod common;

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use std::path::Path;

/// One test: the DPC++ runtime crashes when several threads submit at once.
#[test]
fn sycl_megakernel_parity() {
    let device = match Device::new_sycl(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: no SYCL device: {e}");
            return;
        }
    };
    if !candle_core::sycl_backend::mega::DecodePlan::available() {
        panic!("the SYCL bridge has no megakernel; build it from this tree and set JOSHUA_SYCL_LIBRARY");
    }
    let dir = common::model_dir("sycl-megakernel");

    // f32 fixtures: GQA + Q/K norms + head_dim 6 (qwen3), fused QKV with
    // bias and MHA (qwen), multi-section RoPE (qwen3vl), partial interleaved
    // RoPE with a fused [gate ‖ up] FFN (chatglm), and a three-layer stack.
    let cases: Vec<(&str, Box<dyn Fn(&Path)>, f32)> = vec![
        ("qwen3", Box::new(|p: &Path| common::write_tiny_gguf(p, "qwen3")), 1e-5),
        ("qwen", Box::new(|p: &Path| common::write_tiny_qwen_gguf(p, "qwen")), 1e-5),
        ("qwen3vl", Box::new(|p: &Path| common::write_tiny_qwen_gguf(p, "qwen3vl")), 1e-5),
        ("chatglm", Box::new(|p: &Path| common::write_tiny_qwen_gguf(p, "chatglm")), 1e-5),
        ("qwen3 x3", Box::new(|p: &Path| common::write_tiny_qwen3_layers(p, 3)), 1e-5),
        // Block formats at real-model widths: Q4_K / Q6_K take the
        // multi-row GEMV lanes, Q8_0 (the tied head) the one-row kernel.
        ("qwen3 k-quants", Box::new(write_quantized_qwen3), 1e-3),
    ];
    for (name, write, tol) in cases {
        let path = dir.join(format!("{}.gguf", name.replace(' ', "_")));
        write(&path);
        eprintln!("{name}...");
        check(&path, &device, name, tol);
        eprintln!("{name} OK");
    }

    // Tiny fixtures do not exercise real weight shapes or logit spreads. Run
    // the same prefill / cache-growth / chunk / rollback comparison on a real
    // dense Qwen-family GGUF when one is supplied:
    //   JOSHUA_MEGAKERNEL_GGUF=/models/qwen3.gguf \
    //   JOSHUA_MEGAKERNEL_GGUF_TOL=1e-3 (optional, default 1e-3)
    if let Ok(path) = std::env::var("JOSHUA_MEGAKERNEL_GGUF") {
        let tol = std::env::var("JOSHUA_MEGAKERNEL_GGUF_TOL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e-3);
        eprintln!("real model {path} (tol {tol:e})...");
        check(Path::new(&path), &device, "real model", tol);
        eprintln!("real model OK");
    }
}

fn load(path: &Path, device: &Device, mega: bool) -> joshua::model::QuantizedModel {
    // Read at load time by the loader.
    if mega {
        std::env::set_var("JOSHUA_MEGAKERNEL", "1");
    } else {
        std::env::remove_var("JOSHUA_MEGAKERNEL");
    }
    let bytes = std::fs::read(path).unwrap();
    let content = joshua::gguf_ext::read_header(&mut std::io::Cursor::new(&bytes[..]))
        .unwrap()
        .to_candle_content()
        .unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let model = joshua::model::QuantizedModel::from_gguf_mmap(content, &mut cursor, device, None, None, 0).unwrap();
    std::env::remove_var("JOSHUA_MEGAKERNEL");
    model
}

fn step(model: &mut joshua::model::QuantizedModel, device: &Device, tokens: &[u32], offset: usize) -> Vec<f32> {
    let input = Tensor::new(tokens, device).unwrap().unsqueeze(0).unwrap();
    model.forward(&input, offset).unwrap().flatten_all().unwrap().to_vec1().unwrap()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32
}

thread_local! {
    static MAX_DIFF: std::cell::Cell<f32> = const { std::cell::Cell::new(0.0) };
}

/// [`common::assert_close`], recording the largest difference seen.
fn close(got: &[f32], expect: &[f32], tol: f32, what: &str) {
    common::assert_close(got, expect, tol, what);
    let d = got.iter().zip(expect).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    MAX_DIFF.with(|m| m.set(m.get().max(d)));
}

/// Prefill, decode with the reference run's greedy picks, a multi-token
/// chunk, more decode, a rollback and decode again, comparing every step.
fn check(path: &Path, device: &Device, name: &str, tol: f32) {
    let mut reference = load(path, device, false);
    let mut fused = load(path, device, true);
    let before = joshua::megakernel::fused_steps();
    MAX_DIFF.with(|m| m.set(0.0));

    let prompt = [1u32, 4, 5, 6, 2];
    let mut pos = 0;
    let mut expect = step(&mut reference, device, &prompt, pos);
    let got = step(&mut fused, device, &prompt, pos);
    close(&got, &expect, tol, &format!("{name} prefill"));
    pos += prompt.len();

    let decode = |reference: &mut joshua::model::QuantizedModel,
                      fused: &mut joshua::model::QuantizedModel,
                      expect: &mut Vec<f32>,
                      pos: &mut usize,
                      n: usize,
                      what: &str| {
        for i in 0..n {
            let tok = [argmax(expect)];
            *expect = step(reference, device, &tok, *pos);
            let got = step(fused, device, &tok, *pos);
            close(&got, expect, tol, &format!("{name} {what} step {i}"));
            *pos += 1;
        }
    };
    // Past the KV cache's first capacity (16 rows), so it grows mid-run.
    decode(&mut reference, &mut fused, &mut expect, &mut pos, 20, "decode");

    // A multi-token chunk through the op-at-a-time path reads the rows the
    // megakernel wrote; the decode after it reads the chunk's rows.
    let chunk = [7u32, 8, 9];
    expect = step(&mut reference, device, &chunk, pos);
    let got = step(&mut fused, device, &chunk, pos);
    close(&got, &expect, tol, &format!("{name} chunk"));
    pos += chunk.len();
    decode(&mut reference, &mut fused, &mut expect, &mut pos, 6, "decode after chunk");

    // Roll back four positions and decode a different continuation.
    pos -= 4;
    assert!(reference.truncate_kv_cache(pos).unwrap() && fused.truncate_kv_cache(pos).unwrap());
    expect = step(&mut reference, device, &[10], pos);
    let got = step(&mut fused, device, &[10], pos);
    close(&got, &expect, tol, &format!("{name} after rollback"));
    pos += 1;
    decode(&mut reference, &mut fused, &mut expect, &mut pos, 5, "decode after rollback");

    // 20 + 6 + 1 + 5 single-token steps; none of them fell back.
    let fused_steps = joshua::megakernel::fused_steps() - before;
    assert_eq!(fused_steps, 32, "{name}: decode steps that ran fused");
    eprintln!("  largest logit difference: {:e}", MAX_DIFF.with(|m| m.get()));
}

/// A two-layer qwen3 model at widths the block kernels take: hidden 256,
/// 4 query heads and 2 KV heads of 64 dims (Q/K norms), FFN 512, vocab 320.
/// Q4_K for Q/K/O/gate/up, Q6_K for V/down (the usual llama.cpp mix) and a
/// Q8_0 tied embedding / head.
fn write_quantized_qwen3(path: &Path) {
    const EMB: usize = 256;
    const FFN: usize = 512;
    const VOCAB: usize = 320;
    const LAYERS: usize = 2;
    let (heads, kv_heads, hd) = (4usize, 2usize, 64usize);
    let q = |data: Vec<f32>, shape: &[usize], dtype: GgmlDType| {
        let t = Tensor::from_vec(data, shape, &Device::Cpu).unwrap();
        QTensor::quantize(&t, dtype).unwrap()
    };
    // Larger weights than common::weights so the logits spread.
    let w = |n: usize, seed: u32| common::weights(n, seed).into_iter().map(|x| x * 4.0).collect::<Vec<f32>>();
    let norm = |n: usize, seed: u32| common::weights(n, seed).into_iter().map(|x| 1.0 + x).collect::<Vec<f32>>();
    let key = |s: &str| format!("qwen3.{s}");
    let metadata: Vec<(String, gguf_file::Value)> = vec![
        ("general.architecture".into(), gguf_file::Value::String("qwen3".into())),
        (key("attention.head_count"), gguf_file::Value::U32(heads as u32)),
        (key("attention.head_count_kv"), gguf_file::Value::U32(kv_heads as u32)),
        (key("attention.key_length"), gguf_file::Value::U32(hd as u32)),
        (key("block_count"), gguf_file::Value::U32(LAYERS as u32)),
        (key("embedding_length"), gguf_file::Value::U32(EMB as u32)),
        (key("context_length"), gguf_file::Value::U32(512)),
        (key("attention.layer_norm_rms_epsilon"), gguf_file::Value::F32(1e-6)),
        (key("rope.freq_base"), gguf_file::Value::F32(1_000_000.0)),
        (key("feed_forward_length"), gguf_file::Value::U32(FFN as u32)),
        ("tokenizer.ggml.eos_token_id".into(), gguf_file::Value::U32(3)),
    ];
    let mut tensors: Vec<(String, QTensor)> = vec![
        ("token_embd.weight".into(), q(w(VOCAB * EMB, 1), &[VOCAB, EMB], GgmlDType::Q8_0)),
        ("output_norm.weight".into(), q(norm(EMB, 2), &[EMB], GgmlDType::F32)),
    ];
    for l in 0..LAYERS {
        let s = 100 * (l as u32 + 1);
        let p = |n: &str| format!("blk.{l}.{n}");
        tensors.push((p("attn_norm.weight"), q(norm(EMB, s), &[EMB], GgmlDType::F32)));
        tensors.push((p("ffn_norm.weight"), q(norm(EMB, s + 1), &[EMB], GgmlDType::F32)));
        tensors.push((p("attn_q_norm.weight"), q(norm(hd, s + 2), &[hd], GgmlDType::F32)));
        tensors.push((p("attn_k_norm.weight"), q(norm(hd, s + 3), &[hd], GgmlDType::F32)));
        tensors.push((p("attn_q.weight"), q(w(heads * hd * EMB, s + 4), &[heads * hd, EMB], GgmlDType::Q4K)));
        tensors.push((p("attn_k.weight"), q(w(kv_heads * hd * EMB, s + 5), &[kv_heads * hd, EMB], GgmlDType::Q4K)));
        tensors.push((p("attn_v.weight"), q(w(kv_heads * hd * EMB, s + 6), &[kv_heads * hd, EMB], GgmlDType::Q6K)));
        tensors.push((p("attn_output.weight"), q(w(EMB * heads * hd, s + 7), &[EMB, heads * hd], GgmlDType::Q4K)));
        tensors.push((p("ffn_gate.weight"), q(w(FFN * EMB, s + 8), &[FFN, EMB], GgmlDType::Q4K)));
        tensors.push((p("ffn_up.weight"), q(w(FFN * EMB, s + 9), &[FFN, EMB], GgmlDType::Q4K)));
        tensors.push((p("ffn_down.weight"), q(w(EMB * FFN, s + 10), &[EMB, FFN], GgmlDType::Q6K)));
    }
    let md: Vec<(&str, &gguf_file::Value)> = metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let ts: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut file = std::fs::File::create(path).unwrap();
    gguf_file::write(&mut file, &md, &ts).unwrap();
}
