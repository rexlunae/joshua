//! Shared helpers for integration tests: synthesise tiny but structurally
//! valid GGUF models plus a matching `tokenizer.json`, so the full engine
//! pipeline can be exercised without network access or model downloads.
#![allow(dead_code)]

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use std::fs::File;
use std::path::{Path, PathBuf};

/// Deterministic pseudo-random weights in roughly [-0.1, 0.1].
pub fn weights(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            // xorshift32
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state % 2000) as f32 / 1000.0 - 1.0) * 0.1
        })
        .collect()
}

pub fn qtensor(data: Vec<f32>, shape: &[usize]) -> QTensor {
    let t = Tensor::from_vec(data, shape, &Device::Cpu).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

/// A minimal WordLevel tokenizer with a 16-token vocabulary.
pub const TOKENIZER_JSON: &str = r#"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [
        {"id": 3, "content": "</s>", "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true}
    ],
    "normalizer": null,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": null,
    "decoder": null,
    "model": {
        "type": "WordLevel",
        "vocab": {"<unk>": 0, "hello": 1, "world": 2, "</s>": 3,
                  "a": 4, "b": 5, "c": 6, "d": 7, "e": 8, "f": 9,
                  "g": 10, "h": 11, "i": 12, "j": 13, "k": 14, "l": 15},
        "unk_token": "<unk>"
    }
}"#;

/// Write a tiny but structurally valid GGUF for the given architecture:
/// 16-token vocab, 8-dim embedding, 2 heads, 1 transformer block, tied
/// output head.  Per-arch quirks are exercised deliberately: qwen2 uses
/// grouped-query attention plus QKV biases, qwen3 uses GQA, a head_dim
/// decoupled from the embedding width, Q/K norms, and last-token pooling.
pub fn write_tiny_gguf(path: &Path, arch: &str) {
    const VOCAB: usize = 16;
    const EMB: usize = 8;
    const FFN: usize = 16;
    let (heads, kv_heads, head_dim) = match arch {
        "llama" => (2usize, 2usize, 4usize),
        "qwen2" => (2, 1, 4),
        "qwen3" => (2, 1, 6),
        other => panic!("unsupported synthetic arch {other}"),
    };

    let key = |suffix: &str| format!("{arch}.{suffix}");
    let mut metadata: Vec<(String, gguf_file::Value)> = vec![
        (
            "general.architecture".to_string(),
            gguf_file::Value::String(arch.to_string()),
        ),
        (
            key("attention.head_count"),
            gguf_file::Value::U32(heads as u32),
        ),
        (
            key("attention.head_count_kv"),
            gguf_file::Value::U32(kv_heads as u32),
        ),
        (key("block_count"), gguf_file::Value::U32(1)),
        (key("embedding_length"), gguf_file::Value::U32(EMB as u32)),
        (key("context_length"), gguf_file::Value::U32(64)),
        (
            key("attention.layer_norm_rms_epsilon"),
            gguf_file::Value::F32(1e-5),
        ),
        (key("rope.freq_base"), gguf_file::Value::F32(10_000.0)),
        (key("feed_forward_length"), gguf_file::Value::U32(FFN as u32)),
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            gguf_file::Value::U32(3),
        ),
        (
            "tokenizer.ggml.bos_token_id".to_string(),
            gguf_file::Value::U32(3),
        ),
        (
            "tokenizer.ggml.unknown_token_id".to_string(),
            gguf_file::Value::U32(0),
        ),
        // Embedded SPM-style vocab so external GGUF consumers (llama.cpp via
        // the joshua-llamacpp-npu adapter) can load the file too.  Joshua's
        // own engine tokenises with tokenizer.json instead.
        (
            "tokenizer.ggml.model".to_string(),
            gguf_file::Value::String("llama".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            gguf_file::Value::Array(
                [
                    "<unk>", "hello", "world", "</s>", "a", "b", "c", "d", "e", "f", "g", "h",
                    "i", "j", "k", "l",
                ]
                .iter()
                .map(|s| gguf_file::Value::String(s.to_string()))
                .collect(),
            ),
        ),
        (
            "tokenizer.ggml.scores".to_string(),
            gguf_file::Value::Array((0..16).map(|i| gguf_file::Value::F32(-(i as f32))).collect()),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            gguf_file::Value::Array(
                // 2 = unknown, 3 = control, 1 = normal (llama.cpp encoding).
                [2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    .iter()
                    .map(|&t| gguf_file::Value::I32(t))
                    .collect(),
            ),
        ),
        // A minimal chat template so complete() renders through the
        // GGUF-template path.  "hello" stands in for a turn delimiter so
        // the tiny WordLevel vocab can tokenise the rendered prompt.
        (
            "tokenizer.chat_template".to_string(),
            gguf_file::Value::String(
                "{% for message in messages %}hello {{ message.content }} \
                 {% endfor %}{% if add_generation_prompt %}world{% endif %}"
                    .to_string(),
            ),
        ),
    ];
    match arch {
        "llama" => {
            metadata.push((
                key("rope.dimension_count"),
                gguf_file::Value::U32(head_dim as u32),
            ));
        }
        "qwen3" => {
            metadata.push((
                key("attention.key_length"),
                gguf_file::Value::U32(head_dim as u32),
            ));
            // F32 activations so parity with the F32 test weights is exact.
            metadata.push(("general.dtype".to_string(), gguf_file::Value::U32(0)));
            // Qwen3-Embedding style last-token pooling.
            metadata.push((key("pooling_type"), gguf_file::Value::U32(3)));
        }
        _ => {}
    }

    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;
    let ones = |n: usize| vec![1.0f32; n];
    let mut tensors: Vec<(String, QTensor)> = vec![
        (
            "token_embd.weight".to_string(),
            qtensor(weights(VOCAB * EMB, 1), &[VOCAB, EMB]),
        ),
        ("output_norm.weight".to_string(), qtensor(ones(EMB), &[EMB])),
        (
            "blk.0.attn_norm.weight".to_string(),
            qtensor(ones(EMB), &[EMB]),
        ),
        (
            "blk.0.ffn_norm.weight".to_string(),
            qtensor(ones(EMB), &[EMB]),
        ),
        (
            "blk.0.attn_q.weight".to_string(),
            qtensor(weights(q_dim * EMB, 2), &[q_dim, EMB]),
        ),
        (
            "blk.0.attn_k.weight".to_string(),
            qtensor(weights(kv_dim * EMB, 3), &[kv_dim, EMB]),
        ),
        (
            "blk.0.attn_v.weight".to_string(),
            qtensor(weights(kv_dim * EMB, 4), &[kv_dim, EMB]),
        ),
        (
            "blk.0.attn_output.weight".to_string(),
            qtensor(weights(EMB * q_dim, 5), &[EMB, q_dim]),
        ),
        (
            "blk.0.ffn_gate.weight".to_string(),
            qtensor(weights(FFN * EMB, 6), &[FFN, EMB]),
        ),
        (
            "blk.0.ffn_down.weight".to_string(),
            qtensor(weights(EMB * FFN, 7), &[EMB, FFN]),
        ),
        (
            "blk.0.ffn_up.weight".to_string(),
            qtensor(weights(FFN * EMB, 8), &[FFN, EMB]),
        ),
    ];
    if arch == "qwen2" {
        tensors.push((
            "blk.0.attn_q.bias".to_string(),
            qtensor(weights(q_dim, 9), &[q_dim]),
        ));
        tensors.push((
            "blk.0.attn_k.bias".to_string(),
            qtensor(weights(kv_dim, 10), &[kv_dim]),
        ));
        tensors.push((
            "blk.0.attn_v.bias".to_string(),
            qtensor(weights(kv_dim, 11), &[kv_dim]),
        ));
    }
    if arch == "qwen3" {
        tensors.push((
            "blk.0.attn_q_norm.weight".to_string(),
            qtensor(weights(head_dim, 12), &[head_dim]),
        ));
        tensors.push((
            "blk.0.attn_k_norm.weight".to_string(),
            qtensor(weights(head_dim, 13), &[head_dim]),
        ));
    }

    let metadata_refs: Vec<(&str, &gguf_file::Value)> =
        metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> =
        tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let mut file = File::create(path).unwrap();
    gguf_file::write(&mut file, &metadata_refs, &tensor_refs).unwrap();
}

/// Back-compat wrapper: the original llama-arch test model.
pub fn write_tiny_llama_gguf(path: &Path) {
    write_tiny_gguf(path, "llama");
}

/// Write a tiny `deepseek2` GGUF with the KV up-projection in the legacy
/// combined `attn_kv_b` form (both Joshua and llama.cpp take the unabsorbed
/// attention path). See [`write_deepseek2_gguf`].
pub fn write_tiny_deepseek2_gguf(path: &Path) {
    write_deepseek2_gguf(path, false);
}

/// Write a tiny `deepseek2` GGUF in the modern MLA-split form (`attn_k_b` /
/// `attn_v_b`, `key_length_mla` / `value_length_mla` set) that recent
/// Kimi-K2 conversions ship. Joshua reconstructs the combined projection from
/// the split tensors; llama.cpp uses its absorbed MLA path — the two are
/// algebraically identical, so their logits must still agree.
pub fn write_tiny_deepseek2_mla_gguf(path: &Path) {
    write_deepseek2_gguf(path, true);
}

/// Write a tiny but structurally valid `deepseek2` GGUF exercising the full
/// DeepSeek-V3 / Kimi-K2 feature set: MLA attention with Q-LoRA, a leading
/// dense layer plus a fine-grained MoE layer, sigmoid gating with a selection
/// bias, group-limited routing, and a shared expert.  `split_mla` chooses
/// between the pre-split (`attn_k_b`/`attn_v_b`) and legacy combined
/// (`attn_kv_b`) KV up-projection encodings.
pub fn write_deepseek2_gguf(path: &Path, split_mla: bool) {
    const VOCAB: usize = 16;
    const EMB: usize = 8;
    const H: usize = 2; // heads
    const R: usize = 4; // qk_rope_head_dim
    const NP: usize = 4; // qk_nope_head_dim
    const KH: usize = R + NP; // per-head key dim (8)
    const VH: usize = 4; // v_head_dim
    const LQ: usize = 6; // q_lora_rank
    const LKV: usize = 6; // kv_lora_rank
    const NFF: usize = 16; // dense ffn
    const NE: usize = 4; // experts
    const NFE: usize = 8; // expert ffn
    const NLAYER: usize = 2;

    let u32v = |v: u32| gguf_file::Value::U32(v);
    let f32v = |v: f32| gguf_file::Value::F32(v);
    let key = |s: &str| format!("deepseek2.{s}");
    let mut metadata: Vec<(String, gguf_file::Value)> = vec![
        (
            "general.architecture".to_string(),
            gguf_file::Value::String("deepseek2".to_string()),
        ),
        (key("attention.head_count"), u32v(H as u32)),
        (key("attention.head_count_kv"), u32v(H as u32)),
        (key("block_count"), u32v(NLAYER as u32)),
        (key("embedding_length"), u32v(EMB as u32)),
        (key("context_length"), u32v(64)),
        (key("attention.layer_norm_rms_epsilon"), f32v(1e-5)),
        (key("attention.key_length"), u32v(KH as u32)),
        (key("attention.value_length"), u32v(VH as u32)),
        (key("rope.dimension_count"), u32v(R as u32)),
        (key("rope.freq_base"), f32v(10_000.0)),
        (key("attention.q_lora_rank"), u32v(LQ as u32)),
        (key("attention.kv_lora_rank"), u32v(LKV as u32)),
        (key("feed_forward_length"), u32v(NFF as u32)),
        (key("leading_dense_block_count"), u32v(1)),
        (key("expert_count"), u32v(NE as u32)),
        (key("expert_used_count"), u32v(2)),
        (key("expert_feed_forward_length"), u32v(NFE as u32)),
        (key("expert_shared_count"), u32v(1)),
        (key("expert_weights_scale"), f32v(2.5)),
        (key("expert_weights_norm"), gguf_file::Value::Bool(true)),
        (key("expert_gating_func"), u32v(2)), // sigmoid
        (key("expert_group_count"), u32v(2)),
        (key("expert_group_used_count"), u32v(1)),
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            gguf_file::Value::U32(3),
        ),
        (
            "tokenizer.ggml.bos_token_id".to_string(),
            gguf_file::Value::U32(3),
        ),
        (
            "tokenizer.ggml.unknown_token_id".to_string(),
            gguf_file::Value::U32(0),
        ),
        (
            "tokenizer.ggml.model".to_string(),
            gguf_file::Value::String("llama".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            gguf_file::Value::Array(
                [
                    "<unk>", "hello", "world", "</s>", "a", "b", "c", "d", "e", "f", "g", "h", "i",
                    "j", "k", "l",
                ]
                .iter()
                .map(|s| gguf_file::Value::String(s.to_string()))
                .collect(),
            ),
        ),
        (
            "tokenizer.ggml.scores".to_string(),
            gguf_file::Value::Array((0..16).map(|i| gguf_file::Value::F32(-(i as f32))).collect()),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            gguf_file::Value::Array(
                [2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    .iter()
                    .map(|&t| gguf_file::Value::I32(t))
                    .collect(),
            ),
        ),
        (
            "tokenizer.chat_template".to_string(),
            gguf_file::Value::String(
                "{% for message in messages %}hello {{ message.content }} \
                 {% endfor %}{% if add_generation_prompt %}world{% endif %}"
                    .to_string(),
            ),
        ),
    ];
    if split_mla {
        // Advertise the pre-split MLA head dims so llama.cpp reads attn_k_b /
        // attn_v_b and takes its absorbed MLA path.
        metadata.push((key("attention.key_length_mla"), u32v(KH as u32)));
        metadata.push((key("attention.value_length_mla"), u32v(VH as u32)));
    }

    let ones = |n: usize| vec![1.0f32; n];
    let mut tensors: Vec<(String, QTensor)> = vec![
        (
            "token_embd.weight".to_string(),
            qtensor(weights(VOCAB * EMB, 1), &[VOCAB, EMB]),
        ),
        ("output_norm.weight".to_string(), qtensor(ones(EMB), &[EMB])),
    ];

    let mut seed = 10u32;
    let mut next = |n: usize| {
        seed = seed.wrapping_add(7).wrapping_mul(2_654_435_761) | 1;
        weights(n, seed)
    };
    for i in 0..NLAYER {
        let p = format!("blk.{i}");
        tensors.push((format!("{p}.attn_norm.weight"), qtensor(ones(EMB), &[EMB])));
        tensors.push((format!("{p}.ffn_norm.weight"), qtensor(ones(EMB), &[EMB])));
        // MLA attention (Q-LoRA + combined KV-B).
        tensors.push((format!("{p}.attn_q_a.weight"), qtensor(next(LQ * EMB), &[LQ, EMB])));
        tensors.push((format!("{p}.attn_q_a_norm.weight"), qtensor(ones(LQ), &[LQ])));
        tensors.push((
            format!("{p}.attn_q_b.weight"),
            qtensor(next(H * KH * LQ), &[H * KH, LQ]),
        ));
        tensors.push((
            format!("{p}.attn_kv_a_mqa.weight"),
            qtensor(next((LKV + R) * EMB), &[LKV + R, EMB]),
        ));
        tensors.push((format!("{p}.attn_kv_a_norm.weight"), qtensor(ones(LKV), &[LKV])));
        // Draw the KV up-projection from a single source (per head: A = [NP,LKV]
        // mapping the latent to k_nope, B = [VH,LKV] mapping it to v) regardless
        // of encoding, so the legacy and split forms are the *same* model — the
        // RNG stream stays identical either way — and can be compared directly.
        let a = next(H * NP * LKV); // head-major [H][NP][LKV]
        let bmat = next(H * VH * LKV); // head-major [H][VH][LKV]
        if split_mla {
            // attn_k_b: ggml {NP, LKV, H} → candle [H, LKV, NP], element[h][l][np] = A[h][np][l].
            let mut kb = vec![0f32; H * LKV * NP];
            for h in 0..H {
                for l in 0..LKV {
                    for np in 0..NP {
                        kb[h * LKV * NP + l * NP + np] = a[h * NP * LKV + np * LKV + l];
                    }
                }
            }
            tensors.push((format!("{p}.attn_k_b.weight"), qtensor(kb, &[H, LKV, NP])));
            // attn_v_b: ggml {LKV, VH, H} → candle [H, VH, LKV] = B directly.
            tensors.push((format!("{p}.attn_v_b.weight"), qtensor(bmat.clone(), &[H, VH, LKV])));
        } else {
            // Combined kv_b: per head, rows [A (NP×LKV); B (VH×LKV)].
            let mut kv = vec![0f32; H * (NP + VH) * LKV];
            for h in 0..H {
                let base = h * (NP + VH) * LKV;
                for r in 0..NP {
                    for l in 0..LKV {
                        kv[base + r * LKV + l] = a[h * NP * LKV + r * LKV + l];
                    }
                }
                for r in 0..VH {
                    for l in 0..LKV {
                        kv[base + (NP + r) * LKV + l] = bmat[h * VH * LKV + r * LKV + l];
                    }
                }
            }
            tensors.push((format!("{p}.attn_kv_b.weight"), qtensor(kv, &[H * (NP + VH), LKV])));
        }
        tensors.push((
            format!("{p}.attn_output.weight"),
            qtensor(next(EMB * H * VH), &[EMB, H * VH]),
        ));

        if i == 0 {
            // Dense SwiGLU layer.
            tensors.push((format!("{p}.ffn_gate.weight"), qtensor(next(NFF * EMB), &[NFF, EMB])));
            tensors.push((format!("{p}.ffn_up.weight"), qtensor(next(NFF * EMB), &[NFF, EMB])));
            tensors.push((format!("{p}.ffn_down.weight"), qtensor(next(EMB * NFF), &[EMB, NFF])));
        } else {
            // MoE layer: router (+ bias), routed experts, shared expert.
            tensors.push((format!("{p}.ffn_gate_inp.weight"), qtensor(next(NE * EMB), &[NE, EMB])));
            tensors.push((format!("{p}.exp_probs_b.bias"), qtensor(next(NE), &[NE])));
            tensors.push((
                format!("{p}.ffn_gate_exps.weight"),
                qtensor(next(NE * NFE * EMB), &[NE, NFE, EMB]),
            ));
            tensors.push((
                format!("{p}.ffn_up_exps.weight"),
                qtensor(next(NE * NFE * EMB), &[NE, NFE, EMB]),
            ));
            tensors.push((
                format!("{p}.ffn_down_exps.weight"),
                qtensor(next(NE * EMB * NFE), &[NE, EMB, NFE]),
            ));
            tensors.push((
                format!("{p}.ffn_gate_shexp.weight"),
                qtensor(next(NFE * EMB), &[NFE, EMB]),
            ));
            tensors.push((
                format!("{p}.ffn_up_shexp.weight"),
                qtensor(next(NFE * EMB), &[NFE, EMB]),
            ));
            tensors.push((
                format!("{p}.ffn_down_shexp.weight"),
                qtensor(next(EMB * NFE), &[EMB, NFE]),
            ));
        }
    }

    let metadata_refs: Vec<(&str, &gguf_file::Value)> =
        metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> =
        tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut file = File::create(path).unwrap();
    gguf_file::write(&mut file, &metadata_refs, &tensor_refs).unwrap();
}

/// Write a GGUF with a known-to-llama.cpp but unimplemented architecture.
pub fn write_unsupported_gguf(path: &Path) {
    let arch = gguf_file::Value::String("mamba".to_string());
    let metadata: Vec<(&str, &gguf_file::Value)> = vec![("general.architecture", &arch)];
    let mut file = File::create(path).unwrap();
    gguf_file::write(&mut file, &metadata, &[]).unwrap();
}

/// Create a fresh model directory under the target tmp area.
pub fn model_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("joshua-tests")
        .join(format!("{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).unwrap();
    dir
}
