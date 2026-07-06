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
