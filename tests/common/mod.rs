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

/// Q4_K quantized tensor — used by writers that exercise the K-quant matmul
/// paths (Metal needs dims that are multiples of 256 for Q4_K blocks).
pub fn qtensor_q4k(data: Vec<f32>, shape: &[usize]) -> QTensor {
    let t = Tensor::from_vec(data, shape, &Device::Cpu).unwrap();
    QTensor::quantize(&t, GgmlDType::Q4K).unwrap()
}

/// Load a GGUF on the CPU the way the engine does: the tolerant header (raw
/// dtype ids) projected onto candle's `Content` — `Content::read` itself
/// rejects a file holding a dtype candle cannot name (IQ2_XXS, Q1_0, …) —
/// with the weights borrowed from a memory mapping when `mmap` is set, else
/// streamed from the reader.  The loader re-reads the raw header from the
/// reader for the tensors the projection drops.
pub fn load_model(path: &Path, mmap: bool) -> joshua::model::QuantizedModel {
    let bytes = std::fs::read(path).unwrap();
    let content = joshua::gguf_ext::read_header(&mut std::io::Cursor::new(&bytes[..]))
        .unwrap()
        .to_candle_content()
        .unwrap();
    let mmap = mmap.then(|| {
        // Safety: the file is read-only for the lifetime of the mapping.
        std::sync::Arc::new(unsafe { memmap2::Mmap::map(&File::open(path).unwrap()) }.unwrap())
    });
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    joshua::model::QuantizedModel::from_gguf_mmap(content, &mut cursor, &Device::Cpu, mmap, None, 0)
        .unwrap()
}

/// Last-position logits for `tokens` fed at `offset` (CPU).
pub fn logits(model: &mut joshua::model::QuantizedModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    let input = Tensor::new(tokens, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
    model
        .forward(&input, offset)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Element-wise `|a - b| < tol`, naming the first diverging logit.
pub fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < tol, "{what}: logit {i} diverges: {x} vs {y}");
    }
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
        (key("context_length"), gguf_file::Value::U32(512)),
        (
            key("attention.layer_norm_rms_epsilon"),
            gguf_file::Value::F32(1e-5),
        ),
        (key("rope.freq_base"), gguf_file::Value::F32(10_000.0)),
        (
            key("feed_forward_length"),
            gguf_file::Value::U32(FFN as u32),
        ),
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
            gguf_file::Value::Array(
                (0..16)
                    .map(|i| gguf_file::Value::F32(-(i as f32)))
                    .collect(),
            ),
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
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let mut file = File::create(path).unwrap();
    gguf_file::write(&mut file, &metadata_refs, &tensor_refs).unwrap();
}

/// Back-compat wrapper: the original llama-arch test model.
pub fn write_tiny_llama_gguf(path: &Path) {
    write_tiny_gguf(path, "llama");
}

/// Like [`write_tiny_llama_gguf`] but with `EMB = 64` (2 heads → head_dim 32)
/// and small-integer weights.
///
/// Two reasons this variant exists:
/// - candle's Metal backend routes single-token decode through its SDPA kernel,
///   which only supports head dims ≥ 32 (real llama/qwen/gemma models all
///   qualify; the EMB=8 toy does not).
/// - small-integer weights quantise almost exactly in Q8_0, so the logits are
///   well-separated and greedy argmax is identical on CPU and Metal despite
///   different accumulation orders.
pub fn write_tiny_llama_gguf_hd32(path: &Path) {
    const VOCAB: usize = 16;
    const EMB: usize = 64;
    const FFN: usize = 16;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = EMB / HEADS; // 32 — Metal SDPA's minimum
    const Q_DIM: usize = HEADS * HEAD_DIM;

    let key = |suffix: &str| format!("llama.{suffix}");
    let metadata: Vec<(String, gguf_file::Value)> = vec![
        (
            "general.architecture".to_string(),
            gguf_file::Value::String("llama".to_string()),
        ),
        (
            key("attention.head_count"),
            gguf_file::Value::U32(HEADS as u32),
        ),
        (
            key("attention.head_count_kv"),
            gguf_file::Value::U32(HEADS as u32),
        ),
        (key("block_count"), gguf_file::Value::U32(2)),
        (key("embedding_length"), gguf_file::Value::U32(EMB as u32)),
        (key("context_length"), gguf_file::Value::U32(512)),
        (
            key("attention.layer_norm_rms_epsilon"),
            gguf_file::Value::F32(1e-5),
        ),
        (key("rope.freq_base"), gguf_file::Value::F32(10_000.0)),
        (
            key("rope.dimension_count"),
            gguf_file::Value::U32(HEAD_DIM as u32),
        ),
        (
            key("feed_forward_length"),
            gguf_file::Value::U32(FFN as u32),
        ),
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
            gguf_file::Value::Array(
                (0..16)
                    .map(|i| gguf_file::Value::F32(-(i as f32)))
                    .collect(),
            ),
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
    ];

    // Deterministic small integers in [-8, 8]: quantise almost exactly in Q8_0
    // (scale = 8/127) and keep logits well-separated, so greedy decoding is
    // stable across CPU/Metal accumulation-order differences.
    let w = |n: usize, seed: usize| {
        (0..n)
            .map(|i| ((((seed + 1) * 37 + i * 13) % 17) as i32 - 8) as f32)
            .collect::<Vec<f32>>()
    };
    let ones = |n: usize| vec![1.0f32; n];

    let mut tensors: Vec<(String, QTensor)> = vec![
        (
            "token_embd.weight".to_string(),
            qtensor(w(VOCAB * EMB, 1), &[VOCAB, EMB]),
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
            "blk.1.attn_norm.weight".to_string(),
            qtensor(ones(EMB), &[EMB]),
        ),
        (
            "blk.1.ffn_norm.weight".to_string(),
            qtensor(ones(EMB), &[EMB]),
        ),
    ];
    for (layer, seed) in [(0usize, 2usize), (1, 10)] {
        tensors.push((
            format!("blk.{layer}.attn_q.weight"),
            qtensor(w(Q_DIM * EMB, seed), &[Q_DIM, EMB]),
        ));
        tensors.push((
            format!("blk.{layer}.attn_k.weight"),
            qtensor(w(Q_DIM * EMB, seed + 1), &[Q_DIM, EMB]),
        ));
        tensors.push((
            format!("blk.{layer}.attn_v.weight"),
            qtensor(w(Q_DIM * EMB, seed + 2), &[Q_DIM, EMB]),
        ));
        tensors.push((
            format!("blk.{layer}.attn_output.weight"),
            qtensor(w(EMB * Q_DIM, seed + 3), &[EMB, Q_DIM]),
        ));
        tensors.push((
            format!("blk.{layer}.ffn_gate.weight"),
            qtensor(w(FFN * EMB, seed + 4), &[FFN, EMB]),
        ));
        tensors.push((
            format!("blk.{layer}.ffn_down.weight"),
            qtensor(w(EMB * FFN, seed + 5), &[EMB, FFN]),
        ));
        tensors.push((
            format!("blk.{layer}.ffn_up.weight"),
            qtensor(w(FFN * EMB, seed + 6), &[FFN, EMB]),
        ));
    }

    let metadata_refs: Vec<(&str, &gguf_file::Value)> =
        metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let mut file = File::create(path).unwrap();
    gguf_file::write(&mut file, &metadata_refs, &tensor_refs).unwrap();
}

/// The 16-token SentencePiece-style vocabulary, special-token ids and
/// trivial chat template shared by the tiny MoE fixtures.
pub fn tiny_tokenizer_metadata() -> Vec<(String, gguf_file::Value)> {
    vec![
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
        gguf_file::Value::Array(
            (0..16)
                .map(|i| gguf_file::Value::F32(-(i as f32)))
                .collect(),
        ),
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
    ]
}

/// Write a tiny `deepseek2` GGUF with the KV up-projection in the legacy
/// combined `attn_kv_b` form (both Joshua and llama.cpp take the unabsorbed
/// attention path). DeepSeek-V3 / Kimi-K2 style (sigmoid gating + selection
/// bias). See [`write_deepseek2_gguf`].
pub fn write_tiny_deepseek2_gguf(path: &Path) {
    write_deepseek2_gguf(path, false, false);
}

/// Write a tiny `deepseek2` GGUF in the modern MLA-split form (`attn_k_b` /
/// `attn_v_b`, `key_length_mla` / `value_length_mla` set) that recent
/// Kimi-K2 conversions ship. Joshua reconstructs the combined projection from
/// the split tensors; llama.cpp uses its absorbed MLA path — the two are
/// algebraically identical, so their logits must still agree.
pub fn write_tiny_deepseek2_mla_gguf(path: &Path) {
    write_deepseek2_gguf(path, true, false);
}

/// Write a tiny DeepSeek-V2-style `deepseek2` GGUF: softmax gating with no
/// selection bias and `group_limited_greedy` routing (group score = the best
/// single expert), exercising the V2 branch of the group router.
pub fn write_tiny_deepseek2_v2_gguf(path: &Path) {
    write_deepseek2_gguf(path, false, true);
}

/// Write a tiny but structurally valid `deepseek2` GGUF exercising the full
/// DeepSeek-V3 / Kimi-K2 feature set: MLA attention with Q-LoRA, a leading
/// dense layer plus a fine-grained MoE layer, sigmoid gating with a selection
/// bias, group-limited routing, and a shared expert.  `split_mla` chooses
/// between the pre-split (`attn_k_b`/`attn_v_b`) and legacy combined
/// (`attn_kv_b`) KV up-projection encodings.  `softmax_v2` selects DeepSeek-V2
/// style routing (softmax gating, no `exp_probs_b` selection bias) instead of
/// the DeepSeek-V3 / Kimi-K2 default (sigmoid gating with a bias).
pub fn write_deepseek2_gguf(path: &Path, split_mla: bool, softmax_v2: bool) {
    write_deepseek_family_gguf(path, false, split_mla, softmax_v2);
}

/// Write a tiny `deepseek` (DeepSeek-MoE) GGUF: GQA attention (2 query heads
/// over 1 KV head) with full-width RoPE, a leading dense layer, then a
/// softmax-routed MoE layer with a shared expert and no weight
/// normalisation — the same MoE stack as [`write_deepseek2_gguf`], minus MLA.
pub fn write_tiny_deepseek_gguf(path: &Path) {
    write_deepseek_family_gguf(path, true, false, true);
}

/// Shared body of the `deepseek` (`gqa`) and `deepseek2` fixture writers.
fn write_deepseek_family_gguf(path: &Path, gqa: bool, split_mla: bool, softmax_v2: bool) {
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
    const KV: usize = 1; // GQA kv heads (`deepseek` only)

    let arch = if gqa { "deepseek" } else { "deepseek2" };
    let u32v = |v: u32| gguf_file::Value::U32(v);
    let f32v = |v: f32| gguf_file::Value::F32(v);
    let key = |s: &str| format!("{arch}.{s}");
    let mut metadata: Vec<(String, gguf_file::Value)> = vec![
        (
            "general.architecture".to_string(),
            gguf_file::Value::String(arch.to_string()),
        ),
        (key("attention.head_count"), u32v(H as u32)),
        (key("block_count"), u32v(NLAYER as u32)),
        (key("embedding_length"), u32v(EMB as u32)),
        (key("context_length"), u32v(512)),
        (key("attention.layer_norm_rms_epsilon"), f32v(1e-5)),
        (key("attention.key_length"), u32v(KH as u32)),
        (key("rope.freq_base"), f32v(10_000.0)),
        (key("feed_forward_length"), u32v(NFF as u32)),
        (key("leading_dense_block_count"), u32v(1)),
        (key("expert_count"), u32v(NE as u32)),
        (key("expert_used_count"), u32v(2)),
        (key("expert_feed_forward_length"), u32v(NFE as u32)),
        (key("expert_shared_count"), u32v(1)),
    ];
    if gqa {
        // DeepSeek-MoE: plain GQA, RoPE over the whole head, softmax routing
        // without normalisation (llama.cpp defaults for `deepseek`).
        metadata.extend([
            (key("attention.head_count_kv"), u32v(KV as u32)),
            (key("attention.value_length"), u32v(KH as u32)),
            (key("rope.dimension_count"), u32v(KH as u32)),
            (key("expert_weights_scale"), f32v(1.0)),
        ]);
    } else {
        metadata.extend([
            (key("attention.head_count_kv"), u32v(H as u32)),
            (key("attention.value_length"), u32v(VH as u32)),
            (key("rope.dimension_count"), u32v(R as u32)),
            (key("attention.q_lora_rank"), u32v(LQ as u32)),
            (key("attention.kv_lora_rank"), u32v(LKV as u32)),
            (key("expert_weights_scale"), f32v(2.5)),
            (key("expert_weights_norm"), gguf_file::Value::Bool(true)),
            // 1 = softmax (DeepSeek-V2), 2 = sigmoid (DeepSeek-V3 / Kimi-K2).
            (
                key("expert_gating_func"),
                u32v(if softmax_v2 { 1 } else { 2 }),
            ),
            (key("expert_group_count"), u32v(2)),
            (key("expert_group_used_count"), u32v(1)),
        ]);
    }
    metadata.extend(tiny_tokenizer_metadata());
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
        if gqa {
            tensors.push((format!("{p}.attn_q.weight"), qtensor(next(H * KH * EMB), &[H * KH, EMB])));
            tensors.push((format!("{p}.attn_k.weight"), qtensor(next(KV * KH * EMB), &[KV * KH, EMB])));
            tensors.push((format!("{p}.attn_v.weight"), qtensor(next(KV * KH * EMB), &[KV * KH, EMB])));
            tensors.push((
                format!("{p}.attn_output.weight"),
                qtensor(next(EMB * H * KH), &[EMB, H * KH]),
            ));
        } else {
            // MLA attention (Q-LoRA + combined KV-B).
            tensors.push((
                format!("{p}.attn_q_a.weight"),
                qtensor(next(LQ * EMB), &[LQ, EMB]),
            ));
            tensors.push((
                format!("{p}.attn_q_a_norm.weight"),
                qtensor(ones(LQ), &[LQ]),
            ));
            tensors.push((
                format!("{p}.attn_q_b.weight"),
                qtensor(next(H * KH * LQ), &[H * KH, LQ]),
            ));
            tensors.push((
                format!("{p}.attn_kv_a_mqa.weight"),
                qtensor(next((LKV + R) * EMB), &[LKV + R, EMB]),
            ));
            tensors.push((
                format!("{p}.attn_kv_a_norm.weight"),
                qtensor(ones(LKV), &[LKV]),
            ));
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
                tensors.push((
                    format!("{p}.attn_v_b.weight"),
                    qtensor(bmat.clone(), &[H, VH, LKV]),
                ));
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
                tensors.push((
                    format!("{p}.attn_kv_b.weight"),
                    qtensor(kv, &[H * (NP + VH), LKV]),
                ));
            }
            tensors.push((
                format!("{p}.attn_output.weight"),
                qtensor(next(EMB * H * VH), &[EMB, H * VH]),
            ));
        }

        if i == 0 {
            // Dense SwiGLU layer.
            tensors.push((
                format!("{p}.ffn_gate.weight"),
                qtensor(next(NFF * EMB), &[NFF, EMB]),
            ));
            tensors.push((
                format!("{p}.ffn_up.weight"),
                qtensor(next(NFF * EMB), &[NFF, EMB]),
            ));
            tensors.push((
                format!("{p}.ffn_down.weight"),
                qtensor(next(EMB * NFF), &[EMB, NFF]),
            ));
        } else {
            // MoE layer: router (+ bias for V3/Kimi), routed experts, shared expert.
            tensors.push((
                format!("{p}.ffn_gate_inp.weight"),
                qtensor(next(NE * EMB), &[NE, EMB]),
            ));
            if !softmax_v2 && !gqa {
                tensors.push((format!("{p}.exp_probs_b.bias"), qtensor(next(NE), &[NE])));
            }
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
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut file = File::create(path).unwrap();
    gguf_file::write(&mut file, &metadata_refs, &tensor_refs).unwrap();
}

/// Write a tiny but structurally valid `qwen3moe` GGUF exercising the full
/// Qwen3-MoE feature set: GQA attention with per-head Q/K norms and a
/// half-split RoPE over a head_dim decoupled from the embedding width, plus a
/// fine-grained MoE (softmax router, `norm_topk_prob`, no shared experts).
pub fn write_tiny_qwen3moe_gguf(path: &Path) {
    write_tiny_qwen_gguf(path, "qwen3moe");
}

/// Write a tiny but structurally valid GGUF for any Qwen-family
/// architecture the native Qwen loader serves, with each architecture's
/// distinguishing features:
///
/// * `qwen` — fused `attn_qkv` + bias (MHA), dense FFN.
/// * `qwen2moe` — Q/K/V biases, MoE without top-k normalisation, a shared
///   expert gated by `ffn_gate_inp_shexp`.
/// * `qwen2vl` / `qwen3vl` / `qwen3vlmoe` — multi-section RoPE (sections
///   chosen so the text mask actually freezes a frequency),
///   Q/K norm for the Qwen3 generation.
/// * `qwen3moe` — byte-identical to the historical `qwen3moe` fixture.
/// * `qwen3next` / `qwen35` / `qwen35moe` — a Gated DeltaNet layer followed
///   by a gated-attention layer with partial RoPE and `post_attention_norm`;
///   `qwen3next` uses the fused `ssm_ba`, Qwen3.5 split `ssm_beta` /
///   `ssm_alpha`; four value heads share two key heads.
pub fn write_tiny_qwen_gguf(path: &Path, arch: &str) {
    const VOCAB: usize = 16;
    const EMB: usize = 8;
    const NE: usize = 4; // experts
    const NFE: usize = 8; // expert ffn
    const NFF: usize = 16; // dense ffn

    let hybrid = matches!(arch, "qwen3next" | "qwen35" | "qwen35moe" | "qwen4exp");
    let moe = matches!(
        arch,
        "qwen2moe" | "qwen3moe" | "qwen3vlmoe" | "qwen3next" | "qwen35moe" | "qwen4exp"
    );
    let shared_expert = matches!(arch, "qwen2moe" | "qwen3next" | "qwen35moe" | "qwen4exp");
    // qwen4exp: hyper-connections (2 streams, rank 3), QSA on the attention
    // layer (blocks of 2, budget 2 cells + tail) and PLE on the DeltaNet layer
    // (3-grams, 2 heads per n-gram, 2-wide rows, kernel 3).
    let hyper = arch == "qwen4exp";
    let (hc, hc_rank) = (2usize, 3usize);
    let (idx_heads, idx_dim) = (2usize, 6usize);
    let (ple_dim, ple_rows, ple_kernel) = (2usize, 40usize, 3usize);
    let qk_norm = arch.starts_with("qwen3") || arch == "qwen4exp";
    let biases = matches!(arch, "qwen2moe" | "qwen2vl");
    let fused_qkv = arch == "qwen";
    let h = 2usize;
    let kv = if arch == "qwen" { 2 } else { 1 };
    let hd = if arch == "qwen3moe" || arch == "qwen" || arch == "qwen2moe" {
        4
    } else {
        8
    };
    let n_layer = if arch == "qwen3moe" { 1 } else { 2 };
    // Gated DeltaNet dims: 2 key heads shared by 4 value heads, so the
    // grouped (Qwen3-Next) and tiled (Qwen3.5) sharing layouts differ.
    let (nk, nv, sd, d_conv) = (2usize, 4usize, 4usize, 4usize);
    let conv_dim = 2 * nk * sd + nv * sd;

    let u32v = |v: u32| gguf_file::Value::U32(v);
    let f32v = |v: f32| gguf_file::Value::F32(v);
    let key = |s: &str| format!("{arch}.{s}");
    let mut metadata: Vec<(String, gguf_file::Value)> = vec![
        (
            "general.architecture".to_string(),
            gguf_file::Value::String(arch.to_string()),
        ),
        (key("attention.head_count"), u32v(h as u32)),
        (key("attention.head_count_kv"), u32v(kv as u32)),
        (key("block_count"), u32v(n_layer as u32)),
        (key("embedding_length"), u32v(EMB as u32)),
        (key("context_length"), u32v(512)),
        (key("attention.layer_norm_rms_epsilon"), f32v(1e-5)),
        (key("attention.key_length"), u32v(hd as u32)),
        (key("rope.freq_base"), f32v(10_000.0)),
    ];
    if moe {
        metadata.extend([
            (key("expert_count"), u32v(NE as u32)),
            (key("expert_used_count"), u32v(2)),
            (key("expert_feed_forward_length"), u32v(NFE as u32)),
            (
                key("expert_shared_feed_forward_length"),
                u32v(if shared_expert { NFE as u32 } else { 0 }),
            ),
        ]);
    } else {
        // Qwen1 stores the doubled HF intermediate size (llama.cpp halves it).
        let ff = if arch == "qwen" { 2 * NFF } else { NFF };
        metadata.push((key("feed_forward_length"), u32v(ff as u32)));
    }
    if arch == "qwen3moe" {
        metadata.push((
            key("attention.norm_topk_prob"),
            gguf_file::Value::Bool(true),
        ));
    }
    if hyper {
        // Python int lists convert to I32 arrays; the hash constants are U64.
        let i32s = |v: &[i32]| gguf_file::Value::Array(v.iter().map(|&x| gguf_file::Value::I32(x)).collect());
        let u64s = |v: &[u64]| gguf_file::Value::Array(v.iter().map(|&x| gguf_file::Value::U64(x)).collect());
        metadata.extend([
            (key("hyper_connection.count"), u32v(hc as u32)),
            (key("hyper_connection.low_rank"), u32v(hc_rank as u32)),
            (key("attention.indexer.head_count"), u32v(idx_heads as u32)),
            (key("attention.indexer.key_length"), u32v(idx_dim as u32)),
            (key("attention.indexer.top_k"), u32v(2)),
            (key("attention.compress_ratios"), i32s(&[0, 2])),
            (key("ple.layers"), i32s(&[0])),
            (key("ple.ngram_size"), u32v(3)),
            (key("ple.heads_per_ngram"), u32v(2)),
            (key("ple.conv_kernel"), u32v(ple_kernel as u32)),
            (key("ple.eos_token_id"), u32v(3)),
            (key("ple.layer_multipliers"), u64s(&[3, 1_000_003, 24_000_000_000_017])),
            (key("ple.head_offsets"), u64s(&[0, 10, 20, 30])),
            (key("ple.head_vocab_sizes"), u64s(&[10, 9, 10, 7])),
            (key("embedding_length_per_layer_input"), u32v(ple_dim as u32)),
        ]);
    }
    let sections = |s: [u32; 4]| {
        gguf_file::Value::Array(s.iter().map(|&x| gguf_file::Value::U32(x)).collect())
    };
    match arch {
        // M-RoPE over 4 pairs with sections [1, 1, 0, 1]: pair 2 is frozen.
        "qwen2vl" => metadata.push((key("rope.dimension_sections"), sections([1, 1, 0, 1]))),
        // IM-RoPE over 4 pairs with sections [2, 1, 0, 0]: pair 2 is frozen.
        "qwen3vl" | "qwen3vlmoe" => {
            metadata.push((key("rope.dimension_sections"), sections([2, 1, 0, 0])))
        }
        "qwen35" | "qwen35moe" => {
            metadata.push((key("rope.dimension_sections"), sections([1, 1, 0, 0])))
        }
        // IM-RoPE over 2 pairs with sections [1, 0, 0, 1]: pair 1 is frozen.
        "qwen4exp" => metadata.push((key("rope.dimension_sections"), sections([1, 0, 0, 1]))),
        _ => {}
    }
    if hybrid {
        metadata.extend([
            (key("rope.dimension_count"), u32v((hd / 2) as u32)),
            (key("full_attention_interval"), u32v(2)),
            (key("ssm.conv_kernel"), u32v(d_conv as u32)),
            (key("ssm.state_size"), u32v(sd as u32)),
            (key("ssm.group_count"), u32v(nk as u32)),
            (key("ssm.time_step_rank"), u32v(nv as u32)),
            (key("ssm.inner_size"), u32v((nv * sd) as u32)),
        ]);
    }
    metadata.extend(tiny_tokenizer_metadata());

    let ones = |n: usize| vec![1.0f32; n];
    let mut tensors: Vec<(String, QTensor)> = vec![
        (
            "token_embd.weight".to_string(),
            qtensor(weights(VOCAB * EMB, 1), &[VOCAB, EMB]),
        ),
        ("output_norm.weight".to_string(), qtensor(ones(EMB), &[EMB])),
    ];
    if hyper {
        tensors.pop(); // the final hyper-connection mixer is the output norm
    }

    let mut seed = 10u32;
    let mut next = |n: usize| {
        seed = seed.wrapping_add(7).wrapping_mul(2_654_435_761) | 1;
        weights(n, seed)
    };
    // Gammas near 1 (the converter folds zero-centred ones to 1 + w).
    let gamma = |w: Vec<f32>| w.into_iter().map(|x| 1.0 + 3.0 * x).collect::<Vec<f32>>();
    let q_dim = h * hd * if hybrid { 2 } else { 1 };
    if hyper {
        let wide = hc * EMB;
        tensors.push(("output_hc_norm.weight".to_string(), qtensor(gamma(next(wide)), &[hc, EMB])));
        tensors.push(("output_hc_down.weight".to_string(), qtensor(next(hc_rank * wide).iter().map(|x| x * 10.0).collect(), &[hc_rank, wide])));
        tensors.push(("output_hc_up.weight".to_string(), qtensor(next(wide * hc_rank).iter().map(|x| x * 10.0).collect(), &[wide, hc_rank])));
        tensors.push((
            "per_layer_token_embd.weight".to_string(),
            qtensor(next(ple_rows * ple_dim).iter().map(|x| x * 10.0).collect(), &[ple_rows, ple_dim]),
        ));
    }
    for i in 0..n_layer {
        let p = format!("blk.{i}");
        let mut push = |name: &str, data: Vec<f32>, shape: &[usize]| {
            tensors.push((format!("{p}.{name}"), qtensor(data, shape)));
        };
        if hyper {
            let wide = hc * EMB;
            for m in ["hc_attn", "hc_ffn"] {
                push(&format!("{m}_norm.weight"), gamma(next(wide)), &[hc, EMB]);
                push(&format!("{m}_down.weight"), next(hc_rank * wide).iter().map(|x| x * 10.0).collect(), &[hc_rank, wide]);
                push(&format!("{m}_up.weight"), next(wide * hc_rank).iter().map(|x| x * 10.0).collect(), &[wide, hc_rank]);
                push(&format!("{m}_inject.weight"), next(hc * wide).iter().map(|x| x * 10.0).collect(), &[hc, wide]);
            }
            if i == 0 {
                push("ple_key.weight", next(wide * EMB).iter().map(|x| x * 10.0).collect(), &[wide, EMB]);
                push("ple_value.weight", next(EMB * EMB).iter().map(|x| x * 10.0).collect(), &[EMB, EMB]);
                for n in ["ple_norm_key", "ple_norm_query", "ple_norm_conv"] {
                    push(&format!("{n}.weight"), gamma(next(wide)), &[hc, EMB]);
                }
                push("ple_conv1d.weight", next(wide * ple_kernel).iter().map(|x| x * 10.0).collect(), &[wide, ple_kernel]);
            } else {
                push("indexer.q_proj.weight", next(idx_heads * idx_dim * EMB).iter().map(|x| x * 10.0).collect(), &[idx_heads * idx_dim, EMB]);
                push("indexer.k_proj.weight", next(idx_dim * EMB).iter().map(|x| x * 10.0).collect(), &[idx_dim, EMB]);
                push("indexer.q_norm.weight", gamma(next(idx_dim)), &[idx_dim]);
                push("indexer.k_norm.weight", gamma(next(idx_dim)), &[idx_dim]);
            }
        } else {
            push("attn_norm.weight", ones(EMB), &[EMB]);
            push(
                if hybrid {
                    "post_attention_norm.weight"
                } else {
                    "ffn_norm.weight"
                },
                ones(EMB),
                &[EMB],
            );
        }
        if hybrid && i == 0 {
            // Gated DeltaNet layer.
            push("attn_qkv.weight", next(conv_dim * EMB), &[conv_dim, EMB]);
            push("attn_gate.weight", next(nv * sd * EMB), &[nv * sd, EMB]);
            // Conv taps scaled up so the kernel visibly mixes positions.
            push(
                "ssm_conv1d.weight",
                next(conv_dim * d_conv).iter().map(|x| x * 10.0).collect(),
                &[conv_dim, d_conv],
            );
            push("ssm_dt.bias", next(nv), &[nv]);
            push(
                "ssm_a",
                next(nv).iter().map(|x| -1.0 - x.abs() * 10.0).collect(),
                &[nv],
            );
            if arch == "qwen3next" {
                push("ssm_ba.weight", next(2 * nv * EMB), &[2 * nv, EMB]);
            } else {
                push("ssm_beta.weight", next(nv * EMB), &[nv, EMB]);
                push("ssm_alpha.weight", next(nv * EMB), &[nv, EMB]);
            }
            push("ssm_norm.weight", ones(sd), &[sd]);
            push("ssm_out.weight", next(EMB * nv * sd), &[EMB, nv * sd]);
        } else {
            if fused_qkv {
                let n = q_dim + 2 * kv * hd;
                push("attn_qkv.weight", next(n * EMB), &[n, EMB]);
                push("attn_qkv.bias", next(n), &[n]);
            } else {
                push("attn_q.weight", next(q_dim * EMB), &[q_dim, EMB]);
                push("attn_k.weight", next(kv * hd * EMB), &[kv * hd, EMB]);
                push("attn_v.weight", next(kv * hd * EMB), &[kv * hd, EMB]);
                if biases {
                    push("attn_q.bias", next(q_dim), &[q_dim]);
                    push("attn_k.bias", next(kv * hd), &[kv * hd]);
                    push("attn_v.bias", next(kv * hd), &[kv * hd]);
                }
            }
            push("attn_output.weight", next(EMB * h * hd), &[EMB, h * hd]);
            if qk_norm {
                push("attn_q_norm.weight", ones(hd), &[hd]);
                push("attn_k_norm.weight", ones(hd), &[hd]);
            }
        }
        if moe {
            push("ffn_gate_inp.weight", next(NE * EMB), &[NE, EMB]);
            push(
                "ffn_gate_exps.weight",
                next(NE * NFE * EMB),
                &[NE, NFE, EMB],
            );
            push("ffn_up_exps.weight", next(NE * NFE * EMB), &[NE, NFE, EMB]);
            push(
                "ffn_down_exps.weight",
                next(NE * EMB * NFE),
                &[NE, EMB, NFE],
            );
            if shared_expert {
                push("ffn_gate_inp_shexp.weight", next(EMB), &[EMB]);
                push("ffn_gate_shexp.weight", next(NFE * EMB), &[NFE, EMB]);
                push("ffn_up_shexp.weight", next(NFE * EMB), &[NFE, EMB]);
                push("ffn_down_shexp.weight", next(EMB * NFE), &[EMB, NFE]);
            }
        } else {
            push("ffn_gate.weight", next(NFF * EMB), &[NFF, EMB]);
            push("ffn_up.weight", next(NFF * EMB), &[NFF, EMB]);
            push("ffn_down.weight", next(EMB * NFF), &[EMB, NFF]);
        }
    }

    let metadata_refs: Vec<(&str, &gguf_file::Value)> =
        metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut file = File::create(path).unwrap();
    gguf_file::write(&mut file, &metadata_refs, &tensor_refs).unwrap();
}

/// Tiny `qwen3moe` GGUF scaled up enough to run on Metal.
///
/// The CPU writer above uses head_dim=4, which Metal rejects (head dims must
/// be >= 32).  This variant uses the real model's head_dim=128 and enough
/// width for Q4_K block alignment (last dim a multiple of 256), plus 128
/// experts with top-8 routing like the real Qwen3-30B-A3B.  Norms are F32
/// (they are not multiples of 256, so they cannot be Q4_K).
pub fn write_tiny_qwen3moe_metal_gguf(path: &Path) {
    const VOCAB: usize = 16;
    const EMB: usize = 256;
    const H: usize = 2; // heads (H*HD = 256)
    const KV: usize = 1; // kv heads
    const HD: usize = 128; // head_dim: real Qwen3-30B-A3B uses 128
    const NFE: usize = 256; // expert ffn (mult of 256 for Q4K)
    const NE: usize = 128; // experts — match real Qwen3-30B-A3B
    const NLAYER: usize = 2;

    let u32v = |v: u32| gguf_file::Value::U32(v);
    let f32v = |v: f32| gguf_file::Value::F32(v);
    let key = |s: &str| format!("qwen3moe.{s}");
    let mut metadata: Vec<(String, gguf_file::Value)> = vec![
        (
            "general.architecture".to_string(),
            gguf_file::Value::String("qwen3moe".to_string()),
        ),
        (key("attention.head_count"), u32v(H as u32)),
        (key("attention.head_count_kv"), u32v(KV as u32)),
        (key("block_count"), u32v(NLAYER as u32)),
        (key("embedding_length"), u32v(EMB as u32)),
        (key("context_length"), u32v(512)),
        (key("attention.layer_norm_rms_epsilon"), f32v(1e-5)),
        (key("attention.key_length"), u32v(HD as u32)),
        (key("rope.freq_base"), f32v(10_000.0)),
        (key("expert_count"), u32v(NE as u32)),
        (key("expert_used_count"), u32v(8)),
        (key("expert_feed_forward_length"), u32v(NFE as u32)),
        (key("expert_shared_feed_forward_length"), u32v(0)),
        (
            key("attention.norm_topk_prob"),
            gguf_file::Value::Bool(true),
        ),
    ];
    metadata.extend(tiny_tokenizer_metadata());

    let ones = |n: usize| vec![1.0f32; n];
    let mut tensors: Vec<(String, QTensor)> = vec![
        (
            "token_embd.weight".to_string(),
            qtensor_q4k(weights(VOCAB * EMB, 1), &[VOCAB, EMB]),
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
        tensors.push((
            format!("{p}.attn_q.weight"),
            qtensor_q4k(next(H * HD * EMB), &[H * HD, EMB]),
        ));
        tensors.push((
            format!("{p}.attn_k.weight"),
            qtensor_q4k(next(KV * HD * EMB), &[KV * HD, EMB]),
        ));
        tensors.push((
            format!("{p}.attn_v.weight"),
            qtensor_q4k(next(KV * HD * EMB), &[KV * HD, EMB]),
        ));
        tensors.push((
            format!("{p}.attn_output.weight"),
            qtensor_q4k(next(EMB * H * HD), &[EMB, H * HD]),
        ));
        tensors.push((format!("{p}.attn_q_norm.weight"), qtensor(ones(HD), &[HD])));
        tensors.push((format!("{p}.attn_k_norm.weight"), qtensor(ones(HD), &[HD])));
        tensors.push((
            format!("{p}.ffn_gate_inp.weight"),
            qtensor(next(NE * EMB), &[NE, EMB]),
        ));
        tensors.push((
            format!("{p}.ffn_gate_exps.weight"),
            qtensor_q4k(next(NE * NFE * EMB), &[NE, NFE, EMB]),
        ));
        tensors.push((
            format!("{p}.ffn_up_exps.weight"),
            qtensor_q4k(next(NE * NFE * EMB), &[NE, NFE, EMB]),
        ));
        tensors.push((
            format!("{p}.ffn_down_exps.weight"),
            qtensor_q4k(next(NE * EMB * NFE), &[NE, EMB, NFE]),
        ));
    }

    let metadata_refs: Vec<(&str, &gguf_file::Value)> =
        metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut file = File::create(path).unwrap();
    gguf_file::write(&mut file, &metadata_refs, &tensor_refs).unwrap();
}
pub fn write_unsupported_gguf(path: &Path) {
    let arch = gguf_file::Value::String("mamba".to_string());
    let metadata: Vec<(&str, &gguf_file::Value)> = vec![("general.architecture", &arch)];
    let mut file = File::create(path).unwrap();
    gguf_file::write(&mut file, &metadata, &[]).unwrap();
}

// ─── DeepSeek-V4 test writer ──────────────────────────────────────────────────
//
// The deepseek4 loader reads tensors candle cannot *name* (IQ2_XXS experts,
// I32 routed-id tables), so the tiny model is written by hand rather than via
// candle's `gguf_file::write`.  The layout below mirrors candle's writer
// exactly (version 2, 32-byte alignment, per-tensor data padding) so the
// same file parses identically through candle's reader and Joshua's.

/// GGUF dtype ids used by the hand-rolled writer.
const DTYPE_F32: u32 = 0;
const DTYPE_F16: u32 = 1;
const DTYPE_Q8_0: u32 = 8;
const DTYPE_Q2_K: u32 = 10;
const DTYPE_Q4_K: u32 = 12;
const DTYPE_IQ2_XXS: u32 = 16;
const DTYPE_I32: u32 = 26;

/// A raw tensor for the hand-rolled writer: dtype id + candle-order dims +
/// exact file bytes (block layout included).
pub struct RawTensor {
    pub name: String,
    pub dtype: u32,
    pub dims: Vec<usize>,
    pub data: Vec<u8>,
}

impl RawTensor {
    pub fn f32(name: &str, data: Vec<f32>, dims: &[usize]) -> Self {
        let data = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            name: name.into(),
            dtype: DTYPE_F32,
            dims: dims.to_vec(),
            data,
        }
    }

    pub fn f16(name: &str, data: Vec<f32>, dims: &[usize]) -> Self {
        let data = data
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        Self {
            name: name.into(),
            dtype: DTYPE_F16,
            dims: dims.to_vec(),
            data,
        }
    }

    pub fn i32(name: &str, data: Vec<i32>, dims: &[usize]) -> Self {
        let data = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            name: name.into(),
            dtype: DTYPE_I32,
            dims: dims.to_vec(),
            data,
        }
    }

    /// Quantize f32 data to Q8_0 via candle (matches the other writers).
    pub fn q8_0(name: &str, data: Vec<f32>, dims: &[usize]) -> Self {
        let t = Tensor::from_vec(data, dims, &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&t, GgmlDType::Q8_0).unwrap();
        Self {
            name: name.into(),
            dtype: DTYPE_Q8_0,
            dims: qt.shape().dims().to_vec(),
            data: qt.data().unwrap().to_vec(),
        }
    }

    /// Quantize f32 data to Q2_K via candle (the real DeepSeek-V4-Flash
    /// files store the routed down projections this way).  Same alignment
    /// contract as [`RawTensor::q4k`].
    pub fn q2k(name: &str, data: Vec<f32>, dims: &[usize]) -> Self {
        let t = Tensor::from_vec(data, dims, &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&t, GgmlDType::Q2K).unwrap();
        Self {
            name: name.into(),
            dtype: DTYPE_Q2_K,
            dims: qt.shape().dims().to_vec(),
            data: qt.data().unwrap().to_vec(),
        }
    }

    /// Quantize f32 data to Q4_K via candle.
    ///
    /// The total element count must be a multiple of QK_K = 256 (candle's
    /// `QTensor::quantize` requires it), and the innermost dim must be
    /// block-aligned too for the matmul contract — the deepseek4 tiny-model
    /// weights chosen for this (attn_kv, shared-expert gate/up) satisfy both.
    pub fn q4k(name: &str, data: Vec<f32>, dims: &[usize]) -> Self {
        let t = Tensor::from_vec(data, dims, &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&t, GgmlDType::Q4K).unwrap();
        Self {
            name: name.into(),
            dtype: DTYPE_Q4_K,
            dims: qt.shape().dims().to_vec(),
            data: qt.data().unwrap().to_vec(),
        }
    }

    /// Encode f32 data as IQ2_XXS blocks.
    ///
    /// The test model does not need *accurate* quantisation — it needs valid
    /// blocks that exercise the decode path with varied, deterministic data.
    /// Values must stay *realistic*, though: candle's Q8_0 matmul quantizes
    /// the activation side to f16, so if the synthetic IQ2_XXS weights decode
    /// to huge magnitudes the MoE products overflow to inf.  The codebook's
    /// largest byte is 43, so a block scale of 2^-10 caps decoded weights at
    /// 43 * 3.875 / 1024 ≈ 0.16 (real weights are ~±0.1), which keeps every
    /// intermediate under f16/f32 range.
    /// Layout matches llama.cpp: 2-byte fp16 scale, then 8 groups of 32
    /// values: 4 code bytes + one u32 of sign bits (7 per 8-value group) with
    /// the scale index in bits 28..31.
    pub fn iq2xxs(name: &str, data: Vec<f32>, dims: &[usize]) -> Self {
        assert!(
            data.len().is_multiple_of(256),
            "iq2xxs test writer needs a multiple of 256 elements"
        );
        let mut out = Vec::with_capacity(data.len() / 256 * 66);
        for (b, chunk) in data.chunks_exact(256).enumerate() {
            // fp16 scale 2^-10 keeps decoded values ~±0.16.
            out.extend_from_slice(&half::f16::from_f32(1.0 / 1024.0).to_le_bytes());
            for ib32 in 0..8usize {
                let base = ib32 * 32;
                let mut lo = [0u8; 4];
                for l in 0..4usize {
                    let group = &chunk[base + l * 8..base + l * 8 + 8];
                    let sum: u32 = group.iter().map(|x| x.abs() as u32).sum::<u32>();
                    let code = ((ib32 * 53 + l * 97 + b * 13) as u32 + sum) % 256;
                    lo[l] = code as u8;
                }
                let mut hi: u32 = 0;
                for l in 0..4usize {
                    let group = &chunk[base + l * 8..base + l * 8 + 8];
                    let mut sbits: u8 = 0;
                    for (j, &v) in group.iter().enumerate().take(7) {
                        if v < 0.0 {
                            sbits |= 1 << j;
                        }
                    }
                    hi |= (sbits as u32) << (7 * l);
                }
                hi |= ((b % 16) as u32) << 28;
                out.extend_from_slice(&lo);
                out.extend_from_slice(&hi.to_le_bytes());
            }
        }
        Self {
            name: name.into(),
            dtype: DTYPE_IQ2_XXS,
            dims: dims.to_vec(),
            data: out,
        }
    }
}

fn value_type_id(v: &gguf_file::Value) -> u32 {
    match v {
        gguf_file::Value::U8(_) => 0,
        gguf_file::Value::I8(_) => 1,
        gguf_file::Value::U16(_) => 2,
        gguf_file::Value::I16(_) => 3,
        gguf_file::Value::U32(_) => 4,
        gguf_file::Value::I32(_) => 5,
        gguf_file::Value::F32(_) => 6,
        gguf_file::Value::Bool(_) => 7,
        gguf_file::Value::String(_) => 8,
        gguf_file::Value::Array(_) => 9,
        gguf_file::Value::U64(_) => 10,
        gguf_file::Value::I64(_) => 11,
        gguf_file::Value::F64(_) => 12,
    }
}

fn write_value(w: &mut impl std::io::Write, v: &gguf_file::Value) {
    let write_string = |w: &mut dyn std::io::Write, s: &str| {
        let b = s.as_bytes();
        w.write_all(&(b.len() as u64).to_le_bytes()).unwrap();
        w.write_all(b).unwrap();
    };
    match v {
        gguf_file::Value::U8(x) => w.write_all(&[*x]).unwrap(),
        gguf_file::Value::I8(x) => w.write_all(&[*x as u8]).unwrap(),
        gguf_file::Value::U16(x) => w.write_all(&x.to_le_bytes()).unwrap(),
        gguf_file::Value::I16(x) => w.write_all(&x.to_le_bytes()).unwrap(),
        gguf_file::Value::U32(x) => w.write_all(&x.to_le_bytes()).unwrap(),
        gguf_file::Value::I32(x) => w.write_all(&x.to_le_bytes()).unwrap(),
        gguf_file::Value::U64(x) => w.write_all(&x.to_le_bytes()).unwrap(),
        gguf_file::Value::I64(x) => w.write_all(&x.to_le_bytes()).unwrap(),
        gguf_file::Value::F32(x) => w.write_all(&x.to_le_bytes()).unwrap(),
        gguf_file::Value::F64(x) => w.write_all(&x.to_le_bytes()).unwrap(),
        gguf_file::Value::Bool(x) => w.write_all(&[u8::from(*x)]).unwrap(),
        gguf_file::Value::String(s) => write_string(w, s),
        gguf_file::Value::Array(a) => {
            let vt: u32 = if a.is_empty() {
                4
            } else {
                match a[0] {
                    gguf_file::Value::U8(_) => 0,
                    gguf_file::Value::I8(_) => 1,
                    gguf_file::Value::U16(_) => 2,
                    gguf_file::Value::I16(_) => 3,
                    gguf_file::Value::U32(_) => 4,
                    gguf_file::Value::I32(_) => 5,
                    gguf_file::Value::F32(_) => 6,
                    gguf_file::Value::Bool(_) => 7,
                    gguf_file::Value::String(_) => 8,
                    gguf_file::Value::U64(_) => 10,
                    gguf_file::Value::I64(_) => 11,
                    gguf_file::Value::F64(_) => 12,
                    _ => panic!("nested arrays unsupported"),
                }
            };
            w.write_all(&vt.to_le_bytes()).unwrap();
            w.write_all(&(a.len() as u64).to_le_bytes()).unwrap();
            for e in a {
                write_value(w, e);
            }
        }
    }
}

/// Write a GGUF in candle's exact on-disk layout, but for arbitrary dtypes.
pub fn write_raw_gguf(path: &Path, metadata: &[(String, gguf_file::Value)], tensors: &[RawTensor]) {
    let mut w = Vec::new();
    w.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
    w.extend_from_slice(&2u32.to_le_bytes()); // version 2
    w.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    w.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    let write_string = |w: &mut Vec<u8>, s: &str| {
        let b = s.as_bytes();
        w.extend_from_slice(&(b.len() as u64).to_le_bytes());
        w.extend_from_slice(b);
    };
    for (k, v) in metadata {
        write_string(&mut w, k);
        w.extend_from_slice(&value_type_id(v).to_le_bytes());
        write_value(&mut w, v);
    }
    // Tensor infos: dims reversed (GGUF order), 32-byte-aligned offsets.
    let mut offset = 0usize;
    let mut data_offsets = Vec::with_capacity(tensors.len());
    for t in tensors {
        write_string(&mut w, &t.name);
        w.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in t.dims.iter().rev() {
            w.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        w.extend_from_slice(&t.dtype.to_le_bytes());
        w.extend_from_slice(&(offset as u64).to_le_bytes());
        let size = t.data.len();
        let padding = 31 - (31 + size) % 32;
        data_offsets.push(offset);
        offset += size + padding;
    }
    // Header padded to 32 bytes before the data region.
    let padding = 31 - (31 + w.len()) % 32;
    w.resize(w.len() + padding, 0u8);
    let data_start = w.len();
    for (t, &off) in tensors.iter().zip(data_offsets.iter()) {
        assert_eq!(
            w.len() - data_start,
            off,
            "data offsets must be relative to the aligned data start"
        );
        w.extend_from_slice(&t.data);
        let padding = 31 - (31 + t.data.len()) % 32;
        w.resize(w.len() + padding, 0u8);
    }
    std::fs::write(path, w).unwrap();
}

/// Write a tiny but structurally valid `deepseek4` GGUF exercising the
/// dtypes candle cannot represent: IQ2_XXS routed experts and an I32
/// tid2eid table in the hash layer.
/// Options controlling which tensors the tiny `deepseek4` writer emits.
#[derive(Default, Clone, Copy)]
pub struct TinyDeepseek4Opts {
    /// Also emit `output.weight` as IQ2_XXS (a dtype candle cannot name).
    pub iq2xxs_output: bool,
    /// Layer 0 is a CSA layer (compressor + lightning indexer tensors).
    pub compress: bool,
    /// Emit K-quant (Q4_K) weights for the attention-KV and shared-expert
    /// projections instead of F16 — exercises the K-quant decode path on
    /// both the mmap and streamed loaders.
    pub kquant_weights: bool,
    /// Use only dtypes candle can name (Q8_0 experts, F32 id table instead of
    /// I32) so the model loads through `ModelWeights::from_gguf`, whose
    /// streamed entry point has no raw header.
    pub candle_only: bool,
    /// Store the routed down projections as Q2_K with a 256-wide expert
    /// (the real V4-Flash layout: IQ2_XXS gate/up, Q2_K down), instead of
    /// Q8_0 with a 128-wide one.
    pub q2k_down: bool,
    /// Override routed intermediate width for multi-block shard fixtures.
    pub expert_width: Option<usize>,
    /// Write a `deepseek41` (DeepSeek-V4.1) file instead: 4 layers with
    /// compress ratios [2, 2, 1, 0] — layer 0 a gated ratio-2 source that
    /// owns the index keys and scores top-k, layer 1 a reader that scores
    /// layer 0's keys (and carries the engram), layer 2 a gateless ratio-1
    /// source owning its keys, layer 3 window-only — no hyper-connection
    /// head, no hash layers, randomized norms.
    pub v41: bool,
}

pub fn write_tiny_deepseek4_gguf(path: &Path) {
    write_tiny_deepseek4_gguf_opts(path, TinyDeepseek4Opts::default());
}

/// The tiny `deepseek41` (DeepSeek-V4.1) fixture; see [`TinyDeepseek4Opts::v41`].
pub fn write_tiny_deepseek41_gguf(path: &Path) {
    write_tiny_deepseek4_gguf_opts(
        path,
        TinyDeepseek4Opts {
            v41: true,
            ..Default::default()
        },
    );
}

/// Like [`write_tiny_deepseek4_gguf`], but also emits `output.weight` as an
/// IQ2_XXS tensor (a dtype candle cannot name) so callers can verify the
/// loader's raw-header-aware presence check picks it up instead of silently
/// tying the output head to the input embeddings.
pub fn write_tiny_deepseek4_gguf_iq2xxs_output(path: &Path) {
    write_tiny_deepseek4_gguf_opts(
        path,
        TinyDeepseek4Opts {
            iq2xxs_output: true,
            ..Default::default()
        },
    );
}

/// Like [`write_tiny_deepseek4_gguf`], but layer 0 is a CSA layer
/// (`compress_ratios = [4, 0]`): emits the main compressor + lightning
/// indexer tensors so the compressed-KV cache write path (scatter into
/// `comp`/`lid`) is exercised by forward passes that span a full block.
pub fn write_tiny_deepseek4_gguf_compress(path: &Path) {
    write_tiny_deepseek4_gguf_opts(
        path,
        TinyDeepseek4Opts {
            compress: true,
            ..Default::default()
        },
    );
}

/// Like [`write_tiny_deepseek4_gguf`], but the attention-KV projection and
/// the shared-expert gate/up projections are Q4_K instead of F16.
pub fn write_tiny_deepseek4_gguf_kquant(path: &Path) {
    write_tiny_deepseek4_gguf_opts(
        path,
        TinyDeepseek4Opts {
            kquant_weights: true,
            ..Default::default()
        },
    );
}

/// Like [`write_tiny_deepseek4_gguf`], but every tensor uses a dtype candle's
/// `GgmlDType` can name (routed experts Q8_0 instead of IQ2_XXS, id table F32
/// instead of I32) so the file also loads through `ModelWeights::from_gguf`
/// (no raw header).
/// Like [`write_tiny_deepseek4_gguf`] with the real V4-Flash expert layout:
/// IQ2_XXS gate/up and a Q2_K down projection over a 256-wide expert.
pub fn write_tiny_deepseek4_gguf_q2k_down(path: &Path) {
    write_tiny_deepseek4_gguf_opts(
        path,
        TinyDeepseek4Opts {
            q2k_down: true,
            ..Default::default()
        },
    );
}

pub fn write_tiny_deepseek4_gguf_candle_only(path: &Path) {
    write_tiny_deepseek4_gguf_opts(
        path,
        TinyDeepseek4Opts {
            candle_only: true,
            ..Default::default()
        },
    );
}

pub fn write_tiny_deepseek4_gguf_opts(path: &Path, opts: TinyDeepseek4Opts) {
    let TinyDeepseek4Opts {
        iq2xxs_output,
        compress,
        kquant_weights,
        candle_only,
        q2k_down,
        expert_width,
        v41,
    } = opts;
    let arch = if v41 { "deepseek41" } else { "deepseek4" };
    const VOCAB: usize = 16;
    // EMB must be a multiple of the IQ2_XXS block size (256): each expert's
    // in-dimension is EMB, and candle requires the innermost dim of a QTensor
    // to be block-aligned.  The real DeepSeek-V4-Flash files satisfy this
    // with EMB = 4096.
    const EMB: usize = 256;
    let nlayer: usize = if v41 { 4 } else { 2 };
    const NHEAD: usize = 4;
    const HEAD_DIM: usize = 16;
    const ROPE_DIM: usize = 8;
    const Q_LORA: usize = 8;
    const O_GROUPS: usize = 2;
    const O_LORA: usize = 4;
    const NE: usize = 8;
    // Expert ffn width (out of gate/up, in of down): 256 so a Q2_K down
    // projection has a block-aligned contraction dim, 128 otherwise.
    let nfe: usize = expert_width.unwrap_or(if q2k_down { 256 } else { 128 });
    const NUSED: usize = 2;
    const N_SHARED: usize = 1;
    const HC: usize = 2;
    let n_hash: usize = if v41 { 0 } else { 1 };
    // Lightning indexer (CSA layers only).
    const INDEX_NHEAD: usize = 4;
    const INDEX_HD: usize = 16;

    let u32v = |v: u32| gguf_file::Value::U32(v);
    let f32v = |v: f32| gguf_file::Value::F32(v);
    let key = |s: &str| format!("{arch}.{s}");
    let ratios: Vec<u32> = if v41 {
        vec![2, 2, 1, 0]
    } else {
        vec![if compress { 4 } else { 0 }, 0]
    };
    let mut metadata: Vec<(String, gguf_file::Value)> = vec![
        (
            "general.architecture".to_string(),
            gguf_file::Value::String(arch.to_string()),
        ),
        (
            "general.name".to_string(),
            gguf_file::Value::String(if v41 { "DeepSeek-V4.1" } else { "DeepSeek-V4" }.to_string()),
        ),
        (key("block_count"), u32v(nlayer as u32)),
        (key("attention.head_count"), u32v(NHEAD as u32)),
        (key("embedding_length"), u32v(EMB as u32)),
        (key("attention.layer_norm_rms_epsilon"), f32v(1e-5)),
        (key("attention.q_lora_rank"), u32v(Q_LORA as u32)),
        (key("attention.key_length"), u32v(HEAD_DIM as u32)),
        (key("rope.dimension_count"), u32v(ROPE_DIM as u32)),
        (
            key("attention.compress_ratios"),
            gguf_file::Value::Array(ratios.iter().map(|&r| u32v(r)).collect()),
        ),
        (key("attention.sliding_window"), u32v(8)),
        (key("attention.output_group_count"), u32v(O_GROUPS as u32)),
        (key("attention.output_lora_rank"), u32v(O_LORA as u32)),
        (key("hyper_connection.count"), u32v(HC as u32)),
        (key("hyper_connection.sinkhorn_iterations"), u32v(2)),
        (key("hyper_connection.epsilon"), f32v(1e-6)),
        (key("expert_count"), u32v(NE as u32)),
        (key("expert_used_count"), u32v(NUSED as u32)),
        (key("expert_shared_count"), u32v(N_SHARED as u32)),
        (key("hash_layer_count"), u32v(n_hash as u32)),
        (key("expert_weights_scale"), f32v(1.0)),
        (key("expert_gating_func"), u32v(2)),
        (key("rope.freq_base"), f32v(10_000.0)),
        // 512-token context: long-prefill engine tests (chunked vs single,
        // KV continuation) need headroom beyond the old 32-token cap.
        (key("context_length"), u32v(512)),
        ("tokenizer.ggml.eos_token_id".to_string(), u32v(3)),
        ("tokenizer.ggml.bos_token_id".to_string(), u32v(3)),
        ("tokenizer.ggml.unknown_token_id".to_string(), u32v(0)),
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
            gguf_file::Value::Array(
                (0..16)
                    .map(|i| gguf_file::Value::F32(-(i as f32)))
                    .collect(),
            ),
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
    ];
    if compress || v41 {
        metadata.extend([
            (
                key("attention.indexer.head_count"),
                u32v(INDEX_NHEAD as u32),
            ),
            (key("attention.indexer.key_length"), u32v(INDEX_HD as u32)),
            (key("attention.indexer.top_k"), u32v(2)),
        ]);
    }
    // Engram on layer 1: 3-grams, 2 heads per n-gram, 16-wide rows.
    const ENGRAM_KEY: usize = 16;
    let engram_primes: [u64; 4] = [5, 7, 11, 13];
    let engram_rows = engram_primes.iter().sum::<u64>() as usize;
    if v41 {
        let u64s = |v: &[u64]| gguf_file::Value::Array(v.iter().map(|&x| gguf_file::Value::U64(x)).collect());
        let offsets: Vec<u64> = engram_primes
            .iter()
            .scan(0, |acc, &p| {
                let o = *acc;
                *acc += p;
                Some(o)
            })
            .collect();
        metadata.extend([
            (key("attention.compress_rope_freq_base"), f32v(40_000.0)),
            (key("engram.layer_ids"), gguf_file::Value::Array(vec![gguf_file::Value::I32(1)])),
            (key("engram.head_count"), u32v(2)),
            (key("engram.key_length"), u32v(ENGRAM_KEY as u32)),
            (key("engram.max_ngram_size"), u32v(3)),
            (key("engram.multipliers"), u64s(&[1_000_003, 2_000_029, 3_000_017])),
            (key("engram.primes"), u64s(&engram_primes)),
            (key("engram.offsets"), u64s(&offsets)),
            // Case folding: tokens 4..=15 map pairwise onto one id.
            (
                key("engram.token_map"),
                gguf_file::Value::Array((0..VOCAB as i32).map(|t| gguf_file::Value::I32(if t < 4 { t } else { 4 + (t - 4) / 2 })).collect()),
            ),
            (key("engram.pad_id"), u32v(2)),
        ]);
    }

    // Norm gammas: ones for V4 (as it always was), randomized for V4.1 so a
    // misplaced norm moves the logits.
    let mut norm_seed = 500u32;
    let mut ones = |n: usize| {
        norm_seed += 1;
        if v41 {
            weights(n, norm_seed).iter().map(|w| 1.0 + 3.0 * w).collect()
        } else {
            vec![1.0f32; n]
        }
    };
    let mut seed = 10u32;
    let mut next = |n: usize| {
        seed = seed.wrapping_add(7).wrapping_mul(2_654_435_761) | 1;
        weights(n, seed)
    };

    let mut tensors: Vec<RawTensor> = vec![
        RawTensor::f16("token_embd.weight", weights(VOCAB * EMB, 1), &[VOCAB, EMB]),
        RawTensor::f32("output_norm.weight", ones(EMB), &[EMB]),
    ];
    if v41 {
        tensors.push(RawTensor::f16("output.weight", weights(VOCAB * EMB, 2), &[VOCAB, EMB]));
    } else {
        tensors.extend([
            RawTensor::f16(
                "output_hc_fn.weight",
                next(HC * (HC * EMB)),
                &[HC, HC * EMB],
            ),
            RawTensor::f32("output_hc_base.weight", vec![1.0; HC], &[HC]),
            RawTensor::f32("output_hc_scale.weight", vec![1.0; HC], &[HC]),
        ]);
    }
    let o_group_dim = (NHEAD / O_GROUPS) * HEAD_DIM;
    for i in 0..nlayer {
        let p = format!("blk.{i}");
        tensors.push(RawTensor::f32(
            &format!("{p}.attn_norm.weight"),
            ones(EMB),
            &[EMB],
        ));
        tensors.push(RawTensor::f32(
            &format!("{p}.ffn_norm.weight"),
            ones(EMB),
            &[EMB],
        ));
        tensors.push(RawTensor::f16(
            &format!("{p}.attn_q_a.weight"),
            next(Q_LORA * EMB),
            &[Q_LORA, EMB],
        ));
        tensors.push(RawTensor::f32(
            &format!("{p}.attn_q_a_norm.weight"),
            ones(Q_LORA),
            &[Q_LORA],
        ));
        tensors.push(RawTensor::f16(
            &format!("{p}.attn_q_b.weight"),
            next(NHEAD * HEAD_DIM * Q_LORA),
            &[NHEAD * HEAD_DIM, Q_LORA],
        ));
        // Q4_K needs the innermost dim (the matmul contraction dim) to be a
        // multiple of QK_K = 256; EMB = 256 satisfies it.
        let kv_data = next(HEAD_DIM * EMB);
        tensors.push(if kquant_weights {
            RawTensor::q4k(&format!("{p}.attn_kv.weight"), kv_data, &[HEAD_DIM, EMB])
        } else {
            RawTensor::f16(&format!("{p}.attn_kv.weight"), kv_data, &[HEAD_DIM, EMB])
        });
        tensors.push(RawTensor::f32(
            &format!("{p}.attn_kv_a_norm.weight"),
            ones(HEAD_DIM),
            &[HEAD_DIM],
        ));
        tensors.push(RawTensor::f16(
            &format!("{p}.attn_output_a.weight"),
            next(O_GROUPS * O_LORA * o_group_dim),
            &[O_GROUPS * O_LORA, o_group_dim],
        ));
        tensors.push(RawTensor::f16(
            &format!("{p}.attn_output_b.weight"),
            next(EMB * O_GROUPS * O_LORA),
            &[EMB, O_GROUPS * O_LORA],
        ));
        tensors.push(RawTensor::f32(
            &format!("{p}.attn_sinks.weight"),
            if v41 { next(NHEAD) } else { vec![1.0; NHEAD] },
            &[NHEAD],
        ));

        if v41 {
            // Sources: layers 0 (ratio 2, gated) and 2 (ratio 1, no gate).
            if i == 0 || i == 2 {
                tensors.push(RawTensor::f16(
                    &format!("{p}.attn_compressor_kv.weight"),
                    next(HEAD_DIM * EMB),
                    &[HEAD_DIM, EMB],
                ));
                if i == 0 {
                    tensors.push(RawTensor::f16(
                        &format!("{p}.attn_compressor_gate.weight"),
                        next(HEAD_DIM * EMB),
                        &[HEAD_DIM, EMB],
                    ));
                }
                tensors.push(RawTensor::f32(
                    &format!("{p}.attn_compressor_norm.weight"),
                    ones(HEAD_DIM),
                    &[HEAD_DIM],
                ));
                // Index-key owners.
                tensors.push(RawTensor::f16(
                    &format!("{p}.indexer.attn_k.weight"),
                    next(INDEX_HD * HEAD_DIM),
                    &[INDEX_HD, HEAD_DIM],
                ));
                tensors.push(RawTensor::f32(
                    &format!("{p}.indexer.k_norm.weight"),
                    ones(INDEX_HD),
                    &[INDEX_HD],
                ));
            }
            // Index sources: every compressed layer but none of the window-only one.
            if i < 3 {
                tensors.push(RawTensor::f16(
                    &format!("{p}.indexer.proj.weight"),
                    next(INDEX_NHEAD * EMB),
                    &[INDEX_NHEAD, EMB],
                ));
                tensors.push(RawTensor::f16(
                    &format!("{p}.indexer.attn_q_b.weight"),
                    next(INDEX_NHEAD * INDEX_HD * Q_LORA),
                    &[INDEX_NHEAD * INDEX_HD, Q_LORA],
                ));
            }
            if i == 1 {
                let n_cols = 4;
                tensors.push(RawTensor::f16(
                    &format!("{p}.engram_embd.weight"),
                    next(engram_rows * ENGRAM_KEY),
                    &[engram_rows, ENGRAM_KEY],
                ));
                tensors.push(RawTensor::f16(
                    &format!("{p}.engram_wkv.weight"),
                    next((HC + 1) * EMB * n_cols * ENGRAM_KEY),
                    &[(HC + 1) * EMB, n_cols * ENGRAM_KEY],
                ));
                tensors.push(RawTensor::f32(&format!("{p}.engram_q.weight"), ones(HC * EMB), &[HC, EMB]));
                tensors.push(RawTensor::f32(&format!("{p}.engram_k.weight"), ones(HC * EMB), &[HC, EMB]));
            }
        }

        // CSA layer 0: main compressor + lightning indexer (and its own
        // ratio-4 compressor).  Both write their caches with a scatter whose
        // index is a broadcast view — see `deepseek4_compressed_layers_write_their_caches`.
        if compress && i == 0 {
            let cdim = 2 * HEAD_DIM; // coff=2 for CSA
            tensors.push(RawTensor::f16(
                &format!("{p}.attn_compressor_kv.weight"),
                next(cdim * EMB),
                &[cdim, EMB],
            ));
            tensors.push(RawTensor::f16(
                &format!("{p}.attn_compressor_gate.weight"),
                next(cdim * EMB),
                &[cdim, EMB],
            ));
            tensors.push(RawTensor::f32(
                &format!("{p}.attn_compressor_ape.weight"),
                next(4 * cdim),
                &[4, cdim],
            ));
            tensors.push(RawTensor::f32(
                &format!("{p}.attn_compressor_norm.weight"),
                ones(HEAD_DIM),
                &[HEAD_DIM],
            ));
            tensors.push(RawTensor::f16(
                &format!("{p}.indexer.proj.weight"),
                next(INDEX_NHEAD * EMB),
                &[INDEX_NHEAD, EMB],
            ));
            tensors.push(RawTensor::f16(
                &format!("{p}.indexer.attn_q_b.weight"),
                next(INDEX_NHEAD * INDEX_HD * Q_LORA),
                &[INDEX_NHEAD * INDEX_HD, Q_LORA],
            ));
            let icdim = 2 * INDEX_HD;
            tensors.push(RawTensor::f16(
                &format!("{p}.indexer_compressor_kv.weight"),
                next(icdim * EMB),
                &[icdim, EMB],
            ));
            tensors.push(RawTensor::f16(
                &format!("{p}.indexer_compressor_gate.weight"),
                next(icdim * EMB),
                &[icdim, EMB],
            ));
            tensors.push(RawTensor::f32(
                &format!("{p}.indexer_compressor_ape.weight"),
                next(4 * icdim),
                &[4, icdim],
            ));
            tensors.push(RawTensor::f32(
                &format!("{p}.indexer_compressor_norm.weight"),
                ones(INDEX_HD),
                &[INDEX_HD],
            ));
        }

        // MoE: routed experts (IQ2_XXS gate/up, Q8_0 down) + shared expert.
        tensors.push(RawTensor::f32(
            &format!("{p}.ffn_gate_inp.weight"),
            next(NE * EMB),
            &[NE, EMB],
        ));
        if i >= n_hash {
            tensors.push(RawTensor::f32(
                &format!("{p}.exp_probs_b.bias"),
                next(NE),
                &[NE],
            ));
        }
        let gate_up = next(NE * nfe * EMB);
        let up = next(NE * nfe * EMB);
        let down = next(NE * nfe * EMB);
        // `candle_only` uses Q8_0 for the routed experts so the whole file
        // stays loadable without the raw header (IQ2_XXS is not a dtype
        // candle's `Content` can carry).
        let exps = |name: &str, data: Vec<f32>| {
            if candle_only {
                RawTensor::q8_0(name, data, &[NE, nfe, EMB])
            } else {
                RawTensor::iq2xxs(name, data, &[NE, nfe, EMB])
            }
        };
        tensors.push(exps(&format!("{p}.ffn_gate_exps.weight"), gate_up));
        tensors.push(exps(&format!("{p}.ffn_up_exps.weight"), up));
        tensors.push(if q2k_down {
            RawTensor::q2k(&format!("{p}.ffn_down_exps.weight"), down, &[NE, EMB, nfe])
        } else {
            RawTensor::q8_0(&format!("{p}.ffn_down_exps.weight"), down, &[NE, EMB, nfe])
        });
        // Shared expert: F16 normally, Q4_K in the k-quant variant (gate/up
        // have EMB = 256 as their contraction dim; down's is nfe = 128, so it
        // stays F16).
        let shexp = |name: &str, data: Vec<f32>| {
            if kquant_weights {
                RawTensor::q4k(name, data, &[nfe, EMB])
            } else {
                RawTensor::f16(name, data, &[nfe, EMB])
            }
        };
        tensors.push(shexp(
            &format!("{p}.ffn_gate_shexp.weight"),
            next(nfe * EMB),
        ));
        tensors.push(shexp(&format!("{p}.ffn_up_shexp.weight"), next(nfe * EMB)));
        tensors.push(RawTensor::f16(
            &format!("{p}.ffn_down_shexp.weight"),
            next(EMB * nfe),
            &[EMB, nfe],
        ));
        if i < n_hash {
            // Routed-id table: I32, [vocab, n_expert_used], ids in [0, NE).
            let ids: Vec<i32> = (0..VOCAB).map(|t| ((t * 3 + 1) % NE) as i32).collect();
            let ids = [ids.clone(), ids.clone()].concat();
            // `candle_only` writes the same values as F32 — the loader reads
            // either (I32 goes through the raw header, F32 through candle).
            if candle_only {
                let f32ids: Vec<f32> = ids.iter().map(|&v| v as f32).collect();
                tensors.push(RawTensor::f32(
                    &format!("{p}.ffn_gate_tid2eid.weight"),
                    f32ids,
                    &[VOCAB, NUSED],
                ));
            } else {
                tensors.push(RawTensor::i32(
                    &format!("{p}.ffn_gate_tid2eid.weight"),
                    ids,
                    &[VOCAB, NUSED],
                ));
            }
        }

        // Hyper-connection mixing: [hc*d] → [(2+hc)*hc], plus scale/base.
        let hc_out = (2 + HC) * HC;
        tensors.push(RawTensor::f16(
            &format!("{p}.hc_attn_fn.weight"),
            next(hc_out * HC * EMB),
            &[hc_out, HC * EMB],
        ));
        tensors.push(RawTensor::f32(
            &format!("{p}.hc_attn_base.weight"),
            next(hc_out),
            &[hc_out],
        ));
        let hc_scale = if v41 { vec![0.7, 1.3, 0.9] } else { vec![1.0, 1.0, 1.0] };
        tensors.push(RawTensor::f32(
            &format!("{p}.hc_attn_scale.weight"),
            hc_scale.clone(),
            &[3],
        ));
        tensors.push(RawTensor::f16(
            &format!("{p}.hc_ffn_fn.weight"),
            next(hc_out * HC * EMB),
            &[hc_out, HC * EMB],
        ));
        tensors.push(RawTensor::f32(
            &format!("{p}.hc_ffn_base.weight"),
            next(hc_out),
            &[hc_out],
        ));
        tensors.push(RawTensor::f32(
            &format!("{p}.hc_ffn_scale.weight"),
            hc_scale,
            &[3],
        ));
    }

    if iq2xxs_output {
        // Distinct from token_embd (seed 1) so the test can tell which head
        // was actually used.
        tensors.push(RawTensor::iq2xxs(
            "output.weight",
            weights(VOCAB * EMB, 999),
            &[VOCAB, EMB],
        ));
    }

    write_raw_gguf(path, &metadata, &tensors);
}

/// How [`write_tiny_bonsai_gguf`] stores its matrices.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BonsaiWeights {
    /// Bonsai's formats: Q1_0 embedding + attention, Q2_0 FFN + head.
    LowBit,
    /// The same values dequantized to F32 (a candle-readable twin).
    Dequantized,
}

/// Write a tiny `qwen3` GGUF in the layout Bonsai models ship: Q1_0
/// (1-bit) token embedding and attention projections, Q2_0 (2-bit) FFN and
/// output head, F32 norms.  Widths are block multiples (128-dim embedding,
/// 2 heads of 64, one KV head, 128-wide FFN, 2 layers).  With
/// [`BonsaiWeights::Dequantized`] every matrix holds exactly the values its
/// low-bit blocks decode to, stored as F32.
pub fn write_tiny_bonsai_gguf(path: &Path, weights_as: BonsaiWeights) {
    use joshua::low_bit::{blocks_as_bytes, quantize_q1_0, quantize_q2_0, BlockQ1_0, BlockQ2_0};
    use joshua::raw_block::{dequantize, RawBlock};
    const VOCAB: usize = 16;
    const EMB: usize = 128;
    const HD: usize = 64;
    const NFF: usize = 128;
    let (h, kv, n_layer) = (2usize, 1usize, 2usize);

    let arch = "qwen3";
    let key = |s: &str| format!("{arch}.{s}");
    let u32v = gguf_file::Value::U32;
    let metadata = vec![
        ("general.architecture".to_string(), gguf_file::Value::String(arch.into())),
        (key("attention.head_count"), u32v(h as u32)),
        (key("attention.head_count_kv"), u32v(kv as u32)),
        (key("block_count"), u32v(n_layer as u32)),
        (key("embedding_length"), u32v(EMB as u32)),
        (key("feed_forward_length"), u32v(NFF as u32)),
        (key("context_length"), u32v(512)),
        (key("attention.layer_norm_rms_epsilon"), gguf_file::Value::F32(1e-6)),
        (key("attention.key_length"), u32v(HD as u32)),
        (key("rope.freq_base"), gguf_file::Value::F32(1_000_000.0)),
    ];

    let mut seed = 0u32;
    let mut tensors = Vec::new();
    let mut matrix = |name: String, dims: &[usize], q1: bool| {
        seed += 1;
        let w = weights(dims.iter().product(), seed * 7919);
        let (dtype, bytes, decoded) = if q1 {
            let b = quantize_q1_0(&w);
            let mut d = vec![0f32; w.len()];
            dequantize(&b, &mut d).unwrap();
            (BlockQ1_0::GGML_TYPE, blocks_as_bytes(&b).to_vec(), d)
        } else {
            let b = quantize_q2_0(&w);
            let mut d = vec![0f32; w.len()];
            dequantize(&b, &mut d).unwrap();
            (BlockQ2_0::GGML_TYPE, blocks_as_bytes(&b).to_vec(), d)
        };
        tensors.push(match weights_as {
            BonsaiWeights::LowBit => RawTensor {
                name,
                dtype,
                dims: dims.to_vec(),
                data: bytes,
            },
            BonsaiWeights::Dequantized => RawTensor::f32(&name, decoded, dims),
        });
    };
    matrix("token_embd.weight".into(), &[VOCAB, EMB], true);
    for l in 0..n_layer {
        let p = |n: &str| format!("blk.{l}.{n}.weight");
        matrix(p("attn_q"), &[h * HD, EMB], true);
        matrix(p("attn_k"), &[kv * HD, EMB], true);
        matrix(p("attn_v"), &[kv * HD, EMB], true);
        matrix(p("attn_output"), &[EMB, h * HD], true);
        matrix(p("ffn_gate"), &[NFF, EMB], false);
        matrix(p("ffn_up"), &[NFF, EMB], false);
        matrix(p("ffn_down"), &[EMB, NFF], false);
    }
    matrix("output.weight".into(), &[VOCAB, EMB], false);
    let norm = |name: String, n: usize, seed: u32| {
        RawTensor::f32(&name, weights(n, seed).iter().map(|v| 1.0 + 5.0 * v).collect(), &[n])
    };
    for l in 0..n_layer {
        tensors.push(norm(format!("blk.{l}.attn_norm.weight"), EMB, 101 + l as u32));
        tensors.push(norm(format!("blk.{l}.ffn_norm.weight"), EMB, 201 + l as u32));
        tensors.push(norm(format!("blk.{l}.attn_q_norm.weight"), HD, 301 + l as u32));
        tensors.push(norm(format!("blk.{l}.attn_k_norm.weight"), HD, 401 + l as u32));
    }
    tensors.push(norm("output_norm.weight".into(), EMB, 501));
    write_raw_gguf(path, &metadata, &tensors);
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
