//! Exercise the real generation loop with a deterministic byte-token backend.
mod common;
use joshua::{
    npu::{NpuBackend, NpuSession},
    Engine, EngineOptions, GenerationOptions,
};
use std::{path::Path, sync::Arc};

struct BytesBackend;
struct BytesSession;
impl NpuBackend for BytesBackend {
    fn name(&self) -> String {
        "byte-decoding-test".into()
    }
    fn create_session(&self, _: &Path, _: u32) -> Result<Box<dyn NpuSession>, String> {
        Ok(Box::new(BytesSession))
    }
}
impl NpuSession for BytesSession {
    fn vocab_size(&self) -> usize {
        8
    }
    fn forward(&mut self, tokens: &[u32], _: usize) -> Result<Vec<f32>, String> {
        let next = match tokens.last() {
            Some(4) => 5,
            Some(5) => 6,
            Some(6) => 7,
            Some(7) => 3,
            _ => 4,
        };
        let mut logits = vec![-100.0; 8];
        logits[next] = 100.0;
        Ok(logits)
    }
    fn reset(&mut self) -> bool {
        true
    }
}

#[test]
fn generation_preserves_characters_split_across_byte_tokens() {
    use tokenizers::{decoders::byte_fallback::ByteFallback, models::bpe::BPE, Tokenizer};
    let dir = common::model_dir("output-byte-decoding");
    let path = dir.join("model.gguf");
    common::write_tiny_llama_gguf(&path);
    let vocabulary = [
        "<unk>", "hello", "world", "</s>", "<0xE2>", "<0x82>", "<0xAC>", "!",
    ];
    let vocab: [(String, u32); 8] = std::array::from_fn(|i| (vocabulary[i].to_owned(), i as u32));
    let mut tokenizer = Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".into())
            .byte_fallback(true)
            .build()
            .unwrap(),
    );
    tokenizer.with_decoder(Some(ByteFallback::default()));
    tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
    let mut options = EngineOptions::with_n_ctx(128);
    options.prefill_chunk_size = Some(2);
    let engine = Engine::with_options(&path, options)
        .unwrap()
        .with_npu_backend(Arc::new(BytesBackend));
    let generation = GenerationOptions {
        temperature: 0.0,
        max_tokens: 8,
        ..Default::default()
    };
    for prompt in ["hello", "hellohellohellohellohellohello"] {
        let (text, usage, _, _) = engine.complete_raw(prompt, &generation).unwrap();
        assert_eq!(text, "€!");
        assert_eq!(usage.completion_tokens, 4);
    }
    let truncated = GenerationOptions {
        max_tokens: 2,
        ..generation.clone()
    };
    let (text, usage, _, _) = engine.complete_raw("hello", &truncated).unwrap();
    assert_eq!(
        text, "",
        "a token limit must not emit half a Unicode character"
    );
    assert_eq!(usage.completion_tokens, 2);
    let stopped = GenerationOptions {
        stop_sequences: vec!["€".into()],
        ..generation
    };
    let (text, _, _, _) = engine.complete_raw("hello", &stopped).unwrap();
    assert_eq!(text, "");
    std::fs::remove_dir_all(dir).unwrap();
}
