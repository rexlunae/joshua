//! End-to-end tests for Whisper speech-to-text.
//!
//! A tiny but structurally valid Whisper checkpoint is synthesised on the
//! fly: `Whisper::load` is driven by a `VarMap`-backed `VarBuilder`, which
//! creates every parameter the model asks for (with candle's standard
//! initialisers) — the map is then saved as `model.safetensors`, guaranteeing
//! the exact tensor names and shapes the loader expects.  Together with a
//! generated WAV file this exercises the full pipeline with no downloads:
//! WAV → resample → mel spectrogram → encoder → greedy decode → detokenise.

mod common;

use std::io::Cursor;
use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::whisper::{model::Whisper, Config};
use joshua::whisper::{wav_to_pcm16k, WhisperEngine};

/// Whisper-style tokenizer: a few text tokens plus the control tokens the
/// decoder prompt needs.
const WHISPER_TOKENIZER_JSON: &str = r#"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [
        {"id": 10, "content": "<|endoftext|>", "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true},
        {"id": 11, "content": "<|startoftranscript|>", "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true},
        {"id": 12, "content": "<|en|>", "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true},
        {"id": 13, "content": "<|transcribe|>", "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true},
        {"id": 14, "content": "<|translate|>", "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true},
        {"id": 15, "content": "<|notimestamps|>", "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true}
    ],
    "normalizer": null,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": null,
    "decoder": null,
    "model": {
        "type": "WordLevel",
        "vocab": {"the": 0, "quick": 1, "brown": 2, "fox": 3, "jumps": 4,
                  "over": 5, "lazy": 6, "dog": 7, "hello": 8, "world": 9,
                  "<|endoftext|>": 10, "<|startoftranscript|>": 11, "<|en|>": 12,
                  "<|transcribe|>": 13, "<|translate|>": 14, "<|notimestamps|>": 15},
        "unk_token": "the"
    }
}"#;

/// Synthesise a tiny whisper model directory.
fn write_tiny_whisper(dir: &PathBuf) {
    let config = Config {
        num_mel_bins: 80,
        max_source_positions: 1500,
        d_model: 8,
        encoder_attention_heads: 2,
        encoder_layers: 1,
        vocab_size: 16,
        max_target_positions: 24,
        decoder_attention_heads: 2,
        decoder_layers: 1,
        suppress_tokens: vec![],
    };

    // Drive the loader once against a VarMap: every parameter it requests is
    // created (candle's default initialisers), then saved as safetensors.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let _model = Whisper::load(&vb, config.clone()).expect("synthetic whisper init");
    varmap
        .save(dir.join("model.safetensors"))
        .expect("save synthetic weights");

    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "num_mel_bins": config.num_mel_bins,
            "max_source_positions": config.max_source_positions,
            "d_model": config.d_model,
            "encoder_attention_heads": config.encoder_attention_heads,
            "encoder_layers": config.encoder_layers,
            "vocab_size": config.vocab_size,
            "max_target_positions": config.max_target_positions,
            "decoder_attention_heads": config.decoder_attention_heads,
            "decoder_layers": config.decoder_layers,
            "suppress_tokens": [],
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), WHISPER_TOKENIZER_JSON).unwrap();
}

/// One second of a 440 Hz tone as an in-memory 16 kHz mono WAV.
fn tone_wav() -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
        for i in 0..16000 {
            let v = (i as f32 / 16000.0 * 440.0 * std::f32::consts::TAU).sin();
            writer.write_sample((v * i16::MAX as f32 * 0.5) as i16).unwrap();
        }
        writer.finalize().unwrap();
    }
    buf.into_inner()
}

#[test]
fn whisper_transcribes_wav_end_to_end() {
    let dir = common::model_dir("whisper-tiny-synth");
    write_tiny_whisper(&dir);

    let engine = WhisperEngine::new(&dir).expect("whisper engine should load");
    assert_eq!(engine.model_name(), dir.file_name().unwrap().to_str().unwrap());

    let wav = tone_wav();
    let result = engine
        .transcribe_wav(&wav, Some("en"), false)
        .expect("transcription should run");

    // Random weights produce arbitrary text — the pipeline mechanics are
    // what is under test: duration bookkeeping, decode-loop termination,
    // and special tokens never leaking into the output.
    assert!((result.duration - 1.0).abs() < 0.01, "1 s of audio");
    assert_eq!(result.language.as_deref(), Some("en"));
    assert!(
        !result.text.contains("<|"),
        "special tokens must not leak: {:?}",
        result.text
    );

    // A second call must work (mutex + KV flush per segment).
    engine
        .transcribe_wav(&wav, None, false)
        .expect("second transcription");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn whisper_rejects_unknown_language_and_empty_audio() {
    let dir = common::model_dir("whisper-tiny-errs");
    write_tiny_whisper(&dir);
    let engine = WhisperEngine::new(&dir).expect("whisper engine should load");

    let err = engine
        .transcribe_wav(&tone_wav(), Some("xx"), false)
        .map(|_| ())
        .unwrap_err();
    assert!(err.to_string().contains("not supported"), "got: {err}");

    let err = engine.transcribe(&[], None, false).map(|_| ()).unwrap_err();
    assert!(err.to_string().contains("empty audio"), "got: {err}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn wav_ingest_handles_stereo_and_other_rates() {
    // 0.5 s of 24 kHz stereo f32 WAV.
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 24000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
        for i in 0..12000 {
            let v = (i as f32 / 24000.0 * 220.0 * std::f32::consts::TAU).sin() * 0.3;
            writer.write_sample(v).unwrap();
            writer.write_sample(-v).unwrap(); // opposite phase: mixes to ~0
        }
        writer.finalize().unwrap();
    }
    let pcm = wav_to_pcm16k(&buf.into_inner()).unwrap();
    assert_eq!(pcm.len(), 8000, "0.5 s at 16 kHz");
    // Opposite-phase stereo cancels in the mono mix.
    assert!(pcm.iter().all(|v| v.abs() < 1e-3));
}
