//! Speech-to-text with Whisper — pure Rust.
//!
//! Wraps candle's Whisper implementation (encoder, decoder, and the mel
//! spectrogram code are all Rust) behind a small engine mirroring
//! [`crate::engine::Engine`]'s layout conventions:
//!
//! ```text
//! whisper-tiny/
//! ├── model.safetensors    ← weights (e.g. from openai/whisper-tiny)
//! ├── config.json          ← model dimensions
//! └── tokenizer.json       ← HuggingFace tokenizer
//! ```
//!
//! The mel filterbank is generated at load time (librosa-compatible Slaney
//! filters — the same ones Whisper trains with), so no auxiliary data files
//! are needed.  Audio arrives either as raw 16 kHz mono PCM or as a WAV file
//! (any sample rate/channel count; mixed down and linearly resampled).
//!
//! Decoding is greedy per 30-second segment, matching candle's reference
//! decode loop: the full token sequence is re-fed each step (Whisper's KV
//! cache only covers cross-attention) and generation stops at
//! `<|endoftext|>` or the model's target-position cap.

use std::io::Cursor;
use std::path::Path;
use std::sync::Mutex;

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as wm, audio, model::Whisper, Config};
use tokenizers::Tokenizer;

use crate::error::{JoshuaError, Result};

/// A transcription result.
#[derive(Debug, Clone)]
pub struct Transcription {
    /// The transcribed (or translated) text.
    pub text: String,
    /// Audio duration in seconds.
    pub duration: f64,
    /// Language token used (e.g. `"en"`), when the model is multilingual.
    pub language: Option<String>,
}

/// Whisper speech-to-text engine.
///
/// `Send + Sync`; the model (which owns KV caches) is guarded by a mutex, so
/// transcriptions serialise — matching the one-model-instance memory profile
/// expected of a sidecar STT model.
pub struct WhisperEngine {
    model: Mutex<Whisper>,
    tokenizer: Tokenizer,
    config: Config,
    mel_filters: Vec<f32>,
    device: Device,
    model_name: String,
    sot_token: u32,
    eot_token: u32,
    transcribe_token: Option<u32>,
    translate_token: Option<u32>,
    no_timestamps_token: Option<u32>,
}

impl WhisperEngine {
    /// Load a Whisper model directory (`model.safetensors` + `config.json`
    /// + `tokenizer.json`).
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json"))
                .map_err(|e| JoshuaError::ModelLoad(format!("whisper config.json: {e}")))?,
        )
        .map_err(|e| JoshuaError::ModelLoad(format!("whisper config.json parse: {e}")))?;

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| JoshuaError::ModelLoad(format!("whisper tokenizer load: {e}")))?;

        let weights = find_safetensors(dir)?;
        let device = Device::Cpu;
        // SAFETY: same contract as the GGUF mmap — weight files are treated
        // as immutable once downloaded.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], wm::DTYPE, &device) }
            .map_err(|e| JoshuaError::ModelLoad(format!("whisper weights: {e}")))?;
        let model = Whisper::load(&vb, config.clone())
            .map_err(|e| JoshuaError::ModelLoad(format!("whisper model init: {e}")))?;

        let mel_filters = mel_filterbank(config.num_mel_bins, wm::N_FFT, wm::SAMPLE_RATE);

        let token = |s: &str| tokenizer.token_to_id(s);
        let sot_token = token(wm::SOT_TOKEN).ok_or_else(|| {
            JoshuaError::ModelLoad(format!("whisper tokenizer lacks {}", wm::SOT_TOKEN))
        })?;
        let eot_token = token(wm::EOT_TOKEN).ok_or_else(|| {
            JoshuaError::ModelLoad(format!("whisper tokenizer lacks {}", wm::EOT_TOKEN))
        })?;

        let model_name = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("whisper")
            .to_string();

        tracing::info!(
            "Whisper model '{}' ready (mels={}, vocab={})",
            model_name,
            config.num_mel_bins,
            config.vocab_size
        );

        Ok(Self {
            model: Mutex::new(model),
            transcribe_token: token(wm::TRANSCRIBE_TOKEN),
            translate_token: token(wm::TRANSLATE_TOKEN),
            no_timestamps_token: token(wm::NO_TIMESTAMPS_TOKEN),
            tokenizer,
            config,
            mel_filters,
            device,
            model_name,
            sot_token,
            eot_token,
        })
    }

    /// The model directory name, used as the API model identifier.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Transcribe a WAV file (any sample rate / channel count).
    pub fn transcribe_wav(
        &self,
        wav_bytes: &[u8],
        language: Option<&str>,
        translate: bool,
    ) -> Result<Transcription> {
        let samples = wav_to_pcm16k(wav_bytes)?;
        self.transcribe(&samples, language, translate)
    }

    /// Transcribe raw 16 kHz mono PCM samples in `[-1, 1]`.
    pub fn transcribe(
        &self,
        pcm: &[f32],
        language: Option<&str>,
        translate: bool,
    ) -> Result<Transcription> {
        if pcm.is_empty() {
            return Err(JoshuaError::InvalidRequest("empty audio".to_string()));
        }
        let duration = pcm.len() as f64 / wm::SAMPLE_RATE as f64;

        let mel = audio::pcm_to_mel(&self.config, pcm, &self.mel_filters);
        let n_frames = mel.len() / self.config.num_mel_bins;
        let mel = Tensor::from_vec(mel, (1, self.config.num_mel_bins, n_frames), &self.device)
            .map_err(|e| JoshuaError::Inference(e.to_string()))?;

        // Resolve the language token for multilingual checkpoints.
        let language = language.map(str::to_lowercase);
        let language_token = match &language {
            Some(lang) => Some(self.language_token(lang)?),
            // English-only checkpoints have no language tokens at all.
            None => self.tokenizer.token_to_id("<|en|>"),
        };

        let mut model = self
            .model
            .lock()
            .map_err(|e| JoshuaError::Inference(format!("whisper model lock poisoned: {e}")))?;

        let mut text = String::new();
        let mut seek = 0;
        while seek < n_frames {
            let segment_len = usize::min(n_frames - seek, wm::N_FRAMES);
            let segment = mel
                .narrow(2, seek, segment_len)
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;
            let piece = self.decode_segment(&mut model, &segment, language_token, translate)?;
            text.push_str(&piece);
            seek += segment_len;
        }

        Ok(Transcription {
            text: text.trim().to_string(),
            duration,
            language: language.or_else(|| language_token.map(|_| "en".to_string())),
        })
    }

    /// Greedy-decode one ≤30 s mel segment.
    fn decode_segment(
        &self,
        model: &mut Whisper,
        mel: &Tensor,
        language_token: Option<u32>,
        translate: bool,
    ) -> Result<String> {
        let features = model
            .encoder
            .forward(mel, true)
            .map_err(|e| JoshuaError::Inference(e.to_string()))?;

        let mut tokens: Vec<u32> = vec![self.sot_token];
        if let Some(lang) = language_token {
            tokens.push(lang);
            let task = if translate {
                self.translate_token
            } else {
                self.transcribe_token
            };
            if let Some(task) = task {
                tokens.push(task);
            }
        }
        if let Some(no_ts) = self.no_timestamps_token {
            tokens.push(no_ts);
        }
        let n_prompt = tokens.len();

        let sample_len = self.config.max_target_positions / 2;
        for i in 0..sample_len {
            let input = Tensor::new(tokens.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;
            // The KV cache only covers cross-attention, so the full sequence
            // is re-fed each step, flushing on the first.
            let ys = model
                .decoder
                .forward(&input, &features, i == 0)
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;
            let logits = model
                .decoder
                .final_linear(
                    &ys.i((..1, tokens.len() - 1..))
                        .map_err(|e| JoshuaError::Inference(e.to_string()))?,
                )
                .and_then(|l| l.i(0)?.i(0))
                .and_then(|l| l.to_vec1::<f32>())
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;

            let next = self.argmax_suppressed(&logits);
            if next == self.eot_token || tokens.len() >= self.config.max_target_positions {
                break;
            }
            tokens.push(next);
        }

        // Skip the prompt tokens and strip any specials from the output.
        self.tokenizer
            .decode(&tokens[n_prompt..], true)
            .map_err(|e| JoshuaError::Inference(e.to_string()))
    }

    /// Greedy argmax honouring the model's suppress list and never emitting
    /// the start-of-transcript token.
    fn argmax_suppressed(&self, logits: &[f32]) -> u32 {
        let mut best = self.eot_token;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            let id = i as u32;
            if id == self.sot_token || self.config.suppress_tokens.contains(&id) {
                continue;
            }
            if v > best_val {
                best_val = v;
                best = id;
            }
        }
        best
    }

    /// Resolve `<|xx|>` for a two-letter language code.
    fn language_token(&self, lang: &str) -> Result<u32> {
        self.tokenizer
            .token_to_id(&format!("<|{lang}|>"))
            .ok_or_else(|| {
                JoshuaError::InvalidRequest(format!(
                    "language '{lang}' is not supported by this whisper model"
                ))
            })
    }
}

// ─── Audio ingest ───────────────────────────────────────────────────────────

/// Decode a WAV file to 16 kHz mono f32 PCM.
///
/// Channels are averaged; other sample rates are linearly resampled — fine
/// for speech (Whisper's mel front-end low-passes well below Nyquist).
pub fn wav_to_pcm16k(bytes: &[u8]) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|e| JoshuaError::InvalidRequest(format!("invalid WAV: {e}")))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| JoshuaError::InvalidRequest(format!("WAV read: {e}")))?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| JoshuaError::InvalidRequest(format!("WAV read: {e}")))?
        }
    };

    // Mix down to mono.
    let mono: Vec<f32> = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();

    // Resample to 16 kHz.
    let src_rate = spec.sample_rate as usize;
    if src_rate == wm::SAMPLE_RATE {
        return Ok(mono);
    }
    let ratio = src_rate as f64 / wm::SAMPLE_RATE as f64;
    let out_len = (mono.len() as f64 / ratio) as usize;
    let resampled = (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;
            let a = mono[idx.min(mono.len() - 1)];
            let b = mono[(idx + 1).min(mono.len() - 1)];
            a * (1.0 - frac) + b * frac
        })
        .collect();
    Ok(resampled)
}

// ─── Mel filterbank ─────────────────────────────────────────────────────────

/// Librosa-compatible Slaney mel filterbank (`htk=False, norm="slaney"`) —
/// the filters Whisper was trained with.  Row-major `n_mels × (n_fft/2+1)`.
fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let f_max = sample_rate as f64 / 2.0;

    // Slaney mel scale: linear below 1 kHz, logarithmic above.
    let hz_to_mel = |hz: f64| {
        if hz < 1000.0 {
            hz * 3.0 / 200.0
        } else {
            15.0 + (hz / 1000.0).ln() * 27.0 / 6.4f64.ln()
        }
    };
    let mel_to_hz = |mel: f64| {
        if mel < 15.0 {
            mel * 200.0 / 3.0
        } else {
            1000.0 * ((mel - 15.0) * 6.4f64.ln() / 27.0).exp()
        }
    };

    let mel_max = hz_to_mel(f_max);
    let band_hz: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_max * i as f64 / (n_mels + 1) as f64))
        .collect();
    let fft_hz: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    let mut filters = vec![0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let (lower, center, upper) = (band_hz[m], band_hz[m + 1], band_hz[m + 2]);
        // Slaney area normalisation.
        let enorm = 2.0 / (upper - lower);
        for (k, &hz) in fft_hz.iter().enumerate() {
            let rising = (hz - lower) / (center - lower);
            let falling = (upper - hz) / (upper - center);
            let weight = rising.min(falling).max(0.0);
            filters[m * n_freqs + k] = (weight * enorm) as f32;
        }
    }
    filters
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Find the first `.safetensors` file in `dir`.
fn find_safetensors(dir: &Path) -> Result<std::path::PathBuf> {
    let preferred = dir.join("model.safetensors");
    if preferred.exists() {
        return Ok(preferred);
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            return Ok(path);
        }
    }
    Err(JoshuaError::ModelLoad(format!(
        "no .safetensors weights found in {dir:?}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_filterbank_shape_and_energy() {
        let filters = mel_filterbank(80, 400, 16000);
        assert_eq!(filters.len(), 80 * 201);
        // Every band must have some energy, and no negative weights.
        for m in 0..80 {
            let row = &filters[m * 201..(m + 1) * 201];
            assert!(row.iter().all(|&w| w >= 0.0));
            assert!(row.iter().sum::<f32>() > 0.0, "band {m} is empty");
        }
    }

    #[test]
    fn wav_roundtrip_and_resample() {
        // 1 kHz sine at 8 kHz stereo — should mix down and upsample to 16 kHz.
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 8000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
            for i in 0..8000 {
                let v = (i as f32 / 8000.0 * 1000.0 * std::f32::consts::TAU).sin();
                let s = (v * i16::MAX as f32 * 0.5) as i16;
                writer.write_sample(s).unwrap();
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }
        let pcm = wav_to_pcm16k(&buf.into_inner()).unwrap();
        assert_eq!(pcm.len(), 16000, "1 s of audio at 16 kHz");
        assert!(pcm.iter().any(|&v| v.abs() > 0.1), "signal survived");
    }
}
