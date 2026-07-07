//! Joshua CLI — start the inference server or run a one-shot completion.
//!
//! # Usage
//!
//! ```text
//! # Start the OpenAI-compatible server
//! joshua serve --model ./weights/gemma-3-270m-it-q4_k_m.gguf
//!
//! # One-shot chat completion
//! joshua run --model ./weights/gemma-3-270m-it-q4_k_m.gguf "What is Rust?"
//!
//! # Embed text
//! joshua embed --model ./weights/nomic-embed.gguf "Hello world"
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand, ValueEnum};
use tracing_subscriber::EnvFilter;

use joshua::{
    engine::Engine, server, types::GenerationOptions, ChatMessage, EngineOptions, HugePages,
    PageSize,
};

/// Physical-memory backing for the model mapping (CLI form of [`HugePages`]).
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum HugePagesArg {
    /// Normal pages; file-backed mmap shared via the page cache (default).
    #[default]
    Off,
    /// Transparent huge pages (MADV_HUGEPAGE) on the file-backed mmap.
    Transparent,
    /// Explicit huge pages at the system default size (anonymous copy).
    Huge,
    /// Explicit 2 MiB "large" pages (anonymous copy).
    #[value(name = "2mb")]
    TwoMb,
    /// Explicit 1 GiB "huge" pages (anonymous copy).
    #[value(name = "1gb")]
    OneGb,
}

impl From<HugePagesArg> for HugePages {
    fn from(arg: HugePagesArg) -> Self {
        match arg {
            HugePagesArg::Off => HugePages::Off,
            HugePagesArg::Transparent => HugePages::Transparent,
            HugePagesArg::Huge => HugePages::Explicit(PageSize::Default),
            HugePagesArg::TwoMb => HugePages::Explicit(PageSize::TwoMiB),
            HugePagesArg::OneGb => HugePages::Explicit(PageSize::OneGiB),
        }
    }
}

/// An mmap-based LLM inference engine — a Rust clone of Cactus.
#[derive(Parser)]
#[command(name = "joshua", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the OpenAI-compatible HTTP API server.
    Serve {
        /// Path to the GGUF model file.
        #[arg(short, long, env = "JOSHUA_MODEL_PATH")]
        model: PathBuf,
        /// Address to listen on.  Defaults to localhost; bind `0.0.0.0` only
        /// after enabling `--api-key` and/or TLS, since the API is otherwise
        /// unauthenticated.
        #[arg(short, long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Context window size in tokens.
        #[arg(long, default_value_t = 4096)]
        n_ctx: u32,
        /// Physical-memory backing for the model: off, transparent, huge,
        /// 2mb, or 1gb (Linux only for the huge-page modes).
        #[arg(long, value_enum, default_value_t = HugePagesArg::Off)]
        huge_pages: HugePagesArg,
        /// NPU vendor plugin (a cdylib exporting the joshua_npu_* ABI).
        #[arg(long, env = "JOSHUA_NPU_PLUGIN")]
        npu_plugin: Option<PathBuf>,
        /// Load the NPU plugin in-process instead of the isolated shim
        /// (lower overhead; a plugin crash takes the server down).
        #[arg(long, default_value_t = false)]
        npu_in_process: bool,
        /// Whisper model directory to mount at /v1/audio/transcriptions.
        #[arg(long, env = "JOSHUA_WHISPER_MODEL")]
        whisper_model: Option<PathBuf>,
        /// Maximum concurrent generations/embeddings; excess requests get a
        /// 503.  Bounds peak memory from concurrent model instances.
        /// Defaults to the machine's parallelism.
        #[arg(long, env = "JOSHUA_MAX_CONCURRENCY")]
        max_concurrency: Option<usize>,
        /// Hard ceiling on tokens generated per request, capping the
        /// client-supplied max_tokens.
        #[arg(long, env = "JOSHUA_MAX_OUTPUT_TOKENS")]
        max_output_tokens: Option<u32>,
        /// Require this API key on /v1 routes (Authorization: Bearer <key>).
        #[arg(long, env = "JOSHUA_API_KEY")]
        api_key: Option<String>,
        /// Serve HTTPS with this PEM certificate chain (needs --tls-key and
        /// a build with the `tls` cargo feature).
        #[arg(long, env = "JOSHUA_TLS_CERT", requires = "tls_key")]
        tls_cert: Option<PathBuf>,
        /// PEM private key for --tls-cert.
        #[arg(long, env = "JOSHUA_TLS_KEY", requires = "tls_cert")]
        tls_key: Option<PathBuf>,
    },
    /// Run a single chat completion and print the response.
    Run {
        /// Path to the GGUF model file.
        #[arg(short, long, env = "JOSHUA_MODEL_PATH")]
        model: PathBuf,
        /// The user message to send to the model.
        prompt: String,
        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 256)]
        max_tokens: u32,
        /// Sampling temperature (0 = greedy).
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        /// Context window size in tokens.
        #[arg(long, default_value_t = 4096)]
        n_ctx: u32,
        /// Physical-memory backing for the model: off, transparent, huge,
        /// 2mb, or 1gb (Linux only for the huge-page modes).
        #[arg(long, value_enum, default_value_t = HugePagesArg::Off)]
        huge_pages: HugePagesArg,
        /// NPU vendor plugin (a cdylib exporting the joshua_npu_* ABI).
        #[arg(long, env = "JOSHUA_NPU_PLUGIN")]
        npu_plugin: Option<PathBuf>,
        /// Load the NPU plugin in-process instead of the isolated shim
        /// (lower overhead; a plugin crash takes the server down).
        #[arg(long, default_value_t = false)]
        npu_in_process: bool,
    },
    /// Transcribe a WAV file with a Whisper model.
    Transcribe {
        /// Whisper model directory (model.safetensors + config.json + tokenizer.json).
        #[arg(short, long, env = "JOSHUA_WHISPER_MODEL")]
        model: PathBuf,
        /// Path to the WAV file.
        audio: PathBuf,
        /// Spoken language as a two-letter code (default: auto/en).
        #[arg(long)]
        language: Option<String>,
        /// Translate to English instead of transcribing.
        #[arg(long, default_value_t = false)]
        translate: bool,
    },
    /// Embed one or more texts and print their vector representations.
    Embed {
        /// Path to the GGUF embedding model file.
        #[arg(short, long, env = "JOSHUA_MODEL_PATH")]
        model: PathBuf,
        /// Texts to embed.
        texts: Vec<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialise tracing (respects RUST_LOG env var).
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            model,
            addr,
            n_ctx,
            huge_pages,
            npu_plugin,
            npu_in_process,
            whisper_model,
            max_concurrency,
            max_output_tokens,
            api_key,
            tls_cert,
            tls_key,
        } => {
            // Fail fast, before the (potentially slow) model load, when TLS
            // flags are passed to a build compiled without TLS support.
            #[cfg(not(feature = "tls"))]
            if tls_cert.is_some() || tls_key.is_some() {
                anyhow::bail!(
                    "this build has no TLS support — rebuild with `cargo build --features tls`"
                );
            }

            let opts = EngineOptions::with_n_ctx(n_ctx).huge_pages(huge_pages.into());
            let mut engine = Engine::with_options(&model, opts)?;
            if let Some(plugin) = npu_plugin {
                engine = engine.with_npu_backend(npu_backend(&plugin, npu_in_process)?);
            }
            if let Some(max) = max_concurrency {
                engine = engine.with_max_concurrency(max);
            }
            if let Some(max) = max_output_tokens {
                engine = engine.with_max_output_tokens(max);
            }
            let whisper = whisper_model
                .map(joshua::whisper::WhisperEngine::new)
                .transpose()?
                .map(Arc::new);
            let state = Arc::new(server::ServerState {
                engine: Arc::new(engine),
                whisper,
                api_key,
            });
            match (tls_cert, tls_key) {
                (Some(cert), Some(key)) => {
                    #[cfg(feature = "tls")]
                    server::serve_with_state_tls(state, &addr, &cert, &key).await?;
                    #[cfg(not(feature = "tls"))]
                    {
                        let _ = (cert, key, state);
                        unreachable!("rejected above before the model load");
                    }
                }
                // clap's `requires` rejects one flag without the other.
                _ => server::serve_with_state(state, &addr).await?,
            }
        }

        Commands::Transcribe {
            model,
            audio,
            language,
            translate,
        } => {
            let whisper = joshua::whisper::WhisperEngine::new(&model)?;
            let bytes = std::fs::read(&audio)?;
            let result = whisper.transcribe_wav(&bytes, language.as_deref(), translate)?;
            println!("{}", result.text);
            eprintln!("\n[duration: {:.1}s]", result.duration);
        }

        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            n_ctx,
            huge_pages,
            npu_plugin,
            npu_in_process,
        } => {
            let opts = EngineOptions::with_n_ctx(n_ctx).huge_pages(huge_pages.into());
            let mut engine = Engine::with_options(&model, opts)?;
            if let Some(plugin) = npu_plugin {
                engine = engine.with_npu_backend(npu_backend(&plugin, npu_in_process)?);
            }
            let messages = vec![ChatMessage::text("user".to_string(), prompt)];
            let options = GenerationOptions {
                max_tokens,
                temperature,
                ..GenerationOptions::default()
            };
            let (text, usage, prefill_tps, decode_tps) = engine.complete(&messages, &options)?;
            println!("{text}");
            eprintln!(
                "\n[tokens: prompt={} completion={} | prefill={:.1}t/s decode={:.1}t/s]",
                usage.prompt_tokens, usage.completion_tokens, prefill_tps, decode_tps
            );
        }

        Commands::Embed { model, texts } => {
            if texts.is_empty() {
                anyhow::bail!("At least one text is required");
            }
            let engine = Engine::new(&model)?;
            let embeddings = engine.embed(&texts)?;
            for (i, emb) in embeddings.iter().enumerate() {
                let preview: Vec<String> = emb.iter().take(8).map(|v| format!("{v:.4}")).collect();
                println!(
                    "Text {i}: dim={} [{}, ...]",
                    emb.len(),
                    preview.join(", ")
                );
            }
        }
    }

    Ok(())
}

/// Build the NPU backend for a vendor plugin: process-isolated shim by
/// default, in-process when explicitly requested.
fn npu_backend(
    plugin: &std::path::Path,
    in_process: bool,
) -> anyhow::Result<Arc<dyn joshua::npu::NpuBackend>> {
    if in_process {
        Ok(Arc::new(
            joshua::npu::InProcessBackend::load(plugin).map_err(anyhow::Error::msg)?,
        ))
    } else {
        Ok(Arc::new(
            joshua::npu::ShimBackend::locate(plugin).map_err(anyhow::Error::msg)?,
        ))
    }
}
