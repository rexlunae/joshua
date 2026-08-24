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

use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Parser, Subcommand, ValueEnum};
use tracing_subscriber::EnvFilter;

use joshua::{
    engine::Engine, server, types::GenerationOptions, ChatMessage, ComputeBackend, EngineOptions,
    HugePages, MlockMode, MmapMode, PageSize,
};

/// Values for `--device` (CLI form of [`ComputeBackend`]).
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum DeviceArg {
    /// Best backend this build supports: CUDA, then Metal, then CPU.
    #[default]
    Auto,
    /// CPU inference.
    Cpu,
    /// Apple Metal (build with `--features metal`).
    Metal,
    /// NVIDIA CUDA (build with `--features cuda`).
    Cuda,
}

impl From<DeviceArg> for ComputeBackend {
    fn from(arg: DeviceArg) -> Self {
        match arg {
            DeviceArg::Auto => ComputeBackend::Auto,
            DeviceArg::Cpu => ComputeBackend::Cpu,
            DeviceArg::Metal => ComputeBackend::Metal,
            DeviceArg::Cuda => ComputeBackend::Cuda,
        }
    }
}

/// Physical-memory backing for the model mapping (CLI form of [`HugePages`]).
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
enum HugePagesArg {
    /// Normal pages; file-backed mmap shared via the page cache (the default
    /// off Linux — `transparent` is the Linux default).
    #[default]
    Off,
    /// Transparent huge pages (MADV_HUGEPAGE) on the file-backed mmap; the
    /// default on Linux, where it is best-effort and silently keeps normal
    /// pages when the kernel cannot honour them.
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

/// The CLI default huge-page strategy: `transparent` on Linux (best-effort
/// 2 MiB pages for the model mapping at no cost when unsupported), `off`
/// elsewhere — macOS has no file-backed huge-page API to ask for.
fn default_huge_pages() -> HugePagesArg {
    if cfg!(target_os = "linux") {
        HugePagesArg::Transparent
    } else {
        HugePagesArg::Off
    }
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

/// Values for `--mlock-hot-weights` (CLI form of [`MlockMode`]).
#[derive(Debug, Clone, Copy, ValueEnum)]
enum MlockArg {
    /// Lock the hot set; degrade to advisory pinning (with a warning) when
    /// the memlock limit is too low.
    #[value(alias = "true")]
    On,
    /// Lock the hot set; fail the load when the memlock limit is too low.
    #[value(alias = "strict")]
    Required,
    /// Do not lock (explicitly disable, overriding an env var).
    #[value(alias = "false")]
    Off,
}

impl From<MlockArg> for MlockMode {
    fn from(arg: MlockArg) -> Self {
        match arg {
            MlockArg::On => MlockMode::On,
            MlockArg::Required => MlockMode::Required,
            MlockArg::Off => MlockMode::Off,
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
        /// Compute backend: auto, cpu, metal, or cuda.  An explicit GPU
        /// request fails the load when the backend is not built in or is
        /// unavailable at runtime.
        #[arg(long, env = "JOSHUA_DEVICE", value_enum, default_value_t = DeviceArg::Auto)]
        device: DeviceArg,
        /// Physical-memory backing for the model: off, transparent, huge,
        /// 2mb, or 1gb (Linux only for the huge-page modes).  Defaults to
        /// transparent on Linux (best-effort 2 MiB pages), off elsewhere.
        #[arg(long, value_enum, default_value_t = default_huge_pages())]
        huge_pages: HugePagesArg,
        /// Require the model to be memory-mappable: a compressed model file
        /// (a gzip/zstd/... stream, or one the filesystem stores compressed)
        /// becomes a load error instead of a warning.
        #[arg(long, default_value_t = false)]
        mmap: bool,
        /// Optimise the mapping for a model far larger than RAM: no readahead,
        /// so pages fault in only as weights are genuinely touched.
        #[arg(long, env = "JOSHUA_LAZY_WEIGHTS", default_value_t = false)]
        lazy_weights: bool,
        /// Prefetch the whole model file into the OS page cache at load
        /// (MADV_WILLNEED), so a model that fits in RAM is fully resident
        /// before the first request and inference never re-reads weights from
        /// disk.  Defaults to on when the model file fits in RAM; pass
        /// `--prefetch-model=false` to force it off.
        #[arg(long, env = "JOSHUA_PREFETCH_MODEL", num_args = 0..=1, default_missing_value = "true")]
        prefetch_model: Option<bool>,
        /// Prefetch the always-touched weights (embeddings, norms, attention,
        /// routers, shared experts, output) at load and advise random access
        /// on routed experts, so the per-token working set stays resident.
        /// Defaults to on when the model file is larger than RAM; pass
        /// `--pin-hot-weights=false` to force it off.
        #[arg(long, env = "JOSHUA_PIN_HOT_WEIGHTS", num_args = 0..=1, default_missing_value = "true")]
        pin_hot_weights: Option<bool>,
        /// Lock the always-touched weights into RAM (mlock).  Needs the
        /// process memlock limit to cover the hot set: `LimitMEMLOCK=infinity`
        /// (systemd), `ulimit -l unlimited`, or /etc/security/limits.conf.
        /// On a systemd *user* session the hard limit is inherited from the
        /// login session, so a unit's LimitMEMLOCK alone is silently capped —
        /// apply `sudo prlimit --pid <user manager pid> --memlock=-1:-1` for a
        /// live fix, or re-login.  `=required` fails the load instead of
        /// degrading when the limit is too low; the limit is checked before
        /// any mlock attempt, with one warning naming limit vs required size.
        #[arg(long, env = "JOSHUA_MLOCK_HOT_WEIGHTS", num_args = 0..=1, default_missing_value = "on", value_enum)]
        mlock_hot_weights: Option<MlockArg>,
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
        /// Compute backend: auto, cpu, metal, or cuda.  An explicit GPU
        /// request fails the load when the backend is not built in or is
        /// unavailable at runtime.
        #[arg(long, env = "JOSHUA_DEVICE", value_enum, default_value_t = DeviceArg::Auto)]
        device: DeviceArg,
        /// Physical-memory backing for the model: off, transparent, huge,
        /// 2mb, or 1gb (Linux only for the huge-page modes).  Defaults to
        /// transparent on Linux (best-effort 2 MiB pages), off elsewhere.
        #[arg(long, value_enum, default_value_t = default_huge_pages())]
        huge_pages: HugePagesArg,
        /// Require the model to be memory-mappable: a compressed model file
        /// (a gzip/zstd/... stream, or one the filesystem stores compressed)
        /// becomes a load error instead of a warning.
        #[arg(long, default_value_t = false)]
        mmap: bool,
        /// Optimise the mapping for a model far larger than RAM: no readahead,
        /// so pages fault in only as weights are genuinely touched.
        #[arg(long, env = "JOSHUA_LAZY_WEIGHTS", default_value_t = false)]
        lazy_weights: bool,
        /// Prefetch the whole model file into the OS page cache at load
        /// (MADV_WILLNEED), so a model that fits in RAM is fully resident
        /// before the first request.  Defaults to on when the model file fits
        /// in RAM; pass `--prefetch-model=false` to force it off.
        #[arg(long, env = "JOSHUA_PREFETCH_MODEL", num_args = 0..=1, default_missing_value = "true")]
        prefetch_model: Option<bool>,
        /// Prefetch the always-touched weights at load and advise random
        /// access on routed experts.  Defaults to on when the model file is
        /// larger than RAM; pass `--pin-hot-weights=false` to force it off.
        #[arg(long, env = "JOSHUA_PIN_HOT_WEIGHTS", num_args = 0..=1, default_missing_value = "true")]
        pin_hot_weights: Option<bool>,
        /// Lock the always-touched weights into RAM (mlock).  Needs the
        /// process memlock limit to cover the hot set: `LimitMEMLOCK=infinity`
        /// (systemd), `ulimit -l unlimited`, or /etc/security/limits.conf.
        /// On a systemd *user* session the hard limit is inherited from the
        /// login session, so a unit's LimitMEMLOCK alone is silently capped —
        /// apply `sudo prlimit --pid <user manager pid> --memlock=-1:-1` for a
        /// live fix, or re-login.  `=required` fails the load instead of
        /// degrading when the limit is too low; the limit is checked before
        /// any mlock attempt, with one warning naming limit vs required size.
        #[arg(long, env = "JOSHUA_MLOCK_HOT_WEIGHTS", num_args = 0..=1, default_missing_value = "on", value_enum)]
        mlock_hot_weights: Option<MlockArg>,
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
        /// Compute backend: auto, cpu, metal, or cuda.
        #[arg(long, env = "JOSHUA_DEVICE", value_enum, default_value_t = DeviceArg::Auto)]
        device: DeviceArg,
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
            device,
            huge_pages,
            mmap,
            lazy_weights,
            prefetch_model,
            pin_hot_weights,
            mlock_hot_weights,
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

            let (auto_pin, auto_prefetch) = cache_plan(&model);
            let pin_hot = pin_hot_weights.unwrap_or(auto_pin);
            let prefetch = prefetch_model.unwrap_or(auto_prefetch);
            let opts = EngineOptions::with_n_ctx(n_ctx)
                .backend(device.into())
                .huge_pages(huge_pages.into())
                .mmap(mmap_mode(mmap))
                .lazy_weights(lazy_weights)
                .prefetch_whole_model(prefetch)
                .pin_hot_weights(pin_hot)
                .mlock_hot_weights(
                    mlock_hot_weights
                        .map(MlockMode::from)
                        .unwrap_or(MlockMode::Off),
                );
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
            device,
            huge_pages,
            mmap,
            lazy_weights,
            prefetch_model,
            pin_hot_weights,
            mlock_hot_weights,
            npu_plugin,
            npu_in_process,
        } => {
            let (auto_pin, auto_prefetch) = cache_plan(&model);
            let pin_hot = pin_hot_weights.unwrap_or(auto_pin);
            let prefetch = prefetch_model.unwrap_or(auto_prefetch);
            let opts = EngineOptions::with_n_ctx(n_ctx)
                .backend(device.into())
                .huge_pages(huge_pages.into())
                .mmap(mmap_mode(mmap))
                .lazy_weights(lazy_weights)
                .prefetch_whole_model(prefetch)
                .pin_hot_weights(pin_hot)
                .mlock_hot_weights(
                    mlock_hot_weights
                        .map(MlockMode::from)
                        .unwrap_or(MlockMode::Off),
                );
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

        Commands::Embed {
            model,
            device,
            texts,
        } => {
            if texts.is_empty() {
                anyhow::bail!("At least one text is required");
            }
            let opts = EngineOptions::default().backend(device.into());
            let engine = Engine::with_options(&model, opts)?;
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

/// Translate the `--mmap` flag: passing it makes memory mapping an explicit
/// request, so a model file that cannot be mapped usefully fails the load
/// instead of merely warning.
fn mmap_mode(explicit: bool) -> MmapMode {
    if explicit {
        MmapMode::Required
    } else {
        MmapMode::Auto
    }
}

/// Total physical RAM in bytes; `None` when unreadable.
///
/// Linux reads `/proc/meminfo`; macOS asks `sysctl(3)` for `hw.memsize`
/// (`sysctlbyname` would need a NUL-terminated string, so the MIB route is
/// used).  Other platforms have no detection and return `None`, which simply
/// leaves the page-cache options at their manual defaults.
#[cfg(target_os = "linux")]
fn total_ram_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// macOS: `hw.memsize` via `sysctl(3)` — physical RAM, unlike `hw.physmem`'s
/// legacy 32-bit truncation.
#[cfg(target_os = "macos")]
fn total_ram_bytes() -> Option<u64> {
    let mut mib: [libc::c_int; 2] = [libc::CTL_HW, libc::HW_MEMSIZE];
    let mut size: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: `mib` names the request, `size`/`len` are a valid out-buffer,
    // and no new-value pointer is passed.
    let rc = unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            2,
            (&mut size as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && size > 0).then_some(size)
}

/// Neither Linux nor macOS: no RAM detection.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn total_ram_bytes() -> Option<u64> {
    None
}

/// Size in bytes of the model file (a directory argument is resolved to the
/// `.gguf` inside it).
fn model_file_size(model: &Path) -> Option<u64> {
    if model.is_file() {
        return model.metadata().ok().map(|m| m.len());
    }
    joshua::find_gguf_in_dir(model)
        .ok()
        .and_then(|p| p.metadata().ok())
        .map(|m| m.len())
}

/// Pure decision behind [`cache_plan`]: given model file size and total RAM,
/// return `(pin_hot_weights, prefetch_whole_model)`.
///
/// A model that fits in RAM is prefetched whole into the page cache at load
/// (the entire file is resident before the first request); a model larger
/// than RAM instead gets the dense/expert split, where only the
/// always-touched ranges are prefetched and the routed experts page in on
/// demand.  When either size is unknown both stay off (manual control).
fn cache_plan_for(file: Option<u64>, ram: Option<u64>) -> (bool, bool) {
    match (file, ram) {
        (Some(file), Some(ram)) if file > ram => (true, false),
        (Some(_), Some(_)) => (false, true),
        _ => (false, false),
    }
}

/// Auto decisions for the page-cache options, from the model size vs RAM:
/// `(pin_hot_weights, prefetch_whole_model)`.  Logged so operators see which
/// way the auto-resolution went and why.
fn cache_plan(model: &Path) -> (bool, bool) {
    let file = model_file_size(model);
    let ram = total_ram_bytes();
    match cache_plan_for(file, ram) {
        plan @ (true, false) => {
            tracing::info!(
                "page-cache auto: model {:.1} GiB vs RAM {:.1} GiB — model exceeds RAM, \
                 pinning hot weights only",
                file.unwrap_or(0) as f64 / 2f64.powi(30),
                ram.unwrap_or(0) as f64 / 2f64.powi(30),
            );
            plan
        }
        plan @ (false, true) => {
            tracing::info!(
                "page-cache auto: model {:.1} GiB vs RAM {:.1} GiB — prefetching the whole \
                 model into the page cache",
                file.unwrap_or(0) as f64 / 2f64.powi(30),
                ram.unwrap_or(0) as f64 / 2f64.powi(30),
            );
            plan
        }
        _ => {
            tracing::info!(
                "page-cache auto: cannot size model/RAM, leaving prefetch and pinning off"
            );
            (false, false)
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A model that fits in RAM is prefetched whole; a model larger than RAM
    /// gets the dense/expert pinning split instead.
    #[test]
    fn cache_plan_splits_on_whether_the_model_fits_in_ram() {
        let ram = Some(24 * 1024 * 1024 * 1024); // 24 GiB, like this Mac

        // 400 MiB model: prefetch the whole file into the page cache.
        assert_eq!(cache_plan_for(Some(400 * 1024 * 1024), ram), (false, true));
        // Exactly RAM-sized still prefetches (the cache is reclaimable).
        assert_eq!(cache_plan_for(ram, ram), (false, true));
        // Larger than RAM: only the hot set is prefetched.
        assert_eq!(
            cache_plan_for(Some(64 * 1024 * 1024 * 1024), ram),
            (true, false)
        );
        // Unknown model size or unknown RAM: both off (manual control).
        assert_eq!(cache_plan_for(None, ram), (false, false));
        assert_eq!(cache_plan_for(Some(1), None), (false, false));
        assert_eq!(cache_plan_for(None, None), (false, false));
    }

    /// The huge-page default is transparent on Linux (best-effort there) and
    /// off on platforms with no file-backed huge-page API.
    #[test]
    fn default_huge_pages_follows_the_platform() {
        let expected = if cfg!(target_os = "linux") {
            HugePagesArg::Transparent
        } else {
            HugePagesArg::Off
        };
        assert_eq!(default_huge_pages(), expected);
    }

    /// RAM detection works where supported and never panics elsewhere.
    #[test]
    fn total_ram_bytes_reads_without_panicking() {
        if let Some(bytes) = total_ram_bytes() {
            assert!(bytes > 0, "reported RAM must be positive");
        }
    }
}
