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

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use joshua::{engine::Engine, server, types::GenerationOptions, ChatMessage};

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
        /// Address to listen on.
        #[arg(short, long, default_value = "0.0.0.0:8080")]
        addr: String,
        /// Context window size in tokens.
        #[arg(long, default_value_t = 4096)]
        n_ctx: u32,
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
        Commands::Serve { model, addr, n_ctx } => {
            let engine = Arc::new(Engine::with_n_ctx(&model, n_ctx)?);
            server::serve(engine, &addr).await?;
        }

        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            n_ctx,
        } => {
            let engine = Engine::with_n_ctx(&model, n_ctx)?;
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
