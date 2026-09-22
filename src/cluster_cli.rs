//! One-shot CPU DeepSeek V4 generation on an explicitly configured cluster.

use std::{
    fs::File,
    io::BufReader,
    net::{Ipv4Addr, SocketAddr},
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{ensure, Context, Result};
use clap::{Args, ValueEnum};
use joshua::{
    distributed::{
        collective::MulticastConfig, deepseek4::DeepSeekCluster, session::ClusterSession,
        tcp::TcpConfig,
    },
    gguf_ext,
    template::ChatTemplate,
    ChatMessage,
};
use tokenizers::Tokenizer;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Transport {
    Tcp,
    Udp,
}

#[derive(Args)]
pub struct ClusterRun {
    /// Immutable local DeepSeek V4 GGUF, identical on every rank.
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    rank: usize,
    #[arg(long, default_value_t = 2)]
    world_size: usize,
    /// Fresh random v4 UUID shared by all ranks. Never reuse after a failure.
    #[arg(long)]
    session: Uuid,
    #[arg(long, value_enum, default_value_t = Transport::Tcp)]
    transport: Transport,
    /// Explicit TCP endpoints in rank order, identical on every rank.
    #[arg(long, value_delimiter = ',', required_if_eq("transport", "tcp"))]
    peers: Vec<SocketAddr>,
    #[arg(long, default_value = "239.255.88.1")]
    group: Ipv4Addr,
    #[arg(long, default_value_t = 48888)]
    port: u16,
    #[arg(long, default_value = "127.0.0.1")]
    interface: Ipv4Addr,
    #[arg(long, default_value_t = 120)]
    timeout_seconds: u64,
    #[arg(long, default_value_t = 4096)]
    n_ctx: usize,
    #[arg(long, default_value_t = 32)]
    max_tokens: usize,
    /// Identical prefill chunk size on all ranks; bounds activation workspace.
    #[arg(long, default_value_t = 32)]
    prefill_chunk: usize,
    /// Defaults to tokenizer.json next to the GGUF; not needed with --tokens.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Treat the prompt as preformatted text instead of rendering the GGUF chat template.
    #[arg(long, requires = "prompt")]
    raw_prompt: bool,
    /// Pre-tokenized input for correctness tests; prints generated IDs as JSON.
    #[arg(
        long,
        value_delimiter = ',',
        conflicts_with = "prompt",
        required_unless_present = "prompt"
    )]
    tokens: Option<Vec<u32>>,
    /// Chat message (or preformatted text with --raw-prompt), identical on all ranks.
    #[arg(required_unless_present = "tokens")]
    prompt: Option<String>,
}

fn decode_key(key: &str) -> Result<Vec<u8>> {
    ensure!(
        key.len() == 64 && key.bytes().all(|b| b.is_ascii_hexdigit()),
        "JOSHUA_CLUSTER_KEY must contain exactly 64 hexadecimal characters"
    );
    (0..64)
        .step_by(2)
        .map(|i| u8::from_str_radix(&key[i..i + 2], 16).context("invalid cluster key encoding"))
        .collect()
}

fn greedy(logits: &[f32]) -> Result<u32> {
    ensure!(!logits.is_empty(), "empty model logits");
    ensure!(
        logits.iter().all(|v| v.is_finite()),
        "nonfinite model logits"
    );
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    u32::try_from(best).context("vocabulary exceeds token ID range")
}

impl ClusterRun {
    pub fn run(self) -> Result<()> {
        ensure!(
            (1..=262_144).contains(&self.n_ctx),
            "--n-ctx must be in 1..=262144"
        );
        ensure!(
            (1..=self.n_ctx).contains(&self.max_tokens),
            "invalid --max-tokens"
        );
        ensure!(
            (1..=512).contains(&self.prefill_chunk),
            "--prefill-chunk must be in 1..=512"
        );
        let key = std::env::var("JOSHUA_CLUSTER_KEY")
            .context("set JOSHUA_CLUSTER_KEY to 64 hex characters of random key material")?;
        let key = decode_key(&key)?;
        let timeout = Duration::from_secs(self.timeout_seconds);
        let cluster = Arc::new(match self.transport {
            Transport::Tcp => {
                ensure!(
                    self.peers.len() == self.world_size,
                    "--peers must match --world-size"
                );
                ClusterSession::tcp(
                    self.rank,
                    self.session,
                    &key,
                    TcpConfig {
                        peers: self.peers,
                        timeout,
                    },
                )?
            }
            Transport::Udp => {
                ensure!(self.peers.is_empty(), "--peers requires --transport tcp");
                ClusterSession::udp(
                    self.rank,
                    self.world_size,
                    self.session,
                    &key,
                    MulticastConfig {
                        group: self.group,
                        port: self.port,
                        interface: self.interface,
                        timeout,
                        ..Default::default()
                    },
                )?
            }
        });

        let header = gguf_ext::read_header(&mut BufReader::new(File::open(&self.model)?))?;
        let mut eos = Vec::new();
        if let Some(value) = header.metadata.get("tokenizer.ggml.eos_token_id") {
            eos.push(value.to_u32().context("invalid EOS metadata")?);
        }
        let (input, tokenizer) = if let Some(tokens) = self.tokens {
            (tokens, None)
        } else {
            let path = self.tokenizer.unwrap_or_else(|| {
                self.model
                    .parent()
                    .unwrap_or(std::path::Path::new("."))
                    .join("tokenizer.json")
            });
            let tokenizer = Tokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("loading cluster tokenizer: {e}"))?;
            let prompt = self.prompt.context("missing prompt")?;
            let text = if self.raw_prompt {
                prompt
            } else {
                let source = header
                    .metadata
                    .get("tokenizer.chat_template")
                    .context(
                        "GGUF has no chat template; supply preformatted text with --raw-prompt",
                    )?
                    .to_string()?;
                let special = |key: &str| {
                    header
                        .metadata
                        .get(key)
                        .and_then(|v| v.to_u32().ok())
                        .and_then(|id| tokenizer.id_to_token(id))
                        .unwrap_or_default()
                };
                ChatTemplate::new(
                    source,
                    special("tokenizer.ggml.bos_token_id"),
                    special("tokenizer.ggml.eos_token_id"),
                )
                .render(&[ChatMessage::text("user", prompt)], None)
                .map_err(anyhow::Error::msg)?
            };
            let input = tokenizer
                .encode(text, self.raw_prompt)
                .map_err(|e| anyhow::anyhow!("encoding cluster prompt: {e}"))?
                .get_ids()
                .to_vec();
            for marker in ["<｜end▁of▁sentence｜>", "</s>", "<|im_end|>"] {
                if let Some(id) = tokenizer.token_to_id(marker) {
                    eos.push(id);
                }
            }
            (input, Some(tokenizer))
        };
        ensure!(!input.is_empty(), "prompt must contain at least one token");
        ensure!(
            input
                .len()
                .checked_add(self.max_tokens)
                .is_some_and(|n| n <= self.n_ctx),
            "prompt plus --max-tokens exceeds --n-ctx"
        );
        eos.sort_unstable();
        eos.dedup();
        cluster.agree(&serde_json::to_vec(&(
            "joshua-cluster-run-v1",
            &input,
            self.n_ctx,
            self.max_tokens,
            self.prefill_chunk,
            &eos,
        ))?)?;
        eprintln!(
            "rank={}: loading CPU DeepSeek V4 shards (dense/shared weights and KV replicated)",
            self.rank
        );
        let mut model = DeepSeekCluster::load(&self.model, self.n_ctx, Arc::clone(&cluster))?;
        let (local, full) = model.expert_shard_bytes();
        eprintln!(
            "rank={} routed_expert_bytes={} full_routed_expert_bytes={} (encoded weights, not RSS)",
            self.rank, local, full
        );
        cluster.agree(b"joshua-cluster-model-ready-v1")?;
        let start = Instant::now();
        let mut logits = Vec::new();
        let mut offset = 0;
        for chunk in input.chunks(self.prefill_chunk) {
            logits = model.forward(chunk, offset)?;
            offset += chunk.len();
        }
        let mut generated = Vec::new();
        for step in 0..self.max_tokens {
            let token = if self.rank == 0 { greedy(&logits)? } else { 0 };
            // Two 16-bit pieces preserve every u32 token ID exactly through f32.
            let mut wire = [(token & 0xffff) as f32, (token >> 16) as f32];
            cluster.broadcast(&mut wire)?;
            let token = wire[0] as u32 | ((wire[1] as u32) << 16);
            if eos.contains(&token) {
                break;
            }
            generated.push(token);
            if step + 1 < self.max_tokens {
                logits = model.forward(&[token], offset)?;
                offset += 1;
            }
        }
        cluster.agree(b"joshua-cluster-generation-finished-v1")?;
        if self.rank == 0 {
            match tokenizer {
                Some(tokenizer) => println!(
                    "{}",
                    tokenizer
                        .decode(&generated, true)
                        .map_err(|e| anyhow::anyhow!("decoding cluster output: {e}"))?
                ),
                None => println!("{}", serde_json::to_string(&generated)?),
            }
            eprintln!(
                "prompt_tokens={} generated_tokens={} inference_seconds={:.3}",
                input.len(),
                generated.len(),
                start.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        run: ClusterRun,
    }

    #[test]
    fn parser_requires_inputs_and_tcp_membership() {
        let id = Uuid::new_v4().to_string();
        let mut args = vec![
            "test",
            "--model",
            "model.gguf",
            "--rank",
            "0",
            "--session",
            &id,
        ];
        assert!(TestCli::try_parse_from(&args).is_err());
        args.extend(["--tokens", "1,2"]);
        args.extend(["--transport", "tcp"]);
        assert!(TestCli::try_parse_from(&args).is_err());
        args.extend(["--peers", "127.0.0.1:48888,127.0.0.1:48889"]);
        let cli = TestCli::try_parse_from(&args).unwrap();
        assert_eq!(cli.run.tokens.unwrap(), vec![1, 2]);
        args.push("conflicting prompt");
        assert!(TestCli::try_parse_from(&args).is_err());
    }

    #[test]
    fn key_validation_and_greedy_ties() {
        assert!(decode_key(&"é".repeat(32)).is_err());
        assert!(decode_key("").is_err());
        let key = rand::random::<[u8; 32]>();
        let hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(decode_key(&hex).unwrap(), key);
        assert_eq!(greedy(&[0.0, 2.0, 2.0]).unwrap(), 1);
        assert!(greedy(&[]).is_err());
        assert!(greedy(&[f32::NAN]).is_err());
    }
}
