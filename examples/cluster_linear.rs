//! A static-cluster linear-layer proof, not a distributed language-model runner.

use std::{
    fs::File,
    io::BufReader,
    net::{Ipv4Addr, SocketAddr},
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{ensure, Context, Result};
use candle_core::{
    quantized::{gguf_file, GgmlDType, QTensor},
    Device, Tensor,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use joshua::{
    distributed::{
        collective::{AllReduceGroup, MulticastConfig, MAX_PARTICIPANTS},
        discovery::{Discovery, NodeInfo},
        partition::{plan_partition, LinkObservation, NodeCapacity, TensorWorkload},
        shard::{dtype_layout, ShardedTensor},
        tcp::{TcpAllReduceGroup, TcpConfig},
    },
    gguf_ext,
};
use uuid::Uuid;

#[derive(Parser)]
#[command(about = "Experimental GGUF linear-layer sharding proof (not full LLM inference)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write a small deterministic Q8_0 linear weight, without a model download.
    Fixture {
        #[arg(long)]
        output: PathBuf,
    },
    /// Sum local shards and compare with an unsharded matvec.
    Local {
        #[command(flatten)]
        matrix: MatrixArgs,
        #[arg(long, default_value_t = 2)]
        shards: usize,
    },
    /// Run one rank; every rank needs identical weights, tensor, session and key.
    Rank {
        #[command(flatten)]
        matrix: MatrixArgs,
        #[arg(long)]
        rank: usize,
        #[arg(long, default_value_t = 2)]
        world_size: usize,
        /// Select TCP explicitly on networks without multicast; never switches mid-job.
        #[arg(long, value_enum, default_value_t = Transport::Udp)]
        transport: Transport,
        /// TCP listen addresses in rank order; identical on every rank.
        #[arg(long, value_delimiter = ',', required_if_eq("transport", "tcp"))]
        peers: Vec<SocketAddr>,
        /// Fresh shared v4 UUID for this job (never reuse after a restart).
        #[arg(long)]
        session: Uuid,
        #[arg(long, default_value = "239.255.88.1")]
        group: Ipv4Addr,
        #[arg(long, default_value_t = 48888)]
        port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        interface: Ipv4Addr,
        #[arg(long, default_value_t = 10)]
        timeout_seconds: u64,
        #[arg(long, default_value_t = 1)]
        steps: usize,
        /// Also touch ALL weights to check the result; disables shard-only residency.
        #[arg(long)]
        verify: bool,
    },
    /// Advertise a caller-persisted node identity and print advisory peers.
    Discover {
        #[arg(long)]
        node_id: Uuid,
        /// Advertised future control port; this proof does not start a TCP server.
        #[arg(long, default_value_t = 5359)]
        port: u16,
        #[arg(long, default_value_t = 30)]
        seconds: u64,
    },
    /// Print an allocation for this matrix using validated JSON node/link records.
    Plan {
        #[command(flatten)]
        matrix: MatrixArgs,
        /// JSON array of NodeCapacity, including per-node runtime reservations.
        #[arg(long)]
        nodes: PathBuf,
        /// Optional JSON array of measured LinkObservation records.
        #[arg(long)]
        links: Option<PathBuf>,
    },
}

#[derive(Args)]
struct MatrixArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "linear.weight")]
    tensor: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Transport {
    Udp,
    Tcp,
}

enum Collective {
    Udp(AllReduceGroup),
    Tcp(TcpAllReduceGroup),
}

impl Collective {
    fn all_reduce_sum(&mut self, values: &mut [f32]) -> Result<()> {
        match self {
            Self::Udp(group) => group.all_reduce_sum(values),
            Self::Tcp(group) => group.all_reduce_sum(values),
        }
    }
}

fn fixture(output: &PathBuf) -> Result<()> {
    let weights: Vec<f32> = (0..4 * 256)
        .map(|i| ((i * 17 % 101) as f32 - 50.0) / 64.0)
        .collect();
    let tensor = Tensor::from_vec(weights, (4, 256), &Device::Cpu)?;
    let quantized = QTensor::quantize(&tensor, GgmlDType::Q8_0)?;
    // Refuse to overwrite a file that could already be mapped by another rank.
    let mut file = File::create_new(output)?;
    gguf_file::write(&mut file, &[], &[("linear.weight", &quantized)])?;
    Ok(())
}

fn load(matrix: &MatrixArgs, rank: usize, count: usize) -> Result<ShardedTensor> {
    let file = File::open(&matrix.model)?;
    let header = gguf_ext::read_header(&mut BufReader::new(&file))?;
    let info = header
        .tensors
        .get(&matrix.tensor)
        .context("tensor not found in GGUF")?;
    let shard = ShardedTensor::map_file(&file, info, header.tensor_data_offset, rank, count)?;
    ensure!(
        shard.input_width() <= 1 << 24 && shard.output_width() <= 1 << 24,
        "matrix exceeds this example's activation size limit"
    );
    Ok(shard)
}

fn input(width: usize) -> Vec<f32> {
    (0..width)
        .map(|i| ((i % 29) as f32 - 14.0) / 29.0)
        .collect()
}

fn compare(actual: &[f32], expected: &[f32]) -> Result<f32> {
    ensure!(actual.len() == expected.len(), "output shape mismatch");
    let mut max_error = 0.0f32;
    for (&a, &b) in actual.iter().zip(expected) {
        let error = (a - b).abs();
        ensure!(
            a.is_finite() && b.is_finite() && error <= 1e-4 + 1e-4 * b.abs(),
            "sharded output differs from reference: {a} versus {b}"
        );
        max_error = max_error.max(error);
    }
    Ok(max_error)
}

fn key_from_env() -> Result<Vec<u8>> {
    let key = std::env::var("JOSHUA_CLUSTER_KEY").map_err(|_| {
        anyhow::anyhow!("set JOSHUA_CLUSTER_KEY to 64 hex characters of random key material")
    })?;
    decode_key(&key)
}

fn decode_key(key: &str) -> Result<Vec<u8>> {
    ensure!(
        key.len() == 64 && key.bytes().all(|b| b.is_ascii_hexdigit()),
        "JOSHUA_CLUSTER_KEY must contain exactly 64 hexadecimal characters"
    );
    (0..64)
        .step_by(2)
        .map(|i| u8::from_str_radix(&key[i..i + 2], 16).context("invalid key encoding"))
        .collect()
}

fn plan(matrix: &MatrixArgs, nodes: &PathBuf, links: Option<&PathBuf>) -> Result<()> {
    let mut file = BufReader::new(File::open(&matrix.model)?);
    let header = gguf_ext::read_header(&mut file)?;
    let tensor = header
        .tensors
        .get(&matrix.tensor)
        .context("tensor not found in GGUF")?;
    ensure!(tensor.dims.len() == 2, "planning requires a matrix");
    let layout = dtype_layout(tensor.dtype)?;
    let workload = TensorWorkload::from_layout(
        &matrix.tensor,
        tensor.dims[1] as u64,
        tensor.dims[0] as u64,
        layout.block_elements as u64,
        layout.block_bytes as u64,
    )?;
    let nodes: Vec<NodeCapacity> = serde_json::from_reader(BufReader::new(File::open(nodes)?))?;
    let links: Vec<LinkObservation> = match links {
        Some(path) => serde_json::from_reader(BufReader::new(File::open(path)?))?,
        None => Vec::new(),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&plan_partition(&nodes, &[workload], &links)?)?
    );
    Ok(())
}

fn local(matrix: &MatrixArgs, shards: usize) -> Result<Vec<f32>> {
    ensure!(shards > 0 && shards <= 256, "shards must be in 1..=256");
    let full = load(matrix, 0, 1)?;
    let x = input(full.input_width());
    let reference = full.forward(&x)?;
    let mut sum = vec![0.0f32; full.output_width()];
    for rank in 0..shards {
        let shard = load(matrix, rank, shards)?;
        for (out, part) in sum.iter_mut().zip(shard.forward(&x)?) {
            *out += part;
        }
    }
    let max_error = compare(&sum, &reference)?;
    println!("local shards={shards} max_abs_error={max_error}");
    println!("{}", serde_json::to_string(&sum)?);
    Ok(sum)
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Fixture { output } => fixture(&output),
        Command::Local { matrix, shards } => local(&matrix, shards).map(|_| ()),
        Command::Rank {
            matrix,
            rank,
            world_size,
            transport,
            peers,
            session,
            group,
            port,
            interface,
            timeout_seconds,
            steps,
            verify,
        } => {
            ensure!(
                (1..=MAX_PARTICIPANTS).contains(&world_size) && rank < world_size,
                "invalid rank or world size"
            );
            ensure!((1..=1000).contains(&steps), "steps must be in 1..=1000");
            let key = key_from_env()?;
            let timeout = Duration::from_secs(timeout_seconds);
            let mut collective = match transport {
                Transport::Udp => {
                    ensure!(peers.is_empty(), "--peers requires --transport tcp");
                    Collective::Udp(AllReduceGroup::new(
                        rank,
                        world_size,
                        session,
                        &key,
                        MulticastConfig {
                            group,
                            port,
                            interface,
                            timeout,
                            ..Default::default()
                        },
                    )?)
                }
                Transport::Tcp => {
                    ensure!(
                        peers.len() == world_size,
                        "--peers must contain exactly --world-size addresses in rank order"
                    );
                    Collective::Tcp(TcpAllReduceGroup::new(
                        rank,
                        session,
                        &key,
                        TcpConfig { peers, timeout },
                    )?)
                }
            };
            let shard = load(&matrix, rank, world_size)?;
            shard.prefetch()?;
            let x = input(shard.input_width());
            for step in 0..steps {
                let mut output = shard.forward(&x)?;
                collective.all_reduce_sum(&mut output)?;
                if verify {
                    let reference = load(&matrix, 0, 1)?.forward(&x)?;
                    eprintln!(
                        "rank={rank} step={step} max_abs_error={}",
                        compare(&output, &reference)?
                    );
                }
                println!("{}", serde_json::to_string(&output)?);
            }
            Ok(())
        }
        Command::Discover {
            node_id,
            port,
            seconds,
        } => {
            ensure!((1..=3600).contains(&seconds), "seconds must be in 1..=3600");
            let mut discovery = Discovery::new(NodeInfo::local(node_id), port)?;
            let end = Instant::now() + Duration::from_secs(seconds);
            while Instant::now() < end {
                discovery.poll(
                    end.saturating_duration_since(Instant::now())
                        .min(Duration::from_secs(5)),
                )?;
                println!(
                    "coordinator={} peers={:?}",
                    discovery.coordinator().uuid,
                    discovery.peers()
                );
            }
            discovery.shutdown()
        }
        Command::Plan {
            matrix,
            nodes,
            links,
        } => plan(&matrix, &nodes, links.as_ref()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_and_local_shards_match() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("joshua-linear-{}.gguf", uuid::Uuid::new_v4()));
        fixture(&path)?;
        let matrix = MatrixArgs {
            model: path.clone(),
            tensor: "linear.weight".into(),
        };
        let result = local(&matrix, 3);
        std::fs::remove_file(path)?;
        assert_eq!(result?.len(), 4);
        Ok(())
    }

    #[test]
    fn comparisons_reject_nonfinite_and_bad_shapes() {
        assert!(compare(&[f32::NAN], &[0.0]).is_err());
        assert!(compare(&[0.0], &[f32::INFINITY]).is_err());
        assert!(compare(&[1.0], &[]).is_err());
        assert!(compare(&[1.0], &[0.0]).is_err());
    }

    #[test]
    fn key_parser_rejects_invalid_encodings_without_echoing_them() {
        let key: String = (0..32).map(|i| format!("{i:02x}")).collect();
        assert_eq!(decode_key(&key).unwrap(), (0..32).collect::<Vec<_>>());
        assert!(decode_key("").is_err());
        assert!(decode_key(&"é".repeat(32)).is_err());
        assert!(decode_key(&"x".repeat(64)).is_err());
    }

    #[test]
    fn transport_cli_defaults_to_udp_and_requires_tcp_peers() -> Result<()> {
        let session = Uuid::new_v4().to_string();
        let base = [
            "cluster_linear",
            "rank",
            "--model",
            "model.gguf",
            "--rank",
            "0",
            "--session",
            &session,
        ];
        assert!(matches!(
            Cli::try_parse_from(base)?.command,
            Command::Rank {
                transport: Transport::Udp,
                peers,
                ..
            } if peers.is_empty()
        ));
        let mut args = base.to_vec();
        args.extend(["--transport", "tcp"]);
        assert!(Cli::try_parse_from(&args).is_err());
        args.extend(["--peers", "127.0.0.1:48888,127.0.0.1:48889"]);
        assert!(matches!(
            Cli::try_parse_from(&args)?.command,
            Command::Rank {
                transport: Transport::Tcp,
                peers,
                ..
            } if peers.len() == 2
        ));
        Ok(())
    }

    #[test]
    fn tcp_quantized_shards_match_reference_across_steps() -> Result<()> {
        use std::net::TcpListener;

        let path = std::env::temp_dir().join(format!("joshua-tcp-{}.gguf", Uuid::new_v4()));
        fixture(&path)?;
        let result = (|| -> Result<()> {
            let matrix = MatrixArgs {
                model: path.clone(),
                tensor: "linear.weight".into(),
            };
            let full = load(&matrix, 0, 1)?;
            let x = input(full.input_width());
            let reference = full.forward(&x)?;
            let reservations: Vec<_> = (0..3)
                .map(|_| TcpListener::bind("127.0.0.1:0"))
                .collect::<std::io::Result<_>>()?;
            let peers: Vec<_> = reservations
                .iter()
                .map(TcpListener::local_addr)
                .collect::<std::io::Result<_>>()?;
            drop(reservations);
            let session = Uuid::new_v4();
            let key = rand::random::<[u8; 32]>();
            let mut ranks = Vec::new();
            for rank in 0..peers.len() {
                ranks.push((
                    TcpAllReduceGroup::new(
                        rank,
                        session,
                        &key,
                        TcpConfig {
                            peers: peers.clone(),
                            timeout: Duration::from_secs(5),
                        },
                    )?,
                    load(&matrix, rank, peers.len())?,
                ));
            }
            std::thread::scope(|scope| {
                let handles: Vec<_> = ranks
                    .into_iter()
                    .map(|(mut collective, shard)| {
                        let x = &x;
                        let reference = &reference;
                        scope.spawn(move || -> Result<()> {
                            shard.prefetch()?;
                            for _ in 0..3 {
                                let mut output = shard.forward(x)?;
                                collective.all_reduce_sum(&mut output)?;
                                compare(&output, reference)?;
                            }
                            Ok(())
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().expect("rank thread panicked")?;
                }
                Ok(())
            })
        })();
        std::fs::remove_file(path)?;
        result
    }
}
