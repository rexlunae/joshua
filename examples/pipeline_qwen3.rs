//! CPU-only static pipeline proof. Each worker is a separate process and owns
//! a contiguous layer range; run `--help` for plan/worker/run commands.
use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use joshua::distributed::pipeline::{Input, Limits, Pipeline, Plan, Worker};
use std::{
    net::{SocketAddr, TcpListener},
    path::PathBuf,
    time::Instant,
};
use uuid::Uuid;

#[derive(Parser)]
#[command(about = "Experimental CPU Qwen3 dense pipeline")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect/hash a local immutable model and save the shared stage manifest.
    Plan {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Exclusive layer ends, including the model's final layer count.
        #[arg(long, value_delimiter = ',')]
        ends: Vec<usize>,
        #[arg(long, value_delimiter = ',')]
        budgets_mib: Vec<u64>,
        #[arg(long, default_value_t = 4096)]
        context: usize,
        #[arg(long, default_value_t = 128)]
        chunk: usize,
        #[arg(long, default_value_t = 8)]
        sessions: usize,
        #[arg(long, default_value_t = 2)]
        queue_depth: usize,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 30_000)]
        timeout_ms: u64,
        #[arg(long, default_value_t = 256)]
        coordinator_budget_mib: u64,
    },
    /// One worker handles one coordinator connection then exits. Use a fresh
    /// job UUID after restart; authentication does not encrypt activation data.
    Worker {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        rank: usize,
        #[arg(long)]
        listen: SocketAddr,
        #[arg(long)]
        job: Uuid,
        #[arg(long, env = "JOSHUA_PIPELINE_KEY", hide_env_values = true)]
        key: String,
        #[arg(long)]
        no_mmap: bool,
    },
    /// Greedy token-ID generation; no tokenizer/server API integration yet.
    Run {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long, value_delimiter = ',')]
        peers: Vec<SocketAddr>,
        #[arg(long)]
        job: Uuid,
        #[arg(long, env = "JOSHUA_PIPELINE_KEY", hide_env_values = true)]
        key: String,
        #[arg(long, value_delimiter = ',')]
        tokens: Vec<u32>,
        #[arg(long, default_value_t = 16)]
        decode: usize,
        #[arg(long, default_value_t = 1)]
        requests: usize,
    },
}

fn read_plan(path: PathBuf) -> Result<Plan> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() <= 64 * 1024, "manifest too large");
    let plan: Plan = serde_json::from_slice(&bytes)?;
    plan.validate()?;
    Ok(plan)
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Plan {
            model,
            output,
            ends,
            budgets_mib,
            context,
            chunk,
            sessions,
            queue_depth,
            batch,
            timeout_ms,
            coordinator_budget_mib,
        } => {
            let budgets = budgets_mib
                .iter()
                .map(|m| {
                    m.checked_mul(1024 * 1024)
                        .ok_or_else(|| anyhow::anyhow!("budget overflow"))
                })
                .collect::<Result<Vec<_>>>()?;
            let plan = Plan::from_gguf(
                model,
                &ends,
                &budgets,
                Limits {
                    context,
                    chunk,
                    sessions,
                    queue_depth,
                    batch,
                    timeout_ms,
                    coordinator_memory_budget: coordinator_budget_mib
                        .checked_mul(1024 * 1024)
                        .ok_or_else(|| anyhow::anyhow!("budget overflow"))?,
                },
            )?;
            std::fs::write(output, serde_json::to_vec_pretty(&plan)?)?;
        }
        Command::Worker {
            model,
            plan,
            rank,
            listen,
            job,
            key,
            no_mmap,
        } => {
            let worker = Worker::load(model, read_plan(plan)?, rank, !no_mmap)?;
            let listener = TcpListener::bind(listen)?;
            let listen = listener.local_addr()?;
            eprintln!(
                "stage={rank} reservation_bytes={} listen={listen}",
                worker.reserved_bytes
            );
            worker.serve(listener, job, key.as_bytes())?;
        }
        Command::Run {
            plan,
            peers,
            job,
            key,
            tokens,
            decode,
            requests,
        } => {
            let plan = read_plan(plan)?;
            ensure!(
                !tokens.is_empty()
                    && decode > 0
                    && requests > 0
                    && requests <= plan.limits.sessions
                    && requests <= plan.limits.batch,
                "invalid request count/prompt/decode"
            );
            ensure!(
                tokens
                    .len()
                    .checked_add(decode - 1)
                    .is_some_and(|n| n <= plan.limits.context),
                "context exhausted"
            );
            let chunk = plan.limits.chunk;
            let mut pipeline = Pipeline::connect(plan, &peers, job, key.as_bytes())?;
            let ids = (0..requests)
                .map(|_| pipeline.open())
                .collect::<Result<Vec<_>>>()?;
            let t = Instant::now();
            let mut logits = Vec::new();
            for (i, tokens) in tokens.chunks(chunk).enumerate() {
                logits = pipeline.forward_batch(
                    &ids.iter()
                        .map(|&id| Input {
                            session: id,
                            offset: i * chunk,
                            tokens: tokens.to_vec(),
                        })
                        .collect::<Vec<_>>(),
                )?;
            }
            let prefill_ms = t.elapsed().as_secs_f64() * 1000.;
            let t = Instant::now();
            let mut generated = vec![Vec::with_capacity(decode); requests];
            for step in 0..decode {
                let choices: Vec<u32> = logits
                    .iter()
                    .map(|row| {
                        // Stable lowest-index argmax, as in local greedy sampling.
                        row.iter()
                            .enumerate()
                            .fold((0, f32::NEG_INFINITY), |best, (i, &x)| {
                                if x > best.1 {
                                    (i, x)
                                } else {
                                    best
                                }
                            })
                            .0 as u32
                    })
                    .collect();
                for (result, &token) in generated.iter_mut().zip(&choices) {
                    result.push(token);
                }
                if step + 1 < decode {
                    logits = pipeline.forward_batch(
                        &ids.iter()
                            .zip(&choices)
                            .map(|(&id, &token)| Input {
                                session: id,
                                offset: tokens.len() + step,
                                tokens: vec![token],
                            })
                            .collect::<Vec<_>>(),
                    )?;
                }
            }
            println!(
                "{}",
                serde_json::json!({"generated":generated,"prefill_ms":prefill_ms,
                "decode_ms":t.elapsed().as_secs_f64()*1000.,"stages":pipeline.metrics()})
            );
            for id in ids {
                pipeline.close(id)?;
            }
            pipeline.shutdown()?;
        }
    }
    Ok(())
}
