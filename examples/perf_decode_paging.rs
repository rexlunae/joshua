//! CPU decode through the public engine, including optional bounded residency.
//! Usage: perf_decode_paging MODEL PROMPT CONTEXT TOKENS REPEATS LOCK_BYTES
//! Run under the same RAM/swap limits for each candidate. Locking is required
//! when LOCK_BYTES is nonzero, so a memlock failure cannot masquerade as a run
//! with resident weights. Every completion has fresh KV state and greedy
//! sampling; loading is timed separately. This does not evict the page cache.

use std::time::Instant;

use joshua::{
    ChatMessage, ComputeBackend, Engine, EngineOptions, GenerationOptions, HugePages, MlockMode,
};

fn read_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/self/io")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("read_bytes: ")?.trim().parse().ok())
}

fn locked_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmLck:")?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
        })?
        .checked_mul(1024)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() == 6,
        "usage: perf_decode_paging MODEL PROMPT CONTEXT TOKENS REPEATS LOCK_BYTES"
    );
    let context: u32 = args[2].parse()?;
    let tokens: u32 = args[3].parse()?;
    let repeats: usize = args[4].parse()?;
    let lock_bytes: u64 = args[5].parse()?;
    anyhow::ensure!(
        context > 0 && tokens > 0 && repeats > 0,
        "counts must be positive"
    );
    let prompt = std::fs::read_to_string(&args[1])?;
    let options = EngineOptions::with_n_ctx(context)
        .backend(ComputeBackend::Cpu)
        .huge_pages(HugePages::Off)
        .prefetch_whole_model(false)
        .pin_hot_weights(false)
        .mlock_hot_weights(if lock_bytes == 0 {
            MlockMode::Off
        } else {
            MlockMode::Required
        })
        .mlock_weight_budget(Some(lock_bytes));
    let start = Instant::now();
    let engine = Engine::with_options(&args[0], options)?;
    println!(
        "load_s={:.6} lock_budget_bytes={lock_bytes} locked_bytes={:?}",
        start.elapsed().as_secs_f64(),
        locked_bytes()
    );
    let messages = [ChatMessage::text("user".to_owned(), prompt)];
    let generation = GenerationOptions {
        max_tokens: tokens,
        temperature: 0.0,
        ..Default::default()
    };
    for repeat in 0..repeats {
        let before = read_bytes();
        let start = Instant::now();
        let (text, usage, prefill_tps, decode_tps) = engine.complete(&messages, &generation)?;
        let completion_s = start.elapsed().as_secs_f64();
        let io_bytes = before.zip(read_bytes()).map(|(a, b)| b.saturating_sub(a));
        println!("repeat={repeat} prompt={} completion={} completion_s={completion_s:.6} prefill_tps={prefill_tps:.6} decode_tps={decode_tps:.6} completion_read_bytes={io_bytes:?} text={text:?}",
            usage.prompt_tokens, usage.completion_tokens);
    }
    Ok(())
}
