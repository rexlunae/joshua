//! Greedy-parity check for speculative decoding on a recurrent (hybrid)
//! architecture.
//!
//! Speculative decoding is only sound if the accepted tokens are exactly the
//! ones plain decoding would have produced.  For a model with Gated DeltaNet
//! layers that is not free: their state is *running* rather than append-only,
//! so a rolled-back draft has to restore a pre-pass snapshot **and replay the
//! accepted prefix** ([`joshua::native_session`]).  A bug there does not crash
//! — it silently diverges.
//!
//! This runs the same greedy prompt twice through one engine, once with
//! speculative decoding off and once on, and requires the generated token
//! sequences to match exactly.
//!
//! ```sh
//! cargo run --release --example spec_parity -- model.gguf "Write a haiku about Rust." 48
//! ```

use joshua::engine::{Engine, EngineOptions};
use joshua::types::GenerationOptions;
use joshua::{ChatMessage, SpeculativeConfig};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model = match args.next() {
        Some(m) => m,
        None => {
            eprintln!("usage: spec_parity <model.gguf> [prompt] [max_tokens]");
            std::process::exit(2);
        }
    };
    let prompt = args
        .next()
        .unwrap_or_else(|| "Write a haiku about Rust.".to_string());
    let max_tokens: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(48);

    // Greedy: a draft is then accepted iff it is the target's argmax, so the
    // two runs must agree token for token or the comparison is meaningless.
    let options = GenerationOptions {
        max_tokens,
        temperature: 0.0,
        ..GenerationOptions::default()
    };

    let plain = Engine::with_options(
        &model,
        EngineOptions::with_n_ctx(2048).speculative(None),
    )?;
    let msgs = vec![ChatMessage::text("user".to_string(), prompt.clone())];
    let (plain_text, plain_usage, plain_pf, plain_dec) = plain.complete(&msgs, &options)?;

    let spec = Engine::with_options(
        &model,
        EngineOptions::with_n_ctx(2048)
            .speculative(Some(SpeculativeConfig::with_max_draft(4))),
    )?;
    let (spec_text, spec_usage, spec_pf, spec_dec) = spec.complete(&msgs, &options)?;
    let stats = spec.speculative_stats();

    // The CLI's progress writes use `\r`, so a log is not a reliable place to
    // read the response back out of.  Drop each text in its own file.
    let plain_path = std::env::var("SPEC_PARITY_PLAIN_OUT")
        .unwrap_or_else(|_| "/tmp/spec_parity_plain.txt".to_string());
    let spec_path = std::env::var("SPEC_PARITY_SPEC_OUT")
        .unwrap_or_else(|_| "/tmp/spec_parity_spec.txt".to_string());
    std::fs::write(&plain_path, &plain_text).ok();
    std::fs::write(&spec_path, &spec_text).ok();
    println!("plain_text -> {plain_path} ({} chars)", plain_text.len());
    println!("spec_text  -> {spec_path} ({} chars)", spec_text.len());

    println!("--- plain ---");
    println!("{plain_text}");
    println!(
        "tokens={} prefill={plain_pf:.2}t/s decode={plain_dec:.2}t/s",
        plain_usage.completion_tokens
    );
    println!("--- speculative ---");
    println!("{spec_text}");
    println!(
        "tokens={} prefill={spec_pf:.2}t/s decode={spec_dec:.2}t/s \
         drafted={} accepted={} ({:.0}%) over {} verify steps",
        spec_usage.completion_tokens,
        stats.drafted,
        stats.accepted,
        stats.acceptance_rate() * 100.0,
        stats.verify_steps
    );

    println!("=== VERDICT ===");
    let same_text = plain_text == spec_text;
    let same_len = plain_usage.completion_tokens == spec_usage.completion_tokens;
    if same_text && same_len {
        let speedup = if plain_dec > 0.0 { spec_dec / plain_dec } else { 0.0 };
        println!("IDENTICAL ({} tokens)", plain_usage.completion_tokens);
        println!("decode speedup: {speedup:.2}x ({plain_dec:.2} -> {spec_dec:.2} t/s)");
        Ok(())
    } else {
        eprintln!("MISMATCH: text_equal={same_text} len_equal={same_len}");
        eprintln!("  plain len {} vs spec len {}", plain_text.len(), spec_text.len());
        for (i, (a, b)) in plain_text.chars().zip(spec_text.chars()).enumerate() {
            if a != b {
                eprintln!("  first diff at char {i}: {a:?} vs {b:?}");
                eprintln!("  plain: ...{}", &plain_text[..i.min(plain_text.len())].chars().rev().take(40).collect::<String>().chars().rev().collect::<String>());
                eprintln!("  spec : ...{}", &spec_text[..i.min(spec_text.len())].chars().rev().take(40).collect::<String>().chars().rev().collect::<String>());
                break;
            }
        }
        std::process::exit(1)
    }
}