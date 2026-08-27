//! Token-level greedy generation with top-5 logit dump for DSV4 debugging.
//!
//! Usage:
//!   cargo run --release --example debug_dsv4 -- <model.gguf> <tokenizer.json> "<prompt>" [max_tokens] [--chat]
//!
//! With `--chat`, the GGUF-embedded chat template is rendered around the
//! prompt exactly like the engine's `complete_chat` path, so the collapse
//! behaviour can be compared raw vs templated.
//!
//! Prints every generated token with its id, decoded text, the top-5 logits,
//! and per-token decode time, so a collapse can be pinned to an exact
//! absolute KV position.

use std::io::Cursor;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use memmap2::Mmap;
use tokenizers::Tokenizer;

use joshua::gguf_ext;
use joshua::model::QuantizedModel;
use joshua::template::ChatTemplate;
use joshua::types::ChatMessage;

fn topk(flat: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut order: Vec<usize> = (0..flat.len()).collect();
    order.sort_by(|&a, &b| flat[b].total_cmp(&flat[a]));
    order[..k.min(order.len())]
        .iter()
        .map(|&i| (i, flat[i]))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = &args[1];
    let tok_path = &args[2];
    let prompt = &args[3];
    let mut max_tokens: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(80);
    let chat = args.iter().any(|a| a == "--chat");

    // ---- Load exactly like the engine (mmap + raw header + candle content).
    let file = std::fs::File::open(model_path)?;
    let mmap = Arc::new(unsafe { Mmap::map(&file)? });
    let raw = gguf_ext::read_header(&mut Cursor::new(&mmap[..]))?;
    let gguf = raw.to_candle_content()?;
    let device = Device::Cpu;
    let mut cursor = Cursor::new(&mmap[..]);
    let mut model = QuantizedModel::from_gguf_mmap(
        gguf,
        &mut cursor,
        &device,
        Some(Arc::clone(&mmap)),
    )?;
    println!("model loaded");

    let tokenizer = Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("{e}"))?;

    // ---- Optionally render the GGUF chat template (engine's complete_chat path).
    let mut text = prompt.clone();
    if chat {
        let src = raw
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string().ok().cloned())
            .ok_or_else(|| anyhow::anyhow!("no tokenizer.chat_template in GGUF"))?;
        let bos = match raw.metadata.get("tokenizer.ggml.bos_token_id") {
            Some(v) => tokenizer
                .decode(&[v.to_u64().unwrap_or(0) as u32], false)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            None => String::new(),
        };
        let eos = match raw.metadata.get("tokenizer.ggml.eos_token_id") {
            Some(v) => tokenizer
                .decode(&[v.to_u64().unwrap_or(1) as u32], false)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            None => String::new(),
        };
        let tpl = ChatTemplate::new(src, bos, eos);
        let msg = ChatMessage::text("user", prompt.clone());
        text = tpl
            .render(&[msg], None)
            .map_err(|e| anyhow::anyhow!("template render: {e}"))?;
        println!("--- rendered prompt ---\n{text}\n--- end ---");
    }

    let tokens: Vec<u32> = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    println!(
        "prompt: {} tokens -> {:?}",
        tokens.len(),
        tokens.iter().take(16).collect::<Vec<_>>()
    );

    let d = DType::F32;
    if tokens.len() > 64 {
        println!("NOTE: prompt already > 64 tokens; reducing max_tokens to 16");
        max_tokens = max_tokens.min(16);
    }

    // ---- Prefill.  The DSV4 forward returns ONLY the last position's
    // logits as [1, n_vocab], so the first generated token's logits come
    // straight out of prefill; each decode step then feeds exactly one token.
    let mut input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
    let mut logits = model.forward(&input, 0)?;
    let mut pos = tokens.len();
    let last_logits: Vec<f32> = logits.to_dtype(d)?.flatten_all()?.to_vec1()?;
    println!(
        "prefill done (ctx={pos}); top-5 at last prompt token: {:?}",
        topk(&last_logits, 5)
    );

    // ---- Greedy decode loop with per-token diagnostics.  The prefill logits
    // predict the token at absolute position `pos`; feed the chosen token at
    // that exact position (matching the engine's decode_loop), then advance.
    let mut gen = 0usize;
    while gen < max_tokens {
        let t0 = std::time::Instant::now();
        let flat: Vec<f32> = logits.to_dtype(d)?.flatten_all()?.to_vec1()?;
        let dt = t0.elapsed().as_secs_f32();

        let top = topk(&flat, 5);
        let (best, _) = top[0];
        let tok = tokenizer
            .decode(&[best as u32], false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let max_l = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let logsum: f32 = flat.iter().map(|x| (x - max_l).exp()).sum::<f32>().ln() + max_l;
        let top5_sum: f32 = top.iter().map(|(_, l)| l).sum();

        println!(
            "pos={pos:>4} gen={gen:>3} tok={best:>6} {tok:?} dt={dt:.3}s top5={top:?} lse={logsum:.2} top5sum={top5_sum:.2}",
        );

        let t1 = std::time::Instant::now();
        input = Tensor::new(&[best as u32], &device)?.unsqueeze(0)?;
        logits = model.forward(&input, pos)?;
        if gen % 16 == 0 {
            eprintln!("[step {gen}: forward {:.3}s]", t1.elapsed().as_secs_f32());
        }
        pos += 1;
        gen += 1;
    }
    Ok(())
}
