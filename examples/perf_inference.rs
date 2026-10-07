//! End-to-end CPU inference timing and full-logit capture on real GGUFs.
//! Usage: perf_inference MODEL TOKENIZER PROMPT TOKENS DECODE REPEATS OUTPUT THREADS
//! Timings exclude tokenization and logit file I/O. Each repeat starts with
//! an empty cache; decode uses the model's greedy tokens, never fixed routes.

use std::io::{Cursor, Write};
use std::sync::Arc;
use std::time::Instant;

use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() == 8,
        "usage: perf_inference MODEL TOKENIZER PROMPT TOKENS DECODE REPEATS OUTPUT THREADS"
    );
    let tokens: usize = args[3].parse()?;
    let decode: usize = args[4].parse()?;
    let repeats: usize = args[5].parse()?;
    let threads: usize = args[7].parse()?;
    anyhow::ensure!(
        tokens > 0 && repeats > 0 && threads > 0,
        "counts must be positive"
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let tokenizer =
        tokenizers::Tokenizer::from_file(&args[1]).map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompt = std::fs::read_to_string(&args[2])?;
    let encoded = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    anyhow::ensure!(!encoded.get_ids().is_empty(), "empty prompt");
    let ids: Vec<u32> = encoded
        .get_ids()
        .iter()
        .copied()
        .cycle()
        .take(tokens)
        .collect();
    let file = std::fs::File::open(&args[0])?;
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file)? });
    let mut cursor = Cursor::new(&mmap[..]);
    let content = joshua::gguf_ext::read_header(&mut cursor)?.to_candle_content()?;
    let dev = Device::Cpu;
    let start = Instant::now();
    let mut model = QuantizedModel::from_gguf_mmap(
        content,
        &mut cursor,
        &dev,
        Some(Arc::clone(&mmap)),
        None,
        tokens + decode + 16,
    )?;
    println!(
        "load_s={:.6} threads={threads} prefill={tokens} decode={decode}",
        start.elapsed().as_secs_f64()
    );
    let input = Tensor::new(ids.as_slice(), &dev)?.reshape((1, tokens))?;
    let mut output = std::io::BufWriter::new(std::fs::File::create(&args[6])?);
    for repeat in 0..repeats {
        anyhow::ensure!(model.clear_kv_cache(), "model cannot clear its cache");
        let start = Instant::now();
        let mut logits = model.forward(&input, 0)?;
        let prefill_s = start.elapsed().as_secs_f64();
        let mut captures = vec![logits.flatten_all()?.to_vec1::<f32>()?];
        let mut generated = Vec::with_capacity(decode);
        let mut decode_s = 0.0;
        for step in 0..decode {
            let start = Instant::now();
            let next = logits
                .argmax(candle_core::D::Minus1)?
                .flatten_all()?
                .to_vec1::<u32>()?[0];
            generated.push(next);
            let input = Tensor::new(&[next], &dev)?.reshape((1, 1))?;
            logits = model.forward(&input, tokens + step)?;
            decode_s += start.elapsed().as_secs_f64();
            captures.push(logits.flatten_all()?.to_vec1::<f32>()?);
        }
        println!("repeat={repeat} prefill_s={prefill_s:.6} decode_s={decode_s:.6} prefill_tps={:.3} decode_tps={:.3} ids={generated:?}",
            tokens as f64 / prefill_s, decode as f64 / decode_s);
        for capture in captures {
            for value in capture {
                output.write_all(&value.to_le_bytes())?;
            }
        }
    }
    output.flush()?;
    Ok(())
}
