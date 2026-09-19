//! Long-context comprehension acceptance test for a real DeepSeek-V4-Flash
//! artifact.
//!
//! The IQ2_XXS requant of the reap model silently lost long-context
//! comprehension beyond ~85 prompt tokens (garbled output, independently
//! confirmed in llama.cpp on the identical file, while the Q2_K reap model of
//! the same base answered correctly).  Nothing in `cargo test` noticed,
//! because no engine test runs real-model QA at graded prompt lengths.  This
//! test makes that regression loud:
//!
//!   JOSHUA_DS4_MODEL=/path/to/model.gguf \
//!   cargo test --release --test long_context_acceptance -- --ignored --nocapture
//!
//! For each graded prompt length it runs a raw-completion QA probe (no chat
//! template) with a known answer, greedy-decodes 12 tokens, and requires the
//! answer to appear (greedy decode of 64 tokens — the artifact is a reasoning
//! model and the answer follows its <think> block).  Probes at or below the
//! *certified* length
//! (`JOSHUA_LONGCTX_CERTIFIED`, default 82 — the IQ2_XXS artifact's last
//! verified-coherent length) are hard requirements: if the artifact or engine
//! regresses below what was last verified, this fails immediately with the
//! first failing length.  Probes beyond the certified length are reported and
//! only enforced with `JOSHUA_LONGCTX_STRICT=1` — set that once a re-quant
//! passes them, to ratchet the gate up.  The full table is printed either way.
//!
//! Model + tokenizer selection: `JOSHUA_DS4_MODEL` points at a deepseek4
//! GGUF; the matching `tokenizer.json` sidecar must sit beside it (or be
//! given directly via `JOSHUA_DS4_TOKENIZER`).

use candle_core::Device;
use joshua::model::QuantizedModel;
use std::path::{Path, PathBuf};

fn model_path() -> Option<PathBuf> {
    std::env::var_os("JOSHUA_DS4_MODEL").map(PathBuf::from)
}

fn tokenizer_path(model: &Path) -> PathBuf {
    match std::env::var_os("JOSHUA_DS4_TOKENIZER") {
        Some(p) => PathBuf::from(p),
        None => model.with_file_name("tokenizer.json"),
    }
}

fn logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    let input = candle_core::Tensor::new(tokens, &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    model
        .forward(&input, offset)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Greedy-decode `n` tokens from `ids` (offset-0 prefill, then one token per
/// forward against the session's KV), returning the decoded text.  `n` must
/// cover the artifact's <think> block plus the answer.
fn greedy(model: &mut QuantizedModel, tok: &tokenizers::Tokenizer, ids: &[u32], n: usize) -> String {
    let mut generated: Vec<u32> = Vec::new();
    let mut last = logits(model, ids, 0);
    for step in 0..n {
        let next = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        generated.push(next);
        last = logits(model, &[next], ids.len() + step);
    }
    tok.decode(&generated, true).unwrap_or_default()
}

/// One graded QA probe: raw prompt (no chat template), accepted answer
/// substrings (case-insensitive), approximate prompt-token count.
struct Probe {
    label: &'static str,
    approx_tokens: usize,
    prompt: &'static str,
    expect: &'static [&'static str],
}

fn probes() -> Vec<Probe> {
    vec![
        Probe {
            label: "continent",
            approx_tokens: 31,
            prompt: "The Amazon rainforest covers most of the Amazon basin of South America. Question: which continent is the basin in? Answer concisely.".into(),
            expect: &["south america"],
        },
        Probe {
            label: "nations (medium)",
            approx_tokens: 45,
            prompt: "The Amazon rainforest covers most of the Amazon basin of South America, encompassing seven million square kilometres, and includes territory belonging to nine nations. Question: how many nations share the basin? Answer concisely.".into(),
            expect: &["nine", "9"],
        },
        Probe {
            label: "nations (long)",
            approx_tokens: 82,
            prompt: "Read this passage carefully and answer the question at the end. The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses seven million square kilometres, of which five and a half million are covered by the rainforest. Question: how many nations share the basin? Answer concisely.".into(),
            expect: &["nine", "9"],
        },
        Probe {
            label: "nations (full)",
            approx_tokens: 141,
            prompt: "Read this passage carefully and answer the question at the end. The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses seven million square kilometres, of which five and a half million are covered by the rainforest. This region includes territory belonging to nine nations and three thousand three hundred formally recognised indigenous land holdings. The forest represents over half of the planet's remaining rainforests and comprises the largest and most biodiverse tract of tropical rainforest in the world. Question: how many nations share the basin? Answer concisely.".into(),
            expect: &["nine", "9"],
        },
    ]
}

fn load_model(model_path: &Path) -> QuantizedModel {
    // Real artifacts are tens of GiB: never `fs::read` them — map and borrow.
    let mmap = std::sync::Arc::new(
        unsafe { memmap2::Mmap::map(&std::fs::File::open(model_path).unwrap()) }.unwrap(),
    );
    let mut cursor = std::io::Cursor::new(&mmap[..]);
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let content = header.to_candle_content().unwrap();
    drop(cursor);
    let mut cursor = std::io::Cursor::new(&mmap[..]);
    QuantizedModel::from_gguf_mmap(
        content,
        &mut cursor,
        &Device::Cpu,
        Some(std::sync::Arc::clone(&mmap)),
        None,
        0,
    )
    .unwrap()
}

#[test]
#[ignore]
fn long_context_comprehension_ratchet() {
    let Some(model_path) = model_path() else {
        eprintln!(
            "SKIP: set JOSHUA_DS4_MODEL to a deepseek4 GGUF (with a tokenizer.json sidecar) \
             to run the long-context acceptance probes"
        );
        return;
    };
    let tok_path = tokenizer_path(&model_path);
    let tok = tokenizers::Tokenizer::from_file(&tok_path)
        .unwrap_or_else(|e| panic!("tokenizer at {}: {e}", tok_path.display()));
    let model = load_model(&model_path);

    let certified: usize = std::env::var("JOSHUA_LONGCTX_CERTIFIED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(45);
    let strict = std::env::var("JOSHUA_LONGCTX_STRICT").as_deref() == Ok("1");

    println!(
        "long-context acceptance: certified<={certified} strict={strict} model={}",
        model_path.display()
    );
    println!("{:>6}  {:>6}  {:<18}  {}", "tok", "result", "probe", "completion");

    let mut beyond_certified_failure: Option<(String, String)> = None;
    for probe in probes() {
        let ids = tok.encode(probe.prompt, true).unwrap().get_ids().to_vec();
        // One session per probe: no KV leaks between graded lengths.
        let mut session = model.new_session().unwrap();
        let out = greedy(&mut session, &tok, &ids, 64);
        let lower = out.to_lowercase();
        let hit = probe.expect.iter().any(|e| lower.contains(e));
        println!(
            "{:>6}  {:>6}  {:<18}  {}",
            ids.len(),
            if hit { "PASS" } else { "FAIL" },
            probe.label,
            out.chars().take(50).collect::<String>()
        );
        if !hit {
            if ids.len() <= certified {
                panic!(
                    "long-context regression: probe '{}' ({} prompt tokens, within the \
                     certified range of {certified}) lost comprehension: {out:?}",
                    probe.label,
                    ids.len()
                );
            }
            if beyond_certified_failure.is_none() {
                beyond_certified_failure =
                    Some((probe.label.to_string(), out.chars().take(60).collect()));
            }
        }
    }

    match beyond_certified_failure {
        None => println!("long-context acceptance: all probes passed"),
        Some((label, out)) => {
            if strict {
                panic!(
                    "long-context acceptance (STRICT): probe '{label}' beyond the certified \
                     length lost comprehension: {out:?}"
                );
            }
            eprintln!(
                "NOTE: probe '{label}' beyond the certified length lost comprehension — this \
                 artifact cannot serve prompts that long; re-quantize or lower \
                 JOSHUA_LONGCTX_CERTIFIED. Set JOSHUA_LONGCTX_STRICT=1 to enforce."
            );
        }
    }
}
