//! DeepSeek-V4.1 (`deepseek41`), served by the `deepseek4` loader: the
//! lagged hyper-connection mix with no learned head, KV compressed on a few
//! source layers and shared by the layers after them, index keys derived
//! from the shared latent, no per-head q norm, and engram n-gram tables.

mod common;

use candle_core::{Device, Tensor};
use common::{assert_close, load_model, logits};
use joshua::model::{Architecture, QuantizedModel};
use joshua::placement::{EngramResidency, EngramTier};

const TOKENS: [u32; 14] = [1, 4, 2, 7, 5, 9, 3, 6, 8, 4, 1, 12, 13, 5];

#[test]
fn router_dimensions_are_checked_on_every_load_path() {
    for v41 in [false, true] {
        for (router_shape, router_bias_shape, tensor) in [
            (Some([9, 256]), None, "ffn_gate_inp.weight"),
            (Some([7, 256]), None, "ffn_gate_inp.weight"),
            (Some([8, 255]), None, "ffn_gate_inp.weight"),
            (None, Some(&[7][..]), "exp_probs_b.bias"),
            (None, Some(&[9][..]), "exp_probs_b.bias"),
            (None, Some(&[2, 4][..]), "exp_probs_b.bias"),
        ] {
            let dir = common::model_dir("router-dimensions");
            let path = dir.join("model.gguf");
            common::write_tiny_deepseek4_gguf_opts(&path, common::TinyDeepseek4Opts {
                v41, router_shape, router_bias_shape, ..Default::default()
            });
            for mmap in [false, true] {
                let error = common::try_load_model(&path, mmap).err().expect("invalid router must fail to load");
                let message = error.to_string();
                assert!(message.contains(tensor) && message.contains("n_expert=8") && message.contains("got"), "{message}");
            }
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
}

#[test]
fn non_finite_routing_fails_with_layer_and_token_context() {
    for v41 in [false, true] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let dir = common::model_dir("router-nonfinite");
            let path = dir.join("model.gguf");
            common::write_tiny_deepseek4_gguf_opts(&path, common::TinyDeepseek4Opts {
                v41, router_nonfinite: Some(value), ..Default::default()
            });
            for mmap in [false, true] {
                for tokens in [&TOKENS[..1], &TOKENS[..3]] {
                    let input = Tensor::new(tokens, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
                    for all_rows in [false, true] {
                        let mut model = load_model(&path, mmap);
                        let result = if all_rows {
                            model.forward_all_logits(&input, 0)
                        } else {
                            model.forward(&input, 0)
                        };
                        let message = result.unwrap_err().to_string();
                        assert!(message.contains("non-finite routing weight at layer 1, token 0, slot 0"), "{message}");
                    }
                }
            }
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
}

#[test]
fn ref41_dtype_mix_is_finite_and_preserves_prefill_decode_parity() {
    let dir = common::model_dir("ref41-dtype-mix");
    let path = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_opts(&path, common::TinyDeepseek4Opts {
        v41: true, v41_tied_head: true, ref41_dtype_mix: true, ..Default::default()
    });
    for mmap in [false, true] {
        let mut decode = load_model(&path, mmap);
        let mut expected_rows = Vec::new();
        for position in 0..TOKENS.len() {
            let got = logits(&mut decode, &TOKENS[position..=position], position);
            assert!(got.iter().all(|value| value.is_finite()));
            let mut prefill = load_model(&path, mmap);
            let expected = logits(&mut prefill, &TOKENS[..=position], 0);
            assert_close(&got, &expected, 1e-5, "ref41 dtype mix");
            expected_rows.push(got);
        }
        let mut verification = load_model(&path, mmap);
        for (offset, end) in [(0, 5), (5, TOKENS.len())] {
            let input = Tensor::new(&TOKENS[offset..end], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
            let output = verification.forward_all_logits(&input, offset).unwrap();
            assert_eq!(output.dims(), [1, end - offset, 16]);
            let rows = output.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
            for (actual, expected) in rows.iter().zip(&expected_rows[offset..end]) {
                assert_close(actual, expected, 1e-5, "ref41 all-row logits");
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

fn fixture(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = common::model_dir(name);
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek41_gguf(&model);
    (dir, model)
}

#[test]
fn deepseek41_is_a_supported_architecture() {
    let arch = Architecture::from_name("deepseek41").expect("deepseek41 supported");
    assert!(arch.is_native() && arch.is_deepseek4());
    assert_eq!(arch.display_name(), "DeepSeek-V4.1");
}

/// Logits at every position match an independent float64 NumPy
/// transcription of llama.cpp's `src/models/deepseek41.cpp`
/// (`tests/data/deepseek41_reference_logits.txt`).  The 14-token prompt is
/// longer than the 8-token window and the top-k budget of 2, so every
/// compressed-attention path (gated ratio-2 pooling, gateless ratio 1,
/// shared streams, shared index keys, top-k selection) and the engram are
/// exercised; each alone moves the logits far past the tolerance.
#[test]
fn deepseek41_matches_llama_cpp_reference_at_every_position() {
    let want: Vec<Vec<f32>> = include_str!("data/deepseek41_reference_logits.txt")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect();
    assert_eq!(want.len(), TOKENS.len());

    let (dir, model) = fixture("deepseek41-ref");
    for mmap in [true, false] {
        let mut m = load_model(&model, mmap);
        for (pos, (&tok, w)) in TOKENS.iter().zip(&want).enumerate() {
            let got = logits(&mut m, &[tok], pos);
            assert_close(
                &got,
                w,
                1e-5,
                &format!("deepseek41 (mmap={mmap}) position {pos}"),
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[cfg(any(feature = "vulkan", feature = "sycl"))]
#[test]
fn deepseek41_accelerators_match_independent_float64_reference() {
    let want: Vec<Vec<f32>> = include_str!("data/deepseek41_reference_logits.txt")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect();
    let mut devices = Vec::new();
    #[cfg(feature = "vulkan")]
    if let Ok(dev) = Device::new_vulkan(0) {
        devices.push(dev);
    }
    #[cfg(feature = "sycl")]
    if let Ok(dev) = Device::new_sycl(0) {
        devices.push(dev);
    }
    let (dir, path) = fixture("deepseek41-accelerator-f64-ref");
    for dev in devices {
        let mut file = std::fs::File::open(&path).unwrap();
        let content = joshua::gguf_ext::read_header(&mut file)
            .unwrap()
            .to_candle_content()
            .unwrap();
        let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
        let mut model = QuantizedModel::from_gguf_mmap_placed(
            content,
            &mut file,
            &dev,
            &Device::Cpu,
            Some(mmap),
            None,
            0,
            None,
            joshua::EngramTier::Auto,
        )
        .unwrap();
        for (pos, (&token, expected)) in TOKENS.iter().zip(&want).enumerate() {
            let input = Tensor::new(&[[token]], &dev).unwrap();
            let got = model
                .forward(&input, pos)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_close(&got, expected, 1e-5, &format!("{dev:?} decode {pos}"));
        }
        assert!(model.clear_kv_cache());
        for (offset, end) in [(0, 5), (5, TOKENS.len())] {
            let input = Tensor::new(&TOKENS[offset..end], &dev)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let got = model
                .forward_all_logits(&input, offset)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let expected: Vec<f32> = want[offset..end].iter().flatten().copied().collect();
            assert_close(
                &got,
                &expected,
                1e-5,
                &format!("{dev:?} prefill {offset}..{end}"),
            );
        }
    }
    std::fs::remove_dir_all(dir).ok();
}

/// A whole-prompt prefill, a chunked prefill and the layer-streamed prefill
/// (which carries the lagged mix between layers inside the stream tensor)
/// agree with token-by-token decode, and decoding continues identically.
#[test]
fn deepseek41_prefill_paths_agree() {
    use joshua::stream_prefill::Chunk;
    let (dir, model) = fixture("deepseek41-prefill");

    let mut step = load_model(&model, true);
    let mut want = Vec::new();
    for (pos, &tok) in TOKENS.iter().enumerate() {
        want = logits(&mut step, &[tok], pos);
    }

    let mut whole = load_model(&model, true);
    assert_close(
        &logits(&mut whole, &TOKENS, 0),
        &want,
        1e-4,
        "whole prefill",
    );

    let mut chunked = load_model(&model, true);
    let _ = logits(&mut chunked, &TOKENS[..5], 0);
    assert_close(
        &logits(&mut chunked, &TOKENS[5..], 5),
        &want,
        1e-4,
        "chunked prefill",
    );

    let mut streamed = load_model(&model, true);
    let chunks = [
        Chunk {
            tokens: &TOKENS[..4],
            pos: 0,
        },
        Chunk {
            tokens: &TOKENS[4..9],
            pos: 4,
        },
        Chunk {
            tokens: &TOKENS[9..],
            pos: 9,
        },
    ];
    let got: Vec<f32> = streamed
        .prefill_streamed(&chunks, &Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_close(&got, &want, 1e-4, "layer-streamed prefill");

    let next = TOKENS.len();
    let a = logits(&mut whole, &[3], next);
    assert_close(
        &logits(&mut streamed, &[3], next),
        &a,
        1e-4,
        "decode after streamed prefill",
    );
    assert_close(
        &logits(&mut step, &[3], next),
        &a,
        1e-4,
        "decode after incremental",
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The batched decode (per-sequence caches and engram histories, one shared
/// MoE pass) matches each sequence decoded alone.
#[test]
fn deepseek41_batched_decode_matches_single_sequences() {
    let (dir, model) = fixture("deepseek41-batched");
    let (a, b) = ([1u32, 4, 2, 7], [9u32, 3, 6, 8]);
    let mut single = |toks: &[u32]| {
        let mut m = load_model(&model, true);
        toks.iter()
            .enumerate()
            .map(|(p, &t)| logits(&mut m, &[t], p))
            .collect::<Vec<_>>()
    };
    let (want_a, want_b) = (single(&a), single(&b));

    let mut m = load_model(&model, true);
    let QuantizedModel::DeepSeek4(w) = &mut m else {
        panic!("deepseek41 loads through the deepseek4 loader");
    };
    for p in 0..a.len() {
        let ta = Tensor::new(&a[p..p + 1], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let tb = Tensor::new(&b[p..p + 1], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let got = w.forward_sequences(&[(&ta, p), (&tb, p)]).unwrap();
        assert_close(
            &got[0],
            &want_a[p],
            1e-4,
            &format!("sequence a, position {p}"),
        );
        assert_close(
            &got[1],
            &want_b[p],
            1e-4,
            &format!("sequence b, position {p}"),
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Clearing the cache also forgets the engram's token history.
#[test]
fn deepseek41_clear_resets_all_state() {
    let (dir, model) = fixture("deepseek41-clear");
    let mut m = load_model(&model, true);
    let fresh = logits(&mut m, &TOKENS[..6], 0);
    let _ = logits(&mut m, &[8, 9], 6);
    assert!(m.clear_kv_cache());
    assert_close(
        &logits(&mut m, &TOKENS[..6], 0),
        &fresh,
        1e-5,
        "after clear",
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn deepseek41_runs_through_the_engine() {
    use joshua::types::GenerationOptions;
    let (dir, model) = fixture("deepseek41-engine");
    let engine = joshua::Engine::new(&model).expect("engine must load deepseek41 GGUFs");
    let opts = GenerationOptions {
        max_tokens: 4,
        temperature: 0.0,
        ..Default::default()
    };
    let (_, usage, _, _) = engine
        .complete_raw("hello world", &opts)
        .expect("generation");
    assert!(usage.completion_tokens > 0, "{usage:?}");
    std::fs::remove_dir_all(&dir).ok();
}

/// A GGUF cut short inside its last tensor is refused at load (#175), even
/// though the engine's page-rounded mapping would read the missing bytes as
/// zeros: the layout is checked against the file's real length.
#[test]
fn engine_refuses_a_truncated_gguf() {
    let (dir, model) = fixture("deepseek41-truncated");
    // Cut 8 bytes into the tensor that ends last (the writer pads after it).
    let h = joshua::gguf_ext::read_header(&mut std::fs::File::open(&model).unwrap()).unwrap();
    let end = h
        .tensors
        .values()
        .map(|i| {
            let size = joshua::gguf_ext::type_size_bytes(i.dtype, i.elem_count()).unwrap();
            h.tensor_data_offset + i.offset + size as u64
        })
        .max()
        .unwrap();
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(&model)
        .unwrap();
    f.set_len(end - 8).unwrap();
    drop(f);
    let msg = match joshua::Engine::new(&model) {
        Ok(_) => panic!("a truncated GGUF must not load"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("past the end"), "{msg}");
    std::fs::remove_dir_all(&dir).ok();
}

/// Load the tiny V4.1 fixture through `from_gguf_mmap_placed` with an
/// explicit engram tier, so the placement policy (a separate one from the
/// routed-expert cache) can be observed in isolation.
fn load_engram_tier(path: &std::path::Path, tier: joshua::EngramTier) -> QuantizedModel {
    let bytes = std::fs::read(path).unwrap();
    let content = joshua::gguf_ext::read_header(&mut std::io::Cursor::new(&bytes[..]))
        .unwrap()
        .to_candle_content()
        .unwrap();
    let mmap = std::sync::Arc::new(
        unsafe { memmap2::Mmap::map(&std::fs::File::open(path).unwrap()) }.unwrap(),
    );
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    QuantizedModel::from_gguf_mmap_placed(
        content,
        &mut cursor,
        &Device::Cpu,
        &Device::Cpu,
        Some(mmap),
        None,
        0,
        None,
        tier,
    )
    .unwrap()
}

/// The engram tables are resolved straight from the mapping under every tier
/// and their residency is reported separately from the routed-expert cache.
#[test]
fn engram_placement_is_reported_separately_and_resolves_from_mmap() {
    // Build the fixture once; the tiny model's engram range is mmap-backed.
    let (dir, model) = fixture("deepseek41-engram-tier");
    // Default (`auto`, no ample RAM probe on this host) and explicit `disk`
    // both pin the engram to storage (no promotion).
    for tier in [joshua::EngramTier::Auto, joshua::EngramTier::Disk] {
        let m = load_engram_tier(&model, tier);
        let report = m.engram_residency().expect("fixture has an engram layer");
        assert_eq!(report.layers, 1, "one engram layer in the tiny model");
        assert!(
            report.bytes_total > 0,
            "the mmap-borrowed table reports mapped bytes"
        );
        assert_eq!(
            report.tier,
            EngramResidency::ResolveFromMmap,
            "tier {tier:?} resolves to the mmap/disk residency"
        );
        drop(m);
    }
    // `ram` forces promotion (the mapped range is prefetched); the report
    // still resolves from the mapping but records the promoted tier.
    let m = load_engram_tier(&model, joshua::EngramTier::Ram);
    let report = m.engram_residency().expect("engram layer present");
    assert_eq!(
        report.tier,
        EngramResidency::PromoteToRam,
        "`ram` forces promote-to-RAM"
    );
    drop(m);
    std::fs::remove_dir_all(&dir).ok();
}

/// DeepSeek-V4.1-Flash ships **no `output.weight`**: the LM head is tied to
/// `token_embd`.  The other tests all use the fixture's own `output.weight`,
/// so without this one the tied path — the one the real GGUFs take — is
/// never exercised by a parity check.
///
/// The fixture writes a *distinct* `output.weight` (seed 2) from
/// `token_embd` (seed 1) precisely so the two heads can be told apart: if the
/// loader ignored the tie (or silently used the embedding when a real head
/// exists), these logits would coincide.
#[test]
fn deepseek41_tied_head_loads_and_differs_from_an_explicit_head() {
    let (dir_own, own) = fixture("deepseek41-own-head");

    let dir_tied = common::model_dir("deepseek41-tied-head");
    let tied = dir_tied.join("model.gguf");
    common::write_tiny_deepseek41_gguf_tied(&tied);

    for mmap in [true, false] {
        let mut with_head = load_model(&own, mmap);
        let mut without_head = load_model(&tied, mmap);

        let a = logits(&mut with_head, &TOKENS[..1], 0);
        let b = logits(&mut without_head, &TOKENS[..1], 0);

        assert!(
            b.iter().all(|v| v.is_finite()),
            "tied-head logits must be finite (mmap={mmap}): {b:?}"
        );

        // The tied head projects through token_embd (seed 1); the explicit
        // head uses seed 2.  Different weights, so the logits must differ.
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "tied head must not silently equal the explicit output.weight \
             (mmap={mmap}, max |diff| = {max_diff:e})"
        );

        // Decode must also stay finite and deterministic across a run.
        let mut prev = b;
        for pos in 1..4 {
            let cur = logits(&mut without_head, &TOKENS[pos..=pos], pos);
            assert!(
                cur.iter().all(|v| v.is_finite()),
                "tied-head decode went non-finite at position {pos} (mmap={mmap}): {cur:?}"
            );
            let drift = cur
                .iter()
                .zip(&prev)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                drift.is_finite(),
                "tied-head decode produced a non-finite delta at position {pos}"
            );
            prev = cur;
        }
    }

    std::fs::remove_dir_all(&dir_own).ok();
    std::fs::remove_dir_all(&dir_tied).ok();
}
