//! Native (pure-Rust) validation for the `deepseek4` quantized loader with the
//! dtypes candle cannot represent: IQ2_XXS routed experts and I32 routed-id
//! tables.  These run on the default `cargo test` — no llama.cpp, no network.

mod common;

use candle_core::{Device, Tensor};
use common::logits;
use joshua::model::{Architecture, QuantizedModel};

#[cfg(feature = "distributed")]
fn cluster_sessions(
    count: usize,
) -> Vec<std::sync::Arc<joshua::distributed::session::ClusterSession>> {
    use joshua::distributed::{session::ClusterSession, tcp::TcpConfig};
    let listeners: Vec<_> = (0..count)
        .map(|_| std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect();
    let config = TcpConfig {
        peers: listeners.iter().map(|l| l.local_addr().unwrap()).collect(),
        timeout: std::time::Duration::from_secs(30),
    };
    drop(listeners);
    let session = uuid::Uuid::new_v4();
    let key = rand::random::<[u8; 32]>();
    (0..count)
        .map(|rank| {
            std::sync::Arc::new(
                ClusterSession::tcp(rank, session, &key, config.clone()).unwrap(),
            )
        })
        .collect()
}

#[cfg(feature = "distributed")]
#[test]
fn deepseek4_cluster_matches_single_node_prefill_decode() {
    use joshua::distributed::deepseek4::DeepSeekCluster;
    for (name, opts) in [
        ("q8", common::TinyDeepseek4Opts::default()),
        (
            "q2k-compressed",
            common::TinyDeepseek4Opts {
                compress: true,
                q2k_down: true,
                expert_width: Some(512),
                ..Default::default()
            },
        ),
    ] {
        let dir = common::model_dir(&format!("deepseek4-cluster-{name}"));
        let path = dir.join("model.gguf");
        common::write_tiny_deepseek4_gguf_opts(&path, opts);
        let mut reference = load(&path, true);
        let calls = [
            (&[1, 4, 2, 7, 5, 3, 8, 2, 6][..], 0),
            (&[3][..], 9),
            (&[8][..], 10),
            (&[2][..], 11),
        ];
        let expected: Vec<_> = calls
            .iter()
            .map(|(tokens, offset)| logits(&mut reference, tokens, *offset))
            .collect();
        for count in [1, 2] {
            let sessions = cluster_sessions(count);
            let outputs = std::thread::scope(|scope| {
                let workers: Vec<_> = sessions
                    .into_iter()
                    .map(|session| {
                        let path = &path;
                        let calls = &calls;
                        scope.spawn(move || {
                            let mut model = DeepSeekCluster::load(path, 32, session).unwrap();
                            let (local_bytes, full_bytes) = model.expert_shard_bytes();
                            assert!(local_bytes > 0);
                            assert_eq!(local_bytes * count, full_bytes);
                            calls
                                .iter()
                                .map(|(tokens, offset)| model.forward(tokens, *offset).unwrap())
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                workers
                    .into_iter()
                    .map(|w| w.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for rank in &outputs {
                assert_eq!(rank, &outputs[0], "ranks must produce identical logits");
                for (step, (actual, expected)) in rank.iter().zip(&expected).enumerate() {
                    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
                        let tolerance = 1e-3 * e.abs().max(1.0);
                        assert!((a - e).abs() <= tolerance,
                            "{name}, {count} ranks, step {step}, logit {i}: {a} != {e} (tol {tolerance})");
                    }
                }
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[cfg(feature = "distributed")]
#[test]
fn deepseek4_cluster_uses_rank_zero_routing_despite_peer_drift() {
    use joshua::distributed::deepseek4::DeepSeekCluster;
    let dir = common::model_dir("deepseek4-cluster-routing");
    let root_path = dir.join("root.gguf");
    let peer_path = dir.join("peer.gguf");
    common::write_tiny_deepseek4_gguf(&root_path);
    let mut bytes = std::fs::read(&root_path).unwrap();
    let header = joshua::gguf_ext::read_header(&mut std::io::Cursor::new(&bytes)).unwrap();
    // Deliberately alter only the peer's routers to amplify numerical drift.
    // Real deployments must verify identical model checksums out of band.
    for (name, tensor) in &header.tensors {
        if name.ends_with(".ffn_gate_inp.weight") || name.ends_with(".exp_probs_b.bias") {
            assert_eq!(tensor.dtype, 0);
            let start = (header.tensor_data_offset + tensor.offset) as usize;
            let len = tensor.dims.iter().product::<usize>() * 4;
            bytes[start..start + len].fill(0);
            if name.ends_with(".exp_probs_b.bias") {
                bytes[start..start + 4].copy_from_slice(&100f32.to_le_bytes());
            }
        }
    }
    std::fs::write(&peer_path, bytes).unwrap();
    let tokens = [1, 4, 2, 7, 5];
    let expected = logits(&mut load(&root_path, true), &tokens, 0);
    let perturbed = logits(&mut load(&peer_path, true), &tokens, 0);
    assert!(expected
        .iter()
        .zip(&perturbed)
        .any(|(a, b)| (a - b).abs() > 1e-4));
    let outputs = std::thread::scope(|scope| {
        let workers: Vec<_> = cluster_sessions(2)
            .into_iter()
            .enumerate()
            .map(|(rank, session)| {
                let path = if rank == 0 { &root_path } else { &peer_path };
                let tokens = &tokens;
                scope.spawn(move || {
                    DeepSeekCluster::load(path, 32, session)
                        .unwrap()
                        .forward(tokens, 0)
                        .unwrap()
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(outputs[0], outputs[1]);
    for (&actual, &expected) in outputs[0].iter().zip(&expected) {
        assert!((actual - expected).abs() <= 1e-3 * expected.abs().max(1.0));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "distributed")]
#[test]
fn deepseek4_cluster_cli_generates_like_single_node() {
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, Instant};

    struct ChildGuard(Option<Child>);
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            if let Some(child) = self.0.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    let dir = common::model_dir("deepseek4-cluster-cli");
    let path = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&path);
    let mut reference = load(&path, true);
    let mut next = logits(&mut reference, &[1, 4], 0);
    assert!(next.iter().all(|v| v.is_finite()));
    next = logits(&mut reference, &[2], 2);
    let mut expected = Vec::new();
    for step in 0..3 {
        let best =
            (1..next.len()).fold(0, |best, i| if next[i] > next[best] { i } else { best }) as u32;
        expected.push(best);
        if step < 2 {
            next = logits(&mut reference, &[best], 3 + step);
        }
    }

    for count in [1, 2] {
        let listeners: Vec<_> = (0..count)
            .map(|_| std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap())
            .collect();
        let peers = listeners
            .iter()
            .map(|listener| listener.local_addr().unwrap().to_string())
            .collect::<Vec<_>>()
            .join(",");
        drop(listeners);
        let session = uuid::Uuid::new_v4().to_string();
        let key: String = rand::random::<[u8; 32]>()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let mut children: Vec<_> = (0..count)
            .map(|rank| {
                ChildGuard(Some(
                    Command::new(env!("CARGO_BIN_EXE_joshua"))
                        .args(["cluster-run", "--model"])
                        .arg(&path)
                        .args([
                            "--tokens",
                            "1,4,2",
                            "--n-ctx",
                            "32",
                            "--max-tokens",
                            "3",
                            "--prefill-chunk",
                            "2",
                            "--rank",
                            &rank.to_string(),
                            "--world-size",
                            &count.to_string(),
                            "--session",
                            &session,
                            "--transport",
                            "tcp",
                            "--peers",
                            &peers,
                            "--timeout-seconds",
                            "30",
                        ])
                        .env("JOSHUA_CLUSTER_KEY", &key)
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn()
                        .unwrap(),
                ))
            })
            .collect();
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let complete = children
                .iter_mut()
                .all(|child| child.0.as_mut().unwrap().try_wait().unwrap().is_some());
            if complete {
                break;
            }
            assert!(Instant::now() < deadline, "cluster CLI processes timed out");
            std::thread::sleep(Duration::from_millis(20));
        }
        for (rank, child) in children.iter_mut().enumerate() {
            let output = child.0.take().unwrap().wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "{count} ranks, rank {rank}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stderr).contains("routed_expert_bytes="));
            if rank == 0 {
                let actual: Vec<u32> = serde_json::from_slice(&output.stdout).unwrap();
                assert_eq!(actual, expected, "{count}-rank CLI generation differs");
            } else {
                assert!(output.stdout.is_empty(), "only rank zero prints output");
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "distributed")]
#[test]
fn deepseek4_cluster_mismatched_inputs_fail_and_poison_every_rank() {
    use joshua::distributed::deepseek4::DeepSeekCluster;
    let dir = common::model_dir("deepseek4-cluster-input-mismatch");
    let path = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&path);
    std::thread::scope(|scope| {
        let workers: Vec<_> = cluster_sessions(2)
            .into_iter()
            .enumerate()
            .map(|(rank, session)| {
                let path = &path;
                scope.spawn(move || {
                    let mut model = DeepSeekCluster::load(path, 32, session).unwrap();
                    let tokens = [1, 4 + rank as u32];
                    let error = model.forward(&tokens, 0).unwrap_err();
                    assert!(error.to_string().contains("disagree"), "{error}");
                    let error = model.forward(&[1, 4], 0).unwrap_err();
                    assert!(error.to_string().contains("session failed"), "{error}");
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "distributed")]
#[test]
fn deepseek4_cluster_invalid_inputs_fail_and_poison_session() {
    use joshua::distributed::deepseek4::DeepSeekCluster;
    let dir = common::model_dir("deepseek4-cluster-invalid-input");
    let path = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&path);
    for (tokens, offset, message) in [
        (vec![16], 0, "out-of-vocabulary"),
        (vec![1], 1, "offset is not contiguous"),
        (vec![1; 33], 0, "context exhausted"),
        (Vec::new(), 0, "must not be empty"),
    ] {
        let session = cluster_sessions(1).pop().unwrap();
        let mut model = DeepSeekCluster::load(&path, 32, session).unwrap();
        let error = model.forward(&tokens, offset).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        let error = model.forward(&[1], 0).unwrap_err();
        assert!(error.to_string().contains("session failed"), "{error}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "distributed")]
#[test]
fn deepseek4_cluster_rejects_more_ranks_than_down_blocks() {
    use joshua::distributed::deepseek4::DeepSeekCluster;
    let dir = common::model_dir("deepseek4-cluster-alignment");
    let path = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_q2k_down(&path);
    std::thread::scope(|scope| {
        let workers: Vec<_> = cluster_sessions(2)
            .into_iter()
            .map(|session| {
                let path = &path;
                scope.spawn(move || {
                    let error = DeepSeekCluster::load(path, 32, session).err().unwrap();
                    assert!(
                        error.to_string().contains("more shards than input blocks"),
                        "{error}"
                    );
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    std::fs::remove_dir_all(dir).unwrap();
}

fn load(model: &std::path::Path, mmap: bool) -> QuantizedModel {
    common::load_model(model, mmap)
}

/// The mmap load path must wire up prefetch handles for every routed expert
/// (their weights live in the mapping), and the streamed path must not.
#[test]
fn deepseek4_mmap_path_wires_expert_prefetch_handles() {
    let dir = common::model_dir("deepseek4-prefetch");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let mmapped = load(&model, true);
    let (backed, total) = match &mmapped {
        QuantizedModel::DeepSeek4(w) => w.mmap_backed_experts(),
        _ => panic!("expected DeepSeek4 model"),
    };
    assert!(total > 0, "tiny model must have routed experts");
    assert_eq!(
        backed, total,
        "mmap path should borrow every routed expert ({backed}/{total})"
    );

    let streamed = load(&model, false);
    let (backed, total) = match &streamed {
        QuantizedModel::DeepSeek4(w) => w.mmap_backed_experts(),
        _ => panic!("expected DeepSeek4 model"),
    };
    assert_eq!(backed, 0, "streamed path has no mapping ({backed}/{total})");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn deepseek4_is_a_supported_architecture() {
    assert_eq!(
        Architecture::from_name("deepseek4"),
        Some(Architecture::DeepSeek4)
    );
}

/// The full loader path: IQ2_XXS experts (mmap-borrowed), I32 tid2eid table,
/// hash + regular MoE layers — produces finite, non-degenerate logits.
#[test]
fn deepseek4_loads_iq2xxs_and_i32_and_produces_finite_logits() {
    let dir = common::model_dir("deepseek4-load");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let mut m = load(&model, true);
    let out = logits(&mut m, &[1, 4, 2, 7, 5], 0);
    assert_eq!(out.len(), 16, "logits must cover the 16-token vocab");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "all logits must be finite: {out:?}"
    );
    let first = out[0];
    assert!(
        out.iter().any(|v| (v - first).abs() > 1e-6),
        "logits are degenerate"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The mmap path (blocks borrowed, decoded inside the matmul) and the
/// streamed path (whole tensor decoded to f32 at load) must agree.
#[test]
fn deepseek4_mmap_matches_streamed_path() {
    let dir = common::model_dir("deepseek4-mmap");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let mut mmapped = load(&model, true);
    let mut streamed = load(&model, false);
    let tokens = [1, 4, 2, 7, 5, 3, 8];
    let a = logits(&mut mmapped, &tokens, 0);
    let b = logits(&mut streamed, &tokens, 0);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let tol = 1e-3 * y.abs().max(1.0);
        assert!(
            (x - y).abs() <= tol,
            "logit {i}: mmap={x} streamed={y} (tol {tol})"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// Incremental decode must give the same next-token logits as prefill.
#[test]
fn deepseek4_prefill_matches_incremental_decode() {
    let dir = common::model_dir("deepseek4-incr");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let tokens = [1u32, 4, 2, 7, 5];
    let mut m = load(&model, true);
    let prefill = logits(&mut m, &tokens, 0);

    let mut m = load(&model, true);
    let mut last = vec![0f32; 16];
    for (i, &t) in tokens.iter().enumerate() {
        last = logits(&mut m, &[t], i);
    }
    for i in 0..16 {
        let tol = 1e-3 * prefill[i].abs().max(1.0);
        assert!(
            (last[i] - prefill[i]).abs() <= tol,
            "logit {i}: prefill={} incremental={}",
            prefill[i],
            last[i]
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// Wraps a `Cursor` whose reads always fail — simulates a header that
/// disappears between candle's parse and Joshua's raw re-read (truncation,
/// failing backend), so the re-read error path is exercised deterministically.
struct FailAllReads<R>(R);

impl<R: std::io::Read> std::io::Read for FailAllReads<R> {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "simulated truncated header",
        ))
    }
}

impl<R: std::io::Seek> std::io::Seek for FailAllReads<R> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.0.seek(pos)
    }
}

/// A CSA layer (`compress_ratios[0] = 4`) writes its compressor and indexer
/// caches with a scatter whose index is a broadcast view of the block
/// positions.  candle's `scatter` rejects non-contiguous index tensors, so
/// both writes must materialize a dense index first — otherwise generation
/// aborts the moment the first compressed block completes.
#[test]
fn deepseek4_compressed_layers_write_their_caches() {
    let dir = common::model_dir("deepseek4-compress");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_compress(&model);

    // 5 tokens = one full compressed block at ratio 4 (plus a remainder):
    // the prefill branch completes a block and scatters into `comp` and
    // `lid` — this is where the non-contiguous index used to abort.
    let tokens = [1u32, 4, 2, 7, 5];
    let mut m = load(&model, true);
    let out = logits(&mut m, &tokens, 0);
    assert!(out.iter().all(|l| l.is_finite()));

    // Incremental decode crosses the block boundary mid-stream (position 3
    // completes the first block) — the other scatter call site.
    let mut m = load(&model, true);
    for (i, &t) in tokens.iter().enumerate() {
        let l = logits(&mut m, &[t], i);
        assert!(
            l.iter().all(|v| v.is_finite()),
            "decode step {i} must produce finite logits"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// An `output.weight` shipped in a dtype candle cannot name (IQ2_XXS) must be
/// picked up via the raw header and used, not reported absent — which would
/// silently tie the output head to the input embeddings.
#[test]
fn deepseek4_iq2xxs_output_weight_is_used_not_tied() {
    let dir = common::model_dir("deepseek4-iq2xxs-output");
    let tied = dir.join("tied.gguf");
    let own = dir.join("own.gguf");
    common::write_tiny_deepseek4_gguf(&tied);
    common::write_tiny_deepseek4_gguf_iq2xxs_output(&own);

    let mut a = load(&tied, true);
    let mut b = load(&own, true);
    let tokens = [1u32, 4, 2, 7, 5];
    let la = logits(&mut a, &tokens, 0);
    let lb = logits(&mut b, &tokens, 0);
    assert!(
        la.iter().zip(&lb).any(|(x, y)| (x - y).abs() > 1e-3),
        "IQ2_XXS output.weight must change the logits (tied model: {la:?}, own head: {lb:?})"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A failure to re-read the raw header must surface as the load error, not be
/// swallowed into "no raw table" (which would later become a misleading
/// "cannot find tensor blk.N.ffn_gate_exps.weight").
#[test]
fn deepseek4_header_reread_failure_is_reported() {
    let dir = common::model_dir("deepseek4-bad-header");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let bytes = std::fs::read(&model).unwrap();
    let content = {
        let mut cursor = std::io::Cursor::new(&bytes[..]);
        let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
        header.to_candle_content().unwrap()
    };

    // The reader passes candle's parse but fails every read afterwards.
    let mut failing = FailAllReads(std::io::Cursor::new(bytes));
    let err = match QuantizedModel::from_gguf_mmap(content, &mut failing, &Device::Cpu, None, None, 0) {
        Err(e) => e,
        Ok(_) => panic!("from_gguf_mmap must fail when the raw header re-read fails"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("GGUF header"),
        "error should name the header re-read, got: {msg}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The engine must accept a GGUF whose dtypes candle cannot name — the
/// tolerant header is what makes the whole file reachable at all.
#[test]
fn deepseek4_engine_accepts_unknown_dtypes() {
    let dir = common::model_dir("deepseek4-engine");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let engine = joshua::Engine::new(&model).expect("engine must load IQ2_XXS GGUFs");
    // The API identifier is the file stem (documented contract); the
    // file's `general.name` ("DeepSeek-V4") is log-only.
    assert_eq!(engine.model_name(), "model");

    std::fs::remove_dir_all(&dir).ok();
}

/// Streamed (no mmap) loads must work for K-quant weights: the byte-size
/// lookup used to re-read the tensor data covers Q2_K..Q8_K, not just the
/// Q4_0..Q8_1 family.  Before the fix this aborted with "no size known for
/// GGUF dtype 12".
#[test]
fn deepseek4_streamed_load_handles_k_quant_weights() {
    let dir = common::model_dir("deepseek4-kquant-streamed");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_kquant(&model);

    let mut m = load(&model, true);
    let out = logits(&mut m, &[1, 4, 2, 7, 5], 0);
    assert_eq!(out.len(), 16);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "all logits must be finite: {out:?}"
    );
    let first = out[0];
    assert!(
        out.iter().any(|v| (v - first).abs() > 1e-6),
        "logits are degenerate"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The mmap path (Q4_K blocks borrowed and matmul'd in place) and the
/// streamed path (Q4_K decoded to f32 at load) must agree for K-quant
/// weights, like they do for the IQ2_XXS experts.
#[test]
fn deepseek4_kquant_mmap_matches_streamed_path() {
    let dir = common::model_dir("deepseek4-kquant-parity");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_kquant(&model);

    let mut mmapped = load(&model, true);
    let mut streamed = load(&model, false);
    let tokens = [1, 4, 2, 7, 5, 3, 8];
    let a = logits(&mut mmapped, &tokens, 0);
    let b = logits(&mut streamed, &tokens, 0);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        // The two paths are genuinely different computations — Q4_K blocks
        // dequantised block-by-block inside the fused matmul vs a plain f32
        // matmul over fully-dequantised weights — so they accumulate in
        // different orders.  On aarch64 the NEON fused kernel measures up to
        // ~0.016 absolute apart from the f32 path on this model (logits in
        // ±1.2).  A bound of `0.02 + 1%` admits that with margin while still
        // catching a real regression (transposed weight, dropped expert,
        // wrong kernel), which shifts logits by O(0.1–1.0) — 5–60× the noise
        // floor.
        let tol = 0.02 + 1e-2 * y.abs();
        assert!(
            (x - y).abs() <= tol,
            "logit {i}: mmap={x} streamed={y} (tol {tol})"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// `ModelWeights::from_gguf` is the public streamed entry point without a raw
/// header; it must fall back to candle's reader instead of erroring with "no
/// raw header for `token_embd.weight`".  The model is written with only
/// candle-nameable dtypes so every tensor is reachable that way.
#[test]
fn deepseek4_from_gguf_without_raw_header_loads() {
    let dir = common::model_dir("deepseek4-fromgguf");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_candle_only(&model);

    let bytes = std::fs::read(&model).unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let content = header.to_candle_content().unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let mut m = joshua::quantized_deepseek4::ModelWeights::from_gguf(
        content,
        &mut cursor,
        &Device::Cpu,
    )
    .expect("from_gguf (no raw header) must load candle-nameable models");

    let input = Tensor::new(&[1u32, 4, 2, 7, 5][..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let out: Vec<f32> = m
        .forward(&input, 0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out.len(), 16);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "all logits must be finite: {out:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The speculative next-step expert prefetch must track routing: empty
/// before the first pass, seeded by prefill's final row, refreshed by every
/// decode step — and every recorded id must be a valid expert index.
///
/// The prediction itself only fires `MADV_WILLNEED` (unobservable here);
/// what the test pins down is the bookkeeping the prediction is derived
/// from, plus that recording routing does not disturb outputs: two freshly
/// loaded models must produce identical logits with identical recorded
/// state.
#[test]
fn deepseek4_speculative_routing_state_tracks_forward_passes() {
    const N_EXPERT: u32 = 8; // fixture's expert count

    let dir = common::model_dir("deepseek4-speculative");
    let model_path = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model_path);

    let mut a = load(&model_path, true);
    // A second instance of the *same* load path: recording routing must not
    // disturb the math, so both must produce identical logits and records.
    // (Cross-load-path parity — mmap vs streamed — has small float deltas by
    // design and is covered by its own tolerance-based tests.)
    let mut b = load(&model_path, true);

    let routed = |m: &QuantizedModel| -> Vec<Vec<u32>> {
        match m {
            QuantizedModel::DeepSeek4(w) => w.last_routed_experts().to_vec(),
            _ => panic!("expected DeepSeek4 model"),
        }
    };
    let assert_valid = |state: &[Vec<u32>]| {
        for ids in state {
            for pair in ids.windows(2) {
                assert!(pair[0] < pair[1], "recorded ids must be sorted+deduped");
            }
            assert!(
                ids.iter().all(|&e| e < N_EXPERT),
                "recorded ids must be valid expert indices: {ids:?}"
            );
        }
    };

    // Nothing has routed yet.
    let initial = routed(&a);
    assert_eq!(initial.len(), 2, "one entry per fixture layer");
    assert!(initial.iter().all(|v| v.is_empty()));

    // Prefill seeds the record from the final row of the prompt.
    let la = logits(&mut a, &[1, 4, 5], 0);
    let after_prefill = routed(&a);
    assert_valid(&after_prefill);
    assert!(
        after_prefill.iter().any(|v| !v.is_empty()),
        "prefill must seed the routing record: {after_prefill:?}"
    );

    // Decode refreshes it; outputs and records stay deterministic across
    // instances (the prefetch advice cannot affect the math).
    let d1a = logits(&mut a, &[7], 3);
    let lb = logits(&mut b, &[1, 4, 5], 0);
    let db = logits(&mut b, &[7], 3);
    assert_valid(&routed(&a));
    assert_valid(&routed(&b));
    assert_eq!(la, lb, "identical loads must produce identical logits");
    assert_eq!(d1a, db, "decode must be deterministic and unaffected");

    // Clearing the cache drops the speculative routing state too: it
    // belongs to whatever ran on this instance before the reset.
    match (&mut a, &mut b) {
        (QuantizedModel::DeepSeek4(wa), QuantizedModel::DeepSeek4(wb)) => {
            wa.clear_kv_cache();
            wb.clear_kv_cache();
        }
        _ => panic!("expected DeepSeek4 models"),
    }
    assert!(
        routed(&a).iter().all(|v| v.is_empty()),
        "clear must reset routing records: {:?}",
        routed(&a)
    );
    assert!(routed(&b).iter().all(|v| v.is_empty()));

    std::fs::remove_dir_all(&dir).ok();
}

/// Batched `forward_sequences` must be exact per-sequence: forwarding two
/// independent one-token sequences together (per-sequence KV, shared MoE)
/// must produce the same logits as forwarding each via the single-sequence
/// path, at any two positions.
#[test]
fn deepseek4_forward_sequences_matches_single_sequence() {
    let dir = common::model_dir("deepseek4-batched");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let make_input = |tokens: &[u32]| {
        Tensor::new(tokens, &Device::Cpu).unwrap().unsqueeze(0).unwrap()
    };
    let a = make_input(&[3]);
    let b = make_input(&[8]);
    let seqs = &[(&a, 0usize), (&b, 5usize)];

    // Single-sequence references, each on a FRESH model: an independent sequence
    // has its own empty KV, so the reference must not carry another sequence's
    // writes.
    let mut m_a = load(&model, true);
    let mut m_b = load(&model, true);
    let refs: Vec<Vec<f32>> = vec![logits(&mut m_a, &[3], 0), logits(&mut m_b, &[8], 5)];

    // Batched forward (per-sequence KV, shared MoE) on a fresh model so the
    // batched KV starts empty, exactly like the single-path references.
    let mut m2 = load(&model, true);
    let batched = match &mut m2 {
        QuantizedModel::DeepSeek4(w) => w.forward_sequences(seqs).unwrap(),
        _ => panic!("expected DeepSeek4 model"),
    };
    assert_eq!(batched.len(), 2, "one logits row per sequence");

    for (s, got) in batched.iter().enumerate() {
        let want = refs.get(s).unwrap();
        assert_eq!(got.len(), want.len(), "logits widths must match (vocab size)");
        // Batched dispatch shares the MoE's quantized matmul and index_add
        // across tokens, which reorders the f32 summation relative to a
        // single-token forward.  The math is identical; the only expected
        // divergence is f32 sum-order rounding (~1e-6).  Use a tight relative
        // tolerance to catch a real logic error (cross-sequence leakage, a
        // wrong per-seq split) while allowing the expected rounding.
        let mut max_rel = 0.0f32;
        let mut max_abs = 0.0f32;
        for (g, v) in got.iter().zip(want.iter()) {
            let denom = (*g).abs().max((*v).abs()).max(1.0e-4);
            max_rel = max_rel.max((*g - *v).abs() / denom);
            max_abs = max_abs.max((*g - *v).abs());
        }
        eprintln!(
            "seq {s}: max relative logit delta {max_rel:.3e} (abs {max_abs:.3e})",
        );
        assert!(
            max_rel <= 1.0e-4,
            "batched dispatch must match single-sequence within f32 rounding (seq {s}: rel {max_rel})"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// Multi-step batched decode must persist per-sequence KV across calls: after
/// feeding token `a` then token `b` to the *same* sequence through two
/// `forward_sequences` calls, the step-2 logits must match feeding `a` then
/// `b` through two sequential single-`forward` calls on a fresh model.  This
/// is the property the persistent `kv_seq` cache exists to guarantee — the
/// single-step test alone cannot catch KV being reset each call.
#[test]
fn deepseek4_forward_sequences_persists_kv_across_steps() {
    let dir = common::model_dir("deepseek4-batched-multistep");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let mut single = load(&model, true);
    let make_input = |tokens: &[u32]| {
        Tensor::new(tokens, &Device::Cpu).unwrap().unsqueeze(0).unwrap()
    };
    let a = make_input(&[3]);
    let b = make_input(&[8]);
    // Sequential single-sequence references: feed [3] at pos 0, then [8] at pos 1.
    let _ = logits(&mut single, &[3], 0);
    let ref_second = logits(&mut single, &[8], 1);

    // Batched: feed the same sequence [3]@0 then [8]@1 through the persistent-KV path.
    let mut batched = load(&model, true);
    let logits = match &mut batched {
        QuantizedModel::DeepSeek4(w) => {
            let _ = w.forward_sequences(&[(&a, 0usize)]).unwrap();
            w.forward_sequences(&[(&b, 1usize)]).unwrap()
        }
        _ => panic!("expected DeepSeek4 model"),
    };
    let got = logits.first().unwrap();

    assert_eq!(got.len(), ref_second.len(), "vocab sizes must match");
    let mut max_rel = 0.0f32;
    for (g, v) in got.iter().zip(ref_second.iter()) {
        let denom = (*g).abs().max((*v).abs()).max(1.0e-4);
        max_rel = max_rel.max((*g - *v).abs() / denom);
    }
    eprintln!("multi-step batched vs sequential: max relative logit delta {max_rel:.3e}");
    assert!(
        max_rel <= 1.0e-4,
        "batched KV must persist across steps (rel {max_rel})"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// `clear_kv_cache` must also drop the batched per-sequence KV.  The sliding
/// window and the compressed tables are position-masked, so stale rows there
/// are harmless, but the CSA compressor's streaming state is not: its
/// "previous window" half carries the last block of whatever ran before, and
/// a same-sized batch started after a reset would fold that into its own
/// block 0 the moment the block completes (position `ratio - 1`).  Runs
/// enough steps on the compressor fixture to cross that boundary and checks
/// every step against the same batch on a fresh model.
#[test]
fn deepseek4_clear_kv_cache_resets_batched_kv() {
    let dir = common::model_dir("deepseek4-batched-clear");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_compress(&model);

    let make_input = |tok: u32| Tensor::new(&[tok], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
    let first: [[u32; 2]; 6] = [[3, 8], [1, 9], [4, 2], [7, 7], [2, 5], [6, 1]];
    let second: [[u32; 2]; 6] = [[5, 1], [9, 4], [2, 8], [3, 3], [8, 6], [1, 2]];
    let run = |m: &mut QuantizedModel, toks: &[[u32; 2]; 6]| -> Vec<Vec<Vec<f32>>> {
        toks.iter()
            .enumerate()
            .map(|(pos, [a, b])| {
                let (a, b) = (make_input(*a), make_input(*b));
                m.forward_sequences(&[(&a, pos), (&b, pos)]).unwrap()
            })
            .collect()
    };

    // Reference: the second batch alone, on a fresh model.
    let mut fresh = load(&model, true);
    let want = run(&mut fresh, &second);

    // Same batch after an unrelated same-sized batch and a reset.
    let mut reused = load(&model, true);
    let _ = run(&mut reused, &first);
    reused.clear_kv_cache();
    let got = run(&mut reused, &second);

    for (step, (g_step, w_step)) in got.iter().zip(&want).enumerate() {
        for (s, (g, w)) in g_step.iter().zip(w_step).enumerate() {
            assert_eq!(g.len(), w.len(), "step {step} seq {s}: vocab sizes must match");
            let mut max_rel = 0.0f32;
            for (x, y) in g.iter().zip(w) {
                let denom = x.abs().max(y.abs()).max(1.0e-4);
                max_rel = max_rel.max((x - y).abs() / denom);
            }
            assert!(
                max_rel <= 1.0e-4,
                "step {step} seq {s}: batched KV must be cleared by clear_kv_cache (rel {max_rel})"
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The real V4-Flash expert layout (IQ2_XXS gate/up over a 256-wide expert,
/// Q2_K down projection) loads from the mapping, runs on the CPU expert
/// kernels, and agrees with the streamed load.
#[test]
fn deepseek4_q2k_down_experts_load_and_match_streamed_path() {
    let dir = common::model_dir("deepseek4-q2k-down");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_q2k_down(&model);
    let tokens = [1u32, 4, 2, 7, 5];
    let mut mapped = load(&model, true);
    let a = logits(&mut mapped, &tokens, 0);
    assert!(a.iter().all(|v| v.is_finite()), "mmap logits: {a:?}");
    let b = logits(&mut mapped, &[3], tokens.len());
    assert!(b.iter().all(|v| v.is_finite()), "mmap decode logits: {b:?}");
    let mut streamed = load(&model, false);
    let c = logits(&mut streamed, &tokens, 0);
    for (i, (x, y)) in a.iter().zip(&c).enumerate() {
        assert!((x - y).abs() < 1e-3, "logit {i}: mmap {x} vs streamed {y}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Long-prefill regression gate: at prompt lengths where real-model long-context
/// cliffs have been observed, the engine must stay self-consistent — a single
/// long prefill, a chunked prefill, and KV-continuation decode must all agree.
/// Catches position / causal-mask / KV drift that short prompts cannot reach.
#[test]
fn deepseek4_long_prefill_consistency() {
    let dir = common::model_dir("deepseek4-longprefill");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    // 300 prompt tokens cycling the tiny model's vocab (ids 1..=15; 0 is pad).
    let n = 300;
    let prompt: Vec<u32> = (0..n).map(|i| 1 + (i % 15) as u32).collect();
    assert!(n < 512, "fixture context is 512");

    // 1. Single long prefill (the reference).
    let mut single = load(&model, true);
    let reference = logits(&mut single, &prompt, 0);

    // 2. Chunked prefill: 3 chunks of 100 on a fresh instance.
    let mut chunked = load(&model, true);
    let mut chunked_last = Vec::new();
    for (ci, chunk) in prompt.chunks(100).enumerate() {
        chunked_last = logits(&mut chunked, chunk, ci * 100);
    }

    // 3. KV continuation: decode 8 greedy tokens from the chunked instance.
    let mut incremental_ids = prompt.clone();
    let mut incremental_last = chunked_last.clone();
    let mut decoded: Vec<u32> = Vec::new();
    for step in 0..8 {
        let next = incremental_last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        decoded.push(next);
        incremental_last = logits(&mut chunked, &[next], n + step);
        incremental_ids.push(next);
    }

    // 4. Reference for the continuation: a fresh instance prefilled over the
    //    full prompt + decoded prefix in one call (no incremental KV).
    let mut fresh = load(&model, true);
    let full_reference = logits(&mut fresh, &incremental_ids, 0);

    // Chunked vs single prefill must agree at the final position.
    assert_eq!(reference.len(), chunked_last.len(), "vocab widths");
    let mut worst = 0.0f32;
    for (i, (g, r)) in chunked_last.iter().zip(&reference).enumerate() {
        worst = worst.max((g - r).abs());
        let _ = i;
    }
    assert!(
        worst < 1e-2,
        "chunked prefill diverges from single prefill at {n} tokens (max abs {worst})"
    );

    // Incremental decode must match the fresh full-prefix reference — this is
    // what breaks when KV placement or position accounting drifts at length.
    assert_eq!(incremental_last.len(), full_reference.len(), "vocab widths");
    let mut worst_kv = 0.0f32;
    for (g, r) in incremental_last.iter().zip(&full_reference) {
        worst_kv = worst_kv.max((g - r).abs());
    }
    assert!(
        worst_kv < 1e-2,
        "KV continuation diverges from fresh full-prefix prefill (max abs {worst_kv})"
    );

    // And the greedy continuation must be self-consistent: the fresh reference
    // re-derived greedy tokens must equal the incremental ones.
    let mut ref_ids = prompt.clone();
    let mut ref_last = reference.clone();
    let mut ref_decoded: Vec<u32> = Vec::new();
    for step in 0..8 {
        let next = ref_last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        ref_decoded.push(next);
        // Re-prefill the whole prefix on a fresh instance each step.
        ref_ids.push(next);
        let mut fresh2 = load(&model, true);
        ref_last = logits(&mut fresh2, &ref_ids, 0);
    }
    assert_eq!(
        decoded, ref_decoded,
        "greedy continuation differs between incremental KV and fresh re-prefill"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Largest relative difference between two logit vectors.
fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vocab widths");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(1.0e-3))
        .fold(0.0, f32::max)
}

/// A causal model gives the same last-token logits for a prompt whether it
/// is prefilled in one pass, streamed in chunks, or fed one token at a
/// time.  On the compressor fixture (layer 0 is a CSA layer: window 8,
/// ratio 4, indexer top-k 2) every length up to the context limit crosses
/// the thresholds the real model crosses at 128 (window), 512-token chunks
/// and 2,048 tokens (the indexer starts dropping blocks).
#[test]
fn deepseek4_compressed_prefill_matches_decode_at_every_length() {
    let dir = common::model_dir("deepseek4-causal-bisect");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_compress(&model);
    // Context is 32: the longest prompt plus one decode step must fit.
    let tokens: Vec<u32> = (0..31u32).map(|i| (i * 7 + 5) % 16).collect();

    // Token-by-token: one decode step per position.
    let mut m = load(&model, true);
    let decode: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(p, t)| logits(&mut m, &[*t], p))
        .collect();

    let mut report = Vec::new();
    let mut worst = 0.0f32;
    for len in 1..=tokens.len() {
        let prompt = &tokens[..len];
        let mut m = load(&model, true);
        let one_pass = logits(&mut m, prompt, 0);
        let mut row = format!(
            "len {len:2}: decode {:.2e}",
            max_rel_diff(&one_pass, &decode[len - 1])
        );
        worst = worst.max(max_rel_diff(&one_pass, &decode[len - 1]));
        for chunk in [3usize, 4, 5, 8] {
            let chunks: Vec<joshua::stream_prefill::Chunk> = (0..len)
                .step_by(chunk)
                .map(|s| joshua::stream_prefill::Chunk {
                    tokens: &prompt[s..(s + chunk).min(len)],
                    pos: s,
                })
                .collect();
            let mut m = load(&model, true);
            let streamed: Vec<f32> = m
                .prefill_streamed(&chunks, &Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let d = max_rel_diff(&one_pass, &streamed);
            worst = worst.max(d);
            row.push_str(&format!(" | chunk{chunk} {d:.2e}"));
        }
        report.push(row);
    }
    eprintln!("{}", report.join("\n"));
    assert!(
        worst <= 1.0e-3,
        "prefill and decode disagree (worst {worst:.3e}):\n{}",
        report.join("\n")
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The same consistency across the HCA layer's ratio-128 blocks: layer 1
/// compresses every 128 tokens, which the real model's HCA layers first do
/// past 128 tokens (a long chat prompt, not a short one).  One-pass prefill,
/// streamed chunks landing on and off the block boundary, and
/// token-by-token decode must agree at lengths straddling 128 and 256.
#[test]
fn deepseek4_hca_prefill_matches_decode_across_blocks() {
    let dir = common::model_dir("deepseek4-hca-bisect");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_compress_hca(&model);
    let tokens: Vec<u32> = (0..300u32).map(|i| (i * 7 + 5) % 16).collect();

    let mut m = load(&model, true);
    let decode: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(p, t)| logits(&mut m, &[*t], p))
        .collect();

    let mut report = Vec::new();
    let mut worst = 0.0f32;
    for len in [100usize, 127, 128, 129, 130, 160, 255, 256, 257, 300] {
        let prompt = &tokens[..len];
        let mut m = load(&model, true);
        let one_pass = logits(&mut m, prompt, 0);
        let d = max_rel_diff(&one_pass, &decode[len - 1]);
        worst = worst.max(d);
        let mut row = format!("len {len:3}: decode {d:.2e}");
        for chunk in [64usize, 100, 128, 512] {
            let chunks: Vec<joshua::stream_prefill::Chunk> = (0..len)
                .step_by(chunk)
                .map(|s| joshua::stream_prefill::Chunk {
                    tokens: &prompt[s..(s + chunk).min(len)],
                    pos: s,
                })
                .collect();
            let mut m = load(&model, true);
            let streamed: Vec<f32> = m
                .prefill_streamed(&chunks, &Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let d = max_rel_diff(&one_pass, &streamed);
            worst = worst.max(d);
            row.push_str(&format!(" | chunk{chunk} {d:.2e}"));
        }
        report.push(row);
    }
    eprintln!("{}", report.join("\n"));
    assert!(
        worst <= 1.0e-3,
        "prefill and decode disagree across HCA blocks (worst {worst:.3e}):\n{}",
        report.join("\n")
    );
    std::fs::remove_dir_all(&dir).ok();
}
