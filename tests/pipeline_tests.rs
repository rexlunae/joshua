#![cfg(feature = "distributed")]
mod common;

use joshua::distributed::pipeline::{
    serve_controller, AgentConfig, Deployment, Input, Limits, Pipeline, Plan, Worker,
};
use std::{
    net::TcpListener,
    path::Path,
    thread::{self, JoinHandle},
    time::Instant,
};
use uuid::Uuid;

const KEY: &[u8] = b"pipeline-test-key-32-bytes-long!!!";

fn limits() -> Limits {
    Limits {
        context: 64,
        chunk: 16,
        sessions: 4,
        queue_depth: 1,
        batch: 16,
        timeout_ms: 2000,
        coordinator_memory_budget: 64 * 1024 * 1024,
    }
}

fn start(
    path: &Path,
    ends: &[usize],
    mmap: bool,
) -> (Pipeline, Vec<JoinHandle<anyhow::Result<()>>>) {
    start_with(path, ends, mmap, limits())
}

fn start_with(
    path: &Path,
    ends: &[usize],
    mmap: bool,
    limits: Limits,
) -> (Pipeline, Vec<JoinHandle<anyhow::Result<()>>>) {
    let plan = Plan::from_gguf(path, ends, &vec![64 * 1024 * 1024; ends.len()], limits).unwrap();
    let job = Uuid::new_v4();
    let mut handles = Vec::new();
    let mut addresses = Vec::new();
    for rank in 0..ends.len() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        addresses.push(listener.local_addr().unwrap());
        let worker = Worker::load(path, plan.clone(), rank, mmap).unwrap();
        handles.push(thread::spawn(move || worker.serve(listener, job, KEY)));
    }
    (
        Pipeline::connect(plan, &addresses, job, KEY).unwrap(),
        handles,
    )
}

#[test]
fn two_and_three_stage_prefill_decode_and_continuation_are_bit_exact() {
    let dir = common::model_dir("pipeline-parity");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    for mmap in [false, true] {
        for ends in [&[1, 3][..], &[2, 3][..], &[1, 2, 3][..]] {
            let (mut pipeline, handles) = start(&path, ends, mmap);
            let a = pipeline.open().unwrap();
            let b = pipeline.open().unwrap();
            let mut local_a = common::load_model(&path, mmap);
            let mut local_b = common::load_model(&path, mmap);
            // Same-session chunks and independent sequences interleave. Queue
            // depth 1 forces actual backpressure while preserving token order.
            let inputs = vec![
                Input {
                    session: a,
                    offset: 0,
                    tokens: vec![1, 4, 2],
                },
                Input {
                    session: b,
                    offset: 0,
                    tokens: vec![7, 5],
                },
                Input {
                    session: a,
                    offset: 3,
                    tokens: vec![7, 5, 9],
                },
                Input {
                    session: b,
                    offset: 2,
                    tokens: vec![1, 8, 3, 2],
                },
            ];
            let outputs = pipeline.forward_batch(&inputs).unwrap();
            for (input, output) in inputs.iter().zip(outputs) {
                let local = if input.session == a {
                    &mut local_a
                } else {
                    &mut local_b
                };
                assert_eq!(output, common::logits(local, &input.tokens, input.offset));
            }
            // Cross geometric KV allocation growth and continue through context.
            for offset in 6..64 {
                let tokens = vec![(offset % 16) as u32];
                let outputs = pipeline
                    .forward_batch(&[
                        Input {
                            session: a,
                            offset,
                            tokens: tokens.clone(),
                        },
                        Input {
                            session: b,
                            offset,
                            tokens: tokens.clone(),
                        },
                    ])
                    .unwrap();
                assert_eq!(outputs[0], common::logits(&mut local_a, &tokens, offset));
                assert_eq!(outputs[1], common::logits(&mut local_b, &tokens, offset));
            }
            assert!(pipeline
                .forward_batch(&[Input {
                    session: a,
                    offset: 64,
                    tokens: vec![1]
                }])
                .is_err());
            pipeline.close(a).unwrap();
            pipeline.close(b).unwrap();
            pipeline.shutdown().unwrap();
            for handle in handles {
                handle.join().unwrap().unwrap();
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn rejection_is_before_mutation_and_cancellation_releases_capacity() {
    let dir = common::model_dir("pipeline-admission");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    let (mut pipeline, handles) = start(&path, &[1, 2, 3], true);
    let ids: Vec<_> = (0..4).map(|_| pipeline.open().unwrap()).collect();
    assert!(pipeline.open().is_err());
    for tokens in [vec![], vec![16], vec![0; 17]] {
        assert!(pipeline
            .forward_batch(&[Input {
                session: ids[0],
                offset: 0,
                tokens
            }])
            .is_err());
    }
    // Invalid second chunk cannot leave the first one committed on any stage.
    assert!(pipeline
        .forward_batch(&[
            Input {
                session: ids[0],
                offset: 0,
                tokens: vec![1]
            },
            Input {
                session: ids[0],
                offset: 2,
                tokens: vec![1]
            },
        ])
        .is_err());
    assert!(pipeline
        .forward_batch(&[Input {
            session: 9999,
            offset: 0,
            tokens: vec![1]
        }])
        .is_err());
    let output = pipeline
        .forward_batch(&[Input {
            session: ids[0],
            offset: 0,
            tokens: vec![1],
        }])
        .unwrap();
    let mut local = common::load_model(&path, true);
    assert_eq!(output[0], common::logits(&mut local, &[1], 0));
    pipeline.close(ids[0]).unwrap();
    assert!(pipeline.close(ids[0]).is_err());
    let fresh = pipeline.open().unwrap();
    assert!(fresh > ids[3]);
    pipeline
        .forward_batch(&[Input {
            session: fresh,
            offset: 0,
            tokens: vec![2, 3],
        }])
        .unwrap();
    pipeline.shutdown().unwrap();
    for h in handles {
        h.join().unwrap().unwrap();
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn plans_reject_wrong_models_ranges_dimensions_and_memory() {
    let dir = common::model_dir("pipeline-plans");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    for ends in [&[0, 3][..], &[2, 1, 3][..], &[1, 2][..], &[4][..]] {
        assert!(Plan::from_gguf(&path, ends, &vec![u64::MAX; ends.len()], limits()).is_err());
    }
    assert!(Plan::from_gguf(&path, &[1, 3], &[1, 1], limits()).is_err());
    let plan = Plan::from_gguf(&path, &[1, 3], &[u64::MAX; 2], limits()).unwrap();
    let mut wrong = plan.clone();
    wrong.hidden += 1;
    assert!(Worker::load(&path, wrong, 0, true).is_err());
    let mut wrong = plan.clone();
    wrong.model_sha256 = "0".repeat(64);
    assert!(Worker::load(&path, wrong, 1, false).is_err());
    assert!(Worker::load(&path, plan, 2, false).is_err());
    common::write_tiny_qwen_gguf(&path, "qwen35");
    assert!(Plan::from_gguf(&path, &[1, 2], &[u64::MAX; 2], limits()).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn dropping_coordinator_invalidates_workers_and_clears_live_sessions() {
    let dir = common::model_dir("pipeline-disconnect");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    let (mut pipeline, handles) = start(&path, &[1, 2, 3], true);
    let id = pipeline.open().unwrap();
    pipeline
        .forward_batch(&[Input {
            session: id,
            offset: 0,
            tokens: vec![1, 2],
        }])
        .unwrap();
    drop(pipeline);
    for h in handles {
        assert!(h.join().unwrap().is_err());
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn idle_sessions_outlive_the_frame_deadline() {
    let dir = common::model_dir("pipeline-idle");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    let mut limits = limits();
    limits.timeout_ms = 100;
    let (mut pipeline, handles) = start_with(&path, &[1, 3], true, limits);
    let id = pipeline.open().unwrap();
    let prefill = pipeline
        .forward_batch(&[Input {
            session: id,
            offset: 0,
            tokens: vec![1, 2],
        }])
        .unwrap();
    assert_eq!(prefill.len(), 1);
    thread::sleep(std::time::Duration::from_millis(400));
    pipeline
        .forward_batch(&[Input {
            session: id,
            offset: 2,
            tokens: vec![3],
        }])
        .unwrap();
    pipeline.close(id).unwrap();
    pipeline.shutdown().unwrap();
    for h in handles {
        h.join().unwrap().unwrap();
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// Diagnostic only: no speed assertion on shared hosts or loopback networking.
#[test]
#[ignore = "explicit CPU loopback microbenchmark; run in release mode with --nocapture"]
fn benchmark_pipeline_loopback() {
    let dir = common::model_dir("pipeline-benchmark");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    let median = |mut times: Vec<f64>| {
        times.sort_by(f64::total_cmp);
        times[times.len() / 2]
    };
    for concurrency in [1, 4] {
        let prompt: Vec<u32> = (0..32).map(|i| (i % 16) as u32).collect();
        let mut models: Vec<_> = (0..concurrency)
            .map(|_| common::load_model(&path, true))
            .collect();
        let mut prefill_times = Vec::new();
        let mut decode_times = Vec::new();
        for trial in 0..6 {
            for model in &mut models {
                assert!(model.clear_kv_cache());
            }
            let prefill = Instant::now();
            for model in &mut models {
                common::logits(model, &prompt[..16], 0);
                common::logits(model, &prompt[16..], 16);
            }
            let prefill_ms = prefill.elapsed().as_secs_f64() * 1000.;
            let decode = Instant::now();
            for offset in 32..40 {
                for model in &mut models {
                    common::logits(model, &[1], offset);
                }
            }
            if trial > 0 {
                prefill_times.push(prefill_ms);
                decode_times.push(decode.elapsed().as_secs_f64() * 1000.);
            }
        }
        println!(
            "local concurrency={concurrency} median_prefill_ms={:.3} median_decode_ms={:.3}",
            median(prefill_times),
            median(decode_times)
        );
        for ends in [&[1, 3][..], &[1, 2, 3][..]] {
            let (mut pipeline, handles) = start(&path, ends, true);
            let mut prefill_times = Vec::new();
            let mut decode_times = Vec::new();
            for trial in 0..6 {
                let ids: Vec<_> = (0..concurrency).map(|_| pipeline.open().unwrap()).collect();
                let prefill = Instant::now();
                let inputs: Vec<_> = (0..2)
                    .flat_map(|chunk| {
                        ids.iter().map({
                            let prompt = &prompt;
                            move |&id| Input {
                                session: id,
                                offset: chunk * 16,
                                tokens: prompt[chunk * 16..(chunk + 1) * 16].to_vec(),
                            }
                        })
                    })
                    .collect();
                pipeline.forward_batch(&inputs).unwrap();
                let prefill_ms = prefill.elapsed().as_secs_f64() * 1000.;
                let decode = Instant::now();
                for offset in 32..40 {
                    pipeline
                        .forward_batch(
                            &ids.iter()
                                .map(|&id| Input {
                                    session: id,
                                    offset,
                                    tokens: vec![1],
                                })
                                .collect::<Vec<_>>(),
                        )
                        .unwrap();
                }
                if trial > 0 {
                    prefill_times.push(prefill_ms);
                    decode_times.push(decode.elapsed().as_secs_f64() * 1000.);
                }
                for id in ids {
                    pipeline.close(id).unwrap();
                }
            }
            println!("pipeline stages={} concurrency={concurrency} median_prefill_ms={:.3} median_decode_ms={:.3} metrics={}",
                ends.len(), median(prefill_times), median(decode_times), serde_json::to_string(&pipeline.metrics()).unwrap());
            pipeline.shutdown().unwrap();
            for h in handles {
                h.join().unwrap().unwrap();
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

// Drive the authenticated wire directly to exercise worker-side validation,
// independently of the coordinator's preflight checks.
struct RawClient {
    stream: std::net::TcpStream,
    job: Uuid,
    rank: usize,
    tx: u64,
    rx: u64,
}

impl RawClient {
    fn tag(&self, client: bool, seq: u64, payload: &[u8]) -> hmac::Hmac<sha2::Sha256> {
        use hmac::Mac;
        let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(KEY).unwrap();
        mac.update(b"joshua-qwen3-pipeline-v1");
        mac.update(self.job.as_bytes());
        mac.update(&(self.rank as u64).to_le_bytes());
        mac.update(&[u8::from(client)]);
        mac.update(&seq.to_le_bytes());
        mac.update(&(payload.len() as u64).to_le_bytes());
        mac.update(payload);
        mac
    }
    fn send(&mut self, control: serde_json::Value, values: &[f32]) {
        use hmac::Mac;
        use std::io::Write;
        let header = serde_json::to_vec(&control).unwrap();
        let mut body = (header.len() as u32).to_le_bytes().to_vec();
        body.extend(header);
        for value in values {
            body.extend(value.to_le_bytes());
        }
        let tag = self.tag(true, self.tx, &body).finalize().into_bytes();
        self.stream
            .write_all(&(body.len() as u32).to_le_bytes())
            .unwrap();
        self.stream.write_all(&body).unwrap();
        self.stream.write_all(&tag).unwrap();
        self.tx += 1;
    }
    fn receive(&mut self) -> std::io::Result<()> {
        use hmac::Mac;
        use std::io::Read;
        let mut length = [0; 4];
        self.stream.read_exact(&mut length)?;
        let mut body = vec![0; u32::from_le_bytes(length) as usize];
        self.stream.read_exact(&mut body)?;
        let mut tag = [0; 32];
        self.stream.read_exact(&mut tag)?;
        self.tag(false, self.rx, &body).verify_slice(&tag).unwrap();
        self.rx += 1;
        Ok(())
    }
}

/// Complete a valid handshake and stop, asserting a clean worker exit.
fn stop_worker(
    address: std::net::SocketAddr,
    job: Uuid,
    plan: &Plan,
    handle: JoinHandle<anyhow::Result<()>>,
) {
    use serde_json::json;
    let mut client = RawClient {
        stream: std::net::TcpStream::connect(address).unwrap(),
        job,
        rank: 0,
        tx: 0,
        rx: 0,
    };
    client.send(json!({"Hello":{"plan":plan,"rank":0}}), &[]);
    client.receive().unwrap();
    client.send(json!("Stop"), &[]);
    client.receive().unwrap();
    handle.join().unwrap().unwrap();
}

#[test]
fn worker_rejects_duplicate_out_of_order_shape_and_authentication_errors() {
    use serde_json::json;
    let dir = common::model_dir("pipeline-worker-protocol");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    let plan = Plan::from_gguf(&path, &[1, 3], &[64 * 1024 * 1024; 2], limits()).unwrap();
    let cases = [
        (
            json!({"Forward":{"id":0,"sequence":0,"offset":1,"tokens":[1]}}),
            vec![],
        ),
        (
            json!({"Forward":{"id":0,"sequence":1,"offset":0,"tokens":[1]}}),
            vec![],
        ),
        (
            json!({"Forward":{"id":0,"sequence":0,"offset":0,"tokens":[1]}}),
            vec![0.0],
        ),
        (json!({"Open":{"id":0}}), vec![]),
    ];
    for (command, values) in cases {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let job = Uuid::new_v4();
        let worker = Worker::load(&path, plan.clone(), 0, true).unwrap();
        let handle = thread::spawn(move || worker.serve(listener, job, KEY));
        let mut client = RawClient {
            stream: std::net::TcpStream::connect(address).unwrap(),
            job,
            rank: 0,
            tx: 0,
            rx: 0,
        };
        client.send(json!({"Hello":{"plan":plan,"rank":0}}), &[]);
        client.receive().unwrap();
        client.send(json!({"Open":{"id":0}}), &[]);
        client.receive().unwrap();
        client.send(command, &values);
        assert!(client.receive().is_err());
        assert!(handle.join().unwrap().is_err());
    }
    // Repeated forward at the same position/sequence must fail after a valid pass.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let job = Uuid::new_v4();
    let worker = Worker::load(&path, plan.clone(), 0, false).unwrap();
    let handle = thread::spawn(move || worker.serve(listener, job, KEY));
    let mut client = RawClient {
        stream: std::net::TcpStream::connect(address).unwrap(),
        job,
        rank: 0,
        tx: 0,
        rx: 0,
    };
    client.send(json!({"Hello":{"plan":plan,"rank":0}}), &[]);
    client.receive().unwrap();
    client.send(json!({"Open":{"id":0}}), &[]);
    client.receive().unwrap();
    let forward = json!({"Forward":{"id":0,"sequence":0,"offset":0,"tokens":[1,2]}});
    client.send(forward.clone(), &[]);
    client.receive().unwrap();
    client.send(forward, &[]);
    assert!(client.receive().is_err());
    assert!(handle.join().unwrap().is_err());
    // A valid HMAC from a different job is refused at handshake without
    // stranding the worker: the real coordinator can still connect.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let job = Uuid::new_v4();
    let worker = Worker::load(&path, plan.clone(), 0, true).unwrap();
    let handle = thread::spawn(move || worker.serve(listener, job, KEY));
    let mut client = RawClient {
        stream: std::net::TcpStream::connect(address).unwrap(),
        job: Uuid::new_v4(),
        rank: 0,
        tx: 0,
        rx: 0,
    };
    client.send(json!({"Hello":{"plan":plan,"rank":0}}), &[]);
    assert!(client.receive().is_err());
    // Authenticated first frames other than a matching Hello are also dropped.
    for first in [json!("Stop"), json!({"Hello":{"plan":plan,"rank":1}})] {
        let mut client = RawClient {
            stream: std::net::TcpStream::connect(address).unwrap(),
            job,
            rank: 0,
            tx: 0,
            rx: 0,
        };
        client.send(first, &[]);
        assert!(client.receive().is_err());
    }
    assert!(!handle.is_finished());
    stop_worker(address, job, &plan, handle);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn stage_failure_releases_bounded_queues_and_permanently_invalidates_coordinator() {
    use serde_json::json;
    let dir = common::model_dir("pipeline-failure");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    let plan = Plan::from_gguf(&path, &[1, 2, 3], &[64 * 1024 * 1024; 3], limits()).unwrap();
    let job = Uuid::new_v4();
    let mut addresses = Vec::new();
    let mut handles = Vec::new();
    for rank in [0, 2] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        addresses.push(listener.local_addr().unwrap());
        let worker = Worker::load(&path, plan.clone(), rank, true).unwrap();
        handles.push(thread::spawn(move || worker.serve(listener, job, KEY)));
    }
    let broken = TcpListener::bind("127.0.0.1:0").unwrap();
    addresses.insert(1, broken.local_addr().unwrap());
    let fake = thread::spawn(move || {
        use hmac::Mac;
        use std::io::{Read, Write};
        let (stream, _) = broken.accept().unwrap();
        let mut server = RawClient {
            stream,
            job,
            rank: 1,
            tx: 0,
            rx: 0,
        };
        // Accept hello and open, then close in the middle of the first forward.
        for _ in 0..2 {
            let mut length = [0; 4];
            server.stream.read_exact(&mut length).unwrap();
            let mut body = vec![0; u32::from_le_bytes(length) as usize];
            server.stream.read_exact(&mut body).unwrap();
            let mut tag = [0; 32];
            server.stream.read_exact(&mut tag).unwrap();
            server
                .tag(true, server.rx, &body)
                .verify_slice(&tag)
                .unwrap();
            server.rx += 1;
            let header = serde_json::to_vec(&json!({"Reply":{"compute_ns":0}})).unwrap();
            let mut body = (header.len() as u32).to_le_bytes().to_vec();
            body.extend(header);
            let tag = server.tag(false, server.tx, &body).finalize().into_bytes();
            server.tx += 1;
            server
                .stream
                .write_all(&(body.len() as u32).to_le_bytes())
                .unwrap();
            server.stream.write_all(&body).unwrap();
            server.stream.write_all(&tag).unwrap();
        }
    });
    let mut pipeline = Pipeline::connect(plan, &addresses, job, KEY).unwrap();
    let id = pipeline.open().unwrap();
    let t = Instant::now();
    let batch: Vec<_> = (0..16)
        .map(|i| Input {
            session: id,
            offset: i,
            tokens: vec![1],
        })
        .collect();
    assert!(pipeline.forward_batch(&batch).is_err());
    assert!(
        t.elapsed().as_secs() < 10,
        "bounded queues deadlocked after worker loss"
    );
    assert!(pipeline.open().is_err());
    assert!(pipeline
        .forward_batch(&[Input {
            session: id,
            offset: 0,
            tokens: vec![1]
        }])
        .is_err());
    fake.join().unwrap();
    for h in handles {
        assert!(h.join().unwrap().is_err());
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn compact_q8_stages_match_local_with_mmap_and_streamed_loading() {
    let dir = common::model_dir("pipeline-quantized");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_quantized_layers(&path, 3);
    for mmap in [false, true] {
        let (mut pipeline, handles) = start(&path, &[1, 2, 3], mmap);
        let id = pipeline.open().unwrap();
        let mut local = common::load_model(&path, mmap);
        for (offset, tokens) in [
            (0, vec![1, 4, 2]),
            (3, vec![3, 8]),
            (5, vec![9]),
            (6, vec![2]),
        ] {
            let out = pipeline
                .forward_batch(&[Input {
                    session: id,
                    offset,
                    tokens: tokens.clone(),
                }])
                .unwrap();
            assert_eq!(out[0], common::logits(&mut local, &tokens, offset));
        }
        pipeline.shutdown().unwrap();
        for h in handles {
            h.join().unwrap().unwrap();
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn assigned_stage_does_not_load_unowned_layers_or_head() {
    use candle_core::quantized::gguf_file;
    let dir = common::model_dir("pipeline-selective-load");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    let mut reader = std::fs::File::open(&path).unwrap();
    let content = gguf_file::Content::read(&mut reader).unwrap();
    let metadata: Vec<_> = content.metadata.into_iter().collect();
    // Leave model metadata intact but deliberately omit every tensor outside
    // stage zero. A load-then-drop implementation would fail on this file.
    let mut reader = std::fs::File::open(&path).unwrap();
    let content = gguf_file::Content::read(&mut reader).unwrap();
    let tensors: Vec<_> = content
        .tensor_infos
        .keys()
        .filter(|name| name.starts_with("blk.0.") || *name == "token_embd.weight")
        .map(|name| {
            (
                name.clone(),
                content
                    .tensor(&mut reader, name, &candle_core::Device::Cpu)
                    .unwrap(),
            )
        })
        .collect();
    let partial = dir.join("stage-zero.gguf");
    gguf_file::write(
        &mut std::fs::File::create(&partial).unwrap(),
        &metadata
            .iter()
            .map(|(n, v)| (n.as_str(), v))
            .collect::<Vec<_>>(),
        &tensors
            .iter()
            .map(|(n, t)| (n.as_str(), t))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let plan = Plan::from_gguf(&partial, &[1, 3], &[64 * 1024 * 1024; 2], limits()).unwrap();
    for mmap in [false, true] {
        assert!(Worker::load(&partial, plan.clone(), 0, mmap).is_ok());
    }
    assert!(Worker::load(&partial, plan, 1, true).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn out_of_band_abort_interrupts_an_inflight_rpc() {
    use hmac::Mac;
    use serde_json::json;
    use std::io::{Read, Write};
    let dir = common::model_dir("pipeline-active-cancel");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    let plan = Plan::from_gguf(&path, &[3], &[64 * 1024 * 1024], limits()).unwrap();
    let job = Uuid::new_v4();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (started_send, started_receive) = std::sync::mpsc::channel();
    let fake = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut server = RawClient {
            stream,
            job,
            rank: 0,
            tx: 0,
            rx: 0,
        };
        for request in 0..3 {
            let mut length = [0; 4];
            server.stream.read_exact(&mut length).unwrap();
            let mut body = vec![0; u32::from_le_bytes(length) as usize];
            server.stream.read_exact(&mut body).unwrap();
            let mut tag = [0; 32];
            server.stream.read_exact(&mut tag).unwrap();
            server
                .tag(true, server.rx, &body)
                .verify_slice(&tag)
                .unwrap();
            server.rx += 1;
            if request == 2 {
                started_send.send(()).unwrap();
                // Withhold the response: cancellation must interrupt a blocked
                // receiver rather than wait for the configured deadline.
                assert!(server.stream.read_exact(&mut length).is_err());
                break;
            }
            let header = serde_json::to_vec(&json!({"Reply":{"compute_ns":0}})).unwrap();
            let mut body = (header.len() as u32).to_le_bytes().to_vec();
            body.extend(header);
            let tag = server.tag(false, server.tx, &body).finalize().into_bytes();
            server.tx += 1;
            server
                .stream
                .write_all(&(body.len() as u32).to_le_bytes())
                .unwrap();
            server.stream.write_all(&body).unwrap();
            server.stream.write_all(&tag).unwrap();
        }
    });
    let mut pipeline = Pipeline::connect(plan, &[address], job, KEY).unwrap();
    let id = pipeline.open().unwrap();
    let abort = pipeline.abort_handle().unwrap();
    let active = thread::spawn(move || {
        assert!(pipeline
            .forward_batch(&[Input {
                session: id,
                offset: 0,
                tokens: vec![1]
            }])
            .is_err());
        assert!(pipeline.open().is_err());
    });
    started_receive
        .recv_timeout(std::time::Duration::from_secs(2))
        .unwrap();
    abort.abort();
    active.join().unwrap();
    fake.join().unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "build pipeline_qwen3 example and set JOSHUA_PIPELINE_EXAMPLE to its absolute path"]
fn pipeline_example_runs_three_separate_worker_processes() {
    use std::{
        io::{BufRead, BufReader},
        process::{Command, Stdio},
    };
    struct ChildGuard(std::process::Child);
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let example = std::env::var("JOSHUA_PIPELINE_EXAMPLE").expect("set JOSHUA_PIPELINE_EXAMPLE");
    let dir = common::model_dir("pipeline-process-smoke");
    let path = dir.join("model.gguf");
    let manifest = dir.join("plan.json");
    common::write_tiny_qwen3_quantized_layers(&path, 3);
    assert!(Command::new(&example)
        .arg("plan")
        .arg("--model")
        .arg(&path)
        .arg("--output")
        .arg(&manifest)
        .args([
            "--ends",
            "1,2,3",
            "--budgets-mib",
            "64,64,64",
            "--context",
            "64",
            "--chunk",
            "3",
            "--sessions",
            "2",
            "--batch",
            "2"
        ])
        .status()
        .unwrap()
        .success());
    let job = Uuid::new_v4().to_string();
    let mut children = Vec::new();
    let mut peers = Vec::new();
    for rank in 0..3 {
        let mut child = ChildGuard(
            Command::new(&example)
                .arg("worker")
                .arg("--model")
                .arg(&path)
                .arg("--plan")
                .arg(&manifest)
                .arg("--rank")
                .arg(rank.to_string())
                .arg("--listen")
                .arg("127.0.0.1:0")
                .arg("--job")
                .arg(&job)
                .env("JOSHUA_PIPELINE_KEY", std::str::from_utf8(KEY).unwrap())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let mut ready = String::new();
        BufReader::new(child.0.stderr.take().unwrap())
            .read_line(&mut ready)
            .unwrap();
        assert!(
            ready.contains("reservation_bytes="),
            "worker did not become ready: {ready}"
        );
        let address = ready.split("listen=").nth(1).unwrap().trim().to_owned();
        assert!(address.parse::<std::net::SocketAddr>().unwrap().port() > 0);
        peers.push(address);
        children.push(child);
    }
    let output = Command::new(&example)
        .arg("run")
        .arg("--plan")
        .arg(&manifest)
        .arg("--peers")
        .arg(peers.join(","))
        .arg("--job")
        .arg(&job)
        .args(["--tokens", "1,4,2,7,5", "--decode", "8", "--requests", "2"])
        .env("JOSHUA_PIPELINE_KEY", std::str::from_utf8(KEY).unwrap())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let mut local = common::load_model(&path, true);
    let mut logits = common::logits(&mut local, &[1, 4, 2, 7, 5], 0);
    let mut expected = Vec::new();
    for step in 0..8 {
        let token = logits
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (i, &x)| {
                if x > best.1 {
                    (i, x)
                } else {
                    best
                }
            })
            .0 as u32;
        expected.push(token);
        if step + 1 < 8 {
            logits = common::logits(&mut local, &[token], 5 + step);
        }
    }
    assert_eq!(result["generated"], serde_json::json!([expected, expected]));
    for child in &mut children {
        assert!(child.0.wait().unwrap().success());
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn incomplete_frame_expires_without_allocating_an_unbounded_payload() {
    use std::io::{Read, Write};
    let dir = common::model_dir("pipeline-frame-limits");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    let mut limits = limits();
    limits.timeout_ms = 100;
    let plan = Plan::from_gguf(&path, &[3], &[64 * 1024 * 1024], limits).unwrap();
    for length in [u32::MAX, 1024] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let job = Uuid::new_v4();
        let worker = Worker::load(&path, plan.clone(), 0, true).unwrap();
        let handle = thread::spawn(move || worker.serve(listener, job, KEY));
        let mut stream = std::net::TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(2)))
            .unwrap();
        stream.write_all(&length.to_le_bytes()).unwrap();
        let t = Instant::now();
        let mut reply = [0; 4];
        assert!(stream.read_exact(&mut reply).is_err());
        assert!(t.elapsed().as_secs() < 2, "stalled frame did not expire");
        // A stalled pre-handshake peer cannot strand the worker.
        stop_worker(address, job, &plan, handle);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// A model-less worker that serves `connections` controllers, then exits.
fn agent(
    config: AgentConfig,
    connections: usize,
) -> (std::net::SocketAddr, JoinHandle<Vec<anyhow::Result<()>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        (0..connections)
            .map(|_| {
                let (stream, _) = listener.accept().unwrap();
                serve_controller(stream, KEY, &config)
            })
            .collect()
    });
    (address, handle)
}

fn assert_matches_local(pipeline: &mut Pipeline, path: &Path) {
    let id = pipeline.open().unwrap();
    let mut local = common::load_model(path, true);
    let prompt = [1, 4, 2, 7, 5, 9];
    let out = pipeline
        .forward_batch(&[Input {
            session: id,
            offset: 0,
            tokens: prompt.to_vec(),
        }])
        .unwrap();
    assert_eq!(out[0], common::logits(&mut local, &prompt, 0));
    for offset in prompt.len()..prompt.len() + 4 {
        let tokens = vec![(offset % 16) as u32];
        let out = pipeline
            .forward_batch(&[Input {
                session: id,
                offset,
                tokens: tokens.clone(),
            }])
            .unwrap();
        assert_eq!(out[0], common::logits(&mut local, &tokens, offset));
    }
    pipeline.close(id).unwrap();
}

#[test]
fn model_less_workers_load_their_stage_from_the_controller_bit_exact() {
    let dir = common::model_dir("pipeline-deploy");
    for quantized in [false, true] {
        let path = dir.join(format!("model-{quantized}.gguf"));
        if quantized {
            common::write_tiny_qwen3_quantized_layers(&path, 3);
        } else {
            common::write_tiny_qwen3_layers(&path, 3);
        }
        for (mmap, ends) in [
            (true, Some(vec![1, 2, 3])),
            (false, Some(vec![2, 3])),
            (true, None),
        ] {
            let workers = ends.as_ref().map_or(2, Vec::len);
            let config = AgentConfig {
                mmap,
                ..AgentConfig::default()
            };
            let (addresses, handles): (Vec<_>, Vec<_>) =
                (0..workers).map(|_| agent(config.clone(), 1)).unzip();
            let mut pipeline = Pipeline::deploy(
                &path,
                &addresses,
                KEY,
                Deployment {
                    ends: ends.clone(),
                    limits: limits(),
                },
            )
            .unwrap();
            assert_eq!(pipeline.plan().stages.len(), workers);
            assert_matches_local(&mut pipeline, &path);
            pipeline.shutdown().unwrap();
            for handle in handles {
                for result in handle.join().unwrap() {
                    result.unwrap();
                }
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cached_slices_are_reused_and_corrupt_ones_are_resent() {
    let dir = common::model_dir("pipeline-deploy-cache");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    let caches = [dir.join("cache-0"), dir.join("cache-1")];
    let (addresses, handles): (Vec<_>, Vec<_>) = caches
        .iter()
        .map(|cache| {
            agent(
                AgentConfig {
                    cache_dir: Some(cache.clone()),
                    ..AgentConfig::default()
                },
                3,
            )
        })
        .collect();
    let deploy = || {
        Pipeline::deploy(
            &path,
            &addresses,
            KEY,
            Deployment {
                ends: Some(vec![1, 3]),
                limits: limits(),
            },
        )
        .unwrap()
    };
    let slice = |cache: &Path| {
        std::fs::read_dir(cache)
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "gguf"))
            .unwrap()
    };
    let mut pipeline = deploy();
    assert_matches_local(&mut pipeline, &path);
    pipeline.shutdown().unwrap();
    let first: Vec<_> = caches.iter().map(|c| slice(c)).collect();
    let modified = |p: &Path| std::fs::metadata(p).unwrap().modified().unwrap();
    let stamps: Vec<_> = first.iter().map(|p| modified(p)).collect();
    // Unchanged slices are reused without a transfer.
    let mut pipeline = deploy();
    assert_matches_local(&mut pipeline, &path);
    pipeline.shutdown().unwrap();
    assert_eq!(stamps, first.iter().map(|p| modified(p)).collect::<Vec<_>>());
    // A damaged cache entry fails its checksum and is replaced.
    let original = std::fs::read(&first[1]).unwrap();
    let mut bytes = original.clone();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&first[1], bytes).unwrap();
    let mut pipeline = deploy();
    assert_matches_local(&mut pipeline, &path);
    pipeline.shutdown().unwrap();
    assert_eq!(std::fs::read(&first[1]).unwrap(), original);
    for handle in handles {
        for result in handle.join().unwrap() {
            result.unwrap();
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn deploy_rejects_wrong_keys_and_impossible_splits() {
    let dir = common::model_dir("pipeline-deploy-reject");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    // A controller with another key cannot assign a stage.
    let (address, handle) = agent(AgentConfig::default(), 1);
    let other = b"another-pipeline-key-32-bytes-long";
    assert!(Pipeline::deploy(&path, &[address], other, Deployment {
        ends: None,
        limits: limits(),
    })
    .is_err());
    assert!(handle.join().unwrap()[0].is_err());
    // More workers than layers.
    let (addresses, handles): (Vec<_>, Vec<_>) =
        (0..4).map(|_| agent(AgentConfig::default(), 1)).unzip();
    assert!(Pipeline::deploy(&path, &addresses, KEY, Deployment {
        ends: None,
        limits: limits(),
    })
    .is_err());
    for handle in handles {
        // The controller gave up after probing; each worker saw it leave.
        assert!(handle.join().unwrap()[0].is_err());
    }
    // Each stage needs its own worker.
    let (address, _) = agent(AgentConfig::default(), 1);
    assert!(Pipeline::deploy(&path, &[address, address], KEY, Deployment {
        ends: None,
        limits: limits(),
    })
    .is_err());
    std::fs::remove_dir_all(dir).unwrap();
}
