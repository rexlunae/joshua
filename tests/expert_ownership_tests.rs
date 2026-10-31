#![cfg(feature = "distributed")]
mod common;

use candle_core::{
    quantized::{gguf_file, QMatMul},
    Device, Module, Tensor,
};
use joshua::{
    distributed::expert::{Coordinator, ExpertSet, Limits, Placement, RowId},
    moe::{self, Expert, ExpertTensor},
};
use std::{fs::File, net::TcpListener, path::Path, sync::Arc, thread};

const KEY: &[u8] = b"complete-expert-tests-32-byte-key!";

fn limits() -> Limits {
    Limits {
        rows: 8,
        routes: 32,
        sessions: 2,
        frame_bytes: 128 * 1024,
        scratch_bytes: 64 * 1024 * 1024,
        resident_bytes_per_owner: 4 * 1024 * 1024,
        timeout_ms: 2000,
    }
}

struct LocalExpert {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}
impl Expert for LocalExpert {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        self.down.forward(
            &(candle_nn::ops::silu(&self.gate.forward(input)?)? * self.up.forward(input)?)?,
        )
    }
}

fn local_experts(path: &Path, mapped: bool) -> Vec<LocalExpert> {
    let mut file = File::open(path).unwrap();
    let content = gguf_file::Content::read(&mut file).unwrap();
    let mapping = mapped.then(|| {
        let bytes = std::fs::read(path).unwrap();
        let mut mapping = memmap2::MmapMut::map_anon(bytes.len()).unwrap();
        mapping.copy_from_slice(&bytes);
        Arc::new(mapping.make_read_only().unwrap())
    });
    let mut projections: Vec<Vec<QMatMul>> = ["gate", "up", "down"]
        .iter()
        .map(|name| {
            let descriptor = ExpertTensor::lookup(
                &content,
                "qwen3moe",
                &format!("blk.0.ffn_{name}_exps.weight"),
                4,
            )
            .unwrap();
            let borrowed = mapping
                .as_ref()
                .and_then(|mapping| descriptor.borrow_host(mapping).unwrap());
            let weights = match borrowed {
                Some(experts) => experts.into_iter().map(|expert| expert.tensor).collect(),
                None => descriptor
                    .read_and_split(&content, &mut file, "qwen3moe", &Device::Cpu)
                    .unwrap(),
            };
            weights
                .into_iter()
                .map(|weight| QMatMul::from_qtensor(weight).unwrap())
                .collect()
        })
        .collect();
    let down = projections.pop().unwrap();
    let up = projections.pop().unwrap();
    let gate = projections.pop().unwrap();
    gate.into_iter()
        .zip(up)
        .zip(down)
        .map(|((gate, up), down)| LocalExpert { gate, up, down })
        .collect()
}

fn start(path: &Path, mapped: bool) -> (Coordinator, Vec<thread::JoinHandle<anyhow::Result<()>>>) {
    start_with_limits(path, mapped, limits())
}

fn start_with_limits(
    path: &Path,
    mapped: bool,
    bounds: Limits,
) -> (Coordinator, Vec<thread::JoinHandle<anyhow::Result<()>>>) {
    let placement = Placement::from_gguf(path, 0, vec![1, 0, 1, 0], bounds).unwrap();
    let mapping = if mapped {
        let bytes = std::fs::read(path).unwrap();
        let mut mapping = memmap2::MmapMut::map_anon(bytes.len()).unwrap();
        mapping.copy_from_slice(&bytes);
        Some(Arc::new(mapping.make_read_only().unwrap()))
    } else {
        None
    };
    let job = uuid::Uuid::new_v4();
    let mut addresses = Vec::new();
    let mut handles = Vec::new();
    for owner in 0..2 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        addresses.push(listener.local_addr().unwrap());
        let experts = ExpertSet::load(path, placement.clone(), owner, mapping.as_ref()).unwrap();
        for expert in 0..4 {
            assert_eq!(experts.get(expert).is_some(), [1, 0, 1, 0][expert] == owner);
        }
        handles.push(thread::spawn(move || {
            experts.serve(listener.accept()?.0, job, KEY)
        }));
    }
    (
        Coordinator::connect(placement, &addresses, job, KEY).unwrap(),
        handles,
    )
}

#[test]
fn idle_workers_survive_a_pause_longer_than_the_frame_timeout() {
    let dir = common::model_dir("expert-ownership-idle");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3moe_gguf(&path);
    let mut bounds = limits();
    bounds.timeout_ms = 250;
    let (mut coordinator, handles) = start_with_limits(&path, false, bounds);
    let input = Tensor::new(&[[0.25f32; 8]], &Device::Cpu).unwrap();
    let ids = Tensor::new(&[[3u32, 1, 3]], &Device::Cpu).unwrap();
    let weights = Tensor::new(&[[0.25f32, -0.5, 0.125]], &Device::Cpu).unwrap();
    let row = [RowId {
        session: 1,
        token: 0,
    }];
    let expected = coordinator.forward(&row, &input, &ids, &weights).unwrap();
    thread::sleep(std::time::Duration::from_millis(600));
    let actual = coordinator.forward(&row, &input, &ids, &weights).unwrap();
    assert_eq!(
        actual.to_vec2::<f32>().unwrap(),
        expected.to_vec2::<f32>().unwrap()
    );
    drop(coordinator);
    for handle in handles {
        assert!(handle.join().unwrap().is_err());
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn frame_admission_and_owner_load_identity_are_checked_before_serving() {
    let dir = common::model_dir("expert-ownership-frame-and-identity");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3moe_gguf(&path);
    for frame_bytes in [1, 1024] {
        let mut bounds = limits();
        bounds.frame_bytes = frame_bytes;
        assert!(Placement::from_gguf(&path, 0, vec![0; 4], bounds)
            .unwrap_err()
            .to_string()
            .contains("frame budget"));
    }
    let plan = Placement::from_gguf(&path, 0, vec![0; 4], limits()).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let mut mapping = memmap2::MmapMut::map_anon(bytes.len()).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    mapping.copy_from_slice(&bytes);
    let mapping = Arc::new(mapping.make_read_only().unwrap());
    assert!(ExpertSet::load(&path, plan.clone(), 0, Some(&mapping))
        .err()
        .unwrap()
        .to_string()
        .contains("mapping identity mismatch"));
    std::fs::write(&path, bytes).unwrap();
    assert!(ExpertSet::load(&path, plan, 0, None)
        .err()
        .unwrap()
        .to_string()
        .contains("model identity mismatch"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn actual_gguf_experts_match_local_decode_prefill_sessions_duplicates_and_ties() {
    let dir = common::model_dir("expert-ownership-parity");
    let path = dir.join("model.gguf");
    for quantized in [false, true] {
        if quantized {
            common::write_tiny_qwen3moe_quantized_gguf(&path);
        } else {
            common::write_tiny_qwen3moe_gguf(&path);
        }
        for mapped in [false, true] {
            let local = local_experts(&path, mapped);
            let (mut coordinator, handles) = start(&path, mapped);
            let hidden = if quantized { 256 } else { 8 };
            for (batch, offset) in [(1, 0), (5, 1), (1, 6), (3, 7)] {
                let rows: Vec<_> = (0..batch)
                    .map(|index| RowId {
                        session: index as u64 % 2,
                        token: offset + index as u64,
                    })
                    .collect();
                let input = Tensor::from_vec(
                    (0..batch * hidden)
                        .map(|index| (index as f32 * 0.017).sin())
                        .collect(),
                    (batch, hidden),
                    &Device::Cpu,
                )
                .unwrap();
                for duplicate in [false, true] {
                    let (ids, weights) = if duplicate {
                        (
                            Tensor::new(vec![vec![3u32, 1, 3]; batch], &Device::Cpu).unwrap(),
                            Tensor::new(vec![vec![0.25f32, -0.5, 0.125]; batch], &Device::Cpu)
                                .unwrap(),
                        )
                    } else {
                        let scores =
                            Tensor::zeros((batch, 4), candle_core::DType::F32, &Device::Cpu)
                                .unwrap();
                        moe::route(&scores, None, 3, true, 1.0, |scores| {
                            candle_nn::ops::softmax(&scores, candle_core::D::Minus1)
                        })
                        .unwrap()
                    };
                    let expected = if batch == 1 {
                        moe::dispatch_decode(
                            "qwen3moe",
                            &local,
                            &Device::Cpu,
                            &input,
                            &ids,
                            &weights,
                            3,
                        )
                        .unwrap()
                        .0
                    } else {
                        moe::dispatch_prefill(
                            "qwen3moe",
                            &local,
                            &Device::Cpu,
                            &input,
                            &ids,
                            &weights,
                            batch,
                            3,
                        )
                        .unwrap()
                        .0
                    };
                    let actual = coordinator.forward(&rows, &input, &ids, &weights).unwrap();
                    assert_eq!(actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(), expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(), "quantized={quantized}, mapped={mapped}, batch={batch}, duplicate={duplicate}");
                }
            }
            drop(coordinator);
            for handle in handles {
                assert!(handle.join().unwrap().is_err());
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn admission_identity_shared_experts_and_fail_stop() {
    let dir = common::model_dir("expert-ownership-invalid");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3moe_gguf(&path);
    let plan = Placement::from_gguf(&path, 0, vec![0; 4], limits()).unwrap();
    for budget in [
        "rows", "routes", "sessions", "bytes", "scratch", "resident", "timeout",
    ] {
        let mut bad = limits();
        match budget {
            "rows" => bad.rows = 0,
            "routes" => bad.routes = 0,
            "sessions" => bad.sessions = 0,
            "bytes" => bad.frame_bytes = usize::MAX,
            "scratch" => bad.scratch_bytes = 1,
            "resident" => bad.resident_bytes_per_owner = 1,
            _ => bad.timeout_ms = 0,
        }
        assert!(
            Placement::from_gguf(&path, 0, vec![0; 4], bad).is_err(),
            "{budget}"
        );
    }
    for owners in [vec![usize::MAX; 4], vec![1; 4], vec![0; 3]] {
        assert!(Placement::from_gguf(&path, 0, owners, limits()).is_err());
    }
    let (mut coordinator, handles) = start(&path, false);
    let input = Tensor::zeros((1, 8), candle_core::DType::F32, &Device::Cpu).unwrap();
    let ids = Tensor::new(&[[4u32]], &Device::Cpu).unwrap();
    let weights = Tensor::new(&[[1f32]], &Device::Cpu).unwrap();
    let row = [RowId {
        session: 1,
        token: 0,
    }];
    assert!(coordinator.forward(&row, &input, &ids, &weights).is_err());
    assert!(coordinator
        .forward(
            &row,
            &input,
            &Tensor::new(&[[0u32]], &Device::Cpu).unwrap(),
            &weights
        )
        .unwrap_err()
        .to_string()
        .contains("job failed"));
    drop(coordinator);
    for handle in handles {
        assert!(handle.join().unwrap().is_err());
    }
    common::write_tiny_qwen_gguf(&path, "qwen2moe");
    assert!(Placement::from_gguf(&path, 0, vec![0; 4], limits())
        .unwrap_err()
        .to_string()
        .contains("shared"));
    assert!(ExpertSet::load(&path, plan, 0, None).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}
