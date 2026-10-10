//! Experimental CPU Qwen3 pipeline: static contiguous stages, local KV, and
//! binary f32 activations over authenticated TCP. One coordinator connection
//! per worker/job; a disconnect destroys every session on that worker.
//!
//! Authentication does not encrypt prompts or activations. Use a private network
//! or an encrypted tunnel. A fresh job UUID and key are required after failure.

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::sync_channel,
        Arc,
    },
    time::{Duration, Instant},
};

use anyhow::{bail, ensure, Context, Result};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{
    gguf_ext,
    quantized_qwen::{CpuPipelineWeights, LayerState},
};

const MAX_HEADER: usize = 64 * 1024;
const MAX_VALUES: usize = 16 * 1024 * 1024;
const VERSION: u32 = 1;
/// Largest slice frame payload; bounds each receive buffer while a stage's
/// weights stream from the controller.
const SLICE_CHUNK: usize = 1024 * 1024;

/// Limits are part of the negotiated plan, not unbounded runtime hints.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limits {
    pub context: usize,
    pub chunk: usize,
    pub sessions: usize,
    pub queue_depth: usize,
    pub batch: usize,
    pub timeout_ms: u64,
    pub coordinator_memory_budget: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            context: 4096,
            chunk: 128,
            sessions: 8,
            queue_depth: 2,
            batch: 8,
            timeout_ms: 30_000,
            coordinator_memory_budget: 256 * 1024 * 1024,
        }
    }
}

impl Limits {
    fn validate(&self) -> Result<()> {
        ensure!(
            (1..=262_144).contains(&self.context),
            "invalid context limit"
        );
        ensure!(
            (1..=1024).contains(&self.chunk) && self.chunk <= self.context,
            "invalid chunk limit"
        );
        ensure!((1..=128).contains(&self.sessions), "invalid session limit");
        ensure!((1..=128).contains(&self.queue_depth), "invalid queue depth");
        ensure!((1..=128).contains(&self.batch), "invalid batch limit");
        ensure!((1..=300_000).contains(&self.timeout_ms), "invalid timeout");
        ensure!(
            self.coordinator_memory_budget > 0,
            "empty coordinator memory budget"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stage {
    pub start: usize,
    pub end: usize,
    /// Planning reservation; includes weights, KV growth and tensor workspace.
    /// This is not an operating-system RSS quota (see README).
    pub memory_budget: u64,
}

/// Fixed model identity and wire schema: Qwen3 dense, [1, rows, hidden], f32 LE.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub version: u32,
    pub model_sha256: String,
    pub layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub limits: Limits,
    pub stages: Vec<Stage>,
}

fn checked_product(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(1u64, |a, b| {
        a.checked_mul(*b).context("pipeline memory size overflow")
    })
}

/// Dense element width forced on quantized matrices by candle's
/// `CANDLE_DEQUANTIZE_ALL` (f32) or `CANDLE_DEQUANTIZE_ALL_F16` (f16).
fn dequantize_width() -> Option<u64> {
    let set = |name| std::env::var(name).is_ok_and(|s| !s.is_empty() && s != "0");
    if set("CANDLE_DEQUANTIZE_ALL") {
        Some(4)
    } else if set("CANDLE_DEQUANTIZE_ALL_F16") {
        Some(2)
    } else {
        None
    }
}

/// Bytes retained by a matrix loaded through `QMatMul::from_arc`: floating
/// GGUF types become f32, quantized types stay compact unless forced dense.
fn matrix_bytes(dtype: u32, elems: usize, encoded: u64, dequantize: Option<u64>) -> Result<u64> {
    let width = if matches!(dtype, 0 | 1 | 30) {
        4
    } else if let Some(width) = dequantize {
        width
    } else {
        return Ok(encoded);
    };
    Ok(encoded.max(checked_product(&[elems as u64, width])?))
}

fn checked_sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0u64, |a, b| {
        a.checked_add(*b).context("pipeline memory size overflow")
    })
}

fn model_header(path: &Path) -> Result<(gguf_ext::GgufHeader, String)> {
    let mut reader = BufReader::new(File::open(path)?);
    let header = gguf_ext::read_header(&mut reader)?;
    ensure!(
        header.architecture().as_deref() == Some("qwen3"),
        "pipeline supports only CPU Qwen3 dense"
    );
    // Full-file identity avoids accepting different weights with identical headers.
    // Startup scan is bounded but deliberately not lazy; weight loading remains so.
    reader.seek(SeekFrom::Start(0))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok((header, format!("{:x}", hash.finalize())))
}

impl Plan {
    /// Inspect a local immutable GGUF and divide at explicit layer boundaries.
    /// `ends` includes the final layer count; budgets correspond to stages.
    pub fn from_gguf(
        path: impl AsRef<Path>,
        ends: &[usize],
        budgets: &[u64],
        limits: Limits,
    ) -> Result<Self> {
        limits.validate()?;
        let (header, digest) = model_header(path.as_ref())?;
        let meta = crate::gguf_meta::Meta::new(&header.metadata, "qwen3");
        ensure!(
            limits.context <= meta.u32("context_length")? as usize,
            "context exceeds model limit"
        );
        let layers = meta.u32("block_count")? as usize;
        ensure!(
            meta.u32_or("nextn_predict_layers", 0) == 0 && meta.u32_or("expert_count", 0) == 0,
            "unsupported Qwen3 configuration"
        );
        let embedding = header
            .tensors
            .get("token_embd.weight")
            .context("missing embedding")?;
        ensure!(embedding.dims.len() == 2, "invalid embedding shape");
        let hidden = meta.u32("embedding_length")? as usize;
        ensure!(embedding.dims[1] == hidden, "embedding width mismatch");
        ensure!(
            ends.len() == budgets.len(),
            "one memory budget per stage required"
        );
        let mut start = 0;
        let stages = ends
            .iter()
            .zip(budgets)
            .map(|(&end, &memory_budget)| {
                let stage = Stage {
                    start,
                    end,
                    memory_budget,
                };
                start = end;
                stage
            })
            .collect();
        let plan = Self {
            version: VERSION,
            model_sha256: digest,
            layers,
            hidden,
            vocab: embedding.dims[0],
            limits,
            stages,
        };
        plan.validate()?;
        for rank in 0..plan.stages.len() {
            plan.check_memory(&header, rank)?;
        }
        Ok(plan)
    }

    pub fn validate(&self) -> Result<()> {
        self.limits.validate()?;
        ensure!(
            self.version == VERSION
                && self.model_sha256.len() == 64
                && self.model_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid model identity/version"
        );
        ensure!(
            self.layers > 0 && self.hidden > 0 && self.vocab > 0,
            "invalid model dimensions"
        );
        ensure!((1..=64).contains(&self.stages.len()), "invalid stage count");
        ensure!(
            self.values_limit()? <= MAX_VALUES,
            "activation frame exceeds limit"
        );
        let mut start = 0;
        for stage in &self.stages {
            ensure!(
                stage.start == start && stage.end > start && stage.end <= self.layers,
                "stages must cover contiguous nonempty layer ranges"
            );
            ensure!(stage.memory_budget > 0, "empty memory budget");
            start = stage.end;
        }
        ensure!(start == self.layers, "incomplete layer coverage");
        // Include bounded stage channels, each active RPC's encode/decode copies,
        // queued token controls, and retained result logits for the whole batch.
        let slots = checked_sum(&[
            checked_product(&[self.stages.len() as u64, self.limits.queue_depth as u64 + 6])?,
            self.limits.batch as u64,
        ])?;
        let frames = checked_product(&[slots, self.values_limit()? as u64, 4])?;
        let controls = checked_product(&[slots, MAX_HEADER as u64])?;
        let outputs = checked_product(&[self.limits.batch as u64, self.vocab as u64, 4])?;
        ensure!(
            checked_sum(&[frames, controls, outputs])? <= self.limits.coordinator_memory_budget,
            "coordinator buffers exceed memory budget; reduce chunk/batch/queue depth"
        );
        Ok(())
    }

    fn values_limit(&self) -> Result<usize> {
        Ok(self
            .limits
            .chunk
            .checked_mul(self.hidden)
            .context("activation size overflow")?
            .max(self.vocab))
    }

    /// `slice`: a stage's subset file, where only the first and (tied) last
    /// stages carry the embedding table.
    fn check_header(&self, header: &gguf_ext::GgufHeader, slice: bool) -> Result<()> {
        ensure!(
            header.architecture().as_deref() == Some("qwen3"),
            "pipeline supports only CPU Qwen3 dense"
        );
        let meta = crate::gguf_meta::Meta::new(&header.metadata, "qwen3");
        let emb = header.tensors.get("token_embd.weight");
        ensure!(slice || emb.is_some(), "missing embedding");
        ensure!(
            self.layers == meta.u32("block_count")? as usize
                && self.hidden == meta.u32("embedding_length")? as usize
                && emb.is_none_or(|emb| emb.dims == [self.vocab, self.hidden])
                && self.limits.context <= meta.u32("context_length")? as usize
                && meta.u32_or("nextn_predict_layers", 0) == 0
                && meta.u32_or("expert_count", 0) == 0,
            "plan/model dimensions or capabilities mismatch"
        );
        Ok(())
    }

    /// Whether stage `rank` loads tensor `name`: its own blocks, plus the
    /// embedding on the first stage and the head on the last.
    fn owns(&self, header: &gguf_ext::GgufHeader, rank: usize, name: &str) -> bool {
        let stage = &self.stages[rank];
        if let Some(rest) = name.strip_prefix("blk.") {
            let layer = rest.split('.').next().and_then(|s| s.parse::<usize>().ok());
            layer.is_some_and(|l| (stage.start..stage.end).contains(&l))
        } else {
            (stage.start == 0 && name == "token_embd.weight")
                || (stage.end == self.layers
                    && (name.starts_with("output.")
                        || name.starts_with("output_norm.")
                        || (name == "token_embd.weight"
                            && !header.tensors.contains_key("output.weight"))))
        }
    }

    /// The tensors stage `rank` loads, in file order.
    fn stage_tensors(&self, header: &gguf_ext::GgufHeader, rank: usize) -> Vec<String> {
        let mut names: Vec<_> = header
            .tensors
            .iter()
            .filter(|(name, _)| self.owns(header, rank, name))
            .map(|(name, info)| (info.offset, name.clone()))
            .collect();
        names.sort();
        names.into_iter().map(|(_, name)| name).collect()
    }

    fn check_memory(&self, header: &gguf_ext::GgufHeader, rank: usize) -> Result<u64> {
        let stage = &self.stages[rank];
        let dequantize = dequantize_width();
        let mut weights = 0u64;
        let mut load_scratch = 0u64;
        let mut intermediate = self.hidden as u64;
        for (name, tensor) in &header.tensors {
            if self.owns(header, rank, name) {
                let elems = tensor
                    .dims
                    .iter()
                    .try_fold(1usize, |a, b| a.checked_mul(*b))
                    .context("tensor size overflow")?;
                let encoded = gguf_ext::type_size_bytes(tensor.dtype, elems)
                    .context("unsupported tensor storage format")?
                    as u64;
                let dense = checked_product(&[elems as u64, 4])?;
                // Norms/biases expand to f32. CPU embedding tables stay compact
                // (floating tables are budgeted dense); matrices, including a
                // tied head, follow `QMatMul::from_arc`.
                let bytes = if tensor.dims.len() == 1 {
                    encoded.max(dense)
                } else {
                    let embedding = stage.start == 0 && name == "token_embd.weight";
                    let matmul = name != "token_embd.weight"
                        || (stage.end == self.layers
                            && !header.tensors.contains_key("output.weight"));
                    let table = if embedding {
                        if matches!(tensor.dtype, 0 | 1 | 30) {
                            encoded.max(dense)
                        } else {
                            encoded
                        }
                    } else {
                        0
                    };
                    let matrix = if matmul {
                        matrix_bytes(tensor.dtype, elems, encoded, dequantize)?
                    } else {
                        0
                    };
                    checked_sum(&[table, matrix])?
                };
                weights = checked_sum(&[weights, bytes])?;
                load_scratch = load_scratch.max(encoded);
                if name.starts_with("blk.") {
                    intermediate = intermediate.max(*tensor.dims.iter().max().unwrap_or(&0) as u64);
                }
            }
        }
        let meta = crate::gguf_meta::Meta::new(&header.metadata, "qwen3");
        let heads = meta.u32("attention.head_count")? as u64;
        ensure!(heads > 0, "invalid head count");
        let kv_heads = meta.u32_or("attention.head_count_kv", heads as u32) as u64;
        let dim = meta.u32_or("attention.key_length", (self.hidden as u64 / heads) as u32) as u64;
        // Geometric KV growth can transiently retain old and new buffers.
        let kv = checked_product(&[
            self.limits.sessions as u64,
            (stage.end - stage.start) as u64,
            2,
            kv_heads,
            dim,
            (self.limits.context.max(16)) as u64,
            4,
            4,
        ])?;
        let attention = checked_product(&[
            heads,
            self.limits.chunk as u64,
            self.limits.context as u64,
            4,
            8,
        ])?;
        let repeated_kv = checked_product(&[heads, dim, self.limits.context as u64, 4, 4])?;
        let work = checked_product(&[intermediate, self.limits.chunk as u64, 4, 16])?;
        let frames = checked_product(&[
            checked_sum(&[
                checked_product(&[self.values_limit()? as u64, 4])?,
                MAX_HEADER as u64,
            ])?,
            6,
        ])?;
        let rope = checked_product(&[meta.u32("context_length")? as u64, dim, 4])?;
        let reserved = checked_sum(&[
            weights,
            load_scratch,
            kv,
            repeated_kv,
            attention,
            work,
            frames,
            rope,
        ])?;
        ensure!(
            reserved <= stage.memory_budget,
            "stage {rank} planning reservation {reserved} exceeds budget {}",
            stage.memory_budget
        );
        Ok(reserved)
    }
}

/// Per-worker timings and wire volume, accumulated over the job.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub calls: u64,
    pub compute_ns: u64,
    pub rpc_ns: u64,
    pub codec_ns: u64,
    pub input_wait_ns: u64,
    pub downstream_wait_ns: u64,
    pub sent_bytes: u64,
    pub received_bytes: u64,
}

#[derive(Serialize, Deserialize)]
enum Command {
    Hello {
        plan: Plan,
        rank: usize,
    },
    Open {
        id: u64,
    },
    Forward {
        id: u64,
        sequence: u64,
        offset: usize,
        tokens: Vec<u32>,
    },
    Close {
        id: u64,
    },
    Stop,
    Reply {
        compute_ns: u64,
    },
    // Controller-managed workers (see [`serve_agent`]): the controller asks
    // for capacity, assigns a stage, then streams that stage's slice.
    Probe,
    Capacity {
        available_bytes: u64,
    },
    Assign {
        plan: Plan,
        rank: usize,
        slice: String,
        bytes: u64,
    },
    Ready {
        cached: bool,
    },
    /// Raw slice bytes travel in the frame payload instead of activations.
    Chunk,
    Commit {
        sha256: String,
    },
    Failed {
        message: String,
    },
}

struct Packet {
    command: Command,
    values: Vec<f32>,
    bytes: Vec<u8>,
}

impl Packet {
    fn control(command: Command) -> Self {
        Self {
            command,
            values: Vec::new(),
            bytes: Vec::new(),
        }
    }

    fn values(command: Command, values: Vec<f32>) -> Self {
        Self {
            command,
            values,
            bytes: Vec::new(),
        }
    }
}

/// Framing: length, JSON control header length, JSON control, binary LE f32,
/// HMAC-SHA256. Direction/job/rank/monotonic counter bind every frame.
struct Wire {
    stream: TcpStream,
    key: Vec<u8>,
    domain: &'static [u8],
    job: Uuid,
    rank: usize,
    client: bool,
    tx: u64,
    rx: u64,
    values_limit: usize,
    bytes_limit: usize,
    timeout: Duration,
    metrics: Metrics,
}

impl Wire {
    fn new(
        stream: TcpStream,
        key: &[u8],
        job: Uuid,
        rank: usize,
        client: bool,
        plan: &Plan,
    ) -> Result<Self> {
        ensure!(
            key.len() >= 32 && job.get_version_num() == 4,
            "use a 32-byte key and fresh v4 job UUID"
        );
        stream.set_nodelay(true)?;
        let timeout = Some(Duration::from_millis(plan.limits.timeout_ms));
        stream.set_read_timeout(timeout)?;
        stream.set_write_timeout(timeout)?;
        Ok(Self {
            stream,
            key: key.to_vec(),
            domain: b"joshua-qwen3-pipeline-v1",
            job,
            rank,
            client,
            tx: 0,
            rx: 0,
            values_limit: plan.values_limit()?,
            bytes_limit: 0,
            timeout: timeout.expect("configured timeout"),
            metrics: Metrics::default(),
        })
    }

    /// A controller-managed connection: no plan yet, bound to the worker's
    /// fresh per-connection `nonce` instead of a preshared job UUID, so a
    /// recorded connection cannot be replayed against a later one.
    fn agent(
        stream: TcpStream,
        key: &[u8],
        nonce: Uuid,
        client: bool,
        timeout: Duration,
    ) -> Result<Self> {
        ensure!(key.len() >= 32, "use a 32-byte key");
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        Ok(Self {
            stream,
            key: key.to_vec(),
            domain: b"joshua-qwen3-agent-v1",
            job: nonce,
            rank: 0,
            client,
            tx: 0,
            rx: 0,
            values_limit: 0,
            bytes_limit: SLICE_CHUNK,
            timeout,
            metrics: Metrics::default(),
        })
    }

    /// Adopt an assigned plan's activation limit and timeout.
    fn configure(&mut self, plan: &Plan) -> Result<()> {
        self.values_limit = plan.values_limit()?;
        self.timeout = Duration::from_millis(plan.limits.timeout_ms);
        self.stream.set_read_timeout(Some(self.timeout))?;
        self.stream.set_write_timeout(Some(self.timeout))?;
        Ok(())
    }

    fn mac(&self, client: bool, sequence: u64, payload: &[u8]) -> Result<Hmac<Sha256>> {
        let mut mac = Hmac::<Sha256>::new_from_slice(&self.key).map_err(anyhow::Error::msg)?;
        mac.update(self.domain);
        mac.update(self.job.as_bytes());
        mac.update(&(self.rank as u64).to_le_bytes());
        mac.update(&[u8::from(client)]);
        mac.update(&sequence.to_le_bytes());
        mac.update(&(payload.len() as u64).to_le_bytes());
        mac.update(payload);
        Ok(mac)
    }

    fn send(&mut self, packet: &Packet) -> Result<()> {
        let t = Instant::now();
        ensure!(
            packet.values.len() <= self.values_limit && packet.values.iter().all(|v| v.is_finite()),
            "invalid output payload"
        );
        ensure!(
            packet.bytes.len() <= self.bytes_limit
                && (packet.bytes.is_empty() || matches!(packet.command, Command::Chunk))
                && (packet.values.is_empty() || packet.bytes.is_empty()),
            "invalid byte payload"
        );
        let header = serde_json::to_vec(&packet.command)?;
        ensure!(header.len() <= MAX_HEADER, "control header too large");
        let mut body =
            Vec::with_capacity(4 + header.len() + packet.values.len() * 4 + packet.bytes.len());
        body.extend_from_slice(&(header.len() as u32).to_le_bytes());
        body.extend_from_slice(&header);
        for value in &packet.values {
            body.extend_from_slice(&value.to_le_bytes());
        }
        body.extend_from_slice(&packet.bytes);
        let tag = self
            .mac(self.client, self.tx, &body)?
            .finalize()
            .into_bytes();
        self.metrics.codec_ns += elapsed_ns(t);
        let deadline = Instant::now() + self.timeout;
        write_until(
            &mut self.stream,
            &(body.len() as u32).to_le_bytes(),
            deadline,
        )?;
        write_until(&mut self.stream, &body, deadline)?;
        write_until(&mut self.stream, &tag, deadline)?;
        self.metrics.sent_bytes += (body.len() + 36) as u64;
        self.tx = self.tx.checked_add(1).context("wire sequence exhausted")?;
        Ok(())
    }

    fn receive(&mut self) -> Result<Packet> {
        let deadline = Instant::now() + self.timeout;
        let mut length = [0u8; 4];
        read_until(&mut self.stream, &mut length, deadline)?;
        let length = u32::from_le_bytes(length) as usize;
        ensure!(
            (4..=4 + MAX_HEADER + (self.values_limit * 4).max(self.bytes_limit)).contains(&length),
            "invalid frame size"
        );
        let mut body = vec![0u8; length];
        read_until(&mut self.stream, &mut body, deadline)?;
        let mut tag = [0u8; 32];
        read_until(&mut self.stream, &mut tag, deadline)?;
        let t = Instant::now();
        self.mac(!self.client, self.rx, &body)?
            .verify_slice(&tag)
            .map_err(|_| anyhow::anyhow!("pipeline authentication/replay failure"))?;
        let h = u32::from_le_bytes(body[..4].try_into()?) as usize;
        ensure!(
            h <= MAX_HEADER && h <= body.len() - 4,
            "invalid header size"
        );
        let command: Command = serde_json::from_slice(&body[4..4 + h])?;
        let payload = &body[4 + h..];
        let (values, bytes) = if matches!(command, Command::Chunk) {
            ensure!(
                payload.len() <= self.bytes_limit,
                "invalid byte payload length"
            );
            (Vec::new(), payload.to_vec())
        } else {
            ensure!(
                payload.len().is_multiple_of(4) && payload.len() / 4 <= self.values_limit,
                "invalid f32 payload length"
            );
            let values: Vec<f32> = payload
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().expect("four bytes")))
                .collect();
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "nonfinite input activation"
            );
            (values, Vec::new())
        };
        self.metrics.codec_ns += elapsed_ns(t);
        self.metrics.received_bytes += (body.len() + 36) as u64;
        self.rx = self.rx.checked_add(1).context("wire sequence exhausted")?;
        Ok(Packet {
            command,
            values,
            bytes,
        })
    }

    /// Block without a deadline until the next frame's first byte arrives, so
    /// an idle authenticated connection keeps its sessions; `receive` then
    /// bounds the frame itself.
    fn wait_for_frame(&mut self) -> Result<()> {
        self.stream.set_read_timeout(None)?;
        loop {
            match self.stream.peek(&mut [0u8; 1]) {
                Ok(0) => return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into()),
                Ok(_) => return Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn rpc(&mut self, packet: &Packet) -> Result<Packet> {
        let t = Instant::now();
        self.send(packet)?;
        let response = self.receive()?;
        let Command::Reply { compute_ns } = response.command else {
            bail!("unexpected worker reply");
        };
        self.metrics.rpc_ns += elapsed_ns(t);
        self.metrics.compute_ns += compute_ns;
        self.metrics.calls += 1;
        Ok(response)
    }
}

fn remaining(deadline: Instant) -> std::io::Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "pipeline frame deadline"))
}

fn read_until(
    stream: &mut TcpStream,
    mut bytes: &mut [u8],
    deadline: Instant,
) -> std::io::Result<()> {
    while !bytes.is_empty() {
        stream.set_read_timeout(Some(remaining(deadline)?))?;
        match stream.read(bytes) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => bytes = &mut bytes[n..],
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn write_until(stream: &mut TcpStream, mut bytes: &[u8], deadline: Instant) -> std::io::Result<()> {
    while !bytes.is_empty() {
        stream.set_write_timeout(Some(remaining(deadline)?))?;
        match stream.write(bytes) {
            Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
            Ok(n) => bytes = &bytes[n..],
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn elapsed_ns(t: Instant) -> u64 {
    t.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

struct Session {
    state: Vec<LayerState>,
    position: usize,
    sequence: u64,
}

/// A CPU stage with shared immutable local weights and bounded session state.
pub struct Worker {
    plan: Plan,
    rank: usize,
    weights: CpuPipelineWeights,
    sessions: HashMap<u64, Session>,
    pub reserved_bytes: u64,
}

impl Worker {
    /// The GGUF must remain immutable for the worker's lifetime, as with Engine.
    /// `mmap=false` preserves the copying/streamed-load fallback, limited to the
    /// assigned layers and owned endpoints rather than the full model.
    pub fn load(path: impl AsRef<Path>, plan: Plan, rank: usize, mmap: bool) -> Result<Self> {
        plan.validate()?;
        ensure!(rank < plan.stages.len(), "invalid stage rank");
        let (header, digest) = model_header(path.as_ref())?;
        ensure!(digest == plan.model_sha256, "model checksum mismatch");
        plan.check_header(&header, false)?;
        Self::from_header(path.as_ref(), header, plan, rank, mmap)
    }

    /// Load a stage from the subset file a controller streamed to this worker.
    /// Its integrity was checked against the controller's digest on receipt;
    /// the full-model checksum in `plan` names the model it was cut from.
    fn load_slice(path: &Path, plan: Plan, rank: usize, mmap: bool) -> Result<Self> {
        plan.validate()?;
        ensure!(rank < plan.stages.len(), "invalid stage rank");
        let header = gguf_ext::read_header(&mut BufReader::new(File::open(path)?))?;
        plan.check_header(&header, true)?;
        ensure!(
            header
                .tensors
                .keys()
                .all(|name| plan.owns(&header, rank, name)),
            "slice carries tensors outside its stage"
        );
        Self::from_header(path, header, plan, rank, mmap)
    }

    fn from_header(
        path: &Path,
        header: gguf_ext::GgufHeader,
        plan: Plan,
        rank: usize,
        mmap: bool,
    ) -> Result<Self> {
        let reserved_bytes = plan.check_memory(&header, rank)?;
        let file = File::open(path)?;
        let mapping = if mmap {
            // SAFETY: caller keeps the local GGUF immutable while weights borrow it.
            Some(Arc::new(unsafe { memmap2::Mmap::map(&file) }?))
        } else {
            None
        };
        let mut reader = BufReader::new(file);
        let stage = &plan.stages[rank];
        let weights =
            CpuPipelineWeights::load(header, &mut reader, mapping, stage.start..stage.end)?;
        Ok(Self {
            plan,
            rank,
            weights,
            sessions: HashMap::new(),
            reserved_bytes,
        })
    }

    /// Serve exactly one authenticated coordinator connection. Connections
    /// whose first frame is not an authenticated `Hello` for this plan and
    /// rank (including malformed or stalled frames) are dropped and the
    /// worker keeps listening. After the handshake, any protocol or
    /// compute failure closes the connection and drops every session; do not
    /// reuse a failed worker job. Idle sessions do not expire; TCP keepalive
    /// detects a vanished coordinator.
    pub fn serve(mut self, listener: TcpListener, job: Uuid, key: &[u8]) -> Result<()> {
        ensure!(
            key.len() >= 32 && job.get_version_num() == 4,
            "use a 32-byte key and fresh v4 job UUID"
        );
        let mut wire = loop {
            let (stream, _) = listener.accept()?;
            let Ok(mut wire) = Wire::new(stream, key, job, self.rank, false, &self.plan) else {
                continue;
            };
            match wire.receive() {
                Ok(Packet {
                    command: Command::Hello { plan, rank },
                    values,
                    ..
                }) if plan == self.plan && rank == self.rank && values.is_empty() => break wire,
                _ => {
                    let _ = wire.stream.shutdown(Shutdown::Both);
                }
            }
        };
        let result = (|| {
            wire.send(&Packet::control(Command::Reply { compute_ns: 0 }))?;
            self.run(&mut wire)
        })();
        let _ = wire.stream.shutdown(Shutdown::Both);
        self.sessions.clear();
        result
    }

    /// The session loop after the handshake, until `Stop` or a failure.
    fn run(&mut self, wire: &mut Wire) -> Result<()> {
        // Best effort; keepalive idle time has one-second granularity.
        keepalive(wire);
        loop {
            wire.wait_for_frame()?;
            let packet = wire.receive()?;
            if matches!(packet.command, Command::Stop) {
                ensure!(packet.values.is_empty(), "unexpected stop payload");
                wire.send(&Packet::control(Command::Reply { compute_ns: 0 }))?;
                return Ok(());
            }
            let t = Instant::now();
            let values = self.execute(packet)?;
            wire.send(&Packet::values(
                Command::Reply {
                    compute_ns: elapsed_ns(t),
                },
                values,
            ))?;
        }
    }

    fn execute(&mut self, packet: Packet) -> Result<Vec<f32>> {
        match packet.command {
            Command::Open { id } => {
                ensure!(
                    packet.values.is_empty() && !self.sessions.contains_key(&id),
                    "duplicate/invalid session open"
                );
                ensure!(
                    self.sessions.len() < self.plan.limits.sessions,
                    "session capacity exceeded"
                );
                self.sessions.insert(
                    id,
                    Session {
                        state: self.weights.new_state(),
                        position: 0,
                        sequence: 0,
                    },
                );
                Ok(Vec::new())
            }
            Command::Close { id } => {
                ensure!(
                    packet.values.is_empty() && self.sessions.remove(&id).is_some(),
                    "unknown/invalid session close"
                );
                Ok(Vec::new())
            }
            Command::Forward {
                id,
                sequence,
                offset,
                tokens,
            } => {
                ensure!(
                    !tokens.is_empty() && tokens.len() <= self.plan.limits.chunk,
                    "invalid token chunk"
                );
                ensure!(
                    tokens.iter().all(|&t| (t as usize) < self.plan.vocab),
                    "token outside vocabulary"
                );
                let end = offset
                    .checked_add(tokens.len())
                    .context("position overflow")?;
                ensure!(end <= self.plan.limits.context, "context exhausted");
                let expected = if self.rank == 0 {
                    0
                } else {
                    tokens.len() * self.plan.hidden
                };
                ensure!(packet.values.len() == expected, "activation shape mismatch");
                let session = self.sessions.get_mut(&id).context("unknown session")?;
                ensure!(
                    offset == session.position && sequence == session.sequence,
                    "out-of-order or duplicate forward"
                );
                let output = self.weights.forward(
                    &mut session.state,
                    &tokens,
                    packet.values,
                    offset,
                    self.plan.hidden,
                )?;
                session.position = end;
                session.sequence = session
                    .sequence
                    .checked_add(1)
                    .context("session sequence exhausted")?;
                Ok(output)
            }
            _ => bail!("unexpected worker command"),
        }
    }
}

fn keepalive(wire: &Wire) {
    // Best effort; keepalive idle time has one-second granularity.
    let keepalive = wire.timeout.max(Duration::from_secs(1));
    let _ = socket2::SockRef::from(&wire.stream)
        .set_tcp_keepalive(&socket2::TcpKeepalive::new().with_time(keepalive));
}

// ─── Controller-managed workers ──────────────────────────────────────────────

/// Settings for a worker started without a model (see [`serve_agent`]).
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// Keep received stage slices here, named by slice identity, and reuse a
    /// slice when a controller assigns the same stage of the same model
    /// again (after re-hashing it). `None` stores each slice in the system
    /// temporary directory and deletes it on unload.
    pub cache_dir: Option<PathBuf>,
    /// Map the received slice instead of copying its weights into memory.
    pub mmap: bool,
    /// Bound on each frame before a stage is assigned, and on the
    /// controller's first frame.
    pub timeout: Duration,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            cache_dir: None,
            mmap: true,
            timeout: Duration::from_secs(30),
        }
    }
}

/// Serve controllers one at a time, forever. Each controller connection may
/// assign this worker one pipeline stage and stream it the stage's slice of
/// the model; the worker then serves that stage until the controller stops
/// or disconnects, unloads it, and waits for the next controller.
pub fn serve_agent(listener: &TcpListener, key: &[u8], config: &AgentConfig) -> Result<()> {
    ensure!(key.len() >= 32, "use a 32-byte key");
    loop {
        let (stream, peer) = listener.accept()?;
        match serve_controller(stream, key, config) {
            Ok(()) => tracing::info!(%peer, "controller released this worker"),
            Err(error) => tracing::warn!(%peer, "controller connection ended: {error:#}"),
        }
    }
}

/// Handle one controller connection; see [`serve_agent`].
pub fn serve_controller(mut stream: TcpStream, key: &[u8], config: &AgentConfig) -> Result<()> {
    ensure!(key.len() >= 32, "use a 32-byte key");
    let nonce = *Uuid::new_v4().as_bytes();
    let deadline = Instant::now() + config.timeout;
    write_until(&mut stream, &nonce, deadline)?;
    let mut theirs = [0u8; 16];
    read_until(&mut stream, &mut theirs, deadline)?;
    let mut wire = Wire::agent(
        stream,
        key,
        session_id(&nonce, &theirs),
        false,
        config.timeout,
    )?;
    // Dropped in order: the stage's weights, then its temporary file.
    let mut loaded: Option<(Worker, RemoveOnDrop)> = None;
    let result = (|| -> Result<()> {
        // The first frame is bounded so an unauthenticated peer cannot hold
        // this single-controller worker; an authenticated one may idle.
        let mut first = true;
        loop {
            if !first {
                wire.wait_for_frame()?;
            }
            first = false;
            let packet = wire.receive()?;
            match packet.command {
                Command::Probe => wire.send(&Packet::control(Command::Capacity {
                    available_bytes: crate::placement::available_ram_bytes().unwrap_or(0),
                }))?,
                Command::Assign {
                    plan,
                    rank,
                    slice,
                    bytes,
                } => {
                    let stage = match receive_stage(&mut wire, config, plan, rank, &slice, bytes) {
                        Ok(stage) => stage,
                        Err(error) => {
                            let _ = wire.send(&Packet::control(Command::Failed {
                                message: format!("{error:#}"),
                            }));
                            return Err(error);
                        }
                    };
                    tracing::info!(
                        rank,
                        reservation_bytes = stage.0.reserved_bytes,
                        "stage loaded from controller"
                    );
                    let (worker, _) = loaded.insert(stage);
                    return worker.run(&mut wire);
                }
                Command::Stop => {
                    return wire.send(&Packet::control(Command::Reply { compute_ns: 0 }));
                }
                _ => bail!("unexpected controller command"),
            }
        }
    })();
    let _ = wire.stream.shutdown(Shutdown::Both);
    drop(loaded);
    result
}

/// Both sides' per-connection randomness, so neither a recorded worker nor
/// a recorded controller can replay an earlier connection.
fn session_id(worker: &[u8; 16], controller: &[u8; 16]) -> Uuid {
    let digest = Sha256::new()
        .chain_update(b"joshua-qwen3-agent-session")
        .chain_update(worker)
        .chain_update(controller)
        .finalize();
    Uuid::from_bytes(digest[..16].try_into().expect("sixteen bytes"))
}

/// Deterministic name of a stage's slice: the model and the stage range
/// fix its content, since [`gguf_ext::subset_layout`] is deterministic.
fn slice_id(plan: &Plan, rank: usize) -> String {
    let stage = &plan.stages[rank];
    let digest = Sha256::new()
        .chain_update(b"joshua-stage-slice-v1")
        .chain_update(plan.model_sha256.as_bytes())
        .chain_update((plan.layers as u64).to_le_bytes())
        .chain_update((stage.start as u64).to_le_bytes())
        .chain_update((stage.end as u64).to_le_bytes())
        .finalize();
    format!("{digest:x}")
}

fn file_sha256(path: &Path) -> Result<(String, u64)> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024];
    let mut length = 0u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        length += n as u64;
    }
    Ok((format!("{:x}", hash.finalize()), length))
}

/// Accept a stage assignment: reuse a verified cached slice or receive it,
/// then load it. Returns the worker and the file to delete on unload.
fn receive_stage(
    wire: &mut Wire,
    config: &AgentConfig,
    plan: Plan,
    rank: usize,
    slice: &str,
    bytes: u64,
) -> Result<(Worker, RemoveOnDrop)> {
    plan.validate()?;
    ensure!(rank < plan.stages.len(), "invalid stage rank");
    ensure!(
        slice == slice_id(&plan, rank),
        "slice identity does not match the assigned stage"
    );
    ensure!(bytes > 0, "empty stage slice");
    wire.configure(&plan)?;
    let (path, temporary) = match &config.cache_dir {
        Some(dir) => {
            std::fs::create_dir_all(dir)?;
            (dir.join(format!("{slice}.gguf")), false)
        }
        None => (
            std::env::temp_dir().join(format!("joshua-slice-{}.gguf", wire.job)),
            true,
        ),
    };
    // Owns a temporary slice from here on, through every early return.
    let cleanup = RemoveOnDrop(temporary.then(|| path.clone()));
    let digest_path = path.with_extension("sha256");
    let cached = !temporary
        && std::fs::read_to_string(&digest_path).is_ok_and(|expected| {
            file_sha256(&path)
                .is_ok_and(|(digest, length)| digest == expected.trim() && length == bytes)
        });
    wire.send(&Packet::control(Command::Ready { cached }))?;
    if !cached {
        let partial = path.with_extension("partial");
        let received = (|| -> Result<()> {
            let mut file = std::io::BufWriter::new(File::create(&partial)?);
            let mut hash = Sha256::new();
            let mut length = 0u64;
            loop {
                let packet = wire.receive()?;
                match packet.command {
                    Command::Chunk => {
                        length = length
                            .checked_add(packet.bytes.len() as u64)
                            .filter(|&n| n <= bytes)
                            .context("stage slice longer than announced")?;
                        hash.update(&packet.bytes);
                        file.write_all(&packet.bytes)?;
                    }
                    Command::Commit { sha256 } => {
                        ensure!(length == bytes, "stage slice shorter than announced");
                        ensure!(
                            sha256 == format!("{:x}", hash.finalize()),
                            "stage slice checksum mismatch"
                        );
                        file.into_inner().map_err(|e| e.into_error())?.sync_all()?;
                        std::fs::rename(&partial, &path)?;
                        if !temporary {
                            std::fs::write(&digest_path, sha256)?;
                        }
                        return Ok(());
                    }
                    _ => bail!("unexpected command while receiving a stage"),
                }
            }
        })();
        if let Err(error) = received {
            let _ = std::fs::remove_file(&partial);
            return Err(error);
        }
    }
    let t = Instant::now();
    let worker = Worker::load_slice(&path, plan, rank, config.mmap)?;
    wire.send(&Packet::control(Command::Reply {
        compute_ns: elapsed_ns(t),
    }))?;
    Ok((worker, cleanup))
}

/// Deletes a temporary stage slice when dropped.
struct RemoveOnDrop(Option<PathBuf>);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        if let Some(path) = &self.0 {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// How a controller places a model on its workers (see [`Pipeline::deploy`]).
#[derive(Clone, Debug, Default)]
pub struct Deployment {
    /// Exclusive layer ends per worker, ending with the layer count. `None`
    /// splits layers in proportion to each worker's reported free memory.
    pub ends: Option<Vec<usize>>,
    pub limits: Limits,
}

/// Contiguous layer ranges whose weight bytes follow `capacities`; every
/// stage gets at least one layer. Unknown (zero) capacities split evenly.
fn split_layers(
    header: &gguf_ext::GgufHeader,
    layers: usize,
    capacities: &[u64],
) -> Result<Vec<usize>> {
    let n = capacities.len();
    ensure!(
        (1..=layers).contains(&n),
        "{n} workers cannot split {layers} layers"
    );
    let mut weights = vec![1u64; layers];
    for (name, info) in &header.tensors {
        let layer = name
            .strip_prefix("blk.")
            .and_then(|rest| rest.split('.').next())
            .and_then(|l| l.parse::<usize>().ok())
            .filter(|&l| l < layers);
        if let Some(layer) = layer {
            let size = gguf_ext::type_size_bytes(info.dtype, info.elem_count()).unwrap_or(0);
            weights[layer] = weights[layer].saturating_add(size as u64);
        }
    }
    let shares: Vec<f64> = if capacities.contains(&0) {
        vec![1.0; n]
    } else {
        capacities.iter().map(|&c| c as f64).collect()
    };
    let total_weight: f64 = weights.iter().map(|&w| w as f64).sum();
    let total_share: f64 = shares.iter().sum();
    let mut ends = Vec::with_capacity(n);
    let (mut end, mut filled, mut share) = (0usize, 0f64, 0f64);
    for (i, s) in shares[..n - 1].iter().enumerate() {
        share += s;
        let target = total_weight * share / total_share;
        let last = layers - (n - 1 - i);
        filled += weights[end] as f64;
        end += 1;
        while end < last && filled + weights[end] as f64 <= target {
            filled += weights[end] as f64;
            end += 1;
        }
        ends.push(end);
    }
    ends.push(layers);
    Ok(ends)
}

/// Frames a byte stream into bounded `Chunk` packets, hashing what it sends.
struct SliceSender<'a> {
    wire: &'a mut Wire,
    hash: Sha256,
    packet: Packet,
}

impl SliceSender<'_> {
    /// Append `length` bytes, each piece filled by `fill` (zeros if it
    /// leaves the piece untouched).
    fn push(
        &mut self,
        mut length: u64,
        mut fill: impl FnMut(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        while length > 0 {
            let old = self.packet.bytes.len();
            let n = (SLICE_CHUNK - old).min(length as usize);
            self.packet.bytes.resize(old + n, 0);
            fill(&mut self.packet.bytes[old..])?;
            length -= n as u64;
            if self.packet.bytes.len() == SLICE_CHUNK {
                self.flush()?;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if !self.packet.bytes.is_empty() {
            self.hash.update(&self.packet.bytes);
            self.wire.send(&self.packet)?;
            self.packet.bytes.clear();
        }
        Ok(())
    }
}

fn connect_agent(address: &SocketAddr, key: &[u8], timeout: Duration) -> Result<Wire> {
    let mut stream = TcpStream::connect_timeout(address, timeout)
        .with_context(|| format!("connecting to worker {address}"))?;
    let deadline = Instant::now() + timeout;
    let mut theirs = [0u8; 16];
    read_until(&mut stream, &mut theirs, deadline)?;
    let nonce = *Uuid::new_v4().as_bytes();
    write_until(&mut stream, &nonce, deadline)?;
    Wire::agent(stream, key, session_id(&theirs, &nonce), true, timeout)
}

/// Stream stage `rank`'s slice of `path` to its worker unless the worker has
/// it cached, then wait for the worker to load it.
fn push_stage(
    wire: &mut Wire,
    path: &Path,
    header: &gguf_ext::GgufHeader,
    plan: &Plan,
    rank: usize,
) -> Result<()> {
    let names = plan.stage_tensors(header, rank);
    let (head, ranges) = gguf_ext::subset_layout(header, &names)?;
    let bytes = ranges
        .iter()
        .try_fold(head.len() as u64, |n, (range, pad)| {
            n.checked_add(range.end - range.start)?
                .checked_add(*pad as u64)
        })
        .context("stage slice size overflow")?;
    wire.send(&Packet::control(Command::Assign {
        plan: plan.clone(),
        rank,
        slice: slice_id(plan, rank),
        bytes,
    }))?;
    let cached = match wire.receive()?.command {
        Command::Ready { cached } => cached,
        Command::Failed { message } => bail!("worker {rank} refused its stage: {message}"),
        _ => bail!("unexpected worker reply"),
    };
    wire.configure(plan)?;
    if !cached {
        let mut file = File::open(path)?;
        let mut out = SliceSender {
            wire,
            hash: Sha256::new(),
            packet: Packet {
                command: Command::Chunk,
                values: Vec::new(),
                bytes: Vec::with_capacity(SLICE_CHUNK),
            },
        };
        let mut rest = &head[..];
        out.push(head.len() as u64, |buf| {
            let (now, later) = rest.split_at(buf.len());
            buf.copy_from_slice(now);
            rest = later;
            Ok(())
        })?;
        for (range, pad) in &ranges {
            file.seek(SeekFrom::Start(range.start))?;
            out.push(range.end - range.start, |buf| Ok(file.read_exact(buf)?))?;
            out.push(*pad as u64, |_| Ok(()))?;
        }
        out.flush()?;
        let sha256 = format!("{:x}", out.hash.finalize());
        wire.send(&Packet::control(Command::Commit { sha256 }))?;
    }
    // Loading may outlast one frame timeout; keepalive detects a lost worker.
    keepalive(wire);
    wire.wait_for_frame()?;
    match wire.receive()?.command {
        Command::Reply { .. } => Ok(()),
        Command::Failed { message } => bail!("worker {rank} failed to load its stage: {message}"),
        _ => bail!("unexpected worker reply"),
    }
}

/// One bounded prefill chunk or independent sequence's decode token. Chunks
/// for a single session must appear in position order; decode choices depend
/// on earlier logits and should be submitted in successive batches.
pub struct Input {
    pub session: u64,
    pub offset: usize,
    pub tokens: Vec<u32>,
}

struct Job {
    index: usize,
    id: u64,
    sequence: u64,
    offset: usize,
    tokens: Vec<u32>,
    values: Vec<f32>,
}

/// Static coordinator. Activations return via the coordinator between stages;
/// workers overlap independent chunks using bounded FIFO channels. No retries
/// after mutation: a failed operation permanently invalidates this instance.
pub struct Pipeline {
    plan: Plan,
    stages: Vec<Wire>,
    sessions: HashMap<u64, (usize, u64)>,
    // Disallow id reuse within a job without an unbounded tombstone registry.
    next_id: u64,
    failed: bool,
    cancelled: Arc<AtomicBool>,
}

impl Pipeline {
    pub fn connect(plan: Plan, addresses: &[SocketAddr], job: Uuid, key: &[u8]) -> Result<Self> {
        plan.validate()?;
        ensure!(
            key.len() >= 32 && job.get_version_num() == 4,
            "use a 32-byte key and fresh v4 job UUID"
        );
        ensure!(
            addresses.len() == plan.stages.len(),
            "one address per stage required"
        );
        ensure!(
            addresses.iter().collect::<HashSet<_>>().len() == addresses.len(),
            "duplicate worker addresses"
        );
        let mut stages = Vec::with_capacity(addresses.len());
        for (rank, address) in addresses.iter().enumerate() {
            let stream =
                TcpStream::connect_timeout(address, Duration::from_millis(plan.limits.timeout_ms))?;
            let mut wire = Wire::new(stream, key, job, rank, true, &plan)?;
            wire.rpc(&Packet::control(Command::Hello {
                plan: plan.clone(),
                rank,
            }))?;
            stages.push(wire);
        }
        Ok(Self {
            plan,
            stages,
            sessions: HashMap::new(),
            next_id: 0,
            failed: false,
            cancelled: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Place the local GGUF at `model` on workers started without a model
    /// ([`serve_agent`]): plan stages from their free memory (or
    /// `deployment.ends`), stream each worker only its stage's tensors, and
    /// return once every stage is loaded. Model data comes only from this
    /// controller. Dropping or shutting down the pipeline unloads the stages.
    pub fn deploy(
        model: impl AsRef<Path>,
        workers: &[SocketAddr],
        key: &[u8],
        deployment: Deployment,
    ) -> Result<Self> {
        let path = model.as_ref();
        deployment.limits.validate()?;
        ensure!(key.len() >= 32, "use a 32-byte key");
        ensure!(
            (1..=64).contains(&workers.len()),
            "a pipeline needs 1 to 64 workers"
        );
        ensure!(
            workers.iter().collect::<HashSet<_>>().len() == workers.len(),
            "duplicate worker addresses"
        );
        let timeout = Duration::from_millis(deployment.limits.timeout_ms);
        let mut stages = workers
            .iter()
            .map(|address| connect_agent(address, key, timeout))
            .collect::<Result<Vec<_>>>()?;
        let mut capacities = Vec::with_capacity(stages.len());
        for (rank, wire) in stages.iter_mut().enumerate() {
            wire.send(&Packet::control(Command::Probe))?;
            match wire.receive()?.command {
                Command::Capacity { available_bytes } => capacities.push(available_bytes),
                Command::Failed { message } => bail!("worker {rank}: {message}"),
                _ => bail!("unexpected worker reply"),
            }
        }
        let header = gguf_ext::read_header(&mut BufReader::new(File::open(path)?))?;
        let ends = match deployment.ends {
            Some(ends) => ends,
            None => {
                let meta = crate::gguf_meta::Meta::new(&header.metadata, "qwen3");
                split_layers(&header, meta.u32("block_count")? as usize, &capacities)?
            }
        };
        // Unknown free memory leaves the stage unbounded by this check.
        let budgets: Vec<u64> = capacities
            .iter()
            .map(|&c| if c == 0 { u64::MAX } else { c })
            .collect();
        let plan = Plan::from_gguf(path, &ends, &budgets, deployment.limits)?;
        std::thread::scope(|scope| -> Result<()> {
            let handles: Vec<_> = stages
                .iter_mut()
                .enumerate()
                .map(|(rank, wire)| {
                    let (header, plan) = (&header, &plan);
                    scope.spawn(move || push_stage(wire, path, header, plan, rank))
                })
                .collect();
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("stage transfer thread panicked"))??;
            }
            Ok(())
        })?;
        Ok(Self {
            plan,
            stages,
            sessions: HashMap::new(),
            next_id: 0,
            failed: false,
            cancelled: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Open sessions; at most `plan().limits.sessions`.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    fn ready(&self) -> Result<()> {
        ensure!(
            !self.failed && !self.cancelled.load(Ordering::Acquire),
            "pipeline failed; restart every worker with a fresh job UUID"
        );
        Ok(())
    }

    fn abort(&mut self) {
        self.failed = true;
        self.cancelled.store(true, Ordering::Release);
        for stage in &self.stages {
            let _ = stage.stream.shutdown(Shutdown::Both);
        }
        self.sessions.clear();
    }

    fn control(&mut self, command: impl Fn() -> Command) -> Result<()> {
        self.ready()?;
        for stage in &mut self.stages {
            if let Err(error) = stage.rpc(&Packet::control(command())) {
                self.abort();
                return Err(error);
            }
        }
        Ok(())
    }

    pub fn open(&mut self) -> Result<u64> {
        self.ready()?;
        ensure!(
            self.sessions.len() < self.plan.limits.sessions,
            "session capacity exceeded"
        );
        let id = self.next_id;
        self.next_id = id.checked_add(1).context("session identifiers exhausted")?;
        self.control(|| Command::Open { id })?;
        self.sessions.insert(id, (0, 0));
        Ok(id)
    }

    /// Cancellation between batches. Active calls are bounded by socket timeouts;
    /// dropping the coordinator interrupts transport and clears worker sessions.
    pub fn close(&mut self, id: u64) -> Result<()> {
        self.ready()?;
        ensure!(self.sessions.contains_key(&id), "unknown session");
        self.control(|| Command::Close { id })?;
        self.sessions.remove(&id);
        Ok(())
    }

    /// Pipeline independent requests/chunks, returning last-position logits in
    /// input order. Validation finishes before any stage mutates KV.
    pub fn forward_batch(&mut self, inputs: &[Input]) -> Result<Vec<Vec<f32>>> {
        self.ready()?;
        ensure!(
            !inputs.is_empty() && inputs.len() <= self.plan.limits.batch,
            "invalid batch size"
        );
        let mut next = self.sessions.clone();
        let mut jobs = Vec::with_capacity(inputs.len());
        for (index, input) in inputs.iter().enumerate() {
            let (position, sequence) = next.get_mut(&input.session).context("unknown session")?;
            ensure!(input.offset == *position, "input offset is not contiguous");
            ensure!(
                !input.tokens.is_empty() && input.tokens.len() <= self.plan.limits.chunk,
                "invalid chunk size"
            );
            ensure!(
                input.tokens.iter().all(|&t| (t as usize) < self.plan.vocab),
                "token outside vocabulary"
            );
            let end = position
                .checked_add(input.tokens.len())
                .context("position overflow")?;
            ensure!(end <= self.plan.limits.context, "context exhausted");
            jobs.push(Job {
                index,
                id: input.session,
                sequence: *sequence,
                offset: input.offset,
                tokens: input.tokens.clone(),
                values: Vec::new(),
            });
            *position = end;
            *sequence = sequence.checked_add(1).context("sequence exhausted")?;
        }
        let stages = &mut self.stages;
        let depth = self.plan.limits.queue_depth;
        let n = inputs.len();
        let hidden = self.plan.hidden;
        let vocab = self.plan.vocab;
        // A separate feeder prevents full upstream/downstream queues from
        // deadlocking the result collector. Dropping a channel on failure
        // propagates backpressure release in both directions.
        let result = std::thread::scope(|scope| -> Result<Vec<Vec<f32>>> {
            let (send, mut receive) = sync_channel::<Job>(depth);
            scope.spawn(move || {
                for job in jobs {
                    if send.send(job).is_err() {
                        break;
                    }
                }
            });
            let count = stages.len();
            let mut handles = Vec::with_capacity(count);
            for (rank, stage) in stages.iter_mut().enumerate() {
                let (send, downstream) = sync_channel::<Job>(depth);
                let upstream = receive;
                receive = downstream;
                handles.push(scope.spawn(move || -> Result<()> {
                    loop {
                        let waiting = Instant::now();
                        let Ok(mut job) = upstream.recv() else {
                            break;
                        };
                        stage.metrics.input_wait_ns += elapsed_ns(waiting);
                        let response = stage.rpc(&Packet::values(
                            Command::Forward {
                                id: job.id,
                                sequence: job.sequence,
                                offset: job.offset,
                                tokens: job.tokens.clone(),
                            },
                            std::mem::take(&mut job.values),
                        ))?;
                        let expected = if rank + 1 == count {
                            vocab
                        } else {
                            job.tokens.len() * hidden
                        };
                        ensure!(
                            response.values.len() == expected,
                            "worker output shape mismatch"
                        );
                        job.values = response.values;
                        let waiting = Instant::now();
                        if send.send(job).is_err() {
                            break;
                        }
                        stage.metrics.downstream_wait_ns += elapsed_ns(waiting);
                    }
                    Ok(())
                }));
            }
            let mut outputs = vec![Vec::new(); n];
            let mut completed = 0;
            for job in receive {
                outputs[job.index] = job.values;
                completed += 1;
            }
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("pipeline stage thread panicked"))??;
            }
            ensure!(completed == n, "pipeline batch interrupted");
            Ok(outputs)
        });
        match result {
            Ok(outputs) => {
                self.sessions = next;
                Ok(outputs)
            }
            Err(error) => {
                self.abort();
                Err(error)
            }
        }
    }

    /// An out-of-band cancellation handle. Aborting an active request aborts
    /// the whole static pipeline job, because other stages may have mutated KV.
    pub fn abort_handle(&self) -> Result<AbortHandle> {
        self.ready()?;
        Ok(AbortHandle {
            streams: self
                .stages
                .iter()
                .map(|s| s.stream.try_clone())
                .collect::<std::io::Result<_>>()?,
            cancelled: Arc::clone(&self.cancelled),
        })
    }

    pub fn metrics(&self) -> Vec<Metrics> {
        self.stages.iter().map(|w| w.metrics.clone()).collect()
    }

    pub fn shutdown(mut self) -> Result<()> {
        self.stop()
    }

    /// Tell every worker to finish, then refuse further calls. Controller-
    /// managed workers unload their stages.
    pub fn stop(&mut self) -> Result<()> {
        let result = self.control(|| Command::Stop);
        self.failed = true;
        result
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        for stage in &self.stages {
            let _ = stage.stream.shutdown(Shutdown::Both);
        }
    }
}

/// Interrupt blocked network calls and invalidate all sessions in the job.
pub struct AbortHandle {
    streams: Vec<TcpStream>,
    cancelled: Arc<AtomicBool>,
}

impl AbortHandle {
    pub fn abort(&self) {
        self.cancelled.store(true, Ordering::Release);
        for stream in &self.streams {
            let _ = stream.shutdown(Shutdown::Both);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::matrix_bytes;

    #[test]
    fn matrix_reservation_follows_qmatmul_representation() {
        // Q8_0 (type 8): 32 values in a 34-byte block.
        assert_eq!(matrix_bytes(8, 32, 34, None).unwrap(), 34);
        assert_eq!(matrix_bytes(8, 32, 34, Some(4)).unwrap(), 128);
        assert_eq!(matrix_bytes(8, 32, 34, Some(2)).unwrap(), 64);
        // Floating types always load as f32.
        assert_eq!(matrix_bytes(1, 32, 64, None).unwrap(), 128);
        assert_eq!(matrix_bytes(0, 32, 128, Some(2)).unwrap(), 128);
    }
}
