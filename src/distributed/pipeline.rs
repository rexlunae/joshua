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
    path::Path,
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

    fn check_header(&self, header: &gguf_ext::GgufHeader) -> Result<()> {
        let meta = crate::gguf_meta::Meta::new(&header.metadata, "qwen3");
        let emb = header
            .tensors
            .get("token_embd.weight")
            .context("missing embedding")?;
        ensure!(
            self.layers == meta.u32("block_count")? as usize
                && self.hidden == meta.u32("embedding_length")? as usize
                && emb.dims == [self.vocab, self.hidden]
                && self.limits.context <= meta.u32("context_length")? as usize
                && meta.u32_or("nextn_predict_layers", 0) == 0
                && meta.u32_or("expert_count", 0) == 0,
            "plan/model dimensions or capabilities mismatch"
        );
        Ok(())
    }

    fn check_memory(&self, header: &gguf_ext::GgufHeader, rank: usize) -> Result<u64> {
        let stage = &self.stages[rank];
        let mut weights = 0u64;
        let mut load_scratch = 0u64;
        let mut intermediate = self.hidden as u64;
        for (name, tensor) in &header.tensors {
            let local = if let Some(rest) = name.strip_prefix("blk.") {
                let layer = rest.split('.').next().and_then(|s| s.parse::<usize>().ok());
                layer.is_some_and(|l| (stage.start..stage.end).contains(&l))
            } else {
                (stage.start == 0 && name == "token_embd.weight")
                    || (stage.end == self.layers
                        && (name.starts_with("output.")
                            || name.starts_with("output_norm.")
                            || (name == "token_embd.weight"
                                && !header.tensors.contains_key("output.weight"))))
            };
            if local {
                let elems = tensor
                    .dims
                    .iter()
                    .try_fold(1usize, |a, b| a.checked_mul(*b))
                    .context("tensor size overflow")?;
                let encoded = gguf_ext::type_size_bytes(tensor.dtype, elems)
                    .context("unsupported tensor storage format")?
                    as u64;
                // Norms/biases and floating-point linear weights expand to f32;
                // block-quantized matrices remain compact, including mmap loads.
                let expanded = tensor.dims.len() == 1 || matches!(tensor.dtype, 0 | 1 | 30);
                let bytes = if expanded {
                    encoded.max(checked_product(&[elems as u64, 4])?)
                } else {
                    encoded
                };
                let copies = if name == "token_embd.weight"
                    && stage.start == 0
                    && stage.end == self.layers
                    && !header.tensors.contains_key("output.weight")
                {
                    2
                } else {
                    1
                };
                weights = checked_sum(&[weights, checked_product(&[bytes, copies])?])?;
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
}

struct Packet {
    command: Command,
    values: Vec<f32>,
}

/// Framing: length, JSON control header length, JSON control, binary LE f32,
/// HMAC-SHA256. Direction/job/rank/monotonic counter bind every frame.
struct Wire {
    stream: TcpStream,
    key: Vec<u8>,
    job: Uuid,
    rank: usize,
    client: bool,
    tx: u64,
    rx: u64,
    values_limit: usize,
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
            job,
            rank,
            client,
            tx: 0,
            rx: 0,
            values_limit: plan.values_limit()?,
            timeout: timeout.expect("configured timeout"),
            metrics: Metrics::default(),
        })
    }

    fn mac(&self, client: bool, sequence: u64, payload: &[u8]) -> Result<Hmac<Sha256>> {
        let mut mac = Hmac::<Sha256>::new_from_slice(&self.key).map_err(anyhow::Error::msg)?;
        mac.update(b"joshua-qwen3-pipeline-v1");
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
        let header = serde_json::to_vec(&packet.command)?;
        ensure!(header.len() <= MAX_HEADER, "control header too large");
        let mut body = Vec::with_capacity(4 + header.len() + packet.values.len() * 4);
        body.extend_from_slice(&(header.len() as u32).to_le_bytes());
        body.extend_from_slice(&header);
        for value in &packet.values {
            body.extend_from_slice(&value.to_le_bytes());
        }
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
            (4..=4 + MAX_HEADER + self.values_limit * 4).contains(&length),
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
        let values = &body[4 + h..];
        ensure!(
            values.len().is_multiple_of(4) && values.len() / 4 <= self.values_limit,
            "invalid f32 payload length"
        );
        let values: Vec<f32> = values
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().expect("four bytes")))
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite input activation"
        );
        let command = serde_json::from_slice(&body[4..4 + h])?;
        self.metrics.codec_ns += elapsed_ns(t);
        self.metrics.received_bytes += (body.len() + 36) as u64;
        self.rx = self.rx.checked_add(1).context("wire sequence exhausted")?;
        Ok(Packet { command, values })
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
        plan.check_header(&header)?;
        let reserved_bytes = plan.check_memory(&header, rank)?;
        let file = File::open(path.as_ref())?;
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

    /// Serve exactly one coordinator connection. Any protocol/compute failure
    /// closes it and drops every session; do not reuse a failed worker job.
    pub fn serve(mut self, listener: TcpListener, job: Uuid, key: &[u8]) -> Result<()> {
        ensure!(
            key.len() >= 32 && job.get_version_num() == 4,
            "use a 32-byte key and fresh v4 job UUID"
        );
        let (stream, _) = listener.accept()?;
        let mut wire = Wire::new(stream, key, job, self.rank, false, &self.plan)?;
        let result = (|| {
            let hello = wire.receive()?;
            let Command::Hello { plan, rank } = hello.command else {
                bail!("expected handshake");
            };
            ensure!(
                plan == self.plan && rank == self.rank && hello.values.is_empty(),
                "stage plan mismatch"
            );
            wire.send(&Packet {
                command: Command::Reply { compute_ns: 0 },
                values: Vec::new(),
            })?;
            loop {
                let packet = wire.receive()?;
                if matches!(packet.command, Command::Stop) {
                    ensure!(packet.values.is_empty(), "unexpected stop payload");
                    wire.send(&Packet {
                        command: Command::Reply { compute_ns: 0 },
                        values: Vec::new(),
                    })?;
                    break;
                }
                let t = Instant::now();
                let values = self.execute(packet)?;
                wire.send(&Packet {
                    command: Command::Reply {
                        compute_ns: elapsed_ns(t),
                    },
                    values,
                })?;
            }
            Ok(())
        })();
        let _ = wire.stream.shutdown(Shutdown::Both);
        self.sessions.clear();
        result
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
            wire.rpc(&Packet {
                command: Command::Hello {
                    plan: plan.clone(),
                    rank,
                },
                values: Vec::new(),
            })?;
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
            if let Err(error) = stage.rpc(&Packet {
                command: command(),
                values: Vec::new(),
            }) {
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
                        let response = stage.rpc(&Packet {
                            command: Command::Forward {
                                id: job.id,
                                sequence: job.sequence,
                                offset: job.offset,
                                tokens: job.tokens.clone(),
                            },
                            values: std::mem::take(&mut job.values),
                        })?;
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
        self.control(|| Command::Stop)
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
