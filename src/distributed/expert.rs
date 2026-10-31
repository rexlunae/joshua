//! Complete-expert CPU layer prototype, not an Engine backend or tensor sharding.
//!
//! The coordinator supplies authoritative routes. Workers own complete packed
//! gate/up/down matrices and return **unweighted** rows. Reassembly uses the
//! existing MoE dispatch, preserving decode slot order and prefill expert order.
//! One synchronous forward is in flight; independent sessions share weights,
//! not KV. No timer batching, retry, migration, fallback, or discovery is implied.
//! Authentication is not encryption. Use a fresh job/key on a trusted network
//! or encrypted tunnel; all workers with the job key are trusted participants.

use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    net::{SocketAddr, TcpStream},
    path::Path,
    sync::Arc,
    time::Duration,
};

use anyhow::{ensure, Context, Result};
use candle_core::{
    quantized::{gguf_file, GgmlDType, QMatMul},
    Device, Module, Tensor,
};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::pipeline::ExpertChannel;
use crate::moe::{self, Expert, ExpertTensor};

/// Admission reservations, not OS RSS quotas. Tensor workspace includes a
/// conservative dense single-projection allowance for CPU kernel temporaries.
/// Frames, metadata and retained unweighted route outputs are also reserved.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Limits {
    pub rows: usize,
    pub routes: usize,
    pub sessions: usize,
    pub frame_bytes: usize,
    pub scratch_bytes: usize,
    pub resident_bytes_per_owner: usize,
    pub timeout_ms: u64,
}

impl Limits {
    fn validate(&self) -> Result<()> {
        ensure!((1..=4096).contains(&self.rows), "invalid row bound");
        ensure!((1..=16384).contains(&self.routes), "invalid route bound");
        ensure!((1..=128).contains(&self.sessions), "invalid session bound");
        ensure!(
            (1..=16 * 1024 * 1024).contains(&self.frame_bytes),
            "invalid byte bound"
        );
        ensure!(
            (1..=120_000).contains(&self.timeout_ms),
            "invalid timeout bound"
        );
        ensure!(self.resident_bytes_per_owner > 0, "empty resident budget");
        Ok(())
    }
}

/// Immutable placement for one layer and one exact GGUF revision. Constructed
/// from the file, never accepted as an unchecked remote manifest.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Placement {
    version: u32,
    architecture: String,
    model_sha256: [u8; 32],
    layer: usize,
    hidden: usize,
    width: usize,
    formats: [u32; 3],
    projection_bytes: [usize; 3],
    owners: Vec<usize>,
    limits: Limits,
}

fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |size, value| {
        size.checked_mul(*value).context("size overflow")
    })
}

fn digest(reader: &mut impl Read) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hash.finalize().into())
}

impl Placement {
    pub fn from_gguf(
        path: &Path,
        layer: usize,
        owners: Vec<usize>,
        limits: Limits,
    ) -> Result<Self> {
        limits.validate()?;
        ensure!(
            !owners.is_empty() && owners.len() <= 4096,
            "invalid expert count"
        );
        ensure!(owners.iter().all(|&owner| owner < 32), "invalid owner rank");
        let n_owners = owners.iter().copied().max().unwrap() + 1;
        ensure!(
            n_owners <= 32 && (0..n_owners).all(|owner| owners.contains(&owner)),
            "owners must be dense, nonempty ranks"
        );
        let mut file = File::open(path)?;
        let model_sha256 = digest(&mut file)?;
        file.rewind()?;
        let content = gguf_file::Content::read(&mut file)?;
        let architecture = content
            .metadata
            .get("general.architecture")
            .context("missing architecture")?
            .to_string()?
            .clone();
        ensure!(
            architecture == "qwen3moe",
            "only qwen3moe (no shared experts) is supported"
        );
        ensure!(
            !content
                .tensor_infos
                .keys()
                .any(|name| name.contains("shexp")),
            "shared experts must remain coordinator-local; unsupported model"
        );
        let projections = ["gate", "up", "down"].map(|name| {
            ExpertTensor::lookup(
                &content,
                "qwen3moe",
                &format!("blk.{layer}.ffn_{name}_exps.weight"),
                owners.len(),
            )
        });
        let [gate, up, down] = projections;
        let (gate, up, down) = (gate?, up?, down?);
        let (width, hidden) = gate.expert_shape();
        ensure!(
            (1..=65536).contains(&hidden) && (1..=65536).contains(&width),
            "unsupported expert dimensions"
        );
        ensure!(
            up.expert_shape() == (width, hidden) && down.expert_shape() == (hidden, width),
            "projection shape mismatch"
        );
        let mut projection_bytes = [0; 3];
        let mut formats = [0; 3];
        for (index, projection) in [&gate, &up, &down].iter().enumerate() {
            ensure!(
                matches!(
                    projection.dtype(),
                    GgmlDType::F32
                        | GgmlDType::Q2K
                        | GgmlDType::Q4K
                        | GgmlDType::Q6K
                        | GgmlDType::Q8_0
                ),
                "unsupported complete-expert format"
            );
            projection_bytes[index] = projection
                .bytes_per_expert()
                .context("unsliceable expert")?;
            formats[index] = projection.dtype() as u32;
            let end = projection
                .base()
                .checked_add(product(&[projection_bytes[index], owners.len()])?)
                .context("file range overflow")?;
            ensure!(
                end as u64 <= file.metadata()?.len(),
                "expert data past file end"
            );
        }
        let per_expert = projection_bytes.iter().try_fold(0usize, |sum, bytes| {
            sum.checked_add(*bytes).context("resident overflow")
        })?;
        for owner in 0..n_owners {
            ensure!(
                product(&[
                    owners.iter().filter(|&&rank| rank == owner).count(),
                    per_expert
                ])? <= limits.resident_bytes_per_owner,
                "resident expert budget exceeded"
            );
        }
        let result = Self {
            version: 1,
            architecture,
            model_sha256,
            layer,
            hidden,
            width,
            formats,
            projection_bytes,
            owners,
            limits,
        };
        ensure!(
            result.scratch_reservation()? <= result.limits.scratch_bytes,
            "scratch budget exceeded"
        );
        Ok(result)
    }

    fn scratch_reservation(&self) -> Result<usize> {
        let tensors = product(&[
            self.limits.routes,
            self.hidden
                .checked_add(self.width * 3)
                .context("workspace overflow")?,
            4,
            8,
        ])?;
        let projection = product(&[self.hidden, self.width, 4, 3])?;
        let frames = product(&[self.owner_count(), self.limits.frame_bytes + 64 * 1024, 8])?;
        tensors
            .checked_add(projection)
            .and_then(|size| size.checked_add(frames))
            .context("scratch overflow")
    }

    fn identity(&self) -> Result<[u8; 32]> {
        Ok(Sha256::digest(serde_json::to_vec(self)?).into())
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }
    pub fn expert_count(&self) -> usize {
        self.owners.len()
    }
    pub fn owner_count(&self) -> usize {
        self.owners.iter().copied().max().unwrap() + 1
    }
    pub fn limits(&self) -> &Limits {
        &self.limits
    }
}

/// Complete SwiGLU expert shared across requests. Quantized projections stay
/// compact; F32 projections use the existing dense execution path.
pub struct CompleteExpert {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl Expert for CompleteExpert {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.down
            .forward(&(candle_nn::ops::silu(&self.gate.forward(xs)?)? * self.up.forward(xs)?)?)
    }
}

/// Only this owner's experts are loaded. `None` means streamed compact load;
/// a supplied immutable mapping is identity-checked and borrowed where possible.
/// The caller must keep mapped model files immutable for the mapping's lifetime.
pub struct ExpertSet {
    placement: Placement,
    owner: usize,
    experts: BTreeMap<usize, CompleteExpert>,
}

impl ExpertSet {
    pub fn load(
        path: &Path,
        placement: Placement,
        owner: usize,
        mmap: Option<&Arc<Mmap>>,
    ) -> Result<Self> {
        ensure!(owner < placement.owner_count(), "unknown owner");
        let actual = Placement::from_gguf(
            path,
            placement.layer,
            placement.owners.clone(),
            placement.limits.clone(),
        )?;
        ensure!(actual == placement, "model identity mismatch");
        let mut file = File::open(path)?;
        if let Some(mmap) = mmap {
            let size = file.metadata()?.len() as usize;
            ensure!(mmap.len() >= size, "mapping truncated");
            ensure!(
                digest(&mut &mmap[..size])? == placement.model_sha256,
                "mapping identity mismatch"
            );
        }
        let content = gguf_file::Content::read(&mut file)?;
        let [gate, up, down] = ["gate", "up", "down"].map(|name| {
            ExpertTensor::lookup(
                &content,
                "qwen3moe",
                &format!("blk.{}.ffn_{name}_exps.weight", placement.layer),
                placement.expert_count(),
            )
        });
        let (gate, up, down) = (gate?, up?, down?);
        let mut experts = BTreeMap::new();
        for (expert, &rank) in placement.owners.iter().enumerate() {
            if rank != owner {
                continue;
            }
            let mut load = |projection: &ExpertTensor| -> Result<QMatMul> {
                let tensor = projection.read_expert(&mut file, expert, mmap)?;
                Ok(if tensor.dtype() == GgmlDType::F32 {
                    QMatMul::Tensor(tensor.dequantize(&Device::Cpu)?)
                } else {
                    QMatMul::QTensor(Arc::new(tensor))
                })
            };
            experts.insert(
                expert,
                CompleteExpert {
                    gate: load(&gate)?,
                    up: load(&up)?,
                    down: load(&down)?,
                },
            );
        }
        Ok(Self {
            placement,
            owner,
            experts,
        })
    }

    pub fn get(&self, expert: usize) -> Option<&CompleteExpert> {
        self.experts.get(&expert)
    }

    /// Serve one explicitly configured coordinator connection. Any invalid
    /// frame/identity/sequence/route or compute failure terminates the job.
    pub fn serve(self, stream: TcpStream, job: Uuid, key: &[u8]) -> Result<()> {
        let mut channel = channel(stream, key, job, self.owner, false, &self.placement)?;
        let mut sequence = 0u64;
        loop {
            let bytes = channel.receive()?;
            let (request, values): (Message, _) = decode(&bytes, &self.placement)?;
            ensure!(
                request.identity == self.placement.identity()? && request.forward == sequence,
                "placement or forward mismatch"
            );
            validate_rows(&request.rows, &self.placement, self.owner)?;
            ensure!(
                !request.decode
                    || request
                        .rows
                        .iter()
                        .map(|row| row.row)
                        .collect::<HashSet<_>>()
                        .len()
                        <= 1,
                "decode must contain one token position"
            );
            ensure!(
                request.rows.len() * self.placement.hidden == values.len(),
                "input shape mismatch"
            );
            let mut output = Vec::with_capacity(values.len());
            let mut start = 0;
            while start < request.rows.len() {
                let expert = request.rows[start].expert;
                let mut end = start + 1;
                while !request.decode
                    && end < request.rows.len()
                    && request.rows[end].expert == expert
                {
                    end += 1;
                }
                let input = Tensor::from_slice(
                    &values[start * self.placement.hidden..end * self.placement.hidden],
                    (end - start, self.placement.hidden),
                    &Device::Cpu,
                )?;
                output.extend(
                    self.experts
                        .get(&expert)
                        .context("unowned expert")?
                        .forward(&input)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                );
                start = end;
            }
            channel.send(encode(&request, &output, &self.placement)?)?;
            sequence = sequence.checked_add(1).context("forward exhausted")?;
        }
    }
}

fn channel(
    stream: TcpStream,
    key: &[u8],
    job: Uuid,
    owner: usize,
    client: bool,
    placement: &Placement,
) -> Result<ExpertChannel> {
    ExpertChannel::new(
        stream,
        key,
        job,
        owner,
        client,
        Duration::from_millis(placement.limits.timeout_ms),
        placement.limits.frame_bytes,
    )
}

/// Coordinator-assigned position. Tokens are positions, not vocabulary IDs.
/// Sessions carry no remote KV; the limit bounds distinct sessions per forward.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RowId {
    pub session: u64,
    pub token: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct RouteRow {
    route: usize,
    row: RowId,
    expert: usize,
}

#[derive(Serialize, Deserialize)]
struct Message {
    identity: [u8; 32],
    forward: u64,
    decode: bool,
    rows: Vec<RouteRow>,
}

fn validate_rows(rows: &[RouteRow], placement: &Placement, owner: usize) -> Result<()> {
    ensure!(
        rows.len() <= placement.limits.routes,
        "route bound exceeded"
    );
    let mut routes = HashSet::new();
    let mut positions = HashSet::new();
    let mut sessions = HashSet::new();
    let mut previous = None;
    for row in rows {
        ensure!(
            row.route < placement.limits.routes && routes.insert(row.route),
            "invalid or duplicate route ID"
        );
        ensure!(
            placement.owners.get(row.expert) == Some(&owner),
            "unowned expert"
        );
        let order = (row.expert, row.route);
        ensure!(
            previous.is_none_or(|prior| prior < order),
            "noncanonical bucket order"
        );
        previous = Some(order);
        positions.insert(row.row);
        sessions.insert(row.row.session);
    }
    ensure!(
        positions.len() <= placement.limits.rows && sessions.len() <= placement.limits.sessions,
        "row/session bound exceeded"
    );
    Ok(())
}

fn encode(message: &Message, values: &[f32], placement: &Placement) -> Result<Vec<u8>> {
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "nonfinite expert payload"
    );
    let header = serde_json::to_vec(message)?;
    let size = header
        .len()
        .checked_add(product(&[values.len(), 4])?)
        .and_then(|size| size.checked_add(4))
        .context("frame overflow")?;
    ensure!(size <= placement.limits.frame_bytes, "byte bound exceeded");
    let mut bytes = Vec::with_capacity(size);
    bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&header);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn decode(bytes: &[u8], placement: &Placement) -> Result<(Message, Vec<f32>)> {
    ensure!(
        (4..=placement.limits.frame_bytes).contains(&bytes.len()),
        "invalid byte count"
    );
    let header = u32::from_le_bytes(bytes[..4].try_into()?) as usize;
    ensure!(header <= bytes.len() - 4, "invalid metadata length");
    let message: Message = serde_json::from_slice(&bytes[4..4 + header])?;
    ensure!(
        message.rows.len() <= placement.limits.routes,
        "route bound exceeded"
    );
    let payload = &bytes[4 + header..];
    ensure!(
        payload.len() == product(&[message.rows.len(), placement.hidden, 4])?,
        "payload shape mismatch"
    );
    let values: Vec<f32> = payload
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().expect("four bytes")))
        .collect();
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "nonfinite expert payload"
    );
    Ok((message, values))
}

struct ReturnedExpert(Tensor);
impl Expert for ReturnedExpert {
    fn forward(&self, _xs: &Tensor) -> candle_core::Result<Tensor> {
        Ok(self.0.clone())
    }
}

/// Static placement and bounded synchronous backpressure: no second forward
/// enters until this one completes. A failure permanently poisons the job and
/// closes every worker connection; partial output is never returned.
pub struct Coordinator {
    placement: Placement,
    channels: Vec<ExpertChannel>,
    sequence: u64,
    failed: bool,
}

impl Coordinator {
    pub fn connect(
        placement: Placement,
        addresses: &[SocketAddr],
        job: Uuid,
        key: &[u8],
    ) -> Result<Self> {
        ensure!(
            addresses.len() == placement.owner_count(),
            "owner address count mismatch"
        );
        let channels = addresses
            .iter()
            .enumerate()
            .map(|(owner, address)| {
                channel(
                    TcpStream::connect_timeout(
                        address,
                        Duration::from_millis(placement.limits.timeout_ms),
                    )?,
                    key,
                    job,
                    owner,
                    true,
                    &placement,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            placement,
            channels,
            sequence: 0,
            failed: false,
        })
    }

    /// CPU f32 activations [rows, hidden], authoritative IDs/weights [rows, k].
    /// Decode is exactly one row, matching local dispatch's slot accumulation;
    /// larger batches use local prefill's expert-major accumulation order.
    pub fn forward(
        &mut self,
        rows: &[RowId],
        input: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        ensure!(!self.failed, "expert job failed; create a fresh job");
        let result = self.forward_inner(rows, input, ids, weights);
        if result.is_err() {
            self.failed = true;
            for channel in &self.channels {
                channel.shutdown();
            }
        }
        result
    }

    fn forward_inner(
        &mut self,
        rows: &[RowId],
        input: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let placement = &self.placement;
        ensure!(
            !rows.is_empty() && rows.len() <= placement.limits.rows,
            "row bound exceeded"
        );
        ensure!(
            rows.iter().copied().collect::<HashSet<_>>().len() == rows.len(),
            "duplicate row ID"
        );
        ensure!(
            rows.iter()
                .map(|row| row.session)
                .collect::<HashSet<_>>()
                .len()
                <= placement.limits.sessions,
            "session bound exceeded"
        );
        ensure!(
            input.device().is_cpu() && ids.device().is_cpu() && weights.device().is_cpu(),
            "CPU-only expert prototype"
        );
        ensure!(
            input.dims() == [rows.len(), placement.hidden],
            "activation shape mismatch"
        );
        let (n_rows, selected) = ids.dims2()?;
        let routes = product(&[n_rows, selected])?;
        ensure!(
            n_rows == rows.len()
                && selected > 0
                && routes <= placement.limits.routes
                && weights.dims() == ids.dims(),
            "routing shape/bound mismatch"
        );
        let route_ids = ids.flatten_all()?.to_vec1::<u32>()?;
        let route_weights = weights.flatten_all()?.to_vec1::<f32>()?;
        ensure!(
            route_weights.iter().all(|weight| weight.is_finite()),
            "nonfinite route weight"
        );
        let values = input.flatten_all()?.to_vec1::<f32>()?;
        let identity = placement.identity()?;
        let mut requests: Vec<Message> = (0..placement.owner_count())
            .map(|_| Message {
                identity,
                forward: self.sequence,
                decode: rows.len() == 1,
                rows: Vec::new(),
            })
            .collect();
        for (route, &expert) in route_ids.iter().enumerate() {
            let expert = expert as usize;
            let owner = *placement
                .owners
                .get(expert)
                .context("invalid routed expert")?;
            requests[owner].rows.push(RouteRow {
                route,
                row: rows[route / selected],
                expert,
            });
        }
        let mut packets = Vec::with_capacity(requests.len());
        for request in &mut requests {
            request.rows.sort_by_key(|row| (row.expert, row.route));
            let mut bucket = Vec::with_capacity(request.rows.len() * placement.hidden);
            for row in &request.rows {
                let start = row.route / selected * placement.hidden;
                bucket.extend_from_slice(&values[start..start + placement.hidden]);
            }
            packets.push(encode(request, &bucket, placement)?);
        }
        let responses = std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .channels
                .iter_mut()
                .zip(packets)
                .map(|(channel, packet)| {
                    scope.spawn(move || {
                        channel.send(packet)?;
                        decode(&channel.receive()?, placement)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("expert RPC panicked"))?
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let mut outputs: Vec<Vec<f32>> =
            (0..placement.expert_count()).map(|_| Vec::new()).collect();
        for ((response, values), request) in responses.into_iter().zip(&requests) {
            ensure!(
                response.identity == identity
                    && response.forward == self.sequence
                    && response.decode == request.decode
                    && response.rows == request.rows,
                "response route/identity mismatch"
            );
            for (row, values) in response
                .rows
                .iter()
                .zip(values.chunks_exact(placement.hidden))
            {
                outputs[row.expert].extend_from_slice(values);
            }
        }
        let experts = outputs
            .into_iter()
            .map(|values| {
                let count = values.len() / placement.hidden;
                Ok(ReturnedExpert(Tensor::from_vec(
                    values,
                    (count, placement.hidden),
                    &Device::Cpu,
                )?))
            })
            .collect::<Result<Vec<_>>>()?;
        let (output, _) = if rows.len() == 1 {
            let mut offsets = vec![0usize; experts.len()];
            let mut slots = Vec::with_capacity(selected);
            for &expert in &route_ids {
                slots.push(
                    experts[expert as usize]
                        .0
                        .narrow(0, offsets[expert as usize], 1)?,
                );
                offsets[expert as usize] += 1;
            }
            let output = Tensor::stack(&slots, 0)?
                .broadcast_mul(&weights.reshape((selected, 1, 1))?)?
                .sum(0)?;
            (output, route_ids)
        } else {
            moe::dispatch_prefill(
                "remote-qwen3moe",
                &experts,
                &Device::Cpu,
                input,
                ids,
                weights,
                rows.len(),
                selected,
            )?
        };
        self.sequence = self.sequence.checked_add(1).context("forward exhausted")?;
        Ok(output)
    }
}

impl Drop for Coordinator {
    fn drop(&mut self) {
        for channel in &self.channels {
            channel.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{net::TcpListener, sync::mpsc, thread};

    const KEY: &[u8] = &[17; 32];

    fn placement(owners: Vec<usize>) -> Placement {
        Placement {
            version: 1,
            architecture: "qwen3moe".into(),
            model_sha256: [7; 32],
            layer: 0,
            hidden: 2,
            width: 2,
            formats: [0; 3],
            projection_bytes: [16; 3],
            owners,
            limits: Limits {
                rows: 2,
                routes: 4,
                sessions: 2,
                frame_bytes: 4096,
                scratch_bytes: 1024 * 1024,
                resident_bytes_per_owner: 1024,
                timeout_ms: 2000,
            },
        }
    }

    fn message(placement: &Placement) -> Message {
        Message {
            identity: placement.identity().unwrap(),
            forward: 0,
            decode: true,
            rows: vec![RouteRow {
                route: 0,
                row: RowId {
                    session: 1,
                    token: 0,
                },
                expert: 0,
            }],
        }
    }

    #[test]
    fn codec_and_route_bounds_fail_closed() {
        let placement = placement(vec![0, 1]);
        let mut request = message(&placement);
        let bytes = encode(&request, &[1., 2.], &placement).unwrap();
        assert_eq!(decode(&bytes, &placement).unwrap().1, [1., 2.]);
        assert!(decode(&bytes[..bytes.len() - 1], &placement).is_err());
        assert!(decode(&[255; 4], &placement).is_err());
        assert!(decode(&vec![0; 4097], &placement).is_err());
        assert!(encode(&request, &[f32::NAN, 0.], &placement).is_err());
        let mut nonfinite = bytes;
        let end = nonfinite.len();
        nonfinite[end - 4..].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(decode(&nonfinite, &placement).is_err());
        assert!(validate_rows(&request.rows, &placement, 1).is_err());
        request.rows.push(request.rows[0].clone());
        assert!(validate_rows(&request.rows, &placement, 0).is_err());
        request.rows[1].route = 1;
        validate_rows(&request.rows, &placement, 0).unwrap();
        request.rows[0].route = 2;
        assert!(validate_rows(&request.rows, &placement, 0).is_err());
    }

    #[test]
    fn later_owner_can_reply_first_without_changing_decode_accumulation() {
        let placement = placement(vec![0, 1]);
        let job = Uuid::new_v4();
        let (finished, wait) = mpsc::channel();
        let mut wait = Some(wait);
        let mut addresses = Vec::new();
        let mut handles = Vec::new();
        for owner in 0..2 {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            addresses.push(listener.local_addr().unwrap());
            let placement = placement.clone();
            let wait = if owner == 0 { wait.take() } else { None };
            let finished = finished.clone();
            handles.push(thread::spawn(move || -> Result<()> {
                let mut channel =
                    channel(listener.accept()?.0, KEY, job, owner, false, &placement)?;
                let (request, _) = decode(&channel.receive()?, &placement)?;
                if let Some(wait) = wait {
                    wait.recv_timeout(Duration::from_secs(1))?;
                }
                let value = if owner == 0 { 1. } else { 1e10 };
                let values = vec![value; request.rows.len() * placement.hidden];
                channel.send(encode(&request, &values, &placement)?)?;
                if owner == 1 {
                    finished.send(())?;
                }
                Ok(())
            }));
        }
        let mut coordinator = Coordinator::connect(placement, &addresses, job, KEY).unwrap();
        let input = Tensor::zeros((1, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let ids = Tensor::new(&[[1u32, 0, 1]], &Device::Cpu).unwrap();
        let weights = Tensor::new(&[[1f32, 1., -1.]], &Device::Cpu).unwrap();
        let output = coordinator
            .forward(
                &[RowId {
                    session: 1,
                    token: 0,
                }],
                &input,
                &ids,
                &weights,
            )
            .unwrap();
        let local = vec![
            ReturnedExpert(Tensor::new(&[[1f32, 1.]], &Device::Cpu).unwrap()),
            ReturnedExpert(Tensor::new(&[[1e10f32, 1e10]], &Device::Cpu).unwrap()),
        ];
        let expected =
            moe::dispatch_decode("test", &local, &Device::Cpu, &input, &ids, &weights, 3)
                .unwrap()
                .0;
        assert_eq!(
            output.to_vec2::<f32>().unwrap(),
            expected.to_vec2::<f32>().unwrap()
        );
        for handle in handles {
            handle.join().unwrap().unwrap();
        }
    }

    #[test]
    fn invalid_reply_or_authentication_permanently_fails_job() {
        for fault in [
            "sequence",
            "identity",
            "route",
            "mode",
            "authentication",
            "disconnect",
        ] {
            let placement = placement(vec![0]);
            let job = Uuid::new_v4();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let remote = placement.clone();
            let handle = thread::spawn(move || -> Result<()> {
                let key: &[u8] = if fault == "authentication" {
                    &[18; 32]
                } else {
                    KEY
                };
                let mut channel = channel(listener.accept()?.0, key, job, 0, false, &remote)?;
                if fault == "disconnect" {
                    return Ok(());
                }
                let (mut request, values) = decode(&channel.receive()?, &remote)?;
                match fault {
                    "sequence" => request.forward += 1,
                    "identity" => request.identity[0] ^= 1,
                    "route" => request.rows[0].row.token += 1,
                    "mode" => request.decode = false,
                    _ => {}
                }
                channel.send(encode(&request, &values, &remote)?)
            });
            let mut coordinator = Coordinator::connect(placement, &[address], job, KEY).unwrap();
            let input = Tensor::zeros((1, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
            let ids = Tensor::new(&[[0u32]], &Device::Cpu).unwrap();
            let weights = Tensor::new(&[[1f32]], &Device::Cpu).unwrap();
            let rows = [RowId {
                session: 1,
                token: 0,
            }];
            assert!(
                coordinator.forward(&rows, &input, &ids, &weights).is_err(),
                "{fault}"
            );
            assert!(coordinator
                .forward(&rows, &input, &ids, &weights)
                .unwrap_err()
                .to_string()
                .contains("job failed"));
            let result = handle.join().unwrap();
            assert_eq!(result.is_err(), fault == "authentication");
        }
    }

    #[test]
    fn worker_rejects_invalid_requests_before_expert_execution() {
        for fault in [
            "sequence",
            "identity",
            "owner",
            "duplicate",
            "mode",
            "rows",
            "sessions",
        ] {
            let mut placement = placement(vec![0]);
            if fault == "rows" {
                placement.limits.rows = 1;
            }
            if fault == "sessions" {
                placement.limits.sessions = 1;
            }
            let job = Uuid::new_v4();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let remote = ExpertSet {
                placement: placement.clone(),
                owner: 0,
                experts: BTreeMap::new(),
            };
            let handle =
                thread::spawn(move || remote.serve(listener.accept().unwrap().0, job, KEY));
            let mut channel = channel(
                TcpStream::connect(address).unwrap(),
                KEY,
                job,
                0,
                true,
                &placement,
            )
            .unwrap();
            let mut request = message(&placement);
            match fault {
                "sequence" => request.forward += 1,
                "identity" => request.identity[0] ^= 1,
                "owner" => request.rows[0].expert = 1,
                "duplicate" => request.rows.push(request.rows[0].clone()),
                _ => {
                    request.rows.push(RouteRow {
                        route: 1,
                        row: RowId {
                            session: 2,
                            token: 1,
                        },
                        expert: 0,
                    });
                    request.decode = fault == "mode";
                }
            }
            let values = vec![0f32; request.rows.len() * placement.hidden];
            channel
                .send(encode(&request, &values, &placement).unwrap())
                .unwrap();
            assert!(channel.receive().is_err(), "{fault}");
            let error = handle.join().unwrap().unwrap_err().to_string();
            assert!(
                !error.contains("unowned expert") || fault == "owner",
                "{fault}: {error}"
            );
        }
    }
}
