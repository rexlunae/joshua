//! Bounded, synchronous all-reduce for a manually configured, trusted static cluster.
//!
//! Every rank must use the same fresh random session UUID, membership size, PSK,
//! tensor length and collective order. Discovery is not a source of membership or
//! trust. The PSK authenticates packets but does not encrypt them; any member with
//! the shared key can impersonate another member.
//!
//! Inputs are retransmitted until every rank acknowledges receiving every input.
//! Success therefore never silently omits a rank. A timeout poisons the group and
//! leaves the caller's tensor unchanged: recovery requires a new session on every
//! rank. UDP cannot guarantee globally atomic success during a partition (one rank
//! can succeed while another times out). Brief completion-ack linger and replay
//! during the next call mitigate lost final acknowledgements, not that limitation.
//! TCP fallback, FEC, dynamic membership and performance guarantees are deferred.

use anyhow::{bail, ensure, Context, Result};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use socket2::{Domain, Protocol, Socket, Type};
use std::io::ErrorKind;
use std::net::{Ipv4Addr, SocketAddrV4, UdpSocket};
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

pub const MAX_PARTICIPANTS: usize = 64;
pub const MAX_TENSOR_ELEMENTS: usize = 262_144;
/// Maximum aggregate input storage, in f32 elements (16 MiB).
pub const MAX_TOTAL_ELEMENTS: usize = 4_194_304;

const MAGIC: &[u8; 4] = b"JAR1";
const VERSION: u8 = 1;
const HEADER: usize = 48;
const TAG: usize = 32;
const CHUNK_ELEMENTS: usize = 320;
const MAX_PACKET: usize = HEADER + CHUNK_ELEMENTS * 4 + TAG;
const DATA: u8 = 1;
const READY: u8 = 2;
type Auth = Hmac<Sha256>;

/// IPv4 multicast transport settings. All ranks must join the same group/port.
#[derive(Clone, Debug)]
pub struct MulticastConfig {
    pub group: Ipv4Addr,
    pub port: u16,
    /// Local interface address, or unspecified to use the OS-selected interface.
    pub interface: Ipv4Addr,
    pub timeout: Duration,
    pub retransmit_interval: Duration,
}

impl Default for MulticastConfig {
    fn default() -> Self {
        Self {
            group: Ipv4Addr::new(239, 255, 88, 1),
            port: 48888,
            interface: Ipv4Addr::UNSPECIFIED,
            timeout: Duration::from_secs(10),
            retransmit_interval: Duration::from_millis(25),
        }
    }
}

/// A single explicit rank in a fixed-membership session; calls require exclusive access.
pub struct AllReduceGroup {
    socket: UdpSocket,
    destination: SocketAddrV4,
    config: MulticastConfig,
    rank: usize,
    peers: usize,
    session: Uuid,
    auth: Auth,
    step: u64,
    poisoned: bool,
    previous_len: Option<usize>,
    #[cfg(test)]
    faults: Faults,
    #[cfg(test)]
    test_destinations: Option<Vec<SocketAddrV4>>,
}

#[derive(Clone, Copy)]
struct Envelope {
    session: Uuid,
    step: u64,
    peers: usize,
    total: usize,
}

struct Packet<'a> {
    kind: u8,
    rank: usize,
    chunk: usize,
    payload: &'a [u8],
}

impl AllReduceGroup {
    /// Create a socket before starting any rank's first collective.
    ///
    /// Generate `session` once with `Uuid::new_v4()` and distribute it and a
    /// high-entropy PSK out of band. Never reuse a session after restarting a rank.
    pub fn new(
        rank: usize,
        world_size: usize,
        session: Uuid,
        psk: &[u8],
        config: MulticastConfig,
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_PARTICIPANTS).contains(&world_size) && rank < world_size,
            "invalid static rank or membership size"
        );
        ensure!(
            session.get_version_num() == 4,
            "session must be a random v4 UUID"
        );
        ensure!(
            psk.len() >= 32,
            "all-reduce requires a PSK of at least 32 bytes"
        );
        ensure!(config.group.is_multicast(), "group must be IPv4 multicast");
        ensure!(config.port != 0, "multicast port must be nonzero");
        ensure!(
            (Duration::from_millis(50)..=Duration::from_secs(120)).contains(&config.timeout),
            "timeout must be between 50 milliseconds and 120 seconds"
        );
        ensure!(
            config.retransmit_interval >= Duration::from_millis(5)
                && config.retransmit_interval <= config.timeout / 4,
            "retransmit interval must be at least 5 milliseconds and at most timeout / 4"
        );
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
        socket.set_reuse_address(true)?;
        socket.bind(&SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, config.port).into())?;
        socket.set_multicast_if_v4(&config.interface)?;
        socket.set_multicast_ttl_v4(1)?;
        socket.set_multicast_loop_v4(true)?;
        socket.join_multicast_v4(&config.group, &config.interface)?;
        socket.set_nonblocking(true)?;
        let destination = SocketAddrV4::new(config.group, config.port);
        Ok(Self {
            socket: socket.into(),
            destination,
            config,
            rank,
            peers: world_size,
            session,
            auth: Auth::new_from_slice(psk).expect("HMAC accepts arbitrary key lengths"),
            step: 0,
            poisoned: false,
            previous_len: None,
            #[cfg(test)]
            faults: Faults::default(),
            #[cfg(test)]
            test_destinations: None,
        })
    }

    /// Sum equal-length f32 vectors in ascending rank order.
    ///
    /// All ranks must call this in the same order, including for empty vectors.
    /// On any transport failure/timeout the input stays unchanged and this group
    /// cannot be reused. Local size-validation errors do not consume a step.
    pub fn all_reduce_sum(&mut self, values: &mut [f32]) -> Result<()> {
        ensure!(
            !self.poisoned,
            "all-reduce group is poisoned; create a new session"
        );
        ensure!(
            values.len() <= MAX_TENSOR_ELEMENTS
                && values
                    .len()
                    .checked_mul(self.peers)
                    .is_some_and(|n| n <= MAX_TOTAL_ELEMENTS),
            "all-reduce tensor exceeds bounded storage limits"
        );
        let next_step = self
            .step
            .checked_add(1)
            .context("collective step exhausted")?;
        self.poisoned = true;
        let inputs = self.exchange(values)?;
        let total = values.len();
        for (index, value) in values.iter_mut().enumerate() {
            *value = (0..self.peers).fold(0.0, |sum, rank| sum + inputs[rank * total + index]);
        }
        self.previous_len = Some(values.len());
        self.step = next_step;
        self.poisoned = false;
        Ok(())
    }

    fn exchange(&mut self, values: &[f32]) -> Result<Vec<f32>> {
        let deadline = Instant::now() + self.config.timeout;
        let envelope = Envelope {
            session: self.session,
            step: self.step,
            peers: self.peers,
            total: values.len(),
        };
        let chunks = values.len().div_ceil(CHUNK_ELEMENTS);
        let mut inputs = vec![0.0; self.peers * values.len()];
        inputs[self.rank * values.len()..(self.rank + 1) * values.len()].copy_from_slice(values);
        let mut seen = vec![false; self.peers * chunks];
        let mut remaining = (self.peers - 1) * chunks;
        let mut ready = vec![false; self.peers];
        ready[self.rank] = remaining == 0;
        let mut linger_until = None;
        let mut next_round = Instant::now();
        let mut next_ready = next_round;
        let mut next_previous_ack = next_round;
        let mut send_chunk = chunks;
        // One extra byte rejects oversized/truncated UDP datagrams without allocation.
        let mut receive = [0u8; MAX_PACKET + 1];
        let mut transmit = [0u8; MAX_PACKET];
        loop {
            let now = Instant::now();
            if linger_until.is_some_and(|end| now >= end) {
                return Ok(inputs);
            }
            if now >= deadline {
                bail!(
                    "all-reduce step {} timed out waiting for all {} ranks",
                    self.step,
                    self.peers
                );
            }
            // Bounded batches preserve receive progress and the deadline even under traffic.
            for _ in 0..64 {
                match self.socket.recv_from(&mut receive) {
                    Ok((len, _)) => {
                        if let Some(packet) = decode(&receive[..len], envelope, &self.auth) {
                            if packet.kind == READY {
                                ready[packet.rank] = true;
                            } else if packet.rank != self.rank {
                                accept_data(
                                    packet,
                                    values.len(),
                                    chunks,
                                    &mut inputs,
                                    &mut seen,
                                    &mut remaining,
                                );
                            }
                        } else if let Some(total) = self.previous_len {
                            let previous = Envelope {
                                step: self.step - 1,
                                total,
                                ..envelope
                            };
                            if Instant::now() >= next_previous_ack
                                && decode(&receive[..len], previous, &self.auth).is_some()
                            {
                                self.send(previous, READY, 0, &[], &mut transmit)?;
                                next_previous_ack =
                                    Instant::now() + self.config.retransmit_interval;
                            }
                        }
                    }
                    Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                    Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error).context("receive all-reduce multicast"),
                }
            }
            ready[self.rank] = remaining == 0;
            let now = Instant::now();
            if ready.iter().all(|&r| r) && linger_until.is_none() {
                linger_until = Some((now + self.config.retransmit_interval * 2).min(deadline));
                next_ready = now;
            }
            if ready[self.rank] && now >= next_ready {
                self.send(envelope, READY, 0, &[], &mut transmit)?;
                next_ready = now + self.config.retransmit_interval;
            }
            if linger_until.is_none() {
                if send_chunk == chunks && now >= next_round {
                    send_chunk = 0;
                }
                for _ in 0..16 {
                    if send_chunk == chunks || Instant::now() >= deadline {
                        break;
                    }
                    let start = send_chunk * CHUNK_ELEMENTS;
                    let end = (start + CHUNK_ELEMENTS).min(values.len());
                    self.send(
                        envelope,
                        DATA,
                        send_chunk,
                        &values[start..end],
                        &mut transmit,
                    )?;
                    send_chunk += 1;
                    if send_chunk == chunks {
                        next_round = Instant::now() + self.config.retransmit_interval;
                    }
                }
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn send(
        &mut self,
        envelope: Envelope,
        kind: u8,
        chunk: usize,
        values: &[f32],
        buffer: &mut [u8; MAX_PACKET],
    ) -> Result<()> {
        #[cfg(test)]
        if self.faults.drop_packet(kind) {
            return Ok(());
        }
        let len = encode(envelope, kind, self.rank, chunk, values, &self.auth, buffer);
        #[cfg(test)]
        if let Some(destinations) = &self.test_destinations {
            for destination in destinations {
                self.send_datagram(&buffer[..len], *destination)?;
            }
            return Ok(());
        }
        self.send_datagram(&buffer[..len], self.destination)
    }

    fn send_datagram(&self, bytes: &[u8], destination: SocketAddrV4) -> Result<()> {
        match self.socket.send_to(bytes, destination) {
            Ok(n) if n == bytes.len() => Ok(()),
            // The periodic retry covers transient send-buffer pressure as well as loss.
            Err(error)
                if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) =>
            {
                Ok(())
            }
            Err(error) => Err(error).context("send all-reduce multicast"),
            Ok(_) => bail!("incomplete all-reduce datagram send"),
        }
    }
}

fn accept_data(
    packet: Packet<'_>,
    total: usize,
    chunks: usize,
    inputs: &mut [f32],
    seen: &mut [bool],
    remaining: &mut usize,
) {
    let slot = packet.rank * chunks + packet.chunk;
    if seen[slot] {
        return;
    }
    seen[slot] = true;
    *remaining -= 1;
    let start = packet.rank * total + packet.chunk * CHUNK_ELEMENTS;
    for (out, bytes) in inputs[start..]
        .iter_mut()
        .zip(packet.payload.chunks_exact(4))
    {
        *out = f32::from_bits(u32::from_be_bytes(bytes.try_into().unwrap()));
    }
}

fn encode(
    envelope: Envelope,
    kind: u8,
    rank: usize,
    chunk: usize,
    values: &[f32],
    auth: &Auth,
    buffer: &mut [u8; MAX_PACKET],
) -> usize {
    buffer[..HEADER].fill(0);
    buffer[..4].copy_from_slice(MAGIC);
    buffer[4] = VERSION;
    buffer[5] = kind;
    buffer[8..24].copy_from_slice(envelope.session.as_bytes());
    buffer[24..32].copy_from_slice(&envelope.step.to_be_bytes());
    buffer[32..34].copy_from_slice(&(rank as u16).to_be_bytes());
    buffer[34..36].copy_from_slice(&(envelope.peers as u16).to_be_bytes());
    buffer[36..40].copy_from_slice(&(envelope.total as u32).to_be_bytes());
    buffer[40..44].copy_from_slice(&(chunk as u32).to_be_bytes());
    buffer[44..46].copy_from_slice(&((values.len() * 4) as u16).to_be_bytes());
    for (value, out) in values.iter().zip(buffer[HEADER..].chunks_exact_mut(4)) {
        out.copy_from_slice(&value.to_bits().to_be_bytes());
    }
    let end = HEADER + values.len() * 4;
    let mut mac = auth.clone();
    mac.update(&buffer[..end]);
    buffer[end..end + TAG].copy_from_slice(&mac.finalize().into_bytes());
    end + TAG
}

/// Parse and authenticate without allocating, and only in the expected bounded context.
fn decode<'a>(bytes: &'a [u8], expected: Envelope, auth: &Auth) -> Option<Packet<'a>> {
    if !(HEADER + TAG..=MAX_PACKET).contains(&bytes.len())
        || &bytes[..4] != MAGIC
        || bytes[4] != VERSION
        || bytes[6..8] != [0, 0]
        || bytes[46..48] != [0, 0]
        || bytes[8..24] != *expected.session.as_bytes()
        || u64::from_be_bytes(bytes[24..32].try_into().ok()?) != expected.step
    {
        return None;
    }
    let kind = bytes[5];
    let rank = u16::from_be_bytes(bytes[32..34].try_into().ok()?) as usize;
    let peers = u16::from_be_bytes(bytes[34..36].try_into().ok()?) as usize;
    let total = u32::from_be_bytes(bytes[36..40].try_into().ok()?) as usize;
    let chunk = u32::from_be_bytes(bytes[40..44].try_into().ok()?) as usize;
    let len = u16::from_be_bytes(bytes[44..46].try_into().ok()?) as usize;
    if peers != expected.peers
        || !(1..=MAX_PARTICIPANTS).contains(&peers)
        || rank >= peers
        || total != expected.total
        || total > MAX_TENSOR_ELEMENTS
        || total.checked_mul(peers)? > MAX_TOTAL_ELEMENTS
        || bytes.len() != HEADER + len + TAG
    {
        return None;
    }
    match kind {
        DATA => {
            if chunk >= total.div_ceil(CHUNK_ELEMENTS)
                || len != (total - chunk * CHUNK_ELEMENTS).min(CHUNK_ELEMENTS) * 4
            {
                return None;
            }
        }
        READY if chunk == 0 && len == 0 => {}
        _ => return None,
    }
    let mut mac = auth.clone();
    mac.update(&bytes[..HEADER + len]);
    mac.verify_slice(&bytes[HEADER + len..]).ok()?;
    Some(Packet {
        kind,
        rank,
        chunk,
        payload: &bytes[HEADER..HEADER + len],
    })
}

#[cfg(test)]
#[derive(Default)]
struct Faults {
    data: usize,
    ready: usize,
}

#[cfg(test)]
impl Faults {
    fn drop_packet(&mut self, kind: u8) -> bool {
        let count = if kind == DATA {
            &mut self.data
        } else {
            &mut self.ready
        };
        if *count == 0 {
            false
        } else {
            *count -= 1;
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &[u8] = b"test-only-static-cluster-key-32bytes";

    fn config() -> MulticastConfig {
        let reservation = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        MulticastConfig {
            port: reservation.local_addr().unwrap().port(),
            interface: Ipv4Addr::LOCALHOST,
            timeout: Duration::from_secs(3),
            retransmit_interval: Duration::from_millis(10),
            ..MulticastConfig::default()
        }
    }

    fn envelope(total: usize) -> Envelope {
        Envelope {
            session: Uuid::new_v4(),
            step: 7,
            peers: 2,
            total,
        }
    }

    // The CI firewall denies multicast sendto (EPERM), but permits real loopback
    // UDP. Fan out the same authenticated datagrams to independently bound sockets.
    // The ignored multicast test below exercises the production socket path on a LAN.
    fn loopback_fanout(groups: &mut [&mut AllReduceGroup]) {
        let mut destinations = Vec::new();
        for group in groups.iter_mut() {
            group.socket = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
            group.socket.set_nonblocking(true).unwrap();
            let std::net::SocketAddr::V4(address) = group.socket.local_addr().unwrap() else {
                unreachable!()
            };
            destinations.push(address);
        }
        for group in groups {
            group.test_destinations = Some(destinations.clone());
        }
    }

    #[test]
    fn packet_roundtrip_and_rejection() {
        let auth = Auth::new_from_slice(KEY).unwrap();
        let expected = envelope(3);
        let mut buffer = [0; MAX_PACKET];
        let n = encode(expected, DATA, 1, 0, &[1.25, -2.0, 9.0], &auth, &mut buffer);
        let packet = decode(&buffer[..n], expected, &auth).unwrap();
        assert_eq!((packet.kind, packet.rank, packet.chunk), (DATA, 1, 0));
        assert_eq!(&packet.payload[..4], &1.25f32.to_bits().to_be_bytes());
        assert!(decode(
            &buffer[..n],
            Envelope {
                session: Uuid::new_v4(),
                ..expected
            },
            &auth
        )
        .is_none());
        assert!(decode(
            &buffer[..n],
            Envelope {
                step: 8,
                ..expected
            },
            &auth
        )
        .is_none());
        assert!(decode(
            &buffer[..n],
            Envelope {
                total: 4,
                ..expected
            },
            &auth
        )
        .is_none());
        let other_key = Auth::new_from_slice(b"another-test-only-key-of-32-bytes!").unwrap();
        assert!(decode(&buffer[..n], expected, &other_key).is_none());
        // Every header, payload and tag byte is authenticated (including reserved bytes).
        for index in 0..n {
            buffer[index] ^= 1;
            assert!(
                decode(&buffer[..n], expected, &auth).is_none(),
                "byte {index}"
            );
            buffer[index] ^= 1;
        }
        for len in 0..n {
            assert!(decode(&buffer[..len], expected, &auth).is_none());
        }
        assert!(decode(&[0; MAX_PACKET + 1], expected, &auth).is_none());
        for (kind, rank, chunk, data) in [
            (DATA, 2, 0, vec![1.0; 3]),
            (DATA, 1, 1, vec![1.0; 3]),
            (DATA, 1, 0, vec![1.0; 2]),
            (READY, 1, 1, vec![]),
            (READY, 1, 0, vec![1.0]),
            (3, 1, 0, vec![]),
        ] {
            let len = encode(expected, kind, rank, chunk, &data, &auth, &mut buffer);
            assert!(decode(&buffer[..len], expected, &auth).is_none());
        }
        for invalid in [
            Envelope {
                peers: 0,
                ..expected
            },
            Envelope {
                peers: MAX_PARTICIPANTS + 1,
                ..expected
            },
            Envelope {
                total: MAX_TENSOR_ELEMENTS + 1,
                ..expected
            },
            Envelope {
                peers: MAX_PARTICIPANTS,
                total: MAX_TENSOR_ELEMENTS,
                ..expected
            },
        ] {
            let len = encode(invalid, READY, 0, 0, &[], &auth, &mut buffer);
            assert!(decode(&buffer[..len], invalid, &auth).is_none());
        }
        let n = encode(expected, READY, 1, 0, &[], &auth, &mut buffer);
        assert_eq!(decode(&buffer[..n], expected, &auth).unwrap().kind, READY);
    }

    #[test]
    fn duplicate_chunk_is_ignored() {
        let auth = Auth::new_from_slice(KEY).unwrap();
        let expected = envelope(3);
        let mut buffer = [0; MAX_PACKET];
        let mut inputs = vec![0.0; 6];
        let mut seen = vec![false; 2];
        let mut remaining = 1;
        for data in [[1.0, -2.0, 3.0], [99.0, 99.0, 99.0]] {
            let n = encode(expected, DATA, 1, 0, &data, &auth, &mut buffer);
            accept_data(
                decode(&buffer[..n], expected, &auth).unwrap(),
                3,
                1,
                &mut inputs,
                &mut seen,
                &mut remaining,
            );
        }
        assert_eq!(remaining, 0);
        assert_eq!(&inputs[3..], &[1.0, -2.0, 3.0]);
    }

    fn two_rank_rounds(drop_packets: bool, multicast: bool) {
        let settings = config();
        let session = Uuid::new_v4();
        let mut first = AllReduceGroup::new(0, 2, session, KEY, settings.clone()).unwrap();
        let mut second = AllReduceGroup::new(1, 2, session, KEY, settings).unwrap();
        if !multicast {
            loopback_fanout(&mut [&mut first, &mut second]);
        }
        if drop_packets {
            first.faults = Faults { data: 7, ready: 3 };
            second.faults = Faults { data: 4, ready: 2 };
        }
        thread::scope(|scope| {
            for (rank, mut group) in [(0, first), (1, second)] {
                scope.spawn(move || {
                    for (step, len) in [CHUNK_ELEMENTS * 2 + 17, 5, 0, CHUNK_ELEMENTS + 1]
                        .into_iter()
                        .enumerate()
                    {
                        // Different rank inputs and partial final chunks, not identical tensors.
                        let input = |r: usize, i: usize| {
                            if r == 0 {
                                i as f32 * 0.5 - step as f32
                            } else {
                                3.0 - i as f32 * 0.25 + step as f32 * 2.0
                            }
                        };
                        let mut values: Vec<_> = (0..len).map(|i| input(rank, i)).collect();
                        // Rank 0 can queue future-step data while rank 1 finishes lingering.
                        if rank == 1 {
                            thread::sleep(Duration::from_millis(15));
                        }
                        group.all_reduce_sum(&mut values).unwrap();
                        let expected: Vec<_> =
                            (0..len).map(|i| input(0, i) + input(1, i)).collect();
                        assert_eq!(values, expected);
                    }
                    assert_eq!(group.step, 4);
                    if drop_packets {
                        assert_eq!((group.faults.data, group.faults.ready), (0, 0));
                    }
                });
            }
        });
    }

    #[test]
    fn loopback_consecutive_nonuniform_partial_vectors() {
        two_rank_rounds(false, false);
    }

    #[test]
    fn loopback_retries_deterministically_dropped_data_and_acks() {
        two_rank_rounds(true, false);
    }

    #[test]
    #[ignore = "requires host IPv4 multicast permission; run with --ignored on a multicast-capable host"]
    fn multicast_loopback_consecutive_rounds_and_loss_recovery() {
        two_rank_rounds(true, true);
    }

    #[test]
    fn missing_rank_timeout_preserves_tensor_and_poisons() {
        let mut settings = config();
        settings.timeout = Duration::from_millis(80);
        let mut group = AllReduceGroup::new(0, 2, Uuid::new_v4(), KEY, settings).unwrap();
        loopback_fanout(&mut [&mut group]);
        let mut values = vec![1.0, -2.0, 3.5];
        let before = values.clone();
        let start = Instant::now();
        assert!(group
            .all_reduce_sum(&mut values)
            .unwrap_err()
            .to_string()
            .contains("timed out"));
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(values, before);
        assert!(group
            .all_reduce_sum(&mut values)
            .unwrap_err()
            .to_string()
            .contains("poisoned"));
        assert_eq!(values, before);
    }

    #[test]
    fn mismatched_lengths_cannot_succeed() {
        let mut settings = config();
        settings.timeout = Duration::from_millis(100);
        let session = Uuid::new_v4();
        let mut first = AllReduceGroup::new(0, 2, session, KEY, settings.clone()).unwrap();
        let mut second = AllReduceGroup::new(1, 2, session, KEY, settings).unwrap();
        loopback_fanout(&mut [&mut first, &mut second]);
        thread::scope(|scope| {
            for (len, group) in [(1, &mut first), (2, &mut second)] {
                scope.spawn(move || {
                    let mut values = vec![7.0; len];
                    assert!(group
                        .all_reduce_sum(&mut values)
                        .unwrap_err()
                        .to_string()
                        .contains("timed out"));
                    assert_eq!(values, vec![7.0; len]);
                });
            }
        });
    }

    #[test]
    fn completion_ack_prevents_leaving_a_peer_without_data() {
        let mut settings = config();
        settings.timeout = Duration::from_millis(100);
        let session = Uuid::new_v4();
        let mut first = AllReduceGroup::new(0, 2, session, KEY, settings.clone()).unwrap();
        let mut second = AllReduceGroup::new(1, 2, session, KEY, settings).unwrap();
        loopback_fanout(&mut [&mut first, &mut second]);
        first.faults.data = usize::MAX;
        thread::scope(|scope| {
            for group in [&mut first, &mut second] {
                scope.spawn(move || {
                    let mut values = vec![7.0];
                    assert!(group
                        .all_reduce_sum(&mut values)
                        .unwrap_err()
                        .to_string()
                        .contains("timed out"));
                    assert_eq!(values, vec![7.0]);
                    assert!(group.poisoned);
                });
            }
        });
    }

    #[test]
    fn rank_order_is_deterministic_despite_reordered_arrival() {
        let settings = config();
        let session = Uuid::new_v4();
        let mut groups: Vec<_> = (0..3)
            .map(|rank| AllReduceGroup::new(rank, 3, session, KEY, settings.clone()).unwrap())
            .collect();
        loopback_fanout(&mut groups.iter_mut().collect::<Vec<_>>());
        groups[0].faults.data = 4;
        thread::scope(|scope| {
            for (rank, mut group) in groups.into_iter().enumerate() {
                scope.spawn(move || {
                    let mut values = vec![[1.0e20, -1.0e20, 3.0][rank]];
                    group.all_reduce_sum(&mut values).unwrap();
                    assert_eq!(values, vec![3.0]);
                });
            }
        });
    }

    #[test]
    fn rejects_invalid_configuration_and_oversized_storage() {
        let settings = config();
        let session = Uuid::new_v4();
        assert!(AllReduceGroup::new(0, 2, session, b"too short", settings.clone()).is_err());
        assert!(AllReduceGroup::new(2, 2, session, KEY, settings.clone()).is_err());
        assert!(AllReduceGroup::new(0, 0, session, KEY, settings.clone()).is_err());
        assert!(
            AllReduceGroup::new(0, MAX_PARTICIPANTS + 1, session, KEY, settings.clone()).is_err()
        );
        assert!(AllReduceGroup::new(0, 2, Uuid::nil(), KEY, settings.clone()).is_err());
        assert!(AllReduceGroup::new(
            0,
            2,
            session,
            KEY,
            MulticastConfig {
                group: Ipv4Addr::LOCALHOST,
                ..settings.clone()
            }
        )
        .is_err());
        assert!(AllReduceGroup::new(
            0,
            2,
            session,
            KEY,
            MulticastConfig {
                timeout: Duration::ZERO,
                ..settings.clone()
            }
        )
        .is_err());
        let mut group = AllReduceGroup::new(0, MAX_PARTICIPANTS, session, KEY, settings).unwrap();
        let mut values = vec![1.0; MAX_TENSOR_ELEMENTS];
        assert!(group.all_reduce_sum(&mut values).is_err());
        assert!(values.iter().all(|&v| v == 1.0));
        assert!(!group.poisoned);
        assert_eq!(group.step, 0);
    }
}
