//! Authenticated, bounded TCP ring all-reduce for an explicitly configured cluster.
//!
//! Each rank connects to its successor and accepts its predecessor, reusing those
//! connections across calls. Inputs circulate around the ring, then are summed in
//! ascending rank order (not arrival order). Storage is bounded by the UDP limits:
//! 64 ranks, 262,144 elements per tensor and 4,194,304 aggregate input elements.
//! Fixed-size authenticated frames use bounded concurrent nonblocking I/O so large
//! vectors cannot deadlock on socket buffers. One deadline covers setup and exchange.
//!
//! All ranks need identical ordered peer addresses, a fresh random v4 session UUID,
//! a high-entropy PSK of at least 32 bytes, and the same tensor lengths/call order.
//! Authentication is not encryption; a member with the PSK can impersonate another.
//! Discovery never supplies membership or trust. Select TCP before the collective;
//! never switch transports after a partially failed UDP collective.
//!
//! Failures preserve the caller's input and poison the group; drop it and establish
//! a fresh session on every rank to recover. As with UDP, failures cannot guarantee
//! globally atomic success (a peer may fail after another has returned). This is a
//! synchronous, static ring, not a fault-tolerant or performance-tuned collective.

use super::collective::{MAX_PARTICIPANTS, MAX_TENSOR_ELEMENTS, MAX_TOTAL_ELEMENTS};
use anyhow::{bail, ensure, Context, Result};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

type Auth = Hmac<Sha256>;
const MAGIC: &[u8; 8] = b"JTCPAR01";
const HEADER: usize = 88;
const CHUNK_ELEMENTS: usize = 1024;
const TAG: usize = 32;
const FRAME_LEN: usize = HEADER + CHUNK_ELEMENTS * 4 + TAG;
const HELLO: u32 = 1;
const DATA: u32 = 2;
const READY: u32 = 3;

/// Explicit endpoints ordered by rank; no discovery or coordinator is involved.
#[derive(Clone, Debug)]
pub struct TcpConfig {
    /// Distinct unicast addresses with nonzero ports; wildcards are not allowed.
    pub peers: Vec<SocketAddr>,
    /// Overall deadline per collective, between 50 milliseconds and 120 seconds.
    pub timeout: Duration,
}

/// A fixed rank in an authenticated static TCP ring.
pub struct TcpAllReduceGroup {
    listener: TcpListener,
    streams: Option<(TcpStream, TcpStream)>,
    config: TcpConfig,
    rank: usize,
    session: Uuid,
    membership: [u8; 32],
    auth: Auth,
    step: u64,
    poisoned: bool,
}

#[derive(Clone, Copy)]
struct Frame {
    kind: u32,
    rank: usize,
    total: usize,
    round: usize,
    origin: usize,
    chunk: usize,
}

impl TcpAllReduceGroup {
    /// Bind immediately; connect only on the first collective so constructors can
    /// run sequentially. Distribute the PSK and `Uuid::new_v4()` session out of band.
    pub fn new(rank: usize, session: Uuid, psk: &[u8], config: TcpConfig) -> Result<Self> {
        ensure!(
            (1..=MAX_PARTICIPANTS).contains(&config.peers.len()) && rank < config.peers.len(),
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
        ensure!(
            (Duration::from_millis(50)..=Duration::from_secs(120)).contains(&config.timeout),
            "timeout must be between 50 milliseconds and 120 seconds"
        );
        for (index, peer) in config.peers.iter().enumerate() {
            let ip = peer.ip().to_canonical();
            ensure!(
                peer.port() != 0
                    && !ip.is_unspecified()
                    && !ip.is_multicast()
                    && !matches!(ip, IpAddr::V4(addr) if addr.is_broadcast()),
                "TCP peers require explicit unicast addresses and nonzero ports"
            );
            ensure!(
                !config.peers[..index]
                    .iter()
                    .any(|other| other.ip().to_canonical() == ip && other.port() == peer.port()),
                "TCP peer addresses must be distinct"
            );
        }
        let membership = membership_digest(&config.peers);
        let listener =
            TcpListener::bind(config.peers[rank]).context("binding TCP ring listener")?;
        listener.set_nonblocking(true)?;
        Ok(Self {
            listener,
            streams: None,
            config,
            rank,
            session,
            membership,
            auth: Auth::new_from_slice(psk).expect("HMAC accepts arbitrary key lengths"),
            step: 0,
            poisoned: false,
        })
    }

    /// Sum equal-length f32 vectors in ascending rank order, including empty calls.
    /// Exchange errors leave inputs unchanged and poison this group. Local size
    /// validation errors neither poison the group nor consume a step.
    pub fn all_reduce_sum(&mut self, values: &mut [f32]) -> Result<()> {
        ensure!(
            !self.poisoned,
            "all-reduce group is poisoned; create a new session"
        );
        let peers = self.config.peers.len();
        ensure!(
            values.len() <= MAX_TENSOR_ELEMENTS
                && values
                    .len()
                    .checked_mul(peers)
                    .is_some_and(|n| n <= MAX_TOTAL_ELEMENTS),
            "all-reduce tensor exceeds bounded storage limits"
        );
        let next_step = self
            .step
            .checked_add(1)
            .context("collective step exhausted")?;
        let deadline = Instant::now() + self.config.timeout;
        self.poisoned = true;
        let result = self.exchange(values, deadline).and_then(|mut inputs| {
            // Fold into rank zero's storage; no caller-visible writes before success.
            for index in 0..values.len() {
                inputs[index] =
                    (0..peers).fold(0.0, |sum, rank| sum + inputs[rank * values.len() + index]);
            }
            remaining(deadline)?;
            Ok(inputs)
        });
        match result {
            Ok(inputs) => {
                values.copy_from_slice(&inputs[..values.len()]);
                self.step = next_step;
                self.poisoned = false;
                Ok(())
            }
            Err(error) => {
                if let Some((incoming, outgoing)) = self.streams.take() {
                    let _ = incoming.shutdown(Shutdown::Both);
                    let _ = outgoing.shutdown(Shutdown::Both);
                }
                Err(error)
            }
        }
    }

    fn connect(&mut self, deadline: Instant) -> Result<()> {
        if self.streams.is_some() {
            return Ok(());
        }
        let successor = self.config.peers[(self.rank + 1) % self.config.peers.len()];
        let outgoing = loop {
            let budget = remaining(deadline)?.min(Duration::from_millis(10));
            match TcpStream::connect_timeout(&successor, budget) {
                Ok(stream) => break stream,
                Err(error)
                    if matches!(
                        error.kind(),
                        ErrorKind::ConnectionRefused
                            | ErrorKind::TimedOut
                            | ErrorKind::WouldBlock
                            | ErrorKind::Interrupted
                    ) =>
                {
                    pause(deadline)?;
                }
                Err(error) => return Err(error).context("connecting TCP ring successor"),
            }
        };
        outgoing.set_nonblocking(true)?;
        outgoing.set_nodelay(true)?;
        let incoming = loop {
            remaining(deadline)?;
            match self.listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == ErrorKind::WouldBlock => pause(deadline)?,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(error).context("accepting TCP ring predecessor"),
            }
        };
        incoming.set_nonblocking(true)?;
        incoming.set_nodelay(true)?;
        self.streams = Some((incoming, outgoing));
        Ok(())
    }

    fn exchange(&mut self, values: &[f32], deadline: Instant) -> Result<Vec<f32>> {
        let peers = self.config.peers.len();
        let total = values.len();
        // All allocations depend solely on validated local sizes, never wire fields.
        let mut inputs = vec![0.0; peers * total];
        inputs[self.rank * total..(self.rank + 1) * total].copy_from_slice(values);
        if peers == 1 {
            return Ok(inputs);
        }
        self.connect(deadline)?;
        let predecessor = (self.rank + peers - 1) % peers;
        let hello = Frame {
            kind: HELLO,
            rank: self.rank,
            total,
            round: 0,
            origin: self.rank,
            chunk: 0,
        };
        self.transfer(
            hello,
            Frame {
                rank: predecessor,
                origin: predecessor,
                ..hello
            },
            &[],
            &mut [],
            deadline,
        )?;
        for round in 0..peers - 1 {
            let origin = (self.rank + peers - round) % peers;
            let received_origin = (origin + peers - 1) % peers;
            for chunk in 0..total.div_ceil(CHUNK_ELEMENTS) {
                let offset = chunk * CHUNK_ELEMENTS;
                let len = (total - offset).min(CHUNK_ELEMENTS);
                let send = Frame {
                    kind: DATA,
                    round,
                    origin,
                    chunk,
                    ..hello
                };
                let receive = Frame {
                    rank: predecessor,
                    origin: received_origin,
                    ..send
                };
                // The source and destination ranks differ; a bounded scratch chunk
                // avoids aliasing while keeping storage within aggregate limits.
                let mut chunk_values = [0.0; CHUNK_ELEMENTS];
                self.transfer(
                    send,
                    receive,
                    &inputs[origin * total + offset..origin * total + offset + len],
                    &mut chunk_values[..len],
                    deadline,
                )?;
                inputs[received_origin * total + offset..received_origin * total + offset + len]
                    .copy_from_slice(&chunk_values[..len]);
            }
        }
        // Circulating completion tokens proves every member reached the end of the
        // data exchange. It does not promise atomic success after a network failure.
        for round in 0..peers - 1 {
            let origin = (self.rank + peers - round) % peers;
            let send = Frame {
                kind: READY,
                round,
                origin,
                ..hello
            };
            self.transfer(
                send,
                Frame {
                    rank: predecessor,
                    origin: (origin + peers - 1) % peers,
                    ..send
                },
                &[],
                &mut [],
                deadline,
            )?;
        }
        Ok(inputs)
    }

    fn header(&self, frame: Frame) -> [u8; HEADER] {
        let mut bytes = [0; HEADER];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..24].copy_from_slice(self.session.as_bytes());
        bytes[24..56].copy_from_slice(&self.membership);
        bytes[56..64].copy_from_slice(&self.step.to_be_bytes());
        for (field, value) in [
            frame.rank as u32,
            frame.total as u32,
            frame.round as u32,
            frame.origin as u32,
            frame.chunk as u32,
            frame.kind,
        ]
        .into_iter()
        .enumerate()
        {
            bytes[64 + field * 4..68 + field * 4].copy_from_slice(&value.to_be_bytes());
        }
        bytes
    }

    fn encode(&self, frame: Frame, values: &[f32]) -> [u8; FRAME_LEN] {
        let mut bytes = [0; FRAME_LEN];
        bytes[..HEADER].copy_from_slice(&self.header(frame));
        for (dest, value) in bytes[HEADER..].chunks_exact_mut(4).zip(values.iter()) {
            dest.copy_from_slice(&value.to_bits().to_be_bytes());
        }
        let mut auth = self.auth.clone();
        auth.update(&bytes[..FRAME_LEN - TAG]);
        bytes[FRAME_LEN - TAG..].copy_from_slice(&auth.finalize().into_bytes());
        bytes
    }

    fn decode(&self, bytes: &[u8], expected: Frame, values: &mut [f32]) -> Result<()> {
        ensure!(bytes.len() == FRAME_LEN, "invalid TCP ring frame length");
        let mut auth = self.auth.clone();
        auth.update(&bytes[..FRAME_LEN - TAG]);
        auth.verify_slice(&bytes[FRAME_LEN - TAG..])
            .map_err(|_| anyhow::anyhow!("TCP ring authentication failed"))?;
        ensure!(
            bytes[..HEADER] == self.header(expected),
            "TCP ring session, membership or collective frame mismatch"
        );
        ensure!(
            bytes[HEADER + values.len() * 4..FRAME_LEN - TAG]
                .iter()
                .all(|&byte| byte == 0),
            "invalid TCP ring frame padding"
        );
        for (value, source) in values.iter_mut().zip(bytes[HEADER..].chunks_exact(4)) {
            *value = f32::from_bits(u32::from_be_bytes(source.try_into().unwrap()));
        }
        Ok(())
    }

    fn transfer(
        &mut self,
        send: Frame,
        receive: Frame,
        source: &[f32],
        destination: &mut [f32],
        deadline: Instant,
    ) -> Result<()> {
        let transmit = self.encode(send, source);
        let mut received = [0; FRAME_LEN];
        let (incoming, outgoing) = self.streams.as_mut().context("TCP ring not connected")?;
        duplex(incoming, outgoing, &transmit, &mut received, deadline)?;
        self.decode(&received, receive, destination)
    }
}

fn membership_digest(peers: &[SocketAddr]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(MAGIC);
    digest.update((peers.len() as u32).to_be_bytes());
    for peer in peers {
        match peer {
            SocketAddr::V4(address) => {
                digest.update([4]);
                digest.update(address.ip().octets());
                digest.update(address.port().to_be_bytes());
            }
            SocketAddr::V6(address) => {
                digest.update([6]);
                digest.update(address.ip().octets());
                digest.update(address.port().to_be_bytes());
                digest.update(address.flowinfo().to_be_bytes());
                digest.update(address.scope_id().to_be_bytes());
            }
        }
    }
    digest.finalize().into()
}

fn remaining(deadline: Instant) -> Result<Duration> {
    let budget = deadline.saturating_duration_since(Instant::now());
    ensure!(!budget.is_zero(), "TCP all-reduce timed out");
    Ok(budget)
}

fn pause(deadline: Instant) -> Result<()> {
    thread::sleep(remaining(deadline)?.min(Duration::from_millis(1)));
    Ok(())
}

/// Advance both directions on every iteration; neither send nor receive can hold
/// up its counterpart, and partial traffic never resets the absolute deadline.
fn duplex(
    incoming: &mut TcpStream,
    outgoing: &mut TcpStream,
    transmit: &[u8],
    received: &mut [u8],
    deadline: Instant,
) -> Result<()> {
    let (mut sent, mut read) = (0, 0);
    while sent < transmit.len() || read < received.len() {
        remaining(deadline)?;
        let before = (sent, read);
        if sent < transmit.len() {
            match outgoing.write(&transmit[sent..]) {
                Ok(0) => bail!("TCP ring successor closed"),
                Ok(count) => sent += count,
                Err(error)
                    if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => {}
                Err(error) => return Err(error).context("writing TCP ring frame"),
            }
        }
        if read < received.len() {
            match incoming.read(&mut received[read..]) {
                Ok(0) => bail!("TCP ring predecessor closed"),
                Ok(count) => read += count,
                Err(error)
                    if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => {}
                Err(error) => return Err(error).context("reading TCP ring frame"),
            }
        }
        if before == (sent, read) {
            pause(deadline)?;
        }
    }
    remaining(deadline)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &[u8] = b"tcp-ring-test-only-key-32-bytes!!!";

    fn config(peers: usize) -> TcpConfig {
        let reservations: Vec<_> = (0..peers)
            .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
            .collect();
        TcpConfig {
            peers: reservations
                .iter()
                .map(|listener| listener.local_addr().unwrap())
                .collect(),
            timeout: Duration::from_secs(10),
        }
    }

    fn groups(peers: usize) -> Vec<TcpAllReduceGroup> {
        let config = config(peers);
        let session = Uuid::new_v4();
        (0..peers)
            .map(|rank| TcpAllReduceGroup::new(rank, session, KEY, config.clone()).unwrap())
            .collect()
    }

    #[test]
    fn loopback_ring_many_steps_empty_and_large_vectors() {
        for peers in [1, 2, 3, 5] {
            let groups = groups(peers);
            thread::scope(|scope| {
                for mut group in groups {
                    scope.spawn(move || {
                        let mut connection = None;
                        for (step, len) in [0, 7, MAX_TENSOR_ELEMENTS, 1025, 0, 1]
                            .into_iter()
                            .enumerate()
                        {
                            let input = |rank: usize, index: usize| {
                                index as f32 * 0.25 + rank as f32 - step as f32
                            };
                            let mut values: Vec<_> =
                                (0..len).map(|index| input(group.rank, index)).collect();
                            if group.rank == 0 {
                                thread::sleep(Duration::from_millis(2));
                            }
                            group.all_reduce_sum(&mut values).unwrap();
                            if let Some((incoming, outgoing)) = &group.streams {
                                let endpoints = (
                                    incoming.peer_addr().unwrap(),
                                    outgoing.local_addr().unwrap(),
                                );
                                assert_eq!(*connection.get_or_insert(endpoints), endpoints);
                            }
                            for (index, value) in values.into_iter().enumerate() {
                                assert_eq!(
                                    value,
                                    (0..peers).fold(0.0, |sum, rank| sum + input(rank, index))
                                );
                            }
                        }
                        assert_eq!(group.step, 6);
                    });
                }
            });
        }
    }

    #[test]
    fn non_associative_values_sum_in_ascending_rank_order() {
        thread::scope(|scope| {
            for mut group in groups(3) {
                scope.spawn(move || {
                    let mut values = [[1e20, 3.0], [-1e20, 1e20], [1.0, -1e20]][group.rank];
                    group.all_reduce_sum(&mut values).unwrap();
                    assert_eq!(values, [1.0, 0.0]);
                });
            }
        });
    }

    #[test]
    fn wrong_key_session_membership_length_and_step_fail_closed() {
        for fault in 0..5 {
            let mut groups = groups(3);
            match fault {
                0 => groups[1].auth = Auth::new_from_slice(&[9; 32]).unwrap(),
                1 => groups[1].session = Uuid::new_v4(),
                2 => {
                    let mut peers = groups[1].config.peers.clone();
                    peers.swap(0, 2);
                    groups[1].membership = membership_digest(&peers);
                }
                3 => {}
                4 => groups[1].step = 1,
                _ => unreachable!(),
            }
            thread::scope(|scope| {
                for mut group in groups {
                    scope.spawn(move || {
                        group.config.timeout = Duration::from_secs(1);
                        let len = if fault == 3 && group.rank == 1 { 2 } else { 3 };
                        let mut values = vec![7.0; len];
                        assert!(group.all_reduce_sum(&mut values).is_err());
                        assert_eq!(values, vec![7.0; len]);
                        assert!(group.poisoned);
                        assert!(group.streams.is_none());
                        assert!(group.all_reduce_sum(&mut values).is_err());
                    });
                }
            });
        }
    }

    #[test]
    fn missing_rank_times_out_without_changing_input_and_poisons() {
        for missing_successor in [true, false] {
            let mut config = config(3);
            config.timeout = Duration::from_millis(80);
            let _successor =
                (!missing_successor).then(|| TcpListener::bind(config.peers[1]).unwrap());
            let mut group = TcpAllReduceGroup::new(0, Uuid::new_v4(), KEY, config).unwrap();
            let mut values = [1.0, -2.0, 3.5];
            let start = Instant::now();
            assert!(group
                .all_reduce_sum(&mut values)
                .unwrap_err()
                .to_string()
                .contains("timed out"));
            assert!(start.elapsed() < Duration::from_secs(1));
            assert_eq!(values, [1.0, -2.0, 3.5]);
            assert!(group
                .all_reduce_sum(&mut values)
                .unwrap_err()
                .to_string()
                .contains("poisoned"));
        }
    }

    #[test]
    fn malformed_and_unauthenticated_wire_frames_preserve_input() {
        for authenticated in [false, true] {
            let mut groups = groups(2);
            let mut peer = groups.pop().unwrap();
            let mut group = groups.pop().unwrap();
            let mut frame = peer.encode(
                Frame {
                    kind: if authenticated { 99 } else { HELLO },
                    rank: 1,
                    total: 1,
                    round: 0,
                    origin: 1,
                    chunk: 0,
                },
                &[],
            );
            if !authenticated {
                frame[FRAME_LEN - 1] ^= 1;
            }
            thread::scope(|scope| {
                scope.spawn(move || {
                    let deadline = Instant::now() + Duration::from_secs(1);
                    peer.connect(deadline).unwrap();
                    let (incoming, outgoing) = peer.streams.as_mut().unwrap();
                    // The victim may close immediately after reading the bad frame.
                    let _ = duplex(incoming, outgoing, &frame, &mut [0; FRAME_LEN], deadline);
                });
                let mut values = [7.0];
                let error = group.all_reduce_sum(&mut values).unwrap_err();
                assert!(
                    error.to_string().contains(if authenticated {
                        "frame mismatch"
                    } else {
                        "authentication failed"
                    }),
                    "{error}"
                );
                assert_eq!(values, [7.0]);
                assert!(group.poisoned);
            });
        }
    }

    #[test]
    fn later_step_length_mismatch_preserves_the_new_input() {
        thread::scope(|scope| {
            for mut group in groups(3) {
                scope.spawn(move || {
                    group.all_reduce_sum(&mut [1.0]).unwrap();
                    let mut values = vec![7.0; group.rank];
                    assert!(group.all_reduce_sum(&mut values).is_err());
                    assert_eq!(values, vec![7.0; group.rank]);
                    assert!(group.poisoned);
                    assert_eq!(group.step, 1);
                });
            }
        });
    }

    #[test]
    fn frame_authentication_and_strict_expected_metadata() {
        let mut group = groups(1).pop().unwrap();
        let expected = Frame {
            kind: DATA,
            rank: 0,
            total: 3,
            round: 0,
            origin: 0,
            chunk: 0,
        };
        let mut bytes = group.encode(expected, &[1.25, -2.0, 9.0]);
        let mut values = [0.0; 3];
        group.decode(&bytes, expected, &mut values).unwrap();
        assert_eq!(values, [1.25, -2.0, 9.0]);
        // Every byte, including padding, is authenticated.
        for index in 0..FRAME_LEN {
            bytes[index] ^= 1;
            assert!(group.decode(&bytes, expected, &mut values).is_err());
            bytes[index] ^= 1;
        }
        for len in [0, HEADER, FRAME_LEN - TAG, FRAME_LEN - 1] {
            assert!(group.decode(&bytes[..len], expected, &mut values).is_err());
        }
        assert!(group
            .decode(&[0; FRAME_LEN + 1], expected, &mut values)
            .is_err());
        for frame in [
            Frame {
                kind: 99,
                ..expected
            },
            Frame {
                rank: 1,
                ..expected
            },
            Frame {
                total: usize::MAX,
                ..expected
            },
            Frame {
                round: 1,
                ..expected
            },
            Frame {
                origin: 1,
                ..expected
            },
            Frame {
                chunk: 1,
                ..expected
            },
        ] {
            let invalid = group.encode(frame, &[1.0; 3]);
            assert!(group.decode(&invalid, expected, &mut values).is_err());
        }
        let bad_padding = group.encode(expected, &[1.0; 4]);
        assert!(group.decode(&bad_padding, expected, &mut values).is_err());
        group.step += 1;
        assert!(group.decode(&bytes, expected, &mut values).is_err());
    }

    #[test]
    fn validates_configuration_and_storage_before_networking() {
        let config = config(1);
        let session = Uuid::new_v4();
        assert!(TcpAllReduceGroup::new(1, session, KEY, config.clone()).is_err());
        assert!(TcpAllReduceGroup::new(0, Uuid::nil(), KEY, config.clone()).is_err());
        assert!(TcpAllReduceGroup::new(0, session, &[0; 31], config.clone()).is_err());
        for peers in [
            vec![],
            vec![config.peers[0]; 2],
            vec![config.peers[0]; MAX_PARTICIPANTS + 1],
            vec!["127.0.0.1:0".parse().unwrap()],
            vec!["0.0.0.0:1234".parse().unwrap()],
            vec!["[::]:1234".parse().unwrap()],
            vec!["[::ffff:0.0.0.0]:1234".parse().unwrap()],
            vec!["[::ffff:239.1.2.3]:1234".parse().unwrap()],
            vec![
                "127.0.0.1:1234".parse().unwrap(),
                "[::ffff:127.0.0.1]:1234".parse().unwrap(),
            ],
            vec!["239.1.2.3:1234".parse().unwrap()],
            vec!["255.255.255.255:1234".parse().unwrap()],
        ] {
            assert!(TcpAllReduceGroup::new(
                0,
                session,
                KEY,
                TcpConfig {
                    peers,
                    ..config.clone()
                }
            )
            .is_err());
        }
        for timeout in [Duration::from_millis(49), Duration::from_secs(121)] {
            assert!(TcpAllReduceGroup::new(
                0,
                session,
                KEY,
                TcpConfig {
                    timeout,
                    ..config.clone()
                }
            )
            .is_err());
        }
        let mut group = TcpAllReduceGroup::new(0, session, KEY, config).unwrap();
        let mut values = vec![1.0; MAX_TENSOR_ELEMENTS + 1];
        assert!(group.all_reduce_sum(&mut values).is_err());
        assert!(values.iter().all(|&value| value == 1.0));
        assert!(!group.poisoned);
        assert_eq!(group.step, 0);
        group.all_reduce_sum(&mut [2.0]).unwrap();
        let mut group = TcpAllReduceGroup::new(0, Uuid::new_v4(), KEY, self::config(17)).unwrap();
        let mut values = vec![1.0; MAX_TOTAL_ELEMENTS / 17 + 1];
        assert!(group.all_reduce_sum(&mut values).is_err());
        assert!(!group.poisoned);
        assert_eq!(group.step, 0);
    }

    #[test]
    fn partial_frame_trickle_cannot_extend_deadline() {
        let mut config = config(2);
        config.timeout = Duration::from_millis(80);
        let successor = TcpListener::bind(config.peers[1]).unwrap();
        let mut group = TcpAllReduceGroup::new(0, Uuid::new_v4(), KEY, config.clone()).unwrap();
        thread::scope(|scope| {
            scope.spawn(move || {
                let mut predecessor = TcpStream::connect(config.peers[0]).unwrap();
                let (_outgoing, _) = successor.accept().unwrap();
                for _ in 0..20 {
                    if predecessor.write_all(&[0]).is_err() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
            });
            let start = Instant::now();
            let mut values = [7.0];
            assert!(group
                .all_reduce_sum(&mut values)
                .unwrap_err()
                .to_string()
                .contains("timed out"));
            assert!(start.elapsed() < Duration::from_millis(500));
            assert_eq!(values, [7.0]);
            assert!(group.poisoned);
        });
    }
}
