//! Authenticated point-to-point link probing for the static partitioner.
//!
//! A [`ProbeResponder`] runs on every node; a prober connects to each peer with
//! [`probe_link`] (or [`measure_links`]) and turns live measurements into the
//! [`LinkObservation`] records that [`super::partition::plan_partition`] consumes:
//!
//! * **latency**: several ping-pong rounds of small authenticated control frames.
//!   The reported one-way latency is half the **median** round trip, so a single
//!   scheduling hiccup does not move the estimate.
//! * **bandwidth**: several timed bulk transfers of a bounded payload, each
//!   acknowledged by the responder once every byte has been read. The minimum
//!   ping round trip is subtracted from each transfer (never more than half of it),
//!   and the **median** transfer rate is reported.
//!
//! Control frames carry an HMAC-SHA256 tag over the session UUID, both sides'
//! per-connection nonces and a strictly increasing sequence number, so a
//! non-member cannot answer for a peer and frames cannot be replayed across
//! connections. Bulk payload bytes are deliberately *not* authenticated (hashing
//! would bound the measurement by SHA-256 throughput); only their count is, in the
//! acknowledgement. An on-path attacker can always delay traffic, so measurements
//! are only as trustworthy as the network path. Authentication is not encryption.
//!
//! Probing is one-shot and synchronous: it does not run during inference, and
//! nothing here re-plans a live cluster (see [`super::partition::RepartitionMonitor`]).

use super::partition::LinkObservation;
use anyhow::{bail, ensure, Context, Result};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

type Auth = Hmac<Sha256>;
const MAGIC: &[u8; 8] = b"JPROBE01";
const BODY: usize = 8 + 16 + 16 + 16 + 4 + 8 + 8;
const TAG: usize = 32;
const FRAME_LEN: usize = BODY + TAG;
const CHUNK: usize = 64 * 1024;
const HELLO: u32 = 1;
const WELCOME: u32 = 2;
const PING: u32 = 3;
const PONG: u32 = 4;
const BULK: u32 = 5;
const BULK_ACK: u32 = 6;
const DONE: u32 = 7;
const DONE_ACK: u32 = 8;

/// Hard upper bounds, independent of configuration, on what a responder accepts.
pub const MAX_PING_ROUNDS: usize = 1024;
pub const MAX_BULK_ROUNDS: usize = 32;
pub const MAX_BULK_BYTES: usize = 256 * 1024 * 1024;

/// Probe shape for a prober, and the per-connection limits of a responder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeConfig {
    /// Ping-pong rounds (1..=[`MAX_PING_ROUNDS`]).
    pub ping_rounds: usize,
    /// Bytes per bulk transfer (1..=[`MAX_BULK_BYTES`]).
    pub bulk_bytes: usize,
    /// Timed bulk transfers (1..=[`MAX_BULK_ROUNDS`]).
    pub bulk_rounds: usize,
    /// One deadline covering connection, every round and teardown (50 ms..=120 s).
    pub timeout: Duration,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            ping_rounds: 16,
            bulk_bytes: 8 * 1024 * 1024,
            bulk_rounds: 3,
            timeout: Duration::from_secs(10),
        }
    }
}

impl ProbeConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            (1..=MAX_PING_ROUNDS).contains(&self.ping_rounds),
            "ping rounds must be in 1..={MAX_PING_ROUNDS}"
        );
        ensure!(
            (1..=MAX_BULK_BYTES).contains(&self.bulk_bytes),
            "bulk bytes must be in 1..={MAX_BULK_BYTES}"
        );
        ensure!(
            (1..=MAX_BULK_ROUNDS).contains(&self.bulk_rounds),
            "bulk rounds must be in 1..={MAX_BULK_ROUNDS}"
        );
        ensure!(
            (Duration::from_millis(50)..=Duration::from_secs(120)).contains(&self.timeout),
            "probe timeout must be between 50 milliseconds and 120 seconds"
        );
        Ok(())
    }
}

/// Raw samples and robust summaries of one probed link.
#[derive(Clone, Debug, PartialEq)]
pub struct LinkMeasurement {
    /// Ping round trips in seconds, in measurement order.
    pub rtt_seconds: Vec<f64>,
    /// Bulk transfer rates in bytes per second, in measurement order.
    pub bandwidth_samples: Vec<f64>,
    /// Half the median round trip.
    pub latency_seconds: f64,
    /// Median of `bandwidth_samples`.
    pub bandwidth_bytes_per_second: f64,
}

impl LinkMeasurement {
    /// The planner input for this link between two node IDs.
    pub fn observation(&self, from: impl Into<String>, to: impl Into<String>) -> LinkObservation {
        LinkObservation {
            from: from.into(),
            to: to.into(),
            latency_seconds: self.latency_seconds,
            bandwidth_bytes_per_second: self.bandwidth_bytes_per_second,
        }
    }
}

/// Median of nonempty finite samples (mean of the two middle values when even).
pub fn median(samples: &[f64]) -> Result<f64> {
    ensure!(!samples.is_empty(), "no samples");
    ensure!(
        samples.iter().all(|x| x.is_finite()),
        "samples must be finite"
    );
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Ok(if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Frame {
    kind: u32,
    seq: u64,
    value: u64,
}

struct Channel {
    stream: TcpStream,
    session: Uuid,
    auth: Auth,
    client_nonce: [u8; 16],
    server_nonce: [u8; 16],
}

impl Channel {
    fn body(&self, frame: Frame) -> [u8; BODY] {
        let mut bytes = [0; BODY];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..24].copy_from_slice(self.session.as_bytes());
        bytes[24..40].copy_from_slice(&self.client_nonce);
        bytes[40..56].copy_from_slice(&self.server_nonce);
        bytes[56..60].copy_from_slice(&frame.kind.to_be_bytes());
        bytes[60..68].copy_from_slice(&frame.seq.to_be_bytes());
        bytes[68..76].copy_from_slice(&frame.value.to_be_bytes());
        bytes
    }

    fn send(&mut self, frame: Frame, deadline: Instant) -> Result<()> {
        let mut bytes = [0; FRAME_LEN];
        bytes[..BODY].copy_from_slice(&self.body(frame));
        let mut auth = self.auth.clone();
        auth.update(&bytes[..BODY]);
        bytes[BODY..].copy_from_slice(&auth.finalize().into_bytes());
        write_until(&mut self.stream, &bytes, deadline)
    }

    /// Read one frame; the kind and sequence must match, the value is returned.
    fn receive(&mut self, kind: u32, seq: u64, deadline: Instant) -> Result<u64> {
        let mut bytes = [0; FRAME_LEN];
        read_until(&mut self.stream, &mut bytes, deadline)?;
        let mut auth = self.auth.clone();
        auth.update(&bytes[..BODY]);
        auth.verify_slice(&bytes[BODY..])
            .map_err(|_| anyhow::anyhow!("probe frame authentication failed"))?;
        let value = u64::from_be_bytes(bytes[68..76].try_into().unwrap());
        ensure!(
            bytes[..BODY] == self.body(Frame { kind, seq, value }),
            "probe session, nonce, kind or sequence mismatch"
        );
        Ok(value)
    }
}

/// One node's probe endpoint. Serves probers one connection at a time.
pub struct ProbeResponder {
    listener: TcpListener,
    session: Uuid,
    auth: Auth,
    limits: ProbeConfig,
}

impl ProbeResponder {
    /// Bind a listener. `limits` caps what any one prober may request; its
    /// `timeout` bounds each served connection (and the wait for it).
    pub fn bind(
        address: SocketAddr,
        session: Uuid,
        psk: &[u8],
        limits: ProbeConfig,
    ) -> Result<Self> {
        check_identity(session, psk)?;
        limits.validate()?;
        let listener = TcpListener::bind(address).context("binding probe listener")?;
        listener.set_nonblocking(true)?;
        Ok(Self {
            listener,
            session,
            auth: Auth::new_from_slice(psk).expect("HMAC accepts arbitrary key lengths"),
            limits,
        })
    }

    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    /// Accept and serve exactly one prober within the configured timeout.
    /// Returns the number of bulk payload bytes received.
    pub fn serve_one(&self) -> Result<u64> {
        let deadline = Instant::now() + self.limits.timeout;
        let stream = loop {
            remaining(deadline)?;
            match self.listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    thread::sleep(remaining(deadline)?.min(Duration::from_millis(1)))
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(error).context("accepting prober"),
            }
        };
        stream.set_nonblocking(false)?;
        stream.set_nodelay(true)?;
        let mut channel = Channel {
            stream,
            session: self.session,
            auth: self.auth.clone(),
            client_nonce: [0; 16],
            server_nonce: [0; 16],
        };
        // The hello carries the client's nonce; read it raw, then authenticate.
        let mut hello = [0; FRAME_LEN];
        read_until(&mut channel.stream, &mut hello, deadline)?;
        channel.client_nonce.copy_from_slice(&hello[24..40]);
        let mut auth = channel.auth.clone();
        auth.update(&hello[..BODY]);
        auth.verify_slice(&hello[BODY..])
            .map_err(|_| anyhow::anyhow!("probe hello authentication failed"))?;
        ensure!(
            hello[..BODY]
                == channel.body(Frame {
                    kind: HELLO,
                    seq: 0,
                    value: 0
                }),
            "probe hello session or format mismatch"
        );
        channel.server_nonce = rand::random();
        channel.send(
            Frame {
                kind: WELCOME,
                seq: 0,
                value: 0,
            },
            deadline,
        )?;
        let (mut pings, mut bulks, mut received) = (0usize, 0usize, 0u64);
        let mut scratch = vec![0u8; CHUNK];
        for seq in 1u64.. {
            let mut bytes = [0; FRAME_LEN];
            read_until(&mut channel.stream, &mut bytes, deadline)?;
            let kind = u32::from_be_bytes(bytes[56..60].try_into().unwrap());
            let value = u64::from_be_bytes(bytes[68..76].try_into().unwrap());
            let mut auth = channel.auth.clone();
            auth.update(&bytes[..BODY]);
            auth.verify_slice(&bytes[BODY..])
                .map_err(|_| anyhow::anyhow!("probe frame authentication failed"))?;
            ensure!(
                bytes[..BODY] == channel.body(Frame { kind, seq, value }),
                "probe session, nonce or sequence mismatch"
            );
            match kind {
                PING => {
                    pings += 1;
                    ensure!(pings <= self.limits.ping_rounds, "too many probe pings");
                    channel.send(
                        Frame {
                            kind: PONG,
                            seq,
                            value,
                        },
                        deadline,
                    )?;
                }
                BULK => {
                    bulks += 1;
                    ensure!(bulks <= self.limits.bulk_rounds, "too many bulk probes");
                    ensure!(
                        value > 0 && value <= self.limits.bulk_bytes as u64,
                        "bulk probe size outside responder limits"
                    );
                    let mut left = value as usize;
                    while left > 0 {
                        let n = left.min(CHUNK);
                        read_until(&mut channel.stream, &mut scratch[..n], deadline)?;
                        left -= n;
                    }
                    received += value;
                    channel.send(
                        Frame {
                            kind: BULK_ACK,
                            seq,
                            value,
                        },
                        deadline,
                    )?;
                }
                DONE => {
                    channel.send(
                        Frame {
                            kind: DONE_ACK,
                            seq,
                            value: received,
                        },
                        deadline,
                    )?;
                    return Ok(received);
                }
                _ => bail!("unexpected probe frame kind {kind}"),
            }
        }
        unreachable!("sequence numbers are exhausted only after 2^64 frames")
    }
}

/// Probe one responder: `ping_rounds` round trips, then `bulk_rounds` transfers.
pub fn probe_link(
    address: SocketAddr,
    session: Uuid,
    psk: &[u8],
    config: &ProbeConfig,
) -> Result<LinkMeasurement> {
    check_identity(session, psk)?;
    config.validate()?;
    let deadline = Instant::now() + config.timeout;
    let stream = loop {
        let budget = remaining(deadline)?.min(Duration::from_millis(100));
        match TcpStream::connect_timeout(&address, budget) {
            Ok(stream) => break stream,
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::ConnectionRefused | ErrorKind::TimedOut | ErrorKind::Interrupted
                ) =>
            {
                thread::sleep(remaining(deadline)?.min(Duration::from_millis(5)));
            }
            Err(error) => return Err(error).context("connecting to probe responder"),
        }
    };
    stream.set_nodelay(true)?;
    let mut channel = Channel {
        stream,
        session,
        auth: Auth::new_from_slice(psk).expect("HMAC accepts arbitrary key lengths"),
        client_nonce: rand::random(),
        server_nonce: [0; 16],
    };
    channel.send(
        Frame {
            kind: HELLO,
            seq: 0,
            value: 0,
        },
        deadline,
    )?;
    let mut welcome = [0; FRAME_LEN];
    read_until(&mut channel.stream, &mut welcome, deadline)?;
    channel.server_nonce.copy_from_slice(&welcome[40..56]);
    let mut auth = channel.auth.clone();
    auth.update(&welcome[..BODY]);
    auth.verify_slice(&welcome[BODY..])
        .map_err(|_| anyhow::anyhow!("probe welcome authentication failed"))?;
    ensure!(
        welcome[..BODY]
            == channel.body(Frame {
                kind: WELCOME,
                seq: 0,
                value: 0
            }),
        "probe welcome session or nonce mismatch"
    );

    let mut seq = 0u64;
    let mut rtt_seconds = Vec::with_capacity(config.ping_rounds);
    for _ in 0..config.ping_rounds {
        seq += 1;
        let start = Instant::now();
        channel.send(
            Frame {
                kind: PING,
                seq,
                value: seq,
            },
            deadline,
        )?;
        let echoed = channel.receive(PONG, seq, deadline)?;
        rtt_seconds.push(start.elapsed().as_secs_f64());
        ensure!(echoed == seq, "probe pong value mismatch");
    }
    let min_rtt = rtt_seconds.iter().copied().fold(f64::INFINITY, f64::min);

    let payload = vec![0u8; CHUNK.min(config.bulk_bytes)];
    let mut bandwidth_samples = Vec::with_capacity(config.bulk_rounds);
    for _ in 0..config.bulk_rounds {
        seq += 1;
        let start = Instant::now();
        channel.send(
            Frame {
                kind: BULK,
                seq,
                value: config.bulk_bytes as u64,
            },
            deadline,
        )?;
        let mut left = config.bulk_bytes;
        while left > 0 {
            let n = left.min(payload.len());
            write_until(&mut channel.stream, &payload[..n], deadline)?;
            left -= n;
        }
        let acknowledged = channel.receive(BULK_ACK, seq, deadline)?;
        let total = start.elapsed().as_secs_f64();
        ensure!(
            acknowledged == config.bulk_bytes as u64,
            "bulk acknowledgement byte count mismatch"
        );
        // The acknowledgement adds roughly one round trip of pure latency.
        let transfer = (total - min_rtt).max(total / 2.0).max(1e-9);
        bandwidth_samples.push(config.bulk_bytes as f64 / transfer);
    }
    seq += 1;
    channel.send(
        Frame {
            kind: DONE,
            seq,
            value: 0,
        },
        deadline,
    )?;
    let received = channel.receive(DONE_ACK, seq, deadline)?;
    ensure!(
        received == (config.bulk_bytes as u64) * config.bulk_rounds as u64,
        "responder byte total mismatch"
    );
    let latency_seconds = median(&rtt_seconds)? / 2.0;
    let bandwidth_bytes_per_second = median(&bandwidth_samples)?;
    ensure!(
        bandwidth_bytes_per_second.is_finite() && bandwidth_bytes_per_second > 0.0,
        "measured bandwidth is not finite and positive"
    );
    Ok(LinkMeasurement {
        rtt_seconds,
        bandwidth_samples,
        latency_seconds,
        bandwidth_bytes_per_second,
    })
}

/// Probe each `(node ID, responder address)` peer from `local_id`, sequentially
/// so transfers do not compete for the local link. Any failure fails the call;
/// substitute explicit constants with [`super::partition::resolve_links`] instead
/// of inventing a measurement for an unreachable peer.
pub fn measure_links(
    local_id: &str,
    peers: &[(String, SocketAddr)],
    session: Uuid,
    psk: &[u8],
    config: &ProbeConfig,
) -> Result<Vec<LinkObservation>> {
    ensure!(!local_id.is_empty(), "local node ID must not be empty");
    let mut links = Vec::with_capacity(peers.len());
    for (id, address) in peers {
        ensure!(
            !id.is_empty() && id != local_id,
            "peer IDs must be nonempty and differ from the local node"
        );
        let measurement = probe_link(*address, session, psk, config)
            .with_context(|| format!("probing link {local_id} -> {id} at {address}"))?;
        links.push(measurement.observation(local_id, id.clone()));
    }
    Ok(links)
}

fn check_identity(session: Uuid, psk: &[u8]) -> Result<()> {
    ensure!(
        session.get_version_num() == 4,
        "probe session must be a random v4 UUID"
    );
    ensure!(
        psk.len() >= 32,
        "probing requires a PSK of at least 32 bytes"
    );
    Ok(())
}

fn remaining(deadline: Instant) -> Result<Duration> {
    let budget = deadline.saturating_duration_since(Instant::now());
    ensure!(!budget.is_zero(), "link probe timed out");
    Ok(budget)
}

fn read_until(stream: &mut TcpStream, mut bytes: &mut [u8], deadline: Instant) -> Result<()> {
    while !bytes.is_empty() {
        stream.set_read_timeout(Some(remaining(deadline)?))?;
        match stream.read(bytes) {
            Ok(0) => bail!("probe peer closed the connection"),
            Ok(n) => bytes = &mut bytes[n..],
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                bail!("link probe timed out")
            }
            Err(error) => return Err(error).context("reading probe stream"),
        }
    }
    Ok(())
}

fn write_until(stream: &mut TcpStream, mut bytes: &[u8], deadline: Instant) -> Result<()> {
    while !bytes.is_empty() {
        stream.set_write_timeout(Some(remaining(deadline)?))?;
        match stream.write(bytes) {
            Ok(0) => bail!("probe peer closed the connection"),
            Ok(n) => bytes = &bytes[n..],
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                bail!("link probe timed out")
            }
            Err(error) => return Err(error).context("writing probe stream"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::partition::{
        plan_partition, resolve_links, LinkDefaults, NodeCapacity, TensorWorkload,
    };

    const KEY: &[u8] = b"link-probe-test-only-key-32-bytes!!";

    fn small() -> ProbeConfig {
        ProbeConfig {
            ping_rounds: 9,
            bulk_bytes: 4 * 1024 * 1024,
            bulk_rounds: 3,
            timeout: Duration::from_secs(20),
        }
    }

    fn responder(session: Uuid) -> ProbeResponder {
        ProbeResponder::bind("127.0.0.1:0".parse().unwrap(), session, KEY, small()).unwrap()
    }

    #[test]
    fn median_is_robust_to_one_outlier() -> Result<()> {
        assert_eq!(median(&[3.0, 1.0, 1000.0])?, 3.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 1000.0])?, 3.0);
        assert!(median(&[]).is_err());
        assert!(median(&[1.0, f64::NAN]).is_err());
        Ok(())
    }

    #[test]
    fn loopback_probe_measures_latency_and_bandwidth() -> Result<()> {
        let session = Uuid::new_v4();
        let server = responder(session);
        let address = server.local_addr()?;
        let measurement = thread::scope(|scope| {
            let served = scope.spawn(|| server.serve_one());
            let measured = probe_link(address, session, KEY, &small());
            assert_eq!(
                served.join().expect("responder panicked").unwrap(),
                3 * 4 * 1024 * 1024
            );
            measured
        })?;
        assert_eq!(measurement.rtt_seconds.len(), 9);
        assert_eq!(measurement.bandwidth_samples.len(), 3);
        assert!(measurement.rtt_seconds.iter().all(|&rtt| rtt > 0.0));
        // Loopback: generous bounds that still reject unit errors.
        assert!(
            measurement.latency_seconds > 0.0 && measurement.latency_seconds < 0.05,
            "{measurement:?}"
        );
        assert!(
            measurement.bandwidth_bytes_per_second > 10e6,
            "{measurement:?}"
        );
        eprintln!(
            "loopback probe: median one-way latency {:.1} us, median bandwidth {:.2} GB/s, \
             rtt samples {:?}",
            measurement.latency_seconds * 1e6,
            measurement.bandwidth_bytes_per_second / 1e9,
            measurement.rtt_seconds
        );
        let link = measurement.observation("a", "b");
        assert_eq!((link.from.as_str(), link.to.as_str()), ("a", "b"));
        Ok(())
    }

    #[test]
    fn measured_links_feed_the_planner_with_constant_overrides() -> Result<()> {
        let session = Uuid::new_v4();
        let servers = [responder(session), responder(session)];
        let peers = vec![
            ("b".to_string(), servers[0].local_addr()?),
            ("c".to_string(), servers[1].local_addr()?),
        ];
        let measured = thread::scope(|scope| {
            for server in &servers {
                scope.spawn(move || server.serve_one().unwrap());
            }
            measure_links("a", &peers, session, KEY, &small())
        })?;
        assert_eq!(measured.len(), 2);
        assert!(measured
            .iter()
            .all(|l| l.from == "a" && l.latency_seconds > 0.0));
        let nodes: Vec<_> = ["a", "b", "c"]
            .into_iter()
            .map(|id| NodeCapacity {
                id: id.into(),
                available_bytes: 1 << 20,
                reserved_bytes: 0,
                compute_weight: 1.0,
            })
            .collect();
        let work = [TensorWorkload {
            id: "w".into(),
            input_blocks: 300,
            columns_per_block: 32,
            bytes_per_input_block: 34,
        }];
        // A constant override for b<->c replaces nothing measured; the fallback
        // covers no pair because every pair is now known.
        let slow = LinkObservation {
            from: "c".into(),
            to: "b".into(),
            latency_seconds: 0.1,
            bandwidth_bytes_per_second: 1e6,
        };
        let links = resolve_links(
            &nodes,
            &measured,
            std::slice::from_ref(&slow),
            Some(LinkDefaults {
                latency_seconds: 1.0,
                bandwidth_bytes_per_second: 1.0,
            }),
        )?;
        assert_eq!(links.len(), 3);
        let plan = plan_partition(&nodes, &work, &links)?;
        let counts: Vec<_> = plan.tensors[0]
            .shards
            .iter()
            .map(|s| s.input_blocks.end - s.input_blocks.start)
            .collect();
        // b and c share the slow link; a only sees fast measured loopback links.
        assert!(counts[0] > counts[1] && counts[0] > counts[2], "{counts:?}");
        Ok(())
    }

    #[test]
    fn wrong_key_or_session_fails_closed() {
        for (session, key) in [
            (Uuid::new_v4(), KEY),
            (
                Uuid::nil(),
                b"other-probe-test-only-key-32-bytes!".as_slice(),
            ),
        ] {
            let server_session = Uuid::new_v4();
            let server = ProbeResponder::bind(
                "127.0.0.1:0".parse().unwrap(),
                server_session,
                KEY,
                ProbeConfig {
                    timeout: Duration::from_secs(2),
                    ..small()
                },
            )
            .unwrap();
            let address = server.local_addr().unwrap();
            let session = if session.is_nil() {
                server_session
            } else {
                session
            };
            thread::scope(|scope| {
                let served = scope.spawn(|| server.serve_one());
                let config = ProbeConfig {
                    timeout: Duration::from_secs(2),
                    ..small()
                };
                assert!(probe_link(address, session, key, &config).is_err());
                assert!(served.join().unwrap().is_err());
            });
        }
    }

    #[test]
    fn responder_limits_bound_prober_requests() {
        let session = Uuid::new_v4();
        let server = ProbeResponder::bind(
            "127.0.0.1:0".parse().unwrap(),
            session,
            KEY,
            ProbeConfig {
                bulk_bytes: 1024,
                timeout: Duration::from_secs(2),
                ..small()
            },
        )
        .unwrap();
        let address = server.local_addr().unwrap();
        thread::scope(|scope| {
            let served = scope.spawn(|| server.serve_one());
            let config = ProbeConfig {
                bulk_bytes: 2048,
                timeout: Duration::from_secs(2),
                ..small()
            };
            assert!(probe_link(address, session, KEY, &config).is_err());
            assert!(served.join().unwrap().is_err());
        });
    }

    #[test]
    fn invalid_configuration_and_identity_are_rejected() {
        let address: SocketAddr = "127.0.0.1:1".parse().unwrap();
        for bad in [
            ProbeConfig {
                ping_rounds: 0,
                ..small()
            },
            ProbeConfig {
                bulk_bytes: MAX_BULK_BYTES + 1,
                ..small()
            },
            ProbeConfig {
                bulk_rounds: 0,
                ..small()
            },
            ProbeConfig {
                timeout: Duration::from_millis(1),
                ..small()
            },
        ] {
            assert!(probe_link(address, Uuid::new_v4(), KEY, &bad).is_err());
        }
        assert!(probe_link(address, Uuid::nil(), KEY, &small()).is_err());
        assert!(probe_link(address, Uuid::new_v4(), b"short", &small()).is_err());
        assert!(
            measure_links("a", &[("a".into(), address)], Uuid::new_v4(), KEY, &small()).is_err()
        );
        let server = responder(Uuid::new_v4());
        let unused = ProbeConfig {
            timeout: Duration::from_millis(60),
            ..small()
        };
        let quiet = ProbeResponder {
            limits: unused,
            ..server
        };
        assert!(quiet.serve_one().is_err(), "no prober must time out");
    }
}
