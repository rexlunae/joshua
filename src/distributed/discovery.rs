//! Advisory LAN discovery; mDNS announcements are **not authenticated**.
//!
//! Never use membership, resource claims, or coordinator election to authorize
//! inference. Authenticate peers separately and freeze an explicit participant
//! set before starting a distributed operation.
//!
//! The caller supplies a UUID and must persist it outside this API (for example
//! in its node configuration). Call [`Discovery::poll`] continuously: it publishes
//! a changing TXT heartbeat every five seconds, rather than relying on the much
//! longer mDNS cache refresh interval.
//! These are discovery-only heartbeats, not all-reduce health checks; they do
//! not authorize participant replacement or provide collective failover.
//!
//! TXT validation applies to the properties exposed by `mdns-sd`; that library
//! canonicalizes duplicate wire keys by keeping the first value. Distinct live
//! service instances claiming the same UUID are excluded from peer selection.

use anyhow::{anyhow, ensure, Context, Result};
use mdns_sd::{Receiver, ServiceDaemon, ServiceEvent, ServiceInfo};
use std::collections::{BTreeMap, HashMap};
use std::net::IpAddr;
use std::time::{Duration, Instant};
use uuid::Uuid;

pub const SERVICE_TYPE: &str = "_joshua._tcp.local.";
pub const DEFAULT_PORT: u16 = 5359;
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
pub const PEER_TTL: Duration = Duration::from_secs(15);
const MAX_PEERS: usize = 256;
const MAX_TXT_BYTES: usize = 1024;
const MAX_ADDRESSES: usize = 32;

/// Untrusted advertised capabilities. Memory values are MiB, zero means unknown.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeInfo {
    pub uuid: Uuid,
    pub version: String,
    pub ram_mb: u64,
    pub ram_avail_mb: u64,
    pub cpu_cores: u32,
    pub simd: String,
    /// Discovery-service uptime in seconds at the latest advertisement.
    /// This is never OS boot age or a wall-clock timestamp.
    pub uptime: u64,
}

impl NodeInfo {
    /// Collect local capabilities using a caller-owned, persistently stored UUID.
    /// No identity is generated or persisted by this module.
    pub fn local(uuid: Uuid) -> Self {
        let (ram_mb, ram_avail_mb) = local_resources();
        Self {
            uuid,
            version: env!("CARGO_PKG_VERSION").to_owned(),
            ram_mb,
            ram_avail_mb,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get().min(u32::MAX as usize) as u32)
                .unwrap_or(1),
            simd: runtime_simd().to_owned(),
            uptime: 0,
        }
    }

    fn properties(&self, heartbeat: u64) -> BTreeMap<String, String> {
        [
            ("uuid", self.uuid.to_string()),
            ("version", self.version.clone()),
            ("ram_mb", self.ram_mb.to_string()),
            ("ram_avail_mb", self.ram_avail_mb.to_string()),
            ("cpu_cores", self.cpu_cores.to_string()),
            ("simd", self.simd.clone()),
            ("uptime", self.uptime.to_string()),
            ("heartbeat", heartbeat.to_string()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect()
    }
}

/// A live, compatible advertisement, not an authenticated inference participant.
#[derive(Clone, Debug)]
pub struct PeerInfo {
    pub node: NodeInfo,
    /// Routable IP addresses. IPv6 link-local addresses are omitted because this
    /// representation does not carry the required interface scope.
    pub addresses: Vec<IpAddr>,
    pub port: u16,
    pub last_seen: Instant,
}

struct Advertisement {
    peer: PeerInfo,
    heartbeat: u64,
    age_at_last_seen: Duration,
}

struct PeerTable {
    local: NodeInfo,
    local_observed_at: Instant,
    by_service: BTreeMap<String, Advertisement>,
}

impl PeerTable {
    fn new(local: NodeInfo, now: Instant) -> Self {
        Self {
            local,
            local_observed_at: now,
            by_service: BTreeMap::new(),
        }
    }

    fn local_age(&self, now: Instant) -> Duration {
        Duration::from_secs(self.local.uptime)
            .saturating_add(now.saturating_duration_since(self.local_observed_at))
    }

    fn local_node(&self, now: Instant) -> NodeInfo {
        let mut node = self.local.clone();
        node.uptime = self.local_age(now).as_secs();
        node
    }

    fn upsert(
        &mut self,
        fullname: &str,
        node: NodeInfo,
        heartbeat: u64,
        mut addresses: Vec<IpAddr>,
        port: u16,
        now: Instant,
    ) -> Result<()> {
        self.expire(now);
        ensure!(
            fullname.len() <= 255 && fullname.to_ascii_lowercase().ends_with(SERVICE_TYPE),
            "invalid service name"
        );
        ensure!(port != 0, "invalid service port");
        ensure!(
            !addresses.is_empty() && addresses.len() <= MAX_ADDRESSES,
            "invalid address count"
        );
        ensure!(
            node.version == self.local.version,
            "incompatible discovery version"
        );
        if node.uuid == self.local.uuid {
            return Ok(());
        }
        let key = fullname.to_ascii_lowercase();
        let mut age_at_last_seen = Duration::from_secs(node.uptime);
        if let Some(previous) = self.by_service.get(&key) {
            ensure!(previous.peer.node.uuid == node.uuid, "service changed UUID");
            // Address/cache re-resolution is not evidence of a new heartbeat.
            if previous.heartbeat == heartbeat {
                return Ok(());
            }
            if node.uptime >= previous.peer.node.uptime {
                // Keep age monotonic despite integer rounding and network jitter.
                // A lower advertised uptime instead denotes a restarted service.
                age_at_last_seen = age_at_last_seen.max(
                    previous
                        .age_at_last_seen
                        .saturating_add(now.saturating_duration_since(previous.peer.last_seen)),
                );
            }
        } else {
            ensure!(self.by_service.len() < MAX_PEERS, "peer limit reached");
        }
        addresses.sort_unstable();
        addresses.dedup();
        self.by_service.insert(
            key,
            Advertisement {
                peer: PeerInfo {
                    node,
                    addresses,
                    port,
                    last_seen: now,
                },
                heartbeat,
                age_at_last_seen,
            },
        );
        Ok(())
    }

    fn remove(&mut self, fullname: &str) {
        self.by_service.remove(&fullname.to_ascii_lowercase());
    }

    fn expire(&mut self, now: Instant) {
        self.by_service
            .retain(|_, entry| is_live(entry.peer.last_seen, now));
    }

    fn live_entries(&self, now: Instant) -> Vec<&Advertisement> {
        let mut counts = HashMap::new();
        for entry in self.by_service.values() {
            if is_live(entry.peer.last_seen, now) {
                *counts.entry(entry.peer.node.uuid).or_insert(0usize) += 1;
            }
        }
        self.by_service
            .values()
            .filter(|entry| {
                is_live(entry.peer.last_seen, now) && counts[&entry.peer.node.uuid] == 1
            })
            .collect()
    }

    fn peers(&self, now: Instant) -> Vec<PeerInfo> {
        let mut peers: Vec<_> = self
            .live_entries(now)
            .into_iter()
            .map(|entry| entry.peer.clone())
            .collect();
        peers.sort_unstable_by_key(|peer| peer.node.uuid);
        peers
    }

    fn coordinator(&self, now: Instant) -> NodeInfo {
        let (mut node, age) = self
            .live_entries(now)
            .into_iter()
            .map(|entry| {
                (
                    entry.peer.node.clone(),
                    entry
                        .age_at_last_seen
                        .saturating_add(now.saturating_duration_since(entry.peer.last_seen)),
                )
            })
            .chain(std::iter::once((self.local.clone(), self.local_age(now))))
            .max_by_key(|(node, age)| (*age, node.uuid))
            .expect("the local node is always a candidate");
        node.uptime = age.as_secs();
        node
    }
}

fn is_live(last_seen: Instant, now: Instant) -> bool {
    now.saturating_duration_since(last_seen) < PEER_TTL
}

/// Owns an mDNS daemon and its advisory peer table.
///
/// Heartbeats are driven by [`Self::poll`], not a separate worker thread. If the
/// caller stops polling, other nodes intentionally expire this node after 15s.
pub struct Discovery {
    daemon: ServiceDaemon,
    events: Receiver<ServiceEvent>,
    table: PeerTable,
    port: u16,
    fullname: String,
    heartbeat: u64,
    next_heartbeat: Instant,
    stopped: bool,
}

impl Discovery {
    /// Register and browse Joshua nodes. `port` is the advertised application
    /// port, not the mDNS transport port. Zero is rejected. Service uptime starts
    /// at zero here, regardless of the supplied `node.uptime`.
    pub fn new(mut node: NodeInfo, port: u16) -> Result<Self> {
        let started = Instant::now();
        node.uptime = 0;
        ensure!(port != 0, "discovery requires a nonzero application port");
        let properties = node.properties(0);
        parse_properties(
            properties
                .iter()
                .map(|(key, value)| (key.as_str(), Some(value.as_bytes()))),
        )?;
        let service = service_info(&node, port, 0)?;
        let fullname = service.get_fullname().to_owned();
        let daemon = ServiceDaemon::new().context("create mDNS daemon")?;
        let events = match daemon.browse(SERVICE_TYPE) {
            Ok(events) => events,
            Err(error) => {
                let _ = daemon.shutdown();
                return Err(error.into());
            }
        };
        if let Err(error) = daemon.register(service) {
            let _ = daemon.shutdown();
            return Err(error.into());
        }
        Ok(Self {
            daemon,
            events,
            table: PeerTable::new(node, started),
            port,
            fullname,
            heartbeat: 0,
            next_heartbeat: Instant::now() + HEARTBEAT_INTERVAL,
            stopped: false,
        })
    }

    /// Process one event, or wait up to `timeout`, servicing heartbeat deadlines
    /// while waiting. Malformed or incompatible remote advertisements are ignored.
    pub fn poll(&mut self, timeout: Duration) -> Result<()> {
        ensure!(!self.stopped, "discovery is shut down");
        let deadline = Instant::now()
            .checked_add(timeout)
            .context("discovery timeout overflow")?;
        loop {
            let now = Instant::now();
            self.table.expire(now);
            if now >= self.next_heartbeat {
                let heartbeat = self.heartbeat.wrapping_add(1);
                // Re-registering updates TXT and sends an unsolicited response.
                // A changing property makes mdns-sd emit ServiceResolved again.
                self.daemon.register(service_info(
                    &self.table.local_node(now),
                    self.port,
                    heartbeat,
                )?)?;
                self.heartbeat = heartbeat;
                self.next_heartbeat = now + HEARTBEAT_INTERVAL;
            }
            let wait = deadline
                .saturating_duration_since(now)
                .min(self.next_heartbeat.saturating_duration_since(now));
            match self.events.recv_timeout(wait) {
                Ok(event) => {
                    self.handle_event(event, Instant::now());
                    return Ok(());
                }
                Err(mdns_sd::RecvTimeoutError::Timeout) => {
                    if Instant::now() >= deadline {
                        self.table.expire(Instant::now());
                        return Ok(());
                    }
                }
                Err(error) => return Err(anyhow!("mDNS event receiver: {error}")),
            }
        }
    }

    fn handle_event(&mut self, event: ServiceEvent, now: Instant) {
        match event {
            ServiceEvent::ServiceResolved(service) => {
                if service.ty_domain != SERVICE_TYPE || service.addresses.len() > MAX_ADDRESSES {
                    return;
                }
                let Ok((node, heartbeat)) = parse_properties(
                    service
                        .get_properties()
                        .iter()
                        .map(|property| (property.key(), property.val())),
                ) else {
                    return;
                };
                let addresses = service
                    .addresses
                    .iter()
                    .map(|address| address.to_ip_addr())
                    .filter(|address| match address {
                        IpAddr::V4(address) => !address.is_unspecified() && !address.is_multicast(),
                        IpAddr::V6(address) => {
                            !address.is_unspecified()
                                && !address.is_multicast()
                                && !address.is_unicast_link_local()
                        }
                    })
                    .collect();
                let _ = self.table.upsert(
                    &service.fullname,
                    node,
                    heartbeat,
                    addresses,
                    service.port,
                    now,
                );
            }
            ServiceEvent::ServiceRemoved(_, fullname) => self.table.remove(&fullname),
            _ => {}
        }
    }

    /// Live compatible peers, sorted by UUID, excluding self and ambiguous UUIDs.
    /// Expired entries are filtered even if the caller has not polled recently.
    pub fn peers(&self) -> Vec<PeerInfo> {
        self.table.peers(Instant::now())
    }

    /// Advisory election including self: longest-running discovery service wins;
    /// highest UUID (byte order) breaks ties. Remote ages are extrapolated using
    /// local monotonic time, not compared wall clocks. Advertising whole seconds
    /// and network delays limit precision. This is not consensus or authorization.
    pub fn coordinator(&self) -> NodeInfo {
        self.table.coordinator(Instant::now())
    }

    /// Send an mDNS goodbye before stopping the daemon. Idempotent; cleanup is
    /// attempted even if unregister fails. Each acknowledgement waits at most 2s.
    pub fn shutdown(&mut self) -> Result<()> {
        if self.stopped {
            return Ok(());
        }
        self.stopped = true;
        let unregister = self
            .daemon
            .unregister(&self.fullname)
            .map_err(anyhow::Error::from)
            .and_then(|ack| {
                ack.recv_timeout(Duration::from_secs(2))
                    .context("wait for mDNS goodbye")
            });
        let _ = self.daemon.stop_browse(SERVICE_TYPE);
        let shutdown = self
            .daemon
            .shutdown()
            .map_err(anyhow::Error::from)
            .and_then(|ack| {
                ack.recv_timeout(Duration::from_secs(2))
                    .context("wait for mDNS shutdown")
            });
        self.table.by_service.clear();
        unregister?;
        shutdown?;
        Ok(())
    }
}

impl Drop for Discovery {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn service_info(node: &NodeInfo, port: u16, heartbeat: u64) -> Result<ServiceInfo> {
    let properties: HashMap<_, _> = node.properties(heartbeat).into_iter().collect();
    Ok(ServiceInfo::new(
        SERVICE_TYPE,
        &node.uuid.to_string(),
        &format!("joshua-{}.local.", node.uuid),
        "",
        port,
        properties,
    )?
    .enable_addr_auto())
}

fn parse_properties<'a>(
    properties: impl IntoIterator<Item = (&'a str, Option<&'a [u8]>)>,
) -> Result<(NodeInfo, u64)> {
    let mut fields = BTreeMap::new();
    let mut bytes = 0usize;
    for (key, value) in properties {
        ensure!(fields.len() < 8, "too many TXT properties");
        ensure!(key.len() <= 16 && key.is_ascii(), "invalid TXT key");
        let value = value.context("missing TXT value")?;
        ensure!(value.len() <= 64, "TXT value too long");
        bytes = bytes
            .checked_add(key.len() + value.len())
            .context("TXT length overflow")?;
        ensure!(bytes <= MAX_TXT_BYTES, "TXT properties too large");
        let key = key.to_ascii_lowercase();
        ensure!(
            matches!(
                key.as_str(),
                "uuid"
                    | "version"
                    | "ram_mb"
                    | "ram_avail_mb"
                    | "cpu_cores"
                    | "simd"
                    | "uptime"
                    | "heartbeat"
            ),
            "unknown TXT property"
        );
        ensure!(
            fields.insert(key, std::str::from_utf8(value)?).is_none(),
            "duplicate TXT property"
        );
    }
    ensure!(fields.len() == 8, "missing TXT properties");
    let uuid_text = fields["uuid"];
    ensure!(uuid_text.len() == 36, "noncanonical UUID");
    let uuid = Uuid::parse_str(uuid_text).context("invalid UUID")?;
    ensure!(
        !uuid.is_nil() && uuid.to_string().eq_ignore_ascii_case(uuid_text),
        "invalid UUID"
    );
    let version = fields["version"];
    ensure!(
        !version.is_empty()
            && version
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b".-_+".contains(&b)),
        "invalid version"
    );
    let simd = fields["simd"];
    ensure!(
        matches!(simd, "scalar" | "sse2" | "avx2" | "avx512f" | "neon"),
        "invalid SIMD capability"
    );
    let ram_mb = decimal(fields["ram_mb"])?;
    let ram_avail_mb = decimal(fields["ram_avail_mb"])?;
    ensure!(ram_avail_mb <= ram_mb, "available memory exceeds total");
    let cpu_cores = u32::try_from(decimal(fields["cpu_cores"])?)?;
    ensure!(cpu_cores != 0, "invalid core count");
    Ok((
        NodeInfo {
            uuid,
            version: version.to_owned(),
            ram_mb,
            ram_avail_mb,
            cpu_cores,
            simd: simd.to_owned(),
            uptime: decimal(fields["uptime"])?,
        },
        decimal(fields["heartbeat"])?,
    ))
}

fn decimal(value: &str) -> Result<u64> {
    ensure!(
        !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
        "invalid unsigned integer"
    );
    value.parse().context("unsigned integer overflow")
}

fn runtime_simd() -> &'static str {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return "avx512f";
        }
        if std::is_x86_feature_detected!("avx2") {
            return "avx2";
        }
        if std::is_x86_feature_detected!("sse2") {
            return "sse2";
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return "neon";
    }
    "scalar"
}

#[cfg(target_os = "linux")]
fn local_resources() -> (u64, u64) {
    std::fs::read_to_string("/proc/meminfo")
        .map(|text| parse_meminfo(&text))
        .unwrap_or_default()
}

#[cfg(not(target_os = "linux"))]
fn local_resources() -> (u64, u64) {
    (0, 0)
}

#[cfg(any(target_os = "linux", test))]
fn parse_meminfo(text: &str) -> (u64, u64) {
    let field = |name: &str| {
        text.lines().find_map(|line| {
            let mut parts = line.split_whitespace();
            if parts.next()? != name {
                return None;
            }
            let value = parts.next()?.parse::<u64>().ok()?;
            (parts.next()? == "kB" && parts.next().is_none()).then_some(value / 1024)
        })
    };
    let total = field("MemTotal:").unwrap_or(0);
    let available = field("MemAvailable:").unwrap_or(0).min(total);
    (total, available)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u128, uptime: u64) -> NodeInfo {
        NodeInfo {
            uuid: Uuid::from_u128(id),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            ram_mb: 8192,
            ram_avail_mb: 4096,
            cpu_cores: 4,
            simd: "scalar".to_owned(),
            uptime,
        }
    }

    fn insert(table: &mut PeerTable, name: &str, node: NodeInfo, beat: u64, now: Instant) {
        table
            .upsert(
                &format!("{name}.{SERVICE_TYPE}"),
                node,
                beat,
                vec!["192.0.2.1".parse().unwrap()],
                DEFAULT_PORT,
                now,
            )
            .unwrap();
    }

    fn parse(fields: &BTreeMap<String, String>) -> Result<(NodeInfo, u64)> {
        parse_properties(
            fields
                .iter()
                .map(|(key, value)| (key.as_str(), Some(value.as_bytes()))),
        )
    }

    #[test]
    fn add_upsert_remove_and_self_filter() {
        let now = Instant::now();
        let mut table = PeerTable::new(node(1, 100), now);
        insert(&mut table, "self", node(1, 100), 0, now);
        assert!(table.peers(now).is_empty());
        insert(&mut table, "two", node(2, 200), 0, now);
        assert_eq!(table.peers(now).len(), 1);
        let mut updated = node(2, 200);
        updated.ram_avail_mb = 2048;
        insert(
            &mut table,
            "two",
            updated.clone(),
            1,
            now + HEARTBEAT_INTERVAL,
        );
        assert_eq!(table.peers(now + HEARTBEAT_INTERVAL)[0].node, updated);
        assert_eq!(
            table.peers(now + HEARTBEAT_INTERVAL)[0].last_seen,
            now + HEARTBEAT_INTERVAL
        );
        table.remove(&format!("TWO.{SERVICE_TYPE}"));
        assert!(table.peers(now + HEARTBEAT_INTERVAL).is_empty());
    }

    #[test]
    fn exact_expiry_and_repeated_cache_events_do_not_refresh() {
        let now = Instant::now();
        let mut table = PeerTable::new(node(1, 100), now);
        insert(&mut table, "two", node(2, 200), 0, now);
        insert(&mut table, "two", node(2, 200), 0, now + HEARTBEAT_INTERVAL);
        assert_eq!(
            table.peers(now + PEER_TTL - Duration::from_nanos(1)).len(),
            1
        );
        assert!(table.peers(now + PEER_TTL).is_empty());
        table.expire(now + PEER_TTL);
        assert!(table.by_service.is_empty());
    }

    #[test]
    fn changing_heartbeat_keeps_peer_live() {
        let now = Instant::now();
        let mut table = PeerTable::new(node(1, 100), now);
        for beat in 0..20 {
            let time = now + HEARTBEAT_INTERVAL * beat as u32;
            insert(&mut table, "two", node(2, 200), beat, time);
            assert_eq!(table.peers(time).len(), 1);
        }
        assert_eq!(table.peers(now + Duration::from_secs(109)).len(), 1);
        assert!(table.peers(now + Duration::from_secs(110)).is_empty());
    }

    #[test]
    fn election_uses_service_uptime_then_greatest_uuid() {
        let now = Instant::now();
        let mut table = PeerTable::new(node(1, 100), now);
        assert_eq!(table.coordinator(now).uuid, Uuid::from_u128(1));
        insert(&mut table, "two", node(2, 200), 0, now);
        insert(&mut table, "three", node(3, 200), 0, now);
        insert(&mut table, "four", node(4, 150), 0, now);
        assert_eq!(table.coordinator(now).uuid, Uuid::from_u128(3));
        table.remove(&format!("three.{SERVICE_TYPE}"));
        assert_eq!(table.coordinator(now).uuid, Uuid::from_u128(2));
        assert_eq!(table.coordinator(now + PEER_TTL).uuid, Uuid::from_u128(1));
    }

    #[test]
    fn established_local_service_outlives_late_join_despite_larger_uuid() {
        let now = Instant::now();
        let mut table = PeerTable::new(node(1, 0), now);
        let join = now + Duration::from_secs(8);
        insert(&mut table, "late", node(999, 0), 0, join);
        assert_eq!(table.coordinator(join).uuid, Uuid::from_u128(1));
        let refresh = join + HEARTBEAT_INTERVAL;
        insert(&mut table, "late", node(999, 5), 1, refresh);
        let winner = table.coordinator(refresh + Duration::from_secs(3));
        assert_eq!(winner.uuid, Uuid::from_u128(1));
        assert_eq!(winner.uptime, 16);
    }

    #[test]
    fn remote_coordinator_ages_and_departure_elects_next_longest() {
        let now = Instant::now();
        let mut table = PeerTable::new(node(1, 0), now);
        insert(&mut table, "first", node(2, 60), 0, now);
        insert(&mut table, "second", node(3, 30), 0, now);
        let join = now + Duration::from_secs(8);
        insert(&mut table, "late", node(999, 0), 0, join);
        let winner = table.coordinator(join);
        assert_eq!(winner.uuid, Uuid::from_u128(2));
        assert_eq!(winner.uptime, 68);
        let refresh = now + Duration::from_secs(10);
        insert(&mut table, "first", node(2, 70), 1, refresh);
        insert(&mut table, "second", node(3, 40), 1, refresh);
        table.remove(&format!("first.{SERVICE_TYPE}"));
        let winner = table.coordinator(refresh + Duration::from_secs(1));
        assert_eq!(winner.uuid, Uuid::from_u128(3));
        assert_eq!(winner.uptime, 41);
        insert(
            &mut table,
            "late",
            node(999, 12),
            1,
            now + Duration::from_secs(20),
        );
        // The second service expires; the local service predates the late join.
        assert_eq!(
            table.coordinator(now + Duration::from_secs(25)).uuid,
            Uuid::from_u128(1)
        );
    }

    #[test]
    fn peer_age_is_monotonic_between_heartbeats_but_resets_on_restart() {
        let now = Instant::now();
        let mut table = PeerTable::new(node(1, 0), now);
        insert(&mut table, "peer", node(2, 10), 0, now);
        insert(
            &mut table,
            "peer",
            node(2, 15),
            1,
            now + Duration::from_millis(5900),
        );
        assert_eq!(
            table.coordinator(now + Duration::from_millis(6100)).uptime,
            16
        );
        let restart = now + Duration::from_secs(7);
        insert(&mut table, "peer", node(2, 0), 0, restart);
        assert_eq!(table.coordinator(restart).uuid, Uuid::from_u128(1));
    }

    #[test]
    fn equal_service_ages_keep_deterministic_tie_break_as_time_advances() {
        let now = Instant::now();
        let mut table = PeerTable::new(node(1, 0), now);
        insert(&mut table, "peer", node(2, 0), 0, now);
        assert_eq!(table.coordinator(now).uuid, Uuid::from_u128(2));
        let refresh = now + HEARTBEAT_INTERVAL;
        insert(&mut table, "peer", node(2, 5), 1, refresh);
        assert_eq!(
            table.coordinator(refresh + Duration::from_secs(2)).uuid,
            Uuid::from_u128(2)
        );
    }

    #[test]
    fn duplicate_uuid_is_ambiguous_until_conflict_leaves() {
        let now = Instant::now();
        let mut table = PeerTable::new(node(1, 100), now);
        insert(&mut table, "two", node(2, 200), 0, now);
        insert(&mut table, "imposter", node(2, 300), 0, now);
        assert!(table.peers(now).is_empty());
        assert_eq!(table.coordinator(now).uuid, Uuid::from_u128(1));
        table.remove(&format!("imposter.{SERVICE_TYPE}"));
        assert_eq!(table.peers(now).len(), 1);
    }

    #[test]
    fn incompatible_version_and_identity_changes_rejected() {
        let now = Instant::now();
        let mut table = PeerTable::new(node(1, 100), now);
        insert(&mut table, "two", node(2, 200), 0, now);
        for mut other in [node(3, 300), node(2, 300)] {
            if other.uuid == Uuid::from_u128(2) {
                other.version = "incompatible".to_owned();
            }
            assert!(table
                .upsert(
                    &format!("two.{SERVICE_TYPE}"),
                    other,
                    1,
                    vec!["192.0.2.1".parse().unwrap()],
                    DEFAULT_PORT,
                    now + HEARTBEAT_INTERVAL,
                )
                .is_err());
        }
        assert_eq!(table.peers(now)[0].node, node(2, 200));
        assert_eq!(table.peers(now)[0].last_seen, now);
    }

    #[test]
    fn txt_round_trip_and_malformed_fields() {
        let original = node(2, 100);
        let properties = original.properties(123);
        assert_eq!(parse(&properties).unwrap(), (original, 123));
        for (key, value) in [
            ("uuid", "not-a-uuid"),
            ("uuid", "00000000-0000-0000-0000-000000000000"),
            ("version", ""),
            ("version", "bad version"),
            ("ram_mb", "-1"),
            ("ram_mb", "18446744073709551616"),
            ("ram_mb", "1"),
            ("ram_avail_mb", "9000"),
            ("cpu_cores", "4294967296"),
            ("cpu_cores", "0"),
            ("cpu_cores", "+1"),
            ("simd", "unknown"),
            ("uptime", "1.2"),
            ("heartbeat", " 1"),
        ] {
            let mut malformed = properties.clone();
            malformed.insert(key.to_owned(), value.to_owned());
            assert!(parse(&malformed).is_err(), "{key}={value}");
        }
        let mut missing = properties.clone();
        missing.remove("heartbeat");
        assert!(parse(&missing).is_err());
        let mut oversized = properties.clone();
        oversized.insert("version".to_owned(), "x".repeat(65));
        assert!(parse(&oversized).is_err());
        let mut duplicate = properties.clone();
        duplicate.insert("UUID".to_owned(), Uuid::from_u128(3).to_string());
        assert!(parse(&duplicate).is_err());
        assert!(parse_properties([("uuid", Some(&[0xff][..]))]).is_err());
        assert!(parse_properties([("uuid", None)]).is_err());
    }

    #[test]
    fn peer_table_is_bounded() {
        let now = Instant::now();
        let mut table = PeerTable::new(node(1, 100), now);
        for id in 2..MAX_PEERS + 2 {
            insert(&mut table, &id.to_string(), node(id as u128, 0), 0, now);
        }
        assert!(table
            .upsert(
                &format!("overflow.{SERVICE_TYPE}"),
                node(999, 0),
                0,
                vec!["192.0.2.1".parse().unwrap()],
                DEFAULT_PORT,
                now,
            )
            .is_err());
        assert_eq!(table.peers(now).len(), MAX_PEERS);
    }

    #[test]
    fn resource_parser_is_conservative() {
        assert_eq!(
            parse_meminfo("MemTotal: 8388608 kB\nMemAvailable: 2097152 kB\n"),
            (8192, 2048)
        );
        assert_eq!(parse_meminfo("MemFree: 1000 kB\n"), (0, 0));
        assert_eq!(parse_meminfo("MemTotal: 1024 kB\n"), (1, 0));
        assert_eq!(
            parse_meminfo("MemTotal: 1024 kB\nMemAvailable: 2048 kB\n"),
            (1, 1)
        );
        assert_eq!(parse_meminfo("MemTotal: 18446744073709551616 kB\n"), (0, 0));
        assert_eq!(parse_meminfo("MemTotal: 1024 MB\n"), (0, 0));
    }

    #[test]
    fn local_collection_preserves_caller_identity() {
        let identity = Uuid::from_u128(42);
        let local = NodeInfo::local(identity);
        assert_eq!(local.uuid, identity);
        assert_eq!(local.uptime, 0);
        assert!(local.cpu_cores >= 1);
        assert!(local.ram_avail_mb <= local.ram_mb);
        assert!(parse(&local.properties(0)).is_ok());
    }

    #[test]
    fn heartbeat_updates_txt_and_service_uptime_without_changing_identity() {
        let now = Instant::now();
        let table = PeerTable::new(node(42, 0), now);
        let initial = service_info(&table.local_node(now), DEFAULT_PORT, 0).unwrap();
        let refresh =
            service_info(&table.local_node(now + HEARTBEAT_INTERVAL), DEFAULT_PORT, 1).unwrap();
        assert_eq!(initial.get_fullname(), refresh.get_fullname());
        assert_eq!(initial.get_property_val_str("uptime"), Some("0"));
        assert_eq!(refresh.get_property_val_str("uptime"), Some("5"));
        assert_eq!(initial.get_property_val_str("heartbeat"), Some("0"));
        assert_eq!(refresh.get_property_val_str("heartbeat"), Some("1"));
    }
}
