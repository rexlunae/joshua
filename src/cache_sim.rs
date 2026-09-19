//! Offline replay of a routing trace ([`crate::route_trace`]) against
//! expert-cache policies.
//!
//! The question a trace answers: with `N` device slots, what hit rate could
//! *any* policy reach on this routing, and how close is the policy the
//! loader runs?  And with a host page cache of `M` experts in front of the
//! disk, how many visits per token still reach the disk — with the two tiers
//! holding different experts (exclusive) or the page cache carrying a copy of
//! the card (inclusive)?
//!
//! Visit semantics mirror the deepseek4 dispatch: per call and per layer the
//! routed experts are deduplicated (one lookup per distinct expert); a decode
//! miss is uploaded (inserted) right away; a prefill only looks the pool up,
//! except that its last row's experts are requested at its end.  The
//! routing-frequency hot set counts one visit per distinct expert per decode
//! call (and the last row per layer of a prefill), is re-selected every
//! [`crate::hot_experts::REFRESH_STEPS`] decode steps, capped at
//! [`crate::residency::HOT_SET_SHARE`] of the slots, protected from eviction
//! and re-acquired on every refresh.
//!
//! Run it on a trace with `cargo run --release --example cache_sim -- trace.csv
//! --slots 2067 --host-slots 7400`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::route_trace::Phase;

/// A `(layer, expert)` key.
pub type Key = (u32, u32);

/// One forward pass of the trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    /// Prefill or decode.
    pub phase: Phase,
    /// Per layer, the routed `(row, expert)` visits in trace order
    /// (duplicates kept; a batched decode has several rows).
    pub layers: Vec<Vec<(u32, u32)>>,
}

/// A parsed routing trace.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Trace {
    /// The calls in order.
    pub calls: Vec<Call>,
}

impl Trace {
    /// Parse the CSV written by [`crate::route_trace::Tracer`].
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut calls: Vec<Call> = Vec::new();
        let mut current: Option<u64> = None;
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("call") || line.starts_with('#') {
                continue;
            }
            let mut f = line.split(',');
            let mut next = |what: &str| {
                f.next()
                    .map(str::trim)
                    .ok_or_else(|| format!("line {}: missing {what}", n + 1))
            };
            let call: u64 = next("call")?
                .parse()
                .map_err(|e| format!("line {}: call: {e}", n + 1))?;
            let phase = match next("phase")? {
                "p" => Phase::Prefill,
                "d" => Phase::Decode,
                other => return Err(format!("line {}: phase `{other}`", n + 1)),
            };
            let row: u32 = next("row")?
                .parse()
                .map_err(|e| format!("line {}: row: {e}", n + 1))?;
            let layer: usize = next("layer")?
                .parse()
                .map_err(|e| format!("line {}: layer: {e}", n + 1))?;
            let expert: u32 = next("expert")?
                .parse()
                .map_err(|e| format!("line {}: expert: {e}", n + 1))?;
            if f.next().is_some() {
                return Err(format!("line {}: too many fields", n + 1));
            }
            if current != Some(call) {
                calls.push(Call {
                    phase,
                    layers: Vec::new(),
                });
                current = Some(call);
            }
            let c = calls.last_mut().expect("pushed above");
            if c.layers.len() <= layer {
                c.layers.resize(layer + 1, Vec::new());
            }
            c.layers[layer].push((row, expert));
        }
        Ok(Self { calls })
    }

    /// Read and parse a trace file.
    pub fn from_path(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&text)
    }

    /// Number of decode calls.
    pub fn decode_steps(&self) -> usize {
        self.calls
            .iter()
            .filter(|c| c.phase == Phase::Decode)
            .count()
    }

    /// Distinct `(layer, expert)` keys visited.
    pub fn distinct_keys(&self) -> usize {
        let mut set = HashSet::new();
        for c in &self.calls {
            for (l, visits) in c.layers.iter().enumerate() {
                for &(_, e) in visits {
                    set.insert((l as u32, e));
                }
            }
        }
        set.len()
    }

    /// The lookups a call performs, per layer: distinct experts in first-seen
    /// order (the dispatch buckets per expert).
    fn lookups(call: &Call) -> Vec<Vec<Key>> {
        call.layers
            .iter()
            .enumerate()
            .map(|(l, visits)| {
                let mut seen = HashSet::new();
                visits
                    .iter()
                    .filter(|(_, e)| seen.insert(*e))
                    .map(|&(_, e)| (l as u32, e))
                    .collect()
            })
            .collect()
    }

    /// What the hot-set counters record for a call: every distinct expert
    /// per layer of a decode step; the last row's experts per layer of a
    /// prefill (which are also the experts the prefill seeds the pool with).
    fn recorded(call: &Call) -> Vec<Vec<Key>> {
        match call.phase {
            Phase::Decode => Self::lookups(call),
            Phase::Prefill => call
                .layers
                .iter()
                .enumerate()
                .map(|(l, visits)| {
                    let last_row = visits.iter().map(|(r, _)| *r).max();
                    let mut last: Vec<Key> = visits
                        .iter()
                        .filter(|(r, _)| Some(*r) == last_row)
                        .map(|&(_, e)| (l as u32, e))
                        .collect();
                    last.sort_unstable();
                    last.dedup();
                    last
                })
                .collect(),
        }
    }
}

/// An LRU tier of `slots` experts with an eviction-protected hot set.
struct Lru {
    slots: usize,
    /// key → recency clock.
    map: HashMap<Key, u64>,
    /// recency clock → key (oldest first).
    order: BTreeMap<u64, Key>,
    hot: HashSet<Key>,
    clock: u64,
    inserts: u64,
    evictions: u64,
    refused: u64,
}

impl Lru {
    fn new(slots: usize) -> Self {
        Self {
            slots,
            map: HashMap::new(),
            order: BTreeMap::new(),
            hot: HashSet::new(),
            clock: 0,
            inserts: 0,
            evictions: 0,
            refused: 0,
        }
    }

    fn contains(&self, k: Key) -> bool {
        self.map.contains_key(&k)
    }

    /// Refresh `k`'s recency if resident; whether it was.
    fn touch(&mut self, k: Key) -> bool {
        let Some(old) = self.map.get(&k).copied() else {
            return false;
        };
        self.order.remove(&old);
        self.clock += 1;
        self.order.insert(self.clock, k);
        self.map.insert(k, self.clock);
        true
    }

    /// Insert `k` (or touch it), evicting the LRU non-hot expert when full.
    /// Refused (not inserted) when every resident expert is hot.
    fn insert(&mut self, k: Key) {
        if self.touch(k) || self.slots == 0 {
            return;
        }
        if self.map.len() >= self.slots {
            let victim = self
                .order
                .iter()
                .find(|(_, key)| !self.hot.contains(key))
                .map(|(c, key)| (*c, *key));
            match victim {
                Some((c, key)) => {
                    self.order.remove(&c);
                    self.map.remove(&key);
                    self.evictions += 1;
                }
                None => {
                    self.refused += 1;
                    return;
                }
            }
        }
        self.clock += 1;
        self.order.insert(self.clock, k);
        self.map.insert(k, self.clock);
        self.inserts += 1;
    }

    fn remove(&mut self, k: Key) {
        if let Some(c) = self.map.remove(&k) {
            self.order.remove(&c);
        }
    }

    fn set_hot(&mut self, hot: &[Key]) {
        self.hot = hot.iter().copied().collect();
    }
}

/// The routing-frequency hot-set policy of [`crate::hot_experts`], replayed.
struct HotSet {
    budget: usize,
    hits: HashMap<Key, u64>,
    last_used: HashMap<Key, u64>,
    step: u64,
}

impl HotSet {
    fn new(budget: usize) -> Self {
        Self {
            budget,
            hits: HashMap::new(),
            last_used: HashMap::new(),
            step: 0,
        }
    }

    fn record(&mut self, keys: &[Key]) {
        for &k in keys {
            *self.hits.entry(k).or_insert(0) += 1;
            self.last_used.insert(k, self.step);
        }
    }

    /// Advance the decode clock; whether a refresh is due.
    fn begin_decode(&mut self) -> bool {
        self.step += 1;
        self.budget > 0 && self.step.is_multiple_of(crate::hot_experts::REFRESH_STEPS)
    }

    /// The hot set in priority order: frequency, then recency.
    fn select(&self) -> Vec<Key> {
        let mut all: Vec<(Key, u64, u64)> = self
            .hits
            .iter()
            .map(|(k, h)| (*k, *h, self.last_used.get(k).copied().unwrap_or(0)))
            .collect();
        all.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)).then(a.0.cmp(&b.0)));
        all.into_iter()
            .take(self.budget)
            .map(|(k, _, _)| k)
            .collect()
    }
}

/// A single-tier policy to replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Plain LRU: decode misses inserted, LRU evicted.
    Lru,
    /// LRU with the routing-frequency hot set protected and re-acquired
    /// every refresh — what the loader runs.  The hot budget is the given
    /// share of the slots (¾ by default).
    LruHot {
        /// Hot-set share of the slots as a fraction (`numerator`, `denominator`).
        share: (usize, usize),
    },
    /// The `slots` most-visited keys of the *whole* trace, fixed in advance
    /// (an oracle bound on any frequency-based static placement).
    StaticTop,
    /// Belady's offline optimum: on a miss, evict the resident key whose
    /// next visit is farthest away.  The ceiling for demand caching at this
    /// slot count.
    Belady,
}

impl Policy {
    /// A short name for tables.
    pub fn name(&self) -> String {
        match self {
            Policy::Lru => "lru".into(),
            Policy::LruHot { share } => format!("lru+hot {}/{}", share.0, share.1),
            Policy::StaticTop => "static top-N".into(),
            Policy::Belady => "belady (optimal)".into(),
        }
    }
}

/// Replay settings.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Device slots.
    pub slots: usize,
    /// Insert every prefill visit too (the loader does not: prefill is
    /// lookup-only, plus the last row).
    pub prefill_inserts: bool,
}

/// The outcome of a single-tier replay.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    /// The policy replayed.
    pub policy: String,
    /// Device slots.
    pub slots: usize,
    /// Decode steps replayed.
    pub decode_steps: u64,
    /// Decode lookups and hits.
    pub decode_visits: u64,
    pub decode_hits: u64,
    /// Prefill lookups and hits.
    pub prefill_visits: u64,
    pub prefill_hits: u64,
    /// Inserts (uploads), evictions, refused inserts.
    pub uploads: u64,
    pub evictions: u64,
    pub refused: u64,
}

impl Report {
    /// Decode hit rate, 0–1.
    pub fn decode_hit_rate(&self) -> f64 {
        if self.decode_visits == 0 {
            0.0
        } else {
            self.decode_hits as f64 / self.decode_visits as f64
        }
    }
}

/// Replay `trace` under `policy`.
pub fn simulate(trace: &Trace, policy: Policy, cfg: &Config) -> Report {
    match policy {
        Policy::Belady => simulate_belady(trace, cfg),
        Policy::StaticTop => simulate_static(trace, cfg),
        Policy::Lru | Policy::LruHot { .. } => simulate_lru(trace, policy, cfg),
    }
}

fn simulate_lru(trace: &Trace, policy: Policy, cfg: &Config) -> Report {
    let mut rep = Report {
        policy: policy.name(),
        slots: cfg.slots,
        ..Default::default()
    };
    let mut tier = Lru::new(cfg.slots);
    let hot_budget = match policy {
        Policy::LruHot { share } => cfg.slots * share.0 / share.1.max(1),
        _ => 0,
    };
    let mut hot = HotSet::new(hot_budget);
    for call in &trace.calls {
        let decode = call.phase == Phase::Decode;
        if decode {
            rep.decode_steps += 1;
            if hot.begin_decode() {
                let set = hot.select();
                tier.set_hot(&set);
                for key in set {
                    tier.insert(key);
                }
            }
        }
        for keys in Trace::lookups(call) {
            for key in keys {
                let hit = tier.touch(key);
                if decode {
                    rep.decode_visits += 1;
                    rep.decode_hits += hit as u64;
                    if !hit {
                        tier.insert(key);
                    }
                } else {
                    rep.prefill_visits += 1;
                    rep.prefill_hits += hit as u64;
                    if !hit && cfg.prefill_inserts {
                        tier.insert(key);
                    }
                }
            }
        }
        for keys in Trace::recorded(call) {
            hot.record(&keys);
            if !decode && !cfg.prefill_inserts {
                // The last prompt row seeds the pool for the decode that follows.
                for key in keys {
                    tier.insert(key);
                }
            }
        }
    }
    rep.uploads = tier.inserts;
    rep.evictions = tier.evictions;
    rep.refused = tier.refused;
    rep
}

fn simulate_static(trace: &Trace, cfg: &Config) -> Report {
    let mut rep = Report {
        policy: Policy::StaticTop.name(),
        slots: cfg.slots,
        ..Default::default()
    };
    let mut count: HashMap<Key, u64> = HashMap::new();
    for call in trace.calls.iter().filter(|c| c.phase == Phase::Decode) {
        for keys in Trace::lookups(call) {
            for key in keys {
                *count.entry(key).or_insert(0) += 1;
            }
        }
    }
    let mut ranked: Vec<(Key, u64)> = count.into_iter().collect();
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let resident: HashSet<Key> = ranked.into_iter().take(cfg.slots).map(|(k, _)| k).collect();
    rep.uploads = resident.len() as u64;
    for call in &trace.calls {
        let decode = call.phase == Phase::Decode;
        if decode {
            rep.decode_steps += 1;
        }
        for keys in Trace::lookups(call) {
            for key in keys {
                let hit = resident.contains(&key) as u64;
                if decode {
                    rep.decode_visits += 1;
                    rep.decode_hits += hit;
                } else {
                    rep.prefill_visits += 1;
                    rep.prefill_hits += hit;
                }
            }
        }
    }
    rep
}

/// One event of the flattened Belady sequence.
#[derive(Clone, Copy)]
enum Event {
    /// A lookup (`decode`), inserting on a miss when `inserts`.
    Visit { decode: bool, inserts: bool },
    /// The prefill's last-row seeding: an insert that is not a lookup.
    Seed,
}

fn simulate_belady(trace: &Trace, cfg: &Config) -> Report {
    let mut rep = Report {
        policy: Policy::Belady.name(),
        slots: cfg.slots,
        ..Default::default()
    };
    // Flatten the lookups; prefill visits count as lookups (and inform the
    // next-use distances) but only decode misses insert, as in the loader —
    // plus the prefill's last-row seed, which the loader uploads.
    let mut seq: Vec<(Key, Event)> = Vec::new();
    for call in &trace.calls {
        let decode = call.phase == Phase::Decode;
        if decode {
            rep.decode_steps += 1;
        }
        for keys in Trace::lookups(call) {
            for key in keys {
                seq.push((
                    key,
                    Event::Visit {
                        decode,
                        inserts: decode || cfg.prefill_inserts,
                    },
                ));
            }
        }
        if !decode && !cfg.prefill_inserts {
            for keys in Trace::recorded(call) {
                for key in keys {
                    seq.push((key, Event::Seed));
                }
            }
        }
    }
    let n = seq.len();
    // Next *visit* of each key after position i (seeds are not uses).
    let mut next_use = vec![u64::MAX; n];
    let mut last_seen: HashMap<Key, usize> = HashMap::new();
    for i in (0..n).rev() {
        if let Some(&j) = last_seen.get(&seq[i].0) {
            next_use[i] = j as u64;
        }
        if matches!(seq[i].1, Event::Visit { .. }) {
            last_seen.insert(seq[i].0, i);
        }
    }
    let mut resident: HashMap<Key, u64> = HashMap::new(); // key → next use
    let mut by_next: BTreeSet<(u64, Key)> = BTreeSet::new();
    for (i, &(key, event)) in seq.iter().enumerate() {
        let hit = resident.contains_key(&key);
        let inserts = match event {
            Event::Visit { decode, inserts } => {
                if decode {
                    rep.decode_visits += 1;
                    rep.decode_hits += hit as u64;
                } else {
                    rep.prefill_visits += 1;
                    rep.prefill_hits += hit as u64;
                }
                inserts
            }
            Event::Seed => true,
        };
        if hit {
            let old = resident[&key];
            by_next.remove(&(old, key));
            resident.insert(key, next_use[i]);
            by_next.insert((next_use[i], key));
        } else if inserts && cfg.slots > 0 {
            if next_use[i] == u64::MAX {
                // Never visited again: inserting it can only cost a slot.
                continue;
            }
            if resident.len() >= cfg.slots {
                // Evict the resident key used farthest in the future.
                let victim = by_next.iter().next_back().copied().expect("non-empty");
                by_next.remove(&victim);
                resident.remove(&victim.1);
                rep.evictions += 1;
            }
            resident.insert(key, next_use[i]);
            by_next.insert((next_use[i], key));
            rep.uploads += 1;
        }
    }
    rep
}

/// Settings for the two-tier (device pool + host page cache) replay.
#[derive(Debug, Clone, Copy)]
pub struct TwoTierConfig {
    /// Device slots (the VRAM expert cache).
    pub device_slots: usize,
    /// Host page-cache capacity in experts (RAM left for experts ÷ expert bytes).
    pub host_slots: usize,
    /// Hot-set share of the device slots.
    pub hot_share: (usize, usize),
    /// Exclusive tiers: the host tier is not advised for device-resident
    /// experts and an uploaded expert's host pages are dropped (what the
    /// loader does now).  Inclusive: the hot-set refresh and the speculative
    /// prefetch advise every expert's host pages, resident on the device or
    /// not, and uploads leave their pages in place (the previous behaviour).
    pub exclusive: bool,
}

/// The outcome of a two-tier replay.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TwoTierReport {
    /// Decode steps replayed.
    pub decode_steps: u64,
    /// Decode lookups: served by the device, by the host page cache, or by
    /// the disk.
    pub decode_visits: u64,
    pub decode_device_hits: u64,
    pub decode_host_hits: u64,
    pub decode_disk: u64,
    /// The same for prefill lookups.
    pub prefill_visits: u64,
    pub prefill_device_hits: u64,
    pub prefill_host_hits: u64,
    pub prefill_disk: u64,
    /// Uploads and device evictions.
    pub uploads: u64,
    pub device_evictions: u64,
    /// Disk reads made by uploads (an upload's pages were not in RAM).
    pub upload_disk: u64,
    /// Disk reads made by host prefetches (the hot-set refresh and the
    /// speculative next-step advice pulling in a cold expert).
    pub prefetch_disk: u64,
}

impl TwoTierReport {
    /// Disk reads per decode step: demand misses plus uploads and prefetches
    /// whose pages were cold.  Prefill reads are excluded (they are
    /// per-prompt, not per-token).
    pub fn disk_reads_per_decode_step(&self) -> f64 {
        if self.decode_steps == 0 {
            0.0
        } else {
            (self.decode_disk + self.upload_disk + self.prefetch_disk) as f64
                / self.decode_steps as f64
        }
    }
}

/// Replay `trace` through a device pool (LRU + protected hot set, the
/// loader's policy) backed by a host page cache (LRU) backed by the disk.
pub fn simulate_two_tier(trace: &Trace, cfg: &TwoTierConfig) -> TwoTierReport {
    let mut rep = TwoTierReport::default();
    let mut device = Lru::new(cfg.device_slots);
    let mut host = Lru::new(cfg.host_slots);
    let hot_budget = cfg.device_slots * cfg.hot_share.0 / cfg.hot_share.1.max(1);
    let mut hot = HotSet::new(hot_budget);
    let mut previous_routed: Vec<Key> = Vec::new();

    // A host prefetch (`MADV_WILLNEED`): a cold expert is read from the disk.
    let prefetch = |key: Key, host: &mut Lru, rep: &mut TwoTierReport| {
        if !host.touch(key) {
            rep.prefetch_disk += 1;
            host.insert(key);
        }
    };
    // Upload `key`: reads its host pages (disk if cold), inserts it on the
    // device; with exclusive tiers the host pages then go.
    let upload = |key: Key, device: &mut Lru, host: &mut Lru, rep: &mut TwoTierReport| {
        if device.contains(key) {
            device.touch(key);
            return;
        }
        if !host.touch(key) {
            rep.upload_disk += 1;
            host.insert(key);
        }
        device.insert(key);
        if device.contains(key) {
            rep.uploads += 1;
            if cfg.exclusive {
                host.remove(key);
            }
        }
    };

    for call in &trace.calls {
        let decode = call.phase == Phase::Decode;
        if decode {
            rep.decode_steps += 1;
            if hot.begin_decode() {
                let set = hot.select();
                device.set_hot(&set);
                for key in set {
                    // The refresh advises the host pages (unless the expert is
                    // on the device and the tiers are exclusive) and asks
                    // for an upload.
                    if !(cfg.exclusive && device.contains(key)) {
                        prefetch(key, &mut host, &mut rep);
                    }
                    upload(key, &mut device, &mut host, &mut rep);
                }
            }
            // Speculative prefetch of the previous step's routing.
            for &key in &previous_routed {
                if !(cfg.exclusive && device.contains(key)) {
                    prefetch(key, &mut host, &mut rep);
                }
            }
        }
        let mut routed_now: Vec<Key> = Vec::new();
        for keys in Trace::lookups(call) {
            for key in keys {
                let on_device = device.touch(key);
                let (visits, dev_hits, host_hits, disk) = if decode {
                    (
                        &mut rep.decode_visits,
                        &mut rep.decode_device_hits,
                        &mut rep.decode_host_hits,
                        &mut rep.decode_disk,
                    )
                } else {
                    (
                        &mut rep.prefill_visits,
                        &mut rep.prefill_device_hits,
                        &mut rep.prefill_host_hits,
                        &mut rep.prefill_disk,
                    )
                };
                *visits += 1;
                if on_device {
                    *dev_hits += 1;
                } else {
                    // Host run: the pages come from RAM or from the disk and
                    // stay in RAM afterwards.
                    if host.touch(key) {
                        *host_hits += 1;
                    } else {
                        *disk += 1;
                        host.insert(key);
                    }
                    if decode {
                        upload(key, &mut device, &mut host, &mut rep);
                    }
                }
                if decode {
                    routed_now.push(key);
                }
            }
        }
        for keys in Trace::recorded(call) {
            hot.record(&keys);
            if !decode {
                for key in keys {
                    upload(key, &mut device, &mut host, &mut rep);
                }
            }
        }
        if decode {
            previous_routed = routed_now;
        }
    }
    rep.device_evictions = device.evictions;
    rep
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-row decode call.
    fn decode(layers: &[&[u32]]) -> Call {
        Call {
            phase: Phase::Decode,
            layers: layers
                .iter()
                .map(|l| l.iter().map(|&e| (0, e)).collect())
                .collect(),
        }
    }

    /// A prefill call given as per-layer rows of experts.
    fn prefill(layers: &[&[&[u32]]]) -> Call {
        Call {
            phase: Phase::Prefill,
            layers: layers
                .iter()
                .map(|rows| {
                    rows.iter()
                        .enumerate()
                        .flat_map(|(r, es)| es.iter().map(move |&e| (r as u32, e)))
                        .collect()
                })
                .collect(),
        }
    }

    #[test]
    fn parses_the_tracer_format() {
        let text =
            "call,phase,row,layer,expert\n0,p,0,0,3\n0,p,1,0,3\n0,p,0,1,7\n1,d,0,0,1\n1,d,0,1,2\n";
        let t = Trace::parse(text).unwrap();
        assert_eq!(
            t,
            Trace {
                calls: vec![prefill(&[&[&[3], &[3]], &[&[7]]]), decode(&[&[1], &[2]])]
            }
        );
        assert_eq!(t.decode_steps(), 1);
        assert_eq!(t.distinct_keys(), 4);
        assert!(Trace::parse("0,x,0,0,1").is_err());
        assert!(Trace::parse("0,d,0,0,1,9").is_err(), "too many fields");
        assert!(Trace::parse("0,d,0,0").is_err(), "too few fields");
    }

    /// A cyclic sweep one expert wider than the cache: LRU never hits,
    /// Belady keeps all but one of the keys and hits on them.
    #[test]
    fn belady_beats_lru_on_a_cyclic_sweep() {
        let calls: Vec<Call> = (0..40).map(|i| decode(&[&[(i % 4) as u32]])).collect();
        let trace = Trace { calls };
        let cfg = Config {
            slots: 3,
            prefill_inserts: false,
        };
        let lru = simulate(&trace, Policy::Lru, &cfg);
        assert_eq!(lru.decode_visits, 40);
        assert_eq!(lru.decode_hits, 0, "{lru:?}");
        let opt = simulate(&trace, Policy::Belady, &cfg);
        assert_eq!(opt.decode_visits, 40);
        // After the first sweep, 2 of every 4 visits hit (3 slots, 4 keys,
        // one eviction per sweep): 36 later visits → at least 18 hits.
        assert!(opt.decode_hits >= 18, "{opt:?}");
        assert!(opt.decode_hits > lru.decode_hits);
        let stat = simulate(&trace, Policy::StaticTop, &cfg);
        assert_eq!(
            stat.decode_hits, 30,
            "top-3 of 4 equally used keys: 3/4 of visits"
        );
    }

    /// The protected hot set survives a sweep that would evict it under
    /// plain LRU: a frequent expert keeps hitting after the refresh.
    #[test]
    fn hot_set_protects_the_frequent_expert() {
        // Expert 0 every step; a sweep of fresh experts fills the rest.
        let mut calls = Vec::new();
        for i in 0..200u32 {
            calls.push(decode(&[&[0, 100 + i, 100 + i + 1]]));
        }
        let trace = Trace { calls };
        let cfg = Config {
            slots: 2,
            prefill_inserts: false,
        };
        let plain = simulate(&trace, Policy::Lru, &cfg);
        let hot = simulate(&trace, Policy::LruHot { share: (1, 2) }, &cfg);
        // With one protected slot expert 0 always hits after the first
        // refresh (step 64); LRU evicts it every step.
        assert!(
            hot.decode_hits > plain.decode_hits,
            "hot {hot:?} vs lru {plain:?}"
        );
        assert!(hot.decode_hits >= 200 - 64, "{hot:?}");
    }

    /// Prefill is lookup-only except for its last row, which seeds the pool
    /// under every policy — Belady included, so its ceiling starts the
    /// decode from the loader's cache state.
    #[test]
    fn prefill_seeds_only_the_last_row() {
        let trace = Trace {
            calls: vec![
                prefill(&[&[&[1, 2], &[3, 4], &[5, 6]]]), // last row (5, 6)
                decode(&[&[5, 6]]),
                decode(&[&[1, 2]]),
            ],
        };
        let cfg = Config {
            slots: 4,
            prefill_inserts: false,
        };
        for policy in [Policy::Lru, Policy::Belady] {
            let r = simulate(&trace, policy, &cfg);
            assert_eq!(r.prefill_visits, 6, "{r:?}");
            assert_eq!(r.prefill_hits, 0, "{r:?}");
            assert_eq!(r.decode_visits, 4, "{r:?}");
            assert_eq!(
                r.decode_hits, 2,
                "the seeded (5, 6) hit, (1, 2) miss: {r:?}"
            );
        }
        let all = simulate(
            &trace,
            Policy::Lru,
            &Config {
                slots: 6,
                prefill_inserts: true,
            },
        );
        assert_eq!(all.decode_hits, 4);
    }

    /// A batched decode (several rows per call) is one lookup per distinct
    /// expert, and a prefill's seed is its last *row*, not the last
    /// `batch × k` entries.
    #[test]
    fn batched_rows_do_not_widen_the_prefill_seed() {
        let batched = Call {
            phase: Phase::Decode,
            layers: vec![vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (2, 5)]],
        };
        let trace = Trace {
            calls: vec![
                batched,
                prefill(&[&[&[7, 8], &[9, 10], &[11, 12]]]),
                decode(&[&[9, 10, 11, 12]]),
            ],
        };
        let cfg = Config {
            slots: 8,
            prefill_inserts: false,
        };
        let r = simulate(&trace, Policy::Lru, &cfg);
        // The batched call: 5 distinct experts, all misses.
        // The last decode: (11, 12) seeded, (9, 10) not.
        assert_eq!(r.decode_visits, 5 + 4, "{r:?}");
        assert_eq!(r.decode_hits, 2, "{r:?}");
    }

    /// Exclusive tiers: once the hot set pins expert 2 on the one device
    /// slot, the two host slots hold experts 0 and 1 and no visit reaches
    /// the disk.  Inclusive tiers advise expert 2's host pages every step
    /// too (the speculative prefetch), which takes a host slot from 0 or 1
    /// and sends one of them to the disk on every step — and that prefetch
    /// itself is a disk read that the report counts.
    #[test]
    fn exclusive_tiers_stop_the_duplicate_from_evicting_the_host_experts() {
        // Expert 2 is visited one step more than 0 and 1, so the refresh at
        // step 64 makes it the hot one; until then the one-slot device
        // thrashes under either policy.
        let mut calls = vec![decode(&[&[2]])];
        calls.extend(std::iter::repeat_n(decode(&[&[0, 1, 2]]), 300));
        let trace = Trace { calls };
        let base = TwoTierConfig {
            device_slots: 1,
            host_slots: 2,
            hot_share: (1, 1),
            exclusive: true,
        };
        let ex = simulate_two_tier(&trace, &base);
        let inc = simulate_two_tier(
            &trace,
            &TwoTierConfig {
                exclusive: false,
                ..base
            },
        );
        assert_eq!(ex.decode_visits, 901);
        assert_eq!(inc.decode_visits, 901);
        // Exclusive: nothing from the disk after the refresh (≤ 3 reads per
        // step for the 64 thrashing steps before it, uploads and
        // prefetches included).
        assert!(
            ex.decode_disk + ex.upload_disk + ex.prefetch_disk <= 3 * 64 + 3,
            "{ex:?}"
        );
        assert!(ex.decode_device_hits >= 300 - 64, "{ex:?}");
        // Inclusive: two demand disk reads per step for the ~236 steps
        // after it, plus the cold prefetch of the device-resident expert.
        assert!(inc.decode_disk >= 2 * 230, "{inc:?}");
        assert!(inc.prefetch_disk >= 230, "{inc:?}");
        assert!(ex.disk_reads_per_decode_step() < inc.disk_reads_per_decode_step());
    }
}
