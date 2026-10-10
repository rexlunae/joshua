//! Deterministic, static input-column planning from coordinator-validated measurements.
//!
//! This is a memory-bounded heuristic, not a latency optimizer or live rebalancer.
//! Discovery advertisements are not validated capacity measurements. The caller must
//! supply trustworthy capacities and reserve replicated tensors, KV caches, collective
//! buffers and other runtime memory on **every** node before planning.

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeCapacity {
    pub id: String,
    /// Total bytes available to this model, including `reserved_bytes`.
    pub available_bytes: u64,
    /// Per-node replicated-model and runtime memory, counted once across all tensors.
    pub reserved_bytes: u64,
    /// Positive relative compute throughput per unit of usable memory.
    pub compute_weight: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorWorkload {
    pub id: String,
    pub input_blocks: u64,
    /// Quantization block alignment in input columns (one for unquantized tensors).
    pub columns_per_block: u64,
    /// Storage bytes for one input block across ALL output rows, including block metadata.
    pub bytes_per_input_block: u64,
}

impl TensorWorkload {
    pub fn from_layout(
        id: impl Into<String>,
        input_columns: u64,
        output_rows: u64,
        columns_per_block: u64,
        bytes_per_block_per_row: u64,
    ) -> Result<Self> {
        ensure!(columns_per_block > 0, "block alignment must be positive");
        ensure!(
            input_columns > 0 && input_columns.is_multiple_of(columns_per_block),
            "input columns must contain whole, nonempty blocks"
        );
        ensure!(
            output_rows > 0 && bytes_per_block_per_row > 0,
            "output rows and block storage must be positive"
        );
        let tensor = Self {
            id: id.into(),
            input_blocks: input_columns / columns_per_block,
            columns_per_block,
            bytes_per_input_block: output_rows
                .checked_mul(bytes_per_block_per_row)
                .context("tensor block byte count overflow")?,
        };
        tensor.validate()?;
        Ok(tensor)
    }

    fn validate(&self) -> Result<()> {
        ensure!(!self.id.is_empty(), "tensor ID must not be empty");
        ensure!(
            self.input_blocks > 0 && self.columns_per_block > 0 && self.bytes_per_input_block > 0,
            "tensor dimensions and storage must be positive"
        );
        self.input_blocks
            .checked_mul(self.columns_per_block)
            .context("tensor column count overflow")?;
        self.input_blocks
            .checked_mul(self.bytes_per_input_block)
            .context("tensor byte count overflow")?;
        Ok(())
    }
}

/// A known, measured undirected link. Unknown links receive no invented measurements.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkObservation {
    pub from: String,
    pub to: String,
    pub latency_seconds: f64,
    pub bandwidth_bytes_per_second: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankShard {
    pub rank: usize,
    pub node_id: String,
    pub input_blocks: Range<u64>,
    pub input_columns: Range<u64>,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorPartition {
    pub tensor_id: String,
    pub shards: Vec<RankShard>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeMemory {
    pub rank: usize,
    pub node_id: String,
    pub reserved_bytes: u64,
    pub tensor_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionPlan {
    /// Sorted by tensor ID; each tensor's shards are sorted by rank.
    pub tensors: Vec<TensorPartition>,
    /// Ranks are assigned in stable node-ID order.
    pub nodes: Vec<NodeMemory>,
}

/// Plan all tensors under a single per-node memory budget.
///
/// Every rank receives at least one quantization block of every tensor. The caller
/// must choose a smaller participating cluster if that is impossible. Remaining
/// blocks use capped, integer weighted water filling with score proportional to
/// usable RAM * compute weight / measured network penalty. The network penalty is
/// `1 + worst_latency / 1ms + 1GiB/s / worst_bandwidth` over known incident links.
/// These reference scales are heuristic, not a promised P99 or wall-clock target.
///
/// Minimum shards for all tensors are reserved up front; larger-byte blocks are
/// allocated first to reduce fragmentation. Integer packing can still reject a
/// feasible arrangement; this is not an exact bin-packing solver.
pub fn plan_partition(
    nodes: &[NodeCapacity],
    tensors: &[TensorWorkload],
    links: &[LinkObservation],
) -> Result<PartitionPlan> {
    ensure!(
        !nodes.is_empty(),
        "at least one participating rank is required"
    );
    ensure!(!tensors.is_empty(), "at least one tensor is required");
    let mut nodes: Vec<_> = nodes.iter().collect();
    nodes.sort_by(|a, b| a.id.cmp(&b.id));
    let mut ids = BTreeMap::new();
    let mut budget = Vec::with_capacity(nodes.len());
    for (rank, node) in nodes.iter().enumerate() {
        ensure!(!node.id.is_empty(), "node ID must not be empty");
        ensure!(
            ids.insert(node.id.as_str(), rank).is_none(),
            "duplicate node ID"
        );
        ensure!(
            node.compute_weight.is_finite() && node.compute_weight > 0.0,
            "compute weights must be finite and positive"
        );
        budget.push(
            node.available_bytes
                .checked_sub(node.reserved_bytes)
                .context("node reserve exceeds available memory")?,
        );
    }
    let scores = speed_scores(&nodes, &budget, &ids, links)?;
    let mut tensors: Vec<_> = tensors.iter().collect();
    tensors.sort_by(|a, b| {
        b.bytes_per_input_block
            .cmp(&a.bytes_per_input_block)
            .then_with(|| a.id.cmp(&b.id))
    });
    let mut tensor_ids = BTreeSet::new();
    let mut minimum = 0u64;
    let mut required = 0u64;
    let ranks = u64::try_from(nodes.len()).context("rank count overflow")?;
    for tensor in &tensors {
        tensor.validate()?;
        ensure!(tensor_ids.insert(&tensor.id), "duplicate tensor ID");
        ensure!(
            tensor.input_blocks >= ranks,
            "tensor {} has fewer blocks than participating ranks",
            tensor.id
        );
        minimum = minimum
            .checked_add(tensor.bytes_per_input_block)
            .context("minimum shard storage overflow")?;
        required = required
            .checked_add(
                tensor
                    .input_blocks
                    .checked_mul(tensor.bytes_per_input_block)
                    .context("tensor storage overflow")?,
            )
            .context("global model storage overflow")?;
    }
    let aggregate = budget.iter().try_fold(0u64, |sum, &value| {
        sum.checked_add(value)
            .context("aggregate capacity overflow")
    })?;
    ensure!(required <= aggregate, "insufficient aggregate model memory");
    for bytes in &mut budget {
        *bytes = bytes
            .checked_sub(minimum)
            .context("node cannot hold one block of every tensor plus its reserve")?;
    }
    let mut planned = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let caps: Vec<_> = budget
            .iter()
            .map(|bytes| bytes / tensor.bytes_per_input_block)
            .collect();
        let extras = water_fill(tensor.input_blocks - ranks, &caps, &scores)?;
        let mut start = 0u64;
        let mut shards = Vec::with_capacity(nodes.len());
        for (rank, node) in nodes.iter().enumerate() {
            let count = extras[rank] + 1;
            let end = start.checked_add(count).context("block range overflow")?;
            let bytes = count
                .checked_mul(tensor.bytes_per_input_block)
                .context("shard storage overflow")?;
            budget[rank] = budget[rank]
                .checked_sub(extras[rank] * tensor.bytes_per_input_block)
                .context("shard exceeds node memory")?;
            shards.push(RankShard {
                rank,
                node_id: node.id.clone(),
                input_blocks: start..end,
                input_columns: start * tensor.columns_per_block..end * tensor.columns_per_block,
                bytes,
            });
            start = end;
        }
        ensure!(start == tensor.input_blocks, "incomplete block assignment");
        planned.push(TensorPartition {
            tensor_id: tensor.id.clone(),
            shards,
        });
    }
    planned.sort_by(|a, b| a.tensor_id.cmp(&b.tensor_id));
    let memory = nodes
        .iter()
        .enumerate()
        .map(|(rank, node)| {
            let total_bytes = node.available_bytes - budget[rank];
            NodeMemory {
                rank,
                node_id: node.id.clone(),
                reserved_bytes: node.reserved_bytes,
                tensor_bytes: total_bytes - node.reserved_bytes,
                total_bytes,
            }
        })
        .collect();
    Ok(PartitionPlan {
        tensors: planned,
        nodes: memory,
    })
}

fn speed_scores(
    nodes: &[&NodeCapacity],
    budget: &[u64],
    ids: &BTreeMap<&str, usize>,
    links: &[LinkObservation],
) -> Result<Vec<f64>> {
    let mut latency = vec![0.0f64; nodes.len()];
    let mut bandwidth = vec![f64::INFINITY; nodes.len()];
    let mut seen = BTreeSet::new();
    for link in links {
        let from = *ids
            .get(link.from.as_str())
            .context("unknown link endpoint")?;
        let to = *ids.get(link.to.as_str()).context("unknown link endpoint")?;
        ensure!(from != to, "self links are not supported");
        ensure!(
            seen.insert((from.min(to), from.max(to))),
            "duplicate link observation"
        );
        ensure!(
            link.latency_seconds.is_finite() && link.latency_seconds >= 0.0,
            "link latency must be finite and nonnegative"
        );
        ensure!(
            link.bandwidth_bytes_per_second.is_finite() && link.bandwidth_bytes_per_second > 0.0,
            "link bandwidth must be finite and positive"
        );
        for rank in [from, to] {
            latency[rank] = latency[rank].max(link.latency_seconds);
            bandwidth[rank] = bandwidth[rank].min(link.bandwidth_bytes_per_second);
        }
    }
    // Log space avoids overflow for valid but extremely disparate measurements.
    let logs: Vec<_> = nodes
        .iter()
        .enumerate()
        .map(|(rank, node)| {
            let terms = [
                0.0,
                latency[rank].ln() - 0.001f64.ln(),
                1_073_741_824f64.ln() - bandwidth[rank].ln(),
            ];
            let largest = terms.into_iter().fold(f64::NEG_INFINITY, f64::max);
            let penalty = largest + terms.iter().map(|x| (x - largest).exp()).sum::<f64>().ln();
            node.compute_weight.ln() + (budget[rank].max(1) as f64).ln() - penalty
        })
        .collect();
    let largest = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(logs
        .iter()
        .map(|score| (score - largest).exp().max(f64::MIN_POSITIVE))
        .collect())
}

fn water_fill(mut remaining: u64, caps: &[u64], scores: &[f64]) -> Result<Vec<u64>> {
    let capacity = caps.iter().try_fold(0u64, |sum, &cap| {
        sum.checked_add(cap).context("block capacity overflow")
    })?;
    ensure!(
        remaining <= capacity,
        "insufficient whole-block memory after reserving all tensor minima"
    );
    let mut assigned = vec![0u64; caps.len()];
    while remaining > 0 {
        let active: Vec<_> = (0..caps.len())
            .filter(|&rank| assigned[rank] < caps[rank])
            .collect();
        ensure!(!active.is_empty(), "no capacity for remaining blocks");
        // Renormalize after caps remove high-weight nodes, preventing underflow.
        let largest = active.iter().map(|&i| scores[i]).fold(0.0f64, f64::max);
        let total: f64 = active.iter().map(|&i| scores[i] / largest).sum();
        let shares: Vec<_> = active
            .iter()
            .map(|&rank| (rank, remaining as f64 * (scores[rank] / largest / total)))
            .collect();
        let saturated: Vec<_> = shares
            .iter()
            .filter(|&&(rank, share)| share >= (caps[rank] - assigned[rank]) as f64)
            .map(|&(rank, _)| rank)
            .collect();
        if !saturated.is_empty() {
            for rank in saturated {
                let count = (caps[rank] - assigned[rank]).min(remaining);
                assigned[rank] += count;
                remaining -= count;
            }
            continue;
        }
        let before = remaining;
        for &(rank, share) in &shares {
            let count = (share.floor() as u64)
                .min(caps[rank] - assigned[rank])
                .min(remaining);
            assigned[rank] += count;
            remaining -= count;
        }
        let mut fractions = shares;
        fractions.sort_by(|(a, x), (b, y)| y.fract().total_cmp(&x.fract()).then_with(|| a.cmp(b)));
        for (rank, _) in fractions {
            if remaining > 0 && assigned[rank] < caps[rank] {
                assigned[rank] += 1;
                remaining -= 1;
            }
        }
        if remaining == before {
            bail!("unable to assign remaining blocks");
        }
    }
    Ok(assigned)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImbalanceRecommendation {
    KeepStaticPlan,
    /// Drain in-flight work and obtain a new cluster-wide plan before changing shards.
    DrainAndReplan,
}

/// Compare a common measurement window across ranks. Never mutates a live plan.
/// For sustained-imbalance detection with a proposed plan see [`RepartitionMonitor`].
/// Zero, nonfinite, negative, and empty measurements are rejected.
pub fn monitor_imbalance(durations_seconds: &[f64]) -> Result<ImbalanceRecommendation> {
    ensure!(!durations_seconds.is_empty(), "no rank measurements");
    ensure!(
        durations_seconds.iter().all(|x| x.is_finite() && *x > 0.0),
        "rank durations must be finite and positive"
    );
    let min = durations_seconds
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max = durations_seconds.iter().copied().fold(0.0, f64::max);
    Ok(if max / 2.0 > min {
        ImbalanceRecommendation::DrainAndReplan
    } else {
        ImbalanceRecommendation::KeepStaticPlan
    })
}

/// Constant link values for node pairs that have no measurement or override.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkDefaults {
    pub latency_seconds: f64,
    pub bandwidth_bytes_per_second: f64,
}

/// Merge planner link inputs with precedence `overrides` > `measured` > `fallback`.
///
/// `measured` typically comes from `probe::measure_links`, possibly from several
/// nodes; when both directions of a pair were measured, the conservative value
/// (larger latency, smaller bandwidth) is kept. `overrides` are operator-supplied
/// constants that replace any measurement of the same unordered pair; duplicates
/// there are rejected. `fallback`, when given, fills every remaining pair of
/// `nodes`; without it unknown pairs stay unknown (the planner adds no penalty).
/// Output is sorted by endpoint IDs; values are validated by [`plan_partition`].
pub fn resolve_links(
    nodes: &[NodeCapacity],
    measured: &[LinkObservation],
    overrides: &[LinkObservation],
    fallback: Option<LinkDefaults>,
) -> Result<Vec<LinkObservation>> {
    fn key(link: &LinkObservation) -> (String, String) {
        let (a, b) = (link.from.clone(), link.to.clone());
        if a <= b {
            (a, b)
        } else {
            (b, a)
        }
    }
    let mut links: BTreeMap<(String, String), LinkObservation> = BTreeMap::new();
    for link in measured {
        let pair = key(link);
        let merged = match links.remove(&pair) {
            Some(known) => LinkObservation {
                latency_seconds: known.latency_seconds.max(link.latency_seconds),
                bandwidth_bytes_per_second: known
                    .bandwidth_bytes_per_second
                    .min(link.bandwidth_bytes_per_second),
                ..known
            },
            None => link.clone(),
        };
        links.insert(pair, merged);
    }
    let mut overridden = BTreeSet::new();
    for link in overrides {
        let pair = key(link);
        ensure!(
            overridden.insert(pair.clone()),
            "duplicate link override {} <-> {}",
            pair.0,
            pair.1
        );
        links.insert(pair, link.clone());
    }
    if let Some(defaults) = fallback {
        ensure!(
            defaults.latency_seconds.is_finite()
                && defaults.latency_seconds >= 0.0
                && defaults.bandwidth_bytes_per_second.is_finite()
                && defaults.bandwidth_bytes_per_second > 0.0,
            "fallback link values must be finite, nonnegative latency and positive bandwidth"
        );
        let mut ids: Vec<_> = nodes.iter().map(|node| node.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        for (index, a) in ids.iter().enumerate() {
            for b in &ids[index + 1..] {
                links
                    .entry((a.to_string(), b.to_string()))
                    .or_insert_with(|| LinkObservation {
                        from: a.to_string(),
                        to: b.to_string(),
                        latency_seconds: defaults.latency_seconds,
                        bandwidth_bytes_per_second: defaults.bandwidth_bytes_per_second,
                    });
            }
        }
    }
    Ok(links.into_values().collect())
}

/// One rank's timing over a common measurement window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankTiming {
    pub node_id: String,
    /// Time spent computing this rank's shards.
    pub busy_seconds: f64,
    /// Time spent blocked in collectives waiting for slower peers.
    pub wait_seconds: f64,
}

/// When a [`RepartitionMonitor`] proposes a new plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepartitionPolicy {
    /// Load imbalance `max(busy) / mean(busy) - 1` tolerated per window.
    pub max_imbalance: f64,
    /// Consecutive windows above `max_imbalance` before proposing a plan.
    pub sustained_windows: usize,
    /// Largest per-replan change of a node's compute weight (factor, >= 1).
    pub max_weight_step: f64,
}

impl Default for RepartitionPolicy {
    fn default() -> Self {
        Self {
            max_imbalance: 0.10,
            sustained_windows: 3,
            max_weight_step: 4.0,
        }
    }
}

/// Load imbalance `max(busy) / mean(busy) - 1` of one window (zero is perfect).
///
/// Ranks that only record collective waits can report `busy = window - wait`.
/// Wait times are validated but otherwise informational: in a barrier
/// collective the slowest rank waits least.
pub fn load_imbalance(timings: &[RankTiming]) -> Result<f64> {
    validate_timings(timings)?;
    let max = timings
        .iter()
        .map(|t| t.busy_seconds)
        .fold(0.0f64, f64::max);
    let mean = timings.iter().map(|t| t.busy_seconds).sum::<f64>() / timings.len() as f64;
    Ok(max / mean - 1.0)
}

fn validate_timings(timings: &[RankTiming]) -> Result<()> {
    ensure!(!timings.is_empty(), "no rank timings");
    let mut ids = BTreeSet::new();
    for timing in timings {
        ensure!(
            ids.insert(timing.node_id.as_str()),
            "duplicate rank timing for {}",
            timing.node_id
        );
        ensure!(
            timing.busy_seconds.is_finite() && timing.busy_seconds > 0.0,
            "busy time must be finite and positive"
        );
        ensure!(
            timing.wait_seconds.is_finite() && timing.wait_seconds >= 0.0,
            "wait time must be finite and nonnegative"
        );
    }
    Ok(())
}

/// Correct compute weights from measured busy times.
///
/// The planner assigns blocks in proportion to its node scores, so a rank's
/// measured rate is `blocks / busy` and the score that balances busy time is
/// `old_score / busy`. Only `compute_weight` changes (memory budgets and link
/// penalties are unchanged), scaled by `mean(busy) / busy` and clamped to
/// `[1 / max_step, max_step]` to damp noisy windows.
pub fn rebalance_weights(
    nodes: &[NodeCapacity],
    timings: &[RankTiming],
    max_step: f64,
) -> Result<Vec<NodeCapacity>> {
    validate_timings(timings)?;
    ensure!(
        max_step.is_finite() && max_step >= 1.0,
        "weight step must be finite and at least one"
    );
    ensure!(
        timings.len() == nodes.len(),
        "need exactly one timing per planned node"
    );
    let mean = timings.iter().map(|t| t.busy_seconds).sum::<f64>() / timings.len() as f64;
    nodes
        .iter()
        .map(|node| {
            let timing = timings
                .iter()
                .find(|t| t.node_id == node.id)
                .with_context(|| format!("no timing for node {}", node.id))?;
            let factor = (mean / timing.busy_seconds).clamp(1.0 / max_step, max_step);
            let compute_weight = node.compute_weight * factor;
            ensure!(
                compute_weight.is_finite() && compute_weight > 0.0,
                "rebalanced compute weight is not finite and positive"
            );
            Ok(NodeCapacity {
                compute_weight,
                ..node.clone()
            })
        })
        .collect()
}

/// A proposed replacement plan. Applying it requires draining in-flight work,
/// resharding every rank and resuming; that protocol is not implemented here.
#[derive(Clone, Debug, PartialEq)]
pub struct Repartition {
    pub nodes: Vec<NodeCapacity>,
    pub plan: PartitionPlan,
    /// Imbalance of the accumulated windows that triggered the proposal.
    pub observed_imbalance: f64,
}

/// Detects sustained imbalance from per-window rank timings and proposes a plan.
///
/// Advisory only: it never mutates a live plan. Feed windows in order for the
/// participant set of the current plan. After a proposal (or any balanced
/// window) the streak restarts, so one slow window never triggers a replan.
#[derive(Clone, Debug)]
pub struct RepartitionMonitor {
    policy: RepartitionPolicy,
    streak: usize,
    accumulated: BTreeMap<String, (f64, f64)>,
}

impl RepartitionMonitor {
    pub fn new(policy: RepartitionPolicy) -> Result<Self> {
        ensure!(
            policy.max_imbalance.is_finite() && policy.max_imbalance >= 0.0,
            "imbalance threshold must be finite and nonnegative"
        );
        ensure!(
            policy.sustained_windows > 0,
            "sustained window count must be positive"
        );
        ensure!(
            policy.max_weight_step.is_finite() && policy.max_weight_step >= 1.0,
            "weight step must be finite and at least one"
        );
        Ok(Self {
            policy,
            streak: 0,
            accumulated: BTreeMap::new(),
        })
    }

    /// Consecutive imbalanced windows seen so far.
    pub fn streak(&self) -> usize {
        self.streak
    }

    /// Feed one window. Returns a new plan for `nodes`/`tensors`/`links` once the
    /// imbalance exceeded the threshold for `sustained_windows` windows in a row.
    /// The correction uses busy times summed over that streak.
    pub fn observe(
        &mut self,
        timings: &[RankTiming],
        nodes: &[NodeCapacity],
        tensors: &[TensorWorkload],
        links: &[LinkObservation],
    ) -> Result<Option<Repartition>> {
        let imbalance = load_imbalance(timings)?;
        if imbalance <= self.policy.max_imbalance {
            self.streak = 0;
            self.accumulated.clear();
            return Ok(None);
        }
        if self.accumulated.len() != timings.len()
            || timings
                .iter()
                .any(|t| !self.accumulated.contains_key(&t.node_id))
        {
            // First window, or the participant set changed: restart the evidence.
            self.streak = 0;
            self.accumulated.clear();
        }
        for timing in timings {
            let entry = self
                .accumulated
                .entry(timing.node_id.clone())
                .or_insert((0.0, 0.0));
            entry.0 += timing.busy_seconds;
            entry.1 += timing.wait_seconds;
        }
        self.streak += 1;
        if self.streak < self.policy.sustained_windows {
            return Ok(None);
        }
        let summed: Vec<_> = self
            .accumulated
            .iter()
            .map(|(id, &(busy, wait))| RankTiming {
                node_id: id.clone(),
                busy_seconds: busy,
                wait_seconds: wait,
            })
            .collect();
        self.streak = 0;
        self.accumulated.clear();
        let observed_imbalance = load_imbalance(&summed)?;
        let nodes = rebalance_weights(nodes, &summed, self.policy.max_weight_step)?;
        let plan = plan_partition(&nodes, tensors, links)?;
        Ok(Some(Repartition {
            nodes,
            plan,
            observed_imbalance,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, bytes: u64) -> NodeCapacity {
        NodeCapacity {
            id: id.into(),
            available_bytes: bytes,
            reserved_bytes: 0,
            compute_weight: 1.0,
        }
    }

    fn tensor(id: &str, blocks: u64, bytes: u64) -> TensorWorkload {
        TensorWorkload {
            id: id.into(),
            input_blocks: blocks,
            columns_per_block: 32,
            bytes_per_input_block: bytes,
        }
    }

    fn counts(plan: &PartitionPlan) -> Vec<u64> {
        plan.tensors[0]
            .shards
            .iter()
            .map(|shard| shard.input_blocks.end - shard.input_blocks.start)
            .collect()
    }

    #[test]
    fn equal_nodes_balance_with_aligned_complete_ranges() -> Result<()> {
        let plan = plan_partition(
            &[node("b", 1000), node("a", 1000)],
            &[tensor("w", 20, 18)],
            &[],
        )?;
        assert_eq!(counts(&plan), vec![10, 10]);
        assert_eq!(plan.tensors[0].shards[0].input_blocks, 0..10);
        assert_eq!(plan.tensors[0].shards[1].input_blocks, 10..20);
        assert_eq!(plan.tensors[0].shards[0].input_columns, 0..320);
        assert_eq!(plan.tensors[0].shards[1].input_columns, 320..640);
        assert_eq!(plan.nodes[0].node_id, "a");
        Ok(())
    }

    #[test]
    fn heterogeneous_ram_is_proportional_and_bounded() -> Result<()> {
        let nodes = [node("a", 64), node("b", 32), node("c", 16)];
        let plan = plan_partition(&nodes, &[tensor("w", 112, 1)], &[])?;
        assert_eq!(counts(&plan), vec![64, 32, 16]);
        let plan = plan_partition(&nodes, &[tensor("w", 56, 1)], &[])?;
        for (actual, ideal) in counts(&plan).iter().zip([32u64, 16, 8]) {
            assert!(actual.abs_diff(ideal) <= 1);
        }
        Ok(())
    }

    #[test]
    fn reserves_and_all_tensors_share_one_global_budget() -> Result<()> {
        let mut nodes = [node("a", 100), node("b", 100)];
        nodes[0].reserved_bytes = 30;
        nodes[1].reserved_bytes = 10;
        let tensors = [tensor("x", 8, 10), tensor("y", 8, 10)];
        let plan = plan_partition(&nodes, &tensors, &[])?;
        assert_eq!(plan.nodes[0].tensor_bytes, 70);
        assert_eq!(plan.nodes[1].tensor_bytes, 90);
        for memory in &plan.nodes {
            assert_eq!(memory.total_bytes, 100);
            let sum: u64 = plan
                .tensors
                .iter()
                .map(|t| t.shards[memory.rank].bytes)
                .sum();
            assert_eq!(sum, memory.tensor_bytes);
        }
        assert!(plan_partition(&nodes, &[tensor("x", 9, 10), tensor("y", 8, 10)], &[]).is_err());
        Ok(())
    }

    #[test]
    fn layout_counts_storage_across_output_rows() -> Result<()> {
        let work = TensorWorkload::from_layout("q4", 128, 7, 32, 18)?;
        assert_eq!(work.input_blocks, 4);
        assert_eq!(work.bytes_per_input_block, 126);
        let plan = plan_partition(&[node("a", 252), node("b", 252)], &[work], &[])?;
        assert_eq!(counts(&plan), vec![2, 2]);
        assert!(TensorWorkload::from_layout("q4", 129, 7, 32, 18).is_err());
        assert!(TensorWorkload::from_layout("q4", 128, u64::MAX, 32, 18).is_err());
        Ok(())
    }

    #[test]
    fn ordering_is_independent_and_integer_ties_use_node_id() -> Result<()> {
        let a = node("a", 100);
        let b = node("b", 100);
        let x = tensor("x", 5, 2);
        let y = tensor("y", 9, 1);
        let first = plan_partition(&[a.clone(), b.clone()], &[x.clone(), y.clone()], &[])?;
        let second = plan_partition(&[b, a], &[y, x], &[])?;
        assert_eq!(first, second);
        assert_eq!(counts(&first), vec![3, 2]);
        Ok(())
    }

    #[test]
    fn known_bottleneck_links_and_compute_affect_assignment() -> Result<()> {
        let mut nodes = [node("a", 1000), node("b", 1000), node("c", 1000)];
        let work = [tensor("w", 300, 1)];
        let baseline = plan_partition(&nodes, &work, &[])?;
        assert_eq!(counts(&baseline), vec![100; 3]);
        let links = [
            LinkObservation {
                from: "a".into(),
                to: "b".into(),
                latency_seconds: 0.0001,
                bandwidth_bytes_per_second: 1e10,
            },
            LinkObservation {
                from: "b".into(),
                to: "c".into(),
                latency_seconds: 0.1,
                bandwidth_bytes_per_second: 1e6,
            },
        ];
        let linked = counts(&plan_partition(&nodes, &work, &links)?);
        assert!(linked[0] > linked[1] && linked[1] == linked[2]);
        assert_eq!(
            plan_partition(&nodes, &work, &links)?,
            plan_partition(&nodes, &work, &[links[1].clone(), links[0].clone()])?
        );
        for (latency, bandwidth) in [(0.1, 1e10), (0.0001, 1e6)] {
            let mut isolated = links.clone();
            isolated[1].latency_seconds = latency;
            isolated[1].bandwidth_bytes_per_second = bandwidth;
            let changed = counts(&plan_partition(&nodes, &work, &isolated)?);
            assert!(changed[0] > changed[2]);
        }
        nodes[1].compute_weight = 2.0;
        let weighted = counts(&plan_partition(&nodes, &work, &[])?);
        assert!(weighted[1] > weighted[0]);
        Ok(())
    }

    #[test]
    fn invalid_capacities_dimensions_weights_and_overflows_are_rejected() {
        let work = [tensor("w", 4, 10)];
        assert!(plan_partition(&[], &work, &[]).is_err());
        assert!(plan_partition(&[node("a", 100)], &[], &[]).is_err());
        assert!(plan_partition(&[node("a", 100), node("a", 100)], &work, &[]).is_err());
        assert!(plan_partition(&[node("", 100)], &work, &[]).is_err());
        assert!(plan_partition(&[node("a", 39)], &work, &[]).is_err());
        assert!(plan_partition(&[node("a", 0), node("b", 100)], &work, &[]).is_err());
        for weight in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut bad = node("a", 100);
            bad.compute_weight = weight;
            assert!(plan_partition(&[bad], &work, &[]).is_err());
        }
        let mut bad = node("a", 100);
        bad.reserved_bytes = 101;
        assert!(plan_partition(&[bad], &work, &[]).is_err());
        assert!(plan_partition(&[node("a", 100)], &[tensor("w", 0, 10)], &[]).is_err());
        assert!(plan_partition(&[node("a", 100)], &[tensor("w", 2, 0)], &[]).is_err());
        assert!(
            plan_partition(&[node("a", 100)], &[work[0].clone(), work[0].clone()], &[]).is_err()
        );
        assert!(plan_partition(&[node("a", u64::MAX)], &[tensor("w", u64::MAX, 10)], &[]).is_err());
        assert!(plan_partition(&[node("a", u64::MAX), node("b", 1)], &work, &[]).is_err());
    }

    #[test]
    fn every_rank_requires_nonempty_shards_and_whole_block_capacity() {
        let nodes = [node("a", 15), node("b", 15)];
        assert!(plan_partition(&nodes, &[tensor("w", 1, 10)], &[]).is_err());
        assert!(plan_partition(&nodes, &[tensor("w", 3, 10)], &[]).is_err());
        assert!(plan_partition(
            &[node("a", 15), node("b", 100)],
            &[tensor("x", 2, 10), tensor("y", 2, 10)],
            &[]
        )
        .is_err());
    }

    #[test]
    fn invalid_topology_is_rejected() {
        let nodes = [node("a", 100), node("b", 100)];
        let work = [tensor("w", 4, 10)];
        let valid = LinkObservation {
            from: "a".into(),
            to: "b".into(),
            latency_seconds: 0.001,
            bandwidth_bytes_per_second: 1e9,
        };
        for bandwidth in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut bad = valid.clone();
            bad.bandwidth_bytes_per_second = bandwidth;
            assert!(plan_partition(&nodes, &work, &[bad]).is_err());
        }
        for latency in [-1.0, f64::NAN, f64::INFINITY] {
            let mut bad = valid.clone();
            bad.latency_seconds = latency;
            assert!(plan_partition(&nodes, &work, &[bad]).is_err());
        }
        for endpoint in ["a", "unknown"] {
            let mut bad = valid.clone();
            bad.to = endpoint.into();
            assert!(plan_partition(&nodes, &work, &[bad]).is_err());
        }
        let mut reversed = valid.clone();
        std::mem::swap(&mut reversed.from, &mut reversed.to);
        assert!(plan_partition(&nodes, &work, &[valid, reversed]).is_err());
    }

    #[test]
    fn monitoring_only_recommends_drained_replanning_above_twofold() -> Result<()> {
        assert_eq!(
            monitor_imbalance(&[1.0, 2.0])?,
            ImbalanceRecommendation::KeepStaticPlan
        );
        assert_eq!(
            monitor_imbalance(&[1.0, 2.01])?,
            ImbalanceRecommendation::DrainAndReplan
        );
        assert_eq!(
            monitor_imbalance(&[f64::MAX, f64::MAX])?,
            ImbalanceRecommendation::KeepStaticPlan
        );
        assert!(monitor_imbalance(&[]).is_err());
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(monitor_imbalance(&[1.0, invalid]).is_err());
        }
        Ok(())
    }

    #[test]
    fn water_fill_handles_extreme_weights_and_large_integer_counts() -> Result<()> {
        assert_eq!(
            water_fill(10, &[2, 10], &[1.0, f64::MIN_POSITIVE])?,
            vec![2, 8]
        );
        let count = u64::MAX - 1;
        let filled = water_fill(count, &[count / 2, count / 2], &[1.0, 1.0])?;
        assert_eq!(filled, vec![count / 2; 2]);
        Ok(())
    }

    #[test]
    fn integer_water_fill_preserves_counts_and_caps() -> Result<()> {
        for a in 0..6 {
            for b in 0..6 {
                for c in 0..6 {
                    let caps = [a, b, c];
                    for total in 0..=a + b + c {
                        for scores in [[1.0, 1.0, 1.0], [0.001, 1.0, 100.0]] {
                            let assigned = water_fill(total, &caps, &scores)?;
                            assert_eq!(assigned.iter().sum::<u64>(), total);
                            assert!(assigned.iter().zip(caps).all(|(&n, cap)| n <= cap));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    const GIB: u64 = 1 << 30;

    /// An 80-layer dense transformer (hidden 8192, FFN 28672) stored as Q8_0:
    /// 560 tensors, ~67.7 GiB in total.
    fn q8_model() -> Result<Vec<TensorWorkload>> {
        let mut tensors = Vec::new();
        for layer in 0..80 {
            for (name, input, output) in [
                ("attn_q", 8192, 8192),
                ("attn_k", 8192, 1024),
                ("attn_v", 8192, 1024),
                ("attn_output", 8192, 8192),
                ("ffn_gate", 8192, 28672),
                ("ffn_up", 8192, 28672),
                ("ffn_down", 28672, 8192),
            ] {
                tensors.push(TensorWorkload::from_layout(
                    format!("blk.{layer}.{name}.weight"),
                    input,
                    output,
                    32,
                    34,
                )?);
            }
        }
        Ok(tensors)
    }

    fn total_bytes(tensors: &[TensorWorkload]) -> u64 {
        tensors
            .iter()
            .map(|t| t.input_blocks * t.bytes_per_input_block)
            .sum()
    }

    /// Per-node work time under the planner's own cost model, where throughput
    /// is `compute_weight * usable memory`: `max(t) / mean(t) - 1`.
    fn modeled_imbalance(nodes: &[NodeCapacity], plan: &PartitionPlan) -> f64 {
        let times: Vec<_> = plan
            .nodes
            .iter()
            .map(|memory| {
                let node = nodes.iter().find(|n| n.id == memory.node_id).unwrap();
                memory.tensor_bytes as f64
                    / (node.compute_weight * (node.available_bytes - node.reserved_bytes) as f64)
            })
            .collect();
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        times.iter().copied().fold(0.0, f64::max) / mean - 1.0
    }

    /// Acceptance (#91): two identical nodes get a perfectly balanced split.
    #[test]
    fn acceptance_two_identical_nodes_split_a_real_model_exactly_in_half() -> Result<()> {
        let tensors = q8_model()?;
        let mut nodes = [node("node-b", 64 * GIB), node("node-a", 64 * GIB)];
        for n in &mut nodes {
            n.reserved_bytes = 2 * GIB;
        }
        let plan = plan_partition(&nodes, &tensors, &[])?;
        for tensor in &plan.tensors {
            let [a, b] = &tensor.shards[..] else {
                panic!("two shards expected")
            };
            assert_eq!(a.bytes, b.bytes, "{}", tensor.tensor_id);
            assert_eq!(a.input_blocks.end, b.input_blocks.start);
        }
        assert_eq!(plan.nodes[0].tensor_bytes, plan.nodes[1].tensor_bytes);
        assert_eq!(plan.nodes[0].tensor_bytes * 2, total_bytes(&tensors));
        assert_eq!(modeled_imbalance(&nodes, &plan), 0.0);
        Ok(())
    }

    /// Acceptance (#91): 64/32/16 GiB nodes; every node fits and the planner's
    /// modeled per-node work time is within 10% (wall clock is not measured here).
    #[test]
    fn acceptance_64_32_16_gib_nodes_fit_ram_with_under_ten_percent_imbalance() -> Result<()> {
        let tensors = q8_model()?;
        let model = total_bytes(&tensors);
        let mut nodes = [
            node("big", 64 * GIB),
            node("mid", 32 * GIB),
            node("small", 16 * GIB),
        ];
        for n in &mut nodes {
            n.reserved_bytes = 2 * GIB;
        }
        // An even three-way split would not fit the 16 GiB node.
        assert!(model / 3 > 14 * GIB);
        let plan = plan_partition(&nodes, &tensors, &[])?;
        for (memory, n) in plan.nodes.iter().zip(&nodes) {
            assert_eq!(memory.node_id, n.id);
            assert!(memory.total_bytes <= n.available_bytes, "{memory:?}");
            assert_eq!(memory.total_bytes, memory.tensor_bytes + 2 * GIB);
        }
        assert_eq!(
            plan.nodes.iter().map(|m| m.tensor_bytes).sum::<u64>(),
            model
        );
        let imbalance = modeled_imbalance(&nodes, &plan);
        assert!(imbalance < 0.10, "modeled imbalance {imbalance}");
        eprintln!(
            "64/32/16 GiB: model {:.2} GiB, tensor GiB per node {:?}, modeled imbalance {:.4}%",
            model as f64 / GIB as f64,
            plan.nodes
                .iter()
                .map(|m| format!("{:.2}", m.tensor_bytes as f64 / GIB as f64))
                .collect::<Vec<_>>(),
            imbalance * 100.0
        );
        Ok(())
    }

    fn link(from: &str, to: &str, latency: f64, bandwidth: f64) -> LinkObservation {
        LinkObservation {
            from: from.into(),
            to: to.into(),
            latency_seconds: latency,
            bandwidth_bytes_per_second: bandwidth,
        }
    }

    #[test]
    fn resolved_links_prefer_overrides_then_measurements_then_fallback() -> Result<()> {
        let nodes = [node("a", 100), node("b", 100), node("c", 100)];
        let measured = [
            link("a", "b", 0.001, 1e9),
            link("b", "a", 0.002, 2e9),
            link("b", "c", 0.001, 1e9),
        ];
        let overrides = [link("c", "b", 0.5, 1e6)];
        let fallback = LinkDefaults {
            latency_seconds: 0.01,
            bandwidth_bytes_per_second: 1e8,
        };
        let links = resolve_links(&nodes, &measured, &overrides, Some(fallback))?;
        assert_eq!(
            links,
            vec![
                link("a", "b", 0.002, 1e9),
                link("a", "c", 0.01, 1e8),
                link("c", "b", 0.5, 1e6),
            ]
        );
        // Without a fallback, unknown pairs stay unknown.
        assert_eq!(resolve_links(&nodes, &measured, &[], None)?.len(), 2);
        assert!(resolve_links(
            &nodes,
            &[],
            &[overrides[0].clone(), link("b", "c", 1.0, 1.0)],
            None
        )
        .is_err());
        for bad in [
            (f64::NAN, 1.0),
            (-1.0, 1.0),
            (0.0, 0.0),
            (0.0, f64::INFINITY),
        ] {
            let defaults = LinkDefaults {
                latency_seconds: bad.0,
                bandwidth_bytes_per_second: bad.1,
            };
            assert!(resolve_links(&nodes, &[], &[], Some(defaults)).is_err());
        }
        plan_partition(&nodes, &[tensor("w", 30, 1)], &links)?;
        Ok(())
    }

    fn timing(id: &str, busy: f64) -> RankTiming {
        RankTiming {
            node_id: id.into(),
            busy_seconds: busy,
            wait_seconds: 0.0,
        }
    }

    /// Simulate one window: every rank's busy time is its blocks over its true
    /// per-block rate; faster ranks wait at the collective for the slowest.
    fn simulate(plan: &PartitionPlan, rates: &[f64]) -> Vec<RankTiming> {
        let busy: Vec<_> = plan
            .nodes
            .iter()
            .map(|memory| {
                let blocks: u64 = plan
                    .tensors
                    .iter()
                    .map(|t| {
                        let r = &t.shards[memory.rank].input_blocks;
                        r.end - r.start
                    })
                    .sum();
                blocks as f64 / rates[memory.rank]
            })
            .collect();
        let slowest = busy.iter().copied().fold(0.0, f64::max);
        plan.nodes
            .iter()
            .zip(busy)
            .map(|(memory, busy)| RankTiming {
                node_id: memory.node_id.clone(),
                busy_seconds: busy,
                wait_seconds: slowest - busy,
            })
            .collect()
    }

    #[test]
    fn sustained_imbalance_proposes_a_plan_that_balances_busy_time() -> Result<()> {
        let nodes = [node("a", 1 << 20), node("b", 1 << 20), node("c", 1 << 20)];
        let work = [tensor("x", 600, 1), tensor("y", 900, 2)];
        // Ranks are equal on paper but really run at 1.0, 0.5 and 2.0 blocks/s.
        let rates = [1.0, 0.5, 2.0];
        let plan = plan_partition(&nodes, &work, &[])?;
        let before = simulate(&plan, &rates);
        let initial = load_imbalance(&before)?;
        assert!(initial > 0.5, "{initial}");
        let mut monitor = RepartitionMonitor::new(RepartitionPolicy::default())?;
        assert!(monitor.observe(&before, &nodes, &work, &[])?.is_none());
        assert!(monitor.observe(&before, &nodes, &work, &[])?.is_none());
        assert_eq!(monitor.streak(), 2);
        let proposal = monitor
            .observe(&before, &nodes, &work, &[])?
            .expect("third imbalanced window triggers");
        assert_eq!(monitor.streak(), 0);
        assert!((proposal.observed_imbalance - initial).abs() < 1e-12);
        let after = load_imbalance(&simulate(&proposal.plan, &rates))?;
        assert!(after < 0.10, "imbalance after replan {after}");
        assert!(proposal.nodes[2].compute_weight > proposal.nodes[0].compute_weight);
        assert!(proposal.nodes[0].compute_weight > proposal.nodes[1].compute_weight);
        eprintln!(
            "simulated repartition: imbalance {:.1}% -> {:.2}%",
            initial * 100.0,
            after * 100.0
        );
        // The balanced plan stays put.
        let balanced = simulate(&proposal.plan, &rates);
        for _ in 0..10 {
            assert!(monitor
                .observe(&balanced, &proposal.nodes, &work, &[])?
                .is_none());
        }
        Ok(())
    }

    #[test]
    fn transient_imbalance_and_membership_changes_reset_the_streak() -> Result<()> {
        let nodes = [node("a", 1000), node("b", 1000)];
        let work = [tensor("w", 100, 1)];
        let mut monitor = RepartitionMonitor::new(RepartitionPolicy {
            sustained_windows: 2,
            ..Default::default()
        })?;
        let slow = [timing("a", 1.0), timing("b", 3.0)];
        let fine = [timing("a", 1.0), timing("b", 1.05)];
        let other = [timing("a", 1.0), timing("c", 3.0)];
        assert!(monitor.observe(&slow, &nodes, &work, &[])?.is_none());
        assert!(monitor.observe(&fine, &nodes, &work, &[])?.is_none());
        assert!(monitor.observe(&slow, &nodes, &work, &[])?.is_none());
        let other_nodes = [node("a", 1000), node("c", 1000)];
        assert!(monitor.observe(&other, &other_nodes, &work, &[])?.is_none());
        assert_eq!(monitor.streak(), 1);
        assert!(monitor.observe(&other, &other_nodes, &work, &[])?.is_some());
        Ok(())
    }

    #[test]
    fn invalid_timings_and_policies_are_rejected() {
        assert!(load_imbalance(&[]).is_err());
        assert!(load_imbalance(&[timing("a", 1.0), timing("a", 1.0)]).is_err());
        for busy in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(load_imbalance(&[timing("a", busy)]).is_err());
        }
        let mut waiting = timing("a", 1.0);
        waiting.wait_seconds = -1.0;
        assert!(load_imbalance(&[waiting]).is_err());
        let nodes = [node("a", 100), node("b", 100)];
        assert!(rebalance_weights(&nodes, &[timing("a", 1.0)], 4.0).is_err());
        assert!(rebalance_weights(&nodes, &[timing("a", 1.0), timing("c", 1.0)], 4.0).is_err());
        assert!(rebalance_weights(&nodes, &[timing("a", 1.0), timing("b", 1.0)], 0.5).is_err());
        let clamped =
            rebalance_weights(&nodes, &[timing("a", 1.0), timing("b", 1000.0)], 1.5).unwrap();
        assert_eq!(clamped[0].compute_weight, 1.5);
        assert_eq!(clamped[1].compute_weight, 1.0 / 1.5);
        for policy in [
            RepartitionPolicy {
                max_imbalance: f64::NAN,
                ..Default::default()
            },
            RepartitionPolicy {
                sustained_windows: 0,
                ..Default::default()
            },
            RepartitionPolicy {
                max_weight_step: 0.9,
                ..Default::default()
            },
        ] {
            assert!(RepartitionMonitor::new(policy).is_err());
        }
    }
}
