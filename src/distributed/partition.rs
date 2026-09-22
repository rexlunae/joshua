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
            input_columns > 0 && input_columns % columns_per_block == 0,
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
}
