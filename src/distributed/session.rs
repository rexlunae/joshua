//! One fail-stop, explicitly configured inference job per rank.
//!
//! All participants must use the same call order. Discovery is not membership.
//! A failed operation permanently invalidates the session; restart every rank
//! with a fresh session UUID. Authentication does not encrypt activations.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::{
    collective::{AllReduceGroup, MulticastConfig, MAX_TENSOR_ELEMENTS, MAX_TOTAL_ELEMENTS},
    tcp::{TcpAllReduceGroup, TcpConfig},
};

const MAX_CALL_ELEMENTS: usize = 16 * 1024 * 1024;

enum Transport {
    Tcp(TcpAllReduceGroup),
    Udp(AllReduceGroup),
}

impl Transport {
    fn reduce(&mut self, values: &mut [f32]) -> Result<()> {
        match self {
            Self::Tcp(group) => group.all_reduce_sum(values),
            Self::Udp(group) => group.all_reduce_sum(values),
        }
    }
}

/// A single ordered stream of collectives, shared by one model's MoE layers.
///
/// Do not interleave independent jobs on this object. Use a fresh transport,
/// session UUID and model for each job.
pub struct ClusterSession {
    rank: usize,
    world_size: usize,
    transport: Mutex<Transport>,
    poisoned: AtomicBool,
    model_claimed: AtomicBool,
}

impl ClusterSession {
    pub fn tcp(rank: usize, session: Uuid, key: &[u8], config: TcpConfig) -> Result<Self> {
        let world_size = config.peers.len();
        let transport = Transport::Tcp(TcpAllReduceGroup::new(rank, session, key, config)?);
        Ok(Self::new(rank, world_size, transport))
    }

    pub fn udp(
        rank: usize,
        world_size: usize,
        session: Uuid,
        key: &[u8],
        config: MulticastConfig,
    ) -> Result<Self> {
        let transport =
            Transport::Udp(AllReduceGroup::new(rank, world_size, session, key, config)?);
        Ok(Self::new(rank, world_size, transport))
    }

    fn new(rank: usize, world_size: usize, transport: Transport) -> Self {
        Self {
            rank,
            world_size,
            transport: Mutex::new(transport),
            poisoned: AtomicBool::new(false),
            model_claimed: AtomicBool::new(false),
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub(crate) fn claim_model(&self) -> Result<()> {
        ensure!(
            !self.model_claimed.swap(true, Ordering::SeqCst),
            "a cluster session can own only one model"
        );
        self.check()
    }

    pub(crate) fn abort(&self) {
        self.poisoned.store(true, Ordering::SeqCst);
    }

    fn check(&self) -> Result<()> {
        ensure!(
            !self.poisoned.load(Ordering::SeqCst),
            "cluster session failed; restart every rank with a fresh session UUID"
        );
        Ok(())
    }

    fn operation<T>(&self, f: impl FnOnce(&mut Transport) -> Result<T>) -> Result<T> {
        self.check()?;
        let result = (|| {
            let mut transport = self.transport.lock().map_err(|_| {
                anyhow::anyhow!("cluster transport lock poisoned; restart every rank")
            })?;
            self.check()?;
            f(&mut transport)
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }

    fn agree_locked(&self, transport: &mut Transport, bytes: &[u8]) -> Result<()> {
        let digest = Sha256::digest(bytes);
        let mut reference: Vec<f32> = digest
            .iter()
            .map(|&byte| if self.rank == 0 { byte as f32 } else { 0.0 })
            .collect();
        transport.reduce(&mut reference)?;
        let matches = reference
            .iter()
            .zip(digest)
            .all(|(&actual, expected)| actual == expected as f32);
        let mut mismatch = [if matches { 0.0 } else { 1.0 }];
        transport.reduce(&mut mismatch)?;
        ensure!(
            mismatch == [0.0],
            "cluster ranks disagree on model, inputs, or collective operation"
        );
        Ok(())
    }

    /// Compare SHA-256 digests of job metadata or inputs without disclosing them.
    /// Every rank observes a mismatch before proceeding.
    pub fn agree(&self, bytes: &[u8]) -> Result<()> {
        self.operation(|transport| self.agree_locked(transport, bytes))
    }

    fn exchange(&self, values: &mut [f32], broadcast: bool) -> Result<()> {
        self.operation(|transport| {
            ensure!(
                values.len() <= MAX_CALL_ELEMENTS,
                "cluster activation exceeds the per-call element limit"
            );
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "nonfinite cluster activation"
            );
            let mut descriptor = Vec::from(b"joshua-cluster-vector-v1".as_slice());
            descriptor.push(u8::from(broadcast));
            descriptor.extend_from_slice(&(values.len() as u64).to_le_bytes());
            self.agree_locked(transport, &descriptor)?;

            // Commit only after every chunk succeeds. Transport frames and
            // aggregate peer storage remain bounded even for a long prefill.
            let mut result = values.to_vec();
            if broadcast && self.rank != 0 {
                result.fill(0.0);
            }
            let chunk_size = MAX_TENSOR_ELEMENTS.min(MAX_TOTAL_ELEMENTS / self.world_size);
            for chunk in result.chunks_mut(chunk_size) {
                transport
                    .reduce(chunk)
                    .context("exchanging cluster activations")?;
                ensure!(
                    chunk.iter().all(|v| v.is_finite()),
                    "nonfinite reduced cluster activation"
                );
            }
            values.copy_from_slice(&result);
            Ok(())
        })
    }

    /// Sum all ranks' partial activations, chunking under transport limits.
    pub fn all_reduce_sum(&self, values: &mut [f32]) -> Result<()> {
        self.exchange(values, false)
    }

    /// Copy rank zero's values to every rank. All ranks supply equal-length buffers.
    pub fn broadcast(&self, values: &mut [f32]) -> Result<()> {
        self.exchange(values, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{net::TcpListener, sync::Arc, time::Duration};

    fn pair() -> [Arc<ClusterSession>; 2] {
        let listeners: Vec<_> = (0..2)
            .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
            .collect();
        let peers = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        drop(listeners);
        let config = TcpConfig {
            peers,
            timeout: Duration::from_secs(5),
        };
        let id = Uuid::new_v4();
        let key = rand::random::<[u8; 32]>();
        std::array::from_fn(|rank| {
            Arc::new(ClusterSession::tcp(rank, id, &key, config.clone()).unwrap())
        })
    }

    #[test]
    fn chunked_reduce_and_root_broadcast() {
        let ranks = pair();
        std::thread::scope(|scope| {
            let handles: Vec<_> = ranks
                .iter()
                .map(|rank| {
                    scope.spawn(move || {
                        rank.agree(b"same job").unwrap();
                        let mut data = vec![rank.rank() as f32 + 1.0; MAX_TENSOR_ELEMENTS + 3];
                        rank.all_reduce_sum(&mut data).unwrap();
                        assert!(data.iter().all(|&v| v == 3.0));
                        data.fill(rank.rank() as f32 + 7.0);
                        rank.broadcast(&mut data).unwrap();
                        assert!(data.iter().all(|&v| v == 7.0));
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn mismatch_fails_all_ranks_and_poisons_session() {
        let ranks = pair();
        std::thread::scope(|scope| {
            let handles: Vec<_> = ranks
                .iter()
                .map(|rank| {
                    scope.spawn(move || {
                        assert!(rank.agree(&[rank.rank() as u8]).is_err());
                        assert!(rank.agree(b"retry").is_err());
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn shape_mismatch_preserves_inputs() {
        let ranks = pair();
        std::thread::scope(|scope| {
            let handles: Vec<_> = ranks
                .iter()
                .map(|rank| {
                    scope.spawn(move || {
                        let mut data = vec![2.0; rank.rank() + 1];
                        let original = data.clone();
                        assert!(rank.all_reduce_sum(&mut data).is_err());
                        assert_eq!(data, original);
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn operation_mismatch_fails_instead_of_mixing_broadcast_and_sum() {
        let ranks = pair();
        std::thread::scope(|scope| {
            let handles: Vec<_> = ranks
                .iter()
                .map(|rank| {
                    scope.spawn(move || {
                        let mut data = [2.0];
                        let result = if rank.rank() == 0 {
                            rank.broadcast(&mut data)
                        } else {
                            rank.all_reduce_sum(&mut data)
                        };
                        assert!(result.is_err());
                        assert_eq!(data, [2.0]);
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn local_failure_invalidates_session_and_model_cannot_be_reused() {
        let ranks = pair();
        let rank = &ranks[0];
        rank.claim_model().unwrap();
        assert!(rank.claim_model().is_err());
        let mut data = [f32::INFINITY];
        assert!(rank.all_reduce_sum(&mut data).is_err());
        assert_eq!(data, [f32::INFINITY]);
        assert!(rank.agree(b"retry").is_err());
    }
}
