//! Expert residency backends.
//!
//! The routing-frequency hot-expert cache ([`crate::hot_experts::HotExpertCache`])
//! is pure policy: it names which `(layer, expert)` pairs to keep hot.  This
//! module is the *executor* — it makes a named expert's weights resident on
//! the active compute device.
//!
//! Today the only backend is **CPU** (best-effort `MADV_WILLNEED` over each
//! expert's borrowed mmap ranges — the historical `--pin-hot-experts`
//! behavior).  The trait exists so a device backend (a CUDA/Metal slot cache
//! that copies hot experts into VRAM) can be added later without touching the
//! policy or the loaders: policy names experts, `acquire`/`release` move
//! bytes, `capacity` sizes the budget.

use std::sync::Arc;

/// The three per-expert weight-tensor residency handles (gate/up/down).
///
/// Each handle can make its byte range in the model mapping resident on the
/// active device.  The MoE loaders already slice every expert tensor into
/// per-expert ranges, so an expert's residency is exactly these three handles.
#[derive(Clone)]
pub struct ExpertHandles {
    /// Gate projection handle.
    pub gate: Arc<dyn crate::mmap_tensor::MmapPrefetch>,
    /// Up projection handle.
    pub up: Arc<dyn crate::mmap_tensor::MmapPrefetch>,
    /// Down projection handle.
    pub down: Arc<dyn crate::mmap_tensor::MmapPrefetch>,
}

impl ExpertHandles {
    /// Ask the backend to make all three weight ranges resident (best effort).
    pub fn prefetch(&self) {
        self.gate.prefetch();
        self.up.prefetch();
        self.down.prefetch();
    }
}

/// Where a hot expert's weights live, and how to make them resident on the
/// active compute device.
///
/// Policy ([`crate::hot_experts::HotExpertCache`]) only names experts; this
/// trait executes residency.  Best-effort: implementations must never fail
/// the caller — a failed hint degrades to demand-faulting, which is the
/// no-cache baseline.
pub trait ExpertResidency: Send + Sync + 'static {
    /// Make `(layer, expert)` resident.  Idempotent: re-acquiring an already
    /// resident expert is a cheap no-op or hit.
    fn acquire(&self, layer: u32, expert: u32);
    /// Release `(layer, expert)` residency (e.g. free a device slot).  A
    /// no-op for the CPU backend, whose eviction is kernel-managed.
    fn release(&self, layer: u32, expert: u32);
    /// The number of experts this backend can hold resident.  Drives the
    /// hot-expert-cache budget on devices; on CPU it is informational (the
    /// budget stays operator-set via `--pin-hot-experts`).
    fn capacity(&self) -> usize;
}

/// CPU residency backend: best-effort `MADV_WILLNEED` over each hot expert's
/// borrowed mmap ranges — the historical `--pin-hot-experts` behavior.
pub struct CpuResidency {
    /// `[layer][expert]` → the expert's handles (`None` when not mmap-backed).
    experts: Vec<Vec<Option<ExpertHandles>>>,
    /// Number of experts that carry handles (informational capacity).
    capacity: usize,
}

impl CpuResidency {
    /// Build from a per-layer table of per-expert handles.
    pub fn new(experts: Vec<Vec<Option<ExpertHandles>>>) -> Self {
        let capacity = experts.iter().flatten().filter(|h| h.is_some()).count();
        Self { experts, capacity }
    }
}

impl ExpertResidency for CpuResidency {
    fn acquire(&self, layer: u32, expert: u32) {
        if let Some(h) = self
            .experts
            .get(layer as usize)
            .and_then(|row| row.get(expert as usize))
            .and_then(|h| h.as_ref())
        {
            h.prefetch();
        }
    }

    fn release(&self, _layer: u32, _expert: u32) {
        // Kernel-managed page cache: nothing to free explicitly.
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

/// No-op residency for architectures/backends without per-expert handles
/// (dense models, the vendored candle `llama` loader, non-mmap loads).
pub struct NoopResidency;

impl ExpertResidency for NoopResidency {
    fn acquire(&self, _layer: u32, _expert: u32) {}
    fn release(&self, _layer: u32, _expert: u32) {}
    fn capacity(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct CountingPrefetch(Arc<AtomicUsize>);

    impl crate::mmap_tensor::MmapPrefetch for CountingPrefetch {
        fn prefetch(&self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn handle(counter: &Arc<AtomicUsize>) -> Arc<dyn crate::mmap_tensor::MmapPrefetch> {
        Arc::new(CountingPrefetch(Arc::clone(counter)))
    }

    /// Acquiring an expert prefetches its three weight tensors and counts
    /// toward capacity.
    #[test]
    fn acquire_prefetches_the_named_expert() {
        let n = Arc::new(AtomicUsize::new(0));
        let handles = ExpertHandles {
            gate: handle(&n),
            up: handle(&n),
            down: handle(&n),
        };
        let res = CpuResidency::new(vec![vec![Some(handles)]]);
        res.acquire(0, 0);
        assert_eq!(n.load(Ordering::Relaxed), 3, "gate+up+down prefetched");
        assert_eq!(res.capacity(), 1);
    }

    /// Out-of-range layers/experts and handle-less experts are safe no-ops.
    #[test]
    fn acquire_is_a_noop_out_of_range_or_without_handles() {
        let n = Arc::new(AtomicUsize::new(0));
        let res = CpuResidency::new(vec![vec![
            None,
            Some(ExpertHandles {
                gate: handle(&n),
                up: handle(&n),
                down: handle(&n),
            }),
        ]]);
        res.acquire(9, 0); // layer out of range
        res.acquire(0, 9); // expert out of range
        res.acquire(0, 0); // no handles
        res.release(0, 1);
        assert_eq!(n.load(Ordering::Relaxed), 0);
        assert_eq!(res.capacity(), 1);
    }

    /// The no-op backend is fully inert.
    #[test]
    fn noop_residency_is_inert() {
        let r = NoopResidency;
        r.acquire(0, 0);
        r.release(0, 0);
        assert_eq!(r.capacity(), 0);
    }
}
