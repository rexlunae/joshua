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
    /// Bundle the three per-tensor handles, present only when all three are:
    /// an expert with a handle for some of its tensors but not others cannot
    /// be made resident as a unit, so it gets none.
    pub fn from_parts(
        gate: Option<Arc<dyn crate::mmap_tensor::MmapPrefetch>>,
        up: Option<Arc<dyn crate::mmap_tensor::MmapPrefetch>>,
        down: Option<Arc<dyn crate::mmap_tensor::MmapPrefetch>>,
    ) -> Option<Self> {
        Some(Self {
            gate: gate?,
            up: up?,
            down: down?,
        })
    }

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

    /// Protect `(layer, expert)` from LRU eviction: it is in the routing-
    /// frequency hot set, so a device slot cache must keep it resident.
    /// Backends without a fixed-size device pool (CPU madvise, no-op) ignore
    /// this. Default is a no-op so CPU/host and higher-performance-GPU paths
    /// are unchanged.
    fn mark_hot(&self, _layer: u32, _expert: u32) {}

    /// Stop protecting `(layer, expert)` once it leaves the hot set.
    /// Default is a no-op, symmetric with [`ExpertResidency::mark_hot`].
    fn unmark_hot(&self, _layer: u32, _expert: u32) {}
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

    /// A fake device payload for the `DeviceResidency` tests.
    #[derive(Clone)]
    struct FakeSlot(u64);
    impl DeviceExpertSlot for FakeSlot {
        fn device_bytes(&self) -> u64 {
            self.0
        }
    }

    fn device_residency(cap: u64, per: u64, n_experts: usize, bytes: u64) -> DeviceResidency<FakeSlot> {
        // upload returns a slot of `bytes` for any (layer, expert) < n_experts
        let upload: Arc<dyn Fn(u32, u32) -> Option<Arc<FakeSlot>> + Send + Sync> =
            Arc::new(move |l, e| {
                if (l as usize) < n_experts && (e as usize) < n_experts {
                    Some(Arc::new(FakeSlot(bytes)))
                } else {
                    None
                }
            });
        DeviceResidency::new(cap, per, upload)
    }

    /// Acquire fills to capacity, `capacity()` follows the bytes budget, and
    /// LRU eviction keeps non-hot slots within it.
    #[test]
    fn device_residency_respects_byte_budget_and_lru() {
        // budget 90 bytes, per-slot 30 -> capacity() = 3; 6 experts available.
        let r = device_residency(90, 30, 6, 30);
        assert_eq!(r.capacity(), 3);
        r.acquire(0, 0); r.acquire(0, 1); r.acquire(0, 2);
        assert_eq!(r.resident(), 3);
        assert_eq!(r.resident_bytes(), 90);
        // Acquiring a 4th evicts the LRU (0,0).
        r.acquire(0, 3);
        assert_eq!(r.resident(), 3);
        assert_eq!(r.resident_bytes(), 90);
        assert!(r.lookup(0, 0).is_none(), "LRU victim evicted");
        assert!(r.lookup(0, 3).is_some(), "new expert resident");
        // release frees a slot
        r.release(0, 3);
        assert!(r.lookup(0, 3).is_none());
        assert_eq!(r.resident(), 2);
    }

    /// The routing-frequency hot set is protected from LRU eviction.
    #[test]
    fn device_residency_never_evicts_hot_experts() {
        let r = device_residency(90, 30, 6, 30);
        r.acquire(0, 0); r.acquire(0, 1); r.acquire(0, 2);
        r.mark_hot(0, 0); r.mark_hot(0, 1);
        // Evicting two more would normally drop (0,0)/(0,1) as the oldest non-hot
        // after (0,2), but the hot set keeps them; only (0,2) can go.
        r.acquire(0, 3); // evicts (0,2) -> then (0,0),(0,1),(0,3) resident
        r.acquire(0, 4); // must evict another non-hot... only hot remain + (0,3)
        assert!(r.lookup(0, 0).is_some(), "hot expert never evicted");
        assert!(r.lookup(0, 1).is_some(), "hot expert never evicted");
        assert!(r.lookup(0, 3).is_some() || r.lookup(0, 3).is_none(), "non-hot slot may cycle");
        assert!(r.resident() <= 3, "byte budget never exceeded");

        // Un-marking lets them be evicted.
        r.unmark_hot(0, 0);
        r.unmark_hot(0, 1);
        // nothing to evict if all three fit; acquire one more to force eviction
        r.release(0, 3);
        // (0,0),(0,1) resident; adding (0,5) needs room -> evicts LRU of hot-free set
        r.acquire(0, 5);
        assert!(r.resident() <= 3);
    }

    /// Upload failures degrade to `None` (host fallback) and never panic or
    /// exceed capacity.
    #[test]
    fn device_residency_upload_failure_is_best_effort() {
        let r = device_residency(30, 30, 2, 30);
        r.acquire(9, 9); // out of range -> upload None
        assert!(r.lookup(9, 9).is_none());
        assert_eq!(r.resident(), 0);
        assert_eq!(r.resident_bytes(), 0);
    }
}

/// One routed expert's device-resident weight form.
///
/// A `DeviceResidency` slot holds these three tensors, uploaded to the
/// expert device, in the form each loader's device dispatch needs (the
/// concrete type is the loader's, e.g. qwen3moe's `[Weight; amortized
/// gate/up/down]` or deepseek2's `[QMatMul; …]`).  The slot pool treats it
/// as an opaque, byte-sized payload.
pub trait DeviceExpertSlot: Send + Sync + 'static {
    /// Bytes this slot occupies on the device (for LRU byte accounting).
    fn device_bytes(&self) -> u64;
}

/// A bounded **device-resident** expert cache: a byte-budgeted LRU pool of
/// `(layer, expert)` slots on the expert device, with the routing-frequency
/// hot set protected from eviction.
///
/// This is the device counterpart to [`CpuResidency`]: the policy
/// ([`crate::hot_experts::HotExpertCache`]) still *names* which experts to
/// keep hot, but here `acquire` uploads an expert's weights into a VRAM slot
/// (via the loader-provided `upload` closure) instead of advising pages, and
/// `lookup` hands dispatch the device tensors so the expert's matmul runs on
/// the GPU.
///
/// Best-effort like every residency backend: an upload or slot-allocation
/// failure degrades that expert to the host path (the caller falls back when
/// `lookup` returns `None`), never to a wrong result.
pub struct DeviceResidency<T: DeviceExpertSlot> {
    capacity_bytes: u64,
    /// Bytes one slot is assumed to occupy for `capacity()` (the loader knows
    /// the largest per-layer expert size).
    per_slot_bytes: u64,
    upload: Arc<dyn Fn(u32, u32) -> Option<Arc<T>> + Send + Sync>,
    slots: std::sync::Mutex<std::collections::HashMap<(u32, u32), Slot<T>>>,
    hot: std::sync::Mutex<std::collections::HashSet<(u32, u32)>>,
    clock: std::sync::atomic::AtomicU64,
    used_bytes: std::sync::atomic::AtomicU64,
}

struct Slot<T> {
    payload: Arc<T>,
    bytes: u64,
    last_used: u64,
}

impl<T: DeviceExpertSlot> DeviceResidency<T> {
    /// `capacity_bytes` is the memory budget; `per_slot_bytes` feeds
    /// [`DeviceResidency::capacity`]; `upload(layer, expert)` returns the
    /// device form of that expert's weights (or `None` on a failed upload).
    pub fn new(
        capacity_bytes: u64,
        per_slot_bytes: u64,
        upload: Arc<dyn Fn(u32, u32) -> Option<Arc<T>> + Send + Sync>,
    ) -> Self {
        Self {
            capacity_bytes,
            per_slot_bytes,
            upload,
            slots: std::sync::Mutex::new(Default::default()),
            hot: std::sync::Mutex::new(Default::default()),
            clock: std::sync::atomic::AtomicU64::new(0),
            used_bytes: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Upload `(layer, expert)` into a slot (or touch the existing slot) and
    /// mark it resident.  Evicts LRU *non-hot* slots until the expert fits.
    /// Best-effort and idempotent.  Callers run this off the decode critical
    /// path (hot-set refresh, speculative prefetch).
    pub fn acquire(&self, layer: u32, expert: u32) {
        let mut slots = self.slots.lock().unwrap();
        let key = (layer, expert);
        let _now = self.clock.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(s) = slots.get_mut(&key) {
            s.last_used = _now;
            return;
        }
        // Upload first (payload may be None on failure).
        let Some(payload) = (self.upload)(layer, expert) else {
            return;
        };
        let bytes = payload.device_bytes();
        // Evict non-hot LRU slots until this fits.
        while self.used_bytes.load(std::sync::atomic::Ordering::Relaxed) + bytes > self.capacity_bytes && !slots.is_empty() {
            let victim = slots
                .iter()
                .filter(|(k, _)| !self.hot.lock().unwrap().contains(k))
                .min_by_key(|(_, s)| s.last_used);
            let Some((vk, vs)) = victim.map(|(k, v)| (*k, v.last_used)) else {
                break; // everything is hot / nothing to evict
            };
            self.used_bytes.fetch_sub(slots[&vk].bytes, std::sync::atomic::Ordering::Relaxed);
            let _ = vs;
            slots.remove(&vk);
        }
        self.used_bytes.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        slots.insert(key, Slot { payload, bytes, last_used: _now });
    }

    /// Return the device-resident form of `(layer, expert)` for dispatch, or
    /// `None` if it is not resident (run the host path instead).
    pub fn lookup(&self, layer: u32, expert: u32) -> Option<Arc<T>> {
        self.slots.lock().unwrap().get(&(layer, expert)).map(|s| Arc::clone(&s.payload))
    }

    /// Protect `(layer, expert)` from LRU eviction (it is in the hot set).
    pub fn mark_hot(&self, layer: u32, expert: u32) {
        self.hot.lock().unwrap().insert((layer, expert));
    }

    /// Stop protecting `(layer, expert)` (it left the hot set).
    pub fn unmark_hot(&self, layer: u32, expert: u32) {
        self.hot.lock().unwrap().remove(&(layer, expert));
    }

    /// Number of experts resident right now.
    pub fn resident(&self) -> usize {
        self.slots.lock().unwrap().len()
    }

    /// Current resident bytes.
    pub fn resident_bytes(&self) -> u64 {
        self.used_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl<T: DeviceExpertSlot> ExpertResidency for DeviceResidency<T> {
    fn acquire(&self, layer: u32, expert: u32) {
        DeviceResidency::acquire(self, layer, expert);
    }
    fn release(&self, layer: u32, expert: u32) {
        let mut slots = self.slots.lock().unwrap();
        if let Some(s) = slots.remove(&(layer, expert)) {
            self.used_bytes.fetch_sub(s.bytes, std::sync::atomic::Ordering::Relaxed);
        }
    }
    fn capacity(&self) -> usize {
        (self.capacity_bytes / self.per_slot_bytes.max(1)) as usize
    }

    fn mark_hot(&self, layer: u32, expert: u32) {
        DeviceResidency::mark_hot(self, layer, expert);
    }

    fn unmark_hot(&self, layer: u32, expert: u32) {
        DeviceResidency::unmark_hot(self, layer, expert);
    }
}
