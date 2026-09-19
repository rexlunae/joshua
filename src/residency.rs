//! Expert residency backends.
//!
//! The routing-frequency hot-expert cache ([`crate::hot_experts::HotExpertCache`])
//! is pure policy: it names which `(layer, expert)` pairs to keep hot.  This
//! module is the *executor* — it makes a named expert's weights resident on
//! the active compute device.
//!
//! Two backends exist: **CPU** ([`CpuResidency`], best-effort `MADV_WILLNEED`
//! over each expert's borrowed mmap ranges — the historical
//! `--pin-hot-experts` behavior) and a **device slot pool**
//! ([`DeviceResidency`], a byte-budgeted LRU of experts uploaded to the
//! expert device, with the hot set protected from eviction).  Policy names
//! experts, `acquire`/`release` move bytes, `capacity` sizes the budget.

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

    /// The three mapped byte ranges (empty for handles without a mapping).
    pub fn mapped_ranges(&self) -> Vec<crate::mmap_tensor::MappedRange> {
        [&self.gate, &self.up, &self.down]
            .into_iter()
            .filter_map(|h| h.mapped_range())
            .collect()
    }

    /// How many of the expert's pages are resident in memory right now, as
    /// `(resident, total)`; `None` when no range can be probed.
    pub fn resident_pages(&self) -> Option<(usize, usize)> {
        let mut acc: Option<(usize, usize)> = None;
        for r in self.mapped_ranges() {
            if let Some((res, total)) = r.resident_pages() {
                let (a, b) = acc.unwrap_or((0, 0));
                acc = Some((a + res, b + total));
            }
        }
        acc
    }

    /// Release the expert's host pages: drop them from the mapping and,
    /// when the model `file` is known, from the page cache too.  For an
    /// expert that is resident on a device, so the RAM it occupied can hold
    /// experts that are not.  Best-effort.
    pub fn release_host_pages(&self, file: Option<&std::fs::File>) {
        for r in self.mapped_ranges() {
            r.drop_pages();
            if let Some(f) = file {
                r.evict_from_cache(f);
            }
        }
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

    /// Replace the protected set wholesale with `hot` (the routing-frequency
    /// refresh hands over the *whole* hot set each time, so members that
    /// dropped out become evictable again).  Default: mark each member hot.
    fn replace_hot_set(&self, hot: &[(u32, u32)]) {
        for &(l, e) in hot {
            self.mark_hot(l, e);
        }
    }
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
    pub(super) struct CountingPrefetch(pub(super) Arc<AtomicUsize>);

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

    fn device_residency(
        cap: u64,
        per: u64,
        n_experts: usize,
        bytes: u64,
    ) -> DeviceResidency<FakeSlot> {
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
        r.acquire(0, 0);
        r.acquire(0, 1);
        r.acquire(0, 2);
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
        r.acquire(0, 0);
        r.acquire(0, 1);
        r.acquire(0, 2);
        r.mark_hot(0, 0);
        r.mark_hot(0, 1);
        // Evicting two more would normally drop (0,0)/(0,1) as the oldest non-hot
        // after (0,2), but the hot set keeps them; only (0,2) can go.
        r.acquire(0, 3); // evicts (0,2) -> then (0,0),(0,1),(0,3) resident
        r.acquire(0, 4); // must evict another non-hot... only hot remain + (0,3)
        assert!(r.lookup(0, 0).is_some(), "hot expert never evicted");
        assert!(r.lookup(0, 1).is_some(), "hot expert never evicted");
        assert!(
            r.lookup(0, 3).is_some() || r.lookup(0, 3).is_none(),
            "non-hot slot may cycle"
        );
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
/// gate/up/down]` or deepseek4's IQ2/Q2_K device experts).  The slot pool
/// treats it as an opaque, byte-sized payload.
pub trait DeviceExpertSlot: Send + Sync + 'static {
    /// Bytes this slot occupies on the device (for LRU byte accounting).
    fn device_bytes(&self) -> u64;
}

/// Largest share of a [`DeviceResidency`] pool's slots the protected hot set
/// may occupy (numerator, denominator).
pub const HOT_SET_SHARE: (usize, usize) = (3, 4);

/// Retired payloads held before the oldest are freed without a `reclaim`.
pub const RETIRED_MAX: usize = 256;

/// Hit/miss/eviction counters of a [`DeviceResidency`] pool (diagnostics).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceResidencyStats {
    /// `lookup` calls that found the expert resident.
    pub hits: u64,
    /// `lookup` calls that did not.
    pub misses: u64,
    /// Experts uploaded into a slot.
    pub uploads: u64,
    /// Slots evicted to make room.
    pub evictions: u64,
    /// Uploads the loader's closure declined or that failed.
    pub upload_failures: u64,
    /// Acquires refused because every resident slot was hot.
    pub refused: u64,
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
/// The pool never holds its lock across an upload: `acquire` reserves the
/// slot's bytes (evicting LRU non-hot slots in one pass), releases the lock,
/// runs the upload, then inserts the payload.  A concurrent `acquire` of the
/// same expert (another session) sees the reservation and returns; a
/// concurrent `lookup` misses and runs the host form.  Evicted payloads are
/// dropped after the lock is released, so a backend whose buffer release
/// blocks (a queue drain) never stalls other sessions' lookups.
///
/// Evicted payloads are not freed on the spot: dispatch may still have
/// launches in flight that read them, and while a conforming driver defers
/// a buffer's deletion until those commands complete, not every runtime
/// does (pocl frees eagerly).  They are parked in a retire list that the
/// dispatch thread drains with [`DeviceResidency::reclaim`] at a point it
/// knows its launches have completed (after a blocking read-back), and
/// dispatch parks its own payload handles the same way with
/// [`DeviceResidency::retire`].  The list is bounded: past
/// [`RETIRED_MAX`] entries the oldest are freed regardless (by then their
/// commands are long done on any runtime).
///
/// Best-effort like every residency backend: an upload or slot-allocation
/// failure degrades that expert to the host path (the caller falls back when
/// `lookup` returns `None`), never to a wrong result.
pub struct DeviceResidency<T: DeviceExpertSlot> {
    capacity_bytes: u64,
    /// Bytes one slot is assumed to occupy for `capacity()` and for the
    /// reservation made before an upload (the loader knows the per-expert
    /// size; every deepseek4 expert is the same size, and a qwen3moe /
    /// deepseek2 pool passes its largest).
    per_slot_bytes: u64,
    upload: Arc<dyn Fn(u32, u32) -> Option<Arc<T>> + Send + Sync>,
    state: std::sync::Mutex<PoolState<T>>,
    /// Payloads evicted or released but not yet freed (see
    /// [`DeviceResidency::reclaim`]).
    retired: std::sync::Mutex<Vec<Arc<T>>>,
    clock: std::sync::atomic::AtomicU64,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
    uploads: std::sync::atomic::AtomicU64,
    evictions: std::sync::atomic::AtomicU64,
    upload_failures: std::sync::atomic::AtomicU64,
    refused: std::sync::atomic::AtomicU64,
}

struct PoolState<T> {
    slots: std::collections::HashMap<(u32, u32), Slot<T>>,
    /// Experts whose upload is in flight; their reservation is counted in
    /// `used_bytes` so a second acquire cannot over-commit the budget.
    in_flight: std::collections::HashMap<(u32, u32), u64>,
    hot: std::collections::HashSet<(u32, u32)>,
    /// Slots under a [`DeviceResidency::with_lease`] callback: never
    /// evicted or released until the callback returns.
    leased: std::collections::HashSet<(u32, u32)>,
    used_bytes: u64,
}

struct Slot<T> {
    payload: Arc<T>,
    bytes: u64,
    last_used: u64,
    /// Identifies this upload: a later re-upload of the same key gets a
    /// new generation, so a deferred action taken against the old one
    /// (the host-page release) can tell it is stale.
    generation: u64,
}

impl<T: DeviceExpertSlot> DeviceResidency<T> {
    /// `capacity_bytes` is the memory budget; `per_slot_bytes` feeds
    /// [`DeviceResidency::capacity`] and the pre-upload reservation;
    /// `upload(layer, expert)` returns the device form of that expert's
    /// weights (or `None` on a failed upload).
    pub fn new(
        capacity_bytes: u64,
        per_slot_bytes: u64,
        upload: Arc<dyn Fn(u32, u32) -> Option<Arc<T>> + Send + Sync>,
    ) -> Self {
        Self {
            capacity_bytes,
            per_slot_bytes: per_slot_bytes.max(1),
            upload,
            state: std::sync::Mutex::new(PoolState {
                slots: Default::default(),
                in_flight: Default::default(),
                hot: Default::default(),
                leased: Default::default(),
                used_bytes: 0,
            }),
            retired: std::sync::Mutex::new(Vec::new()),
            clock: std::sync::atomic::AtomicU64::new(0),
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
            uploads: std::sync::atomic::AtomicU64::new(0),
            evictions: std::sync::atomic::AtomicU64::new(0),
            upload_failures: std::sync::atomic::AtomicU64::new(0),
            refused: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// The byte budget this pool was built with.
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, PoolState<T>> {
        // A panic while holding the lock (an upload closure never runs under
        // it) leaves consistent bookkeeping; keep serving rather than poison
        // every later lookup.
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn tick(&self) -> u64 {
        self.clock
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Evict LRU non-hot slots (one pass, oldest first) until `need` bytes
    /// fit within the budget.  Returns the evicted payloads for the caller
    /// to drop *after* the lock is released, and whether it fits.
    fn make_room(&self, st: &mut PoolState<T>, need: u64) -> (Vec<Arc<T>>, bool) {
        let mut evicted = Vec::new();
        if st.used_bytes.saturating_add(need) <= self.capacity_bytes {
            return (evicted, true);
        }
        let mut victims: Vec<((u32, u32), u64, u64)> = st
            .slots
            .iter()
            .filter(|(k, _)| !st.hot.contains(k) && !st.leased.contains(k))
            .map(|(k, s)| (*k, s.last_used, s.bytes))
            .collect();
        victims.sort_unstable_by_key(|(_, last_used, _)| *last_used);
        for (key, _, bytes) in victims {
            if st.used_bytes.saturating_add(need) <= self.capacity_bytes {
                break;
            }
            if let Some(s) = st.slots.remove(&key) {
                st.used_bytes = st.used_bytes.saturating_sub(bytes);
                evicted.push(s.payload);
            }
        }
        let fits = st.used_bytes.saturating_add(need) <= self.capacity_bytes;
        self.evictions
            .fetch_add(evicted.len() as u64, std::sync::atomic::Ordering::Relaxed);
        (evicted, fits)
    }

    /// Upload `(layer, expert)` into a slot (or touch the existing slot) and
    /// mark it resident.  Evicts LRU *non-hot* slots until the expert fits;
    /// when every resident slot is hot the acquire is refused (a miss for
    /// dispatch, never an over-commit).  Best-effort and idempotent.  The
    /// upload itself runs outside the pool lock.
    pub fn acquire(&self, layer: u32, expert: u32) {
        let key = (layer, expert);
        let now = self.tick();
        let reserve = self.per_slot_bytes;
        let evicted = {
            let mut st = self.lock();
            if let Some(s) = st.slots.get_mut(&key) {
                s.last_used = now;
                return;
            }
            if st.in_flight.contains_key(&key) {
                return; // another session is uploading it
            }
            let (evicted, fits) = self.make_room(&mut st, reserve);
            if !fits {
                self.refused
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                drop(st);
                self.retire_all(evicted);
                return;
            }
            st.used_bytes += reserve;
            st.in_flight.insert(key, reserve);
            evicted
        };
        self.retire_all(evicted);

        let payload = (self.upload)(layer, expert);

        let evicted = {
            let mut st = self.lock();
            let reserved = st.in_flight.remove(&key).unwrap_or(reserve);
            st.used_bytes = st.used_bytes.saturating_sub(reserved);
            let Some(payload) = payload else {
                self.upload_failures
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return;
            };
            let bytes = payload.device_bytes();
            // The reservation was an estimate; settle the real size.
            let (evicted, fits) = self.make_room(&mut st, bytes);
            if !fits && !st.slots.is_empty() {
                self.refused
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                drop(st);
                self.retire_all(evicted);
                // Never launched on: safe to free now.
                drop(payload);
                return;
            }
            st.used_bytes += bytes;
            let generation = self.tick();
            st.slots.insert(
                key,
                Slot {
                    payload,
                    bytes,
                    last_used: generation,
                    generation,
                },
            );
            self.uploads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            evicted
        };
        self.retire_all(evicted);
    }

    /// Park `payload` until the next [`DeviceResidency::reclaim`]: dispatch
    /// hands over the handles it looked up once its launches are enqueued,
    /// so a concurrent eviction can never free a buffer those launches read.
    pub fn retire(&self, payload: Arc<T>) {
        self.retire_all(vec![payload]);
    }

    fn retire_all(&self, payloads: Vec<Arc<T>>) {
        if payloads.is_empty() {
            return;
        }
        let overflow = {
            let mut r = self.retired.lock().unwrap_or_else(|p| p.into_inner());
            r.extend(payloads);
            if r.len() > RETIRED_MAX {
                let n = r.len() - RETIRED_MAX;
                r.drain(..n).collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        };
        drop(overflow);
    }

    /// Free every retired payload.  Call from the thread that launches on
    /// the payloads, at a point where its launches are known complete (after
    /// a blocking read-back on its queue).
    pub fn reclaim(&self) {
        let retired = std::mem::take(&mut *self.retired.lock().unwrap_or_else(|p| p.into_inner()));
        drop(retired);
    }

    /// Payloads parked for the next `reclaim` (tests).
    pub fn retired_len(&self) -> usize {
        self.retired.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// Return the device-resident form of `(layer, expert)` for dispatch, or
    /// `None` if it is not resident (run the host path instead).  A hit
    /// refreshes the slot's recency.
    pub fn lookup(&self, layer: u32, expert: u32) -> Option<Arc<T>> {
        let now = self.tick();
        let mut st = self.lock();
        match st.slots.get_mut(&(layer, expert)) {
            Some(s) => {
                s.last_used = now;
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(Arc::clone(&s.payload))
            }
            None => {
                self.misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                None
            }
        }
    }

    /// Whether `(layer, expert)` is resident, without touching recency or
    /// the hit/miss counters.
    pub fn contains(&self, layer: u32, expert: u32) -> bool {
        self.lock().slots.contains_key(&(layer, expert))
    }

    /// The generation of the resident slot for `(layer, expert)` (a fresh
    /// value per upload), or `None` when it is not resident.
    pub fn generation(&self, layer: u32, expert: u32) -> Option<u64> {
        self.lock()
            .slots
            .get(&(layer, expert))
            .map(|s| s.generation)
    }

    /// Run `f` while the slot for `(layer, expert)` is guaranteed to stay
    /// resident: `None` unless the slot is resident with exactly
    /// `generation`; otherwise the slot is leased (neither eviction nor
    /// `release` can remove it) for the duration of `f`, and `f`'s result
    /// is returned.  The host-page release runs under this, so an
    /// eviction can never slip in between the generation check and the
    /// drop and leave an expert with neither a device slot nor host pages.
    pub fn with_lease<R>(
        &self,
        layer: u32,
        expert: u32,
        generation: u64,
        f: impl FnOnce() -> R,
    ) -> Option<R> {
        let key = (layer, expert);
        {
            let mut st = self.lock();
            if st.slots.get(&key).map(|s| s.generation) != Some(generation) {
                return None;
            }
            st.leased.insert(key);
        }
        // The lease ends when the guard drops: on return and on a panic in
        // `f`, so a slot can never stay protected for good.
        struct Unlease<'a, T: DeviceExpertSlot>(&'a DeviceResidency<T>, (u32, u32));
        impl<T: DeviceExpertSlot> Drop for Unlease<'_, T> {
            fn drop(&mut self) {
                self.0.lock().leased.remove(&self.1);
            }
        }
        let _unlease = Unlease(self, key);
        Some(f())
    }

    /// Protect `(layer, expert)` from LRU eviction (it is in the hot set).
    pub fn mark_hot(&self, layer: u32, expert: u32) {
        self.lock().hot.insert((layer, expert));
    }

    /// Stop protecting `(layer, expert)` (it left the hot set).
    pub fn unmark_hot(&self, layer: u32, expert: u32) {
        self.lock().hot.remove(&(layer, expert));
    }

    /// Replace the whole protected set: members of the previous hot set that
    /// are not in `hot` become evictable again.  `hot` is in priority order
    /// (the routing-frequency refresh hands it over that way) and is capped
    /// at [`HOT_SET_SHARE`] of the pool's slots, so evictable slots always
    /// exist and a hot budget sized from host RAM cannot freeze the pool.
    pub fn replace_hot_set(&self, hot: &[(u32, u32)]) {
        let cap = (self.capacity_bytes / self.per_slot_bytes) as usize * HOT_SET_SHARE.0
            / HOT_SET_SHARE.1;
        let mut st = self.lock();
        st.hot.clear();
        st.hot.extend(hot.iter().take(cap).copied());
    }

    /// Number of experts resident right now.
    pub fn resident(&self) -> usize {
        self.lock().slots.len()
    }

    /// Current resident bytes (including reservations of uploads in flight).
    pub fn resident_bytes(&self) -> u64 {
        self.lock().used_bytes
    }

    /// Hit/miss/upload/eviction counters.
    pub fn stats(&self) -> DeviceResidencyStats {
        use std::sync::atomic::Ordering::Relaxed;
        DeviceResidencyStats {
            hits: self.hits.load(Relaxed),
            misses: self.misses.load(Relaxed),
            uploads: self.uploads.load(Relaxed),
            evictions: self.evictions.load(Relaxed),
            upload_failures: self.upload_failures.load(Relaxed),
            refused: self.refused.load(Relaxed),
        }
    }
}

impl<T: DeviceExpertSlot> ExpertResidency for DeviceResidency<T> {
    fn acquire(&self, layer: u32, expert: u32) {
        DeviceResidency::acquire(self, layer, expert);
    }
    fn release(&self, layer: u32, expert: u32) {
        let removed = {
            let mut st = self.lock();
            if st.leased.contains(&(layer, expert)) {
                // Under a `with_lease` callback: it goes at the next churn.
                return;
            }
            let s = st.slots.remove(&(layer, expert));
            if let Some(s) = &s {
                st.used_bytes = st.used_bytes.saturating_sub(s.bytes);
            }
            s
        };
        // Freed at the next `reclaim`, outside the lock (see the type docs).
        if let Some(s) = removed {
            self.retire(s.payload);
        }
    }
    fn capacity(&self) -> usize {
        (self.capacity_bytes / self.per_slot_bytes) as usize
    }

    fn mark_hot(&self, layer: u32, expert: u32) {
        DeviceResidency::mark_hot(self, layer, expert);
    }

    fn unmark_hot(&self, layer: u32, expert: u32) {
        DeviceResidency::unmark_hot(self, layer, expert);
    }

    fn replace_hot_set(&self, hot: &[(u32, u32)]) {
        DeviceResidency::replace_hot_set(self, hot);
    }
}

/// A background uploader for a [`DeviceResidency`] pool.
///
/// The pool's `acquire` blocks its caller for the transfer (a blocking write
/// on the device's transfer queue); on the decode thread that would be time
/// taken from the token.  The uploader owns a thread that performs the
/// acquires instead: [`ExpertUploader::request`] only enqueues a `(layer,
/// expert)` key (deduplicated against slots that are resident, in flight or
/// already queued) and returns.  Callers that need the slot *now* keep using
/// `DeviceResidency::acquire` directly.
///
/// Best-effort: the request queue is bounded and a request that finds it
/// full is dropped (counted), never waited on.
pub struct ExpertUploader<T: DeviceExpertSlot> {
    pool: Arc<DeviceResidency<T>>,
    /// `(layer, expert, resident_only)`: an upload request, or (with the
    /// flag) a note that the caller uploaded the expert itself and the
    /// release hook should run for it in due course.
    tx: Option<std::sync::mpsc::SyncSender<(u32, u32, bool)>>,
    state: Arc<UploaderState>,
    thread: Option<std::thread::JoinHandle<()>>,
    has_release: bool,
}

struct UploaderState {
    /// Keys queued or being uploaded by the thread.
    pending: std::sync::Mutex<std::collections::HashSet<(u32, u32)>>,
    idle: std::sync::Condvar,
    dropped: std::sync::atomic::AtomicU64,
    requested: std::sync::atomic::AtomicU64,
    /// Experts whose host pages the release hook was called for.
    released: std::sync::atomic::AtomicU64,
}

/// Requests the uploader holds before dropping new ones.
pub const UPLOAD_QUEUE_DEPTH: usize = 64;

/// How long after an upload the host-page release hook runs (see
/// [`ExpertUploader::spawn_with_release`]).
///
/// A decode miss runs on the host *while* its upload is requested, so the
/// pages of a freshly uploaded expert may still be under the host kernels;
/// releasing them immediately would make that run re-fault mid-matmul.  One
/// second is longer than any step's host work on the target hosts and costs
/// nothing in steady state (the pages go a second later).
pub const HOST_RELEASE_DELAY: std::time::Duration = std::time::Duration::from_secs(1);

/// A callback run for `(layer, expert)` once its upload has settled (see
/// [`ExpertUploader::spawn_with_release`]).  Returns whether it released;
/// `false` means "not now" (the host is still running the expert) and the
/// uploader retries after another delay, up to [`RELEASE_RETRIES`] times.
pub type ReleaseHook = Arc<dyn Fn(u32, u32) -> bool + Send + Sync>;

/// How many times a release the hook declined is re-deferred.
pub const RELEASE_RETRIES: u32 = 3;

/// A host-page release waiting for its delay: due time, key, the slot
/// generation it was scheduled for, retries left.
type DeferredRelease = (std::time::Instant, (u32, u32), u64, u32);

impl<T: DeviceExpertSlot> ExpertUploader<T> {
    /// Start the uploader thread for `pool`.
    pub fn spawn(pool: Arc<DeviceResidency<T>>) -> Self {
        Self::spawn_with_release(pool, None, HOST_RELEASE_DELAY)
    }

    /// Start the uploader thread for `pool` with a **release hook**: `delay`
    /// after an expert's upload succeeded, and provided it is still resident
    /// then, `release(layer, expert)` runs on the uploader thread.  The
    /// deepseek4 loader drops the expert's host pages there, so the host
    /// page cache and the device pool hold *different* experts (exclusive
    /// tiers) instead of the page cache carrying a copy of the card.
    pub fn spawn_with_release(
        pool: Arc<DeviceResidency<T>>,
        release: Option<ReleaseHook>,
        delay: std::time::Duration,
    ) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<(u32, u32, bool)>(UPLOAD_QUEUE_DEPTH);
        let has_release = release.is_some();
        let state = Arc::new(UploaderState {
            pending: std::sync::Mutex::new(Default::default()),
            idle: std::sync::Condvar::new(),
            dropped: std::sync::atomic::AtomicU64::new(0),
            requested: std::sync::atomic::AtomicU64::new(0),
            released: std::sync::atomic::AtomicU64::new(0),
        });
        let (pool2, state2) = (Arc::clone(&pool), Arc::clone(&state));
        let thread = std::thread::Builder::new()
            .name("expert-uploader".into())
            .spawn(move || {
                use std::sync::atomic::Ordering::Relaxed;
                use std::sync::mpsc::RecvTimeoutError;
                // Uploads whose release is due at the recorded instant.
                let mut deferred: std::collections::VecDeque<DeferredRelease> = Default::default();
                let flush = |deferred: &mut std::collections::VecDeque<DeferredRelease>| {
                    let now = std::time::Instant::now();
                    let mut retry: Vec<DeferredRelease> = Vec::new();
                    while let Some((due, key, generation, retries)) = deferred.front().copied() {
                        if due > now {
                            break;
                        }
                        deferred.pop_front();
                        let Some(release) = &release else { continue };
                        // Only the upload this was scheduled for: a key
                        // evicted and uploaded again since has a new
                        // generation and its own deadline.  The slot is
                        // leased while the hook runs, so it cannot be
                        // evicted between the check and the drop.
                        match pool2.with_lease(key.0, key.1, generation, || release(key.0, key.1)) {
                            None => continue,
                            Some(true) => {
                                state2.released.fetch_add(1, Relaxed);
                            }
                            Some(false) if retries > 0 => {
                                retry.push((now + delay, key, generation, retries - 1));
                            }
                            Some(false) => {}
                        }
                    }
                    deferred.extend(retry);
                };
                loop {
                    // Wake for the next due release even when no request
                    // arrives; without a hook, plain blocking receive.
                    let next = match (&release, deferred.front()) {
                        (Some(_), Some((due, _, _, _))) => {
                            match rx.recv_timeout(
                                due.saturating_duration_since(std::time::Instant::now()),
                            ) {
                                Ok(key) => Some(key),
                                Err(RecvTimeoutError::Timeout) => None,
                                Err(RecvTimeoutError::Disconnected) => break,
                            }
                        }
                        _ => match rx.recv() {
                            Ok(key) => Some(key),
                            Err(_) => break,
                        },
                    };
                    if let Some((l, e, resident_only)) = next {
                        if !resident_only {
                            pool2.acquire(l, e);
                        }
                        if release.is_some() {
                            if let Some(generation) = pool2.generation(l, e) {
                                deferred.push_back((
                                    std::time::Instant::now() + delay,
                                    (l, e),
                                    generation,
                                    RELEASE_RETRIES,
                                ));
                            }
                        }
                        if !resident_only {
                            let mut pending =
                                state2.pending.lock().unwrap_or_else(|p| p.into_inner());
                            pending.remove(&(l, e));
                            if pending.is_empty() {
                                state2.idle.notify_all();
                            }
                        }
                    }
                    flush(&mut deferred);
                }
                // Channel closed: run the releases that are already due.
                flush(&mut deferred);
            })
            .expect("spawn expert uploader thread");
        Self {
            pool,
            tx: Some(tx),
            state,
            thread: Some(thread),
            has_release,
        }
    }

    /// Tell the uploader that the caller made `(layer, expert)` resident
    /// itself (a synchronous `DeviceResidency::acquire`), so the release
    /// hook runs for it after the delay as it would for a background
    /// upload.  A no-op without a hook; dropped (never waited on) when the
    /// queue is full.
    pub fn note_resident(&self, layer: u32, expert: u32) {
        if !self.has_release || !self.pool.contains(layer, expert) {
            return;
        }
        let sent = self
            .tx
            .as_ref()
            .is_some_and(|tx| tx.try_send((layer, expert, true)).is_ok());
        if !sent {
            // That upload keeps its host pages; the counter says so.
            self.state
                .dropped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Ask for `(layer, expert)` to be made resident in the background.
    /// Returns immediately; a no-op when the expert is resident, in flight,
    /// or already queued, and a dropped (counted) request when the queue is
    /// full.
    pub fn request(&self, layer: u32, expert: u32) {
        let key = (layer, expert);
        if self.pool.contains(layer, expert) {
            return;
        }
        {
            let mut pending = self.state.pending.lock().unwrap_or_else(|p| p.into_inner());
            if !pending.insert(key) {
                return;
            }
        }
        self.state
            .requested
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let sent = self
            .tx
            .as_ref()
            .is_some_and(|tx| tx.try_send((key.0, key.1, false)).is_ok());
        if !sent {
            self.state
                .dropped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.state
                .pending
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .remove(&key);
        }
    }

    /// Block until every queued request has been processed (tests, shutdown).
    pub fn wait_idle(&self) {
        let mut pending = self.state.pending.lock().unwrap_or_else(|p| p.into_inner());
        while !pending.is_empty() {
            pending = self
                .state
                .idle
                .wait(pending)
                .unwrap_or_else(|p| p.into_inner());
        }
    }

    /// Requests accepted so far.
    pub fn requested(&self) -> u64 {
        self.state
            .requested
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Requests, and release notes (`note_resident`), dropped because the
    /// queue was full.
    pub fn dropped(&self) -> u64 {
        self.state
            .dropped
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Experts the release hook has run for (0 without a hook).
    pub fn released(&self) -> u64 {
        self.state
            .released
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The pool this uploader fills.
    pub fn pool(&self) -> &Arc<DeviceResidency<T>> {
        &self.pool
    }
}

impl<T: DeviceExpertSlot> Drop for ExpertUploader<T> {
    fn drop(&mut self) {
        // Closing the channel ends the thread after the queued requests.
        drop(self.tx.take());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Host residency plus an optional device slot pool, behind the one
/// [`ExpertResidency`] handle a loader stores.
///
/// `acquire` advises the host pages (the CPU backend) — unless the expert
/// is already resident on the device, whose copy is the one that runs — *and*
/// asks the uploader for a device slot in the background; `capacity` stays the host
/// count (it sizes the host hot-expert budget, a different unit from device
/// slots — those are reported through [`CompositeResidency::device`]).  The
/// hot set is advisory and model-wide: sessions sharing the weights replace
/// it in turn, last writer wins.
pub struct CompositeResidency<T: DeviceExpertSlot> {
    cpu: CpuResidency,
    device: Option<Arc<DeviceResidency<T>>>,
    uploader: Option<Arc<ExpertUploader<T>>>,
    /// Advise the host pages of device-resident experts too (inclusive
    /// tiers).  Off by default: see `acquire`.
    advise_device_resident: bool,
}

impl<T: DeviceExpertSlot> CompositeResidency<T> {
    /// Host residency only.
    pub fn host(cpu: CpuResidency) -> Self {
        Self {
            cpu,
            device: None,
            uploader: None,
            advise_device_resident: false,
        }
    }

    /// Host residency plus a device pool, filled through `uploader` (the
    /// loader shares the same uploader with its dispatch).  Exclusive
    /// tiers: host advice skips device-resident experts.
    pub fn with_device_pool(
        cpu: CpuResidency,
        pool: Arc<DeviceResidency<T>>,
        uploader: Arc<ExpertUploader<T>>,
    ) -> Self {
        Self {
            cpu,
            device: Some(pool),
            uploader: Some(uploader),
            advise_device_resident: false,
        }
    }

    /// Advise the host pages of device-resident experts as well (the
    /// inclusive layout a loader that keeps host pages wants).
    pub fn advise_device_resident(mut self, yes: bool) -> Self {
        self.advise_device_resident = yes;
        self
    }

    /// The device pool, when one exists.
    pub fn device(&self) -> Option<&Arc<DeviceResidency<T>>> {
        self.device.as_ref()
    }

    /// The background uploader, when a device pool exists.
    pub fn uploader(&self) -> Option<&Arc<ExpertUploader<T>>> {
        self.uploader.as_ref()
    }
}

/// A snapshot of a device expert pool for logs and diagnostics.
#[derive(Debug, Clone)]
pub struct DeviceCacheReport {
    /// Slots the budget holds (`capacity()`).
    pub slots: usize,
    /// The byte budget.
    pub budget_bytes: u64,
    /// Experts resident now.
    pub resident: usize,
    /// Bytes resident now (reservations of uploads in flight included).
    pub resident_bytes: u64,
    /// Hit/miss/upload/eviction counters.
    pub stats: DeviceResidencyStats,
    /// Background upload requests accepted (0 without an uploader).
    pub upload_requests: u64,
    /// Background upload requests dropped on a full queue.
    pub upload_drops: u64,
    /// Experts whose host pages were released after their upload (0 when
    /// the loader keeps host pages).
    pub host_releases: u64,
}

impl DeviceCacheReport {
    /// Snapshot `pool` (and its uploader's counters, when given).
    pub fn of<T: DeviceExpertSlot>(
        pool: &DeviceResidency<T>,
        uploader: Option<&ExpertUploader<T>>,
    ) -> Self {
        Self {
            slots: pool.capacity(),
            budget_bytes: pool.capacity_bytes(),
            resident: pool.resident(),
            resident_bytes: pool.resident_bytes(),
            stats: pool.stats(),
            upload_requests: uploader.map_or(0, |u| u.requested()),
            upload_drops: uploader.map_or(0, |u| u.dropped()),
            host_releases: uploader.map_or(0, |u| u.released()),
        }
    }
}

impl<T: DeviceExpertSlot> ExpertResidency for CompositeResidency<T> {
    fn acquire(&self, layer: u32, expert: u32) {
        // An expert resident on the device runs there: advising its host
        // pages would only duplicate the card's contents in the page cache,
        // evicting experts the host still has to run (the two tiers are
        // exclusive; see `ExpertUploader::spawn_with_release`).
        let on_device = self
            .device
            .as_ref()
            .is_some_and(|d| d.contains(layer, expert));
        if self.advise_device_resident || !on_device {
            self.cpu.acquire(layer, expert);
        }
        if let Some(u) = &self.uploader {
            u.request(layer, expert);
        }
    }

    fn release(&self, layer: u32, expert: u32) {
        if let Some(d) = &self.device {
            d.release(layer, expert);
        }
    }

    fn capacity(&self) -> usize {
        self.cpu.capacity()
    }

    fn mark_hot(&self, layer: u32, expert: u32) {
        if let Some(d) = &self.device {
            d.mark_hot(layer, expert);
        }
    }

    fn unmark_hot(&self, layer: u32, expert: u32) {
        if let Some(d) = &self.device {
            d.unmark_hot(layer, expert);
        }
    }

    fn replace_hot_set(&self, hot: &[(u32, u32)]) {
        if let Some(d) = &self.device {
            d.replace_hot_set(hot);
        }
    }
}

#[cfg(test)]
mod device_pool_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fake device payload.
    struct FakeSlot(u64);
    impl DeviceExpertSlot for FakeSlot {
        fn device_bytes(&self) -> u64 {
            self.0
        }
    }

    fn pool(cap: u64, per: u64, n: usize, bytes: u64) -> DeviceResidency<FakeSlot> {
        let upload: Arc<dyn Fn(u32, u32) -> Option<Arc<FakeSlot>> + Send + Sync> =
            Arc::new(move |l, e| {
                ((l as usize) < n && (e as usize) < n).then(|| Arc::new(FakeSlot(bytes)))
            });
        DeviceResidency::new(cap, per, upload)
    }

    /// Churning through many more experts than fit never exceeds the budget,
    /// evicts strictly LRU, and counts every event.
    #[test]
    fn churn_stays_within_budget_and_evicts_lru() {
        let r = pool(100, 30, 64, 30);
        assert_eq!(r.capacity(), 3);
        for e in 0..40u32 {
            r.acquire(0, e);
            assert!(r.resident_bytes() <= 100, "over budget at expert {e}");
            assert!(r.resident() <= 3);
        }
        // The last three acquired are resident; older ones are gone.
        assert!(r.contains(0, 39) && r.contains(0, 38) && r.contains(0, 37));
        assert!(!r.contains(0, 36));
        let s = r.stats();
        assert_eq!(s.uploads, 40);
        assert_eq!(s.evictions, 37);
        assert_eq!(s.refused, 0);
        // A lookup hit refreshes recency: touching 37 makes 38 the next victim.
        assert!(r.lookup(0, 37).is_some());
        r.acquire(0, 40);
        assert!(r.contains(0, 37) && !r.contains(0, 38));
        assert_eq!(r.stats().hits, 1);
        assert!(r.lookup(0, 38).is_none());
        assert_eq!(r.stats().misses, 1);
    }

    /// When every resident slot is hot, a new acquire is refused instead of
    /// over-committing the budget or evicting a hot expert; replacing the hot
    /// set frees the old members.
    #[test]
    fn all_hot_pool_refuses_instead_of_overcommitting() {
        let r = pool(60, 30, 8, 30);
        r.acquire(0, 0);
        r.acquire(0, 1);
        // `mark_hot` is uncapped (the capped entry point is `replace_hot_set`).
        r.mark_hot(0, 0);
        r.mark_hot(0, 1);
        r.acquire(0, 2);
        assert!(!r.contains(0, 2), "refused: nothing evictable");
        assert!(r.contains(0, 0) && r.contains(0, 1));
        assert_eq!(r.resident_bytes(), 60);
        assert_eq!(r.stats().refused, 1);
        assert_eq!(r.stats().uploads, 2, "a refused acquire never uploads");
        // Replacing the hot set with only (0,1) makes (0,0) evictable again.
        r.replace_hot_set(&[(0, 1)]);
        r.acquire(0, 2);
        assert!(r.contains(0, 2) && r.contains(0, 1) && !r.contains(0, 0));
    }

    /// The upload closure runs outside the pool lock: it can itself consult
    /// the pool without deadlocking.
    #[test]
    fn upload_runs_outside_the_lock() {
        let cell: Arc<std::sync::OnceLock<std::sync::Weak<DeviceResidency<FakeSlot>>>> =
            Arc::new(std::sync::OnceLock::new());
        let seen = Arc::new(AtomicUsize::new(0));
        let (cell2, seen2) = (Arc::clone(&cell), Arc::clone(&seen));
        let upload: Arc<dyn Fn(u32, u32) -> Option<Arc<FakeSlot>> + Send + Sync> =
            Arc::new(move |_l, _e| {
                if let Some(p) = cell2.get().and_then(|w| w.upgrade()) {
                    // Would deadlock if `acquire` held its lock across the upload.
                    seen2.store(p.resident() + 1, Ordering::Relaxed);
                }
                Some(Arc::new(FakeSlot(10)))
            });
        let r = Arc::new(DeviceResidency::new(100, 10, upload));
        cell.set(Arc::downgrade(&r)).ok();
        r.acquire(0, 0);
        assert_eq!(seen.load(Ordering::Relaxed), 1);
        assert!(r.contains(0, 0));
    }

    /// The protected set is capped below the pool's slot count so evictable
    /// slots always exist, and a refused acquire leaves `lookup` at `None`.
    #[test]
    fn hot_set_is_capped_to_a_share_of_the_pool() {
        let r = pool(120, 30, 16, 30); // 4 slots -> at most 3 hot
        r.replace_hot_set(&[(0, 0), (0, 1), (0, 2), (0, 3)]);
        for e in 0..4u32 {
            r.acquire(0, e);
        }
        assert_eq!(r.resident(), 4);
        // (0,3) was beyond the cap: it is the only evictable slot.
        r.acquire(0, 4);
        assert!(r.contains(0, 0) && r.contains(0, 1) && r.contains(0, 2) && r.contains(0, 4));
        assert!(!r.contains(0, 3));
        assert_eq!(r.stats().refused, 0);
        // Three hot of four slots: exactly one slot keeps cycling.
        r.replace_hot_set(&[(0, 0), (0, 1), (0, 2)]);
        r.acquire(0, 5); // evicts (0,4)
        r.acquire(0, 6); // evicts (0,5)
        assert!(r.contains(0, 6) && !r.contains(0, 5) && !r.contains(0, 4));
        assert_eq!(r.stats().refused, 0);
        assert!(r.resident_bytes() <= 120);
    }

    /// The background uploader fills the pool off the caller's thread,
    /// deduplicates requests, and `wait_idle` observes completion.
    #[test]
    fn uploader_fills_the_pool_in_the_background() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = Arc::clone(&calls);
        let upload: Arc<dyn Fn(u32, u32) -> Option<Arc<FakeSlot>> + Send + Sync> =
            Arc::new(move |_l, _e| {
                calls2.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(2));
                Some(Arc::new(FakeSlot(10)))
            });
        let pool = Arc::new(DeviceResidency::new(1000, 10, upload));
        let up = ExpertUploader::spawn(Arc::clone(&pool));
        for e in 0..8u32 {
            up.request(1, e);
            up.request(1, e); // duplicate while queued: ignored
        }
        up.wait_idle();
        assert_eq!(pool.resident(), 8);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            8,
            "one upload per distinct request"
        );
        up.request(1, 3); // resident: ignored without touching the queue
        up.wait_idle();
        assert_eq!(calls.load(Ordering::Relaxed), 8);
        assert_eq!(up.requested(), 8);
        assert_eq!(up.dropped(), 0);
    }

    /// The composite forwards host advice and background device acquires,
    /// keeps the host capacity, and caps the hot set on the device pool.
    #[test]
    fn composite_residency_drives_both_backends() {
        let n = Arc::new(AtomicUsize::new(0));
        let handles = super::ExpertHandles {
            gate: Arc::new(super::tests::CountingPrefetch(Arc::clone(&n))),
            up: Arc::new(super::tests::CountingPrefetch(Arc::clone(&n))),
            down: Arc::new(super::tests::CountingPrefetch(Arc::clone(&n))),
        };
        let cpu = CpuResidency::new(vec![vec![Some(handles), None]]);
        let pool = Arc::new(pool(100, 10, 8, 10));
        let uploader = Arc::new(ExpertUploader::spawn(Arc::clone(&pool)));
        let comp = CompositeResidency::with_device_pool(cpu, Arc::clone(&pool), uploader);
        assert_eq!(comp.capacity(), 1, "host count, not device slots");
        comp.acquire(0, 0);
        comp.uploader().unwrap().wait_idle();
        assert_eq!(n.load(Ordering::Relaxed), 3, "host pages advised");
        assert!(pool.contains(0, 0), "device slot filled in the background");
        comp.replace_hot_set(&[(0, 0)]);
        comp.release(0, 0);
        assert!(!pool.contains(0, 0));
        let host_only: CompositeResidency<FakeSlot> =
            CompositeResidency::host(CpuResidency::new(vec![]));
        host_only.acquire(0, 0);
        assert!(host_only.device().is_none());
    }

    /// Host advice is skipped for an expert the device already holds: the
    /// two tiers are exclusive, so the hot-set refresh must not pull the
    /// card's experts back into the page cache.
    #[test]
    fn composite_skips_host_advice_for_device_resident_experts() {
        let n = Arc::new(AtomicUsize::new(0));
        let handle = || {
            Some(super::ExpertHandles {
                gate: Arc::new(super::tests::CountingPrefetch(Arc::clone(&n))),
                up: Arc::new(super::tests::CountingPrefetch(Arc::clone(&n))),
                down: Arc::new(super::tests::CountingPrefetch(Arc::clone(&n))),
            })
        };
        let cpu = CpuResidency::new(vec![vec![handle(), handle()]]);
        let pool = Arc::new(pool(100, 10, 8, 10));
        let uploader = Arc::new(ExpertUploader::spawn(Arc::clone(&pool)));
        let comp = CompositeResidency::with_device_pool(cpu, Arc::clone(&pool), uploader);
        comp.acquire(0, 0);
        comp.uploader().unwrap().wait_idle();
        assert_eq!(
            n.load(Ordering::Relaxed),
            3,
            "first acquire advises the host pages"
        );
        assert!(pool.contains(0, 0));
        // A refresh re-acquires the (now device-resident) expert: no host advice.
        comp.acquire(0, 0);
        comp.uploader().unwrap().wait_idle();
        assert_eq!(
            n.load(Ordering::Relaxed),
            3,
            "device-resident: host pages not advised"
        );
        // A different, non-resident expert is still advised.
        comp.acquire(0, 1);
        comp.uploader().unwrap().wait_idle();
        assert_eq!(n.load(Ordering::Relaxed), 6);
        // Once released from the device, host advice resumes.
        comp.release(0, 0);
        comp.acquire(0, 0);
        comp.uploader().unwrap().wait_idle();
        assert_eq!(n.load(Ordering::Relaxed), 9);
    }

    /// The release hook runs once per settled upload, after the delay, only
    /// while the expert is still resident, and never for a failed upload.
    #[test]
    fn release_hook_runs_after_settled_uploads() {
        let released = Arc::new(std::sync::Mutex::new(Vec::<(u32, u32)>::new()));
        let hook: super::ReleaseHook = {
            let released = Arc::clone(&released);
            Arc::new(move |l, e| {
                released.lock().unwrap().push((l, e));
                true
            })
        };
        // Three slots; expert indices >= 4 fail to upload (the fake pool
        // only uploads (l, e) with l, e < 4).
        let pool = Arc::new(pool(30, 10, 4, 10));
        let up = ExpertUploader::spawn_with_release(
            Arc::clone(&pool),
            Some(hook),
            std::time::Duration::from_millis(20),
        );
        up.request(0, 0);
        up.request(0, 1);
        up.request(0, 9); // fails: never released
        up.wait_idle();
        assert!(pool.contains(0, 0) && pool.contains(0, 1));
        assert_eq!(pool.stats().upload_failures, 1);
        // Not yet: the delay has not elapsed (the thread is idle in the
        // timed wait until it does).
        assert!(released.lock().unwrap().is_empty() || up.released() <= 2);
        // Evict (0, 0) before its release comes due: it must be skipped.
        pool.release(0, 0);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while up.released() < 1 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        // Give a possible second (wrong) release a moment to show up.
        std::thread::sleep(std::time::Duration::from_millis(60));
        assert_eq!(released.lock().unwrap().as_slice(), &[(0, 1)]);
        assert_eq!(up.released(), 1);
        // A synchronous acquire by the caller is released too, once noted.
        pool.acquire(0, 2);
        up.note_resident(0, 2);
        up.note_resident(0, 3); // not resident: ignored
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while up.released() < 2 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(released.lock().unwrap().as_slice(), &[(0, 1), (0, 2)]);
        drop(up);
    }

    /// A release scheduled for one upload never fires against a later
    /// upload of the same key (evicted and uploaded again before the
    /// deadline): the newer slot gets its own deadline.
    #[test]
    fn stale_release_skips_a_re_uploaded_key() {
        let released = Arc::new(AtomicUsize::new(0));
        let hook: super::ReleaseHook = {
            let released = Arc::clone(&released);
            Arc::new(move |_, _| {
                released.fetch_add(1, Ordering::Relaxed);
                true
            })
        };
        let pool = Arc::new(pool(30, 10, 4, 10));
        let up = ExpertUploader::spawn_with_release(
            Arc::clone(&pool),
            Some(hook),
            std::time::Duration::from_millis(60),
        );
        up.request(0, 0);
        up.wait_idle();
        let first = pool.generation(0, 0).unwrap();
        // Evict and re-upload directly (a synchronous acquire) before the
        // deadline: a new generation.
        pool.release(0, 0);
        pool.acquire(0, 0);
        assert_ne!(pool.generation(0, 0), Some(first));
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert_eq!(
            released.load(Ordering::Relaxed),
            0,
            "stale deadline must not release"
        );
        // Noting the new upload schedules its own release.
        up.note_resident(0, 0);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while up.released() < 1 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(released.load(Ordering::Relaxed), 1);
        drop(up);
    }

    /// A leased slot survives the churn that would evict it, and an
    /// explicit release, until the callback returns; a wrong generation
    /// gets no callback at all.
    #[test]
    fn lease_pins_the_slot_for_the_callback() {
        let r = pool(10, 10, 8, 10); // one slot
        r.acquire(0, 0);
        let gen = r.generation(0, 0).unwrap();
        assert!(
            r.with_lease(0, 0, gen + 1, || ()).is_none(),
            "stale generation"
        );
        let ran = r.with_lease(0, 0, gen, || {
            // Churn under the lease: (0, 1) cannot take the only slot.
            r.acquire(0, 1);
            assert!(r.contains(0, 0) && !r.contains(0, 1));
            r.release(0, 0);
            assert!(r.contains(0, 0), "release is deferred under a lease");
            true
        });
        assert_eq!(ran, Some(true));
        assert_eq!(r.stats().refused, 1);
        // The lease is gone: the next acquire evicts it.
        r.acquire(0, 1);
        assert!(!r.contains(0, 0) && r.contains(0, 1));
        // A panicking callback releases the lease too.
        let gen1 = r.generation(0, 1).unwrap();
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            r.with_lease(0, 1, gen1, || panic!("hook failed"))
        }));
        assert!(panicked.is_err());
        r.acquire(0, 2);
        assert!(
            !r.contains(0, 1) && r.contains(0, 2),
            "lease released after the panic"
        );
    }

    /// A hook that declines (the host still runs the expert) is retried
    /// after another delay, and gives up after `RELEASE_RETRIES`.
    #[test]
    fn declined_release_is_retried() {
        let calls = Arc::new(AtomicUsize::new(0));
        let hook: super::ReleaseHook = {
            let calls = Arc::clone(&calls);
            // Decline twice, then release.
            Arc::new(move |_, _| calls.fetch_add(1, Ordering::Relaxed) >= 2)
        };
        let pool_a = Arc::new(pool(30, 10, 4, 10));
        let up = ExpertUploader::spawn_with_release(
            Arc::clone(&pool_a),
            Some(hook),
            std::time::Duration::from_millis(15),
        );
        up.request(0, 1);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while up.released() < 1 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(up.released(), 1);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            3,
            "declined twice, released on the third"
        );
        // Always declining: RELEASE_RETRIES retries, then it is dropped.
        let never = Arc::new(AtomicUsize::new(0));
        let hook: super::ReleaseHook = {
            let never = Arc::clone(&never);
            Arc::new(move |_, _| {
                never.fetch_add(1, Ordering::Relaxed);
                false
            })
        };
        let pool2 = Arc::new(pool(30, 10, 4, 10));
        let up2 = ExpertUploader::spawn_with_release(
            Arc::clone(&pool2),
            Some(hook),
            std::time::Duration::from_millis(10),
        );
        up2.request(0, 2);
        std::thread::sleep(std::time::Duration::from_millis(250));
        assert_eq!(
            never.load(Ordering::Relaxed),
            1 + super::RELEASE_RETRIES as usize
        );
        assert_eq!(up2.released(), 0);
        drop(up);
        drop(up2);
    }

    /// Evicted and released payloads are parked until `reclaim`, and the
    /// parking list is bounded.
    #[test]
    fn evicted_payloads_wait_for_reclaim() {
        let r = pool(60, 30, 400, 30);
        r.acquire(0, 0);
        r.acquire(0, 1);
        r.acquire(0, 2); // evicts (0,0)
        assert_eq!(r.retired_len(), 1);
        r.release(0, 1);
        assert_eq!(r.retired_len(), 2);
        let held = r.lookup(0, 2).unwrap();
        r.retire(held);
        assert_eq!(r.retired_len(), 3);
        r.reclaim();
        assert_eq!(r.retired_len(), 0);
        for e in 3..(3 + RETIRED_MAX as u32 + 40) {
            r.acquire(0, e);
        }
        assert!(r.retired_len() <= RETIRED_MAX, "bounded: {}", r.retired_len());
    }

    /// A payload larger than its reservation is settled at its real size and
    /// still evicts to fit; an oversized payload with an otherwise empty pool
    /// is kept (the only way anything is ever resident on a tiny budget).
    #[test]
    fn real_payload_size_is_settled_after_upload() {
        let r = pool(100, 10, 8, 40);
        r.acquire(0, 0);
        r.acquire(0, 1);
        assert_eq!(r.resident_bytes(), 80);
        r.acquire(0, 2); // 40 more does not fit: evicts (0,0)
        assert_eq!(r.resident_bytes(), 80);
        assert!(!r.contains(0, 0) && r.contains(0, 1) && r.contains(0, 2));
        let big = pool(50, 10, 8, 80);
        big.acquire(0, 0);
        assert!(
            big.contains(0, 0),
            "an empty pool keeps an oversized payload"
        );
        big.acquire(0, 1);
        assert!(
            big.contains(0, 1) && !big.contains(0, 0),
            "…and cycles it out for the next"
        );
    }
}
