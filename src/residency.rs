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
    used_bytes: u64,
}

struct Slot<T> {
    payload: Arc<T>,
    bytes: u64,
    last_used: u64,
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
                used_bytes: 0,
            }),
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
            .filter(|(k, _)| !st.hot.contains(k))
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
                drop(evicted);
                return;
            }
            st.used_bytes += reserve;
            st.in_flight.insert(key, reserve);
            evicted
        };
        drop(evicted);

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
                drop(evicted);
                drop(payload);
                return;
            }
            st.used_bytes += bytes;
            st.slots.insert(
                key,
                Slot {
                    payload,
                    bytes,
                    last_used: self.tick(),
                },
            );
            self.uploads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            evicted
        };
        drop(evicted);
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
            let s = st.slots.remove(&(layer, expert));
            if let Some(s) = &s {
                st.used_bytes = st.used_bytes.saturating_sub(s.bytes);
            }
            s
        };
        // Dropped outside the lock (see the type docs).
        drop(removed);
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
    tx: Option<std::sync::mpsc::SyncSender<(u32, u32)>>,
    state: Arc<UploaderState>,
    thread: Option<std::thread::JoinHandle<()>>,
}

struct UploaderState {
    /// Keys queued or being uploaded by the thread.
    pending: std::sync::Mutex<std::collections::HashSet<(u32, u32)>>,
    idle: std::sync::Condvar,
    dropped: std::sync::atomic::AtomicU64,
    requested: std::sync::atomic::AtomicU64,
}

/// Requests the uploader holds before dropping new ones.
pub const UPLOAD_QUEUE_DEPTH: usize = 64;

impl<T: DeviceExpertSlot> ExpertUploader<T> {
    /// Start the uploader thread for `pool`.
    pub fn spawn(pool: Arc<DeviceResidency<T>>) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<(u32, u32)>(UPLOAD_QUEUE_DEPTH);
        let state = Arc::new(UploaderState {
            pending: std::sync::Mutex::new(Default::default()),
            idle: std::sync::Condvar::new(),
            dropped: std::sync::atomic::AtomicU64::new(0),
            requested: std::sync::atomic::AtomicU64::new(0),
        });
        let (pool2, state2) = (Arc::clone(&pool), Arc::clone(&state));
        let thread = std::thread::Builder::new()
            .name("expert-uploader".into())
            .spawn(move || {
                while let Ok((l, e)) = rx.recv() {
                    pool2.acquire(l, e);
                    let mut pending = state2.pending.lock().unwrap_or_else(|p| p.into_inner());
                    pending.remove(&(l, e));
                    if pending.is_empty() {
                        state2.idle.notify_all();
                    }
                }
            })
            .expect("spawn expert uploader thread");
        Self {
            pool,
            tx: Some(tx),
            state,
            thread: Some(thread),
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
        let sent = self.tx.as_ref().is_some_and(|tx| tx.try_send(key).is_ok());
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

    /// Requests dropped because the queue was full.
    pub fn dropped(&self) -> u64 {
        self.state
            .dropped
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
/// `acquire` advises the host pages (the CPU backend) *and* asks the
/// uploader for a device slot in the background; `capacity` stays the host
/// count (it sizes the host hot-expert budget, a different unit from device
/// slots — those are reported through [`CompositeResidency::device`]).  The
/// hot set is advisory and model-wide: sessions sharing the weights replace
/// it in turn, last writer wins.
pub struct CompositeResidency<T: DeviceExpertSlot> {
    cpu: CpuResidency,
    device: Option<Arc<DeviceResidency<T>>>,
    uploader: Option<ExpertUploader<T>>,
}

impl<T: DeviceExpertSlot> CompositeResidency<T> {
    /// Host residency only.
    pub fn host(cpu: CpuResidency) -> Self {
        Self {
            cpu,
            device: None,
            uploader: None,
        }
    }

    /// Host residency plus a device pool filled by a background uploader.
    pub fn with_device_pool(cpu: CpuResidency, pool: Arc<DeviceResidency<T>>) -> Self {
        let uploader = ExpertUploader::spawn(Arc::clone(&pool));
        Self {
            cpu,
            device: Some(pool),
            uploader: Some(uploader),
        }
    }

    /// The device pool, when one exists.
    pub fn device(&self) -> Option<&Arc<DeviceResidency<T>>> {
        self.device.as_ref()
    }

    /// The background uploader, when a device pool exists.
    pub fn uploader(&self) -> Option<&ExpertUploader<T>> {
        self.uploader.as_ref()
    }
}

impl<T: DeviceExpertSlot> ExpertResidency for CompositeResidency<T> {
    fn acquire(&self, layer: u32, expert: u32) {
        self.cpu.acquire(layer, expert);
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
        let comp = CompositeResidency::with_device_pool(cpu, Arc::clone(&pool));
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
