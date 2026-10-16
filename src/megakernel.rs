//! Decode megakernels: a whole single-token decode step in one device
//! launch instead of ~30 launches per layer.
//!
//! Opt-in with `JOSHUA_MEGAKERNEL=1`.  Today the SYCL backend has one
//! (`candle_core::sycl_backend::mega`), used by the Qwen loader for dense
//! pre-norm attention models whose weights all live on the device; anything
//! else keeps the op-at-a-time path.  Prefill, speculative verification and
//! batched decode always take the op-at-a-time path, and the megakernel
//! writes its keys and values into the same per-layer [`SeqKvCache`]
//! buffers, so the two paths interleave freely within a session.
//!
//! `JOSHUA_MEGAKERNEL_GROUPS=<n>` overrides the work-groups each launch
//! uses (default: four per compute unit).
//!
//! [`SeqKvCache`]: crate::attention::SeqKvCache

use candle_core::{Result, Tensor};

use crate::attention::KvCache;

static FUSED_STEPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Decode steps run through a megakernel so far in this process (tests
/// assert the fused path really ran).
pub fn fused_steps() -> usize {
    FUSED_STEPS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether `JOSHUA_MEGAKERNEL` asks for the decode megakernel.
pub fn requested() -> bool {
    matches!(std::env::var("JOSHUA_MEGAKERNEL").as_deref(), Ok(v) if !v.is_empty() && v != "0")
}

/// Work-group override (`JOSHUA_MEGAKERNEL_GROUPS`), 0 for the default.
pub fn groups() -> usize {
    std::env::var("JOSHUA_MEGAKERNEL_GROUPS").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// A loaded model's megakernel.
pub struct Plan {
    #[cfg(feature = "sycl")]
    inner: candle_core::sycl_backend::mega::DecodePlan,
    #[cfg(not(feature = "sycl"))]
    _never: std::convert::Infallible,
}

impl Plan {
    #[cfg(feature = "sycl")]
    pub fn sycl(inner: candle_core::sycl_backend::mega::DecodePlan) -> Self {
        Self { inner }
    }

    /// Run one decode step for the token embedded in `xs` (`[1, 1, hidden]`)
    /// at position `offset`, appending its keys and values to `caches` (one
    /// per layer).  Returns the logits `[1, vocab]`, or `None` when this step
    /// cannot run fused (a cache not yet created or not at `offset`, a
    /// position past the RoPE table); the caller then takes the op-at-a-time
    /// path, which leaves the caches as they were.
    pub fn decode(&self, caches: &mut [&mut KvCache], xs: &Tensor, offset: usize) -> Result<Option<Tensor>> {
        #[cfg(feature = "sycl")]
        {
            let plan = &self.inner;
            if caches.len() != plan.n_layers() || offset >= plan.max_pos() || !xs.device().is_sycl() {
                return Ok(None);
            }
            for c in caches.iter() {
                match c {
                    Some(c) if c.len() == offset && c.k().dtype() == candle_core::DType::F32 => {}
                    _ => return Ok(None),
                }
            }
            let mut kv = Vec::with_capacity(caches.len());
            for c in caches.iter_mut() {
                let c = c.as_mut().expect("checked above");
                c.reserve(1)?;
                kv.push((c.k().clone(), c.v().clone()));
            }
            let x = xs.contiguous()?;
            let logits = plan.decode(&kv, offset, &x)?;
            for c in caches.iter_mut() {
                c.as_mut().expect("checked above").advance(1);
            }
            FUSED_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(Some(logits))
        }
        #[cfg(not(feature = "sycl"))]
        {
            let _ = (caches, xs, offset);
            match self._never {}
        }
    }

    /// Work-groups each launch uses (diagnostics).
    pub fn groups(&self) -> usize {
        #[cfg(feature = "sycl")]
        {
            self.inner.groups()
        }
        #[cfg(not(feature = "sycl"))]
        match self._never {}
    }
}
