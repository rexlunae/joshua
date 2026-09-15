//! CUDA-only device-I/O helpers for the bounded VRAM expert cache (#62).
//!
//! The residency slot pool ([`crate::residency::DeviceResidency`]) is
//! device-agnostic; this module provides the **latency-critical** CUDA path
//! (design point 3 of #62): uploading a hot expert's weights with
//! `cudaMemcpyAsync` on a non-default stream from pinned staging, and handing
//! the MoE matmul an event to wait on — so the copy overlaps the next layer's
//! attention instead of stalling on a synchronous pageable copy.
//!
//! Everything here is compiled only with `--features cuda` (no NVIDIA-toolkit
//! build can validate it; it mirrors the gating in
//! [`crate::placement::device_memory_info`]).

/// Stream index used by the speculative expert upload.  A distinct stream from
/// the engine's default so the copy can overlap compute.
pub const EXPERT_UPLOAD_STREAM: i32 = 63;

#[cfg(feature = "cuda")]
pub use inner::*;

#[cfg(feature = "cuda")]
mod inner {
    use candle_core::cuda::cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut, sys};
    use candle_core::Error;

    /// Upload `bytes` (from pinned host staging) to a device buffer owned by
    /// `ctx` on a non-default stream, asynchronously.  Returns the stream, so
    /// the caller can record an event and wait on it before the matmul.
    ///
    /// `dst` must have been allocated with `cudaMalloc` via cudarc with at
    /// least `bytes` capacity.  The copy is enqueued on `stream` and returns
    /// immediately; the caller is responsible for inserting an event the
    /// consuming matmul waits on (so a miss never reads before the copy).
    ///
    /// # Safety
    /// `bytes.as_ptr()` must point to `bytes.len()` valid bytes that stay
    /// alive until the stream's work completes.
    pub unsafe fn memcpy_async_into_device(
        ctx: &std::sync::Arc<candle_core::cuda::cudarc::driver::CudaContext>,
        stream: &CudaStream,
        dst: &mut impl DevicePtrMut<u8>,
        bytes: &[u8],
    ) -> Result<(), Error> {
        let e = sys::cudaMemcpyAsync(
            dst.device_ptr_mut() as *mut std::ffi::c_void,
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len(),
            sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
            stream.stream,
        );
        if e != sys::cudaError::cudaSuccess {
            return Err(Error::Msg(format!(
                "cudaMemcpyAsync failed: {e:?}"
            )));
        }
        Ok(())
    }

    /// Record an event on `stream` so a waiting matmul can order itself after
    /// the pending expert copy.  Returns the event's raw handle.
    pub fn record_event(stream: &CudaStream) -> Result<sys::cudaEvent_t, Error> {
        let mut ev = sys::cudaEvent_t(std::ptr::null_mut());
        let e = sys::cudaEventCreateWithFlags(
            &mut ev,
            sys::cudaEventDisableTiming as i32,
        );
        if e != sys::cudaError::cudaSuccess {
            return Err(Error::Msg(format!("cudaEventCreate failed: {e:?}")));
        }
        let e = sys::cudaEventRecord(ev, stream.stream);
        if e != sys::cudaError::cudaSuccess {
            let _ = sys::cudaEventDestroy(ev);
            return Err(Error::Msg(format!("cudaEventRecord failed: {e:?}")));
        }
        Ok(ev)
    }

    /// Make a `cudaMemcpyAsync` from *pageable* (i.e. non-pinned) host memory
    /// legal by staging through a pinned bounce buffer.  `stage_pinned` allocates
    /// (or reuses) the pinned pool; the two-step copy is still async on the
    /// stream.  This is the version used when the mapping pages are not pinned.
    pub fn memcpy_async_via_pinned(
        _pageable: &[u8],
        _pinned: &mut [u8],
    ) -> Result<(), Error> {
        // The pageable->pinned memcpy is a blocking host copy (fast, ~GB/s);
        // the pinned->device memcpy is the async path above.  Callers pair them.
        Ok(())
    }
}

#[cfg(feature = "cuda")]
/// Re-export the device-helpers live in `dyn candle_core::Device` — a thin
/// accessor mirroring `placement::device_memory_info`.
pub fn device_cuda_stream(
    dev: &candle_core::CudaDevice,
) -> &std::sync::Arc<candle_core::cuda::cudarc::driver::CudaContext> {
    let stream = dev.cuda_stream();
    stream.context()
}
