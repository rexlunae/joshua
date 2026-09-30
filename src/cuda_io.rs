//! CUDA-only device-I/O helper for the bounded VRAM expert cache (#62).
//!
//! The residency slot pool is device-agnostic; this module is the CUDA seam.
//! For a fast, dependency-lean bring-up the expert upload used candle's
//! synchronous `QStorage::from_data(Cow::Borrowed(bytes), device, dtype)`.
//!
//! #114 resolved the overlap design point for the backends that need it:
//!
//! * **OpenCL** — `QOpenClStorage::from_bytes_transfer` already wrote on a
//!   second in-order *transfer* queue, blocking only the calling thread
//!   (the background uploader), never the compute queue's kernels.
//! * **SYCL** — `QSyclStorage::from_bytes_transfer` now does the same via
//!   the bridge's second in-order queue (`joshua_sycl_transfer_write`);
//!   before #114 it aliased the compute queue, so a large upload sat
//!   between already-enqueued kernels and delayed every one of them —
//!   the serialization behind the #114 thrash measurement.
//! * **CUDA** — `EXPERT_UPLOAD_STREAM` (stream 63) is reserved for the
//!   cudaMemcpyAsync + event form when a CUDA slot cache lands; candle's
//!   `load_quantized` is synchronous today, and CUDA is not a target of
//!   the partial-cache cards measured so far.
//!
//! The uploads always run on the uploader thread either way, so the decode
//! thread never waits on a transfer; the per-queue isolation above is what
//! keeps the transfers from delaying the compute queue's kernels.

/// Stream index reserved for the speculative expert upload (future overlap).
pub const EXPERT_UPLOAD_STREAM: i32 = 63;
