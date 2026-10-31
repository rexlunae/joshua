//! CUDA-only device-I/O helper for the bounded VRAM expert cache (#62).
//!
//! The residency slot pool is device-agnostic; this module is the CUDA seam.
//! DeepSeek4's background uploader calls `QStorage::from_data_transfer`.
//! Its CUDA implementation lives in vendored candle-core, alongside the
//! quantized storage that owns the uploaded buffers.
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
//! * **CUDA** — a lazily created non-blocking cudarc stream is shared by
//!   device clones. Allocation, zero-padding and compact-block H2D copies
//!   use that stream, not the compute stream. The uploader synchronizes
//!   that stream before publishing storage, so the mmap source can be
//!   released and compute can consume the slot safely. This is not a
//!   pending-slot/event handoff or a pinned-memory staging pipeline.
//!
//! These uploads run on the uploader thread. A miss continues on the host.
//! Cudarc may synchronize the context once when creating its first owned
//! stream to enter multi-stream mode; subsequent uploads wait only for the
//! transfer stream. No new raw CUDA FFI is needed here.
//! Separate queues alone do not establish a throughput or overlap benefit:
//! pageable source memory and allocation may still serialize driver work.

/// Legacy reserved index; cudarc now creates the actual transfer stream.
pub const EXPERT_UPLOAD_STREAM: i32 = 63;
