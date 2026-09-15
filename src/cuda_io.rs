//! CUDA-only device-I/O helper for the bounded VRAM expert cache (#62).
//!
//! The residency slot pool is device-agnostic; this module is the CUDA seam.
//! For a fast, dependency-lean bring-up the expert upload uses candle's
//! synchronous `QStorage::from_data(Cow::Borrowed(bytes), device, dtype)`
//! (the issue's "one `QStorage::from_data` / no staging" fallback).  The
//! streamed `cudaMemcpyAsync` + event overlap (design point 3) is the
//! latency-optimisation follow-up and is not implemented yet.

/// Stream index reserved for the speculative expert upload (future overlap).
pub const EXPERT_UPLOAD_STREAM: i32 = 63;
