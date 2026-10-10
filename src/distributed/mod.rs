//! Experimental static-cluster building blocks (RFC #86).
//!
//! GGUF sharding and capacity planning are available without networking.
//! The `distributed` feature also enables CPU DeepSeek V4 tensor-parallel
//! generation, a static CPU Qwen3 layer pipeline, advisory mDNS discovery,
//! authenticated UDP/TCP collectives and authenticated TCP link probing.
//! These APIs do not change [`crate::Engine`]'s single-node execution or
//! automatically admit discovered machines to a cluster.

#[cfg(feature = "distributed")]
pub mod collective;
#[cfg(feature = "distributed")]
pub mod discovery;
pub mod partition;
#[cfg(feature = "distributed")]
pub mod probe;
pub mod shard;
#[cfg(feature = "distributed")]
pub mod tcp;
#[cfg(feature = "distributed")]
pub mod session;
#[cfg(feature = "distributed")]
pub mod deepseek4;
#[cfg(feature = "distributed")]
pub mod pipeline;
#[cfg(feature = "distributed")]
pub mod remote;
