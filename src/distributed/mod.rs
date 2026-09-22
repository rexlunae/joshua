//! Experimental static-cluster building blocks (RFC #86).
//!
//! GGUF sharding and capacity planning are available without networking.
//! The `distributed` feature enables advisory mDNS discovery and authenticated
//! UDP collectives. These APIs do not change [`crate::Engine`]'s single-node
//! execution or automatically admit discovered machines to a cluster.

#[cfg(feature = "distributed")]
pub mod collective;
#[cfg(feature = "distributed")]
pub mod discovery;
pub mod partition;
pub mod shard;
