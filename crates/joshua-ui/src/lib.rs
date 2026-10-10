//! `joshua-ui`: a Dioxus + Bulma web frontend for monitoring and driving a
//! Joshua server (a local instance or the nodes of a cluster).
//!
//! * [`api`] — transport-free request building and response parsing,
//!   unit-tested on the host.
//! * [`client`] — `reqwest` transport (browser `fetch` on `wasm32`).
//! * [`ui`] — the Dioxus components.

pub mod api;
pub mod client;
pub mod ui;
