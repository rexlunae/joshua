# joshua-mock-npu

A pure-Rust reference implementation of the `joshua_npu_*` plugin ABI used by
the [Joshua](https://github.com/rexlunae/joshua) LLM inference engine.

This crate builds a `cdylib` that can be loaded with
`joshua serve --npu-plugin …`. It is primarily a test fixture: it exercises
the plugin protocol deterministically — including crash and hang paths — so
Joshua's host-side isolation (the crash-isolated shim process, circuit
breaker, and timeouts) can be tested without any proprietary vendor SDK.

It is intentionally tiny and dependency-free, and doubles as a reference for
vendors implementing their own `joshua_npu_*` plugins.
