# joshua-llamacpp-npu

A [Joshua](https://github.com/rexlunae/joshua) NPU plugin that bridges to
[llama.cpp](https://github.com/ggerganov/llama.cpp) via the `llama-cpp-2`
crate. It implements the `joshua_npu_*` plugin ABI by driving llama.cpp
directly, so every ggml backend llama.cpp supports — Qualcomm Hexagon NPU,
Huawei CANN, CUDA, Vulkan, OpenCL, Metal, … — works through Joshua's
crash-isolated plugin shim against the same GGUF file, with no model
conversion.

This crate compiles llama.cpp (C++), so it is **not** part of Joshua's
default pure-Rust build. Build it explicitly:

```bash
cargo build --release -p joshua-llamacpp-npu
joshua serve --model m.gguf \
    --npu-plugin target/release/libjoshua_llamacpp_npu.so
```

Environment variables: `JOSHUA_LLAMA_N_GPU_LAYERS` (layer offload, default
all), `JOSHUA_LLAMA_MMPROJ` (multimodal projector for vision), and
`JOSHUA_LLAMA_BACKENDS_DIR` (dynamic ggml backend modules, for NPU backends
built against `dynamic-backends`). See the Joshua README for details.
