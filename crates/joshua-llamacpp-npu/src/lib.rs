//! llama.cpp compatibility plugin for Joshua.
//!
//! Implements Joshua's `joshua_npu_*` plugin ABI (see `joshua::npu`) by
//! driving llama.cpp through the [`llama-cpp-2`] bindings.  Loading this
//! plugin — preferably via the isolated `joshua-npu-shim` — gives Joshua
//! every backend llama.cpp supports **without any vendor adopting Joshua**:
//! vendors keep contributing ggml backends (Qualcomm Hexagon NPU, Huawei
//! CANN, CUDA, Vulkan, OpenCL, Metal, …) and this adapter bridges them.
//!
//! The same GGUF file Joshua already mmaps is passed straight to llama.cpp —
//! no model conversion.  Offload is controlled with
//! `JOSHUA_LLAMA_N_GPU_LAYERS` (default: offload everything llama.cpp's
//! active backend can take).
//!
//! # Building
//!
//! This crate compiles llama.cpp from source and therefore needs CMake and a
//! C++ toolchain — which is exactly why it lives outside the workspace's
//! default members: `cargo build` of Joshua itself stays pure Rust, and the
//! C++ world only ever runs inside the disposable shim process.
//!
//! ```bash
//! cargo build --release -p joshua-llamacpp-npu          # CPU backends
//! cargo build --release -p joshua-llamacpp-npu --features cuda|vulkan|metal
//! joshua serve --model m.gguf --npu-plugin target/release/libjoshua_llamacpp_npu.so
//! ```
//!
//! # Automatic backend loading
//!
//! llama.cpp can register ggml backends at runtime from separate
//! `libggml-<name>` shared modules — the mechanism vendors use to ship NPU
//! backends (Hexagon, CANN) and GPU/CPU variants.  Build this crate with the
//! `dynamic-backends` feature and llama.cpp is compiled with `GGML_BACKEND_DL`;
//! [`load_dynamic_backends`] then discovers and registers those modules on
//! startup, before backend init:
//!
//! ```bash
//! cargo build --release -p joshua-llamacpp-npu --features dynamic-backends
//! # Cross-compile the NPU backend module (needs the vendor SDK) and drop it
//! # where it will be found:
//! JOSHUA_LLAMA_BACKENDS_DIR=/opt/ggml-backends \
//!   joshua serve --model m.gguf --npu-plugin .../libjoshua_llamacpp_npu.so
//! ```
//!
//! So an NPU backend built for the target device is picked up with no code
//! change.  Without the feature only the statically-linked backends (CPU plus
//! any `cuda`/`vulkan`/`metal` feature) are available.
//!
//! Under `dynamic-backends`, ggml itself becomes a set of versioned shared
//! objects (`libggml-base.so.0`, `libggml.so.0`, `libllama.so.0`) that the
//! plugin links against.  The build script bakes an rpath into the plugin so
//! the loader finds them: `$ORIGIN` and `$ORIGIN/../lib` for deployment (ship
//! the libraries next to the plugin, or in a sibling `lib/`), plus the build
//! tree's own lib directory so local builds and tests load with no setup.

use std::ffi::CStr;
use std::num::NonZeroU32;
use std::os::raw::{c_char, c_void};
use std::sync::OnceLock;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::token::LlamaToken;

/// Process-wide llama.cpp runtime (loads ggml backends once).
fn backend() -> Option<&'static LlamaBackend> {
    static BACKEND: OnceLock<Option<LlamaBackend>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            // Discover dynamically-linked ggml backend modules (including NPU
            // backends) before initialising, so they register with ggml.
            load_dynamic_backends();
            LlamaBackend::init().ok()
        })
        .as_ref()
}

/// Auto-load dynamically-linked ggml backend modules.
///
/// llama.cpp's backend registry can be populated at runtime from separate
/// `libggml-<name>` shared modules — the mechanism vendors use to ship NPU
/// backends (Qualcomm Hexagon, Huawei CANN) and GPU/CPU variants without a
/// monolithic build.  With the `dynamic-backends` feature, llama.cpp is built
/// with `GGML_BACKEND_DL` and this scans for those modules and registers them:
///
/// - `JOSHUA_LLAMA_BACKENDS_DIR`, if set, points at the directory to scan
///   (e.g. where a cross-compiled `libggml-hexagon.so` was placed);
/// - otherwise the compile-time default directory is used.
///
/// So an NPU backend built for the target device is picked up with no code
/// change — drop the module in the directory and it registers on startup.
/// Without the `dynamic-backends` feature this is a no-op and only the
/// statically-linked backends (CPU, plus any `cuda`/`vulkan`/`metal` feature)
/// are available.
fn load_dynamic_backends() {
    #[cfg(feature = "dynamic-backends")]
    {
        use llama_cpp_2::llama_backend;
        match std::env::var("JOSHUA_LLAMA_BACKENDS_DIR") {
            Ok(dir) if !dir.is_empty() => {
                eprintln!("joshua-llamacpp-npu: loading ggml backend modules from {dir}");
                llama_backend::load_backends_from_path(std::path::Path::new(&dir));
            }
            _ => llama_backend::load_backends(),
        }
    }
}

/// One generation session: an owned model plus its context.
///
/// `LlamaContext` borrows the model, so the context's lifetime is erased to
/// `'static` and safety is maintained structurally: `model` is boxed (stable
/// address) and `context` is declared first so it drops before the model.
struct LlamaSession {
    context: LlamaContext<'static>,
    /// Multimodal projector context, when `JOSHUA_LLAMA_MMPROJ` points at an
    /// mmproj GGUF for this model (enables vision/audio prompts).
    mtmd: Option<MtmdContext>,
    #[allow(dead_code)]
    model: Box<LlamaModel>,
    n_ctx: u32,
    vocab: u32,
}

impl LlamaSession {
    fn new(model_path: &str, n_ctx: u32) -> Result<Self, String> {
        let backend = backend().ok_or("llama.cpp backend failed to initialise")?;

        let n_gpu_layers: u32 = std::env::var("JOSHUA_LLAMA_N_GPU_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000_000); // offload everything the backend accepts
        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = Box::new(
            LlamaModel::load_from_file(backend, model_path, &model_params)
                .map_err(|e| format!("llama.cpp failed to load {model_path}: {e}"))?,
        );

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx.max(1)))
            .with_n_batch(n_ctx.max(1));
        let context = model
            .new_context(backend, ctx_params)
            .map_err(|e| format!("llama.cpp context creation failed: {e}"))?;
        // SAFETY: `context` borrows `*model`, which is heap-allocated and
        // owned by the same struct; field order guarantees the context is
        // dropped first, and the model box is never moved out.
        let context = unsafe {
            std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(context)
        };

        // Optional multimodal projector (vision/audio) via llama.cpp's mtmd.
        let mtmd = match std::env::var("JOSHUA_LLAMA_MMPROJ") {
            Ok(mmproj) if !mmproj.is_empty() => {
                let params = MtmdContextParams {
                    media_marker: std::ffi::CString::new(
                        llama_cpp_2::mtmd::mtmd_default_marker(),
                    )
                    .expect("marker has no NUL"),
                    ..MtmdContextParams::default()
                };
                let ctx = MtmdContext::init_from_file(&mmproj, &model, &params)
                    .map_err(|e| format!("mtmd projector load failed for {mmproj}: {e}"))?;
                Some(ctx)
            }
            _ => None,
        };

        let vocab = model.n_vocab().max(0) as u32;
        Ok(Self {
            context,
            mtmd,
            model,
            n_ctx,
            vocab,
        })
    }

    /// Tokenise-and-prefill a multimodal prompt via mtmd; returns positions
    /// consumed and last-position logits.
    fn media_prefill(&mut self, prompt: &str, images: &[&[u8]]) -> Result<(u32, Vec<f32>), String> {
        let Some(mtmd) = &self.mtmd else {
            return Err(
                "no multimodal projector loaded — set JOSHUA_LLAMA_MMPROJ to the mmproj GGUF"
                    .to_string(),
            );
        };

        // Fresh sequence: multimodal prefill always starts at position 0.
        self.context.clear_kv_cache();

        let bitmaps: Vec<MtmdBitmap> = images
            .iter()
            .map(|data| {
                MtmdBitmap::from_buffer(mtmd, data, false)
                    .map_err(|e| format!("image decode failed: {e}"))
            })
            .collect::<Result<_, _>>()?;
        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();

        let chunks = mtmd
            .tokenize(
                MtmdInputText {
                    text: prompt.to_string(),
                    add_special: true,
                    parse_special: true,
                },
                &bitmap_refs,
            )
            .map_err(|e| format!("mtmd tokenize failed: {e}"))?;

        let n_past = chunks
            .eval_chunks(mtmd, &self.context, 0, 0, self.n_ctx.max(1) as i32, true)
            .map_err(|e| format!("mtmd eval failed: {e}"))?;

        let logits = self.context.get_logits_ith(-1).to_vec();
        Ok((n_past.max(0) as u32, logits))
    }

    fn forward(&mut self, tokens: &[u32], pos: u32) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("empty token batch".to_string());
        }
        if pos as usize + tokens.len() > self.n_ctx as usize {
            return Err(format!(
                "position {pos} + {} tokens exceeds n_ctx {}",
                tokens.len(),
                self.n_ctx
            ));
        }
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        let last = tokens.len() - 1;
        for (i, &token) in tokens.iter().enumerate() {
            batch
                .add(LlamaToken(token as i32), (pos as usize + i) as i32, &[0], i == last)
                .map_err(|e| format!("llama.cpp batch add failed: {e}"))?;
        }
        self.context
            .decode(&mut batch)
            .map_err(|e| format!("llama.cpp decode failed: {e}"))?;
        Ok(self.context.get_logits_ith(last as i32).to_vec())
    }

    fn reset(&mut self) {
        self.context.clear_kv_cache();
    }
}

// ─── joshua_npu_* ABI ───────────────────────────────────────────────────────

/// # Safety
/// `model_path` must be a valid NUL-terminated string; `out_vocab` and
/// `out_handle` must be valid for writes.
#[no_mangle]
pub unsafe extern "C" fn joshua_npu_init(
    model_path: *const c_char,
    n_ctx: u32,
    out_vocab: *mut u32,
    out_handle: *mut *mut c_void,
) -> i32 {
    if model_path.is_null() || out_vocab.is_null() || out_handle.is_null() {
        return -1;
    }
    let Ok(path) = CStr::from_ptr(model_path).to_str() else {
        return -2;
    };
    let result = std::panic::catch_unwind(|| LlamaSession::new(path, n_ctx));
    match result {
        Ok(Ok(session)) => {
            *out_vocab = session.vocab;
            *out_handle = Box::into_raw(Box::new(session)) as *mut c_void;
            0
        }
        Ok(Err(e)) => {
            eprintln!("joshua-llamacpp-npu: {e}");
            -3
        }
        Err(_) => {
            eprintln!("joshua-llamacpp-npu: panic during init");
            -4
        }
    }
}

/// # Safety
/// `handle` must come from `joshua_npu_init`; `tokens` must be valid for
/// `n_tokens` reads; `out_logits` must be valid for vocab-size writes.
#[no_mangle]
pub unsafe extern "C" fn joshua_npu_forward(
    handle: *mut c_void,
    tokens: *const u32,
    n_tokens: u32,
    pos: u32,
    out_logits: *mut f32,
) -> i32 {
    if handle.is_null() || tokens.is_null() || out_logits.is_null() || n_tokens == 0 {
        return -1;
    }
    let session = &mut *(handle as *mut LlamaSession);
    let tokens = std::slice::from_raw_parts(tokens, n_tokens as usize);
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.forward(tokens, pos)));
    match result {
        Ok(Ok(logits)) => {
            let out = std::slice::from_raw_parts_mut(out_logits, session.vocab as usize);
            let n = logits.len().min(out.len());
            out[..n].copy_from_slice(&logits[..n]);
            0
        }
        Ok(Err(e)) => {
            eprintln!("joshua-llamacpp-npu: {e}");
            -3
        }
        Err(_) => {
            eprintln!("joshua-llamacpp-npu: panic during forward");
            -4
        }
    }
}

/// Optional multimodal entry point: tokenise-and-prefill a prompt whose
/// `<__media__>` markers correspond to `images` (raw encoded bytes), via
/// llama.cpp's `mtmd`.  Requires `JOSHUA_LLAMA_MMPROJ`.
///
/// # Safety
/// `handle` must come from `joshua_npu_init`; `prompt` must be a valid
/// NUL-terminated string; `images`/`image_sizes` must be valid for
/// `n_images` reads; `out_n_past` and `out_logits` must be valid for writes
/// (`out_logits` for vocab-size floats).
#[no_mangle]
pub unsafe extern "C" fn joshua_npu_media_prefill(
    handle: *mut c_void,
    prompt: *const c_char,
    images: *const *const u8,
    image_sizes: *const u64,
    n_images: u32,
    out_n_past: *mut u32,
    out_logits: *mut f32,
) -> i32 {
    if handle.is_null() || prompt.is_null() || out_n_past.is_null() || out_logits.is_null() {
        return -1;
    }
    let session = &mut *(handle as *mut LlamaSession);
    let Ok(prompt) = CStr::from_ptr(prompt).to_str() else {
        return -2;
    };
    let images: Vec<&[u8]> = (0..n_images as usize)
        .map(|i| std::slice::from_raw_parts(*images.add(i), *image_sizes.add(i) as usize))
        .collect();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        session.media_prefill(prompt, &images)
    }));
    match result {
        Ok(Ok((n_past, logits))) => {
            *out_n_past = n_past;
            let out = std::slice::from_raw_parts_mut(out_logits, session.vocab as usize);
            let n = logits.len().min(out.len());
            out[..n].copy_from_slice(&logits[..n]);
            0
        }
        Ok(Err(e)) => {
            eprintln!("joshua-llamacpp-npu: {e}");
            -3
        }
        Err(_) => {
            eprintln!("joshua-llamacpp-npu: panic during media_prefill");
            -4
        }
    }
}

/// # Safety
/// `handle` must come from `joshua_npu_init`.
#[no_mangle]
pub unsafe extern "C" fn joshua_npu_reset(handle: *mut c_void) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let session = &mut *(handle as *mut LlamaSession);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.reset())) {
        Ok(()) => 0,
        Err(_) => -4,
    }
}

/// # Safety
/// `handle` must come from `joshua_npu_init` and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn joshua_npu_free(handle: *mut c_void) {
    if !handle.is_null() {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(Box::from_raw(handle as *mut LlamaSession));
        }));
    }
}
