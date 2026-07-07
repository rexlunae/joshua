//! Mock NPU vendor library for testing Joshua's plugin ABI.
//!
//! Implements the `joshua_npu_*` C ABI (see `joshua::npu` for the contract)
//! entirely in pure Rust, standing in for a real vendor runtime (QNN, CANN,
//! Core ML shims, …).  It behaves like a stateful causal model: every call to
//! `forward` appends the fed tokens to an internal history — the mock's "KV
//! cache" — and the returned logits are a deterministic function of the whole
//! history, so tests can verify position bookkeeping, prefix reuse, and reset
//! semantics exactly.
//!
//! The greedy argmax of the returned logits is always
//! `(sum(history) % 12) + 4`, which keeps generated tokens inside the letter
//! range of the test tokenizer (ids 4–15) and lets tests precompute expected
//! output.
//!
//! Two magic token values simulate vendor-runtime failure modes for the
//! isolation tests (only ever sent by tests, never by the engine):
//!
//! - `4242` — the library aborts the process (simulated driver crash).
//! - `5150` — `forward` sleeps 5 seconds (simulated hang, for timeouts).

use std::ffi::CStr;
use std::os::raw::{c_char, c_void};

const VOCAB: u32 = 16;

/// Token that makes the mock abort the process (crash simulation).
const CRASH_TOKEN: u32 = 4242;
/// Token that makes `forward` hang for 5 s (timeout simulation).
const HANG_TOKEN: u32 = 5150;

struct MockState {
    /// Every token fed so far — the mock's "KV cache".
    history: Vec<u32>,
    n_ctx: u32,
}

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
    // A real vendor library would load/compile its model artifact here; the
    // mock only checks the path is valid UTF-8.
    if CStr::from_ptr(model_path).to_str().is_err() {
        return -2;
    }
    let state = Box::new(MockState {
        history: Vec::new(),
        n_ctx,
    });
    *out_vocab = VOCAB;
    *out_handle = Box::into_raw(state) as *mut c_void;
    0
}

/// # Safety
/// `handle` must come from `joshua_npu_init`; `tokens` must be valid for
/// `n_tokens` reads; `out_logits` must be valid for `VOCAB` writes.
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
    let state = &mut *(handle as *mut MockState);
    let tokens = std::slice::from_raw_parts(tokens, n_tokens as usize);

    if tokens.contains(&CRASH_TOKEN) {
        std::process::abort();
    }
    if tokens.contains(&HANG_TOKEN) {
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    // Positions must append contiguously to the history, like a KV cache.
    if pos as usize != state.history.len() {
        return -3;
    }
    if state.history.len() + tokens.len() > state.n_ctx as usize {
        return -4;
    }
    state.history.extend_from_slice(tokens);

    // Deterministic logits over the whole history: argmax is
    // (sum(history) % 12) + 4, everything else gets a smaller, distinct value.
    let sum: u64 = state.history.iter().map(|&t| t as u64).sum();
    let argmax = ((sum % 12) + 4) as usize;
    let logits = std::slice::from_raw_parts_mut(out_logits, VOCAB as usize);
    for (i, l) in logits.iter_mut().enumerate() {
        // Vary non-argmax logits with the history too, so tests catch any
        // history divergence even away from the argmax.
        *l = ((sum.wrapping_mul(31).wrapping_add(i as u64) % 97) as f32) / 100.0;
    }
    logits[argmax] = 10.0;
    0
}

/// # Safety
/// `handle` must come from `joshua_npu_init`.
#[no_mangle]
pub unsafe extern "C" fn joshua_npu_reset(handle: *mut c_void) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let state = &mut *(handle as *mut MockState);
    state.history.clear();
    0
}

/// # Safety
/// `handle` must come from `joshua_npu_init` and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn joshua_npu_free(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut MockState));
    }
}

/// Optional multimodal entry point (see `joshua::npu`).
///
/// The mock "tokenises" the prompt as whitespace-word count and charges 3
/// positions per image; each consumed position appends a deterministic
/// pseudo-token to the history, so logits (and the greedy argmax formula)
/// stay a pure function of everything fed so far.
///
/// # Safety
/// `handle` must come from `joshua_npu_init`; `prompt` must be a valid
/// NUL-terminated string; `images`/`image_sizes` must be valid for
/// `n_images` reads; `out_logits` must be valid for `VOCAB` writes.
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
    let state = &mut *(handle as *mut MockState);
    let Ok(prompt) = CStr::from_ptr(prompt).to_str() else {
        return -2;
    };

    // Media prefill starts a fresh sequence.
    state.history.clear();

    // Text positions: one pseudo-token per whitespace-separated word.
    for word in prompt.split_whitespace() {
        let tok = word.bytes().map(|b| b as u32).sum::<u32>() % 1000;
        state.history.push(tok);
    }
    // Media positions: three per image, derived from the image bytes.
    for i in 0..n_images as usize {
        let data = std::slice::from_raw_parts(*images.add(i), *image_sizes.add(i) as usize);
        let sum = data.iter().map(|&b| b as u64).sum::<u64>();
        for k in 0..3u64 {
            state.history.push(((sum + k) % 1000) as u32);
        }
    }
    if state.history.is_empty() || state.history.len() > state.n_ctx as usize {
        return -4;
    }
    *out_n_past = state.history.len() as u32;

    // Same logits formula as `joshua_npu_forward`.
    let sum: u64 = state.history.iter().map(|&t| t as u64).sum();
    let argmax = ((sum % 12) + 4) as usize;
    let logits = std::slice::from_raw_parts_mut(out_logits, VOCAB as usize);
    for (i, l) in logits.iter_mut().enumerate() {
        *l = ((sum.wrapping_mul(31).wrapping_add(i as u64) % 97) as f32) / 100.0;
    }
    logits[argmax] = 10.0;
    0
}
