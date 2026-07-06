//! Runtime loading of `joshua_npu_*` vendor plugins (safety layer 2).
//!
//! This is the only module that touches the plugin's C ABI directly.  All
//! `unsafe` is confined here and wrapped in APIs that uphold the ABI's
//! contract (valid pointers, correct buffer sizes, single-threaded handle
//! use via `&mut`).

use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::Library;

use super::{NpuBackend, NpuSession};

type InitFn = unsafe extern "C" fn(*const c_char, u32, *mut u32, *mut *mut c_void) -> i32;
type ForwardFn = unsafe extern "C" fn(*mut c_void, *const u32, u32, u32, *mut f32) -> i32;
type ResetFn = unsafe extern "C" fn(*mut c_void) -> i32;
type FreeFn = unsafe extern "C" fn(*mut c_void);

/// A dynamically loaded vendor plugin.
pub struct VendorLibrary {
    lib: Library,
    path: PathBuf,
}

impl VendorLibrary {
    /// `dlopen` the plugin and verify it exports the full ABI.
    pub fn open(path: &Path) -> std::result::Result<Self, String> {
        // SAFETY: loading a library runs its initialisers; that is the
        // irreducible trust we place in an explicitly configured plugin.
        let lib = unsafe { Library::new(path) }
            .map_err(|e| format!("failed to load NPU plugin {path:?}: {e}"))?;
        let this = Self {
            lib,
            path: path.to_path_buf(),
        };
        // Fail fast on a library that isn't a Joshua plugin.
        for symbol in [
            "joshua_npu_init",
            "joshua_npu_forward",
            "joshua_npu_reset",
            "joshua_npu_free",
        ] {
            this.get_raw(symbol)?;
        }
        Ok(this)
    }

    /// Path the library was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn get_raw(&self, name: &str) -> std::result::Result<*mut c_void, String> {
        let cname = format!("{name}\0");
        // SAFETY: symbol type is checked at the call sites via the typed
        // helpers below; here we only verify presence.
        unsafe {
            self.lib
                .get::<*mut c_void>(cname.as_bytes())
                .map(|s| *s)
                .map_err(|e| format!("NPU plugin {:?} is missing symbol {name}: {e}", self.path))
        }
    }

    fn sym<T: Copy>(&self, name: &str) -> std::result::Result<T, String> {
        let cname = format!("{name}\0");
        // SAFETY: T is one of the ABI function-pointer types; the plugin
        // contract fixes the signatures.
        unsafe {
            self.lib
                .get::<T>(cname.as_bytes())
                .map(|s| *s)
                .map_err(|e| format!("NPU plugin {:?} is missing symbol {name}: {e}", self.path))
        }
    }

    /// Initialise a vendor session for `model_path`.
    pub fn init(
        self: &Arc<Self>,
        model_path: &Path,
        n_ctx: u32,
    ) -> std::result::Result<VendorSession, String> {
        let init: InitFn = self.sym("joshua_npu_init")?;
        let cpath = CString::new(model_path.to_string_lossy().as_bytes())
            .map_err(|_| "model path contains a NUL byte".to_string())?;
        let mut vocab: u32 = 0;
        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: pointers are valid for the duration of the call; cpath is
        // NUL-terminated.
        let rc = unsafe { init(cpath.as_ptr(), n_ctx, &mut vocab, &mut handle) };
        if rc != 0 || handle.is_null() {
            return Err(format!(
                "NPU plugin {:?} init failed for {model_path:?} (code {rc})",
                self.path
            ));
        }
        if vocab == 0 {
            // Free the handle we can't use.
            if let Ok(free) = self.sym::<FreeFn>("joshua_npu_free") {
                unsafe { free(handle) };
            }
            return Err(format!("NPU plugin {:?} reported a zero vocab", self.path));
        }
        Ok(VendorSession {
            lib: Arc::clone(self),
            handle,
            vocab: vocab as usize,
        })
    }
}

/// A live vendor session (an opaque handle into the plugin).
pub struct VendorSession {
    lib: Arc<VendorLibrary>,
    handle: *mut c_void,
    vocab: usize,
}

// SAFETY: the raw handle is only ever used through `&mut self`, so it moves
// between threads but is never used from two at once.  The ABI requires
// plugins to tolerate that (the same requirement llama.cpp places on ggml
// backends).
unsafe impl Send for VendorSession {}

impl VendorSession {
    /// Feed tokens, writing last-token logits into `out` (len == vocab).
    pub fn forward_into(
        &mut self,
        tokens: &[u32],
        pos: usize,
        out: &mut [f32],
    ) -> std::result::Result<(), String> {
        if tokens.is_empty() {
            return Err("empty token batch".to_string());
        }
        if out.len() != self.vocab {
            return Err(format!(
                "logit buffer has {} slots, vocab is {}",
                out.len(),
                self.vocab
            ));
        }
        let forward: ForwardFn = self.lib.sym("joshua_npu_forward")?;
        // SAFETY: handle is live; tokens/out are valid for the given lengths.
        let rc = unsafe {
            forward(
                self.handle,
                tokens.as_ptr(),
                tokens.len() as u32,
                pos as u32,
                out.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!("NPU plugin forward failed (code {rc})"));
        }
        Ok(())
    }

    /// Vocabulary size.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Clear the plugin's internal state.
    pub fn reset(&mut self) -> std::result::Result<(), String> {
        let reset: ResetFn = self.lib.sym("joshua_npu_reset")?;
        // SAFETY: handle is live.
        let rc = unsafe { reset(self.handle) };
        if rc != 0 {
            return Err(format!("NPU plugin reset failed (code {rc})"));
        }
        Ok(())
    }
}

impl Drop for VendorSession {
    fn drop(&mut self) {
        if let Ok(free) = self.lib.sym::<FreeFn>("joshua_npu_free") {
            // SAFETY: handle is live and never used again.
            unsafe { free(self.handle) };
        }
    }
}

// ─── In-process backend ─────────────────────────────────────────────────────

/// Runs a vendor plugin inside the Joshua process (safety layer 2 only).
///
/// Lowest overhead, but a crash in the vendor runtime takes the whole
/// process down — prefer [`super::ShimBackend`] unless the plugin is
/// trusted.
pub struct InProcessBackend {
    lib: Arc<VendorLibrary>,
}

impl InProcessBackend {
    /// Load the plugin at `library`, verifying the ABI.
    pub fn load(library: &Path) -> std::result::Result<Self, String> {
        Ok(Self {
            lib: Arc::new(VendorLibrary::open(library)?),
        })
    }
}

impl NpuBackend for InProcessBackend {
    fn name(&self) -> String {
        format!("npu-inprocess({})", self.lib.path().display())
    }

    fn create_session(
        &self,
        model_path: &Path,
        n_ctx: u32,
    ) -> std::result::Result<Box<dyn NpuSession>, String> {
        Ok(Box::new(self.lib.init(model_path, n_ctx)?))
    }
}

impl NpuSession for VendorSession {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn forward(&mut self, tokens: &[u32], pos: usize) -> std::result::Result<Vec<f32>, String> {
        let mut out = vec![0f32; self.vocab];
        self.forward_into(tokens, pos, &mut out)?;
        Ok(out)
    }

    fn reset(&mut self) -> bool {
        VendorSession::reset(self).is_ok()
    }
}
