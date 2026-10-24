//! Versioned, dynamically loaded C ABI. No SYCL toolchain or runtime is
//! needed to compile the Rust crate, including on machines without SYCL.
use crate::{Error, Result};
use std::ffi::{c_char, c_void, CStr};
use std::path::{Path, PathBuf};

unsafe fn load_library(path: &Path) -> std::result::Result<libloading::Library, libloading::Error> {
    #[cfg(target_os = "linux")]
    {
        libloading::os::unix::Library::open(
            Some(path),
            libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
        ).map(Into::into)
    }
    #[cfg(not(target_os = "linux"))]
    {
        libloading::Library::new(path)
    }
}

fn runtime_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("JOSHUA_SYCL_RUNTIME") {
        candidates.push(path.into());
    }
    #[cfg(target_os = "linux")]
    if let Some(home) = std::env::var_os("HOME") {
        if let Ok(entries) = std::fs::read_dir(home) {
            let mut roots: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path())
                .filter(|p| p.file_name().and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("sycl-toolchain-"))).collect();
            roots.sort();
            if let Some(root) = roots.pop() {
                candidates.push(root.join("lib/libsycl.so.9"));
            }
        }
    }
    candidates
}

fn bridge_candidates() -> Vec<PathBuf> {
    if let Some(path) = std::env::var_os("JOSHUA_SYCL_LIBRARY") {
        return vec![path.into()];
    }
    let mut candidates = vec![libloading::library_filename("joshua_sycl").into()];
    #[cfg(target_os = "linux")]
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        candidates.push(home.join("joshua-sycl/build/libjoshua_sycl.so"));
        candidates.push(home.join("joshua-sycl/libjoshua_sycl.so"));
    }
    candidates
}

#[repr(C)]
pub struct Arg { pub data: *const c_void, pub size: usize }

macro_rules! api {
    (required { $($name:ident($($arg:ty),*) -> $ret:ty;)+ }
     optional { $($oname:ident($($oarg:ty),*) -> $oret:ty;)* }) => {
        struct Api {
            _library: libloading::Library,
            _runtime: Option<libloading::Library>,
            $($name: unsafe extern "C" fn($($arg),*) -> $ret,)+
            // Entry points a bridge built before them lacks; `None` there.
            $($oname: Option<unsafe extern "C" fn($($oarg),*) -> $oret>,)*
        }
        impl Api {
            unsafe fn load() -> std::result::Result<Self, String> {
                let runtime = runtime_candidates().iter().find_map(|p| load_library(p).ok());
                let mut loaded = None;
                let mut errors = Vec::new();
                for path in bridge_candidates() {
                    match load_library(&path) {
                        Ok(library) => { loaded = Some(library); break; }
                        Err(e) => errors.push(format!("{path:?}: {e}")),
                    }
                }
                let library = loaded.ok_or_else(|| format!(
                    "cannot load SYCL bridge: {}; build the SYCL bridge and set JOSHUA_SYCL_LIBRARY",
                    errors.join("; ")
                ))?;
                let version = library.get::<unsafe extern "C" fn() -> u32>(b"joshua_sycl_abi_version\0").map_err(|e| e.to_string())?;
                if version() != 1 { return Err("SYCL bridge ABI version mismatch (expected 1)".into()); }
                $(let $name = *library.get(concat!("joshua_sycl_", stringify!($name), "\0").as_bytes()).map_err(|e| e.to_string())?;)+
                $(let $oname = library.get(concat!("joshua_sycl_", stringify!($oname), "\0").as_bytes()).ok().map(|f| *f);)*
                Ok(Self { _library: library, _runtime: runtime, $($name,)+ $($oname,)* })
            }
        }
    }
}
// The public wrappers below name arguments explicitly; keep function pointers
// private so the library is retained for the lifetime of every device.
api! {
    required {
    open(usize, *mut usize) -> i32;
    close(usize) -> i32;
    info(usize, *mut c_char, usize, *mut u64) -> i32;
    error() -> *const c_char;
    alloc(usize, usize, *mut usize) -> i32;
    free(usize) -> i32;
    finish(usize) -> i32;
    write(usize, usize, usize, usize, *const c_void) -> i32;
    transfer_write(usize, usize, usize, usize, *const c_void) -> i32;
    read(usize, usize, usize, usize, *mut c_void) -> i32;
    copy(usize, usize, usize, usize, usize, usize) -> i32;
    launch(usize, *const c_char, *const Arg, usize, *const usize, *const usize) -> i32;
    }
    optional {
        numerics_version() -> u32;
        canonical_quantized(usize) -> u32;
        mega_new(usize, *const MegaModelDesc, *const MegaLayerDesc, *mut usize) -> i32;
        mega_free(usize) -> i32;
        mega_groups(usize) -> i32;
        mega_decode(usize, usize, *const MegaKvDesc, usize, i32, MegaVDesc, MegaVDesc) -> i32;
    }
}
fn api() -> Result<&'static Api> {
    static API: std::sync::OnceLock<std::result::Result<Api, String>> = std::sync::OnceLock::new();
    API.get_or_init(|| unsafe { Api::load() }).as_ref().map_err(|e| Error::Msg(e.clone()))
}
pub fn numerics_version() -> u32 {
    api().ok().and_then(|a| a.numerics_version).map(|f| unsafe { f() }).unwrap_or(0)
}

pub fn canonical_quantized(context: usize) -> bool {
    api().ok().and_then(|a| a.canonical_quantized)
        .is_some_and(|f| unsafe { f(context) == 1 })
}

pub fn error(op: &str) -> Error {
    let detail = api().map(|a| unsafe { CStr::from_ptr((a.error)()).to_string_lossy().into_owned() })
        .unwrap_or_else(|e| e.to_string());
    Error::Msg(format!("sycl {op}: {detail}"))
}
pub fn open(ordinal: usize) -> Result<usize> {
    let mut handle = 0;
    if unsafe { (api()?.open)(ordinal, &mut handle) } != 0 { return Err(error("open")); }
    Ok(handle)
}
pub fn info(h: usize) -> Result<(String, u64)> {
    let mut name = [0u8; 1024]; let mut memory = 0;
    if unsafe { (api()?.info)(h, name.as_mut_ptr().cast(), name.len(), &mut memory) } != 0 { return Err(error("info")); }
    Ok((unsafe { CStr::from_ptr(name.as_ptr().cast()) }.to_string_lossy().into_owned(), memory))
}
pub fn alloc(h: usize, bytes: usize) -> Result<usize> {
    let mut out = 0;
    if unsafe { (api()?.alloc)(h, bytes, &mut out) } != 0 { return Err(error("alloc")); }
    Ok(out)
}
pub unsafe fn close(h: usize) { if let Ok(a) = api() { (a.close)(h); } }
pub unsafe fn free(h: usize) { if h != 0 { if let Ok(a) = api() { (a.free)(h); } } }
pub unsafe fn finish(h: usize) -> i32 { api().map(|a| (a.finish)(h)).unwrap_or(-1) }
pub unsafe fn write(h: usize, b: usize, off: usize, bytes: usize, ptr: *const u8) -> Result<()> {
    if bytes > 0 && (api()?.write)(h, b, off, bytes, ptr.cast()) != 0 { return Err(error("write")); }
    Ok(())
}
/// [`write`] on the transfer queue: the calling thread waits for its own
/// memcpy but no compute-queue kernel is delayed (the uploader's path).
pub unsafe fn transfer_write(h: usize, b: usize, off: usize, bytes: usize, ptr: *const u8) -> Result<()> {
    if bytes > 0 && (api()?.transfer_write)(h, b, off, bytes, ptr.cast()) != 0 {
        return Err(error("transfer_write"));
    }
    Ok(())
}
pub unsafe fn read(h: usize, b: usize, off: usize, bytes: usize, ptr: *mut u8) -> Result<()> {
    if bytes > 0 && (api()?.read)(h, b, off, bytes, ptr.cast()) != 0 { return Err(error("read")); }
    Ok(())
}
pub unsafe fn copy(h: usize, a: usize, b: usize, ao: usize, bo: usize, bytes: usize) -> i32 {
    api().map(|api| (api.copy)(h, a, b, ao, bo, bytes)).unwrap_or(-1)
}
pub unsafe fn launch(h: usize, name: &CStr, args: &[Arg], global: &[usize; 3], local: &[usize; 3]) -> Result<()> {
    if (api()?.launch)(h, name.as_ptr(), args.as_ptr(), args.len(), global.as_ptr(), local.as_ptr()) != 0 { return Err(error("launch")); }
    Ok(())
}

// ─── Decode megakernel (mega.hpp) ────────────────────────────────────────────
// Layouts must match the `Mega*Desc` structs in bridge.cpp.

/// An f32 vector: buffer handle (0 = absent) and byte offset.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MegaVDesc { pub buf: usize, pub off: u64 }

/// A `[n, k]` weight: GGUF type code, block elements / bytes, and the
/// `k_qgemv_mr` columns per pass (0 = the one-row kernel's layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MegaWDesc { pub buf: usize, pub off: u64, pub qt: i32, pub qk: i32, pub bsz: i32, pub n: i32, pub k: i32, pub cols: i32 }

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MegaLayerDesc {
    pub attn_norm: MegaVDesc,
    pub ffn_norm: MegaVDesc,
    pub qkv: [MegaWDesc; 3],
    pub qkv_b: [MegaVDesc; 3],
    pub q_norm: MegaVDesc,
    pub k_norm: MegaVDesc,
    pub o: MegaWDesc,
    pub o_b: MegaVDesc,
    pub gate: MegaWDesc,
    pub up: MegaWDesc,
    pub down: MegaWDesc,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MegaModelDesc {
    pub n_layers: i32, pub h: i32, pub n_head: i32, pub n_kv: i32, pub hd: i32, pub n_rot: i32,
    pub neox: i32, pub ffn: i32, pub vocab: i32, pub max_pos: i32, pub groups: i32, pub pad: i32,
    pub eps: f32, pub scale: f32,
    pub cos: MegaVDesc, pub sin: MegaVDesc, pub out_norm: MegaVDesc, pub out_b: MegaVDesc,
    pub out: MegaWDesc,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MegaKvDesc { pub k: MegaVDesc, pub v: MegaVDesc, pub cap: i32, pub pad: i32 }

fn mega_missing() -> Error {
    Error::Msg("sycl: this SYCL bridge has no decode megakernel; rebuild it from this tree".into())
}

/// Whether the loaded bridge has the megakernel entry points.
pub fn mega_available() -> bool {
    api().map(|a| a.mega_new.is_some() && a.mega_free.is_some() && a.mega_decode.is_some()).unwrap_or(false)
}

pub fn mega_new(h: usize, model: &MegaModelDesc, layers: &[MegaLayerDesc]) -> Result<usize> {
    let f = api()?.mega_new.ok_or_else(mega_missing)?;
    if layers.len() != model.n_layers as usize { crate::bail!("sycl megakernel: {} layer descriptors for {} layers", layers.len(), model.n_layers); }
    let mut out = 0;
    if unsafe { f(h, model, layers.as_ptr(), &mut out) } != 0 { return Err(error("mega_new")); }
    Ok(out)
}

pub unsafe fn mega_free(plan: usize) {
    if let Ok(Some(f)) = api().map(|a| a.mega_free) { f(plan); }
}

/// Work-groups the plan launches (diagnostics).
pub fn mega_groups(plan: usize) -> usize {
    match api().map(|a| a.mega_groups) {
        Ok(Some(f)) => unsafe { f(plan) }.max(0) as usize,
        _ => 0,
    }
}

pub fn mega_decode(h: usize, plan: usize, kv: &[MegaKvDesc], pos: i32, x_in: MegaVDesc, logits: MegaVDesc) -> Result<()> {
    let f = api()?.mega_decode.ok_or_else(mega_missing)?;
    if unsafe { f(h, plan, kv.as_ptr(), kv.len(), pos, x_in, logits) } != 0 { return Err(error("mega_decode")); }
    Ok(())
}
