//! Versioned, dynamically loaded C ABI. No SYCL toolchain or runtime is
//! needed to compile the Rust crate, including on machines without SYCL.
use crate::{Error, Result};
use std::ffi::{c_char, c_void, CStr};

#[repr(C)]
pub struct Arg { pub data: *const c_void, pub size: usize }

macro_rules! api {
    ($($name:ident($($arg:ty),*) -> $ret:ty;)+) => {
        struct Api {
            _library: libloading::Library,
            $($name: unsafe extern "C" fn($($arg),*) -> $ret,)+
        }
        impl Api {
            unsafe fn load() -> std::result::Result<Self, String> {
                let path = std::env::var_os("JOSHUA_SYCL_LIBRARY")
                    .unwrap_or_else(|| libloading::library_filename("joshua_sycl").into());
                let library = libloading::Library::new(&path)
                    .map_err(|e| format!("cannot load SYCL bridge {path:?}: {e}; build the SYCL bridge and set JOSHUA_SYCL_LIBRARY"))?;
                let version = library.get::<unsafe extern "C" fn() -> u32>(b"joshua_sycl_abi_version\0").map_err(|e| e.to_string())?;
                if version() != 1 { return Err("SYCL bridge ABI version mismatch (expected 1)".into()); }
                $(let $name = *library.get(concat!("joshua_sycl_", stringify!($name), "\0").as_bytes()).map_err(|e| e.to_string())?;)+
                Ok(Self { _library: library, $($name,)+ })
            }
        }
    }
}
// The public wrappers below name arguments explicitly; keep function pointers
// private so the library is retained for the lifetime of every device.
api! {
    open(usize, *mut usize) -> i32;
    close(usize) -> i32;
    info(usize, *mut c_char, usize, *mut u64) -> i32;
    error() -> *const c_char;
    alloc(usize, usize, *mut usize) -> i32;
    free(usize) -> i32;
    finish(usize) -> i32;
    write(usize, usize, usize, usize, *const c_void) -> i32;
    read(usize, usize, usize, usize, *mut c_void) -> i32;
    copy(usize, usize, usize, usize, usize, usize) -> i32;
    launch(usize, *const c_char, *const Arg, usize, *const usize, *const usize) -> i32;
}
fn api() -> Result<&'static Api> {
    static API: std::sync::OnceLock<std::result::Result<Api, String>> = std::sync::OnceLock::new();
    API.get_or_init(|| unsafe { Api::load() }).as_ref().map_err(|e| Error::Msg(e.clone()))
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
