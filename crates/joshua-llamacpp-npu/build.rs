//! Build script for the llama.cpp adapter.
//!
//! Only relevant to the `dynamic-backends` feature. That build compiles
//! llama.cpp with `GGML_BACKEND_DL`, so `libggml`, `libggml-base`, and
//! `libllama` become separate versioned shared objects that the plugin
//! cdylib links against at build time. They live in `llama-cpp-sys-2`'s
//! build output directory, which is *not* on the default runtime search
//! path — so when Joshua's shim `dlopen`s the plugin, the loader cannot
//! resolve `libggml-base.so.0` and the load fails (or, worse, the engine
//! silently falls back to candle).
//!
//! To keep the feature self-contained we bake an rpath into the plugin:
//!   * `$ORIGIN` and `$ORIGIN/../lib` for deployments that ship the ggml
//!     shared objects alongside the plugin (or in a sibling `lib/`), and
//!   * the absolute `llama-cpp-sys-2` lib directory this build linked
//!     against, so local builds and the test-suite resolve the libraries
//!     with no `LD_LIBRARY_PATH` setup.
//!
//! The default (statically-linked) build never enters here, so the pure
//! CPU path keeps its zero build-script footprint.

use std::path::PathBuf;

fn main() {
    // Only the dynamic-backends build links ggml/llama as shared objects;
    // the default build statically links them, so there is nothing to find.
    if std::env::var_os("CARGO_FEATURE_DYNAMIC_BACKENDS").is_none() {
        return;
    }

    // rpath with `$ORIGIN` is an ELF/glibc concept. macOS uses a different
    // install-name scheme (`@loader_path`); Windows has no rpath at all.
    // Keep this Linux-only and let those platforms rely on the loader path.
    if !cfg!(target_os = "linux") {
        return;
    }

    // Deployment layout: ggml libraries shipped next to the plugin, or in a
    // sibling `lib/` directory. Resolved by the dynamic linker at load time,
    // so `$ORIGIN` must reach the linker literally (cargo passes link args
    // verbatim — no shell expansion).
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");

    // Local/dev layout: point straight at the versioned libraries this build
    // linked against so `cargo build -p joshua-llamacpp-npu` produces a
    // plugin that loads with no further setup.
    if let Some(lib_dir) = ggml_lib_dir() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }
}

/// Locate `.../llama-cpp-sys-2-*/out/lib` — the directory holding the
/// versioned `libggml*.so.0` / `libllama.so.0` this build linked against.
///
/// `llama-cpp-sys-2` is a transitive dependency (via `llama-cpp-2`), so its
/// `DEP_LLAMA_*` metadata is not forwarded to this crate's build script. We
/// instead find its sibling build directory next to our own `OUT_DIR`.
fn ggml_lib_dir() -> Option<PathBuf> {
    // OUT_DIR = target/<profile>/build/<this-pkg>-<hash>/out
    // sibling = target/<profile>/build/llama-cpp-sys-2-<hash>/out/lib
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR")?);
    let build_dir = out_dir.parent()?.parent()?; // target/<profile>/build

    // Several `llama-cpp-sys-2-<hash>` directories may exist for different
    // feature sets; pick the most recently built one that actually holds the
    // shared library we need.
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(build_dir).ok()?.flatten() {
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with("llama-cpp-sys-2-")
        {
            continue;
        }
        let lib = entry.path().join("out").join("lib");
        if !lib.join("libggml-base.so.0").exists() {
            continue;
        }
        let mtime = match entry.metadata().and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => continue,
        };
        if best.as_ref().is_none_or(|(t, _)| mtime > *t) {
            best = Some((mtime, lib));
        }
    }
    best.map(|(_, p)| p)
}
