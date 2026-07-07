//! Isolation shim for NPU vendor plugins.
//!
//! Loads a `joshua_npu_*` plugin (see `joshua::npu`) in this process so that
//! a crash or hang in the vendor runtime cannot take down the main Joshua
//! server.  Speaks the newline-delimited-JSON protocol on stdin/stdout with
//! tensors in a shared memory-mapped file; spawned and supervised by
//! `joshua::npu::ShimBackend` — not intended to be run by hand.

use std::io::{BufRead, Write};
use std::sync::Arc;

use memmap2::MmapMut;

use joshua::npu::internal::{b64_decode, Request, Response, VendorLibrary};

fn main() {
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();

    // ── Init ────────────────────────────────────────────────────────────────
    let first = match lines.next() {
        Some(Ok(line)) => line,
        _ => return, // parent went away before init
    };
    let (library, model, n_ctx, shm_path) = match serde_json::from_str::<Request>(&first) {
        Ok(Request::Init {
            library,
            model,
            n_ctx,
            shm,
        }) => (library, model, n_ctx, shm),
        Ok(_) => return reply(Response::err("first request must be init")),
        Err(e) => return reply(Response::err(format!("malformed init request: {e}"))),
    };

    let mut shm = {
        let file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&shm_path)
        {
            Ok(f) => f,
            Err(e) => return reply(Response::err(format!("cannot open shm {shm_path:?}: {e}"))),
        };
        // SAFETY: the host created this private file solely for this session
        // and only touches it between requests.
        match unsafe { MmapMut::map_mut(&file) } {
            Ok(m) => m,
            Err(e) => return reply(Response::err(format!("cannot map shm: {e}"))),
        }
    };

    let lib = match VendorLibrary::open(&library) {
        Ok(lib) => Arc::new(lib),
        Err(e) => return reply(Response::err(e)),
    };
    let mut session = match lib.init(&model, n_ctx) {
        Ok(s) => s,
        Err(e) => return reply(Response::err(e)),
    };
    let vocab = session.vocab();
    let logits_base = n_ctx as usize * 4;
    if logits_base + vocab * 4 > shm.len() {
        return reply(Response::err(format!(
            "vocab {vocab} does not fit the shared logit region"
        )));
    }
    reply(Response {
        vocab: Some(vocab as u32),
        media: Some(session.has_media()),
        ..Response::ok()
    });

    // ── Serve ───────────────────────────────────────────────────────────────
    for line in lines {
        let Ok(line) = line else { break };
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(Request::Forward { pos, n_tokens }) => {
                if n_tokens == 0 || n_tokens > n_ctx {
                    Response::err(format!("invalid n_tokens {n_tokens}"))
                } else {
                    let tokens: Vec<u32> = (0..n_tokens as usize)
                        .map(|i| {
                            u32::from_le_bytes(shm[i * 4..i * 4 + 4].try_into().expect("4 bytes"))
                        })
                        .collect();
                    let mut logits = vec![0f32; vocab];
                    match session.forward_into(&tokens, pos as usize, &mut logits) {
                        Ok(()) => {
                            for (i, logit) in logits.iter().enumerate() {
                                let at = logits_base + i * 4;
                                shm[at..at + 4].copy_from_slice(&logit.to_le_bytes());
                            }
                            Response::ok()
                        }
                        Err(e) => Response::err(e),
                    }
                }
            }
            Ok(Request::MediaPrefill { prompt, images }) => {
                let decoded: Result<Vec<Vec<u8>>, String> = images
                    .iter()
                    .map(|b64| b64_decode(b64).map_err(|e| format!("bad image data: {e}")))
                    .collect();
                match decoded {
                    Err(e) => Response::err(e),
                    Ok(images) => {
                        let mut logits = vec![0f32; vocab];
                        match session.media_prefill_into(&prompt, &images, &mut logits) {
                            Ok(n_past) => {
                                for (i, logit) in logits.iter().enumerate() {
                                    let at = logits_base + i * 4;
                                    shm[at..at + 4].copy_from_slice(&logit.to_le_bytes());
                                }
                                Response {
                                    n_past: Some(n_past as u32),
                                    ..Response::ok()
                                }
                            }
                            Err(e) => Response::err(e),
                        }
                    }
                }
            }
            Ok(Request::Reset) => match session.reset() {
                Ok(()) => Response::ok(),
                Err(e) => Response::err(e),
            },
            Ok(Request::Shutdown) => break,
            Ok(Request::Init { .. }) => Response::err("already initialised"),
            Err(e) => Response::err(format!("malformed request: {e}")),
        };
        reply(response);
    }
    // session/lib drop here, freeing the vendor handle.
}

/// Write one response line to stdout.
fn reply(response: Response) {
    let mut stdout = std::io::stdout().lock();
    if let Ok(line) = serde_json::to_string(&response) {
        let _ = writeln!(stdout, "{line}");
        let _ = stdout.flush();
    }
}
