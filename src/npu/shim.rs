//! Process-isolated NPU backend host (safety layer 3).
//!
//! Spawns the `joshua-npu-shim` binary, which loads the vendor plugin in its
//! own address space.  Control runs over stdin/stdout pipes, tensors over a
//! shared memory-mapped file (see [`super::proto`]).  Every request carries
//! a deadline: on timeout, EOF, or malformed reply the child is killed and
//! the session reports itself dead — the engine then falls back to CPU and
//! a later request starts a fresh shim.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

use memmap2::MmapMut;

use super::proto::{Request, Response, SHM_LOGITS_CAPACITY};
use super::{NpuBackend, NpuSession};

/// Runs vendor plugins in an isolated subprocess per session.
pub struct ShimBackend {
    shim: PathBuf,
    library: PathBuf,
    init_timeout: Duration,
    forward_timeout: Duration,
}

static SHM_COUNTER: AtomicU64 = AtomicU64::new(0);

impl ShimBackend {
    /// Create a backend that isolates the plugin at `library` behind the
    /// shim executable at `shim`.
    ///
    /// Use [`ShimBackend::locate`] to find the shim that ships alongside the
    /// running binary.
    pub fn new(shim: impl Into<PathBuf>, library: impl Into<PathBuf>) -> Self {
        Self {
            shim: shim.into(),
            library: library.into(),
            init_timeout: Duration::from_secs(120),
            forward_timeout: Duration::from_secs(120),
        }
    }

    /// Find the shim executable: `$JOSHUA_NPU_SHIM`, then next to the
    /// current executable.
    pub fn locate(library: impl Into<PathBuf>) -> std::result::Result<Self, String> {
        if let Ok(path) = std::env::var("JOSHUA_NPU_SHIM") {
            return Ok(Self::new(path, library));
        }
        let exe = std::env::current_exe()
            .map_err(|e| format!("cannot locate current executable: {e}"))?;
        let candidate = exe.with_file_name(shim_file_name());
        if candidate.exists() {
            return Ok(Self::new(candidate, library));
        }
        Err(format!(
            "joshua-npu-shim not found (looked at {candidate:?}; set JOSHUA_NPU_SHIM)"
        ))
    }

    /// Timeout for session initialisation (model load/compile). Default 120 s.
    pub fn init_timeout(mut self, timeout: Duration) -> Self {
        self.init_timeout = timeout;
        self
    }

    /// Timeout for a single forward call. Default 120 s.
    pub fn forward_timeout(mut self, timeout: Duration) -> Self {
        self.forward_timeout = timeout;
        self
    }
}

fn shim_file_name() -> &'static str {
    if cfg!(windows) {
        "joshua-npu-shim.exe"
    } else {
        "joshua-npu-shim"
    }
}

impl NpuBackend for ShimBackend {
    fn name(&self) -> String {
        format!("npu-shim({})", self.library.display())
    }

    fn create_session(
        &self,
        model_path: &Path,
        n_ctx: u32,
    ) -> std::result::Result<Box<dyn NpuSession>, String> {
        // Shared-memory file: token region + logit region.
        let shm_path = std::env::temp_dir().join(format!(
            "joshua-npu-{}-{}.shm",
            std::process::id(),
            SHM_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let shm_len = n_ctx as usize * 4 + SHM_LOGITS_CAPACITY;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&shm_path)
            .map_err(|e| format!("cannot create shm file {shm_path:?}: {e}"))?;
        file.set_len(shm_len as u64)
            .map_err(|e| format!("cannot size shm file: {e}"))?;
        // SAFETY: private temp file created above; only this session and its
        // shim child map it, under a strict request/response protocol.
        let shm = unsafe { MmapMut::map_mut(&file) }
            .map_err(|e| format!("cannot map shm file: {e}"))?;

        let mut child = Command::new(&self.shim)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("cannot spawn shim {:?}: {e}", self.shim))?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");

        // Reader thread: forwards response lines; drops the sender on EOF so
        // recv sees the child's death immediately.
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if tx.send(line.clone()).is_err() {
                            break;
                        }
                    }
                }
            }
        });

        let mut session = ShimSession {
            child,
            stdin,
            rx,
            shm,
            shm_path,
            n_ctx,
            vocab: 0,
            dead: false,
            forward_timeout: self.forward_timeout,
        };

        let response = session.request(
            &Request::Init {
                library: self.library.clone(),
                model: model_path.to_path_buf(),
                n_ctx,
                shm: session.shm_path.clone(),
            },
            self.init_timeout,
        )?;
        let vocab = response
            .vocab
            .ok_or_else(|| "shim init reply is missing the vocab size".to_string())?;
        if vocab as usize * 4 > SHM_LOGITS_CAPACITY {
            return Err(format!("vocab {vocab} exceeds the shared logit region"));
        }
        session.vocab = vocab as usize;
        Ok(Box::new(session))
    }
}

/// One live shim subprocess hosting one vendor session.
struct ShimSession {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<String>,
    shm: MmapMut,
    shm_path: PathBuf,
    n_ctx: u32,
    vocab: usize,
    dead: bool,
    forward_timeout: Duration,
}

impl ShimSession {
    /// Send one request and wait (bounded) for its response.
    fn request(
        &mut self,
        request: &Request,
        timeout: Duration,
    ) -> std::result::Result<Response, String> {
        if self.dead {
            return Err("shim process is dead".to_string());
        }
        let line = serde_json::to_string(request).map_err(|e| e.to_string())?;
        if let Err(e) = writeln!(self.stdin, "{line}").and_then(|()| self.stdin.flush()) {
            self.kill();
            return Err(format!("shim stdin write failed: {e}"));
        }
        let reply = match self.rx.recv_timeout(timeout) {
            Ok(reply) => reply,
            Err(RecvTimeoutError::Timeout) => {
                self.kill();
                return Err(format!("shim did not respond within {timeout:?}; killed"));
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.kill();
                return Err("shim process exited unexpectedly".to_string());
            }
        };
        let response: Response = serde_json::from_str(reply.trim()).map_err(|e| {
            self.kill();
            format!("malformed shim reply: {e}")
        })?;
        if response.ok {
            Ok(response)
        } else {
            Err(response
                .error
                .unwrap_or_else(|| "unspecified shim error".to_string()))
        }
    }

    fn kill(&mut self) {
        self.dead = true;
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl NpuSession for ShimSession {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn forward(&mut self, tokens: &[u32], pos: usize) -> std::result::Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("empty token batch".to_string());
        }
        if tokens.len() > self.n_ctx as usize {
            return Err(format!(
                "{} tokens exceed the shm token region ({})",
                tokens.len(),
                self.n_ctx
            ));
        }
        // Write tokens into the shared token region.
        for (i, &token) in tokens.iter().enumerate() {
            self.shm[i * 4..i * 4 + 4].copy_from_slice(&token.to_le_bytes());
        }
        self.request(
            &Request::Forward {
                pos: pos as u32,
                n_tokens: tokens.len() as u32,
            },
            self.forward_timeout,
        )?;
        // Read logits back out of the shared logit region.
        let base = self.n_ctx as usize * 4;
        let mut logits = vec![0f32; self.vocab];
        for (i, logit) in logits.iter_mut().enumerate() {
            let at = base + i * 4;
            *logit = f32::from_le_bytes(self.shm[at..at + 4].try_into().expect("4 bytes"));
        }
        Ok(logits)
    }

    fn reset(&mut self) -> bool {
        self.request(&Request::Reset, self.forward_timeout).is_ok()
    }
}

impl Drop for ShimSession {
    fn drop(&mut self) {
        if !self.dead {
            // Best-effort clean shutdown, then make sure it's gone.
            let _ = serde_json::to_string(&Request::Shutdown)
                .map(|line| writeln!(self.stdin, "{line}"));
        }
        self.kill();
        let _ = std::fs::remove_file(&self.shm_path);
    }
}
