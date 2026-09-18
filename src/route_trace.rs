//! Routing trace: which routed experts every forward pass visited, written
//! to a file for offline cache-policy studies ([`crate::cache_sim`]).
//!
//! Set `JOSHUA_ROUTE_TRACE=<path>` and the MoE loaders that call
//! [`begin_call`] / [`record`] (deepseek4 today) append one CSV row per
//! `(token row, layer, expert)` visit:
//!
//! ```text
//! call,phase,row,layer,expert
//! 0,p,0,0,17
//! 0,p,0,0,203
//! …
//! 1,d,0,0,17
//! ```
//!
//! `call` counts forward passes (a prefill is one call for a plain forward,
//! and one call for the whole layer-streaming sweep), `phase` is `p`
//! (prefill) or `d` (decode), `row` is the token row inside the call.  Rows
//! keep duplicates: a prefill routes many rows to the same expert and the
//! simulator wants the multiplicity.
//!
//! The writer is process-global (one trace file per process), meant for a
//! single-session run: two sessions writing at once interleave their calls.
//! Recording costs one formatted line per visit, so it is off unless the
//! variable is set.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Which kind of forward pass a call is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// A multi-token prompt pass (or one layer-streaming sweep).
    Prefill,
    /// A one-token-per-sequence step.
    Decode,
}

impl Phase {
    fn tag(self) -> char {
        match self {
            Phase::Prefill => 'p',
            Phase::Decode => 'd',
        }
    }
}

/// A routing trace writer.  [`Tracer::global`] is the env-configured one;
/// tests build their own with [`Tracer::create`].
pub struct Tracer {
    out: Mutex<std::io::BufWriter<std::fs::File>>,
    call: AtomicU64,
    phase: AtomicU8,
    rows: AtomicU64,
}

impl Tracer {
    /// Create (truncate) the trace file at `path` and write the header.
    pub fn create(path: &std::path::Path) -> std::io::Result<Self> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(out, "call,phase,row,layer,expert")?;
        Ok(Self {
            out: Mutex::new(out),
            // The first `begin_call` makes this 0.
            call: AtomicU64::new(u64::MAX),
            phase: AtomicU8::new(Phase::Decode.tag() as u8),
            rows: AtomicU64::new(0),
        })
    }

    /// The process-wide tracer: the one [`install`]ed, else the one opened
    /// from `JOSHUA_ROUTE_TRACE` on first use; `None` when the variable is
    /// unset or the file cannot be created (a warning is logged once).
    pub fn global() -> Option<Arc<Tracer>> {
        if let Some(t) = GLOBAL.read().ok().and_then(|g| g.clone()) {
            return Some(t);
        }
        if ENV_CHECKED.swap(true, Ordering::AcqRel) {
            return None;
        }
        let path = std::path::PathBuf::from(std::env::var_os("JOSHUA_ROUTE_TRACE")?);
        match Tracer::create(&path) {
            Ok(t) => {
                tracing::info!("routing trace: writing {}", path.display());
                let t = Arc::new(t);
                if let Ok(mut g) = GLOBAL.write() {
                    g.get_or_insert_with(|| Arc::clone(&t));
                }
                Some(t)
            }
            Err(e) => {
                tracing::warn!("routing trace: cannot create {}: {e}", path.display());
                None
            }
        }
    }

    /// Start a new forward pass of `phase`; rows recorded until the next
    /// call belong to it.  Flushes the previous call's rows.
    pub fn begin_call(&self, phase: Phase) {
        self.call.fetch_add(1, Ordering::Relaxed);
        self.phase.store(phase.tag() as u8, Ordering::Relaxed);
        if let Ok(mut out) = self.out.lock() {
            let _ = out.flush();
        }
    }

    /// Record one layer's routing: `ids` holds `k` expert ids per token row.
    pub fn record(&self, layer: u32, ids: &[u32], k: usize) {
        let call = self.call.load(Ordering::Relaxed);
        let phase = self.phase.load(Ordering::Relaxed) as char;
        let k = k.max(1);
        let Ok(mut out) = self.out.lock() else {
            return;
        };
        for (row, per_row) in ids.chunks(k).enumerate() {
            for &e in per_row {
                let _ = writeln!(out, "{call},{phase},{row},{layer},{e}");
            }
        }
        self.rows.fetch_add(ids.len() as u64, Ordering::Relaxed);
    }

    /// Rows written so far.
    pub fn rows(&self) -> u64 {
        self.rows.load(Ordering::Relaxed)
    }

    /// Flush buffered rows to the file.
    pub fn flush(&self) {
        if let Ok(mut out) = self.out.lock() {
            let _ = out.flush();
        }
    }
}

impl Drop for Tracer {
    fn drop(&mut self) {
        self.flush();
    }
}

static GLOBAL: RwLock<Option<Arc<Tracer>>> = RwLock::new(None);
static ENV_CHECKED: AtomicBool = AtomicBool::new(false);

/// Make `tracer` the process-wide tracer (replacing any), for callers that
/// want the trace without the environment variable (tests, embedders).
pub fn install(tracer: Arc<Tracer>) {
    if let Ok(mut g) = GLOBAL.write() {
        *g = Some(tracer);
    }
}

/// [`Tracer::begin_call`] on the global tracer, if any.
pub fn begin_call(phase: Phase) {
    if let Some(t) = Tracer::global() {
        t.begin_call(phase);
    }
}

/// [`Tracer::record`] on the global tracer, if any.
pub fn record(layer: u32, ids: &[u32], k: usize) {
    if let Some(t) = Tracer::global() {
        t.record(layer, ids, k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_one_row_per_visit_with_call_and_phase() {
        let dir = std::env::temp_dir().join(format!("joshua-trace-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trace.csv");
        let t = Tracer::create(&path).unwrap();
        t.begin_call(Phase::Prefill);
        t.record(0, &[3, 1, 3, 2], 2); // two rows, k = 2
        t.record(1, &[7, 7], 2); // one row's worth, duplicates kept
        t.begin_call(Phase::Decode);
        t.record(0, &[1, 2], 2);
        t.flush();
        assert_eq!(t.rows(), 8);
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "call,phase,row,layer,expert");
        assert_eq!(
            &lines[1..],
            &[
                "0,p,0,0,3",
                "0,p,0,0,1",
                "0,p,1,0,3",
                "0,p,1,0,2",
                "0,p,0,1,7",
                "0,p,0,1,7",
                "1,d,0,0,1",
                "1,d,0,0,2",
            ]
        );
        drop(t);
        std::fs::remove_dir_all(&dir).ok();
    }
}
