//! Per-thread fault slots for the accelerator backends.
//!
//! An indexing kernel cannot return an error, so on an out-of-range id it
//! sets a word in the device's fault buffer, which the host checks and
//! clears at the next read-back or `synchronize`.  One device serves every
//! request of a server at once, so a single shared word would let one
//! thread's fault be consumed (or erased) by another's read-back.  Each
//! thread therefore owns one slot of the buffer for its lifetime: a
//! request's launches and its read-backs happen on the same thread, so the
//! error reaches the request that caused it and nobody else.
//!
//! Slots come from a fixed pool (the buffer holds [`SLOTS`] words); a
//! thread that starts after the pool is exhausted shares slot 0, which
//! degrades to the device-wide word rather than failing.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Number of slots in a device's fault buffer.
pub const SLOTS: usize = 1024;

/// Bytes of a device's fault buffer.
pub const BYTES: usize = SLOTS * 4;

static FREE: Mutex<Vec<u32>> = Mutex::new(Vec::new());
/// Next never-used slot; slot 0 is the shared fallback.
static NEXT: AtomicUsize = AtomicUsize::new(1);

struct Slot(usize);

impl Drop for Slot {
    fn drop(&mut self) {
        if self.0 != 0 {
            FREE.lock().unwrap_or_else(|p| p.into_inner()).push(self.0 as u32);
        }
    }
}

fn acquire() -> Slot {
    if let Some(s) = FREE.lock().unwrap_or_else(|p| p.into_inner()).pop() {
        return Slot(s as usize);
    }
    match NEXT.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| (n < SLOTS).then_some(n + 1)) {
        Ok(n) => Slot(n),
        Err(_) => Slot(0),
    }
}

thread_local! {
    static SLOT: Slot = acquire();
}

/// The calling thread's slot index.
pub fn current() -> usize {
    SLOT.with(|s| s.0)
}
