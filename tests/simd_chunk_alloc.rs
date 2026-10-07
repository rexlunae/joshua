//! Allocation accounting for the row-parallel matmul splitter.
//!
//! `joshua::simd::for_each_row_chunks` / `for_each_row` run once per output row
//! batch of *every* quantized matmul, so anything they allocate is multiplied
//! by the model's layer/expert count on every token.  The chunking must be
//! free: the rows are handed to the worker as a `Range<usize>`, derived from
//! the chunk index, with no per-chunk `Vec<usize>` (which is what rayon's
//! `chunks()` adaptor materialises).
//!
//! This binary installs a counting `#[global_allocator]`.  Counting is opt-in
//! (`count_allocs`) and off by default, so every other test in the binary pays
//! only a relaxed atomic load per allocation, and the rayon workers' work is
//! captured too (the counters are process-global, not thread-local).
//!
//! `baseline_chunks` keeps a copy of the pre-fix `chunks()`-based splitter so
//! the "before" numbers are measured in the same process, on the same host, by
//! the same allocator — and so this file still compiles against the old tree.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use candle_core::quantized::k_quants::BlockQ6K;
use candle_core::quantized::GgmlDType;
use candle_core::{Device, Tensor};
use joshua::quant_matmul::matmul_kquant;
use joshua::simd::{for_each_row, for_each_row_chunks};
use rayon::prelude::*;

// ── Counting allocator ────────────────────────────────────────────────────

static RECORDING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
/// Allocations of at most 32 bytes — where the per-chunk `Vec<usize>` lands
/// for chunks of ≤4 rows, and the bucket a decode step's smallest garbage
/// lives in.
static TINY: AtomicUsize = AtomicUsize::new(0);
/// Allocations of 33..=1024 bytes — a chunk `Vec<usize>` of 5..128 rows.
static SMALL: AtomicUsize = AtomicUsize::new(0);
/// Allocations of 1025..=65536 bytes — chunk `Vec`s of larger chunks, and
/// rayon's own per-`install` job stacks.
static MEDIUM: AtomicUsize = AtomicUsize::new(0);
/// Anything larger.
static BIG: AtomicUsize = AtomicUsize::new(0);

struct Counting;

impl Counting {
    #[inline]
    fn note(size: usize) {
        if RECORDING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size, Ordering::Relaxed);
            match size {
                0..=32 => TINY.fetch_add(1, Ordering::Relaxed),
                33..=1024 => SMALL.fetch_add(1, Ordering::Relaxed),
                1025..=65536 => MEDIUM.fetch_add(1, Ordering::Relaxed),
                _ => BIG.fetch_add(1, Ordering::Relaxed),
            };
        }
    }
}

// SAFETY: every method forwards to `System` unchanged; the counters only ever
// observe sizes, never the pointers.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        Self::note(l.size());
        unsafe { System.alloc(l) }
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        Self::note(l.size());
        unsafe { System.alloc_zeroed(l) }
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        Self::note(new);
        unsafe { System.realloc(p, l, new) }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The counters are process-global, so the census tests take this lock: the
/// default test harness runs them concurrently and another test's allocations
/// would otherwise land in the window.
fn census_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// An allocation census of `f`: total `(count, bytes)` plus the per-size-bucket
/// breakdown.  `install`-level rayon bookkeeping (job stacks, registry slots)
/// is included — it is part of the call's cost, but it does not scale with `n`,
/// which is what the assertions below lean on.
#[derive(Clone, Copy, Debug, Default)]
struct Census {
    count: usize,
    bytes: usize,
    tiny: usize,
    small: usize,
    medium: usize,
    big: usize,
}

impl Census {
    /// Allocations of at most 1 KiB — the size band every per-chunk `Vec<usize>`
    /// falls in for any chunk up to 128 rows.
    fn small_alloc(&self) -> usize {
        self.tiny + self.small
    }
}

fn count_allocs(f: impl FnOnce()) -> Census {
    for c in [&ALLOCS, &BYTES, &TINY, &SMALL, &MEDIUM, &BIG] {
        c.store(0, Ordering::SeqCst);
    }
    RECORDING.store(true, Ordering::SeqCst);
    f();
    RECORDING.store(false, Ordering::SeqCst);
    Census {
        count: ALLOCS.load(Ordering::SeqCst),
        bytes: BYTES.load(Ordering::SeqCst),
        tiny: TINY.load(Ordering::SeqCst),
        small: SMALL.load(Ordering::SeqCst),
        medium: MEDIUM.load(Ordering::SeqCst),
        big: BIG.load(Ordering::SeqCst),
    }
}

/// Low-water census over `reps` runs.
///
/// Rayon's `Registry` grows a thread's cached job stack as a *one-time*
/// event when the stack first needs to be deeper — that growth can land
/// inside a measured window even after warm-up installs (observed on
/// macOS: 6-19 small blocks at n=8/64), but it does not repeat run after
/// run.  A real regression — a per-chunk `Vec<usize>` — allocates on
/// *every* run.  Taking the minimum census therefore keeps the guard
/// deterministic against one-time pool bookkeeping without loosening the
/// bounds it actually enforces.
fn count_allocs_min(reps: usize, mut f: impl FnMut()) -> Census {
    let mut best: Option<Census> = None;
    for _ in 0..reps {
        let c = count_allocs(&mut f);
        best = Some(match best {
            None => c,
            Some(b) if c.count <= b.count => c,
            Some(b) => b,
        });
    }
    best.unwrap()
}

/// The row-parallel pool's chunk count, mirroring `joshua::simd`'s
/// ~4-chunks-per-thread split.
fn pool_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// How many times `joshua::simd`'s splitter calls `f` for `n` rows.
///
/// Mirrors the splitter exactly, including its serial cutoff: with one worker
/// thread (or below `n < 8`) it hands the whole range to a single call rather
/// than splitting.  A host that reports `available_parallelism() == 1` therefore
/// gets one call and one scratch buffer, not a chunk count derived from a
/// parallel split it never takes — assuming the parallel shape there would fail
/// on single-core CI for no production reason.
fn chunk_count(n: usize) -> usize {
    if pool_threads() <= 1 || n < 8 {
        return 1;
    }
    n.div_ceil((n / (pool_threads() * 4)).max(1))
}

// ── The splitter as it was before the fix ─────────────────────────────────

/// The row-parallel pool, built once like `joshua::simd`'s, so the census
/// measures the split and not a pool bootstrap.
fn baseline_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::LazyLock<rayon::ThreadPool> = std::sync::LazyLock::new(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads())
            .build()
            .expect("baseline pool")
    });
    &POOL
}

/// Verbatim pre-fix chunking: rayon's `chunks()` hands the worker a `Vec`.
fn baseline_chunks(n: usize, f: impl Fn(&[usize]) + Send + Sync) {
    if pool_threads() <= 1 || n < 8 {
        let rows: Vec<usize> = (0..n).collect();
        f(&rows);
        return;
    }
    let chunk = (n / (pool_threads() * 4)).max(1);
    baseline_pool().install(|| {
        (0..n).into_par_iter().chunks(chunk).for_each(|rows| f(&rows));
    });
}

// ── Tests ─────────────────────────────────────────────────────────────────

/// The chunked splitter must not allocate on the chunk account: the ≤1 KiB
/// bands (where every per-chunk `Vec<usize>` lands for any chunk up to 128
/// rows) may hold only a small bounded residual, and the total must not grow
/// with `n`.  A strict zero is not a contract rayon offers: its `Registry`
/// may re-allocate a cached job stack inside the measured window — the exact
/// flake that motivated #153 — so this test bounds the residual the same way
/// `for_each_row_allocates_nothing` does and relies on the baseline
/// comparisons and the drift bound below to keep the guard real.
///
/// The old `chunks()` splitter is measured in the same process for the
/// "before" number.
#[test]
fn row_chunk_split_allocates_nothing() {
    let _guard = census_guard();
    // Warm the pool (thread stacks, job queues) before recording.
    for_each_row_chunks(64, |_| {});
    for_each_row_chunks(64, |_| {});
    baseline_chunks(64, |_| {});

    let mut new_totals = Vec::new();
    for n in [8usize, 64, 256, 1024, 4096, 16384] {
        let new = count_allocs_min(3, || for_each_row_chunks(n, |_| {}));
        // The ≤1 KiB bands are where a per-chunk `Vec<usize>` of up to 128
        // rows lands.  This is the low-water mark of three runs: rayon's
        // one-time job-stack growth can land inside any single window (see
        // `count_allocs_min` — observed on macOS as 6-19 small blocks), but
        // it cannot mask a regression that allocates on every rep.  The
        // bound matches `for_each_row_allocates_nothing`'s on purpose — a
        // strict zero here was exactly the latent flake Devin flagged on
        // this PR: the same `Registry` behaviour threatens both tests.
        assert!(
            new.small_alloc() <= 4,
            "for_each_row_chunks n={n} allocated {} small blocks: {new:?}",
            new.small_alloc()
        );
        assert!(new.count <= 8 && new.bytes <= 4096, "for_each_row_chunks n={n}: {new:?}");
        new_totals.push(new.count);

        // Before: one `Vec<usize>` per chunk (`Vec::from_iter` over rayon's
        // chunk sequence), plus a constant handful of rayon allocations.
        let old = count_allocs_min(3, || baseline_chunks(n, |_| {}));
        assert!(
            old.count >= chunk_count(n),
            "baseline n={n}: expected >= {} chunk Vecs, census {old:?}",
            chunk_count(n)
        );
        // The head-to-head comparisons only carry signal once the per-chunk
        // `Vec`s outweigh the constant install bookkeeping both sides pay;
        // at n=8/64 every chunk holds 1-8 rows (8-64-byte `Vec`s) and the
        // two censuses legitimately interleave — that is where the old
        // unconditional comparison flaked.  `old.count >= chunk_count(n)`
        // above still pins the regression shape at those sizes.
        if n >= 256 {
            assert!(
                old.count > new.count,
                "baseline n={n} ({old:?}) should allocate more than the range splitter ({new:?})"
            );
            assert!(
                old.bytes > new.bytes,
                "baseline n={n} ({old:?}) should move more bytes than the range splitter ({new:?})"
            );
        }
    }
    // What is left is rayon's `install` machinery: a constant that must not
    // track `n`.
    let lo = *new_totals.iter().min().unwrap();
    let hi = *new_totals.iter().max().unwrap();
    assert!(hi <= lo + 16, "alloc total drifts with n: {new_totals:?}");
}

/// Same for the per-row entry point every fused kernel (`kquant_dot`,
/// `raw_block`) goes through.
#[test]
fn for_each_row_allocates_nothing() {
    let _guard = census_guard();
    // Warm the pool the way `row_chunk_split_allocates_nothing` does — two
    // installs, not one.  A single warm call leaves rayon free to grow its
    // cached job stack inside the *measured* window, which is what made this
    // test fail intermittently (observed: 3 small blocks at n=8) rather than
    // report anything about the splitter.
    for_each_row(64, |_| {});
    for_each_row(64, |_| {});
    baseline_chunks(64, |_| {});

    let mut new_totals = Vec::new();
    for n in [8usize, 64, 256, 1024, 4096, 16384] {
        // Low-water of three runs: a one-time job-stack growth inside a
        // measured window is what made this census fail on macOS (observed:
        // 8-19 small blocks at n=8/64 even after the double warm-up), while
        // a per-chunk regression recurs every single rep — see
        // `count_allocs_min`.
        let census = count_allocs_min(3, || for_each_row(n, |_| {}));
        // The property is "no allocation tracks the chunk count" — that is what
        // the per-chunk `Vec<usize>` used to do, and what the range splitter
        // removed.  A *strict* zero is not a contract rayon offers: its `Registry`
        // may re-allocate a cached job stack, and that residual lands in the same
        // size bands (see the comment in `row_chunk_split_allocates_nothing`).
        // So bound the residual, then prove below that it does not grow with `n`.
        //
        // At n=16384 a reintroduced per-chunk `Vec` costs hundreds of allocations,
        // so this bound still fails loudly on the actual regression.
        assert!(
            census.small_alloc() <= 4,
            "for_each_row n={n} allocated {} small blocks: {census:?}",
            census.small_alloc()
        );
        assert!(
            census.count <= 8 && census.bytes <= 4096,
            "for_each_row n={n} allocated too much: {census:?}"
        );
        new_totals.push(census.count);
    }
    // The load-bearing check: whatever rayon's constant residual is, it must not
    // track `n`.  A per-chunk allocation would scale with `chunk_count(n)` and
    // blow this apart long before it violated the absolute bounds above.
    let lo = *new_totals.iter().min().unwrap();
    let hi = *new_totals.iter().max().unwrap();
    assert!(hi <= lo + 16, "for_each_row alloc total drifts with n: {new_totals:?}");
}

/// End to end: a decode-shaped quantized matmul allocates exactly one
/// scratch buffer per chunk (the per-task dequantization row) and nothing
/// else.  The count tracks the chunk split, not `n`.
#[test]
fn quantized_matmul_allocates_only_per_chunk_scratch() {
    let _guard = census_guard();
    const K: usize = 1024;
    const M: usize = 1;
    let blocks: Vec<BlockQ6K> = {
        let data: Vec<f32> = (0..512 * K)
            .map(|i| ((((i as u64) * 2654435761) % 100000) as f32 / 1000.0) - 50.0)
            .collect();
        let t = Tensor::from_vec(data, (512, K), &Device::Cpu).unwrap();
        let qt = candle_core::quantized::QTensor::quantize(&t, GgmlDType::Q6K).unwrap();
        let bytes = qt.data().unwrap();
        assert!(bytes.len().is_multiple_of(std::mem::size_of::<BlockQ6K>()));
        let ptr = bytes.as_ptr() as *const BlockQ6K;
        let len = bytes.len() / std::mem::size_of::<BlockQ6K>();
        // SAFETY: `data()` is candle's own quantized storage, T-aligned, and
        // every byte pattern of a k-quant block is a valid block.
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    };
    let lhs: Vec<f32> = (0..M * K).map(|i| ((i * 40503) % 997) as f32 / 317.0 - 1.5).collect();

    // Warm up so the pool's one-time allocations are not counted.
    let mut warm = vec![0f32; M * 512];
    matmul_kquant::<BlockQ6K>((M, K, 512), &lhs, &blocks, &mut warm).unwrap();

    let mut dst = vec![0f32; M * 512];
    // Low-water of three runs: the strict equalities below are exactly the
    // shape rayon's one-time job-stack growth breaks — the stack lands in
    // the medium band, and this test flaked ~3/10 at the unmodified PR head
    // on macOS — while a stray scratch buffer per chunk would recur in
    // every rep.  See `count_allocs_min`.
    let census = count_allocs_min(3, || {
        matmul_kquant::<BlockQ6K>((M, K, 512), &lhs, &blocks, &mut dst).unwrap()
    });
    // One `vec![0f32; k]` (4 KiB) scratch per chunk; no per-chunk `Vec<usize>`
    // on top of it, so the ≤1 KiB bands stay empty.
    assert_eq!(
        census.small_alloc(),
        0,
        "matmul left {} small allocations: {census:?}",
        census.small_alloc()
    );
    assert_eq!(
        census.medium,
        chunk_count(512),
        "one 4 KiB dequant scratch per chunk, census {census:?} (chunks={})",
        chunk_count(512)
    );
    assert!(dst.iter().any(|v| *v != 0.0), "sanity: the matmul did compute");
}

/// Reported with numbers so a regression shows up in the log, not just as a
/// failed assertion.
#[test]
fn report_alloc_table() {
    let _guard = census_guard();
    println!("threads = {}", pool_threads());
    for n in [8usize, 64, 256, 1024, 4096, 16384] {
        for_each_row_chunks(n, |_| {});
        baseline_chunks(n, |_| {});
        let now = count_allocs(|| for_each_row_chunks(n, |_| {}));
        let old = count_allocs(|| baseline_chunks(n, |_| {}));
        println!(
            "n={n:<6} chunks={:<4} new: count={:<3} <=1KiB={:<3} <=64KiB={:<3} bytes={:<7} |              old: count={:<4} <=1KiB={:<4} <=64KiB={:<4} bytes={}",
            chunk_count(n),
            now.count,
            now.small_alloc(),
            now.medium + now.big,
            now.bytes,
            old.count,
            old.small_alloc(),
            old.medium + old.big,
            old.bytes,
        );
    }
}

/// **Microbenchmark only** — no model, no decode loop: the cost of one
/// `for_each_row_chunks` call, range-splitter vs the pre-fix `chunks()`
/// splitter.  Two workers are measured:
///
/// * `empty` — `f` does nothing, so this is the dispatch + chunk-vector cost
///   alone,
/// * `scratch` — `f` allocates and touches one `Vec<f32>` per chunk and does a
///   little arithmetic per row, i.e. the shape of
///   `quant_matmul::try_simd_kquant`'s worker (one dequantized weight row per
///   task).
///
/// The two splitters use *different* rayon pools (the new one the global pool,
/// the baseline its own), so running both in one process would charge the
/// baseline for pool-thread wake-ups the other never pays.  Run each in its own
/// process instead:
///
/// ```sh
/// JOSHUA_SPLITTER=new cargo test --release --test simd_chunk_alloc report_splitter_timing -- --nocapture
/// JOSHUA_SPLITTER=old cargo test --release --test simd_chunk_alloc report_splitter_timing -- --nocapture
/// ```
///
/// (Rounds are interleaved and the minimum is reported; this host's run-to-run
/// spread is around 10%, so only large deltas mean anything.)
#[test]
fn report_splitter_timing() {
    let _guard = census_guard();
    const K: usize = 1024;
    let mode = std::env::var("JOSHUA_SPLITTER").unwrap_or_else(|_| "both".into());

    let empty_new = |n: usize| {
        for_each_row_chunks(n, |_| {});
    };
    let empty_old = |n: usize| {
        baseline_chunks(n, |_| {});
    };
    let sink = std::sync::atomic::AtomicUsize::new(0);
    let scratch_new = |n: usize| {
        for_each_row_chunks(n, |rows| {
            let mut wrow = vec![0f32; K];
            for r in rows {
                wrow[0] = r as f32;
                wrow[K - 1] = sink.load(std::sync::atomic::Ordering::Relaxed) as f32;
            }
        });
    };
    // Diagnostic: the new indexed dispatch *plus* one `Vec<usize>` per chunk,
    // i.e. `new + the allocations the old code did`.  Comparing this against
    // `new` isolates the malloc cost from rayon's `chunks()` machinery.
    let vec_new = |n: usize| {
        let chunk = (n / (pool_threads() * 4)).max(1);
        let nchunks = n.div_ceil(chunk);
        for_each_row_chunks(nchunks, |_| {
            let v: Vec<usize> = (0..chunk).collect();
            std::hint::black_box(v.len());
        });
    };
    let scratch_old = |n: usize| {
        baseline_chunks(n, |rows| {
            let mut wrow = vec![0f32; K];
            for &r in rows {
                wrow[0] = r as f32;
                wrow[K - 1] = sink.load(std::sync::atomic::Ordering::Relaxed) as f32;
            }
        });
    };

    println!("splitter = {mode}, threads = {}", pool_threads());
    println!(
        "{:>7} {:>8} {:>14} {:>14}",
        "n", "chunks", "empty ns/call", "scratch ns/call"
    );
    for n in [16usize, 32, 64, 128, 256, 1024, 4096, 16384] {
        // Warm this process's pool for the variants it will time.
        if mode != "old" {
            scratch_new(n);
            empty_new(n);
        }
        if mode != "new" {
            scratch_old(n);
            empty_old(n);
        }
        if mode == "vec" {
            vec_new(n);
        }
        let rounds = 3;
        let iters_empty = 500;
        let iters_scratch = 50;
        let (mut be_new, mut bo_new) = (f64::INFINITY, f64::INFINITY);
        let (mut be_old, mut bo_old) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..rounds {
            if mode != "old" {
                be_new = be_new.min(time_ns(iters_empty, || empty_new(n)));
                bo_new = bo_new.min(time_ns(iters_scratch, || scratch_new(n)));
            }
            if mode != "new" {
                be_old = be_old.min(time_ns(iters_empty, || empty_old(n)));
                bo_old = bo_old.min(time_ns(iters_scratch, || scratch_old(n)));
            }
            if mode == "vec" {
                be_new = be_new.min(time_ns(iters_empty, || vec_new(n)));
                bo_new = bo_new.min(time_ns(iters_scratch, || vec_new(n)));
            }
        }
        let show = |v: f64| if v.is_finite() { format!("{v:.0}") } else { "-".into() };
        println!(
            "{n:>7} {:>8} {:>14} {:>14}",
            chunk_count(n),
            show(if mode == "old" { be_old } else { be_new }),
            show(if mode == "old" { bo_old } else { bo_new }),
        );
    }
}

fn time_ns<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    let t = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() / iters as f64 * 1e9
}
