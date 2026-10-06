//! One owned model mapping, possibly composed of differently-sized huge pages.
//!
//! Joshua maps the GGUF once and cuts every tensor out of that single mapping
//! (see [`crate::mmap_tensor`]), so the load path needs *one* object that
//! behaves like a byte slice.  `memmap2::Mmap` cannot be that object for a
//! mixed-page mapping: it exposes no constructor for a region the caller
//! already mapped, so a mapping made of two `MAP_HUGETLB` sub-regions of
//! different sizes cannot be handed back as an `Mmap`.
//!
//! [`ModelMapping`] closes that gap.  It is either an ordinary file-backed
//! `Mmap` — today's behaviour, untouched — or a single contiguous virtual
//! range this module owns and unmaps itself.  The range may be backed by a
//! head of 1 GiB pages and a tail of 2 MiB pages, because two `MAP_FIXED`
//! sub-mappings inside one reserved range are simply addresses: a tensor may
//! straddle the boundary and the kernel walks whichever page table entry the
//! touch lands on.  Nothing about the tensors changes.
//!
//! # Why tier at all
//!
//! `PageSize::OneGiB` used to be all-or-nothing: either the whole model fits in
//! 1 GiB pages or the load fails.  Hosts rarely cooperate.  On the reference
//! machine (Ryzen 9 9950X, 80 GiB, DDR5 at 4000 MT/s) the kernel will gather at
//! most 31 of the 59 one-gibibyte pages a 58.1 GiB model needs, even after
//! `drop_caches` and `compact_memory`.  Tiering keeps the pages the host *can*
//! provide — 31 whole gibibytes of single-entry coverage — and pays for the rest
//! with 2 MiB pages, instead of throwing away all 31 because they could not
//! cover everything.
//!
//! The page-split arithmetic is in [`HugeSplit`] and is pure, so the rules that
//! decide how much of the model ends up on which page size can be tested
//! without a pool, a device, or a root-owned sysctl.

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = PAGE_1GIB;
    const MIB: u64 = PAGE_2MIB;

    /// The reference host, measured: a 62,394,667,168-byte GGUF against 31 free
    /// 1 GiB pages (the most the kernel would gather after `drop_caches` and
    /// `compact_memory`) and a large 2 MiB pool.  Tiering must keep all 31
    /// gibibytes and cover the rest with 2 MiB pages.
    #[test]
    fn plan_tiers_to_what_the_host_can_actually_gather() {
        let model = 62_394_667_168;
        let split = plan_huge_split(model, 31 * GIB, 70 * GIB).expect("both pools");
        assert_eq!(split.head_bytes, 31 * GIB, "every available 1 GiB page is used");
        assert_eq!(split.head_pages(), 31);
        assert_eq!(split.tail_bytes % MIB, 0, "tail is whole 2 MiB pages");
        assert!(
            split.total_bytes() >= model,
            "mapping covers the model: {} < {model}",
            split.total_bytes()
        );
        // All-2 MiB for the same model is 29,753 entries; tiering must do
        // materially better, which is the entire point of the change.
        let flat = model.div_ceil(MIB);
        assert!(
            split.page_entries() * 2 < flat,
            "expected >2x fewer entries: {} vs {}",
            split.page_entries(),
            flat
        );
    }

    /// No 1 GiB pages at all: the split degenerates to today's single-size
    /// 2 MiB mapping, with no head.
    #[test]
    fn plan_without_any_1gib_pool_is_all_tail() {
        let model = 62_394_667_168;
        let split = plan_huge_split(model, 0, 70 * GIB).expect("2 MiB pool alone");
        assert_eq!(split.head_bytes, 0);
        assert_eq!(split.tail_bytes, model.div_ceil(MIB) * MIB);
        assert_eq!(split.head_pages(), 0);
    }

    /// Regression: with a sufficient 1 GiB pool and **no** 2 MiB pool, the head
    /// must round *up* to cover the sub-gibibyte remainder.  Rounding it down
    /// left 0.11 GiB with nowhere to go and rejected a mapping that can work.
    #[test]
    fn plan_uses_a_final_whole_gibibyte_when_no_small_pool_exists() {
        let model = 58 * GIB + GIB / 8; // 58.125 GiB
        let split = plan_huge_split(model, 59 * GIB, 0).expect("1 GiB pool alone");
        assert_eq!(split.tail_bytes, 0, "the head absorbs the remainder");
        assert_eq!(split.head_pages(), 59);
        assert!(split.total_bytes() >= model);
        assert_eq!(split.page_entries(), 59, "59 entries for the whole model");
    }

    /// Neither pool, alone or together, can cover the model.
    #[test]
    fn plan_reports_when_the_pools_cannot_cover_the_model() {
        assert!(plan_huge_split(100 * GIB, 31 * GIB, 8 * GIB).is_none());
        // Exactly enough on one axis still has to work.
        assert!(plan_huge_split(2 * MIB, 0, 2 * MIB).is_some());
    }

    /// A model smaller than one page still yields a legal, non-empty split.
    #[test]
    fn plan_handles_a_model_smaller_than_a_page() {
        let split = plan_huge_split(1024, 0, 64 * MIB).expect("tiny model");
        assert_eq!(split.head_bytes, 0);
        assert_eq!(split.tail_bytes, MIB, "rounded to one whole 2 MiB page");
        assert_eq!(split.page_entries(), 1);
    }

    /// The head is always a legal `MAP_HUGE_1GB` sub-mapping length and the tail
    /// a legal `MAP_HUGE_2MB` one, whatever the pool sizes.
    #[test]
    fn plan_invariants_hold_across_pool_shapes() {
        for model in [1u64, MIB - 1, MIB, GIB - 1, GIB, GIB + 1, 58 * GIB + 7] {
            for a1 in [0, GIB, 31 * GIB, 64 * GIB] {
                for a2 in [0, MIB, 4 * GIB, 70 * GIB] {
                    let Some(s) = plan_huge_split(model, a1, a2) else {
                        continue;
                    };
                    assert_eq!(s.head_bytes % GIB, 0, "head is whole gibibytes");
                    assert_eq!(s.tail_bytes % MIB, 0, "tail is whole 2 MiB pages");
                    assert!(s.total_bytes() >= model);
                    assert_eq!(s.head_bytes + s.tail_bytes, s.total_bytes());
                    assert!(
                        s.head_bytes <= model.div_ceil(GIB) * GIB,
                        "head never exceeds a page per gibibyte the model needs"
                    );
                }
            }
        }
    }
}

/// Bytes in the two page sizes this module tiers between.
pub const PAGE_2MIB: u64 = 2 * 1024 * 1024;
pub const PAGE_1GIB: u64 = 1024 * 1024 * 1024;

/// How a model mapping is split between 1 GiB and 2 MiB pages.
///
/// Produced by [`plan_huge_split`] and consumed by the Linux mapping code.  The
/// head is always page-aligned to 1 GiB so a `MAP_HUGETLB` + `MAP_HUGE_1GB`
/// sub-mapping is legal at that address; the tail is a whole number of 2 MiB
/// pages, which also makes the *total* a whole number of 2 MiB pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HugeSplit {
    /// Bytes covered by 1 GiB pages at the start of the mapping.
    pub head_bytes: u64,
    /// Bytes covered by 2 MiB pages after the head.
    pub tail_bytes: u64,
}

impl HugeSplit {
    /// Total length of the mapping, and the only length `munmap` may use.
    pub fn total_bytes(&self) -> u64 {
        self.head_bytes + self.tail_bytes
    }

    /// Whole 1 GiB pages in the head.
    pub fn head_pages(&self) -> u64 {
        self.head_bytes / PAGE_1GIB
    }

    /// Whole 2 MiB pages in the tail.
    pub fn tail_pages(&self) -> u64 {
        self.tail_bytes / PAGE_2MIB
    }

    /// Approximate number of page-table entries the model mapping costs, which
    /// is the whole point: fewer entries, fewer TLB misses on a weight sweep.
    pub fn page_entries(&self) -> u64 {
        self.head_pages() + self.tail_pages()
    }
}

/// Round `n` up to a whole multiple of `page`, or `None` on overflow.
fn round_up(n: u64, page: u64) -> Option<u64> {
    n.checked_add(page - 1).map(|v| v / page * page)
}

/// Decide the 1 GiB / 2 MiB split for a model of `model_len` bytes.
///
/// `avail_1gib` and `avail_2mib` are the bytes the kernel could plausibly hand
/// out at each size (free pool plus unspent surplus — see
/// [`crate::engine::HugePool`]).
///
/// 1 GiB pages are strictly better — each one replaces 512 of the smaller — so
/// the head is taken **as large as the model needs and the pool allows**, and
/// only the remainder falls to 2 MiB pages.  The head is rounded *up* to a whole
/// gibibyte, because a model of 58.11 GiB with 59 GiB of 1 GiB pages and no 2 MiB
/// pool at all should map 59 whole pages rather than fail looking for 0.11 GiB of
/// 2 MiB pages it cannot get.
///
/// Returns `None` when the two pools together cannot cover the model, which the
/// caller turns into a clear error or a normal file-backed mapping.
///
/// Pure: no I/O, no `mmap`, no root.
pub fn plan_huge_split(model_len: u64, avail_1gib: u64, avail_2mib: u64) -> Option<HugeSplit> {
    // Minimum mapping length: a whole number of 2 MiB pages.
    let minimum = round_up(model_len, PAGE_2MIB)?;
    // Head that would cover the model on its own.
    let full_head = round_up(minimum, PAGE_1GIB)?;
    let head_bytes = full_head.min((avail_1gib / PAGE_1GIB) * PAGE_1GIB);
    let tail_capacity = (avail_2mib / PAGE_2MIB) * PAGE_2MIB;
    // The mapping must cover `minimum`; a larger head is allowed to overshoot it.
    let mapped = minimum.max(head_bytes);
    let tail_bytes = mapped - head_bytes;
    (tail_bytes <= tail_capacity).then_some(HugeSplit {
        head_bytes,
        tail_bytes,
    })
}
