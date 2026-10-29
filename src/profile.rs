//! Coherent memory profiles: one name for a whole tier plan.
//!
//! The engine exposes a dozen memory knobs (`--pin-hot-experts`,
//! `--expert-cache`, `--vram-expert-cache`, `--ssd-expert-cache`,
//! `--engram-tier`, `--expert-placement`, …), each with its own `auto|<n>`
//! grammar and its own unit.  Individually they are fine; the *combination* is
//! what actually decides whether a model serves well, and nothing in the CLI
//! states that combination or checks it against the machine.
//!
//! [`MemoryProfile`] names a coherent plan and [`resolve_profile`] turns it into
//! the concrete per-tier settings for a given machine and model, so:
//!
//! * a user who knows the shape of their box (`fits VRAM` / `fits RAM` /
//!   `larger than RAM`) can say so once instead of tuning five flags;
//! * `auto` values are resolved **once, at load**, from measured numbers and
//!   logged, rather than being re-derived implicitly in several places;
//! * an incoherent request is caught at load (naming both values), not at the
//!   first request that OOMs.
//!
//! Precedence is deliberately conservative: a profile only supplies values for
//! settings the operator left unset.  An explicit flag always wins, so adding
//! `--profile` cannot change the behaviour of an existing tuned invocation.

use crate::placement::{EngramTier, ExpertPlacement, SsdExpertCache, VramExpertCache};

/// The share of free RAM a profile may hand to the host expert pool.
///
/// The pool is the Tier-2 cache in front of the SSD tiers; leaving a share for
/// the page cache, KV state and scratch is what keeps a "fits RAM" profile from
/// turning every soft page into a stall.  The engine's own
/// `CACHE_MEMORY_FRACTION` (0.75) is the precedent for the hot-expert budget;
/// the profile uses the same conservative fraction so the two cannot disagree.
pub const PROFILE_HOST_POOL_SHARE: f64 = 0.75;

/// A coherent memory plan, named on the command line with `--profile`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryProfile {
    /// Decide from the machine and the model file.  The default.
    #[default]
    Auto,
    /// The whole model fits the accelerator: keep the routed experts there too.
    Fit,
    /// The model fits in host RAM but not VRAM: experts in RAM, no SSD slab.
    Host,
    /// The model is larger than RAM: pooled hot experts in RAM, the rest
    /// streamed from SSD by the Tier-3 slab, Engram left on disk.
    SsdBacked,
}

impl MemoryProfile {
    /// The stable name (`--profile <name>`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fit => "fit",
            Self::Host => "host",
            Self::SsdBacked => "ssd-backed",
        }
    }
}

impl std::str::FromStr for MemoryProfile {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "fit" | "vram" | "gpu" => Ok(Self::Fit),
            "host" | "cpu" | "ram" => Ok(Self::Host),
            "ssd-backed" | "ssd" | "disk" | "stream" => Ok(Self::SsdBacked),
            other => Err(format!(
                "unknown profile `{other}` (expected auto, fit, host or ssd-backed)"
            )),
        }
    }
}

/// What a machine and model file look like to the resolver.
///
/// All sizes are bytes; `None` means "not measured" and is treated
/// conservatively (never assumed large).  This mirrors how the engine already
/// treats a missing device probe: no evidence means no promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MachineFit {
    /// The model file's size on disk.
    pub model_bytes: u64,
    /// Free accelerator memory at load, when a probe or `--vram-budget` exists.
    pub device_bytes: Option<u64>,
    /// The dense (always-touched) set's device footprint, an upper bound.
    pub dense_device_bytes: u64,
    /// Available host RAM at load, when measurable.
    pub available_ram_bytes: Option<u64>,
}

/// The resolved per-tier settings a profile produces.
///
/// Every field is *the same type the existing flag parses*, so applying a plan
/// is a matter of assigning the unset options — no new tier vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPlan {
    /// Which profile produced this (or `Auto`'s resolved choice).
    pub profile: MemoryProfile,
    /// Where the routed experts run.
    pub placement: ExpertPlacement,
    /// The bounded VRAM expert-cache setting (Tier 1).
    pub vram_expert_cache: VramExpertCache,
    /// The Tier-3 SSD staging-slab setting.
    pub ssd_expert_cache: SsdExpertCache,
    /// Engram residency policy.
    pub engram_tier: EngramTier,
}

/// Whether the **whole model** fits the device with headroom.
///
/// `Fit` keeps the routed experts on the accelerator too, so it needs the whole
/// file — not just the dense set — to fit.  (Checking the dense set alone would
/// classify a 200 GiB model on a 24 GiB card as `fit`, which is exactly the
/// mistake this profile exists to prevent.)
fn model_fits_device(fit: &MachineFit) -> bool {
    let Some(device) = fit.device_bytes else {
        return false;
    };
    // The same 1 GiB of scratch the placement check reserves.
    const HEADROOM: u64 = 1 << 30;
    match fit.model_bytes.checked_add(HEADROOM) {
        Some(required) => required <= device,
        None => false,
    }
}

/// Whether the dense set — which must live on the device either way — fits it.
///
/// Distinct from [`model_fits_device`]: a model far larger than VRAM can still
/// serve with its experts on the host as long as this holds.
fn dense_fits_device(fit: &MachineFit) -> bool {
    let Some(device) = fit.device_bytes else {
        return false;
    };
    const HEADROOM: u64 = 1 << 30;
    match fit.dense_device_bytes.checked_add(HEADROOM) {
        Some(required) => required <= device,
        None => false,
    }
}

/// Whether the whole model (weights) would fit in host RAM.
fn model_fits_ram(fit: &MachineFit) -> bool {
    fit.available_ram_bytes
        .is_some_and(|ram| fit.model_bytes.saturating_sub(ram) == 0)
}

/// Resolve `profile` into concrete settings for `fit`.
///
/// Pure: every input is a number (or `None`), so the rule is unit-testable
/// without a device, a mapping or a live process.  `Auto` picks the narrowest
/// plan the machine can support, in the order fit → host → ssd-backed.
pub fn resolve_profile(profile: MemoryProfile, fit: &MachineFit) -> MemoryPlan {
    let resolved = match profile {
        MemoryProfile::Auto => {
            if model_fits_device(fit) {
                MemoryProfile::Fit
            } else if model_fits_ram(fit) && dense_fits_device(fit) {
                // Experts go to RAM, but the dense set still has to fit VRAM.
                MemoryProfile::Host
            } else {
                MemoryProfile::SsdBacked
            }
        }
        explicit => explicit,
    };

    match resolved {
        // Everything can live on the accelerator: keep the experts there and
        // keep Engram in RAM (the machine has demonstrated it has the room).
        MemoryProfile::Fit => MemoryPlan {
            profile: resolved,
            placement: ExpertPlacement::Device,
            vram_expert_cache: VramExpertCache::Auto,
            ssd_expert_cache: SsdExpertCache::Disabled,
            engram_tier: EngramTier::Ram,
        },
        // Experts in host RAM; no SSD slab (the page cache is enough when the
        // model already fits), Engram promoted only when RAM really is ample
        // (the tier's own floor decides — `Auto`).
        MemoryProfile::Host => MemoryPlan {
            profile: resolved,
            placement: ExpertPlacement::Host,
            vram_expert_cache: VramExpertCache::Disabled,
            ssd_expert_cache: SsdExpertCache::Disabled,
            engram_tier: EngramTier::Auto,
        },
        // Larger than RAM: the tiered plan.  Hot experts pooled in RAM, VRAM
        // cache engaged automatically where the device has room, the rest
        // streamed by the Tier-3 slab, and Engram left on disk (it is a sparse
        // gather and would crowd out experts).
        MemoryProfile::SsdBacked => MemoryPlan {
            profile: resolved,
            placement: ExpertPlacement::Host,
            vram_expert_cache: VramExpertCache::Auto,
            ssd_expert_cache: SsdExpertCache::Auto,
            engram_tier: EngramTier::Disk,
        },
        // `Auto` never survives the match above.
        MemoryProfile::Auto => unreachable!("Auto is resolved before the plan is built"),
    }
}

/// Apply `plan` to the operator's explicit choices, letting each explicit value
/// win.  `None` in the option means "unset, take the profile's value".
///
/// Kept separate from [`resolve_profile`] so the precedence rule is explicit and
/// testable independently of the tier policy.
pub fn apply_plan(
    plan: &MemoryPlan,
    placement: Option<ExpertPlacement>,
    vram_expert_cache: Option<VramExpertCache>,
    ssd_expert_cache: Option<SsdExpertCache>,
    engram_tier: Option<EngramTier>,
) -> (ExpertPlacement, VramExpertCache, SsdExpertCache, EngramTier) {
    (
        placement.unwrap_or(plan.placement),
        vram_expert_cache.unwrap_or(plan.vram_expert_cache),
        ssd_expert_cache.unwrap_or(plan.ssd_expert_cache),
        engram_tier.unwrap_or(plan.engram_tier),
    )
}

/// One line describing the plan, for the load log (the engine already logs the
/// resolved VRAM budget this way, so the effective plan is auditable).
pub fn describe(plan: &MemoryPlan, fit: &MachineFit) -> String {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let ram = match fit.available_ram_bytes {
        Some(b) => format!("{:.1} GiB free RAM", gib(b)),
        None => "RAM unmeasured".to_string(),
    };
    format!(
        "profile `{}`: placement {:?}, vram-expert-cache {:?}, ssd-expert-cache {:?}, \
         engram {:?} ({:.1} GiB model, {})",
        plan.profile.as_str(),
        plan.placement,
        plan.vram_expert_cache,
        plan.ssd_expert_cache,
        plan.engram_tier,
        gib(fit.model_bytes),
        ram
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn machine(model_gib: u64, device_gib: Option<u64>, ram_gib: Option<u64>) -> MachineFit {
        MachineFit {
            model_bytes: model_gib * GIB,
            device_bytes: device_gib.map(|g| g * GIB),
            dense_device_bytes: 12 * GIB,
            available_ram_bytes: ram_gib.map(|g| g * GIB),
        }
    }

    #[test]
    fn profile_names_parse_and_reject() {
        assert_eq!("auto".parse::<MemoryProfile>().unwrap(), MemoryProfile::Auto);
        assert_eq!("FIT".parse::<MemoryProfile>().unwrap(), MemoryProfile::Fit);
        assert_eq!("host".parse::<MemoryProfile>().unwrap(), MemoryProfile::Host);
        assert_eq!(
            "ssd-backed".parse::<MemoryProfile>().unwrap(),
            MemoryProfile::SsdBacked
        );
        assert_eq!("ssd".parse::<MemoryProfile>().unwrap(), MemoryProfile::SsdBacked);
        assert!("sideways".parse::<MemoryProfile>().is_err());
    }

    /// A model that fits VRAM gets the `fit` plan: experts on the device, no
    /// SSD slab, Engram in RAM.
    #[test]
    fn auto_picks_fit_when_the_model_fits_the_device() {
        let plan = resolve_profile(MemoryProfile::Auto, &machine(40, Some(80), Some(120)));
        assert_eq!(plan.profile, MemoryProfile::Fit);
        assert_eq!(plan.placement, ExpertPlacement::Device);
        assert_eq!(plan.ssd_expert_cache, SsdExpertCache::Disabled);
        assert_eq!(plan.engram_tier, EngramTier::Ram);
    }

    /// Fits RAM but not VRAM → `host`: experts in RAM, no SSD slab.
    #[test]
    fn auto_picks_host_when_the_model_fits_ram_only() {
        let plan = resolve_profile(MemoryProfile::Auto, &machine(200, Some(24), Some(256)));
        assert_eq!(plan.profile, MemoryProfile::Host);
        assert_eq!(plan.placement, ExpertPlacement::Host);
        assert_eq!(plan.ssd_expert_cache, SsdExpertCache::Disabled);
        // Engram is left to its own floor rather than promoted blindly.
        assert_eq!(plan.engram_tier, EngramTier::Auto);
    }

    /// Larger than RAM → the tiered plan: SSD slab on, Engram on disk.
    #[test]
    fn auto_picks_ssd_backed_when_the_model_exceeds_ram() {
        let plan = resolve_profile(MemoryProfile::Auto, &machine(397, Some(16), Some(74)));
        assert_eq!(plan.profile, MemoryProfile::SsdBacked);
        assert_eq!(plan.placement, ExpertPlacement::Host);
        assert_eq!(plan.vram_expert_cache, VramExpertCache::Auto);
        assert_eq!(plan.ssd_expert_cache, SsdExpertCache::Auto);
        assert_eq!(plan.engram_tier, EngramTier::Disk);
    }

    /// A model far larger than VRAM is never `fit`, even though its dense set
    /// would fit: `fit` keeps the routed experts on the card too.  This is the
    /// bug the first version of the resolver had.
    #[test]
    fn a_model_larger_than_vram_is_never_fit() {
        // 200 GiB of weights, 24 GiB of VRAM, plenty of RAM, dense set small.
        let plan = resolve_profile(MemoryProfile::Auto, &machine(200, Some(24), Some(256)));
        assert_eq!(
            plan.profile,
            MemoryProfile::Host,
            "200 GiB cannot be `fit` behind 24 GiB of VRAM"
        );
        assert_ne!(plan.placement, ExpertPlacement::Device);
    }

    /// A model that fits neither VRAM nor RAM, and whose dense set still fits
    /// VRAM, is the tiered case.
    #[test]
    fn dense_fits_vram_is_required_for_host_too() {
        // 8 GiB device cannot hold a 12 GiB dense set: not `host` either.
        let plan = resolve_profile(MemoryProfile::Auto, &machine(100, Some(8), Some(256)));
        assert_eq!(plan.profile, MemoryProfile::SsdBacked);
    }

    /// An unmeasured machine never claims to fit: no device probe means not
    /// `fit`, and no RAM figure means not `host`.
    #[test]
    fn auto_is_conservative_without_measurements() {
        let plan = resolve_profile(MemoryProfile::Auto, &machine(10, None, None));
        assert_eq!(plan.profile, MemoryProfile::SsdBacked);

        // No RAM figure but the model genuinely fits the card: `fit` needs no
        // RAM measurement at all, so this one *is* `fit`.  (Only the plans that
        // place experts in RAM depend on the RAM number.)
        let no_ram = resolve_profile(MemoryProfile::Auto, &machine(10, Some(80), None));
        assert_eq!(no_ram.profile, MemoryProfile::Fit);

        // Fits neither the card (200 GiB) nor a measured RAM figure (absent):
        // the tiered plan, never `host`.
        let big = resolve_profile(MemoryProfile::Auto, &machine(200, Some(24), None));
        assert_eq!(big.profile, MemoryProfile::SsdBacked);
    }

    /// An explicit profile is honoured regardless of what the machine reports.
    #[test]
    fn explicit_profiles_ignore_the_measurements() {
        let tiny = machine(1, Some(80), Some(512));
        assert_eq!(
            resolve_profile(MemoryProfile::SsdBacked, &tiny).profile,
            MemoryProfile::SsdBacked
        );
        let huge = machine(10_000, Some(24), Some(128));
        assert_eq!(
            resolve_profile(MemoryProfile::Fit, &huge).profile,
            MemoryProfile::Fit
        );
    }

    /// An explicit flag always beats the profile; only unset values are filled.
    #[test]
    fn explicit_settings_win_over_the_plan() {
        let plan = resolve_profile(MemoryProfile::SsdBacked, &machine(397, Some(16), Some(74)));
        let (placement, vram, ssd, engram) = apply_plan(
            &plan,
            None,
            None,
            Some(SsdExpertCache::Bytes(0)), // operator said: no slab
            None,
        );
        assert_eq!(placement, plan.placement); // filled
        assert_eq!(vram, plan.vram_expert_cache); // filled
        assert_eq!(ssd, SsdExpertCache::Bytes(0)); // operator wins
        assert_eq!(engram, plan.engram_tier); // filled
    }

    /// The load-log line names the profile and every resolved tier.
    #[test]
    fn describe_names_the_plan_and_the_tiers() {
        let fit = machine(397, Some(16), Some(74));
        let line = describe(&resolve_profile(MemoryProfile::SsdBacked, &fit), &fit);
        assert!(line.contains("profile `ssd-backed`"), "{line}");
        assert!(line.contains("ssd-expert-cache Auto"), "{line}");
        assert!(line.contains("397.0 GiB model"), "{line}");
        assert!(line.contains("74.0 GiB free RAM"), "{line}");
    }
}
