//! YaRN long-context RoPE scaling, shared by the DeepSeek loaders.
//!
//! Formulas follow the reference DeepSeek modeling code (and llama.cpp's
//! `ggml_rope_yarn_corr_dims`): frequencies inside the correction range are
//! blended between interpolated and extrapolated values by a linear ramp, and
//! attention logits are temperature-corrected by [`get_mscale`].

use crate::gguf_meta::Meta;

/// Parsed `rope.scaling.*` YaRN parameters.
#[derive(Debug, Clone)]
pub struct YarnConfig {
    pub factor: f32,
    pub orig_context_length: usize,
    /// `mscale_all_dim` recovered from `rope.scaling.yarn_log_multiplier`.
    pub mscale_all_dim: f32,
}

impl YarnConfig {
    /// Read the YaRN parameters when `rope.scaling.type == "yarn"`, else
    /// `None`.  `default_factor` / `default_orig_ctx` fill in keys a
    /// conversion omitted.
    pub fn from_meta(m: &Meta<'_>, default_factor: f32, default_orig_ctx: usize) -> Option<Self> {
        if m.string("rope.scaling.type").as_deref() != Some("yarn") {
            return None;
        }
        Some(Self {
            factor: m.f32_or("rope.scaling.factor", default_factor),
            orig_context_length: m.u32_or(
                "rope.scaling.original_context_length",
                default_orig_ctx as u32,
            ) as usize,
            // llama.cpp stores 0.1 * mscale_all_dim and divides it back out.
            mscale_all_dim: m.f32_or("rope.scaling.yarn_log_multiplier", 0.0) / 0.1,
        })
    }

    /// The attention / cos-sin magnitude scale for these parameters.
    pub fn mscale(&self) -> f32 {
        get_mscale(self.factor, self.mscale_all_dim)
    }
}

/// YaRN attention/temperature scale: `0.1 * mscale * ln(scale) + 1` for
/// `scale > 1`, else 1.
pub fn get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

fn find_correction_dim(num_rot: f32, dim: usize, base: f32, max_pos: usize) -> f32 {
    (dim as f32 * (max_pos as f32 / (num_rot * 2.0 * std::f32::consts::PI)).ln())
        / (2.0 * base.ln())
}

/// The `[low, high]` rotary-dimension range over which frequencies are
/// blended, clamped to `[0, dim - 1]`.
pub fn correction_range(
    low_rot: f32,
    high_rot: f32,
    dim: usize,
    base: f32,
    max_pos: usize,
) -> (f32, f32) {
    let low = find_correction_dim(low_rot, dim, base, max_pos).floor();
    let high = find_correction_dim(high_rot, dim, base, max_pos).ceil();
    (low.max(0.0), high.min(dim as f32 - 1.0))
}

/// `dim` values ramping linearly from 0 at `min` to 1 at `max`.
pub fn linear_ramp(min: f32, mut max: f32, dim: usize) -> Vec<f32> {
    if (min - max).abs() < f32::EPSILON {
        max += 0.001;
    }
    (0..dim)
        .map(|i| (((i as f32) - min) / (max - min)).clamp(0.0, 1.0))
        .collect()
}

/// YaRN inverse frequencies for a `dim`-wide rotary slice (`dim / 2`
/// entries), using DeepSeek's `beta_fast = 32`, `beta_slow = 1`.
pub fn inv_freq(dim: usize, theta: f32, y: &YarnConfig) -> Vec<f32> {
    let half = dim / 2;
    let (low, high) = correction_range(32.0, 1.0, dim, theta, y.orig_context_length);
    let ramp = linear_ramp(low, high, half);
    (0..half)
        .map(|i| {
            let extra = 1f32 / theta.powf((2 * i) as f32 / dim as f32);
            let inter = extra / y.factor;
            let mask = 1.0 - ramp[i];
            inter * (1.0 - mask) + extra * mask
        })
        .collect()
}
