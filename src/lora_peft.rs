//! Hugging Face PEFT LoRA adapters: a directory (or its
//! `adapter_model.safetensors`) with an `adapter_config.json`, read as
//! llama.cpp's `convert_lora_to_gguf.py` would convert it.
//!
//! Each `…layers.{i}.{self_attn|mlp|…}.{module}.lora_A.weight` / `lora_B`
//! pair is mapped to the GGUF projection it targets (`q_proj` → `attn_q`,
//! `gate_up_proj` → the fused `ffn_up`, …) and validated against the model's
//! [`SlotTable`] like a GGUF adapter's pair.  For a model whose GGUF stores
//! Q/K rows permuted for interleaved RoPE ([`SlotTable::rope_permute`]), the
//! `B` rows of `q_proj` / `k_proj` are permuted the same way.
//!
//! PEFT's scale, `lora_alpha / r` (`lora_alpha / √r` with `use_rslora`, and
//! per module under `rank_pattern` / `alpha_pattern`), is folded into each
//! `B`, and the adapter runs with alpha 0, so a request's scale multiplies
//! PEFT's own as it does in PEFT.  Adapters PEFT runs with more than the
//! low-rank pairs (DoRA, activated LoRA, trained biases, `modules_to_save`,
//! embedding or output-head pairs, `fan_in_fan_out` Conv1D weights) are
//! refused.  The weights are mapped, not read, and decoded a pair at a time.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::lora::{invalid, LoraAdapter, LoraLimits, LoraPair, SlotTable};
use crate::Result;

const CONFIG: &str = "adapter_config.json";
const WEIGHTS: &str = "adapter_model.safetensors";

/// A PEFT adapter's two files.
#[derive(Debug, Clone)]
pub struct PeftFiles {
    pub config: PathBuf,
    pub weights: PathBuf,
}

/// The PEFT adapter `path` names: a directory holding `adapter_config.json`
/// and `adapter_model.safetensors`, or a `.safetensors` file with an
/// `adapter_config.json` beside it.  `None` for anything else (a GGUF).
pub fn locate(path: &Path) -> Option<PeftFiles> {
    let (dir, weights) = if path.is_dir() {
        (path.to_path_buf(), path.join(WEIGHTS))
    } else if path.extension().is_some_and(|e| e == "safetensors") {
        (path.parent()?.to_path_buf(), path.to_path_buf())
    } else {
        return None;
    };
    let config = dir.join(CONFIG);
    (config.is_file() || path.is_dir()).then_some(PeftFiles { config, weights })
}

#[derive(Debug, Deserialize)]
struct Config {
    #[serde(default)]
    peft_type: Option<String>,
    r: usize,
    lora_alpha: f32,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    use_dora: bool,
    #[serde(default)]
    bias: Option<String>,
    #[serde(default)]
    fan_in_fan_out: bool,
    #[serde(default)]
    modules_to_save: Option<Vec<String>>,
    /// Per-module ranks and alphas, keyed by regular expressions PEFT
    /// matches against module names, in the config's order (the first
    /// match wins, as in PEFT).
    #[serde(default)]
    rank_pattern: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    alpha_pattern: serde_json::Map<String, serde_json::Value>,
    /// Activated LoRA: the adapter applies only after these tokens.
    #[serde(default)]
    alora_invocation_tokens: Option<Vec<u32>>,
}

/// A `rank_pattern` / `alpha_pattern` entry, compiled.
struct Pattern<T> {
    key: String,
    re: fancy_regex::Regex,
    value: T,
}

fn patterns<T>(
    field: &str,
    map: &serde_json::Map<String, serde_json::Value>,
    value: impl Fn(&serde_json::Value) -> Option<T>,
) -> Result<Vec<Pattern<T>>> {
    map.iter()
        .map(|(key, v)| {
            // PEFT: `re.match(rf"(.*\.)?({key})$", name)`, groups and
            // all, so a key's backreferences count as Python counts them.
            let re = fancy_regex::Regex::new(&format!(r"^(.*\.)?({key})$"))
                .map_err(|e| invalid(format!("{field} key '{key}' is not a valid pattern: {e}")))?;
            let value = value(v).ok_or_else(|| invalid(format!("{field}['{key}'] is not a number")))?;
            Ok(Pattern { key: key.clone(), re, value })
        })
        .collect()
}

/// The value of the first pattern a module matches (PEFT's
/// `get_pattern_key`: an exact key first, then the first matching one).
/// `names` are the module's names to try in turn: its name in the base
/// model, which PEFT matches, then its full key in the weights
/// (`base_model.model.…`), for a config written against that.
fn pick<T: Copy>(patterns: &[Pattern<T>], names: &[&str]) -> Result<Option<T>> {
    for name in names {
        if let Some(p) = patterns.iter().find(|p| p.key == *name) {
            return Ok(Some(p.value));
        }
        for p in patterns {
            let hit = p.re.is_match(name).map_err(|e| {
                invalid(format!("pattern '{}' on '{name}': {e}", p.key))
            })?;
            if hit {
                return Ok(Some(p.value));
            }
        }
    }
    Ok(None)
}

/// What [`Config::scale`] reads, compiled once.
struct Scales {
    r: usize,
    alpha: f32,
    rslora: bool,
    ranks: Vec<Pattern<usize>>,
    alphas: Vec<Pattern<f32>>,
}

impl Config {
    fn check(&self) -> Result<()> {
        if let Some(t) = &self.peft_type {
            if !t.eq_ignore_ascii_case("lora") {
                return Err(invalid(format!("PEFT type '{t}' is not LoRA")));
            }
        }
        if self.use_dora {
            return Err(invalid("DoRA adapters (use_dora) are not supported"));
        }
        if self.bias.as_deref().is_some_and(|b| b != "none") {
            return Err(invalid("adapters with trained biases (bias != \"none\") are not supported"));
        }
        if self.fan_in_fan_out {
            return Err(invalid("fan_in_fan_out (Conv1D) adapters are not supported"));
        }
        if self.modules_to_save.as_ref().is_some_and(|m| !m.is_empty()) {
            return Err(invalid(
                "adapters with modules_to_save carry full weights, which are not supported",
            ));
        }
        if self.alora_invocation_tokens.as_ref().is_some_and(|t| !t.is_empty()) {
            return Err(invalid(
                "activated LoRA adapters (alora_invocation_tokens) are not supported",
            ));
        }
        if !self.lora_alpha.is_finite() {
            return Err(invalid("lora_alpha is not finite"));
        }
        Ok(())
    }

    fn scales(&self) -> Result<Scales> {
        Ok(Scales {
            r: self.r,
            alpha: self.lora_alpha,
            rslora: self.use_rslora,
            ranks: patterns("rank_pattern", &self.rank_pattern, |v| v.as_u64().map(|r| r as usize))?,
            alphas: patterns("alpha_pattern", &self.alpha_pattern, |v| v.as_f64().map(|a| a as f32))?,
        })
    }
}

impl Scales {
    /// PEFT's scale for the module named `name` (its key in the weights,
    /// less `base_model.model.` and the `.lora_A.weight` suffix; `full`
    /// keeps the prefix), whose pair has rank `rank`.  The rank the config gives the module must be
    /// the pair's: one that differs means a pattern was not read as PEFT
    /// reads it, and the scale would be wrong.
    fn scale(&self, name: &str, full: &str, rank: usize) -> Result<f32> {
        let names = [name, full];
        let r = pick(&self.ranks, &names)?.unwrap_or(self.r);
        let alpha = pick(&self.alphas, &names)?.unwrap_or(self.alpha);
        if r != rank {
            return Err(invalid(format!(
                "module '{name}' has rank {rank}, but the config gives it rank {r}"
            )));
        }
        if r == 0 || !alpha.is_finite() {
            return Err(invalid(format!("module '{name}' has rank 0 or a non-finite alpha")));
        }
        Ok(if self.rslora {
            alpha / (r as f32).sqrt()
        } else {
            alpha / r as f32
        })
    }
}

/// The GGUF base tensor (`blk.{i}.{proj}.weight`) a PEFT module maps to.
fn gguf_base(module: &str) -> std::result::Result<String, String> {
    let rest = module
        .split_once(".layers.")
        .map(|(_, rest)| rest)
        .ok_or_else(|| match module.rsplit('.').next() {
            Some("embed_tokens") => "targets the token embedding, which takes no LoRA delta".to_string(),
            Some("lm_head") => "targets the output head, which takes no LoRA delta".to_string(),
            _ => "is not a decoder layer projection".to_string(),
        })?;
    let (layer, path) = rest
        .split_once('.')
        .ok_or_else(|| "is not a decoder layer projection".to_string())?;
    let layer: usize = layer
        .parse()
        .map_err(|_| "is not a decoder layer projection".to_string())?;
    let proj = match path {
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" | "self_attn.dense" => "attn_output",
        "self_attn.qkv_proj" => "attn_qkv",
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" | "mlp.gate_up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        "mlp.shared_expert.gate_proj" => "ffn_gate_shexp",
        "mlp.shared_expert.up_proj" => "ffn_up_shexp",
        "mlp.shared_expert.down_proj" => "ffn_down_shexp",
        p if p.starts_with("mlp.experts.") || p.starts_with("block_sparse_moe.experts.") => {
            return Err("targets routed experts, which take no LoRA delta".into())
        }
        p if p.ends_with(".gate") => return Err("targets an MoE router gate, which takes no LoRA delta".into()),
        _ => return Err(format!("targets '{path}', which this engine has no projection for")),
    };
    Ok(format!("blk.{layer}.{proj}.weight"))
}

/// `b`'s rows (`[n_out, r]`) reordered as `convert_hf_to_gguf.py`'s
/// `permute` reorders Q/K: within each of `heads` heads, the two halves of
/// the rotary pairs are interleaved.
fn permute_rows(b: &[f32], n_out: usize, r: usize, heads: usize) -> Result<Vec<f32>> {
    if heads == 0 || n_out % (2 * heads) != 0 {
        return Err(invalid(format!(
            "cannot permute {n_out} rows for {heads} interleaved-RoPE heads"
        )));
    }
    let half = n_out / heads / 2;
    let mut out = vec![0.0f32; b.len()];
    for h in 0..heads {
        for i in 0..2 {
            for j in 0..half {
                let src = h * 2 * half + i * half + j;
                let dst = h * 2 * half + j * 2 + i;
                out[dst * r..(dst + 1) * r].copy_from_slice(&b[src * r..(src + 1) * r]);
            }
        }
    }
    Ok(out)
}

fn to_f32(name: &str, t: &Tensor) -> Result<(Vec<usize>, Vec<f32>)> {
    let dims = t.dims().to_vec();
    let values: Vec<f32> = t
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1())
        .map_err(|e| invalid(format!("tensor '{name}': {e}")))?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid(format!("tensor '{name}' holds non-finite values")));
    }
    Ok((dims, values))
}

/// Read and validate the PEFT adapter in `files` against `table`.
pub fn load(files: &PeftFiles, table: &SlotTable, limits: &LoraLimits) -> Result<LoraAdapter> {
    let config_bytes = std::fs::read(&files.config)
        .map_err(|e| invalid(format!("{}: {e}", files.config.display())))?;
    let config: Config = serde_json::from_slice(&config_bytes)
        .map_err(|e| invalid(format!("{}: {e}", files.config.display())))?;
    config.check()?;
    let scales = config.scales()?;
    let file = std::fs::File::open(&files.weights)
        .map_err(|e| invalid(format!("{}: {e}", files.weights.display())))?;
    let len = file
        .metadata()
        .map_err(|e| invalid(format!("{}: {e}", files.weights.display())))?
        .len();
    if len > limits.max_bytes {
        return Err(invalid(format!(
            "{} is {len} bytes, over the {}-byte limit",
            files.weights.display(),
            limits.max_bytes
        )));
    }
    // SAFETY: the mapping is read-only and dropped before returning, as for
    // a GGUF adapter.  Tensors are decoded one pair at a time from it, so
    // the file is never held twice over.
    let weights = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| invalid(format!("{}: {e}", files.weights.display())))?;
    let mut hasher = Sha256::new();
    hasher.update(b"joshua/peft/v1");
    hasher.update((config_bytes.len() as u64).to_le_bytes());
    hasher.update(&config_bytes);
    hasher.update(&weights[..]);
    let digest: [u8; 32] = hasher.finalize().into();
    let tensors = candle_core::safetensors::SliceSafetensors::new(&weights)
        .map_err(|e| invalid(format!("{}: {e}", files.weights.display())))?;
    let load = |name: &str| {
        tensors
            .load(name, &Device::Cpu)
            .map_err(|e| invalid(format!("tensor '{name}': {e}")))
    };

    // Pair `<module>.lora_A.weight` with `<module>.lora_B.weight`, in name
    // order so the first error does not depend on the file's order.
    let mut halves: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();
    for (name, _) in tensors.tensors() {
        let entry = |m: &str| m.strip_prefix("base_model.model.").unwrap_or(m).to_string();
        if let Some(m) = name.strip_suffix(".lora_A.weight") {
            halves.entry(entry(m)).or_default().0 = Some(name.clone());
        } else if let Some(m) = name.strip_suffix(".lora_B.weight") {
            halves.entry(entry(m)).or_default().1 = Some(name.clone());
        } else if name.contains("lora_magnitude_vector") {
            return Err(invalid("DoRA adapters are not supported"));
        } else if name.contains("lora_embedding_") {
            return Err(invalid(format!("'{name}' targets the token embedding, which takes no LoRA delta")));
        } else {
            return Err(invalid(format!("tensor '{name}' is not a LoRA weight")));
        }
    }
    if halves.is_empty() {
        return Err(invalid("the adapter holds no LoRA weight pairs"));
    }

    let mut pairs = Vec::with_capacity(halves.len());
    let mut total = 0u64;
    let mut seen = std::collections::BTreeSet::new();
    for (module, ab) in &halves {
        let (Some(a), Some(b)) = ab else {
            return Err(invalid(format!("LoRA pair for '{module}' is missing one component")));
        };
        let base = gguf_base(module).map_err(|why| invalid(format!("'{module}' {why}")))?;
        let (key, n_out, n_in) = table
            .target(&base)
            .map_err(|why| invalid(format!("'{module}' ({base}) {why}")))?;
        if !seen.insert(key) {
            return Err(invalid(format!("'{module}' targets {base} a second time")));
        }
        let full = a.strip_suffix(".lora_A.weight").unwrap_or(a).to_string();
        let (a_dims, a) = to_f32(module, &load(a)?)?;
        let (b_dims, mut b) = to_f32(module, &load(b)?)?;
        let (&[rank, a_in], &[b_out, b_rank]) = (&a_dims[..], &b_dims[..]) else {
            return Err(invalid(format!("'{module}' has incorrect shape (hint: maybe wrong base model?)")));
        };
        if a_in != n_in || b_out != n_out || b_rank != rank {
            return Err(invalid(format!("'{module}' has incorrect shape (hint: maybe wrong base model?)")));
        }
        if rank == 0 || rank > limits.max_rank {
            return Err(invalid(format!(
                "'{module}' has rank {rank}, outside 1..={}",
                limits.max_rank
            )));
        }
        total += (4 * rank * (n_in + n_out)) as u64;
        if total > limits.max_bytes {
            return Err(invalid(format!(
                "the weights decode to more than the {}-byte limit",
                limits.max_bytes
            )));
        }
        if let Some((n_head, n_kv_head)) = table.rope_permute() {
            match key.proj {
                crate::lora::Proj::AttnQ => b = permute_rows(&b, n_out, rank, n_head)?,
                crate::lora::Proj::AttnK => b = permute_rows(&b, n_out, rank, n_kv_head)?,
                _ => {}
            }
        }
        let scale = scales.scale(module, &full, rank)?;
        b.iter_mut().for_each(|v| *v *= scale);
        let pair = LoraPair::new(a, b, rank, n_in, n_out, table.device()).map_err(invalid)?;
        pairs.push((key, Arc::new(pair)));
    }
    Ok(LoraAdapter::from_pairs(digest, table, 0.0, pairs, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_names_map_to_gguf_projections() {
        let cases = [
            ("model.layers.3.self_attn.q_proj", Ok("blk.3.attn_q.weight")),
            ("model.layers.0.mlp.gate_up_proj", Ok("blk.0.ffn_up.weight")),
            ("language_model.model.layers.1.mlp.down_proj", Ok("blk.1.ffn_down.weight")),
            ("model.layers.2.self_attn.qkv_proj", Ok("blk.2.attn_qkv.weight")),
            ("model.layers.0.mlp.shared_expert.up_proj", Ok("blk.0.ffn_up_shexp.weight")),
        ];
        for (module, want) in cases {
            assert_eq!(gguf_base(module).as_deref(), want.map(|s| s), "{module}");
        }
        assert!(gguf_base("lm_head").unwrap_err().contains("output head"));
        assert!(gguf_base("model.embed_tokens").unwrap_err().contains("embedding"));
        assert!(gguf_base("model.layers.0.mlp.experts.3.up_proj").unwrap_err().contains("experts"));
        assert!(gguf_base("model.layers.0.self_attn.rotary").is_err());
    }

    #[test]
    fn rows_permute_as_convert_hf_to_gguf_does() {
        // One head of 4 rows (pairs (0,2), (1,3)) at rank 1: [0, 1, 2, 3]
        // becomes [0, 2, 1, 3].
        assert_eq!(permute_rows(&[0., 1., 2., 3.], 4, 1, 1).unwrap(), [0., 2., 1., 3.]);
        // Two heads at rank 2: each head's rows interleave on their own.
        let b: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let p = permute_rows(&b, 8, 2, 2).unwrap();
        let rows: Vec<f32> = p.chunks(2).map(|r| r[0] / 2.0).collect();
        assert_eq!(rows, [0., 2., 1., 3., 4., 6., 5., 7.]);
        assert!(permute_rows(&b, 8, 2, 3).is_err());
    }

    #[test]
    fn scales_follow_peft() {
        let config = |extra: &str| -> Config {
            serde_json::from_str(&format!(r#"{{"r": 8, "lora_alpha": 16{extra}}}"#)).unwrap()
        };
        let scales = |extra: &str| config(extra).scales().unwrap();
        assert_eq!(scales("").scale("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj", 8).unwrap(), 2.0);
        let rs = scales(r#", "use_rslora": true"#);
        assert!((rs.scale("x", "x", 8).unwrap() - 16.0 / 8f32.sqrt()).abs() < 1e-6);
        let pat = scales(r#", "rank_pattern": {"q_proj": 4}, "alpha_pattern": {"layers.0.self_attn.q_proj": 2}"#);
        assert_eq!(pat.scale("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj", 4).unwrap(), 0.5);
        assert_eq!(pat.scale("model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.q_proj", 4).unwrap(), 4.0);
        assert_eq!(pat.scale("model.layers.1.self_attn.v_proj", "model.layers.1.self_attn.v_proj", 8).unwrap(), 2.0);
        // Keys are regular expressions, matched as PEFT matches them: the
        // whole name, or a dotted suffix of it; the first match wins.
        let re = scales(r#", "rank_pattern": {"q_proj$": 4, "layers\\.[01]\\..*v_proj": 2, "v_proj": 16}"#);
        assert_eq!(re.scale("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj", 4).unwrap(), 4.0);
        assert_eq!(re.scale("model.layers.1.self_attn.v_proj", "model.layers.1.self_attn.v_proj", 2).unwrap(), 8.0);
        assert_eq!(re.scale("model.layers.2.self_attn.v_proj", "model.layers.2.self_attn.v_proj", 16).unwrap(), 1.0);
        // `proj` alone matches no dotted suffix.
        assert_eq!(scales(r#", "rank_pattern": {"proj": 4}"#).scale("model.layers.0.mlp.up_proj", "model.layers.0.mlp.up_proj", 8).unwrap(), 2.0);
        // A rank the config does not give the pair is refused, not mis-scaled.
        assert!(pat.scale("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj", 8).is_err());
        assert!(config(r#", "rank_pattern": {"(": 4}"#).scales().is_err());
        // Python's lookaround, and a key written against the full weight key.
        let look = scales(r#", "rank_pattern": {"q_proj(?<!layers\\.1\\.self_attn\\.q_proj)": 4}"#);
        assert_eq!(look.scale("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj", 4).unwrap(), 4.0);
        assert_eq!(look.scale("model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.q_proj", 8).unwrap(), 2.0);
        let full = scales(r#", "alpha_pattern": {"base_model.model.model.layers.0.self_attn.q_proj": 4}"#);
        let (name, key) = ("model.layers.0.self_attn.q_proj", "base_model.model.model.layers.0.self_attn.q_proj");
        assert_eq!(full.scale(name, key, 8).unwrap(), 0.5);
        assert!(config(r#", "alora_invocation_tokens": [101, 102]"#).check().is_err());
        assert!(config(r#", "alora_invocation_tokens": null"#).check().is_ok());
        assert!(config(r#", "use_dora": true"#).check().is_err());
        assert!(config(r#", "bias": "all""#).check().is_err());
        assert!(config(r#", "modules_to_save": ["lm_head"]"#).check().is_err());
        assert!(config(r#", "modules_to_save": null"#).check().is_ok());
    }
}
