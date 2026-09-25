//! Typed access to a GGUF file's architecture-scoped metadata.
//!
//! llama.cpp namespaces every hyper-parameter under the architecture name
//! (`deepseek2.block_count`, `qwen3moe.expert_count`, …).  Joshua's native
//! loaders each used to carry a private copy of the same small reader and
//! spell out `format!("{arch}.…")` on every key; [`Meta`] is the one copy,
//! taking keys *relative* to the architecture.

use std::collections::HashMap;

use candle_core::quantized::gguf_file;
use candle_core::Result;

/// Metadata reader scoped to one architecture: `m.u32("block_count")` reads
/// `"{arch}.block_count"`.  Missing required keys fail with an error naming
/// the architecture and the full key.
pub struct Meta<'a> {
    md: &'a HashMap<String, gguf_file::Value>,
    arch: &'a str,
}

impl<'a> Meta<'a> {
    pub fn new(md: &'a HashMap<String, gguf_file::Value>, arch: &'a str) -> Self {
        Self { md, arch }
    }

    /// The architecture prefix these keys are read under.
    pub fn arch(&self) -> &'a str {
        self.arch
    }

    /// The full GGUF key for `suffix`.
    pub fn key(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.arch)
    }

    fn get(&self, suffix: &str) -> Option<&'a gguf_file::Value> {
        self.md.get(&self.key(suffix))
    }

    fn required(&self, suffix: &str) -> Result<&'a gguf_file::Value> {
        match self.get(suffix) {
            Some(v) => Ok(v),
            None => candle_core::bail!(
                "{}: missing GGUF metadata key `{}`",
                self.arch,
                self.key(suffix)
            ),
        }
    }

    /// Whether `suffix` is present at all.
    pub fn contains(&self, suffix: &str) -> bool {
        self.get(suffix).is_some()
    }

    pub fn u32(&self, suffix: &str) -> Result<u32> {
        self.required(suffix)?.to_u32()
    }

    /// `suffix` as a `u32`, or `None` when absent or not an integer.
    pub fn u32_opt(&self, suffix: &str) -> Option<u32> {
        self.get(suffix).and_then(|v| v.to_u32().ok())
    }

    pub fn u32_or(&self, suffix: &str, default: u32) -> u32 {
        self.u32_opt(suffix).unwrap_or(default)
    }

    pub fn f32(&self, suffix: &str) -> Result<f32> {
        self.required(suffix)?.to_f32()
    }

    pub fn f32_or(&self, suffix: &str, default: f32) -> f32 {
        self.get(suffix)
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(default)
    }

    pub fn bool_or(&self, suffix: &str, default: bool) -> bool {
        self.get(suffix)
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(default)
    }

    pub fn string(&self, suffix: &str) -> Option<String> {
        self.get(suffix).and_then(|v| v.to_string().ok().cloned())
    }

    /// A per-layer float array, truncated or zero-padded to `n` entries
    /// (all zeros when absent).
    pub fn array_f64(&self, suffix: &str, n: usize) -> Vec<f64> {
        self.array(suffix, n, |v| v.to_f32().map(|x| x as f64).unwrap_or(0.0))
    }

    /// A per-layer integer array, truncated or zero-padded to `n` entries
    /// (all zeros when absent).
    pub fn array_u32(&self, suffix: &str, n: usize) -> Vec<usize> {
        self.array(suffix, n, |v| v.to_u32().unwrap_or(0) as usize)
    }

    /// A per-layer boolean array, truncated or `false`-padded to `n`
    /// entries; `None` when the key is absent.
    pub fn array_bool(&self, suffix: &str, n: usize) -> Option<Vec<bool>> {
        self.contains(suffix)
            .then(|| self.array(suffix, n, |v| v.to_bool().unwrap_or(false)))
    }

    fn array<T: Default + Clone>(
        &self,
        suffix: &str,
        n: usize,
        f: impl Fn(&gguf_file::Value) -> T,
    ) -> Vec<T> {
        let mut out: Vec<T> = match self.get(suffix) {
            Some(gguf_file::Value::Array(arr)) => arr.iter().take(n).map(f).collect(),
            _ => Vec::new(),
        };
        out.resize(n, T::default());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_scoped_to_the_architecture() {
        let md: HashMap<String, gguf_file::Value> = [
            (
                "deepseek2.block_count".to_string(),
                gguf_file::Value::U32(3),
            ),
            (
                "deepseek2.swiglu".to_string(),
                gguf_file::Value::Array(vec![gguf_file::Value::F32(1.5)]),
            ),
        ]
        .into_iter()
        .collect();
        let m = Meta::new(&md, "deepseek2");
        assert_eq!(m.u32("block_count").unwrap(), 3);
        assert_eq!(m.u32_or("expert_count", 7), 7);
        assert!(m.contains("block_count"));
        assert!(!m.contains("expert_count"));
        assert_eq!(m.array_f64("swiglu", 3), vec![1.5, 0.0, 0.0]);
        assert_eq!(m.array_u32("absent", 2), vec![0, 0]);
        let err = m.u32("expert_count").unwrap_err().to_string();
        assert!(err.contains("deepseek2.expert_count"), "{err}");
    }
}
