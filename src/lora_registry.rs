//! The LoRA adapters an engine has loaded, and the scales requests run them
//! at.
//!
//! An adapter is loaded once (at startup with `--lora`/`--lora-scaled`, or
//! later through the API), validated against the base model's projections,
//! and given an id that never changes or gets reused, as llama-server
//! numbers them.  Each adapter carries a server-wide scale, which
//! `POST /lora-adapters` sets and a request without its own `lora` list runs
//! with.  A request's `lora` list (`[{"id": 0, "scale": 0.5}]`) replaces the
//! server-wide scales for that request alone; an adapter it leaves out runs
//! at scale 0, as in llama-server.
//!
//! The adapters and scales a request runs with become one
//! [`ActiveSet`], named by its [`AdapterKey`].  Sets are kept and shared by
//! key, so concurrent requests with the same selection share one set, and
//! the key is folded into the request's cache namespace (see
//! [`crate::types::GenerationOptions::cache_namespace`]): pooled sessions,
//! prefix-cache entries and snapshots computed under one selection are only
//! ever continued under the same one.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::Serialize;

use crate::error::{JoshuaError, Result};
use crate::lora::{ActiveSet, AdapterKey, LoraAdapter, LoraLimits, SlotTable};
use crate::types::LoraScale;

/// How many recently used [`ActiveSet`]s are kept for reuse.
const KEPT_SETS: usize = 8;

/// A loaded adapter, as `GET /lora-adapters` lists it.
#[derive(Debug, Clone, Serialize)]
pub struct LoraInfo {
    /// The adapter's id: what a request's `lora` list names it by.
    pub id: usize,
    /// The name it was loaded under (its file stem unless named).
    pub name: String,
    /// The file it was read from.
    pub path: PathBuf,
    /// The server-wide scale.
    pub scale: f32,
    /// `adapter.lora.task_name`, when the file sets it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_name: Option<String>,
    /// `adapter.lora.prompt_prefix`, when the file sets it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_prefix: Option<String>,
}

struct Entry {
    info: LoraInfo,
    adapter: Arc<LoraAdapter>,
}

#[derive(Default)]
struct Inner {
    entries: Vec<Entry>,
    next_id: usize,
    /// Recently used sets, most recent last.
    sets: Vec<Arc<ActiveSet>>,
}

/// An engine's loaded adapters (see the module docs).
#[derive(Default)]
pub struct Registry {
    inner: Mutex<Inner>,
    limits: LoraLimits,
}

impl Registry {
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Load the adapter at `path`, validated against `table`, under `name`
    /// (its file stem by default) at server-wide scale `scale`.
    pub fn load(
        &self,
        path: &Path,
        name: Option<&str>,
        scale: f32,
        table: &SlotTable,
    ) -> Result<LoraInfo> {
        if !scale.is_finite() {
            return Err(JoshuaError::InvalidRequest(format!(
                "LoRA scale {scale} is not finite"
            )));
        }
        let name = match name {
            Some(name) => name.trim().to_string(),
            None => path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string(),
        };
        if name.is_empty() {
            return Err(JoshuaError::InvalidRequest(
                "a LoRA adapter needs a name".into(),
            ));
        }
        if self.lock().entries.iter().any(|e| e.info.name == name) {
            return Err(JoshuaError::InvalidRequest(format!(
                "a LoRA adapter named `{name}` is already loaded"
            )));
        }
        // Read outside the lock: a large adapter takes a while to decode.
        let adapter = Arc::new(LoraAdapter::load(path, table, &self.limits)?);
        let mut inner = self.lock();
        if inner.entries.iter().any(|e| e.info.name == name) {
            return Err(JoshuaError::InvalidRequest(format!(
                "a LoRA adapter named `{name}` is already loaded"
            )));
        }
        if scale != 0.0 {
            let active = inner.entries.iter().filter(|e| e.info.scale != 0.0).count();
            self.check_active(active + 1)?;
        }
        let info = LoraInfo {
            id: inner.next_id,
            name,
            path: path.to_path_buf(),
            scale,
            task_name: adapter.task_name().map(str::to_string),
            prompt_prefix: adapter.prompt_prefix().map(str::to_string),
        };
        inner.next_id += 1;
        inner.entries.push(Entry {
            info: info.clone(),
            adapter,
        });
        Ok(info)
    }

    /// Unload the adapter named `name`.  Sets already built with it stay
    /// valid for the requests running them; no new request selects it.
    pub fn unload(&self, name: &str) -> Result<LoraInfo> {
        let mut inner = self.lock();
        let Some(i) = inner.entries.iter().position(|e| e.info.name == name) else {
            return Err(JoshuaError::InvalidRequest(format!(
                "no LoRA adapter named `{name}` is loaded"
            )));
        };
        let entry = inner.entries.remove(i);
        let digest = *entry.adapter.digest();
        inner
            .sets
            .retain(|set| set.adapters().iter().all(|(a, _)| *a.digest() != digest));
        Ok(entry.info)
    }

    /// Whether every adapter in `set` is still loaded (an unloaded one's
    /// digest is no longer among the entries).
    pub fn holds(&self, set: &ActiveSet) -> bool {
        let inner = self.lock();
        set.adapters()
            .iter()
            .all(|(a, _)| inner.entries.iter().any(|e| e.adapter.digest() == a.digest()))
    }

    /// The loaded adapters, by id.
    pub fn list(&self) -> Vec<LoraInfo> {
        self.lock().entries.iter().map(|e| e.info.clone()).collect()
    }

    /// The adapter named `name`'s id.
    pub fn id_of(&self, name: &str) -> Option<usize> {
        self.lock()
            .entries
            .iter()
            .find(|e| e.info.name == name)
            .map(|e| e.info.id)
    }

    /// Refuse a server-wide selection of `active` nonzero adapters when a
    /// set of that many could not be built, which would fail every request
    /// that runs the server-wide scales.
    fn check_active(&self, active: usize) -> Result<()> {
        if active > self.limits.max_adapters {
            return Err(JoshuaError::InvalidRequest(format!(
                "{active} LoRA adapters would be active at once; at most {} can be",
                self.limits.max_adapters
            )));
        }
        Ok(())
    }

    /// Set the server-wide scales: each listed adapter's to its scale and
    /// every other one's to 0 (llama-server's `POST /lora-adapters`).
    pub fn set_scales(&self, scales: &[LoraScale]) -> Result<()> {
        let mut inner = self.lock();
        let resolved = resolve(&inner.entries, scales)?;
        self.check_active(resolved.iter().filter(|s| **s != 0.0).count())?;
        for (entry, scale) in inner.entries.iter_mut().zip(resolved) {
            entry.info.scale = scale;
        }
        Ok(())
    }

    /// The set a request runs: `scales` when it sends its own (unlisted
    /// adapters at 0), else the server-wide scales.  `None` when every
    /// scale is 0, which runs the base model.
    pub fn select(&self, scales: Option<&[LoraScale]>) -> Result<Option<Arc<ActiveSet>>> {
        let mut inner = self.lock();
        let resolved = match scales {
            Some(scales) => resolve(&inner.entries, scales)?,
            None => inner.entries.iter().map(|e| e.info.scale).collect(),
        };
        let selection: Vec<(Arc<LoraAdapter>, f32)> = inner
            .entries
            .iter()
            .zip(resolved)
            .filter(|(_, scale)| *scale != 0.0)
            .map(|(e, scale)| (Arc::clone(&e.adapter), scale))
            .collect();
        let digests: Vec<([u8; 32], f32)> =
            selection.iter().map(|(a, s)| (*a.digest(), *s)).collect();
        let Some(key) = AdapterKey::of(&digests) else {
            return Ok(None);
        };
        if let Some(i) = inner.sets.iter().position(|set| set.key() == key) {
            let set = inner.sets.remove(i);
            inner.sets.push(Arc::clone(&set));
            return Ok(Some(set));
        }
        let Some(set) = ActiveSet::new(&selection, &self.limits)? else {
            return Ok(None);
        };
        let set = Arc::new(set);
        inner.sets.push(Arc::clone(&set));
        if inner.sets.len() > KEPT_SETS {
            inner.sets.remove(0);
        }
        Ok(Some(set))
    }
}

/// Each entry's scale under `scales`: its listed scale, or 0.  An id no
/// entry has, a repeated id or a scale that is not finite is refused.
fn resolve(entries: &[Entry], scales: &[LoraScale]) -> Result<Vec<f32>> {
    let mut resolved = vec![0.0f32; entries.len()];
    let mut seen = vec![false; entries.len()];
    for LoraScale { id, scale } in scales {
        let Some(i) = entries.iter().position(|e| e.info.id == *id) else {
            return Err(JoshuaError::InvalidRequest(format!(
                "no LoRA adapter has id {id}"
            )));
        };
        if !scale.is_finite() {
            return Err(JoshuaError::InvalidRequest(format!(
                "LoRA scale {scale} is not finite"
            )));
        }
        if std::mem::replace(&mut seen[i], true) {
            return Err(JoshuaError::InvalidRequest(format!(
                "LoRA adapter {id} is listed twice"
            )));
        }
        resolved[i] = *scale;
    }
    Ok(resolved)
}
