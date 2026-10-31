//! Bounded routing hints and request cache namespaces; neither authorizes token reuse.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use axum::http::{header, HeaderMap};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub const AFFINITY_HEADER: &str = "x-joshua-affinity";
pub(crate) const CACHE_SCOPE_HEADER: &str = "x-joshua-cache-scope";
pub(crate) type Scope = [u8; 32];
const CAPACITY: usize = 4096;
pub(crate) const TTL: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidentScope {
    pub scope: String,
    pub tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheResidency {
    pub entries: Vec<ResidentScope>,
    pub complete: bool,
}

pub(crate) fn namespace(scope: Option<Scope>, context: Option<u32>) -> Option<Scope> {
    context.map_or(scope, |context| {
        Some(digest(&[
            scope.as_ref().map_or(&[], |scope| scope.as_slice()),
            &context.to_le_bytes(),
        ]))
    })
}

pub(crate) fn worker_namespace(scope: Scope, context: Option<u32>, key: Option<&str>) -> Scope {
    let authorization = key.unwrap_or_default();
    namespace(
        Some(digest(&[
            authorization.as_bytes(),
            encode(scope).as_bytes(),
            &context.unwrap_or(0).to_le_bytes(),
        ])),
        context,
    )
    .unwrap()
}

pub(crate) fn digest(parts: &[&[u8]]) -> Scope {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    hash.finalize().into()
}

pub(crate) fn owner(headers: &HeaderMap) -> &[u8] {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
        })
        .map_or(&[], str::as_bytes)
}

pub(crate) fn handle(headers: &HeaderMap) -> Result<Option<String>, &'static str> {
    let Some(value) = headers.get(AFFINITY_HEADER) else {
        return Ok(None);
    };
    let value = value.to_str().map_err(|_| "invalid affinity handle")?;
    if value == "new" {
        return Ok(Some(Uuid::new_v4().to_string()));
    }
    if value.len() != 36 {
        return Err("affinity handle must be 'new' or a UUID");
    }
    Uuid::parse_str(value)
        .map(|id| Some(id.to_string()))
        .map_err(|_| "invalid affinity handle")
}

pub(crate) fn encode(scope: Scope) -> String {
    scope.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) fn cache_scope(headers: &HeaderMap, context: Option<u32>) -> Scope {
    let hint = headers
        .get(CACHE_SCOPE_HEADER)
        .or_else(|| headers.get(AFFINITY_HEADER));
    digest(&[
        owner(headers),
        hint.map_or(&[], |value| value.as_bytes()),
        &context.unwrap_or(0).to_le_bytes(),
    ])
}

struct Entry {
    worker: usize,
    epoch: String,
    expires: Instant,
    touched: u64,
    lease: u64,
    cached_tokens: usize,
}

#[derive(Default, Clone, Copy, Serialize)]
pub struct AffinityStats {
    pub entries: usize,
    pub preferred: u64,
    pub misses: u64,
    pub spillovers: u64,
    pub evicted: u64,
    pub expired: u64,
    pub stale: u64,
    pub residency_misses: u64,
    pub cached_tokens: usize,
}

#[derive(Default)]
pub(crate) struct Directory {
    entries: HashMap<Scope, Entry>,
    tick: u64,
    stats: AffinityStats,
}

impl Directory {
    pub(crate) fn preferred(
        &mut self,
        scope: &Scope,
        now: Instant,
    ) -> Option<(usize, String, u64)> {
        self.expire(now);
        self.entries
            .get(scope)
            .map(|entry| (entry.worker, entry.epoch.clone(), entry.lease))
    }

    pub(crate) fn record(
        &mut self,
        scope: Scope,
        worker: usize,
        epoch: &str,
        preferred: Option<usize>,
        now: Instant,
        lease: u64,
        cached_tokens: usize,
    ) {
        self.expire(now);
        match preferred {
            Some(previous) if previous == worker => self.stats.preferred += 1,
            Some(_) => self.stats.spillovers += 1,
            None => self.stats.misses += 1,
        }
        if !self.entries.contains_key(&scope) && self.entries.len() >= CAPACITY {
            if let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.touched)
                .map(|(key, _)| *key)
            {
                self.entries.remove(&oldest);
                self.stats.evicted += 1;
            }
        }
        self.tick = self.tick.saturating_add(1);
        self.entries.insert(
            scope,
            Entry {
                worker,
                epoch: epoch.to_owned(),
                expires: now + TTL,
                touched: self.tick,
                lease,
                cached_tokens,
            },
        );
    }

    pub(crate) fn invalidate(&mut self, scope: &Scope) {
        if self.entries.remove(scope).is_some() {
            self.stats.stale += 1;
        }
    }

    pub(crate) fn saved_work(&self, scope: &Scope, request_units: u64) -> u64 {
        self.entries.get(scope).map_or(0, |entry| {
            (entry.cached_tokens as u64)
                .saturating_mul(4)
                .min(request_units / 5)
        })
    }

    pub(crate) fn residency(
        &mut self,
        scope: &Scope,
        tokens: Option<usize>,
        complete: bool,
    ) -> bool {
        if let Some(tokens) = tokens {
            if let Some(entry) = self.entries.get_mut(scope) {
                entry.cached_tokens = tokens;
            }
            true
        } else if complete {
            if self.entries.remove(scope).is_some() {
                self.stats.residency_misses += 1;
            }
            false
        } else {
            true
        }
    }

    fn expire(&mut self, now: Instant) {
        let before = self.entries.len();
        self.entries.retain(|_, entry| entry.expires > now);
        self.stats.expired += (before - self.entries.len()) as u64;
    }

    pub(crate) fn stats(&mut self) -> AffinityStats {
        self.expire(Instant::now());
        AffinityStats {
            entries: self.entries.len(),
            cached_tokens: self.entries.values().fold(0usize, |total, entry| {
                total.saturating_add(entry.cached_tokens)
            }),
            ..self.stats
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directory_is_bounded_expires_and_invalidates_incarnations() {
        let mut directory = Directory::default();
        let now = Instant::now();
        for index in 0..CAPACITY + 10 {
            directory.record(digest(&[&index.to_le_bytes()]), 0, "epoch", None, now, 1, 8);
        }
        assert_eq!(directory.entries.len(), CAPACITY);
        assert_eq!(directory.stats.evicted, 10);
        let scope = digest(&[&CAPACITY.to_le_bytes()]);
        assert_eq!(
            directory.preferred(&scope, now),
            Some((0, "epoch".into(), 1))
        );
        directory.invalidate(&scope);
        assert_eq!(directory.preferred(&scope, now), None);
        directory.expire(now + TTL);
        assert!(directory.entries.is_empty());
    }

    #[test]
    fn ownership_and_configuration_are_separate_namespaces() {
        assert_ne!(digest(&[b"ab", b"c"]), digest(&[b"a", b"bc"]));
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Bearer alice".parse().unwrap());
        let alice = cache_scope(&headers, Some(64));
        assert_ne!(alice, cache_scope(&headers, Some(128)));
        headers.insert(header::AUTHORIZATION, "Bearer bob".parse().unwrap());
        assert_ne!(alice, cache_scope(&headers, Some(64)));
        headers.remove(header::AUTHORIZATION);
        headers.insert("x-api-key", "alice".parse().unwrap());
        assert_eq!(alice, cache_scope(&headers, Some(64)));
        headers.insert("x-api-key", "bob".parse().unwrap());
        assert_ne!(alice, cache_scope(&headers, Some(64)));
        headers.insert(AFFINITY_HEADER, "new".parse().unwrap());
        assert_ne!(handle(&headers).unwrap(), handle(&headers).unwrap());
        headers.insert(AFFINITY_HEADER, "arbitrary".parse().unwrap());
        assert!(handle(&headers).is_err());
    }

    #[test]
    fn incomplete_residency_is_unknown_and_complete_absence_invalidates() {
        let mut directory = Directory::default();
        let scope = [1; 32];
        directory.record(scope, 0, "epoch", None, Instant::now(), 7, 10);
        assert!(directory.residency(&scope, None, false));
        assert_eq!(directory.stats().entries, 1);
        assert!(directory.residency(&scope, Some(42), true));
        assert_eq!(directory.stats().cached_tokens, 42);
        assert!(!directory.residency(&scope, None, true));
        assert_eq!(directory.stats().residency_misses, 1);
        assert_eq!(directory.stats().entries, 0);
    }

    #[test]
    fn locality_discount_is_limited_by_residency_and_incoming_work() {
        let mut directory = Directory::default();
        let scope = [1; 32];
        assert_eq!(directory.saved_work(&scope, 100), 0);
        directory.record(scope, 0, "epoch", None, Instant::now(), 1, 2);
        assert_eq!(directory.saved_work(&scope, 100), 8);
        assert_eq!(directory.saved_work(&scope, 20), 4);
        directory.residency(&scope, Some(0), true);
        assert_eq!(directory.saved_work(&scope, 100), 0);
    }

    #[test]
    fn coordinator_and_worker_derive_the_same_namespace() {
        for context in [None, Some(64)] {
            let scope = [3; 32];
            let mut headers = HeaderMap::new();
            headers.insert(header::AUTHORIZATION, "Bearer worker-key".parse().unwrap());
            headers.insert(CACHE_SCOPE_HEADER, encode(scope).parse().unwrap());
            assert_eq!(
                worker_namespace(scope, context, Some("worker-key")),
                namespace(Some(cache_scope(&headers, context)), context).unwrap()
            );
        }
    }
}
