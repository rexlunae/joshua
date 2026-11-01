//! The engine's prefix cache: conversation state parked between requests.
//!
//! The warm session pool keeps a few live sessions, and a request continues
//! (or rewinds) one of them.  That pool is small (0–4 sessions, by free RAM),
//! and a session taken from it is gone from it, so two alternating
//! conversations evict each other and a new chat that shares only its system
//! prompt with an old one either destroys that conversation (rewinding it) or,
//! on a model with recurrent layers, re-reads the whole prompt.  Strata's
//! conversation cache answers both; this is the same idea for Joshua:
//!
//! * **Root entries.**  The state at the end of a prompt's shared head — the
//!   system prompt and tool list every new chat of one client repeats — is
//!   kept for good (evicted only after every other entry), and a new chat
//!   starts from a copy of it.
//! * **Parked conversations.**  A session evicted from the warm pool leaves its
//!   state here instead of being dropped, so the conversation resumes later
//!   without re-reading its history.
//!
//! Entries are detached snapshots (see
//! [`crate::native_session::SessionSnapshot`]), held in host RAM under a byte
//! budget, least recently used first out.  A lookup only matches an entry
//! whose tokens are a strict prefix of the prompt, so a reused state was
//! produced by exactly the tokens it is reused for.

/// What the cache needs to know about a snapshot.
pub trait Snapshot {
    /// Bytes the snapshot holds (its share of the budget).
    fn bytes(&self) -> u64;
}

impl Snapshot for crate::model::ModelSnapshot {
    fn bytes(&self) -> u64 {
        crate::model::ModelSnapshot::bytes(self)
    }
}

/// Root entries kept at most: one per distinct system prompt in use, so a
/// handful of clients (an editor agent, a chat UI, a script) each keep theirs.
pub const MAX_ROOTS: usize = 4;

struct Entry<S> {
    scope: Option<crate::affinity::Scope>,
    expires: Option<std::time::Instant>,
    tokens: Vec<u32>,
    snap: S,
    bytes: u64,
    last_used: u64,
    root: bool,
}

/// What a lookup found.
pub struct Hit<S> {
    /// The state; for a root entry, the caller's own copy.
    pub snap: S,
    /// Tokens it covers.
    pub len: usize,
    /// Whether it came from a root entry (which stays in the cache).
    pub root: bool,
}

/// Counters for the operator (see [`PrefixCache::stats`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    /// Scoped snapshots removed after their idle lease expired.
    pub expired: u64,
    /// Requests that started from a cached entry.
    pub hits: u64,
    /// Prompt tokens those requests did not have to read.
    pub tokens_reused: u64,
    /// Entries stored (roots and parked conversations).
    pub stored: u64,
    /// Entries evicted to make room.
    pub evicted: u64,
    /// Entries currently held.
    pub entries: usize,
    /// Bytes currently held.
    pub bytes: u64,
    /// The byte budget.
    pub budget: u64,
}

/// A byte-budgeted set of parked states, keyed by the tokens they cover.
pub struct PrefixCache<S> {
    entries: Vec<Entry<S>>,
    budget: u64,
    used: u64,
    tick: u64,
    stats: PrefixCacheStats,
}

impl<S: Snapshot> PrefixCache<S> {
    /// An empty cache holding at most `budget` bytes (`0` disables it).
    pub fn new(budget: u64) -> Self {
        Self {
            entries: Vec::new(),
            budget,
            used: 0,
            tick: 0,
            stats: PrefixCacheStats::default(),
        }
    }

    /// Whether the cache stores anything at all.
    pub fn enabled(&self) -> bool {
        self.budget > 0
    }

    fn tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Length of the longest entry that is a strict prefix of `prompt` (a
    /// strict prefix, so at least one prompt token is left to produce the
    /// next-token logits).  `0` when none matches.
    pub fn best_len(&self, prompt: &[u32]) -> usize {
        self.best_len_scoped(prompt, None)
    }

    pub(crate) fn best_len_scoped(
        &self,
        prompt: &[u32],
        scope: Option<crate::affinity::Scope>,
    ) -> usize {
        self.best(prompt, scope)
            .map_or(0, |i| self.entries[i].tokens.len())
    }

    fn best(&self, prompt: &[u32], scope: Option<crate::affinity::Scope>) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.scope == scope
                    && e.expires
                        .is_none_or(|expires| expires > std::time::Instant::now())
                    && e.tokens.len() < prompt.len()
                    && prompt.starts_with(&e.tokens)
            })
            .max_by_key(|(_, e)| (e.tokens.len(), e.last_used))
            .map(|(i, _)| i)
    }

    /// Take the longest entry that is a strict prefix of `prompt`.  A root
    /// entry stays (the caller gets a copy from `duplicate`); a parked
    /// conversation leaves the cache, since the session continuing it will
    /// hold (and later park) a longer state anyway.
    pub fn take<E>(
        &mut self,
        prompt: &[u32],
        duplicate: impl FnOnce(&S) -> Result<S, E>,
    ) -> Result<Option<Hit<S>>, E> {
        self.take_scoped(prompt, None, duplicate)
    }

    pub(crate) fn take_scoped<E>(
        &mut self,
        prompt: &[u32],
        scope: Option<crate::affinity::Scope>,
        duplicate: impl FnOnce(&S) -> Result<S, E>,
    ) -> Result<Option<Hit<S>>, E> {
        let Some(i) = self.best(prompt, scope) else {
            return Ok(None);
        };
        let now = self.tick();
        let hit = if self.entries[i].root {
            let e = &mut self.entries[i];
            e.last_used = now;
            e.expires = scope.map(|_| std::time::Instant::now() + crate::affinity::TTL);
            Hit {
                snap: duplicate(&e.snap)?,
                len: e.tokens.len(),
                root: true,
            }
        } else {
            let e = self.entries.swap_remove(i);
            self.used -= e.bytes;
            Hit {
                len: e.tokens.len(),
                snap: e.snap,
                root: false,
            }
        };
        self.stats.hits += 1;
        self.stats.tokens_reused += hit.len as u64;
        Ok(Some(hit))
    }

    /// Whether a root entry covers exactly `tokens`.
    pub fn has_root(&self, tokens: &[u32]) -> bool {
        self.has_root_scoped(tokens, None)
    }

    pub(crate) fn has_root_scoped(
        &self,
        tokens: &[u32],
        scope: Option<crate::affinity::Scope>,
    ) -> bool {
        self.entries.iter().any(|e| {
            e.scope == scope
                && e.root
                && e.tokens == tokens
                && e.expires
                    .is_none_or(|expires| expires > std::time::Instant::now())
        })
    }

    /// Whether an entry of `bytes` could be stored at all.
    pub fn fits(&self, bytes: u64) -> bool {
        bytes > 0 && bytes <= self.budget
    }

    /// Store `snap` as the state after `tokens`.  An entry for the same tokens
    /// is replaced; then least recently used entries go until it fits —
    /// parked conversations first, roots only when nothing else is left.  A
    /// root also pushes out the oldest root beyond [`MAX_ROOTS`].  Returns
    /// whether it was stored.
    pub fn insert(&mut self, tokens: Vec<u32>, snap: S, root: bool) -> bool {
        self.insert_scoped(tokens, snap, root, None)
    }

    pub(crate) fn insert_scoped(
        &mut self,
        tokens: Vec<u32>,
        snap: S,
        root: bool,
        scope: Option<crate::affinity::Scope>,
    ) -> bool {
        self.expire(std::time::Instant::now());
        let bytes = snap.bytes();
        if tokens.is_empty() || !self.fits(bytes) {
            return false;
        }
        if let Some(i) = self
            .entries
            .iter()
            .position(|e| e.scope == scope && e.tokens == tokens)
        {
            let old = self.entries.swap_remove(i);
            self.used -= old.bytes;
            // A conversation reaching a root's exact tokens keeps it a root.
            return self.insert_scoped(tokens, snap, root || old.root, scope);
        }
        if root {
            while self.entries.iter().filter(|e| e.root).count() >= MAX_ROOTS {
                self.evict(true);
            }
        }
        while self.used + bytes > self.budget {
            if !self.evict(false) && !self.evict(true) {
                return false;
            }
        }
        let now = self.tick();
        self.used += bytes;
        self.entries.push(Entry {
            scope,
            expires: scope.map(|_| std::time::Instant::now() + crate::affinity::TTL),
            tokens,
            snap,
            bytes,
            last_used: now,
            root,
        });
        self.stats.stored += 1;
        true
    }

    /// Drop the least recently used entry of the given kind.
    fn evict(&mut self, root: bool) -> bool {
        let Some(i) = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.root == root)
            .min_by_key(|(_, e)| e.last_used)
            .map(|(i, _)| i)
        else {
            return false;
        };
        let e = self.entries.swap_remove(i);
        self.used -= e.bytes;
        self.stats.evicted += 1;
        true
    }

    /// Drop every parked conversation, keeping the roots (memory pressure).
    pub fn shed(&mut self) {
        while self.evict(false) {}
    }

    /// Drop everything (the model the states belong to is going away).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.used = 0;
    }

    fn expire(&mut self, now: std::time::Instant) {
        self.entries.retain(|entry| {
            if entry.expires.is_some_and(|expires| expires <= now) {
                self.used -= entry.bytes;
                self.stats.expired += 1;
                false
            } else {
                true
            }
        });
    }

    pub(crate) fn visit_residency(&mut self, mut visit: impl FnMut(crate::affinity::Scope, usize)) {
        self.expire(std::time::Instant::now());
        for entry in &self.entries {
            if let Some(scope) = entry.scope {
                visit(scope, entry.tokens.len());
            }
        }
    }

    /// Counters and current occupancy.
    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            entries: self.entries.len(),
            bytes: self.used,
            budget: self.budget,
            ..self.stats
        }
    }
}

/// Byte offset where two rendered prompts stop agreeing, moved back to a
/// `char` boundary of `a`.
pub fn common_text_prefix(a: &str, b: &str) -> usize {
    let mut n = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
    while !a.is_char_boundary(n) {
        n -= 1;
    }
    n
}

/// Length of the longest common prefix of two token sequences.
pub fn common_token_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(u64);
    impl Snapshot for Fake {
        fn bytes(&self) -> u64 {
            self.0
        }
    }

    fn copy(s: &Fake) -> Result<Fake, ()> {
        Ok(Fake(s.0))
    }

    #[test]
    fn expired_scoped_snapshots_are_not_advertised_or_reused() {
        let mut cache = PrefixCache::new(30);
        cache.insert_scoped(vec![1, 2], Fake(10), true, Some([1; 32]));
        cache.insert(vec![1, 2], Fake(10), true);
        cache.entries[0].expires =
            Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
        assert_eq!(cache.best_len_scoped(&[1, 2, 3], Some([1; 32])), 0);
        assert!(!cache.has_root_scoped(&[1, 2], Some([1; 32])));
        cache.visit_residency(|_, _| panic!("expired scoped state must be absent"));
        assert_eq!(cache.stats().expired, 1);
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().bytes, 10);
        assert_eq!(cache.best_len(&[1, 2, 3]), 2);
    }

    #[test]
    fn scopes_isolate_roots_conversations_and_replacement_under_one_budget() {
        let mut cache = PrefixCache::new(30);
        let alice = Some([1; 32]);
        let bob = Some([2; 32]);
        assert!(cache.insert_scoped(vec![1, 2], Fake(10), true, alice));
        assert_eq!(cache.best_len_scoped(&[1, 2, 3], bob), 0);
        assert_eq!(cache.best_len(&[1, 2, 3]), 0);
        assert!(!cache.has_root_scoped(&[1, 2], bob));
        assert!(cache.insert_scoped(vec![1, 2], Fake(10), false, bob));
        assert_eq!(cache.stats().entries, 2);
        assert!(
            !cache
                .take_scoped(&[1, 2, 3], bob, copy)
                .unwrap()
                .unwrap()
                .root
        );
        assert_eq!(cache.stats().entries, 1);
        assert!(
            cache
                .take_scoped(&[1, 2, 3], alice, copy)
                .unwrap()
                .unwrap()
                .root
        );
        for index in 0..100 {
            cache.insert_scoped(vec![index], Fake(10), false, Some([index as u8; 32]));
            assert!(cache.stats().bytes <= 30);
        }
        assert_eq!(cache.best_len_scoped(&[1, 2, 3], bob), 0);
    }

    #[test]
    fn takes_the_longest_strict_prefix() {
        let mut c = PrefixCache::new(100);
        assert!(c.insert(vec![1, 2], Fake(10), true));
        assert!(c.insert(vec![1, 2, 3, 4], Fake(10), false));
        // The exact prompt is not a strict prefix of itself.
        assert_eq!(c.best_len(&[1, 2, 3, 4]), 2);
        assert_eq!(c.best_len(&[1, 2, 3, 4, 5]), 4);
        assert_eq!(c.best_len(&[9]), 0);
        let hit = c.take(&[1, 2, 3, 4, 5], copy).unwrap().unwrap();
        assert_eq!((hit.len, hit.root), (4, false));
        // The parked conversation left; the root stays after a hit.
        let hit = c.take(&[1, 2, 3, 4, 5], copy).unwrap().unwrap();
        assert_eq!((hit.len, hit.root), (2, true));
        assert!(c.has_root(&[1, 2]));
        assert_eq!(c.stats().hits, 2);
        assert_eq!(c.stats().tokens_reused, 6);
    }

    #[test]
    fn evicts_conversations_before_roots() {
        let mut c = PrefixCache::new(30);
        assert!(c.insert(vec![1], Fake(10), true));
        assert!(c.insert(vec![2], Fake(10), false));
        assert!(c.insert(vec![3], Fake(10), false));
        // Room for one more 10: the oldest conversation goes, the root stays.
        assert!(c.insert(vec![4], Fake(10), false));
        assert!(c.has_root(&[1]));
        assert_eq!(c.best_len(&[2, 0]), 0);
        assert_eq!(c.best_len(&[3, 0]), 1);
        // Too big for the budget at all.
        assert!(!c.insert(vec![5], Fake(31), false));
        // Only when no conversation is left does a root go.
        assert!(c.insert(vec![6], Fake(30), false));
        assert_eq!(c.stats().entries, 1);
        assert!(!c.has_root(&[1]));
    }

    #[test]
    fn roots_are_capped_and_replaced_in_place() {
        let mut c = PrefixCache::new(1000);
        for t in 0..MAX_ROOTS as u32 + 2 {
            assert!(c.insert(vec![t], Fake(1), true));
        }
        assert_eq!(c.stats().entries, MAX_ROOTS);
        assert!(!c.has_root(&[0]) && !c.has_root(&[1]));
        // Same tokens replace the entry, and a root stays a root.
        assert!(c.insert(vec![5], Fake(7), false));
        assert!(c.has_root(&[5]));
        assert_eq!(c.stats().bytes, (MAX_ROOTS as u64 - 1) + 7);
    }

    #[test]
    fn disabled_cache_stores_nothing() {
        let mut c: PrefixCache<Fake> = PrefixCache::new(0);
        assert!(!c.enabled());
        assert!(!c.insert(vec![1], Fake(1), true));
    }

    #[test]
    fn text_and_token_prefixes() {
        assert_eq!(common_text_prefix("abcd", "abxd"), 2);
        // Never inside a multi-byte character.
        assert_eq!(common_text_prefix("aé", "aè"), 1);
        assert_eq!(common_token_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
    }
}
