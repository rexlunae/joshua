//! Content identity for complete chunks of post-template prompt tokens.
//!
//! Keys name token prefixes, not cache residency or permission to restore a
//! snapshot. In particular, shared chunks need not be a whole-entry cache hit.

use sha2::{Digest, Sha256};
use std::fmt;

/// Tokens covered by one hash-chain step.
pub const HASH_CHUNK: usize = 16;
/// Tokens covered by one storage chunk (four hash-chain steps).
pub const STORAGE_CHUNK: usize = 64;

/// SHA-256 identity of a complete token prefix.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkKey(pub [u8; 32]);

impl fmt::Debug for ChunkKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Model, tokenizer, template and tenant namespace for a prefix chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Salt([u8; 32]);

impl Salt {
    /// Hash `joshua/prefix-salt/v1`, followed by the four byte strings in
    /// argument order, each prefixed by its byte length as a little-endian u64.
    /// Fingerprints and digests are supplied by the caller; no files are read.
    pub fn new(
        model_fingerprint: &[u8],
        tokenizer_digest: &[u8],
        template_id: &[u8],
        tenant: &[u8],
    ) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"joshua/prefix-salt/v1");
        for component in [model_fingerprint, tokenizer_digest, template_id, tenant] {
            hash.update((component.len() as u64).to_le_bytes());
            hash.update(component);
        }
        Self(hash.finalize().into())
    }
}

/// Incremental chain builder. A step accepts exactly one complete hash chunk.
pub struct ChainBuilder {
    tip: ChunkKey,
}

impl ChainBuilder {
    /// Seed with SHA-256(`joshua/prefix-key/v1` || salt), independent of process.
    pub fn new(salt: &Salt) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"joshua/prefix-key/v1");
        hash.update(salt.0);
        Self::from_tip(ChunkKey(hash.finalize().into()))
    }

    /// Resume after a complete chunk in the same salt namespace.
    pub fn from_tip(tip: ChunkKey) -> Self {
        Self { tip }
    }

    /// The seed before the first step, or the last complete prefix key.
    pub fn tip(&self) -> ChunkKey {
        self.tip
    }

    /// Hash the previous digest followed by 16 little-endian u32 token IDs.
    pub fn push(&mut self, chunk: &[u32; HASH_CHUNK]) -> ChunkKey {
        let mut hash = Sha256::new();
        hash.update(self.tip.0);
        for token in chunk {
            hash.update(token.to_le_bytes());
        }
        self.tip = ChunkKey(hash.finalize().into());
        self.tip
    }
}

/// Key each complete 16-token chunk; an incomplete tail has no key.
pub fn chain(salt: &Salt, tokens: &[u32]) -> Vec<ChunkKey> {
    let mut builder = ChainBuilder::new(salt);
    let mut keys = Vec::with_capacity(tokens.len() / HASH_CHUNK);
    for chunk in tokens.chunks_exact(HASH_CHUNK) {
        keys.push(builder.push(chunk.try_into().unwrap()));
    }
    keys
}

/// Number of equal leading hash chunks, not whole-entry cache-hit length.
pub fn shared_chunks(first: &[ChunkKey], second: &[ChunkKey]) -> usize {
    first
        .iter()
        .zip(second)
        .take_while(|(left, right)| left == right)
        .count()
}

/// Strict-prefix cap: leave at least one prompt token to produce logits.
pub fn usable_chunks(prompt_len: usize) -> usize {
    prompt_len.saturating_sub(1) / HASH_CHUNK
}

/// Borrow the chain and select keys at complete 64-token storage boundaries.
/// These are existing keys (indices 3, 7, 11, ...), never rehashed or regrouped.
pub fn storage_keys(keys: &[ChunkKey]) -> impl Iterator<Item = ChunkKey> + '_ {
    keys.iter()
        .skip(STORAGE_CHUNK / HASH_CHUNK - 1)
        .step_by(STORAGE_CHUNK / HASH_CHUNK)
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix_cache::{common_token_prefix, PrefixCache, Snapshot};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    struct FakeSnapshot;

    impl Snapshot for FakeSnapshot {
        fn bytes(&self) -> u64 {
            1
        }
    }

    fn salt() -> Salt {
        Salt::new(b"m", b"t", b"c", b"")
    }

    fn random_tokens(random: &mut StdRng, len: usize) -> Vec<u32> {
        (0..len).map(|_| random.gen_range(0..4)).collect()
    }

    #[test]
    fn chain_match_is_common_prefix_in_whole_chunks() {
        let mut random = StdRng::seed_from_u64(201);
        for _ in 0..2_000 {
            let shared_len = random.gen_range(0..=200);
            let shared = random_tokens(&mut random, shared_len);
            let mut first = shared.clone();
            let mut second = shared;
            let first_tail = random.gen_range(0..=100);
            let second_tail = random.gen_range(0..=100);
            first.extend(random_tokens(&mut random, first_tail));
            second.extend(random_tokens(&mut random, second_tail));
            assert_eq!(
                shared_chunks(&chain(&salt(), &first), &chain(&salt(), &second)),
                common_token_prefix(&first, &second) / HASH_CHUNK,
            );
        }
    }

    #[test]
    fn chunk_aligned_entry_matches_exactly_when_prefix_cache_would() {
        let mut random = StdRng::seed_from_u64(202);
        for _ in 0..2_000 {
            let entry_len = HASH_CHUNK * random.gen_range(1..=12);
            let entry = random_tokens(&mut random, entry_len);
            let prompt_len = random.gen_range(0..=240);
            let mut prompt = random_tokens(&mut random, prompt_len);
            let shared_len = if random.gen_bool(0.5) {
                entry.len().min(prompt.len())
            } else {
                random.gen_range(0..=entry.len().min(prompt.len()))
            };
            prompt[..shared_len].copy_from_slice(&entry[..shared_len]);
            let mut cache = PrefixCache::new(1);
            assert!(cache.insert(entry.clone(), FakeSnapshot, false));
            let entry_keys = chain(&salt(), &entry);
            let prompt_keys = chain(&salt(), &prompt);
            assert_eq!(
                prompt_keys.starts_with(&entry_keys) && entry.len() < prompt.len(),
                cache.best_len(&prompt) == entry.len(),
            );
        }
    }

    #[test]
    fn keys_never_miss_a_real_hit() {
        let mut random = StdRng::seed_from_u64(203);
        let mut hits = 0;
        for _ in 0..2_000 {
            let prompt_len = random.gen_range(0..=300);
            let prompt = random_tokens(&mut random, prompt_len);
            let prompt_keys = chain(&salt(), &prompt);
            let mut cache = PrefixCache::new(6);
            let mut overlap = 0;
            for _ in 0..random.gen_range(1..=6) {
                let entry_len = random.gen_range(1..=300);
                let mut entry = random_tokens(&mut random, entry_len);
                let shared_len = if random.gen_bool(0.5) {
                    entry_len.min(prompt_len)
                } else {
                    random.gen_range(0..=entry_len.min(prompt_len))
                };
                entry[..shared_len].copy_from_slice(&prompt[..shared_len]);
                overlap = overlap.max(shared_chunks(&chain(&salt(), &entry), &prompt_keys));
                assert!(cache.insert(entry, FakeSnapshot, false));
            }
            let best_len = cache.best_len(&prompt);
            hits += usize::from(best_len >= HASH_CHUNK);
            assert!(best_len / HASH_CHUNK <= overlap);
            assert!(best_len / HASH_CHUNK <= usable_chunks(prompt.len()));
        }
        assert!(hits > 100);
    }

    #[test]
    fn shared_chunks_can_overpromise_a_whole_entry_hit() {
        let prompt = vec![1; 65];
        for entry_len in [32, 65, 80] {
            let mut entry = vec![1; entry_len];
            entry[entry_len - 1] = 2;
            let overlap = shared_chunks(&chain(&salt(), &entry), &chain(&salt(), &prompt));
            assert!(overlap.min(usable_chunks(prompt.len())) > 0);
            let mut cache = PrefixCache::new(1);
            assert!(cache.insert(entry, FakeSnapshot, false));
            assert_eq!(cache.best_len(&prompt), 0);
        }
    }

    #[test]
    fn exact_prompt_is_not_a_usable_whole_entry() {
        for entry_len in [HASH_CHUNK, STORAGE_CHUNK] {
            let entry = vec![1; entry_len];
            let keys = chain(&salt(), &entry);
            assert_eq!(shared_chunks(&keys, &keys), entry_len / HASH_CHUNK);
            assert_eq!(usable_chunks(entry_len), keys.len() - 1);
            let mut cache = PrefixCache::new(1);
            assert!(cache.insert(entry.clone(), FakeSnapshot, false));
            assert_eq!(cache.best_len(&entry), 0);
            let mut prompt = entry;
            prompt.push(2);
            assert_eq!(cache.best_len(&prompt), entry_len);
            assert_eq!(usable_chunks(prompt.len()), keys.len());
        }
    }

    #[test]
    fn salt_separates_every_component() {
        let tokens = vec![1; STORAGE_CHUNK];
        let original = chain(&salt(), &tokens);
        for changed in [
            Salt::new(b"M", b"t", b"c", b""),
            Salt::new(b"m", b"T", b"c", b""),
            Salt::new(b"m", b"t", b"C", b""),
            Salt::new(b"m", b"t", b"c", b"tenant"),
        ] {
            assert_eq!(shared_chunks(&original, &chain(&changed, &tokens)), 0);
        }
        assert_ne!(
            Salt::new(b"ab", b"c", b"", b""),
            Salt::new(b"a", b"bc", b"", b"")
        );
        assert_ne!(
            Salt::new(b"", b"ab", b"c", b""),
            Salt::new(b"", b"a", b"bc", b"")
        );
        assert_ne!(
            Salt::new(b"", b"", b"ab", b"c"),
            Salt::new(b"", b"", b"a", b"bc")
        );
    }

    #[test]
    fn partial_trailing_chunk_is_not_keyed() {
        for len in 0..=160 {
            let tokens = vec![1; len];
            assert_eq!(chain(&salt(), &tokens).len(), len / HASH_CHUNK);
            assert_eq!(
                chain(&salt(), &tokens),
                chain(&salt(), &tokens[..len / HASH_CHUNK * HASH_CHUNK])
            );
        }
    }

    #[test]
    fn usable_chunks_leaves_a_token_for_logits() {
        for (len, expected) in [
            (0, 0),
            (1, 0),
            (15, 0),
            (16, 0),
            (17, 1),
            (32, 1),
            (33, 2),
            (64, 3),
            (65, 4),
        ] {
            assert_eq!(usable_chunks(len), expected);
        }
        assert_eq!(usable_chunks(usize::MAX), (usize::MAX - 1) / HASH_CHUNK);
    }

    #[test]
    fn storage_keys_are_every_fourth_hash_key() {
        for len in 0..=256 {
            let keys = chain(&salt(), &vec![1; len]);
            let storage: Vec<_> = storage_keys(&keys).collect();
            assert_eq!(storage.len(), len / STORAGE_CHUNK);
            for (index, key) in storage.iter().enumerate() {
                assert_eq!(*key, keys[(index + 1) * 4 - 1]);
            }
        }
    }

    #[test]
    fn incremental_chain_equals_batch() {
        let mut random = StdRng::seed_from_u64(204);
        for _ in 0..500 {
            let len = random.gen_range(0..=300);
            let tokens: Vec<u32> = (0..len).map(|_| random.gen()).collect();
            let expected = chain(&salt(), &tokens);
            let mut builder = ChainBuilder::new(&salt());
            let mut keys = Vec::new();
            for chunk in tokens.chunks_exact(HASH_CHUNK) {
                keys.push(builder.push(chunk.try_into().unwrap()));
                builder = ChainBuilder::from_tip(builder.tip());
            }
            assert_eq!(keys, expected);
        }
    }

    #[test]
    fn equal_chunks_at_different_positions_do_not_alias() {
        let mut tokens = vec![1; 2 * HASH_CHUNK];
        let original = chain(&salt(), &tokens);
        assert_ne!(original[0], original[1]);
        tokens[0] = 3;
        let changed = chain(&salt(), &tokens);
        assert_eq!(shared_chunks(&original, &changed), 0);
        assert_ne!(original[1], changed[1]);
    }

    #[test]
    fn golden_vector_pins_the_wire_format() {
        assert_eq!(
            format!("{:?}", ChunkKey(salt().0)),
            "883754f03ec688332b3f42de917ecb2db2cd3c9d4bf4845753d8a82800af4fd1"
        );
        assert_eq!(
            format!("{:?}", ChainBuilder::new(&salt()).tip()),
            "105943b6c76bb6e7480a8b6c022fcf9339fbcc9da20ef1df5d97d2f83eda3c7a"
        );
        let keys = chain(&salt(), &(0..48).collect::<Vec<_>>());
        assert_eq!(
            format!("{:?}", keys[0]),
            "76dcd9e80dd2b1b52bba715ec939de4b436b5ac1efc158aaf3d93a7bdce4ab14"
        );
        assert_eq!(
            format!("{:?}", keys[2]),
            "517aa28c60396ff99079cf8ba7bcce1e2ae232e08ef4ee1ddfedbc917da1d70a"
        );
    }
}
