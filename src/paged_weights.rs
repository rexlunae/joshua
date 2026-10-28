//! Bounded, synchronous cache of quantized GPU matrix tiles backed by mmap.
//!
//! The limit covers live cached weight bytes, not allocator reservations,
//! activations, KV caches, or driver memory. Eviction waits for device work;
//! this intentionally trades overlap for a simple buffer lifetime contract.
//!
//! Opt-in: a cache is only built when a positive byte budget is configured
//! (`JOSHUA_GPU_WEIGHT_CACHE=<MiB>`, read by the `qwen3moe` loader).  For a
//! model larger than device memory the default answer is instead
//! `ExpertPlacement::Host` (see `crate::placement`), which keeps the experts
//! in host RAM on the CPU kernels; this cache is the experimental
//! upload-on-demand alternative.
//! Compressed quantized weights stay in the mmap; an expert's tile is uploaded
//! to the device on demand and evicted (LRU) when the budget is exceeded.
use std::{borrow::Cow, collections::HashMap, sync::{Arc, Mutex}};
use candle_core::{quantized::{GgmlDType, QMatMul, QStorage, QTensor}, Device, Module, Result, Tensor, D};

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct Key { offset: usize, rows: usize, cols: usize, dtype: u32 }
struct Entry { weight: QMatMul, bytes: usize, used: u64 }
#[derive(Default)]
struct State { entries: HashMap<Key, Entry>, bytes: usize, clock: u64 }

pub(crate) struct WeightCache {
    mmap: Arc<memmap2::Mmap>,
    device: Device,
    capacity: usize,
    state: Mutex<State>,
}

pub(crate) struct PagedWeight {
    cache: Arc<WeightCache>,
    tiles: Vec<(Key, usize)>,
    dtype: GgmlDType,
}

impl WeightCache {
    pub(crate) fn new(mmap: Arc<memmap2::Mmap>, device: Device, capacity: usize) -> Result<Arc<Self>> {
        if capacity == 0 { candle_core::bail!("GPU weight cache must have a positive byte budget"); }
        Ok(Arc::new(Self { mmap, device, capacity, state: Mutex::new(State::default()) }))
    }

    pub(crate) fn weight(self: &Arc<Self>, offset: usize, rows: usize, cols: usize, dtype: GgmlDType) -> Result<PagedWeight> {
        if matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16) {
            candle_core::bail!("GPU paging requires quantized weights");
        }
        if rows == 0 || cols == 0 || !cols.is_multiple_of(dtype.block_size()) {
            candle_core::bail!("invalid paged matrix shape {rows}x{cols}");
        }
        let row_bytes = (cols / dtype.block_size()).checked_mul(dtype.type_size())
            .ok_or_else(|| candle_core::Error::Msg("paged row byte count overflow".into()))?;
        let bytes = row_bytes.checked_mul(rows).and_then(|b| offset.checked_add(b))
            .ok_or_else(|| candle_core::Error::Msg("paged tensor byte count overflow".into()))?;
        if bytes > self.mmap.len() { candle_core::bail!("paged matrix exceeds model mapping"); }
        let alignment = if dtype == GgmlDType::Q8K { 4 } else { 2 };
        if !offset.is_multiple_of(alignment) || !row_bytes.is_multiple_of(alignment) {
            candle_core::bail!("paged quantized matrix is not block-aligned");
        }
        let tile_rows = self.capacity / row_bytes;
        if tile_rows == 0 { candle_core::bail!("GPU weight cache needs at least {row_bytes} bytes for one matrix row"); }
        let mut tiles = Vec::new();
        for first in (0..rows).step_by(tile_rows) {
            let count = (rows - first).min(tile_rows);
            tiles.push((Key { offset: offset + first * row_bytes, rows: count, cols, dtype: dtype as u32 }, count * row_bytes));
        }
        Ok(PagedWeight { cache: Arc::clone(self), tiles, dtype })
    }

    fn forward(&self, key: Key, bytes: usize, dtype: GgmlDType, xs: &Tensor) -> Result<Tensor> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.clock = state.clock.wrapping_add(1);
        let clock = state.clock;
        if !state.entries.contains_key(&key) {
            if state.bytes > self.capacity - bytes {
                // No entry is dropped until outstanding work has stopped
                // reading it. The lock also serializes submissions/evictions.
                self.device.synchronize()?;
                while state.bytes > self.capacity - bytes {
                    let oldest = *state.entries.iter().min_by_key(|(_, v)| v.used).unwrap().0;
                    let removed = state.entries.remove(&oldest).unwrap();
                    state.bytes -= removed.bytes;
                    drop(removed);
                }
            }
            let data = &self.mmap[key.offset..key.offset + bytes];
            let storage = QStorage::from_data(Cow::Borrowed(data), &self.device, dtype)?;
            let tensor = QTensor::new(storage, (key.rows, key.cols))?;
            // Do not let CANDLE_DEQUANTIZE_ALL expand the cache to f32.
            let weight = QMatMul::QTensor(Arc::new(tensor));
            state.entries.insert(key, Entry { weight, bytes, used: clock });
            state.bytes += bytes;
        }
        let entry = state.entries.get_mut(&key).unwrap();
        entry.used = clock;
        entry.weight.forward(xs)
    }
}

impl PagedWeight {
    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let outputs = self.tiles.iter().map(|(key, bytes)| self.cache.forward(*key, *bytes, self.dtype, xs))
            .collect::<Result<Vec<_>>>()?;
        if outputs.len() == 1 { Ok(outputs.into_iter().next().unwrap()) }
        else { Tensor::cat(&outputs, D::Minus1) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiled_eviction_matches_resident_weights() {
        let weights = Tensor::from_vec((0..8*32).map(|i| (i % 17) as f32 - 8.0).collect(), (8,32), &Device::Cpu).unwrap();
        let quantized = QTensor::quantize(&weights, GgmlDType::Q8_0).unwrap();
        let path = std::env::temp_dir().join(format!("joshua-paged-{}", std::process::id()));
        std::fs::write(&path, quantized.data().unwrap()).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
        let capacity = 2 * GgmlDType::Q8_0.type_size();
        let cache = WeightCache::new(mmap, Device::Cpu, capacity).unwrap();
        let paged = cache.weight(0, 8, 32, GgmlDType::Q8_0).unwrap();
        assert_eq!(paged.tiles.len(), 4);
        let xs = Tensor::from_vec((0..3*32).map(|i| (i % 7) as f32).collect(), (3,32), &Device::Cpu).unwrap();
        let expected = QMatMul::QTensor(Arc::new(quantized)).forward(&xs).unwrap();
        for _ in 0..2 {
            let actual = paged.forward(&xs).unwrap();
            let diff = (actual - &expected).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(diff < 1e-4, "{diff}");
            let state = cache.state.lock().unwrap();
            assert!(state.bytes <= capacity);
            assert_eq!(state.entries.len(), 1);
        }
        assert!(cache.weight(1, 8, 32, GgmlDType::Q8_0).is_err());
        assert!(cache.weight(0, 9, 32, GgmlDType::Q8_0).is_err());
        std::fs::remove_file(path).unwrap();
    }
}