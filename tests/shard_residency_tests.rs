//! Residency of input-column shards mapped from GGUF files on disk (#90).
//!
//! `ShardedTensor::map_file` maps the whole local file but must only fault in
//! the byte ranges of its own columns. These tests write real GGUF files to a
//! disk-backed directory, evict them from the page cache with
//! `posix_fadvise(DONTNEED)`, run a shard, and then measure the mapping with
//! `mincore(2)` (page-cache residency per page) and `/proc/self/smaps` (pages
//! mapped into this process). Linux only.
#![cfg(target_os = "linux")]

use std::fs::File;
use std::io::BufReader;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use candle_core::{
    quantized::{gguf_file, GgmlDType, QTensor},
    Device, Tensor,
};
use joshua::{distributed::shard::ShardedTensor, gguf_ext};

/// Q8_0, 8 rows of 3,932,160 columns: 4,177,920 bytes per row, ~33 MB in all.
/// Four shards own ~1 MiB of every row, so each row has a shard range that is
/// much larger than a page and off-shard neighbours on both sides of rank 1.
const ROWS: usize = 8;
const WIDTH: usize = 32 * 4 * 30_720;
const SHARDS: usize = 4;
const RANK: usize = 1;

struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn temp_path(name: &str) -> TempFile {
    // CARGO_TARGET_TMPDIR lives under the target directory: disk backed, unlike
    // a tmpfs /tmp, where DONTNEED cannot evict anything.
    TempFile(Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "shard-residency-{}-{name}-{}.gguf",
        std::process::id(),
        uuid::Uuid::new_v4()
    )))
}

fn weights(rows: usize, width: usize, seed: usize) -> Vec<f32> {
    (0..rows * width)
        .map(|i| (((i * 7 + seed * 13) % 251) as f32 - 125.0) / 128.0)
        .collect()
}

fn write_gguf(
    path: &Path,
    metadata: &[(&str, &gguf_file::Value)],
    tensors: &[(&str, &QTensor)],
) -> anyhow::Result<()> {
    let mut file = File::create_new(path)?;
    gguf_file::write(&mut file, metadata, tensors)?;
    // DONTNEED cannot evict dirty pages; make every page clean first.
    file.sync_all()?;
    Ok(())
}

fn quantize(data: Vec<f32>, rows: usize, width: usize) -> anyhow::Result<QTensor> {
    Ok(QTensor::quantize(
        &Tensor::from_vec(data, (rows, width), &Device::Cpu)?,
        GgmlDType::Q8_0,
    )?)
}

fn page_size() -> usize {
    // SAFETY: sysconf has no memory-safety preconditions.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

fn evict(file: &File) -> anyhow::Result<()> {
    // SAFETY: plain syscall on a valid descriptor.
    let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    anyhow::ensure!(rc == 0, "posix_fadvise(DONTNEED) failed: {rc}");
    Ok(())
}

/// The backing device's readahead window (`read_ahead_kb`), 128 KiB if unknown.
fn readahead_bytes(path: &Path) -> usize {
    use std::os::unix::fs::MetadataExt;
    let dev = std::fs::metadata(path).map(|m| m.dev()).unwrap_or(0);
    let (major, minor) = (libc::major(dev), libc::minor(dev));
    std::fs::read_to_string(format!("/sys/class/bdi/{major}:{minor}/read_ahead_kb"))
        .ok()
        .and_then(|kb| kb.trim().parse::<usize>().ok())
        .unwrap_or(128)
        * 1024
}

/// The single live mapping of `path` in this process: (start address, length).
fn mapping_of(path: &Path) -> anyhow::Result<(usize, usize)> {
    let path = std::fs::canonicalize(path)?;
    let path = path.to_str().unwrap();
    let maps = std::fs::read_to_string("/proc/self/maps")?;
    let found: Vec<_> = maps
        .lines()
        .filter(|line| line.ends_with(path))
        .map(|line| {
            let range = line.split_whitespace().next().unwrap();
            let (a, b) = range.split_once('-').unwrap();
            let (a, b) = (
                usize::from_str_radix(a, 16).unwrap(),
                usize::from_str_radix(b, 16).unwrap(),
            );
            (a, b - a)
        })
        .collect();
    anyhow::ensure!(
        found.len() == 1,
        "expected one mapping of {path}: {found:?}"
    );
    Ok(found[0])
}

/// `Rss` of the mapping starting at `start`, from /proc/self/smaps.
fn smaps_rss_bytes(start: usize) -> anyhow::Result<u64> {
    let smaps = std::fs::read_to_string("/proc/self/smaps")?;
    let prefix = format!("{start:x}-");
    let mut lines = smaps.lines().skip_while(|line| !line.starts_with(&prefix));
    anyhow::ensure!(lines.next().is_some(), "mapping not found in smaps");
    for line in lines {
        if let Some(value) = line.strip_prefix("Rss:") {
            let kb: u64 = value.trim().trim_end_matches("kB").trim().parse()?;
            return Ok(kb * 1024);
        }
    }
    anyhow::bail!("no Rss line for mapping")
}

/// Page-cache residency of each page of `[start, start + len)`.
fn resident_pages(start: usize, len: usize) -> anyhow::Result<Vec<bool>> {
    let pages = len.div_ceil(page_size());
    let mut vec = vec![0u8; pages];
    // SAFETY: `start` is the page-aligned start of a live mapping of `len`
    // bytes (found in /proc/self/maps while the owner is alive); mincore only
    // writes one byte per page into `vec`, which has exactly that many bytes.
    let rc = unsafe { libc::mincore(start as *mut libc::c_void, len, vec.as_mut_ptr()) };
    anyhow::ensure!(
        rc == 0,
        "mincore failed: {}",
        std::io::Error::last_os_error()
    );
    Ok(vec.into_iter().map(|byte| byte & 1 == 1).collect())
}

#[derive(Debug)]
struct Residency {
    shard_pages: usize,
    shard_resident: usize,
    off_pages: usize,
    off_resident: usize,
    rss_bytes: u64,
}

impl Residency {
    fn off_fraction(&self) -> f64 {
        self.off_resident as f64 / self.off_pages as f64
    }

    fn report(&self, phase: &str) {
        let page = page_size() as f64 / (1024.0 * 1024.0);
        eprintln!(
            "{phase:>22}: shard {:.2}/{:.2} MiB resident, off-shard {:.2}/{:.2} MiB resident \
             ({:.2}%), smaps Rss {:.2} MiB",
            self.shard_resident as f64 * page,
            self.shard_pages as f64 * page,
            self.off_resident as f64 * page,
            self.off_pages as f64 * page,
            self.off_fraction() * 100.0,
            self.rss_bytes as f64 / (1024.0 * 1024.0)
        );
    }
}

/// Classify every page of the tensor's span as shard (overlaps one of this
/// rank's row ranges) or off-shard (no overlap), then count resident pages.
fn measure(path: &Path, rows: &[Range<usize>], tensor: Range<usize>) -> anyhow::Result<Residency> {
    let page = page_size();
    let (start, len) = mapping_of(path)?;
    let resident = resident_pages(start, len)?;
    let mut shard = vec![false; resident.len()];
    for range in rows {
        for flag in &mut shard[range.start / page..range.end.div_ceil(page)] {
            *flag = true;
        }
    }
    let first = tensor.start.div_ceil(page);
    let last = tensor.end / page;
    let mut result = Residency {
        shard_pages: 0,
        shard_resident: 0,
        off_pages: 0,
        off_resident: 0,
        rss_bytes: smaps_rss_bytes(start)?,
    };
    for page in first..last {
        if shard[page] {
            result.shard_pages += 1;
            result.shard_resident += usize::from(resident[page]);
        } else {
            result.off_pages += 1;
            result.off_resident += usize::from(resident[page]);
        }
    }
    Ok(result)
}

struct Fixture {
    path: TempFile,
    info: gguf_ext::RawTensorInfo,
    data_offset: u64,
}

fn big_fixture() -> anyhow::Result<Fixture> {
    let path = temp_path("big");
    let tensor = quantize(weights(ROWS, WIDTH, 1), ROWS, WIDTH)?;
    write_gguf(&path.0, &[], &[("w", &tensor)])?;
    let header = gguf_ext::read_header(&mut BufReader::new(File::open(&path.0)?))?;
    Ok(Fixture {
        info: header.tensors["w"].clone(),
        data_offset: header.tensor_data_offset,
        path,
    })
}

/// Evict the file, map a fresh shard and return it with its absolute ranges.
fn fresh_shard(
    fixture: &Fixture,
) -> anyhow::Result<(ShardedTensor, Vec<Range<usize>>, Range<usize>)> {
    let file = File::open(&fixture.path.0)?;
    evict(&file)?;
    let shard = ShardedTensor::map_file(&file, &fixture.info, fixture.data_offset, RANK, SHARDS)?;
    let rows: Vec<_> = shard.row_ranges().collect();
    let start = (fixture.data_offset + fixture.info.offset) as usize;
    let row_bytes = WIDTH / 32 * 34;
    Ok((shard, rows, start..start + ROWS * row_bytes))
}

fn input() -> Vec<f32> {
    (0..WIDTH)
        .map(|i| ((i % 29) as f32 - 14.0) / 29.0)
        .collect()
}

#[test]
fn sharded_mapping_faults_in_only_its_own_column_ranges() -> anyhow::Result<()> {
    let fixture = big_fixture()?;
    let x = input();

    // 1. Mapping alone reads nothing.
    let (shard, rows, tensor) = fresh_shard(&fixture)?;
    let lazy = measure(&fixture.path.0, &rows, tensor.clone())?;
    lazy.report("after map_file");
    if lazy.shard_resident + lazy.off_resident > (lazy.shard_pages + lazy.off_pages) / 20 {
        // A tmpfs or otherwise unevictable backing store keeps pages resident
        // regardless of access; the measurement would be meaningless.
        eprintln!(
            "skipping: the page cache for {:?} cannot be evicted",
            fixture.path.0
        );
        return Ok(());
    }
    assert_eq!(lazy.rss_bytes, 0, "mapping must not populate");

    // 2. forward() touches every byte of the shard but no other column.
    let partial = shard.forward(&x)?;
    let used = measure(&fixture.path.0, &rows, tensor.clone())?;
    used.report("after forward");
    assert_eq!(
        used.shard_resident, used.shard_pages,
        "shard pages must be resident"
    );
    let shard_bytes = (used.shard_pages * page_size()) as u64;
    let off_bytes = used.off_resident * page_size();
    // Each row's shard is read as one sequential stream: readahead may run up to
    // two windows (synchronous + asynchronous) past its end, and fault-around
    // may round each end to a 64 KiB block. Anything beyond that would mean the
    // shard is reading other ranks' columns.
    let readahead = readahead_bytes(&fixture.path.0);
    let budget = ROWS * 2 * (2 * readahead + 64 * 1024);
    eprintln!(
        "{:>22}: read_ahead_kb {} -> off-shard budget {:.2} MiB",
        "readahead",
        readahead / 1024,
        budget as f64 / (1024.0 * 1024.0)
    );
    if budget < used.off_pages * page_size() / 2 {
        assert!(
            off_bytes <= budget,
            "off-shard residency {off_bytes} exceeds the readahead budget {budget}: {used:?}"
        );
    } else {
        eprintln!("readahead window too large for a bounded check; see the MADV_RANDOM phase");
    }
    assert!(
        used.rss_bytes < shard_bytes * 3 / 2,
        "mapped RSS {} is not bounded by the shard's {shard_bytes} bytes",
        used.rss_bytes
    );
    drop(shard);

    // 3. prefetch() advises only the shard's row ranges.
    let (shard, rows, tensor) = fresh_shard(&fixture)?;
    shard.prefetch()?;
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut prefetched = measure(&fixture.path.0, &rows, tensor.clone())?;
    while prefetched.shard_resident < prefetched.shard_pages && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
        prefetched = measure(&fixture.path.0, &rows, tensor.clone())?;
    }
    prefetched.report("after prefetch");
    // MADV_WILLNEED is a best-effort hint: the kernel may cap or drop the
    // readahead it schedules (a GitHub runner read 512 of 2048 shard pages),
    // so only require that the hint started reading the shard. What this
    // phase pins down is that the advice stays within the shard's ranges.
    assert!(
        prefetched.shard_resident > 0,
        "WILLNEED read nothing of the shard: {prefetched:?}"
    );
    assert!(
        prefetched.off_fraction() < 0.25,
        "prefetch read off-shard pages: {prefetched:?}"
    );
    assert_eq!(shard.forward(&x)?, partial, "prefetch changed the result");
    drop(shard);

    // 4. With readahead disabled, only pages that share a boundary with the
    // shard may be read: off-shard residency is then essentially zero.
    let (shard, rows, tensor) = fresh_shard(&fixture)?;
    let (start, len) = mapping_of(&fixture.path.0)?;
    // SAFETY: advice on this process's own live read-only mapping; it does not
    // change contents or validity.
    let rc = unsafe { libc::madvise(start as *mut libc::c_void, len, libc::MADV_RANDOM) };
    assert_eq!(rc, 0);
    assert_eq!(shard.forward(&x)?, partial);
    let random = measure(&fixture.path.0, &rows, tensor)?;
    random.report("MADV_RANDOM + forward");
    assert_eq!(random.shard_resident, random.shard_pages);
    // Fault-around may still map up to 64 KiB of already cached neighbours per
    // fault; nothing beyond one such window per row boundary may be read.
    let slack_pages = 2 * ROWS * (64 * 1024 / page_size());
    assert!(
        random.off_resident <= slack_pages,
        "off-shard pages read without readahead: {random:?}"
    );
    drop(shard);

    // The sum of every rank equals the unsharded product.
    let file = File::open(&fixture.path.0)?;
    let full = ShardedTensor::map_file(&file, &fixture.info, fixture.data_offset, 0, 1)?;
    let reference = full.forward(&x)?;
    let mut sum = [0.0f32; ROWS];
    for rank in 0..SHARDS {
        let shard =
            ShardedTensor::map_file(&file, &fixture.info, fixture.data_offset, rank, SHARDS)?;
        for (total, value) in sum.iter_mut().zip(shard.forward(&x)?) {
            *total += value;
        }
    }
    for (a, b) in sum.iter().zip(&reference) {
        assert!((a - b).abs() <= 1e-3 * b.abs().max(1.0), "{a} != {b}");
    }
    Ok(())
}

/// A split model: two self-contained GGUF files written in the llama.cpp
/// `gguf-split` layout (`split.no`/`split.count`/`split.tensors.count`, each
/// part holding a subset of the tensors). Joshua has no automatic split-set
/// loader; this resolves each tensor to its file through the per-file headers,
/// which is what such a loader would do, and checks that shards of a tensor in
/// one part read only that part.
#[test]
fn split_gguf_shards_match_reference_and_touch_only_their_part() -> anyhow::Result<()> {
    const SPLIT_ROWS: usize = 6;
    const SPLIT_WIDTH: usize = 32 * 3 * 4096;
    let parts = [
        temp_path("split-00001-of-00002"),
        temp_path("split-00002-of-00002"),
    ];
    let names = [["blk.0.ffn_up.weight"], ["blk.0.ffn_down.weight"]];
    let mut references = Vec::new();
    for (index, (part, tensor_names)) in parts.iter().zip(&names).enumerate() {
        let data = weights(SPLIT_ROWS, SPLIT_WIDTH, index + 2);
        let tensor = quantize(data, SPLIT_ROWS, SPLIT_WIDTH)?;
        let dequantized = tensor.dequantize(&Device::Cpu)?.to_vec2::<f32>()?;
        references.push((tensor_names[0], dequantized));
        let no = gguf_file::Value::U16(index as u16);
        let count = gguf_file::Value::U16(2);
        let total = gguf_file::Value::I32(2);
        write_gguf(
            &part.0,
            &[
                ("split.no", &no),
                ("split.count", &count),
                ("split.tensors.count", &total),
            ],
            &[(tensor_names[0], &tensor)],
        )?;
    }
    // Resolve tensors to parts from the headers and check the split metadata.
    let mut headers = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        let header = gguf_ext::read_header(&mut BufReader::new(File::open(&part.0)?))?;
        assert_eq!(header.tensors.len(), 1);
        assert!(format!("{:?}", header.metadata["split.no"]).contains(&index.to_string()));
        headers.push(header);
    }
    let x: Vec<f32> = (0..SPLIT_WIDTH)
        .map(|i| ((i % 23) as f32 - 11.0) / 23.0)
        .collect();
    for (name, weights) in &references {
        let owner = headers
            .iter()
            .position(|h| h.tensors.contains_key(*name))
            .expect("tensor resolves to exactly one part");
        let other = 1 - owner;
        let files: Vec<_> = parts
            .iter()
            .map(|p| File::open(&p.0))
            .collect::<Result<_, _>>()?;
        for file in &files {
            evict(file)?;
        }
        let header = &headers[owner];
        let info = &header.tensors[*name];
        let mut sum = vec![0.0f32; SPLIT_ROWS];
        for rank in 0..3 {
            let shard =
                ShardedTensor::map_file(&files[owner], info, header.tensor_data_offset, rank, 3)?;
            for (total, value) in sum.iter_mut().zip(shard.forward(&x)?) {
                *total += value;
            }
        }
        for (row, actual) in weights.iter().zip(&sum) {
            let expected: f32 = row.iter().zip(&x).map(|(w, v)| w * v).sum();
            assert!(
                (actual - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                "{name}: {actual} != {expected}"
            );
        }
        // The other part was never read: map it read-only and check residency.
        // SAFETY: the test owns the file and never modifies it while mapped.
        let probe = unsafe { memmap2::Mmap::map(&files[other])? };
        let pages = resident_pages(probe.as_ptr() as usize, probe.len())?;
        let resident = pages.iter().filter(|&&r| r).count();
        eprintln!(
            "split: shards of {name} (part {}) left part {} with {resident}/{} pages resident",
            owner + 1,
            other + 1,
            pages.len()
        );
        assert_eq!(resident, 0, "a shard of {name} read the other split part");
    }
    Ok(())
}
