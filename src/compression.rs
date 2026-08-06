//! Detecting model files that cannot usefully be memory-mapped.
//!
//! Joshua's whole loading strategy is `mmap`: the GGUF file is mapped once and
//! quantized tensors are reinterpreted in place out of the mapping (see
//! [`crate::mmap_tensor`]).  That only works when the bytes on disk *are* the
//! model.  Two situations break the assumption, and both are easy to hit by
//! accident because the file is still named `.gguf`:
//!
//! * **Container compression** — the file is a gzip/zstd/xz/… stream wrapping
//!   the GGUF.  Mapping it maps the compressed bytes; nothing in the mapping
//!   is a tensor, so the load fails outright (historically with a confusing
//!   "magic mismatch" from the header parser).
//! * **Filesystem compression** — btrfs/ZFS/NTFS store the file compressed and
//!   transparently inflate it on access.  The mapping *works*, which is why
//!   this one goes unnoticed, but every page fault has to decompress a block,
//!   so weights can no longer be paged in cheaply, and the pages are dirty
//!   anonymous-ish copies rather than the clean shared page-cache pages the
//!   engine is designed around.  Load and inference are dramatically slower.
//!
//! [`detect`] recognises both so the caller can say so up front instead of
//! letting the user discover it as an inscrutable parse error or as
//! mysteriously bad throughput.

use std::fs::File;

/// A compression container recognised by its magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Container {
    /// Human-readable format name, e.g. `"gzip"`.
    pub format: &'static str,
    /// A command that turns the file back into plain bytes.
    pub decompress: &'static str,
}

/// Why a model file is a poor candidate for `mmap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// The bytes on disk are a compression container, not the model.
    Container(Container),
    /// The filesystem stores the file compressed (or it is sparse): its
    /// on-disk allocation is far smaller than its length.
    Filesystem {
        /// Bytes actually allocated on disk.
        allocated: u64,
        /// Apparent file length.
        len: u64,
    },
}

impl std::fmt::Display for Compression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Container(c) => write!(
                f,
                "the file is a {} stream, not raw GGUF — mapping it maps the compressed \
                 bytes, so no tensor can be read in place. Decompress it first ({})",
                c.format, c.decompress
            ),
            Self::Filesystem { allocated, len } => write!(
                f,
                "the file occupies {} MiB on disk for {} MiB of content, so the filesystem \
                 is storing it compressed (or it is sparse). The mapping still works, but \
                 every page fault has to decompress a block instead of handing back a \
                 shared page-cache page, which makes loading and inference far slower. \
                 Store the model uncompressed (e.g. `chattr -c`, btrfs `compress=no`, or \
                 `cp` it onto an uncompressed filesystem)",
                allocated / (1024 * 1024),
                len / (1024 * 1024)
            ),
        }
    }
}

/// GGUF's own magic, checked first so a well-formed model short-circuits.
pub const GGUF_MAGIC: &[u8] = b"GGUF";

/// How much of the file's head is examined — enough for the longest magic in
/// [`CONTAINERS`], with room to spare.
const HEAD_BYTES: usize = 8;

/// Magic-byte prefixes of the compression containers a `.gguf` is plausibly
/// wrapped in.  Longest-to-shortest is irrelevant here: no prefix in this
/// table is a prefix of another.
const CONTAINERS: &[(&[u8], Container)] = &[
    (
        b"\x1f\x8b",
        Container {
            format: "gzip",
            decompress: "gunzip",
        },
    ),
    (
        b"\x28\xb5\x2f\xfd",
        Container {
            format: "zstd",
            decompress: "unzstd",
        },
    ),
    (
        b"\xfd7zXZ\x00",
        Container {
            format: "xz",
            decompress: "unxz",
        },
    ),
    (
        b"BZh",
        Container {
            format: "bzip2",
            decompress: "bunzip2",
        },
    ),
    (
        b"\x04\x22\x4d\x18",
        Container {
            format: "lz4",
            decompress: "unlz4",
        },
    ),
    (
        b"\x5d\x00\x00",
        Container {
            format: "lzma",
            decompress: "unlzma",
        },
    ),
    (
        b"\x1f\x9d",
        Container {
            format: "compress(1) .Z",
            decompress: "uncompress",
        },
    ),
    (
        b"PK\x03\x04",
        Container {
            format: "zip archive",
            decompress: "unzip",
        },
    ),
    (
        b"PK\x05\x06",
        Container {
            format: "zip archive",
            decompress: "unzip",
        },
    ),
    (
        b"7z\xbc\xaf\x27\x1c",
        Container {
            format: "7-Zip archive",
            decompress: "7z x",
        },
    ),
];

/// A bare zlib stream has no distinctive magic: the first byte encodes the
/// compression method (`8`) and window size, and the two-byte header is a
/// multiple of 31.  Roughly one arbitrary byte pair in 500 satisfies that, so
/// the test is only trustworthy once the file has already been shown not to be
/// what it claims to be — see [`detect`]'s `expected_magic`.
fn is_zlib(head: &[u8]) -> bool {
    match head {
        [cmf, flg, ..] => cmf & 0x0f == 8 && (u16::from(*cmf) * 256 + u16::from(*flg)) % 31 == 0,
        _ => false,
    }
}

/// Files below this size are never reported as filesystem-compressed: the
/// allocation of a small file is dominated by inline extents, tail packing and
/// block rounding, which the ratio test would misread.  Real models are
/// comfortably above it.
const MIN_FILESYSTEM_CHECK_BYTES: u64 = 8 * 1024 * 1024;

/// Report the file as filesystem-compressed only when its allocation is below
/// this fraction (9/10) of its length, so ordinary slack never trips it.
const FILESYSTEM_RATIO_NUM: u64 = 9;
const FILESYSTEM_RATIO_DEN: u64 = 10;

/// Inspect `file` and report why it cannot usefully be mapped, if so.
///
/// `expected_magic` is the file format's own leading bytes when the caller
/// knows them ([`GGUF_MAGIC`] for a model file, `None` for a format like
/// safetensors that starts with a length rather than a magic).  Supplying it
/// both short-circuits the common case and licenses the weak zlib test, which
/// is only meaningful once the file is known not to be what it claims.
///
/// The head of the file is read positionally, so the handle's read position is
/// neither relied on nor (on Unix) changed.  Returns `None` when the file looks
/// like a plain, uncompressed model — including when the leading bytes are
/// simply unrecognised, since a corrupt or foreign file is the header parser's
/// business rather than this one's.
pub fn detect(file: &File, expected_magic: Option<&[u8]>) -> Option<Compression> {
    if let Some(container) = detect_container(file, expected_magic) {
        return Some(Compression::Container(container));
    }
    detect_filesystem(file)
}

/// Detect compression in a GGUF model file.
pub fn detect_gguf(file: &File) -> Option<Compression> {
    detect(file, Some(GGUF_MAGIC))
}

/// Read up to the first [`HEAD_BYTES`] of `file` without depending on — or,
/// where the platform allows, disturbing — its current read position.
fn read_head(file: &File) -> Option<(usize, [u8; HEAD_BYTES])> {
    let mut head = [0u8; HEAD_BYTES];
    let mut filled = 0;
    while filled < head.len() {
        #[cfg(unix)]
        let n = {
            use std::os::unix::fs::FileExt;
            file.read_at(&mut head[filled..], filled as u64).ok()?
        };
        #[cfg(windows)]
        let n = {
            use std::os::windows::fs::FileExt;
            file.seek_read(&mut head[filled..], filled as u64).ok()?
        };
        #[cfg(not(any(unix, windows)))]
        let n = {
            use std::io::{Seek, SeekFrom};
            let mut f = file;
            f.seek(SeekFrom::Start(filled as u64)).ok()?;
            f.read(&mut head[filled..]).ok()?
        };
        // A short file simply has no room for a magic; stop at EOF.
        if n == 0 {
            break;
        }
        filled += n;
    }
    Some((filled, head))
}

/// Match the leading bytes against the known compression containers.
fn detect_container(file: &File, expected_magic: Option<&[u8]>) -> Option<Container> {
    let (n, head) = read_head(file)?;
    let head = &head[..n];

    if expected_magic.is_some_and(|m| head.starts_with(m)) {
        return None;
    }
    for (magic, container) in CONTAINERS {
        if head.starts_with(magic) {
            return Some(*container);
        }
    }
    // Only reached for a file that was supposed to carry a known magic and
    // does not, so a header that merely *could* be zlib is worth reporting.
    if expected_magic.is_some() && is_zlib(head) {
        return Some(Container {
            format: "zlib",
            decompress: "e.g. `pigz -dz`",
        });
    }
    None
}

/// Compare the file's on-disk allocation with its length.
///
/// A transparently compressed file (btrfs/ZFS `compress=…`, NTFS compression)
/// reports far fewer allocated blocks than its size — as does a sparse file,
/// which is equally bad news for a mapping.
#[cfg(unix)]
fn detect_filesystem(file: &File) -> Option<Compression> {
    use std::os::unix::fs::MetadataExt;

    let md = file.metadata().ok()?;
    let len = md.len();
    if len < MIN_FILESYSTEM_CHECK_BYTES {
        return None;
    }
    // `blocks()` is in 512-byte units by POSIX convention.  Zero means the
    // filesystem does not report allocation (some network filesystems); that
    // is unknown, not compressed.
    let allocated = md.blocks().checked_mul(512)?;
    if allocated == 0 {
        return None;
    }
    if allocated.saturating_mul(FILESYSTEM_RATIO_DEN) < len.saturating_mul(FILESYSTEM_RATIO_NUM) {
        return Some(Compression::Filesystem { allocated, len });
    }
    None
}

/// Non-Unix platforms expose no portable allocation figure in `std`, so only
/// container compression is detectable there.
#[cfg(not(unix))]
fn detect_filesystem(_file: &File) -> Option<Compression> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write `bytes` to a uniquely named temp file and open it for reading.
    fn temp_file(name: &str, bytes: &[u8]) -> File {
        let path =
            std::env::temp_dir().join(format!("joshua-compression-{}-{name}", std::process::id()));
        let mut f = File::create(&path).expect("create temp file");
        f.write_all(bytes).expect("write temp file");
        drop(f);
        let opened = File::open(&path).expect("open temp file");
        let _ = std::fs::remove_file(&path); // unlinked; the handle stays valid
        opened
    }

    #[test]
    fn plain_gguf_is_not_flagged() {
        let f = temp_file("gguf", b"GGUF\x03\x00\x00\x00rest of the header");
        assert_eq!(detect_gguf(&f), None);
    }

    #[test]
    fn gzip_is_detected() {
        let f = temp_file("gz", b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03payload");
        match detect_gguf(&f) {
            Some(Compression::Container(c)) => assert_eq!(c.format, "gzip"),
            other => panic!("expected gzip, got {other:?}"),
        }
    }

    #[test]
    fn common_containers_are_detected() {
        let cases: &[(&str, &[u8], &str)] = &[
            ("zstd", b"\x28\xb5\x2f\xfd\x00\x00\x00\x00", "zstd"),
            ("xz", b"\xfd7zXZ\x00\x00\x00", "xz"),
            ("bz2", b"BZh9payload", "bzip2"),
            ("lz4", b"\x04\x22\x4d\x18payload", "lz4"),
            ("zip", b"PK\x03\x04payload", "zip archive"),
            ("7z", b"7z\xbc\xaf\x27\x1c\x00\x00", "7-Zip archive"),
            ("Z", b"\x1f\x9dpayload!", "compress(1) .Z"),
            // Bare zlib: 0x78 0x9c is the standard default-compression header.
            ("zlib", b"\x78\x9cpayload!", "zlib"),
        ];
        for (name, bytes, expected) in cases {
            let f = temp_file(name, bytes);
            match detect_gguf(&f) {
                Some(Compression::Container(c)) => assert_eq!(&c.format, expected, "for {name}"),
                other => panic!("expected {expected} for {name}, got {other:?}"),
            }
        }
    }

    #[test]
    fn unrecognised_and_tiny_files_are_left_alone() {
        // Not a container and not GGUF: the header parser's problem, not ours.
        assert_eq!(detect_gguf(&temp_file("junk", b"not a model at all")), None);
        // Too short to hold any magic.
        assert_eq!(detect_gguf(&temp_file("tiny", b"G")), None);
        assert_eq!(detect_gguf(&temp_file("empty", b"")), None);
    }

    #[test]
    fn zlib_heuristic_does_not_fire_on_gguf_or_noise() {
        assert!(!is_zlib(b"GGUF"));
        // Right check value, wrong compression method nibble.
        assert!(!is_zlib(b"\x79\x9b"));
        assert!(is_zlib(b"\x78\x01"));
        assert!(is_zlib(b"\x78\xda"));
    }

    /// About one byte pair in 500 passes the zlib header test by chance, so it
    /// is only consulted for a format whose real magic is known and absent.
    #[test]
    fn zlib_heuristic_needs_a_known_magic_to_contradict() {
        // A safetensors file starts with a little-endian header length; this
        // one happens to look like a zlib header.
        let f = temp_file("st", b"\x78\x9c\x00\x00\x00\x00\x00\x00{\"__met");
        assert_eq!(detect(&f, None), None);
        // The same bytes in a file that claimed to be GGUF are worth flagging.
        let f = temp_file("st2", b"\x78\x9c\x00\x00\x00\x00\x00\x00{\"__met");
        assert!(matches!(
            detect(&f, Some(GGUF_MAGIC)),
            Some(Compression::Container(_))
        ));
    }

    /// A sparse file is allocation-poor in exactly the way a transparently
    /// compressed one is, which makes it the portable way to exercise the
    /// ratio test without needing a btrfs mount.
    #[cfg(unix)]
    #[test]
    fn sparse_file_trips_the_allocation_ratio() {
        let path =
            std::env::temp_dir().join(format!("joshua-compression-{}-sparse", std::process::id()));
        let mut f = File::create(&path).expect("create sparse file");
        // A real 64 KiB prefix — so the filesystem reports an allocation at all
        // rather than a flat zero — inside a 64 MiB file whose remainder is a
        // hole, putting the ratio far below the 9/10 threshold.
        f.write_all(&[0x5Au8; 64 * 1024]).expect("write prefix");
        f.set_len(64 * 1024 * 1024).expect("set_len");
        drop(f);
        let opened = File::open(&path).expect("open sparse file");
        let found = detect_gguf(&opened);
        let _ = std::fs::remove_file(&path);

        // Filesystems that always fully allocate (or do not report blocks)
        // legitimately report nothing; accept that rather than fail there.
        match found {
            Some(Compression::Filesystem { allocated, len }) => {
                assert_eq!(len, 64 * 1024 * 1024);
                assert!(allocated < len, "allocated {allocated} len {len}");
            }
            None => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn fully_allocated_file_is_not_flagged() {
        // Incompressible bytes, so the check is not defeated by /tmp itself
        // living on a compressing filesystem.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let bytes: Vec<u8> = (0..12 * 1024 * 1024)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect();
        let f = temp_file("dense", &bytes);
        assert_eq!(detect_gguf(&f), None);
    }

    #[test]
    fn messages_name_the_problem() {
        let container = Compression::Container(Container {
            format: "gzip",
            decompress: "gunzip",
        });
        let msg = container.to_string();
        assert!(msg.contains("gzip") && msg.contains("gunzip"), "{msg}");

        let fs = Compression::Filesystem {
            allocated: 1024 * 1024,
            len: 64 * 1024 * 1024,
        };
        let msg = fs.to_string();
        assert!(msg.contains("1 MiB") && msg.contains("64 MiB"), "{msg}");
    }
}
