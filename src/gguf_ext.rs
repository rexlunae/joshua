//! A GGUF header reader that tolerates quantization types candle does not know.
//!
//! candle maps every tensor's dtype through
//! `GgmlDType::from_u32`, which hard-fails on anything outside its own table:
//!
//! ```text
//! _ => crate::bail!("unknown dtype for tensor {u}")
//! ```
//!
//! Type 39 is `GGML_TYPE_MXFP4`, the format Kimi-K3-class models ship in.
//! Because tensor infos are parsed eagerly, a single MXFP4 tensor makes
//! candle reject the *entire file* at header-read time — before any weight is
//! touched, and regardless of whether the caller intended to decode it.
//!
//! This reader parses the same header but keeps each tensor's dtype as its raw
//! `u32`, so unknown types survive to be handled by [`crate::mxfp4`],
//! [`crate::iq2xxs`] (or reported precisely).  Metadata is decoded into
//! candle's own [`gguf_file::Value`] so existing hyper-parameter code keeps
//! working unchanged.
//!
//! Only the header is read.  Tensor *data* is never touched here — callers
//! borrow it from the memory mapping via [`crate::mmap_tensor`], which is what
//! keeps a model far larger than RAM loadable.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use candle_core::quantized::gguf_file::{self, Value, VersionedMagic};
use candle_core::quantized::GgmlDType;

use crate::{JoshuaError, Result};

const MAGIC: u32 = 0x4655_4747; // "GGUF", little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

/// Guards against a corrupt or hostile header claiming an absurd count and
/// making us attempt a huge allocation before any real data is read.
const MAX_COUNT: u64 = 1 << 24;

/// Per-string cap.  Real GGUF metadata strings are small — tokenizer entries,
/// templates, names — at most a few MiB even for huge vocabs, and an embedded
/// vocab is an *array* of short strings, not one long one.  Candle mirrors
/// llama.cpp's theoretical `GGUF_MAX_STRING_LENGTH` of 1 GiB, but that cap is
/// a DoS liability, not a feature: candle eagerly `vec![0u8; len]`s the claim
/// before reading a single byte, so a hostile header inside an otherwise-real
/// model file (where the "bytes remaining" check trivially passes) can force a
/// ~1 GiB transient allocation per metadata string.  A 1 GiB string would need
/// a 1 GiB buffer to materialize anyway, so accepting it and bounding the
/// allocation are mutually exclusive.  We keep everything candle accepts that
/// can actually load (nested arrays, depth cap, lenient strings) and cap
/// strings at 16 MiB — three orders of magnitude above any real value.
const MAX_STRING_LENGTH: u64 = 1 << 24;

/// Per-array element-count cap, same rationale: real tokenizer arrays
/// (`tokenizer.ggml.tokens` / `.merges` / `.scores`) top out around a million
/// elements; candle's 1 GiB would let a hostile claim drive the element loop
/// into reading and buffering gigabytes of tensor data.
const MAX_ARRAY_ELEMENTS: u64 = 1 << 24;
const MAX_VALUE_DEPTH: usize = 64;

/// GGUF dtype ids candle's `GgmlDType` table can represent.
///
/// Candle maps only 0,1,2,3,6..15,30 (see `GgmlDType::from_u32`, which is
/// crate-private, so the set is mirrored here).
pub fn is_candle_supported(dtype: u32) -> bool {
    matches!(dtype, 0..=3 | 6..=15 | 30)
}

/// Mirror of candle's (crate-private) `GgmlDType::from_u32`.
pub fn ggml_dtype_from_id(dtype: u32) -> Option<GgmlDType> {
    Some(match dtype {
        0 => GgmlDType::F32,
        1 => GgmlDType::F16,
        2 => GgmlDType::Q4_0,
        3 => GgmlDType::Q4_1,
        6 => GgmlDType::Q5_0,
        7 => GgmlDType::Q5_1,
        8 => GgmlDType::Q8_0,
        9 => GgmlDType::Q8_1,
        10 => GgmlDType::Q2K,
        11 => GgmlDType::Q3K,
        12 => GgmlDType::Q4K,
        13 => GgmlDType::Q5K,
        14 => GgmlDType::Q6K,
        15 => GgmlDType::Q8K,
        30 => GgmlDType::BF16,
        _ => return None,
    })
}

/// The reverse of [`ggml_dtype_from_id`]: a candle dtype's raw GGUF id.
///
/// `GgmlDType`'s own discriminant is *not* the GGUF id (candle inserts BF16
/// at index 2, shifting every later type), so the on-disk id must be mapped
/// explicitly when talking to the file format.
pub fn ggml_id_from_dtype(dtype: GgmlDType) -> u32 {
    match dtype {
        GgmlDType::F32 => 0,
        GgmlDType::F16 => 1,
        GgmlDType::BF16 => 30,
        GgmlDType::Q4_0 => 2,
        GgmlDType::Q4_1 => 3,
        GgmlDType::Q5_0 => 6,
        GgmlDType::Q5_1 => 7,
        GgmlDType::Q8_0 => 8,
        GgmlDType::Q8_1 => 9,
        GgmlDType::Q2K => 10,
        GgmlDType::Q3K => 11,
        GgmlDType::Q4K => 12,
        GgmlDType::Q5K => 13,
        GgmlDType::Q6K => 14,
        GgmlDType::Q8K => 15,
        // `GgmlDType` has no non_exhaustive marker; add new candle types here.
    }
}

/// A tensor's location and type, with the dtype left as its raw GGUF id.
#[derive(Debug, Clone)]
pub struct RawTensorInfo {
    /// GGML type id — e.g. 39 for MXFP4. Deliberately not narrowed to
    /// candle's `GgmlDType`, which cannot represent every type.
    pub dtype: u32,
    /// Dimensions, in GGUF order (fastest-varying first).
    pub dims: Vec<usize>,
    /// Byte offset from `tensor_data_offset`.
    pub offset: u64,
}

impl RawTensorInfo {
    /// Total element count.
    pub fn elem_count(&self) -> usize {
        self.dims.iter().product()
    }
}

/// A parsed GGUF header.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, RawTensorInfo>,
    /// Absolute offset at which tensor data begins.
    pub tensor_data_offset: u64,
}

impl GgufHeader {
    /// `general.architecture`, if present.
    pub fn architecture(&self) -> Option<String> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
    }

    /// Per-layer merged byte ranges covering the routed-expert tensors
    /// (`blk.{i}.ffn_gate_exps`, `ffn_down_exps`, `ffn_up_exps`) — the
    /// weights a MoE layer reads when it runs.  Tensors are stored
    /// layer-major and the three expert tensors of a layer are adjacent, so
    /// each layer collapses to one contiguous `[begin, end)` range (including
    /// the small inter-tensor alignment gaps).  Offsets are absolute file
    /// offsets (relative to `tensor_data_offset`).
    ///
    /// Returns `None` for layers missing any expert tensor; callers keep the
    /// index alignment to `n_layer`.
    pub fn layer_expert_ranges(&self, n_layer: usize) -> Vec<Option<(usize, usize)>> {
        let mut starts: Vec<u64> = self.tensors.values().map(|t| t.offset).collect();
        starts.sort_unstable();
        // Byte size of the tensor at `offset` = distance to the next tensor
        // in file order (the GGUF data section is one packed run).
        let end_of = |offset: u64| -> u64 {
            match starts.binary_search(&offset) {
                Ok(i) => starts.get(i + 1).copied().unwrap_or(u64::MAX),
                Err(_) => u64::MAX,
            }
        };
        let base = self.tensor_data_offset;
        (0..n_layer)
            .map(|i| {
                let p = format!("blk.{i}");
                let names = [
                    format!("{p}.ffn_gate_exps.weight"),
                    format!("{p}.ffn_down_exps.weight"),
                    format!("{p}.ffn_up_exps.weight"),
                ];
                let mut begin = u64::MAX;
                let mut end = 0u64;
                for n in &names {
                    if let Some(t) = self.tensors.get(n) {
                        begin = begin.min(t.offset);
                        end = end.max(end_of(t.offset));
                    }
                }
                if begin == u64::MAX || end <= begin {
                    None
                } else {
                    Some((
                        base.saturating_add(begin) as usize,
                        base.saturating_add(end) as usize,
                    ))
                }
            })
            .collect()
    }

    /// Tensor dtype ids present in the file that candle cannot represent.
    ///
    /// Useful for explaining *why* a model needs Joshua's own decoders rather
    /// than failing with candle's opaque "unknown dtype" message.
    pub fn unsupported_by_candle(&self) -> Vec<u32> {
        let mut out: Vec<u32> = self
            .tensors
            .values()
            .map(|t| t.dtype)
            .filter(|d| !is_candle_supported(*d))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// `(name, raw dtype id)` pairs for the tensors candle cannot represent,
    /// sorted by name for deterministic error messages.
    ///
    /// These are exactly the tensors [`Self::to_candle_content`] drops, so a
    /// caller that does not consult the raw header (any loader other than
    /// deepseek4) must treat a non-empty result as "this file needs a decoder
    /// that is not attached" rather than silently proceeding.
    pub fn unsupported_tensors(&self) -> Vec<(String, u32)> {
        let mut out: Vec<(String, u32)> = self
            .tensors
            .iter()
            .filter(|(_, t)| !is_candle_supported(t.dtype))
            .map(|(n, t)| (n.clone(), t.dtype))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Build a candle [`gguf_file::Content`] covering only the tensors candle
    /// can represent.
    ///
    /// Tensors whose dtype is outside candle's table (IQ2_XXS, I32, MXFP4,
    /// …) are dropped: the header reader in candle hard-fails on the first
    /// one, and their data is decoded by Joshua's own loaders, which consult
    /// this raw header instead.
    pub fn to_candle_content(&self) -> Result<gguf_file::Content> {
        let magic = match self.version {
            1 => VersionedMagic::GgufV1,
            2 => VersionedMagic::GgufV2,
            3 => VersionedMagic::GgufV3,
            other => {
                return Err(crate::JoshuaError::ModelLoad(format!(
                    "GGUF header: unsupported version {other}"
                )))
            }
        };
        let mut tensor_infos = HashMap::with_capacity(self.tensors.len());
        for (name, info) in &self.tensors {
            let Some(dtype) = ggml_dtype_from_id(info.dtype) else {
                continue;
            };
            tensor_infos.insert(
                name.clone(),
                gguf_file::TensorInfo {
                    ggml_dtype: dtype,
                    shape: info.dims.clone().into(),
                    offset: info.offset,
                },
            );
        }
        Ok(gguf_file::Content {
            magic,
            metadata: self.metadata.clone(),
            tensor_infos,
            tensor_data_offset: self.tensor_data_offset,
        })
    }
}

/// Reserve for a claimed element count without trusting it.
///
/// A header can claim millions of entries in a handful of bytes, so cap the
/// up-front reservation and let the container grow against data that has
/// actually been read.
fn prealloc(claimed: u64) -> usize {
    const MAX_PREALLOC: u64 = 1024;
    claimed.min(MAX_PREALLOC) as usize
}

fn bad(msg: impl std::fmt::Display) -> JoshuaError {
    JoshuaError::ModelLoad(format!("GGUF header: {msg}"))
}

struct Rdr<'a, R: Read + Seek> {
    r: &'a mut R,
    version: u32,
}

impl<R: Read + Seek> Rdr<'_, R> {
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b).map_err(bad)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.r.read_exact(&mut b).map_err(bad)?;
        Ok(u64::from_le_bytes(b))
    }
    /// Lengths are u64 in GGUF v2+ but u32 in the long-obsolete v1.
    fn len(&mut self) -> Result<u64> {
        if self.version == 1 {
            Ok(self.u32()? as u64)
        } else {
            self.u64()
        }
    }
    fn string(&mut self) -> Result<String> {
        let n = self.len()?;
        if n > MAX_STRING_LENGTH {
            return Err(bad(format!(
                "string of {n} bytes exceeds max {MAX_STRING_LENGTH}"
            )));
        }
        // The claimed length is attacker-controlled: a few-byte file could
        // claim a ~1 GiB string and make us allocate it before a single byte
        // is read.  The stream is seekable, so verify the claim against the
        // bytes actually remaining, then allocate only what can be satisfied.
        // (A legitimate file that genuinely carries a huge metadata string
        // still loads — up to the 1 GiB cap, exactly like candle.)
        let pos = self.r.stream_position().map_err(bad)?;
        let end = self.r.seek(SeekFrom::End(0)).map_err(bad)?;
        self.r.seek(SeekFrom::Start(pos)).map_err(bad)?;
        if n > end - pos {
            return Err(bad(format!(
                "string of {n} bytes exceeds the {} bytes remaining in the stream",
                end - pos
            )));
        }
        let mut buf = vec![0u8; n as usize];
        self.r.read_exact(&mut buf).map_err(bad)?;
        // Real GGUFs in the wild NUL-terminate strings despite the spec, and
        // occasionally carry invalid UTF-8.  candle's own `read_string` is
        // deliberately lenient about both (pops trailing NULs, decodes
        // lossily) precisely because of this — and this reader is now on the
        // main load path, so a file the library reader accepts must not be
        // rejected here.  Match its behaviour exactly.
        while let Some(0) = buf.last() {
            buf.pop();
        }
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn value(&mut self, ty: u32, depth: usize) -> Result<Value> {
        if depth > MAX_VALUE_DEPTH {
            return Err(bad(format!(
                "value nesting depth exceeds max {MAX_VALUE_DEPTH}"
            )));
        }
        let mut one = |n: usize| -> Result<Vec<u8>> {
            let mut b = vec![0u8; n];
            self.r.read_exact(&mut b).map_err(bad)?;
            Ok(b)
        };
        Ok(match ty {
            0 => Value::U8(one(1)?[0]),
            1 => Value::I8(one(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(one(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(one(2)?.try_into().unwrap())),
            4 => Value::U32(u32::from_le_bytes(one(4)?.try_into().unwrap())),
            5 => Value::I32(i32::from_le_bytes(one(4)?.try_into().unwrap())),
            6 => Value::F32(f32::from_le_bytes(one(4)?.try_into().unwrap())),
            7 => Value::Bool(one(1)?[0] != 0),
            8 => Value::String(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let n = self.len()?;
                if n > MAX_ARRAY_ELEMENTS {
                    return Err(bad(format!(
                        "array of {n} elements exceeds max {MAX_ARRAY_ELEMENTS}"
                    )));
                }
                let mut items = Vec::with_capacity(prealloc(n));
                for _ in 0..n {
                    // Nested arrays are legal GGUF (and candle accepts them,
                    // to a depth cap); the depth check above bounds the
                    // recursion.
                    items.push(self.value(elem_ty, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(u64::from_le_bytes(one(8)?.try_into().unwrap())),
            11 => Value::I64(i64::from_le_bytes(one(8)?.try_into().unwrap())),
            12 => Value::F64(f64::from_le_bytes(one(8)?.try_into().unwrap())),
            other => return Err(bad(format!("unknown metadata value type {other}"))),
        })
    }
}

/// Parse a GGUF header, preserving dtypes candle cannot represent.
pub fn read_header<R: Read + Seek>(r: &mut R) -> Result<GgufHeader> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(bad)?;
    if u32::from_le_bytes(magic) != MAGIC {
        return Err(bad("not a GGUF file (bad magic)"));
    }
    let mut rd = Rdr { r, version: 0 };
    let version = rd.u32()?;
    if !(1..=3).contains(&version) {
        return Err(bad(format!("unsupported GGUF version {version}")));
    }
    rd.version = version;

    let tensor_count = rd.len()?;
    let kv_count = rd.len()?;
    if tensor_count > MAX_COUNT || kv_count > MAX_COUNT {
        return Err(bad("implausible tensor/metadata count"));
    }

    // Capacity is deliberately not taken from the header: the count is
    // attacker-controlled, and reserving for it would let a few-byte file
    // trigger a gigabyte of allocation before a single entry is read. The
    // maps grow as real entries arrive.
    let mut metadata = HashMap::with_capacity(prealloc(kv_count));
    for _ in 0..kv_count {
        let key = rd.string()?;
        let ty = rd.u32()?;
        metadata.insert(key, rd.value(ty, 0)?);
    }

    let mut tensors = HashMap::with_capacity(prealloc(tensor_count));
    for _ in 0..tensor_count {
        let name = rd.string()?;
        let n_dims = rd.u32()?;
        if n_dims > 8 {
            return Err(bad(format!("tensor `{name}` claims {n_dims} dimensions")));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(rd.len()? as usize);
        }
        // GGUF stores dims fastest-varying first; row-major consumers want the
        // reverse, matching candle's own convention.
        dims.reverse();
        let dtype = rd.u32()?;
        let offset = rd.u64()?;
        tensors.insert(
            name,
            RawTensorInfo {
                dtype,
                dims,
                offset,
            },
        );
    }

    let alignment = match metadata.get("general.alignment") {
        Some(Value::U8(v)) => *v as u64,
        Some(Value::U16(v)) => *v as u64,
        Some(Value::U32(v)) => *v as u64,
        Some(Value::I8(v)) if *v >= 0 => *v as u64,
        Some(Value::I16(v)) if *v >= 0 => *v as u64,
        Some(Value::I32(v)) if *v >= 0 => *v as u64,
        _ => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(bad(format!("invalid general.alignment {alignment}")));
    }
    let pos = rd.r.stream_position().map_err(bad)?;
    let tensor_data_offset = pos.div_ceil(alignment) * alignment;

    Ok(GgufHeader {
        version,
        metadata,
        tensors,
        tensor_data_offset,
    })
}

/// Byte size of a tensor of `elems` elements in ggml type `dtype`.
///
/// Covers the types Joshua can decode, including MXFP4 and IQ2_XXS, which
/// candle cannot describe at all.
pub fn type_size_bytes(dtype: u32, elems: usize) -> Option<usize> {
    // (block size in elements, bytes per block)
    let (blck, bytes) = match dtype {
        0 => (1, 4),   // F32
        1 => (1, 2),   // F16
        30 => (1, 2),  // BF16
        2 => (32, 18), // Q4_0
        3 => (32, 20), // Q4_1
        6 => (32, 22), // Q5_0
        7 => (32, 24), // Q5_1
        8 => (32, 34), // Q8_0
        9 => (32, 36), // Q8_1
        // K-quants: QK_K = 256 elements per block.  Sizes mirror candle's
        // block structs (k_quants.rs `const _: () = assert!(...)`).
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        26 => (1, 4),  // I32 (routed-expert id tables)
        crate::iq2xxs::GGML_TYPE_IQ2_XXS => (crate::iq2xxs::QK_IQ2_XXS, crate::iq2xxs::BLOCK_BYTES),
        crate::mxfp4::GGML_TYPE_MXFP4 => (crate::mxfp4::QK_MXFP4, 17),
        _ => return None,
    };
    if !elems.is_multiple_of(blck) {
        return None;
    }
    Some(elems / blck * bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Hand-build a GGUF v3 header with one tensor of the given dtype.
    fn header_bytes(dtype: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&1u64.to_le_bytes()); // kv count
                                                  // general.architecture = "kimi-k3"
        let k = b"general.architecture";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        b.extend_from_slice(&8u32.to_le_bytes()); // string
        let v = b"kimi-k3";
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v);
        // tensor "w", dims [64, 8]
        let n = b"w";
        b.extend_from_slice(&(n.len() as u64).to_le_bytes());
        b.extend_from_slice(n);
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&64u64.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes());
        b.extend_from_slice(&dtype.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // offset
        b
    }

    #[test]
    fn parses_a_header_whose_dtype_candle_rejects() {
        // The whole point: MXFP4 (39) must survive header parsing.
        let bytes = header_bytes(crate::mxfp4::GGML_TYPE_MXFP4);
        let h = read_header(&mut Cursor::new(&bytes[..])).unwrap();
        assert_eq!(h.architecture().as_deref(), Some("kimi-k3"));
        let t = h.tensors.get("w").unwrap();
        assert_eq!(t.dtype, 39);
        // Dims reversed into row-major order.
        assert_eq!(t.dims, vec![8, 64]);
        assert_eq!(t.elem_count(), 512);
        assert_eq!(h.unsupported_by_candle(), vec![39]);

        // candle, for comparison, cannot get past this header at all.
        let mut c = Cursor::new(&bytes[..]);
        assert!(
            candle_core::quantized::gguf_file::Content::read(&mut c).is_err(),
            "candle is expected to reject MXFP4; if it stops doing so this shim can go"
        );
    }

    /// Two layers, each with the three expert tensors, plus a trailing dense
    /// tensor: layer ranges must merge gate/down/up into one `[begin, end)`
    /// per layer, spanning the inter-tensor alignment gaps but not the next
    /// layer's start or the trailing tensor.
    #[test]
    fn layer_expert_ranges_merge_per_layer() {
        let base = 4096u64; // tensor_data_offset (arbitrary, must be added)
        let mut h = GgufHeader {
            version: 3,
            metadata: Default::default(),
            tensors: Default::default(),
            tensor_data_offset: base,
        };
        let mut off = 0u64;
        for l in 0..2 {
            for name in [
                format!("blk.{l}.ffn_gate_exps.weight"),
                format!("blk.{l}.ffn_down_exps.weight"),
                format!("blk.{l}.ffn_up_exps.weight"),
            ] {
                h.tensors.insert(
                    name,
                    RawTensorInfo {
                        dtype: 10,
                        dims: vec![256, 32], // 8192 elems; sizes come from deltas anyway
                        offset: off,
                    },
                );
                off += 4096; // simulated tensor size + alignment gap
            }
        }
        h.tensors.insert(
            "output.weight".into(),
            RawTensorInfo {
                dtype: 10,
                dims: vec![256, 8],
                offset: off,
            },
        );

        let ranges = h.layer_expert_ranges(2);
        assert_eq!(ranges.len(), 2);
        for (i, r) in ranges.iter().enumerate() {
            let (begin, end) = r.expect("layer must have expert tensors");
            assert_eq!(begin, base as usize + i * 3 * 4096);
            assert_eq!(end, base as usize + (i + 1) * 3 * 4096);
        }
    }

    #[test]
    fn known_dtypes_are_not_flagged_unsupported() {
        let bytes = header_bytes(0); // F32
        let h = read_header(&mut Cursor::new(&bytes[..])).unwrap();
        assert!(h.unsupported_by_candle().is_empty());
    }

    #[test]
    fn removed_q4_2_q4_3_ids_count_as_unsupported() {
        // candle's table maps 0..=3, 6..=15 and 30. Ids 4 and 5 are the
        // withdrawn Q4_2/Q4_3 and hard-fail there, so reporting them as
        // supported would defeat the point of the diagnostic.
        for dtype in [4u32, 5] {
            let bytes = header_bytes(dtype);
            let h = read_header(&mut Cursor::new(&bytes[..])).unwrap();
            assert_eq!(
                h.unsupported_by_candle(),
                vec![dtype],
                "dtype {dtype} is rejected by candle and must be reported"
            );
            let mut c = Cursor::new(&bytes[..]);
            assert!(
                candle_core::quantized::gguf_file::Content::read(&mut c).is_err(),
                "candle is expected to reject dtype {dtype}"
            );
        }
    }

    #[test]
    fn implausible_counts_do_not_drive_allocation() {
        // A tiny header claiming millions of entries must not reserve for them.
        assert_eq!(prealloc(1 << 24), 1024);
        assert_eq!(prealloc(u64::MAX), 1024);
        // Realistic counts are still reserved exactly.
        assert_eq!(prealloc(37), 37);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = header_bytes(0);
        bytes[0] ^= 0xFF;
        assert!(read_header(&mut Cursor::new(&bytes[..])).is_err());
    }

    #[test]
    fn lenient_strings_nul_terminated_and_invalid_utf8() {
        // Real GGUFs NUL-terminate strings despite the spec, and sometimes
        // carry invalid UTF-8; candle's reader tolerates both, so the
        // tolerant header must too (it is on the main load path).
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&2u64.to_le_bytes()); // kv count

        // general.name = "DeepSeek-V4\0" (NUL-terminated despite the spec).
        let k = b"general.name";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        b.extend_from_slice(&8u32.to_le_bytes()); // string
        let v = b"DeepSeek-V4\0";
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v);

        // general.architecture = "kimi-k3\xff" (invalid UTF-8 tail).
        let k = b"general.architecture";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        b.extend_from_slice(&8u32.to_le_bytes()); // string
        let v = b"kimi-k3\xff";
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v);

        // tensor "w", dims [64, 8]
        let n = b"w";
        b.extend_from_slice(&(n.len() as u64).to_le_bytes());
        b.extend_from_slice(n);
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&64u64.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset

        let h = read_header(&mut Cursor::new(&b[..]))
            .expect("NUL-terminated and invalid-UTF-8 strings must not fail the load");
        assert_eq!(
            h.metadata
                .get("general.name")
                .and_then(|v| v.to_string().ok())
                .map(String::as_str),
            Some("DeepSeek-V4")
        );
        assert_eq!(
            h.architecture().as_deref(),
            Some("kimi-k3\u{FFFD}"),
            "invalid UTF-8 decodes lossily, like candle"
        );
        assert_eq!(h.tensors.get("w").unwrap().dtype, 0);
    }

    #[test]
    fn string_claim_beyond_cap_is_rejected_without_allocating() {
        // A hostile header must not be able to force a large allocation by
        // claiming a huge metadata string: the per-string cap (16 MiB) rejects
        // the claim before any buffer is allocated.  This matters for real
        // model files too — there the "bytes remaining" check trivially
        // passes, so without the cap a claim just under 1 GiB would be
        // eagerly allocated and then fail as a desynced parse.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&1u64.to_le_bytes()); // kv count
        let k = b"general.name";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        b.extend_from_slice(&8u32.to_le_bytes()); // value type: string
        b.extend_from_slice(&(32u64 << 20).to_le_bytes()); // claimed 32 MiB
        let n = b"w";
        b.extend_from_slice(&(n.len() as u64).to_le_bytes());
        b.extend_from_slice(n);
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&64u64.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset

        let msg = read_header(&mut Cursor::new(&b[..]))
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("exceeds max"),
            "oversized string claim must be rejected by the cap, got: {msg}"
        );
    }

    #[test]
    fn string_claim_beyond_stream_is_rejected_without_allocating() {
        // A claim within the cap but beyond the bytes actually remaining in
        // the stream must also be rejected without allocating — a tiny file
        // must not force an 8 MiB (or any) allocation.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&1u64.to_le_bytes()); // kv count
        let k = b"general.name";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        b.extend_from_slice(&8u32.to_le_bytes()); // value type: string
        b.extend_from_slice(&(8u64 << 20).to_le_bytes()); // claimed 8 MiB
        let n = b"w";
        b.extend_from_slice(&(n.len() as u64).to_le_bytes());
        b.extend_from_slice(n);
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&64u64.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset

        let msg = read_header(&mut Cursor::new(&b[..]))
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("bytes remaining"),
            "oversized string claim must be rejected, got: {msg}"
        );
    }

    #[test]
    fn array_element_count_beyond_cap_is_rejected() {
        // Same class as the string cap: a hostile array element-count claim
        // (e.g. `tokenizer.ggml.tokens` with a billion elements) must be
        // rejected up front, not looped over — in a real file the loop would
        // read tensor data as elements and buffer gigabytes.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&1u64.to_le_bytes()); // kv count
        let k = b"tokenizer.ggml.tokens";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        b.extend_from_slice(&9u32.to_le_bytes()); // value type: array
        b.extend_from_slice(&8u32.to_le_bytes()); // element type: string
        b.extend_from_slice(&(32u64 << 20).to_le_bytes()); // claimed 32 MiB elements
        let n = b"w";
        b.extend_from_slice(&(n.len() as u64).to_le_bytes());
        b.extend_from_slice(n);
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&64u64.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset

        let msg = read_header(&mut Cursor::new(&b[..]))
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("exceeds max"),
            "oversized array claim must be rejected by the cap, got: {msg}"
        );
    }

    #[test]
    fn nested_arrays_parse_like_candle() {
        // GGUF permits arrays of arrays; candle accepts them (to a depth
        // cap), so the tolerant reader must too — a file candle loads must
        // not be rejected here.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&1u64.to_le_bytes()); // kv count
        let k = b"general.special";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        // Value::Array([Array([U32(1), U32(2)]), Array([U32(3)])]), serialized
        // exactly as candle's writer would.
        b.extend_from_slice(&9u32.to_le_bytes()); // value type: array
        b.extend_from_slice(&9u32.to_le_bytes()); // element type: array
        b.extend_from_slice(&2u64.to_le_bytes()); // 2 outer elements
        b.extend_from_slice(&4u32.to_le_bytes()); //   inner element type: u32
        b.extend_from_slice(&2u64.to_le_bytes()); //   2 inner elements
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&4u32.to_le_bytes()); //   inner element type: u32
        b.extend_from_slice(&1u64.to_le_bytes()); //   1 inner element
        b.extend_from_slice(&3u32.to_le_bytes());
        let n = b"w";
        b.extend_from_slice(&(n.len() as u64).to_le_bytes());
        b.extend_from_slice(n);
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&64u64.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset

        let h = read_header(&mut Cursor::new(&b[..])).expect("nested arrays must parse");
        match h.metadata.get("general.special") {
            Some(Value::Array(outer)) => {
                assert_eq!(outer.len(), 2);
                match &outer[0] {
                    Value::Array(inner) => {
                        assert_eq!(inner.len(), 2);
                        match (&inner[0], &inner[1]) {
                            (Value::U32(a), Value::U32(c)) => {
                                assert_eq!((*a, *c), (1, 2));
                            }
                            other => panic!("expected u32 elements, got {other:?}"),
                        }
                    }
                    other => panic!("expected inner array, got {other:?}"),
                }
            }
            other => panic!("expected array, got {other:?}"),
        }

        // candle accepts the same bytes — this is a load-compatible file.
        let mut c = Cursor::new(&b[..]);
        candle_core::quantized::gguf_file::Content::read(&mut c)
            .expect("candle must accept nested arrays");
    }

    #[test]
    fn value_nesting_depth_is_capped() {
        // A crafted array-of-arrays chain must not blow the stack: cap the
        // nesting at candle's depth limit (64), exactly like candle.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&1u64.to_le_bytes()); // kv count
        let k = b"general.deep";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        b.extend_from_slice(&9u32.to_le_bytes()); // metadata value type: array
                                                  // 66 nested arrays: elem type
                                                  // array, count 1, then the
                                                  // innermost u32.
        for _ in 0..66 {
            b.extend_from_slice(&9u32.to_le_bytes()); // element type: array
            b.extend_from_slice(&1u64.to_le_bytes()); // count 1
        }
        b.extend_from_slice(&0u32.to_le_bytes()); // element type: u32
        b.extend_from_slice(&1u64.to_le_bytes()); // count 1
        b.extend_from_slice(&42u32.to_le_bytes()); // the value
        let n = b"w";
        b.extend_from_slice(&(n.len() as u64).to_le_bytes());
        b.extend_from_slice(n);
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&64u64.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset

        let msg = read_header(&mut Cursor::new(&b[..]))
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("nesting depth"),
            "deep nesting must be rejected with a depth error, got: {msg}"
        );
    }

    #[test]
    fn mxfp4_size_is_seventeen_bytes_per_thirty_two_elements() {
        assert_eq!(type_size_bytes(crate::mxfp4::GGML_TYPE_MXFP4, 64), Some(34));
        // Not a whole number of blocks.
        assert_eq!(type_size_bytes(crate::mxfp4::GGML_TYPE_MXFP4, 40), None);
        assert_eq!(type_size_bytes(0, 10), Some(40));
    }

    #[test]
    fn k_quant_sizes_match_candle_block_layouts() {
        // QK_K = 256 elements per block; bytes mirror candle's BlockQ*K
        // structs (Q2K: 16+64+4, Q3K: 32+64+12+2, Q4K: 128+12+4,
        // Q5K: 32+128+12+4, Q6K: 192+16+2, Q8K: 4+256+32).
        for (dtype, bytes_per_block) in [(10, 84), (11, 110), (12, 144), (13, 176), (14, 210), (15, 292)] {
            assert_eq!(
                type_size_bytes(dtype, 256),
                Some(bytes_per_block),
                "dtype {dtype}: one block"
            );
            assert_eq!(
                type_size_bytes(dtype, 512),
                Some(2 * bytes_per_block),
                "dtype {dtype}: two blocks"
            );
            // Not a whole number of blocks.
            assert_eq!(type_size_bytes(dtype, 128), None, "dtype {dtype}: half block");
        }
    }
}
