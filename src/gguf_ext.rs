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
//! `u32`, so unknown types survive to be handled by [`crate::mxfp4`] (or
//! reported precisely).  Metadata is decoded into candle's own
//! [`gguf_file::Value`] so existing hyper-parameter code keeps working
//! unchanged.
//!
//! Only the header is read.  Tensor *data* is never touched here — callers
//! borrow it from the memory mapping via [`crate::mmap_tensor`], which is what
//! keeps a model far larger than RAM loadable.

use std::collections::HashMap;
use std::io::{Read, Seek};

use candle_core::quantized::gguf_file::Value;

use crate::{JoshuaError, Result};

const MAGIC: u32 = 0x4655_4747; // "GGUF", little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

/// Guards against a corrupt or hostile header claiming an absurd count and
/// making us attempt a huge allocation before any real data is read.
const MAX_COUNT: u64 = 1 << 24;

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
#[derive(Debug)]
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

    /// Tensor dtype ids present in the file that candle cannot represent.
    ///
    /// Useful for explaining *why* a model needs Joshua's own decoders rather
    /// than failing with candle's opaque "unknown dtype" message.
    pub fn unsupported_by_candle(&self) -> Vec<u32> {
        let mut out: Vec<u32> = self
            .tensors
            .values()
            .map(|t| t.dtype)
            .filter(|d| !matches!(d, 0..=15 | 30))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
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
        if n > MAX_COUNT {
            return Err(bad(format!("string of {n} bytes is implausible")));
        }
        let mut buf = vec![0u8; n as usize];
        self.r.read_exact(&mut buf).map_err(bad)?;
        String::from_utf8(buf).map_err(|e| bad(format!("non-UTF-8 string: {e}")))
    }

    fn value(&mut self, ty: u32) -> Result<Value> {
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
                if elem_ty == 9 {
                    return Err(bad("nested arrays are not permitted"));
                }
                let n = self.len()?;
                if n > MAX_COUNT {
                    return Err(bad(format!("array of {n} elements is implausible")));
                }
                let mut items = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    items.push(self.value(elem_ty)?);
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

    let mut metadata = HashMap::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        let key = rd.string()?;
        let ty = rd.u32()?;
        metadata.insert(key, rd.value(ty)?);
    }

    let mut tensors = HashMap::with_capacity(tensor_count as usize);
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
        tensors.insert(name, RawTensorInfo { dtype, dims, offset });
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
/// Covers the types Joshua can decode, including MXFP4, which candle cannot
/// describe at all.
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

    #[test]
    fn known_dtypes_are_not_flagged_unsupported() {
        let bytes = header_bytes(0); // F32
        let h = read_header(&mut Cursor::new(&bytes[..])).unwrap();
        assert!(h.unsupported_by_candle().is_empty());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = header_bytes(0);
        bytes[0] ^= 0xFF;
        assert!(read_header(&mut Cursor::new(&bytes[..])).is_err());
    }

    #[test]
    fn mxfp4_size_is_seventeen_bytes_per_thirty_two_elements() {
        assert_eq!(
            type_size_bytes(crate::mxfp4::GGML_TYPE_MXFP4, 64),
            Some(34)
        );
        // Not a whole number of blocks.
        assert_eq!(type_size_bytes(crate::mxfp4::GGML_TYPE_MXFP4, 40), None);
        assert_eq!(type_size_bytes(0, 10), Some(40));
    }
}
