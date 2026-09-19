//! GLSL compute kernels for the Vulkan backend, generated as source strings
//! and compiled to SPIR-V at runtime with `naga`.
//!
//! The kernel set mirrors the OpenCL backend's `kernels.cl` op for op:
//! strided / broadcast elementwise ops, strided copies, row reductions,
//! gathers, comparisons, casts, a tiled GEMM plus GEMVs, the fused
//! softmax / RMSNorm / RoPE the attention path calls, and the quantized
//! dequantize-in-kernel GEMV / GEMM over GGUF block formats.
//!
//! Conventions
//! * Every kernel is one complete compute shader (`main`).  Its tensor
//!   arguments are `std430` storage buffers bound in order from binding 0;
//!   scalar arguments are push constants (at most 128 bytes, the guaranteed
//!   minimum); the index descriptors ([`Idx`]) are the last binding,
//!   a small parameter buffer written per launch.
//! * Element indices are `int`; tensors stay below 2^31 elements.
//! * Buffers are typed by element width: `float`/`uint`/`int` for 4-byte
//!   elements, `uvec2` for 8-byte ones, and packed `uint` words for 1- and
//!   2-byte elements.  naga's GLSL front end has no atomics and 8/16-bit
//!   storage is optional, so kernels that *write* packed elements own a
//!   whole word per invocation (read-modify-write of the word is race-free).
//! * `WG` (the work-group size of the row / reduction kernels) is chosen
//!   from the device limit at pipeline build time; the elementwise kernels
//!   use it too and spread large launches over a 2-D grid (`gid()`).
//! * The generic index descriptor `Idx` carries the output shape and up to
//!   three input stride sets so one kernel serves every broadcast /
//!   transposed / narrowed view; outputs are written contiguous row-major.
//!
//! [`Idx`]: super::kernels::Idx

/// Number of dims the index descriptor carries (must match `kernels::MAXD`).
pub const MAXD: usize = 8;

/// The common prelude: version, the `Idx` descriptor, the parameter buffer
/// (always binding `{P}`), and the helpers.
fn prelude(wg: usize, nbuf: usize) -> String {
    format!(
        r#"#version 450
#define WG {wg}
#define MAXD {maxd}
struct Idx {{ int nd; int dims[MAXD]; int s0[MAXD]; int s1[MAXD]; int s2[MAXD]; int o0; int o1; int o2; }};
layout(std430, set = 0, binding = {p}) readonly buffer ParamsB {{ Idx ix; Idx rd; }} p;
float INF() {{ return uintBitsToFloat(0x7F800000u); }}
float NINF() {{ return uintBitsToFloat(0xFF800000u); }}
// Linear index of this invocation over a 2-D grid of 1-D work-groups.
int gid() {{ return int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * uint(WG))); }}
// Row index for one-work-group-per-row kernels (2-D grid of groups).
int grow() {{ return int(gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x); }}
{off0}
{off1}
{off2}
{offr}
"#,
        p = nbuf,
        maxd = MAXD,
        off0 = off_fn("off0", "ix", "s0", "o0"),
        off1 = off_fn("off1", "ix", "s1", "o1"),
        off2 = off_fn("off2", "ix", "s2", "o2"),
        offr = off_fn("offr", "rd", "s0", "0"),
    )
}

/// `int NAME(int lin)`: storage offset of linear output index `lin` under
/// stride set `S` / offset `O` of descriptor `D`.
fn off_fn(name: &str, d: &str, s: &str, o: &str) -> String {
    let o = if o == "0" { "0".to_string() } else { format!("p.{d}.{o}") };
    format!(
        "int {name}(int lin) {{ int off = {o}; for (int d = p.{d}.nd - 1; d >= 0; d--) {{ int dim = p.{d}.dims[d]; int i = lin % dim; lin /= dim; off += i * p.{d}.{s}[d]; }} return off; }}"
    )
}

fn buf(binding: usize, ty: &str, name: &str, ro: bool) -> String {
    let q = if ro { "readonly " } else { "" };
    format!("layout(std430, set = 0, binding = {binding}) {q}buffer B{binding} {{ {ty} {name}[]; }};\n")
}

/// Unary float helpers shared by the elementwise kernels.
const UNARY_FN: &str = r#"
float erf_f(float x) {
    // Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
    float s = x < 0.0 ? -1.0 : 1.0;
    float a = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * a);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}
float round_half_away(float v) { return sign(v) * floor(abs(v) + 0.5); }
float unary_f(float v, int op, float f0, float f1) {
    switch (op) {
        case 0: return exp(v);
        case 1: return log(v);
        case 2: return sin(v);
        case 3: return cos(v);
        case 4: return tanh(v);
        case 5: return -v;
        case 6: return 1.0 / v;
        case 7: return v * v;
        case 8: return sqrt(v);
        case 9: return 0.5 * v * (1.0 + tanh(0.79788456080286535588 * v * (1.0 + 0.044715 * v * v)));
        case 10: return 0.5 * v * (1.0 + erf_f(v * 0.70710678118654752440));
        case 11: return erf_f(v);
        case 12: return v / (1.0 + exp(-v));
        case 13: return abs(v);
        case 14: return ceil(v);
        case 15: return floor(v);
        case 16: return round_half_away(v);
        case 17: return v > 0.0 ? v : 0.0;
        case 18: return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0);
        case 19: return 1.0 / (1.0 + exp(-v));
        case 20: return v * f0 + f1;           // affine
        case 21: return pow(v, f0);            // powf
        case 22: return v > 0.0 ? v : f0 * (exp(v) - 1.0); // elu
        default: return v;
    }
}
"#;

/// Elementwise float unary op (also affine / powf / elu via `op`):
/// `o[i] = f(x[idx(i)])`.  Push: `n, contig, off, op, f0, f1`.
pub fn k_unary(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "float", "o", false);
    s += UNARY_FN;
    s += r#"
layout(push_constant) uniform PC { int n; int contig; int off; int op; float f0; float f1; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n) return;
    int src = pc.contig != 0 ? pc.off + i : off0(i);
    o[i] = unary_f(x[src], pc.op, pc.f0, pc.f1);
}
"#;
    s
}

const BINARY_FN: &str = r#"
float binary_f(float a, float b, int op) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return a > b ? b : a;
        case 5: return a < b ? b : a;
        default: return a;
    }
}
uint binary_u(uint x, uint y, int op) {
    switch (op) {
        case 0: return x + y;
        case 1: return x - y;
        case 2: return x * y;
        case 3: return y == 0u ? 0u : x / y;
        case 4: return x > y ? y : x;
        case 5: return x < y ? y : x;
        default: return x;
    }
}
"#;

/// Binary op over f32 (`is_u32 == 0`) or u32 buffers, strided or contiguous.
/// Push: `n, contig, oa, ob, op`.
pub fn k_binary(wg: usize, u32: bool) -> String {
    let ty = if u32 { "uint" } else { "float" };
    let mut s = prelude(wg, 3);
    s += &buf(0, ty, "a", true);
    s += &buf(1, ty, "b", true);
    s += &buf(2, ty, "o", false);
    s += BINARY_FN;
    s += "layout(push_constant) uniform PC { int n; int contig; int oa; int ob; int op; } pc;\nlayout(local_size_x = WG) in;\nvoid main() {\n    int i = gid();\n    if (i >= pc.n) return;\n    int ia = pc.contig != 0 ? pc.oa + i : off0(i);\n    int ib = pc.contig != 0 ? pc.ob + i : off1(i);\n";
    s += if u32 { "    o[i] = binary_u(a[ia], b[ib], pc.op);\n" } else { "    o[i] = binary_f(a[ia], b[ib], pc.op);\n" };
    s += "}\n";
    s
}

/// Comparison to a packed-u8 output: one invocation per output word
/// (4 elements).  Push: `n, op`.
pub fn k_cmp(wg: usize, u32: bool) -> String {
    let ty = if u32 { "uint" } else { "float" };
    let mut s = prelude(wg, 3);
    s += &buf(0, ty, "a", true);
    s += &buf(1, ty, "b", true);
    s += &buf(2, "uint", "o", false);
    s += &format!(
        r#"
bool cmp_v({ty} x, {ty} y, int op) {{
    switch (op) {{
        case 0: return x == y;
        case 1: return x != y;
        case 2: return x <= y;
        case 3: return x >= y;
        case 4: return x < y;
        default: return x > y;
    }}
}}
layout(push_constant) uniform PC {{ int n; int op; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int w = gid();
    int nw = (pc.n + 3) / 4;
    if (w >= nw) return;
    uint word = 0u;
    for (int k = 0; k < 4; k++) {{
        int i = w * 4 + k;
        if (i < pc.n) {{
            bool r = cmp_v(a[off0(i)], b[off1(i)], pc.op);
            if (r) word |= 1u << (uint(k) * 8u);
        }}
    }}
    o[w] = word;
}}
"#
    );
    s
}

/// `where_cond`: `o[i] = c[i] ? t[i] : f[i]` with a packed-u8 or u32
/// condition over 4-byte (`uint`) or 8-byte (`uvec2`) payloads.
pub fn k_where(wg: usize, cond_u32: bool, elem8: bool) -> String {
    let ty = if elem8 { "uvec2" } else { "uint" };
    let mut s = prelude(wg, 4);
    s += &buf(0, "uint", "c", true);
    s += &buf(1, ty, "t", true);
    s += &buf(2, ty, "f", true);
    s += &buf(3, ty, "o", false);
    let cond = if cond_u32 {
        "uint cv = c[off0(i)];"
    } else {
        "int ci = off0(i); uint cv = (c[ci >> 2] >> (uint(ci & 3) * 8u)) & 0xFFu;"
    };
    s += &format!(
        "layout(push_constant) uniform PC {{ int n; }} pc;\nlayout(local_size_x = WG) in;\nvoid main() {{\n    int i = gid();\n    if (i >= pc.n) return;\n    {cond}\n    o[i] = cv != 0u ? t[off1(i)] : f[off2(i)];\n}}\n"
    );
    s
}

/// Strided copy of 4- or 8-byte elements to a contiguous destination range.
/// Push: `n, dst_off`.
pub fn k_copy_wide(wg: usize, elem8: bool) -> String {
    let ty = if elem8 { "uvec2" } else { "uint" };
    let mut s = prelude(wg, 2);
    s += &buf(0, ty, "x", true);
    s += &buf(1, ty, "o", false);
    s += "layout(push_constant) uniform PC { int n; int dst_off; } pc;\nlayout(local_size_x = WG) in;\nvoid main() {\n    int i = gid();\n    if (i >= pc.n) return;\n    o[pc.dst_off + i] = x[off0(i)];\n}\n";
    s
}

/// Strided copy of 1- or 2-byte elements into a contiguous destination
/// range `[dst_off, dst_off + n)`: one invocation per destination word,
/// keeping the bytes outside the range.  Push: `n, dst_off, w0`
/// (`w0` = first word index).
pub fn k_copy_packed(wg: usize, elem: usize) -> String {
    let (per, bits, mask) = if elem == 1 { (4, 8u32, "0xFFu") } else { (2, 16, "0xFFFFu") };
    let mut s = prelude(wg, 2);
    s += &buf(0, "uint", "x", true);
    s += &buf(1, "uint", "o", false);
    s += &format!(
        r#"
uint src_elem(int e) {{ return (x[e / {per}] >> (uint(e % {per}) * {bits}u)) & {mask}; }}
layout(push_constant) uniform PC {{ int n; int dst_off; int w0; int nw; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int wi = gid();
    if (wi >= pc.nw) return;
    int w = pc.w0 + wi;
    uint word = o[w];
    for (int k = 0; k < {per}; k++) {{
        int e = w * {per} + k - pc.dst_off;
        if (e >= 0 && e < pc.n) {{
            uint v = src_elem(off0(e));
            uint sh = uint(k) * {bits}u;
            word = (word & ~({mask} << sh)) | (v << sh);
        }}
    }}
    o[w] = word;
}}
"#
    );
    s
}

/// copy2d over 4- or 8-byte elements: `d1` rows of `d2` contiguous elements.
/// Push: `d1, d2, src_s, dst_s, src_o, dst_o`.
pub fn k_copy2d_wide(wg: usize, elem8: bool) -> String {
    let ty = if elem8 { "uvec2" } else { "uint" };
    let mut s = prelude(wg, 2);
    s += &buf(0, ty, "x", true);
    s += &buf(1, ty, "o", false);
    s += "layout(push_constant) uniform PC { int d1; int d2; int src_s; int dst_s; int src_o; int dst_o; } pc;\nlayout(local_size_x = WG) in;\nvoid main() {\n    int idx = gid();\n    if (idx >= pc.d1 * pc.d2) return;\n    int i = idx / pc.d2;\n    int j = idx % pc.d2;\n    o[pc.dst_o + i * pc.dst_s + j] = x[pc.src_o + i * pc.src_s + j];\n}\n";
    s
}

/// copy2d over 1- or 2-byte elements: one invocation per destination word
/// in the bounding range; bytes outside the copied rows are kept.
/// Push: `d1, d2, src_s, dst_s, src_o, dst_o, w0, nw`.
pub fn k_copy2d_packed(wg: usize, elem: usize) -> String {
    let (per, bits, mask) = if elem == 1 { (4, 8u32, "0xFFu") } else { (2, 16, "0xFFFFu") };
    let mut s = prelude(wg, 2);
    s += &buf(0, "uint", "x", true);
    s += &buf(1, "uint", "o", false);
    s += &format!(
        r#"
uint src_elem(int e) {{ return (x[e / {per}] >> (uint(e % {per}) * {bits}u)) & {mask}; }}
layout(push_constant) uniform PC {{ int d1; int d2; int src_s; int dst_s; int src_o; int dst_o; int w0; int nw; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int wi = gid();
    if (wi >= pc.nw) return;
    int w = pc.w0 + wi;
    uint word = o[w];
    for (int k = 0; k < {per}; k++) {{
        int e = w * {per} + k - pc.dst_o;
        if (e < 0) continue;
        int i = e / pc.dst_s;
        int j = e - i * pc.dst_s;
        if (pc.dst_s == 0) {{ i = 0; j = e; }}
        if (i < pc.d1 && j < pc.d2) {{
            uint v = src_elem(pc.src_o + i * pc.src_s + j);
            uint sh = uint(k) * {bits}u;
            word = (word & ~({mask} << sh)) | (v << sh);
        }}
    }}
    o[w] = word;
}}
"#
    );
    s
}

/// Fill the strided elements of a 4- or 8-byte buffer with a constant.
/// Push: `n, v_lo, v_hi`.
pub fn k_fill_wide(wg: usize, elem8: bool) -> String {
    let ty = if elem8 { "uvec2" } else { "uint" };
    let mut s = prelude(wg, 1);
    s += &buf(0, ty, "o", false);
    s += "layout(push_constant) uniform PC { int n; uint lo; uint hi; } pc;\nlayout(local_size_x = WG) in;\nvoid main() {\n    int i = gid();\n    if (i >= pc.n) return;\n";
    s += if elem8 { "    o[off0(i)] = uvec2(pc.lo, pc.hi);\n" } else { "    o[off0(i)] = pc.lo;\n" };
    s += "}\n";
    s
}

/// Fill a contiguous range `[off, off + n)` of 1- or 2-byte elements.
/// Push: `n, off, w0, nw, v`.
pub fn k_fill_packed(wg: usize, elem: usize) -> String {
    let (per, bits, mask) = if elem == 1 { (4, 8u32, "0xFFu") } else { (2, 16, "0xFFFFu") };
    let mut s = prelude(wg, 1);
    s += &buf(0, "uint", "o", false);
    s += &format!(
        r#"
layout(push_constant) uniform PC {{ int n; int off; int w0; int nw; uint v; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int wi = gid();
    if (wi >= pc.nw) return;
    int w = pc.w0 + wi;
    uint word = o[w];
    for (int k = 0; k < {per}; k++) {{
        int e = w * {per} + k - pc.off;
        if (e >= 0 && e < pc.n) {{
            uint sh = uint(k) * {bits}u;
            word = (word & ~({mask} << sh)) | ((pc.v & {mask}) << sh);
        }}
    }}
    o[w] = word;
}}
"#
    );
    s
}

/// Storage classes the cast kernel converts between.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cls {
    F32,
    U32,
    I32,
    U8,
    F16,
    BF16,
    I64,
    F64,
}

impl Cls {
    fn packed(self) -> Option<(usize, u32, &'static str)> {
        match self {
            Cls::U8 => Some((4, 8, "0xFFu")),
            Cls::F16 | Cls::BF16 => Some((2, 16, "0xFFFFu")),
            _ => None,
        }
    }
    fn buf_ty(self) -> &'static str {
        match self {
            Cls::F32 => "float",
            Cls::U32 | Cls::U8 | Cls::F16 | Cls::BF16 => "uint",
            Cls::I32 => "int",
            Cls::I64 | Cls::F64 => "uvec2",
        }
    }
    /// GLSL expression reading element at storage offset `e` as a `double`
    /// -free intermediate: floats become `float`, ints become `int`/`uint`.
    fn read(self) -> String {
        match self {
            Cls::F32 => "x[e]".into(),
            Cls::U32 => "x[e]".into(),
            Cls::I32 => "x[e]".into(),
            Cls::U8 => "((x[e / 4] >> (uint(e % 4) * 8u)) & 0xFFu)".into(),
            Cls::F16 => "unpackHalf2x16((x[e / 2] >> (uint(e % 2) * 16u)) & 0xFFFFu).x".into(),
            Cls::BF16 => "uintBitsToFloat(((x[e / 2] >> (uint(e % 2) * 16u)) & 0xFFFFu) << 16u)".into(),
            Cls::I64 => "x[e]".into(),
            Cls::F64 => "x[e]".into(),
        }
    }
}

/// i64 ids to u32 for the indexing kernels: a negative id or one that does
/// not fit in 32 bits becomes `0xFFFFFFFF`, which every bounds check
/// (against a dimension below 2³¹) rejects and reports through the fault
/// word.  Push: `n`.
pub fn k_ids_i64(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "uvec2", "x", true);
    s += &buf(1, "uint", "o", false);
    s += r#"
layout(push_constant) uniform PC { int n; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n) return;
    uvec2 r = x[off0(i)];
    o[i] = r.y != 0u ? 0xFFFFFFFFu : r.x;
}
"#;
    s
}

/// `to_dtype` between two storage classes: strided input, contiguous
/// output.  Values go through `float` (for floats) or `int`/`uint` (ints);
/// an i64 becomes a float from both of its words (f32 precision), and the
/// integer casts of an i64 keep its low word (the values candle's tensors
/// hold here: indices, masks).
/// Push: `n` (+ `nw` for packed outputs).
pub fn k_cast(wg: usize, from: Cls, to: Cls) -> Option<String> {
    use Cls::*;
    // Intermediate value `v` as float (fv) or int (iv), depending on `from`.
    let (float_in, load) = match from {
        F32 | F16 | BF16 => (true, format!("float v = {};", from.read())),
        U32 | U8 => (true, format!("float v = float({});", from.read())),
        I32 => (true, format!("float v = float({});", from.read())),
        I64 => (true, String::new()),
        F64 => return None,
    };
    let _ = float_in;
    // Integer-preserving loads for int-to-int casts (no float rounding).
    let iload = match from {
        U32 => "uint iv = x[e];".to_string(),
        U8 => format!("uint iv = {};", from.read()),
        I32 => "int iv = x[e];".to_string(),
        I64 => "uvec2 r = x[e]; uint iv = r.x;".to_string(),
        _ => String::new(),
    };
    let int_in = !iload.is_empty();
    let mut s = prelude(wg, 2);
    s += &buf(0, from.buf_ty(), "x", true);
    s += &buf(1, to.buf_ty(), "o", false);
    // The output element expression from `v` (float) or `iv` (int).
    let (value, store_ty): (String, &str) = match to {
        F32 => ("v".into(), "float"),
        U32 => (if int_in { "uint(iv)".into() } else { "(v <= 0.0 ? 0u : uint(v))".into() }, "uint"),
        I32 => (if int_in { "int(iv)".into() } else { "int(v)".into() }, "int"),
        I64 => (
            if int_in {
                if from == I32 { "uvec2(uint(iv), iv < 0 ? 0xFFFFFFFFu : 0u)".into() } else { "uvec2(uint(iv), 0u)".into() }
            } else {
                "uvec2(uint(int(v)), int(v) < 0 ? 0xFFFFFFFFu : 0u)".into()
            },
            "uvec2",
        ),
        U8 => (if int_in { "(uint(iv) & 0xFFu)".into() } else { "(v <= 0.0 ? 0u : (v >= 255.0 ? 255u : uint(v)))".into() }, "packed"),
        F16 => ("(packHalf2x16(vec2(v, 0.0)) & 0xFFFFu)".into(), "packed"),
        // Round to nearest even, as candle's f32 → bf16 does.
        BF16 => ("((floatBitsToUint(v) + 0x7FFFu + ((floatBitsToUint(v) >> 16u) & 1u)) >> 16u)".into(), "packed"),
        F64 => return None,
    };
    // Integer inputs also expose `v` for the float-valued targets (an i64
    // from both of its words).
    let load_stmt = match from {
        // Exact when the value fits in 32 bits, f32 precision beyond.
        I64 => format!(
            "{iload} float v = (r.y == (int(r.x) < 0 ? 0xFFFFFFFFu : 0u)) ? float(int(r.x)) : float(int(r.y)) * 4294967296.0 + float(r.x);"
        ),
        _ if int_in => format!("{iload} float v = float(iv);"),
        _ => load,
    };
    if store_ty == "packed" {
        let (per, bits, mask) = to.packed().unwrap();
        s += &format!(
            r#"
layout(push_constant) uniform PC {{ int n; int nw; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int w = gid();
    if (w >= pc.nw) return;
    uint word = 0u;
    for (int k = 0; k < {per}; k++) {{
        int i = w * {per} + k;
        if (i < pc.n) {{
            int e = off0(i);
            {load_stmt}
            uint pv = {value};
            word |= (pv & {mask}) << (uint(k) * {bits}u);
        }}
    }}
    o[w] = word;
}}
"#
        );
    } else {
        s += &format!(
            "layout(push_constant) uniform PC {{ int n; int nw; }} pc;\nlayout(local_size_x = WG) in;\nvoid main() {{\n    int i = gid();\n    if (i >= pc.n) return;\n    int e = off0(i);\n    {load_stmt}\n    o[i] = {value};\n}}\n"
        );
    }
    Some(s)
}

const RED_FN: &str = r#"
float red_f(float a, float b, int op) {
    if (op == 0) return a + b;
    if (op == 1) return a > b ? a : b;
    return a < b ? a : b;
}
float red_init(int op) { return op == 0 ? 0.0 : (op == 1 ? NINF() : INF()); }
"#;

/// Row reduction over the contiguous last dim: one work-group per row.
/// Push: `rows, cols, off, op`.
pub fn k_reduce_last(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "float", "o", false);
    s += RED_FN;
    s += r#"
layout(push_constant) uniform PC { int rows; int cols; int off; int op; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];
void main() {
    int row = grow();
    int t = int(gl_LocalInvocationID.x);
    if (row >= pc.rows) return;
    int base = pc.off + row * pc.cols;
    float acc = red_init(pc.op);
    for (int c = t; c < pc.cols; c += WG) acc = red_f(acc, x[base + c], pc.op);
    sh[t] = acc;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] = red_f(sh[t], sh[t + s], pc.op);
        barrier();
    }
    if (t == 0) o[row] = sh[0];
}
"#;
    s
}

/// Generic reduction: one invocation per output element; `ix` maps output
/// indices to the input base (reduced dims: stride 0, dim 1), `rd`
/// enumerates the reduced sub-space.  Push: `n, count, op`.
pub fn k_reduce_generic(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "float", "o", false);
    s += RED_FN;
    s += r#"
layout(push_constant) uniform PC { int n; int count; int op; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n) return;
    int base = off0(i);
    float acc = red_init(pc.op);
    for (int j = 0; j < pc.count; j++) acc = red_f(acc, x[base + offr(j)], pc.op);
    o[i] = acc;
}
"#;
    s
}

/// ArgMax / ArgMin over the contiguous last dim (u32 output).
/// Push: `rows, cols, off, is_max`.
pub fn k_arg_last(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "uint", "o", false);
    s += r#"
layout(push_constant) uniform PC { int rows; int cols; int off; int is_max; } pc;
layout(local_size_x = WG) in;
shared float shv[WG];
shared uint shi[WG];
void main() {
    int row = grow();
    int t = int(gl_LocalInvocationID.x);
    if (row >= pc.rows) return;
    int base = pc.off + row * pc.cols;
    bool mx = pc.is_max != 0;
    float best = mx ? NINF() : INF();
    uint bi = 0u;
    for (int c = t; c < pc.cols; c += WG) {
        float v = x[base + c];
        bool better = mx ? (v > best) : (v < best);
        if (better) { best = v; bi = uint(c); }
    }
    shv[t] = best; shi[t] = bi;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) {
            float a = shv[t]; float b = shv[t + s];
            bool take = mx ? (b > a || (b == a && shi[t + s] < shi[t])) : (b < a || (b == a && shi[t + s] < shi[t]));
            if (take) { shv[t] = b; shi[t] = shi[t + s]; }
        }
        barrier();
    }
    if (t == 0) o[row] = shi[0];
}
"#;
    s
}

// Out-of-range ids: the CPU backend returns an error, which a kernel
// cannot.  Every indexing kernel binds the device's `fault` buffer last,
// takes the calling thread's slot in it as its last push constant, and
// sets the slot (skipping the element) when an id exceeds the indexed
// dimension; the host checks and clears its slot at the next read-back or
// synchronize and reports the error there, so a bad id never reads or
// writes out of bounds and never turns into a plausible result.

/// `out[l, i, r] = src[l, ids[i], r]` over a contiguous source; u32 ids
/// (`ids_i64`: `uvec2` ids; a non-zero high word is out of range).
/// Push: `n, left, n_ids, right, dim_size, src_off, ids_off`.
pub fn k_index_select(wg: usize, elem8: bool, ids_i64: bool) -> String {
    let ty = if elem8 { "uvec2" } else { "uint" };
    let mut s = prelude(wg, 4);
    s += &buf(0, ty, "src", true);
    s += &buf(1, if ids_i64 { "uvec2" } else { "uint" }, "ids", true);
    s += &buf(2, ty, "o", false);
    s += &buf(3, "uint", "fault", false);
    let id = if ids_i64 {
        "uvec2 rr = ids[pc.ids_off + j]; uint id = rr.y != 0u ? 0xFFFFFFFFu : rr.x;"
    } else {
        "uint id = ids[pc.ids_off + j];"
    };
    s += &format!(
        r#"
layout(push_constant) uniform PC {{ int n; int left; int n_ids; int right; int dim_size; int src_off; int ids_off; int fslot; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int i = gid();
    if (i >= pc.n) return;
    int r = i % pc.right;
    int t = i / pc.right;
    int j = t % pc.n_ids;
    int l = t / pc.n_ids;
    {id}
    if (id >= uint(pc.dim_size)) {{ fault[pc.fslot] = 1u; return; }}
    o[i] = src[pc.src_off + (l * pc.dim_size + int(id)) * pc.right + r];
}}
"#
    );
    s
}

/// gather along `dim`: `ix.s0` maps output index → ids offset, `ix.s1` →
/// src offset with the `dim` coordinate zeroed.  Push: `n, src_dim_stride, dim_size`.
pub fn k_gather(wg: usize) -> String {
    let mut s = prelude(wg, 4);
    s += &buf(0, "uint", "src", true);
    s += &buf(1, "uint", "ids", true);
    s += &buf(2, "uint", "o", false);
    s += &buf(3, "uint", "fault", false);
    s += r#"
layout(push_constant) uniform PC { int n; int src_dim_stride; int dim_size; int fslot; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n) return;
    uint id = ids[off0(i)];
    if (id >= uint(pc.dim_size)) { fault[pc.fslot] = 1u; return; }
    o[i] = src[off1(i) + int(id) * pc.src_dim_stride];
}
"#;
    s
}

/// scatter (set or add) along `dim`: one invocation per position of the
/// ids space with `dim` collapsed (`ix` enumerates that space: s0 = ids,
/// s1 = src, s2 = dst with the dim stride zeroed); the invocation walks
/// `n_j` positions along `dim` in order, so duplicates resolve exactly as
/// the sequential CPU loop does (last write wins / sums in order).
/// Push: `n, n_j, ids_ds, src_ds, dst_ds, dim_size`.
pub fn k_scatter(wg: usize, add: bool) -> String {
    let ty = if add { "float" } else { "uint" };
    let mut s = prelude(wg, 4);
    s += &buf(0, ty, "dst", false);
    s += &buf(1, "uint", "ids", true);
    s += &buf(2, ty, "src", true);
    s += &buf(3, "uint", "fault", false);
    let write = if add {
        "dst[d + int(id) * pc.dst_ds] += v;"
    } else {
        "dst[d + int(id) * pc.dst_ds] = v;"
    };
    s += &format!(
        r#"
layout(push_constant) uniform PC {{ int n; int n_j; int ids_ds; int src_ds; int dst_ds; int dim_size; int fslot; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int i = gid();
    if (i >= pc.n) return;
    int a = off0(i);
    int b = off1(i);
    int d = off2(i);
    for (int j = 0; j < pc.n_j; j++) {{
        uint id = ids[a + j * pc.ids_ds];
        if (id >= uint(pc.dim_size)) {{ fault[pc.fslot] = 1u; continue; }}
        {ty} v = src[b + j * pc.src_ds];
        {write}
    }}
}}
"#
    );
    s
}

/// index_add: `dst[l, ids[j], r] += src[l, j, r]`, one invocation per
/// `(l, r)` looping over `j` (no atomics, CPU order).
/// Push: `n_lr, left, n_ids, right, dim_size, src_off, ids_off`.
pub fn k_index_add(wg: usize) -> String {
    let mut s = prelude(wg, 4);
    s += &buf(0, "float", "dst", false);
    s += &buf(1, "uint", "ids", true);
    s += &buf(2, "float", "src", true);
    s += &buf(3, "uint", "fault", false);
    s += r#"
layout(push_constant) uniform PC { int n_lr; int left; int n_ids; int right; int dim_size; int src_off; int ids_off; int fslot; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n_lr) return;
    int r = i % pc.right;
    int l = i / pc.right;
    for (int j = 0; j < pc.n_ids; j++) {
        uint id = ids[pc.ids_off + j];
        if (id >= uint(pc.dim_size)) { fault[pc.fslot] = 1u; continue; }
        dst[(l * pc.dim_size + int(id)) * pc.right + r] += src[pc.src_off + (l * pc.n_ids + j) * pc.right + r];
    }
}
"#;
    s
}

/// Tiled GEMM: `C[bz][m][n] = sum_k A[bz][m][k] * B[bz][k][n]` with explicit
/// strides for A and B.  `tx`×`tx` invocations per group, each computing a
/// 4×4 block of a `4tx`×`4tx` tile; TK = 16.
/// Push: `M, N, K, sam, sak, sbk, sbn, oa, ob, oc, ba, bb, bc, b_kc`.
pub fn k_gemm(wg: usize, tx: usize) -> String {
    let tm = 4 * tx;
    let tk = 16;
    let nthreads = tx * tx;
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "A", true);
    s += &buf(1, "float", "B", true);
    s += &buf(2, "float", "C", false);
    s += &format!(
        r#"
#define TX {tx}
#define TM {tm}
#define TK {tk}
#define NT {nthreads}
layout(push_constant) uniform PC {{ int M; int N; int K; int sam; int sak; int sbk; int sbn; int oa; int ob; int oc; int ba; int bb; int bc; int b_kc; }} pc;
layout(local_size_x = TX, local_size_y = TX) in;
shared float As[TK * (TM + 1)];
shared float Bs[TK * (TM + 1)];
void main() {{
    int tx = int(gl_LocalInvocationID.x);
    int ty = int(gl_LocalInvocationID.y);
    int m0 = int(gl_WorkGroupID.y) * TM;
    int n0 = int(gl_WorkGroupID.x) * TM;
    int bz = int(gl_WorkGroupID.z);
    int ao = pc.oa + bz * pc.ba;
    int bo = pc.ob + bz * pc.bb;
    int co = pc.oc + bz * pc.bc;
    float acc[16];
    for (int i = 0; i < 16; i++) acc[i] = 0.0;
    int tid = ty * TX + tx;
    for (int k0 = 0; k0 < pc.K; k0 += TK) {{
        for (int i = tid; i < TM * TK; i += NT) {{
            int mm = i / TK;
            int kk = i % TK;
            int gm = m0 + mm;
            int gk = k0 + kk;
            As[kk * (TM + 1) + mm] = (gm < pc.M && gk < pc.K) ? A[ao + gm * pc.sam + gk * pc.sak] : 0.0;
        }}
        if (pc.b_kc != 0) {{
            for (int i = tid; i < TK * TM; i += NT) {{
                int kk = i % TK;
                int nn = i / TK;
                int gk = k0 + kk;
                int gn = n0 + nn;
                Bs[kk * (TM + 1) + nn] = (gk < pc.K && gn < pc.N) ? B[bo + gk * pc.sbk + gn * pc.sbn] : 0.0;
            }}
        }} else {{
            for (int i = tid; i < TK * TM; i += NT) {{
                int kk = i / TM;
                int nn = i % TM;
                int gk = k0 + kk;
                int gn = n0 + nn;
                Bs[kk * (TM + 1) + nn] = (gk < pc.K && gn < pc.N) ? B[bo + gk * pc.sbk + gn * pc.sbn] : 0.0;
            }}
        }}
        barrier();
        for (int kk = 0; kk < TK; kk++) {{
            float a0 = As[kk * (TM + 1) + ty * 4 + 0];
            float a1 = As[kk * (TM + 1) + ty * 4 + 1];
            float a2 = As[kk * (TM + 1) + ty * 4 + 2];
            float a3 = As[kk * (TM + 1) + ty * 4 + 3];
            float b0 = Bs[kk * (TM + 1) + tx * 4 + 0];
            float b1 = Bs[kk * (TM + 1) + tx * 4 + 1];
            float b2 = Bs[kk * (TM + 1) + tx * 4 + 2];
            float b3 = Bs[kk * (TM + 1) + tx * 4 + 3];
            acc[0] += a0 * b0; acc[1] += a0 * b1; acc[2] += a0 * b2; acc[3] += a0 * b3;
            acc[4] += a1 * b0; acc[5] += a1 * b1; acc[6] += a1 * b2; acc[7] += a1 * b3;
            acc[8] += a2 * b0; acc[9] += a2 * b1; acc[10] += a2 * b2; acc[11] += a2 * b3;
            acc[12] += a3 * b0; acc[13] += a3 * b1; acc[14] += a3 * b2; acc[15] += a3 * b3;
        }}
        barrier();
    }}
    for (int r = 0; r < 4; r++) {{
        int gm = m0 + ty * 4 + r;
        if (gm >= pc.M) continue;
        for (int c = 0; c < 4; c++) {{
            int gn = n0 + tx * 4 + c;
            if (gn < pc.N) C[co + gm * pc.N + gn] = acc[r * 4 + c];
        }}
    }}
}}
"#
    );
    s
}

/// GEMV for a transposed weight (B contiguous along k): one work-group per
/// `n` (2-D grid) with the batch on z.  Push: `N, K, sbn, oa, ob, oc, ba, bb, bc`.
pub fn k_gemv_nt(wg: usize) -> String {
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "A", true);
    s += &buf(1, "float", "B", true);
    s += &buf(2, "float", "C", false);
    s += r#"
layout(push_constant) uniform PC { int N; int K; int sbn; int oa; int ob; int oc; int ba; int bb; int bc; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];
void main() {
    int n = grow();
    int bz = int(gl_WorkGroupID.z);
    int t = int(gl_LocalInvocationID.x);
    if (n >= pc.N) return;
    int a = pc.oa + bz * pc.ba;
    int w = pc.ob + bz * pc.bb + n * pc.sbn;
    float acc = 0.0;
    for (int k = t; k < pc.K; k += WG) acc += A[a + k] * B[w + k];
    sh[t] = acc;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier();
    }
    if (t == 0) C[pc.oc + bz * pc.bc + n] = sh[0];
}
"#;
    s
}

/// GEMV for a row-major B (contiguous along n): one invocation per `n`,
/// batch on the group's z.  Push: `N, K, sbk, oa, ob, oc, ba, bb, bc`.
pub fn k_gemv_nn(wg: usize) -> String {
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "A", true);
    s += &buf(1, "float", "B", true);
    s += &buf(2, "float", "C", false);
    s += r#"
layout(push_constant) uniform PC { int N; int K; int sbk; int oa; int ob; int oc; int ba; int bb; int bc; } pc;
layout(local_size_x = WG) in;
void main() {
    int n = gid();
    int bz = int(gl_WorkGroupID.z);
    if (n >= pc.N) return;
    int a = pc.oa + bz * pc.ba;
    int b = pc.ob + bz * pc.bb;
    float acc = 0.0;
    for (int k = 0; k < pc.K; k++) acc += A[a + k] * B[b + k * pc.sbk + n];
    C[pc.oc + bz * pc.bc + n] = acc;
}
"#;
    s
}

/// softmax over the contiguous last dim, one work-group per row.
/// Push: `rows, cols, off`.
pub fn k_softmax_last(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "float", "o", false);
    s += r#"
layout(push_constant) uniform PC { int rows; int cols; int off; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];
void main() {
    int row = grow();
    int t = int(gl_LocalInvocationID.x);
    if (row >= pc.rows) return;
    int r = pc.off + row * pc.cols;
    int w = row * pc.cols;
    float m = NINF();
    for (int c = t; c < pc.cols; c += WG) m = max(m, x[r + c]);
    sh[t] = m;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] = max(sh[t], sh[t + s]);
        barrier();
    }
    m = sh[0];
    barrier();
    float sum = 0.0;
    for (int c = t; c < pc.cols; c += WG) sum += exp(x[r + c] - m);
    sh[t] = sum;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier();
    }
    float inv = 1.0 / sh[0];
    // Recompute exp() so each output element is written once, fully normalized.
    barrier();
    for (int c = t; c < pc.cols; c += WG) o[w + c] = exp(x[r + c] - m) * inv;
}
"#;
    s
}

/// rms_norm: `o = x / sqrt(mean(x^2) + eps) * alpha`, one work-group per row.
/// Push: `rows, cols, off, aoff, eps`.
pub fn k_rmsnorm(wg: usize) -> String {
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "float", "alpha", true);
    s += &buf(2, "float", "o", false);
    s += r#"
layout(push_constant) uniform PC { int rows; int cols; int off; int aoff; float eps; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];
void main() {
    int row = grow();
    int t = int(gl_LocalInvocationID.x);
    if (row >= pc.rows) return;
    int r = pc.off + row * pc.cols;
    int w = row * pc.cols;
    float ss = 0.0;
    for (int c = t; c < pc.cols; c += WG) { float v = x[r + c]; ss += v * v; }
    sh[t] = ss;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier();
    }
    float scale = 1.0 / sqrt(sh[0] / float(pc.cols) + pc.eps);
    for (int c = t; c < pc.cols; c += WG) o[w + c] = x[r + c] * scale * alpha[pc.aoff + c];
}
"#;
    s
}

/// RoPE over `x [b, h, t, d]` with `cos`/`sin` `[t, d/2]` (or `[b, t, d/2]`
/// when `cs_b`); `interleaved` selects candle's `rope_i` pairing.
/// Push: `n_pairs, h, t, d, xoff, coff, soff, cs_b`.
pub fn k_rope(wg: usize, interleaved: bool) -> String {
    let mut s = prelude(wg, 4);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "float", "cs", true);
    s += &buf(2, "float", "sn", true);
    s += &buf(3, "float", "o", false);
    let body = if interleaved {
        r#"
    int base = rest * pc.d + 2 * p;
    float x1 = x[pc.xoff + base];
    float x2 = x[pc.xoff + base + 1];
    o[base] = x1 * c - x2 * s;
    o[base + 1] = x1 * s + x2 * c;
"#
    } else {
        r#"
    int base = rest * pc.d;
    float x1 = x[pc.xoff + base + p];
    float x2 = x[pc.xoff + base + hd2 + p];
    o[base + p] = x1 * c - x2 * s;
    o[base + hd2 + p] = x2 * c + x1 * s;
"#
    };
    s += &format!(
        r#"
layout(push_constant) uniform PC {{ int n_pairs; int h; int t; int d; int xoff; int coff; int soff; int cs_b; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int i = gid();
    if (i >= pc.n_pairs) return;
    int hd2 = pc.d / 2;
    int p = i % hd2;
    int rest = i / hd2;
    int tt = rest % pc.t;
    int bh = rest / pc.t;
    int b = bh / pc.h;
    int ci = (pc.cs_b != 0 ? b * pc.t * hd2 : 0) + tt * hd2 + p;
    float c = cs[pc.coff + ci];
    float s = sn[pc.soff + ci];
{body}
}}
"#
    );
    s
}

/// Byte-addressed helpers over the packed weight buffer `W` and the
/// sub-block dequantizer writing into the private `dq[32]`.
const QUANT_FN: &str = r#"
uint byte_at(uint b) { return (W[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu; }
int i8_at(uint b) { uint v = byte_at(b); return int(v) - ((v & 0x80u) != 0u ? 256 : 0); }
float f16_at(uint b) { return unpackHalf2x16(byte_at(b) | (byte_at(b + 1u) << 8u)).x; }
float bf16_at(uint b) { return uintBitsToFloat((byte_at(b + 1u) << 24u) | (byte_at(b) << 16u)); }
float f32_at(uint b) { return uintBitsToFloat(byte_at(b) | (byte_at(b + 1u) << 8u) | (byte_at(b + 2u) << 16u) | (byte_at(b + 3u) << 24u)); }
uint u32_at(uint b) { return byte_at(b) | (byte_at(b + 1u) << 8u) | (byte_at(b + 2u) << 16u) | (byte_at(b + 3u) << 24u); }
float dq[32];
void k4_scale(uint sc, int is, out uint d1, out uint m1) {
    if (is < 4) { d1 = byte_at(sc + uint(is)) & 63u; m1 = byte_at(sc + uint(is) + 4u) & 63u; }
    else {
        d1 = (byte_at(sc + uint(is) + 4u) & 0xFu) | ((byte_at(sc + uint(is) - 4u) >> 6u) << 4u);
        m1 = (byte_at(sc + uint(is) + 4u) >> 4u) | ((byte_at(sc + uint(is)) >> 6u) << 4u);
    }
}
// Dequantize elements [32*j, 32*j + 32) of the block at byte offset `p`
// (type `qt`) into dq.  Sub-blocks are the unit of parallelism; `j` is
// always 0 for 32-element blocks.
void dequant_sub(int qt, uint p, int j) {
    if (qt == 8) { // Q8_0
        float d = f16_at(p);
        for (int i = 0; i < 32; i++) dq[i] = d * float(i8_at(p + 2u + uint(i)));
    } else if (qt == 2) { // Q4_0
        float d = f16_at(p);
        for (int i = 0; i < 16; i++) {
            uint b = byte_at(p + 2u + uint(i));
            dq[i] = d * (float(b & 0xFu) - 8.0);
            dq[i + 16] = d * (float(b >> 4u) - 8.0);
        }
    } else if (qt == 3) { // Q4_1
        float d = f16_at(p); float m = f16_at(p + 2u);
        for (int i = 0; i < 16; i++) {
            uint b = byte_at(p + 4u + uint(i));
            dq[i] = d * float(b & 0xFu) + m;
            dq[i + 16] = d * float(b >> 4u) + m;
        }
    } else if (qt == 6) { // Q5_0
        float d = f16_at(p);
        uint qh = u32_at(p + 2u);
        for (int i = 0; i < 16; i++) {
            uint b = byte_at(p + 6u + uint(i));
            uint xh0 = ((qh >> uint(i)) << 4u) & 0x10u;
            uint xh1 = (qh >> uint(i + 12)) & 0x10u;
            dq[i] = d * (float((b & 0xFu) | xh0) - 16.0);
            dq[i + 16] = d * (float((b >> 4u) | xh1) - 16.0);
        }
    } else if (qt == 7) { // Q5_1
        float d = f16_at(p); float m = f16_at(p + 2u);
        uint qh = u32_at(p + 4u);
        for (int i = 0; i < 16; i++) {
            uint b = byte_at(p + 8u + uint(i));
            uint xh0 = ((qh >> uint(i)) << 4u) & 0x10u;
            uint xh1 = (qh >> uint(i + 12)) & 0x10u;
            dq[i] = d * float((b & 0xFu) | xh0) + m;
            dq[i + 16] = d * float((b >> 4u) | xh1) + m;
        }
    } else if (qt == 9) { // Q8_1
        float d = f16_at(p);
        for (int i = 0; i < 32; i++) dq[i] = d * float(i8_at(p + 4u + uint(i)));
    } else if (qt == 12 || qt == 13) { // Q4_K / Q5_K
        float d = f16_at(p); float dmin = f16_at(p + 2u);
        int g = j >> 1;
        int hf = j & 1;
        uint sc1; uint m1;
        k4_scale(p + 4u, 2 * g + hf, sc1, m1);
        float d1 = d * float(sc1); float mm1 = dmin * float(m1);
        if (qt == 12) {
            uint qs = p + 16u + 32u * uint(g);
            for (int l = 0; l < 32; l++) {
                uint q = byte_at(qs + uint(l));
                dq[l] = d1 * float(hf != 0 ? (q >> 4u) : (q & 0xFu)) - mm1;
            }
        } else {
            uint qh = p + 16u;
            uint ql = p + 48u + 32u * uint(g);
            uint u = 1u << uint(2 * g + hf);
            for (int l = 0; l < 32; l++) {
                uint q = byte_at(ql + uint(l));
                uint h = byte_at(qh + uint(l));
                dq[l] = d1 * float((hf != 0 ? (q >> 4u) : (q & 0xFu)) + ((h & u) != 0u ? 16u : 0u)) - mm1;
            }
        }
    } else if (qt == 14) { // Q6_K
        int hh = j >> 2;
        int q = j & 3;
        uint ql = p + 64u * uint(hh);
        uint qh = p + 128u + 32u * uint(hh);
        uint sc = p + 192u + 8u * uint(hh);
        float d = f16_at(p + 208u);
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            uint qll = byte_at(ql + uint(l));
            uint qlh = byte_at(ql + uint(l) + 32u);
            uint h = byte_at(qh + uint(l));
            int v;
            int s;
            if (q == 0) { v = int((qll & 0xFu) | (((h >> 0u) & 3u) << 4u)); s = i8_at(sc + uint(is)); }
            else if (q == 1) { v = int((qlh & 0xFu) | (((h >> 2u) & 3u) << 4u)); s = i8_at(sc + uint(is) + 2u); }
            else if (q == 2) { v = int((qll >> 4u) | (((h >> 4u) & 3u) << 4u)); s = i8_at(sc + uint(is) + 4u); }
            else { v = int((qlh >> 4u) | (((h >> 6u) & 3u) << 4u)); s = i8_at(sc + uint(is) + 6u); }
            dq[l] = d * float(s) * float(v - 32);
        }
    } else if (qt == 15) { // Q8_K
        float d = f32_at(p);
        for (int i = 0; i < 32; i++) dq[i] = d * float(i8_at(p + 4u + 32u * uint(j) + uint(i)));
    } else if (qt == 10) { // Q2_K
        int hh = j >> 2;
        int m = j & 3;
        uint sc = p + 8u * uint(hh) + 2u * uint(m);
        uint qs = p + 16u + 32u * uint(hh);
        float d = f16_at(p + 80u); float dmin = f16_at(p + 82u);
        uint shift = 2u * uint(m);
        uint s0 = byte_at(sc); uint s1 = byte_at(sc + 1u);
        float dl0 = d * float(s0 & 0xFu); float ml0 = dmin * float(s0 >> 4u);
        float dl1 = d * float(s1 & 0xFu); float ml1 = dmin * float(s1 >> 4u);
        for (int l = 0; l < 16; l++) {
            dq[l] = dl0 * float((byte_at(qs + uint(l)) >> shift) & 3u) - ml0;
            dq[16 + l] = dl1 * float((byte_at(qs + uint(l) + 16u) >> shift) & 3u) - ml1;
        }
    } else if (qt == 11) { // Q3_K
        uint hm = p;
        uint scp = p + 96u;
        float d = f16_at(p + 108u);
        uint aux0 = u32_at(scp); uint aux1 = u32_at(scp + 4u); uint aux2 = u32_at(scp + 8u);
        uint kmask1 = 0x03030303u; uint kmask2 = 0x0f0f0f0fu;
        uint a0 = (aux0 & kmask2) | (((aux2 >> 0u) & kmask1) << 4u);
        uint a1 = (aux1 & kmask2) | (((aux2 >> 2u) & kmask1) << 4u);
        uint a2 = ((aux0 >> 4u) & kmask2) | (((aux2 >> 4u) & kmask1) << 4u);
        uint a3 = ((aux1 >> 4u) & kmask2) | (((aux2 >> 6u) & kmask1) << 4u);
        int hh = j >> 2;
        int m = j & 3;
        int is = 8 * hh + 2 * m;
        uint w0 = is < 4 ? a0 : (is < 8 ? a1 : (is < 12 ? a2 : a3));
        uint w1 = (is + 1) < 4 ? a0 : ((is + 1) < 8 ? a1 : ((is + 1) < 12 ? a2 : a3));
        uint v0 = (w0 >> (uint(is % 4) * 8u)) & 0xFFu;
        uint v1 = (w1 >> (uint((is + 1) % 4) * 8u)) & 0xFFu;
        int s0 = int(v0) - ((v0 & 0x80u) != 0u ? 256 : 0) - 32;
        int s1 = int(v1) - ((v1 & 0x80u) != 0u ? 256 : 0) - 32;
        uint mask = 1u << uint(4 * hh + m);
        uint shift = 2u * uint(m);
        uint qs = p + 32u + 32u * uint(hh);
        float dl0 = d * float(s0); float dl1 = d * float(s1);
        for (int l = 0; l < 16; l++) {
            int q0 = int((byte_at(qs + uint(l)) >> shift) & 3u) - ((byte_at(hm + uint(l)) & mask) != 0u ? 0 : 4);
            int q1 = int((byte_at(qs + uint(l) + 16u) >> shift) & 3u) - ((byte_at(hm + uint(l) + 16u) & mask) != 0u ? 0 : 4);
            dq[l] = dl0 * float(q0);
            dq[16 + l] = dl1 * float(q1);
        }
    }
}
"#;

/// Quantized GEMV/GEMM: `C[m, n] = sum_k X[m, k] * W[n, k]`; one
/// work-group per `n` (2-D grid), `m` on z; invocations stride over
/// 32-element sub-blocks.  Push: `N, K, qt, qk, bsz, xoff, coff, M`.
pub fn k_qgemv(wg: usize) -> String {
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "X", true);
    s += &buf(1, "uint", "W", true);
    s += &buf(2, "float", "C", false);
    s += QUANT_FN;
    s += r#"
layout(push_constant) uniform PC { int N; int K; int qt; int qk; int bsz; int xoff; int coff; int M; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];
void main() {
    int n = grow();
    int m = int(gl_WorkGroupID.z);
    int t = int(gl_LocalInvocationID.x);
    if (n >= pc.N || m >= pc.M) return;
    int x = pc.xoff + m * pc.K;
    int spb = pc.qk / 32;
    int nsub = pc.K / 32;
    uint row = uint(n) * uint(pc.K / pc.qk) * uint(pc.bsz);
    float acc = 0.0;
    for (int s = t; s < nsub; s += WG) {
        int b = s / spb;
        int jj = s - b * spb;
        dequant_sub(pc.qt, row + uint(b) * uint(pc.bsz), jj);
        int xb = x + s * 32;
        float d = 0.0;
        for (int i = 0; i < 32; i++) d += X[xb + i] * dq[i];
        acc += d;
    }
    sh[t] = acc;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier();
    }
    if (t == 0) C[pc.coff + m * pc.N + n] = sh[0];
}
"#;
    s
}

/// Half / bf16 weight GEMV: `C[m, n] = sum_k X[m, k] * W[n, k]`.
/// Push: `N, K, is_bf16, xoff, coff, M`.
pub fn k_hgemv(wg: usize) -> String {
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "X", true);
    s += &buf(1, "uint", "W", true);
    s += &buf(2, "float", "C", false);
    s += r#"
float h_at(int e, int bf16) {
    uint v = (W[e >> 1] >> (uint(e & 1) * 16u)) & 0xFFFFu;
    return bf16 != 0 ? uintBitsToFloat(v << 16u) : unpackHalf2x16(v).x;
}
layout(push_constant) uniform PC { int N; int K; int is_bf16; int xoff; int coff; int M; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];
void main() {
    int n = grow();
    int m = int(gl_WorkGroupID.z);
    int t = int(gl_LocalInvocationID.x);
    if (n >= pc.N || m >= pc.M) return;
    int x = pc.xoff + m * pc.K;
    int row = n * pc.K;
    float acc = 0.0;
    for (int k = t; k < pc.K; k += WG) acc += X[x + k] * h_at(row + k, pc.is_bf16);
    sh[t] = acc;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier();
    }
    if (t == 0) C[pc.coff + m * pc.N + n] = sh[0];
}
"#;
    s
}

/// Dequantize a whole block tensor to f32: one invocation per 32-element
/// sub-block.  Push: `nsub, qt, qk, bsz`.
pub fn k_dequant(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "uint", "W", true);
    s += &buf(1, "float", "O", false);
    s += QUANT_FN;
    s += r#"
layout(push_constant) uniform PC { int nsub; int qt; int qk; int bsz; } pc;
layout(local_size_x = WG) in;
void main() {
    int s = gid();
    if (s >= pc.nsub) return;
    int spb = pc.qk / 32;
    int b = s / spb;
    int jj = s - b * spb;
    dequant_sub(pc.qt, uint(b) * uint(pc.bsz), jj);
    int o = s * 32;
    for (int i = 0; i < 32; i++) O[o + i] = dq[i];
}
"#;
    s
}

/// Dequantize f16 / bf16 elements to f32.  Push: `n, is_bf16`.
pub fn k_dequant_half(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "uint", "W", true);
    s += &buf(1, "float", "O", false);
    s += r#"
layout(push_constant) uniform PC { int n; int is_bf16; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n) return;
    uint v = (W[i >> 1] >> (uint(i & 1) * 16u)) & 0xFFFFu;
    O[i] = pc.is_bf16 != 0 ? uintBitsToFloat(v << 16u) : unpackHalf2x16(v).x;
}
"#;
    s
}

/// Half-precision embedding gather: `O[j, :] = f32(W[ids[j], :])` for an
/// f16 / bf16 `[vocab, K]` table, one invocation per output element.
/// Push: `n, K, is_bf16, ids_off, vocab, fslot`.
pub fn k_hembed(wg: usize) -> String {
    let mut s = prelude(wg, 4);
    s += &buf(0, "uint", "W", true);
    s += &buf(1, "uint", "ids", true);
    s += &buf(2, "float", "O", false);
    s += &buf(3, "uint", "fault", false);
    s += r#"
float h_at(int e, int bf16) {
    uint v = (W[e >> 1] >> (uint(e & 1) * 16u)) & 0xFFFFu;
    return bf16 != 0 ? uintBitsToFloat(v << 16u) : unpackHalf2x16(v).x;
}
layout(push_constant) uniform PC { int n; int K; int is_bf16; int ids_off; int vocab; int fslot; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n) return;
    int j = i / pc.K;
    int c = i - j * pc.K;
    uint id = ids[pc.ids_off + j];
    if (id >= uint(pc.vocab)) { fault[pc.fslot] = 1u; return; }
    O[i] = h_at(int(id) * pc.K + c, pc.is_bf16);
}
"#;
    s
}

/// Quantized embedding gather: `O[j, :] = dequant(W[ids[j], :])`, one
/// work-group per row striding over sub-blocks.
/// Push: `n_ids, K, qt, qk, bsz, ids_off, vocab`.
pub fn k_qembed(wg: usize) -> String {
    let mut s = prelude(wg, 4);
    s += &buf(0, "uint", "W", true);
    s += &buf(1, "uint", "ids", true);
    s += &buf(2, "float", "O", false);
    s += &buf(3, "uint", "fault", false);
    s += QUANT_FN;
    s += r#"
layout(push_constant) uniform PC { int n_ids; int K; int qt; int qk; int bsz; int ids_off; int vocab; int fslot; } pc;
layout(local_size_x = WG) in;
void main() {
    int j = grow();
    int t = int(gl_LocalInvocationID.x);
    if (j >= pc.n_ids) return;
    uint id = ids[pc.ids_off + j];
    if (id >= uint(pc.vocab)) { fault[pc.fslot] = 1u; return; }
    int spb = pc.qk / 32;
    int nsub = pc.K / 32;
    uint row = id * uint(pc.K / pc.qk) * uint(pc.bsz);
    int o = j * pc.K;
    for (int s = t; s < nsub; s += WG) {
        int b = s / spb;
        int jj = s - b * spb;
        dequant_sub(pc.qt, row + uint(b) * uint(pc.bsz), jj);
        for (int i = 0; i < 32; i++) O[o + s * 32 + i] = dq[i];
    }
}
"#;
    s
}
