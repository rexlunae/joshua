// Native OpenCL kernels for the joshua OpenCL backend.
//
// Every kernel here exists so that a transformer decode step can stay on
// the device: strided / broadcast elementwise ops, strided copies (what
// `contiguous`, `cat`, `narrow`, transposes materialise through), row
// reductions, gathers, comparisons, casts, a tiled GEMM plus a GEMV for the
// single-token case, and the fused softmax / RMSNorm / RoPE that the
// attention path calls through candle-nn's custom ops.  The quantized
// kernels (dequantize-in-kernel GEMV / GEMM over GGUF block formats) live in
// the second half.
//
// Conventions
// * Element indices are `int` (tensors here stay below 2^31 elements);
//   byte offsets into quantized weights are `ulong`.
// * `Idx` carries the output shape and up to three input stride sets so a
//   single kernel serves every broadcast / transposed / narrowed view.
//   Outputs are always written contiguous, row-major.
// * `WG` (the reduction work-group size) is injected by the host through
//   `-D WG=<n>`; the host picks the largest power of two the device allows,
//   at most 256.

#define MAXD 8

typedef struct {
    int nd;
    int dims[MAXD];
    int s0[MAXD];
    int s1[MAXD];
    int s2[MAXD];
    int o0;
    int o1;
    int o2;
} Idx;

// Storage offset of linear (row-major) output index `lin` under stride set `s`.
inline int off_of(int lin, __private const Idx* ix, __private const int* s, int o) {
    int off = o;
    for (int d = ix->nd - 1; d >= 0; d--) {
        int dim = ix->dims[d];
        int i = lin % dim;
        lin /= dim;
        off += i * s[d];
    }
    return off;
}

// ─── Unary ───────────────────────────────────────────────────────────────────

#define OP_EXP 0
#define OP_LOG 1
#define OP_SIN 2
#define OP_COS 3
#define OP_TANH 4
#define OP_NEG 5
#define OP_RECIP 6
#define OP_SQR 7
#define OP_SQRT 8
#define OP_GELU 9
#define OP_GELU_ERF 10
#define OP_ERF 11
#define OP_SILU 12
#define OP_ABS 13
#define OP_CEIL 14
#define OP_FLOOR 15
#define OP_ROUND 16
#define OP_RELU 17
#define OP_SIGN 18
#define OP_SIGMOID 19

inline float unary_f(float v, int op) {
    switch (op) {
        case OP_EXP: return exp(v);
        case OP_LOG: return log(v);
        case OP_SIN: return sin(v);
        case OP_COS: return cos(v);
        case OP_TANH: return tanh(v);
        case OP_NEG: return -v;
        case OP_RECIP: return 1.0f / v;
        case OP_SQR: return v * v;
        case OP_SQRT: return sqrt(v);
        // candle's Gelu is the tanh approximation.
        case OP_GELU: return 0.5f * v * (1.0f + tanh(0.79788456080286535588f * v * (1.0f + 0.044715f * v * v)));
        case OP_GELU_ERF: return 0.5f * v * (1.0f + erf(v * 0.70710678118654752440f));
        case OP_ERF: return erf(v);
        case OP_SILU: return v / (1.0f + exp(-v));
        case OP_ABS: return fabs(v);
        case OP_CEIL: return ceil(v);
        case OP_FLOOR: return floor(v);
        case OP_ROUND: return round(v);
        case OP_RELU: return v > 0.0f ? v : 0.0f;
        case OP_SIGN: return v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
        case OP_SIGMOID: return 1.0f / (1.0f + exp(-v));
        default: return v;
    }
}

__kernel void k_unary_c(__global const float* x, __global float* o, int n, int off, int op) {
    int i = get_global_id(0);
    if (i < n) o[i] = unary_f(x[off + i], op);
}

__kernel void k_unary_s(__global const float* x, __global float* o, int n, Idx ix, int op) {
    int i = get_global_id(0);
    if (i < n) o[i] = unary_f(x[off_of(i, &ix, ix.s0, ix.o0)], op);
}

__kernel void k_affine_c(__global const float* x, __global float* o, int n, int off, float mul, float add) {
    int i = get_global_id(0);
    if (i < n) o[i] = x[off + i] * mul + add;
}

__kernel void k_affine_s(__global const float* x, __global float* o, int n, Idx ix, float mul, float add) {
    int i = get_global_id(0);
    if (i < n) o[i] = x[off_of(i, &ix, ix.s0, ix.o0)] * mul + add;
}

__kernel void k_powf_s(__global const float* x, __global float* o, int n, Idx ix, float e) {
    int i = get_global_id(0);
    if (i < n) o[i] = pow(x[off_of(i, &ix, ix.s0, ix.o0)], e);
}

__kernel void k_elu_s(__global const float* x, __global float* o, int n, Idx ix, float alpha) {
    int i = get_global_id(0);
    if (i < n) {
        float v = x[off_of(i, &ix, ix.s0, ix.o0)];
        o[i] = v > 0.0f ? v : alpha * (exp(v) - 1.0f);
    }
}

// ─── Binary ──────────────────────────────────────────────────────────────────

#define BOP_ADD 0
#define BOP_SUB 1
#define BOP_MUL 2
#define BOP_DIV 3
#define BOP_MIN 4
#define BOP_MAX 5

inline float binary_f(float a, float b, int op) {
    switch (op) {
        case BOP_ADD: return a + b;
        case BOP_SUB: return a - b;
        case BOP_MUL: return a * b;
        case BOP_DIV: return a / b;
        case BOP_MIN: return a > b ? b : a;
        case BOP_MAX: return a < b ? b : a;
        default: return a;
    }
}

__kernel void k_binary_c(__global const float* a, __global const float* b, __global float* o,
                         int n, int oa, int ob, int op) {
    int i = get_global_id(0);
    if (i < n) o[i] = binary_f(a[oa + i], b[ob + i], op);
}

__kernel void k_binary_s(__global const float* a, __global const float* b, __global float* o,
                         int n, Idx ix, int op) {
    int i = get_global_id(0);
    if (i < n) o[i] = binary_f(a[off_of(i, &ix, ix.s0, ix.o0)], b[off_of(i, &ix, ix.s1, ix.o1)], op);
}

// Integer / u32 binary (used by index arithmetic on masks and ids).
__kernel void k_binary_u32_s(__global const uint* a, __global const uint* b, __global uint* o,
                             int n, Idx ix, int op) {
    int i = get_global_id(0);
    if (i < n) {
        uint x = a[off_of(i, &ix, ix.s0, ix.o0)];
        uint y = b[off_of(i, &ix, ix.s1, ix.o1)];
        uint r;
        switch (op) {
            case BOP_ADD: r = x + y; break;
            case BOP_SUB: r = x - y; break;
            case BOP_MUL: r = x * y; break;
            case BOP_DIV: r = y == 0u ? 0u : x / y; break;
            case BOP_MIN: r = x > y ? y : x; break;
            default: r = x < y ? y : x; break;
        }
        o[i] = r;
    }
}

// ─── Comparison → u8 ─────────────────────────────────────────────────────────

#define CMP_EQ 0
#define CMP_NE 1
#define CMP_LE 2
#define CMP_GE 3
#define CMP_LT 4
#define CMP_GT 5

inline uchar cmp_f(float a, float b, int op) {
    switch (op) {
        case CMP_EQ: return a == b;
        case CMP_NE: return a != b;
        case CMP_LE: return a <= b;
        case CMP_GE: return a >= b;
        case CMP_LT: return a < b;
        default: return a > b;
    }
}

__kernel void k_cmp_f32(__global const float* a, __global const float* b, __global uchar* o,
                        int n, Idx ix, int op) {
    int i = get_global_id(0);
    if (i < n) o[i] = cmp_f(a[off_of(i, &ix, ix.s0, ix.o0)], b[off_of(i, &ix, ix.s1, ix.o1)], op);
}

__kernel void k_cmp_u32(__global const uint* a, __global const uint* b, __global uchar* o,
                        int n, Idx ix, int op) {
    int i = get_global_id(0);
    if (i < n) {
        uint x = a[off_of(i, &ix, ix.s0, ix.o0)];
        uint y = b[off_of(i, &ix, ix.s1, ix.o1)];
        uchar r;
        switch (op) {
            case CMP_EQ: r = x == y; break;
            case CMP_NE: r = x != y; break;
            case CMP_LE: r = x <= y; break;
            case CMP_GE: r = x >= y; break;
            case CMP_LT: r = x < y; break;
            default: r = x > y; break;
        }
        o[i] = r;
    }
}

// ─── where_cond (4-byte payload: f32 / u32 / i32) ────────────────────────────

__kernel void k_where_4(__global const uchar* c, __global const uint* t, __global const uint* f,
                        __global uint* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) {
        uchar cv = c[off_of(i, &ix, ix.s0, ix.o0)];
        o[i] = cv ? t[off_of(i, &ix, ix.s1, ix.o1)] : f[off_of(i, &ix, ix.s2, ix.o2)];
    }
}

__kernel void k_where_8(__global const uchar* c, __global const ulong* t, __global const ulong* f,
                        __global ulong* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) {
        uchar cv = c[off_of(i, &ix, ix.s0, ix.o0)];
        o[i] = cv ? t[off_of(i, &ix, ix.s1, ix.o1)] : f[off_of(i, &ix, ix.s2, ix.o2)];
    }
}

// Cond given as u32 (candle masks are often u32 / f32 compared on the host).
__kernel void k_where_u32c_4(__global const uint* c, __global const uint* t, __global const uint* f,
                             __global uint* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) {
        uint cv = c[off_of(i, &ix, ix.s0, ix.o0)];
        o[i] = cv ? t[off_of(i, &ix, ix.s1, ix.o1)] : f[off_of(i, &ix, ix.s2, ix.o2)];
    }
}

// ─── Strided copies (element size in the kernel name) ────────────────────────

__kernel void k_copy_s1(__global const uchar* x, __global uchar* o, int n, Idx ix, int dst_off) {
    int i = get_global_id(0);
    if (i < n) o[dst_off + i] = x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_copy_s2(__global const ushort* x, __global ushort* o, int n, Idx ix, int dst_off) {
    int i = get_global_id(0);
    if (i < n) o[dst_off + i] = x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_copy_s4(__global const uint* x, __global uint* o, int n, Idx ix, int dst_off) {
    int i = get_global_id(0);
    if (i < n) o[dst_off + i] = x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_copy_s8(__global const ulong* x, __global ulong* o, int n, Idx ix, int dst_off) {
    int i = get_global_id(0);
    if (i < n) o[dst_off + i] = x[off_of(i, &ix, ix.s0, ix.o0)];
}

// copy2d: d1 rows of d2 contiguous elements, row strides src_s / dst_s.
__kernel void k_copy2d_4(__global const uint* x, __global uint* o, int d1, int d2,
                         int src_s, int dst_s, int src_o, int dst_o) {
    int j = get_global_id(0);
    int i = get_global_id(1);
    if (i < d1 && j < d2) o[dst_o + i * dst_s + j] = x[src_o + i * src_s + j];
}
__kernel void k_copy2d_8(__global const ulong* x, __global ulong* o, int d1, int d2,
                         int src_s, int dst_s, int src_o, int dst_o) {
    int j = get_global_id(0);
    int i = get_global_id(1);
    if (i < d1 && j < d2) o[dst_o + i * dst_s + j] = x[src_o + i * src_s + j];
}
__kernel void k_copy2d_1(__global const uchar* x, __global uchar* o, int d1, int d2,
                         int src_s, int dst_s, int src_o, int dst_o) {
    int j = get_global_id(0);
    int i = get_global_id(1);
    if (i < d1 && j < d2) o[dst_o + i * dst_s + j] = x[src_o + i * src_s + j];
}
__kernel void k_copy2d_2(__global const ushort* x, __global ushort* o, int d1, int d2,
                         int src_s, int dst_s, int src_o, int dst_o) {
    int j = get_global_id(0);
    int i = get_global_id(1);
    if (i < d1 && j < d2) o[dst_o + i * dst_s + j] = x[src_o + i * src_s + j];
}

// Fill (const_set / zeros) by element size.
__kernel void k_fill_4(__global uint* o, int n, Idx ix, uint v) {
    int i = get_global_id(0);
    if (i < n) o[off_of(i, &ix, ix.s0, ix.o0)] = v;
}
__kernel void k_fill_8(__global ulong* o, int n, Idx ix, ulong v) {
    int i = get_global_id(0);
    if (i < n) o[off_of(i, &ix, ix.s0, ix.o0)] = v;
}
__kernel void k_fill_1(__global uchar* o, int n, Idx ix, uchar v) {
    int i = get_global_id(0);
    if (i < n) o[off_of(i, &ix, ix.s0, ix.o0)] = v;
}
__kernel void k_fill_2(__global ushort* o, int n, Idx ix, ushort v) {
    int i = get_global_id(0);
    if (i < n) o[off_of(i, &ix, ix.s0, ix.o0)] = v;
}

// ─── Casts (strided input, contiguous output) ────────────────────────────────

__kernel void k_cast_f32_u32(__global const float* x, __global uint* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) { float v = x[off_of(i, &ix, ix.s0, ix.o0)]; o[i] = v <= 0.0f ? 0u : (uint)v; }
}
__kernel void k_cast_u32_f32(__global const uint* x, __global float* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) o[i] = (float)x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_cast_f32_u8(__global const float* x, __global uchar* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) { float v = x[off_of(i, &ix, ix.s0, ix.o0)]; o[i] = v <= 0.0f ? 0 : (v >= 255.0f ? 255 : (uchar)v); }
}
__kernel void k_cast_u8_f32(__global const uchar* x, __global float* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) o[i] = (float)x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_cast_f32_i64(__global const float* x, __global long* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) o[i] = (long)x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_cast_i64_f32(__global const long* x, __global float* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) o[i] = (float)x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_cast_u32_i64(__global const uint* x, __global long* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) o[i] = (long)x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_cast_i64_u32(__global const long* x, __global uint* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) o[i] = (uint)x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_cast_u8_u32(__global const uchar* x, __global uint* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) o[i] = (uint)x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_cast_u32_u8(__global const uint* x, __global uchar* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) o[i] = (uchar)x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_cast_f32_f16(__global const float* x, __global half* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) vstore_half(x[off_of(i, &ix, ix.s0, ix.o0)], i, o);
}
__kernel void k_cast_f16_f32(__global const half* x, __global float* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) o[i] = vload_half(off_of(i, &ix, ix.s0, ix.o0), x);
}
__kernel void k_cast_i32_f32(__global const int* x, __global float* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) o[i] = (float)x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_cast_f32_i32(__global const float* x, __global int* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) o[i] = (int)x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_cast_u32_i32(__global const uint* x, __global int* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) o[i] = (int)x[off_of(i, &ix, ix.s0, ix.o0)];
}
__kernel void k_cast_i32_u32(__global const int* x, __global uint* o, int n, Idx ix) {
    int i = get_global_id(0);
    if (i < n) o[i] = (uint)x[off_of(i, &ix, ix.s0, ix.o0)];
}

// ─── Reductions ──────────────────────────────────────────────────────────────

#define RED_SUM 0
#define RED_MAX 1
#define RED_MIN 2

inline float red_f(float a, float b, int op) {
    if (op == RED_SUM) return a + b;
    if (op == RED_MAX) return a > b ? a : b;
    return a < b ? a : b;
}

// Row reduction over the contiguous last dimension: one work-group per row.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void k_reduce_last(__global const float* x, __global float* o, int rows, int cols, int off, int op) {
    __local float sh[WG];
    int row = get_group_id(0);
    int t = get_local_id(0);
    if (row >= rows) return;
    __global const float* r = x + off + (long)row * cols;
    float acc = (op == RED_SUM) ? 0.0f : ((op == RED_MAX) ? -INFINITY : INFINITY);
    for (int c = t; c < cols; c += WG) acc = red_f(acc, r[c], op);
    sh[t] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] = red_f(sh[t], sh[t + s], op);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) o[row] = sh[0];
}

// Generic reduction: one work-item per output element; `ix` maps output
// indices to the input (reduced dims have stride 0 in ix.s0 and dim 1 in
// ix.dims); `rd` enumerates the reduced sub-space (dims + strides in s0).
__kernel void k_reduce_generic(__global const float* x, __global float* o, int n, Idx ix, Idx rd,
                               int count, int op) {
    int i = get_global_id(0);
    if (i >= n) return;
    int base = off_of(i, &ix, ix.s0, ix.o0);
    float acc = (op == RED_SUM) ? 0.0f : ((op == RED_MAX) ? -INFINITY : INFINITY);
    for (int j = 0; j < count; j++) acc = red_f(acc, x[base + off_of(j, &rd, rd.s0, 0)], op);
    o[i] = acc;
}

// ArgMax / ArgMin over the contiguous last dimension (u32 output).
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void k_arg_last(__global const float* x, __global uint* o, int rows, int cols, int off, int is_max) {
    __local float shv[WG];
    __local uint shi[WG];
    int row = get_group_id(0);
    int t = get_local_id(0);
    if (row >= rows) return;
    __global const float* r = x + off + (long)row * cols;
    float best = is_max ? -INFINITY : INFINITY;
    uint bi = 0;
    for (int c = t; c < cols; c += WG) {
        float v = r[c];
        bool better = is_max ? (v > best) : (v < best);
        if (better) { best = v; bi = c; }
    }
    shv[t] = best; shi[t] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) {
            float a = shv[t], b = shv[t + s];
            bool take = is_max ? (b > a || (b == a && shi[t + s] < shi[t])) : (b < a || (b == a && shi[t + s] < shi[t]));
            if (take) { shv[t] = b; shi[t] = shi[t + s]; }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) o[row] = shi[0];
}

// ─── index_select / gather / scatter / index_add ─────────────────────────────
//
// Out-of-range ids: the CPU backend returns an error, which a kernel cannot.
// Every indexing kernel takes the device's `fault` word as its last argument
// and sets it (skipping the element) when an id exceeds the indexed
// dimension; the host checks and clears the word at the next read-back or
// synchronize and reports the error there, so a bad id never reads or
// writes out of bounds and never turns into a plausible result.

// out[l, i, r] = src[l, ids[i], r]; src is contiguous with `dim_size` along
// the selected dim; ids are u32 (i64 ids are cast on the host side kernel).
__kernel void k_index_select_u32_4(__global const uint* src, __global const uint* ids, __global uint* o,
                                   int n, int left, int n_ids, int right, int dim_size, int src_off, int ids_off,
                                   __global uint* fault) {
    int i = get_global_id(0);
    if (i >= n) return;
    int r = i % right;
    int t = i / right;
    int j = t % n_ids;
    int l = t / n_ids;
    uint id = ids[ids_off + j];
    if (id >= (uint)dim_size) { *fault = 1; return; }
    o[i] = src[src_off + ((long)l * dim_size + id) * right + r];
}
__kernel void k_index_select_i64_4(__global const uint* src, __global const long* ids, __global uint* o,
                                   int n, int left, int n_ids, int right, int dim_size, int src_off, int ids_off,
                                   __global uint* fault) {
    int i = get_global_id(0);
    if (i >= n) return;
    int r = i % right;
    int t = i / right;
    int j = t % n_ids;
    int l = t / n_ids;
    long id = ids[ids_off + j];
    if (id < 0 || id >= dim_size) { *fault = 1; return; }
    o[i] = src[src_off + ((long)l * dim_size + id) * right + r];
}
__kernel void k_index_select_u32_8(__global const ulong* src, __global const uint* ids, __global ulong* o,
                                   int n, int left, int n_ids, int right, int dim_size, int src_off, int ids_off,
                                   __global uint* fault) {
    int i = get_global_id(0);
    if (i >= n) return;
    int r = i % right;
    int t = i / right;
    int j = t % n_ids;
    int l = t / n_ids;
    uint id = ids[ids_off + j];
    if (id >= (uint)dim_size) { *fault = 1; return; }
    o[i] = src[src_off + ((long)l * dim_size + id) * right + r];
}

// gather along `dim`: out[..., i, ...] = src[..., ids[..., i, ...], ...].
// `ix` maps output index -> ids offset (s0) ; `src_stride` is src's stride on
// `dim`, `sx` maps output index -> src offset with the `dim` coordinate zeroed
// (s1); u32 ids.
__kernel void k_gather_4(__global const uint* src, __global const uint* ids, __global uint* o,
                         int n, Idx ix, int src_dim_stride, int dim_size, __global uint* fault) {
    int i = get_global_id(0);
    if (i >= n) return;
    uint id = ids[off_of(i, &ix, ix.s0, ix.o0)];
    if (id >= (uint)dim_size) { *fault = 1; return; }
    o[i] = src[off_of(i, &ix, ix.s1, ix.o1) + (int)id * src_dim_stride];
}

// scatter (set or add) along `dim`: one work-item per position of the ids
// space with `dim` collapsed (`ix` enumerates that space: s0 = ids, s1 =
// src, s2 = dst); the work-item walks the `n_j` positions along `dim` in
// order, so duplicate ids resolve exactly as the sequential CPU loop does
// (last write wins / sums in order) with no atomics.
__kernel void k_scatter_set_4(__global uint* dst, __global const uint* ids, __global const uint* src,
                              int n, Idx ix, int n_j, int ids_ds, int src_ds, int dst_ds, int dim_size,
                              __global uint* fault) {
    int i = get_global_id(0);
    if (i >= n) return;
    int a = off_of(i, &ix, ix.s0, ix.o0);
    int b = off_of(i, &ix, ix.s1, ix.o1);
    int d = off_of(i, &ix, ix.s2, ix.o2);
    for (int j = 0; j < n_j; j++) {
        uint id = ids[a + j * ids_ds];
        if (id >= (uint)dim_size) { *fault = 1; continue; }
        dst[d + (int)id * dst_ds] = src[b + j * src_ds];
    }
}

__kernel void k_scatter_add_f32(__global float* dst, __global const uint* ids, __global const float* src,
                                int n, Idx ix, int n_j, int ids_ds, int src_ds, int dst_ds, int dim_size,
                                __global uint* fault) {
    int i = get_global_id(0);
    if (i >= n) return;
    int a = off_of(i, &ix, ix.s0, ix.o0);
    int b = off_of(i, &ix, ix.s1, ix.o1);
    int d = off_of(i, &ix, ix.s2, ix.o2);
    for (int j = 0; j < n_j; j++) {
        uint id = ids[a + j * ids_ds];
        if (id >= (uint)dim_size) { *fault = 1; continue; }
        dst[d + (int)id * dst_ds] += src[b + j * src_ds];
    }
}

// index_add: dst[l, ids[j], r] += src[l, j, r]; one work-item per (l, r),
// looping over j, so no atomics and the order matches the CPU path exactly.
__kernel void k_index_add_f32(__global float* dst, __global const uint* ids, __global const float* src,
                              int n_lr, int left, int n_ids, int right, int dim_size, int src_off, int ids_off,
                              __global uint* fault) {
    int i = get_global_id(0);
    if (i >= n_lr) return;
    int r = i % right;
    int l = i / right;
    for (int j = 0; j < n_ids; j++) {
        uint id = ids[ids_off + j];
        if (id >= (uint)dim_size) { *fault = 1; continue; }
        dst[((long)l * dim_size + id) * right + r] += src[src_off + ((long)l * n_ids + j) * right + r];
    }
}

// ─── Dense GEMM / GEMV ───────────────────────────────────────────────────────

// Tiled GEMM: C[bz][m][n] = sum_k A[bz][m][k] * B[bz][k][n].
// 16x16 work-items per group; each computes a 4x4 block of a 64x64 tile.
// A / B are read through explicit strides so transposed or broadcast operands
// need no materialisation; C is written contiguous row-major.
#define TM 64
#define TN 64
#define TK 16
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void k_gemm(__global const float* A, __global const float* B, __global float* C,
            int M, int N, int K,
            int sam, int sak, int sbk, int sbn,
            int oa, int ob, int oc, int ba, int bb, int bc, int b_kc) {
    __local float As[TK][TM + 1];
    __local float Bs[TK][TN + 1];
    int tx = get_local_id(0), ty = get_local_id(1);
    int m0 = get_group_id(1) * TM, n0 = get_group_id(0) * TN;
    int bz = get_global_id(2);
    A += oa + (long)bz * ba;
    B += ob + (long)bz * bb;
    C += oc + (long)bz * bc;
    float acc[4][4];
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) acc[r][c] = 0.0f;
    int tid = ty * 16 + tx;
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int i = tid; i < TM * TK; i += 256) {
            int mm = i / TK, kk = i % TK;
            int gm = m0 + mm, gk = k0 + kk;
            As[kk][mm] = (gm < M && gk < K) ? A[(long)gm * sam + (long)gk * sak] : 0.0f;
        }
        if (b_kc) {
            // B contiguous along k (a transposed weight): consecutive
            // work-items read consecutive k for coalescing.
            for (int i = tid; i < TK * TN; i += 256) {
                int kk = i % TK, nn = i / TK;
                int gk = k0 + kk, gn = n0 + nn;
                Bs[kk][nn] = (gk < K && gn < N) ? B[(long)gk * sbk + (long)gn * sbn] : 0.0f;
            }
        } else {
            for (int i = tid; i < TK * TN; i += 256) {
                int kk = i / TN, nn = i % TN;
                int gk = k0 + kk, gn = n0 + nn;
                Bs[kk][nn] = (gk < K && gn < N) ? B[(long)gk * sbk + (long)gn * sbn] : 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < TK; kk++) {
            float a[4], b[4];
            for (int r = 0; r < 4; r++) a[r] = As[kk][ty * 4 + r];
            for (int c = 0; c < 4; c++) b[c] = Bs[kk][tx * 4 + c];
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) acc[r][c] += a[r] * b[c];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int r = 0; r < 4; r++) {
        int gm = m0 + ty * 4 + r;
        if (gm >= M) continue;
        for (int c = 0; c < 4; c++) {
            int gn = n0 + tx * 4 + c;
            if (gn < N) C[(long)gm * N + gn] = acc[r][c];
        }
    }
}

// GEMV for a transposed weight (B contiguous along k, e.g. x @ W^T with W
// stored [N, K]): one work-group per (n, batch); the group strides over k.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void k_gemv_nt(__global const float* A, __global const float* B, __global float* C,
               int N, int K, int sbn, int oa, int ob, int oc, int ba, int bb, int bc) {
    __local float sh[WG];
    int n = get_group_id(0);
    int bz = get_group_id(1);
    int t = get_local_id(0);
    if (n >= N) return;
    __global const float* a = A + oa + (long)bz * ba;
    __global const float* w = B + ob + (long)bz * bb + (long)n * sbn;
    float acc = 0.0f;
    for (int k = t; k < K; k += WG) acc += a[k] * w[k];
    sh[t] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) C[oc + (long)bz * bc + n] = sh[0];
}

// GEMV for a row-major B (contiguous along n): one work-item per (n, batch),
// consecutive work-items read consecutive n.
__kernel void k_gemv_nn(__global const float* A, __global const float* B, __global float* C,
                        int N, int K, int sbk, int oa, int ob, int oc, int ba, int bb, int bc) {
    int n = get_global_id(0);
    int bz = get_global_id(1);
    if (n >= N) return;
    __global const float* a = A + oa + (long)bz * ba;
    __global const float* b = B + ob + (long)bz * bb;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += a[k] * b[(long)k * sbk + n];
    C[oc + (long)bz * bc + n] = acc;
}

// ─── Fused attention-path ops ────────────────────────────────────────────────

// softmax over the contiguous last dim: one work-group per row.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void k_softmax_last(__global const float* x, __global float* o, int rows, int cols, int off) {
    __local float sh[WG];
    int row = get_group_id(0);
    int t = get_local_id(0);
    if (row >= rows) return;
    __global const float* r = x + off + (long)row * cols;
    __global float* w = o + (long)row * cols;
    float m = -INFINITY;
    for (int c = t; c < cols; c += WG) m = fmax(m, r[c]);
    sh[t] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] = fmax(sh[t], sh[t + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    m = sh[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f;
    for (int c = t; c < cols; c += WG) {
        float e = exp(r[c] - m);
        w[c] = e;
        sum += e;
    }
    sh[t] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.0f / sh[0];
    for (int c = t; c < cols; c += WG) w[c] *= inv;
}

// rms_norm: o = x / sqrt(mean(x^2) + eps) * alpha, one work-group per row.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void k_rmsnorm(__global const float* x, __global const float* alpha, __global float* o,
               int rows, int cols, int off, int aoff, float eps) {
    __local float sh[WG];
    int row = get_group_id(0);
    int t = get_local_id(0);
    if (row >= rows) return;
    __global const float* r = x + off + (long)row * cols;
    __global float* w = o + (long)row * cols;
    float ss = 0.0f;
    for (int c = t; c < cols; c += WG) { float v = r[c]; ss += v * v; }
    sh[t] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float scale = 1.0f / sqrt(sh[0] / (float)cols + eps);
    for (int c = t; c < cols; c += WG) w[c] = r[c] * scale * alpha[aoff + c];
}

// RoPE, interleaved pairs (candle `rope_i`): x [b, h, t, d], cos/sin [t, d/2]
// (or [b, t, d/2] when cs_b != 0).
__kernel void k_rope_i(__global const float* x, __global const float* cs, __global const float* sn,
                       __global float* o, int n_pairs, int h, int t, int d, int xoff, int coff, int soff, int cs_b) {
    int i = get_global_id(0);
    if (i >= n_pairs) return;
    int hd2 = d / 2;
    int p = i % hd2;
    int rest = i / hd2;
    int tt = rest % t;
    int bh = rest / t;
    int b = bh / h;
    int base = (rest * d) + 2 * p;
    int ci = (cs_b ? b * t * hd2 : 0) + tt * hd2 + p;
    float c = cs[coff + ci], s = sn[soff + ci];
    float x1 = x[xoff + base], x2 = x[xoff + base + 1];
    o[base] = x1 * c - x2 * s;
    o[base + 1] = x1 * s + x2 * c;
}

// RoPE, split halves (candle `rope`): o[i] = x[i] c - x[i+d/2] s ;
// o[i+d/2] = x[i+d/2] c + x[i] s.
__kernel void k_rope(__global const float* x, __global const float* cs, __global const float* sn,
                     __global float* o, int n_pairs, int h, int t, int d, int xoff, int coff, int soff, int cs_b) {
    int i = get_global_id(0);
    if (i >= n_pairs) return;
    int hd2 = d / 2;
    int p = i % hd2;
    int rest = i / hd2;
    int tt = rest % t;
    int bh = rest / t;
    int b = bh / h;
    int base = rest * d;
    int ci = (cs_b ? b * t * hd2 : 0) + tt * hd2 + p;
    float c = cs[coff + ci], s = sn[soff + ci];
    float x1 = x[xoff + base + p], x2 = x[xoff + base + hd2 + p];
    o[base + p] = x1 * c - x2 * s;
    o[base + hd2 + p] = x2 * c + x1 * s;
}

// ─── Quantized weights: dequantize-in-kernel GEMV / GEMM ─────────────────────
//
// Weights are GGUF blocks exactly as on disk, held in a device buffer at a
// byte offset (`woff`, so a page-aligned zero-copy mapping can be used).  The
// matrix is [N, K] row-major in elements (candle's transposed weight): row n
// is K/QK blocks.  Each kernel computes dot(x[m, :], W[n, :]).
//
// Block layouts (little-endian, as in ggml):
//   Q8_0 : { f16 d; i8 qs[32] }                      34 bytes / 32 elems
//   Q4_0 : { f16 d; u8 qs[16] }                      18 bytes / 32 elems (lo nibble = elem i, hi = i+16)
//   Q4_1 : { f16 d; f16 m; u8 qs[16] }               20 bytes / 32 elems
//   Q5_0 : { f16 d; u8 qh[4]; u8 qs[16] }            22 bytes / 32 elems
//   Q5_1 : { f16 d; f16 m; u8 qh[4]; u8 qs[16] }     24 bytes / 32 elems
//   Q8_1 : { f16 d; f16 s; i8 qs[32] }               36 bytes / 32 elems
//   Q4_K : { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] }        144 bytes / 256
//   Q5_K : { f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128] } 176 bytes / 256
//   Q6_K : { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d }        210 bytes / 256
//   Q8_K : { f32 d; i8 qs[256]; i16 bsums[16] }       292 bytes / 256
//   Q2_K : { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin }          84 bytes / 256
//   Q3_K : { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d }      110 bytes / 256
//   F16  : half                                        2 bytes
//   BF16 : ushort                                      2 bytes

inline float f16_at(__global const uchar* p) {
    return vload_half(0, (__global const half*)p);
}
inline float bf16_at(__global const uchar* p) {
    uint u = ((uint)p[1] << 24) | ((uint)p[0] << 16);
    return as_float(u);
}
inline short i16_at(__global const uchar* p) {
    return (short)((ushort)p[0] | ((ushort)p[1] << 8));
}

#define QT_F32 0
#define QT_F16 1
#define QT_Q4_0 2
#define QT_Q4_1 3
#define QT_Q5_0 6
#define QT_Q5_1 7
#define QT_Q8_0 8
#define QT_Q8_1 9
#define QT_Q2_K 10
#define QT_Q3_K 11
#define QT_Q4_K 12
#define QT_Q5_K 13
#define QT_Q6_K 14
#define QT_Q8_K 15
#define QT_BF16 30

// Dequantize elements [32*j, 32*j + 32) of the block at `p` into `out`
// (32 floats).  Sub-blocks are the unit of parallelism: a K-quant block
// holds 256 elements, and one work-item per 32 keeps a work-group busy on
// a row of a few thousand elements.  `j` is always 0 for 32-element blocks.
inline void dequant_sub(int qt, __global const uchar* p, int j, float* out) {
    if (qt == QT_Q8_0) {
        float d = f16_at(p);
        for (int i = 0; i < 32; i++) out[i] = d * (float)((char)p[2 + i]);
    } else if (qt == QT_Q4_0) {
        float d = f16_at(p);
        for (int i = 0; i < 16; i++) {
            uchar b = p[2 + i];
            out[i] = d * ((float)(b & 0x0F) - 8.0f);
            out[i + 16] = d * ((float)(b >> 4) - 8.0f);
        }
    } else if (qt == QT_Q4_1) {
        float d = f16_at(p), m = f16_at(p + 2);
        for (int i = 0; i < 16; i++) {
            uchar b = p[4 + i];
            out[i] = d * (float)(b & 0x0F) + m;
            out[i + 16] = d * (float)(b >> 4) + m;
        }
    } else if (qt == QT_Q5_0) {
        float d = f16_at(p);
        uint qh = (uint)p[2] | ((uint)p[3] << 8) | ((uint)p[4] << 16) | ((uint)p[5] << 24);
        for (int i = 0; i < 16; i++) {
            uchar b = p[6 + i];
            int xh0 = ((qh >> i) << 4) & 0x10;
            int xh1 = (qh >> (i + 12)) & 0x10;
            out[i] = d * (float)(((int)(b & 0x0F) | xh0) - 16);
            out[i + 16] = d * (float)(((int)(b >> 4) | xh1) - 16);
        }
    } else if (qt == QT_Q5_1) {
        float d = f16_at(p), m = f16_at(p + 2);
        uint qh = (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16) | ((uint)p[7] << 24);
        for (int i = 0; i < 16; i++) {
            uchar b = p[8 + i];
            int xh0 = ((qh >> i) << 4) & 0x10;
            int xh1 = (qh >> (i + 12)) & 0x10;
            out[i] = d * (float)((int)(b & 0x0F) | xh0) + m;
            out[i + 16] = d * (float)((int)(b >> 4) | xh1) + m;
        }
    } else if (qt == QT_Q8_1) {
        float d = f16_at(p);
        for (int i = 0; i < 32; i++) out[i] = d * (float)((char)p[4 + i]);
    } else if (qt == QT_Q4_K || qt == QT_Q5_K) {
        // 64-element groups g share a (scale, min) pair per 32-half.
        float d = f16_at(p), dmin = f16_at(p + 2);
        __global const uchar* sc = p + 4;
        int g = j >> 1, hf = j & 1;
        int is = 2 * g + hf;
        uchar sc1, m1;
        if (is < 4) { sc1 = sc[is] & 63; m1 = sc[is + 4] & 63; }
        else { sc1 = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); m1 = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
        float d1 = d * (float)sc1, mm1 = dmin * (float)m1;
        if (qt == QT_Q4_K) {
            __global const uchar* qs = p + 16 + 32 * g;
            for (int l = 0; l < 32; l++) {
                uchar q = qs[l];
                out[l] = d1 * (float)(hf ? (q >> 4) : (q & 0xF)) - mm1;
            }
        } else {
            __global const uchar* qh = p + 16;
            __global const uchar* ql = p + 48 + 32 * g;
            uchar u = (uchar)(1 << (2 * g + hf));
            for (int l = 0; l < 32; l++) {
                uchar q = ql[l];
                out[l] = d1 * (float)((hf ? (q >> 4) : (q & 0xF)) + ((qh[l] & u) ? 16 : 0)) - mm1;
            }
        }
    } else if (qt == QT_Q6_K) {
        int hh = j >> 2, q = j & 3;
        __global const uchar* ql = p + 64 * hh;
        __global const uchar* qh = p + 128 + 32 * hh;
        __global const uchar* sc = p + 192 + 8 * hh;
        float d = f16_at(p + 208);
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int v;
            char s;
            if (q == 0) { v = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4); s = (char)sc[is]; }
            else if (q == 1) { v = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4); s = (char)sc[is + 2]; }
            else if (q == 2) { v = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4); s = (char)sc[is + 4]; }
            else { v = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4); s = (char)sc[is + 6]; }
            out[l] = d * (float)s * (float)(v - 32);
        }
    } else if (qt == QT_Q8_K) {
        float d = as_float((uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24));
        for (int i = 0; i < 32; i++) out[i] = d * (float)((char)p[4 + 32 * j + i]);
    } else if (qt == QT_Q2_K) {
        int hh = j >> 2, m = j & 3;
        __global const uchar* sc = p + 8 * hh + 2 * m;
        __global const uchar* qs = p + 16 + 32 * hh;
        float d = f16_at(p + 80), dmin = f16_at(p + 82);
        int shift = 2 * m;
        uchar s0 = sc[0], s1 = sc[1];
        float dl0 = d * (float)(s0 & 0xF), ml0 = dmin * (float)(s0 >> 4);
        float dl1 = d * (float)(s1 & 0xF), ml1 = dmin * (float)(s1 >> 4);
        for (int l = 0; l < 16; l++) {
            out[l] = dl0 * (float)((qs[l] >> shift) & 3) - ml0;
            out[16 + l] = dl1 * (float)((qs[l + 16] >> shift) & 3) - ml1;
        }
    } else if (qt == QT_Q3_K) {
        __global const uchar* hm = p;
        __global const uchar* scp = p + 96;
        float d = f16_at(p + 108);
        // Unpack the 16 6-bit scales.
        uint aux[4];
        for (int i = 0; i < 4; i++) aux[i] = (uint)scp[4 * i] | ((uint)scp[4 * i + 1] << 8) | ((uint)scp[4 * i + 2] << 16) | ((uint)scp[4 * i + 3] << 24);
        uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
        uint tmp = aux[2];
        uint as4[4];
        as4[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        as4[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        as4[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        as4[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        int hh = j >> 2, m = j & 3;
        int is = 8 * hh + 2 * m;
        int s0 = (int)(char)((as4[is / 4] >> (8 * (is % 4))) & 0xFF) - 32;
        int s1 = (int)(char)((as4[(is + 1) / 4] >> (8 * ((is + 1) % 4))) & 0xFF) - 32;
        uchar mask = (uchar)(1 << (4 * hh + m));
        int shift = 2 * m;
        __global const uchar* qs = p + 32 + 32 * hh;
        float dl0 = d * (float)s0, dl1 = d * (float)s1;
        for (int l = 0; l < 16; l++) {
            int q0 = (int)((qs[l] >> shift) & 3) - ((hm[l] & mask) ? 0 : 4);
            int q1 = (int)((qs[l + 16] >> shift) & 3) - ((hm[l + 16] & mask) ? 0 : 4);
            out[l] = dl0 * (float)q0;
            out[16 + l] = dl1 * (float)q1;
        }
    }
}

// Quantized GEMV/GEMM: C[m, n] = sum_k X[m, k] * W[n, k]; W blocks at
// `w + woff`, `qk` elements per block, `bsz` bytes per block.  One
// work-group per (n, m); the group's work-items stride over 32-element
// sub-blocks, dequantizing each into private memory.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void k_qgemv(__global const float* X, __global const uchar* W, __global float* C,
             int N, int K, int qt, int qk, int bsz, ulong woff, int xoff, int coff, int M) {
    __local float sh[WG];
    int n = get_group_id(0);
    int m = get_group_id(1);
    int t = get_local_id(0);
    if (n >= N || m >= M) return;
    __global const float* x = X + xoff + (long)m * K;
    __global const uchar* row = W + woff + (ulong)n * (ulong)(K / qk) * (ulong)bsz;
    int spb = qk / 32;
    int nsub = K / 32;
    float acc = 0.0f;
    float buf[32];
    for (int s = t; s < nsub; s += WG) {
        int b = s / spb;
        int jj = s - b * spb;
        dequant_sub(qt, row + (ulong)b * bsz, jj, buf);
        __global const float* xb = x + s * 32;
        float d = 0.0f;
        for (int i = 0; i < 32; i++) d += xb[i] * buf[i];
        acc += d;
    }
    sh[t] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) C[coff + (long)m * N + n] = sh[0];
}

// Half / bf16 weight GEMV (no blocks): C[m, n] = sum_k X[m,k] * W[n,k].
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void k_hgemv(__global const float* X, __global const uchar* W, __global float* C,
             int N, int K, int is_bf16, ulong woff, int xoff, int coff, int M) {
    __local float sh[WG];
    int n = get_group_id(0);
    int m = get_group_id(1);
    int t = get_local_id(0);
    if (n >= N || m >= M) return;
    __global const float* x = X + xoff + (long)m * K;
    __global const uchar* row = W + woff + (ulong)n * (ulong)K * 2ul;
    float acc = 0.0f;
    for (int k = t; k < K; k += WG) {
        float w = is_bf16 ? bf16_at(row + 2 * k) : f16_at(row + 2 * k);
        acc += x[k] * w;
    }
    sh[t] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) C[coff + (long)m * N + n] = sh[0];
}

// Dequantize a whole tensor to f32: one work-item per 32-element sub-block.
__kernel void k_dequant(__global const uchar* W, __global float* O, int nsub, int qt, int qk, int bsz, ulong woff) {
    int s = get_global_id(0);
    if (s >= nsub) return;
    int spb = qk / 32;
    int b = s / spb;
    int jj = s - b * spb;
    float buf[32];
    dequant_sub(qt, W + woff + (ulong)b * bsz, jj, buf);
    __global float* o = O + (long)s * 32;
    for (int i = 0; i < 32; i++) o[i] = buf[i];
}

__kernel void k_dequant_half(__global const uchar* W, __global float* O, int n, int is_bf16, ulong woff) {
    int i = get_global_id(0);
    if (i >= n) return;
    O[i] = is_bf16 ? bf16_at(W + woff + 2 * i) : f16_at(W + woff + 2 * i);
}

// Quantized embedding gather: O[j, :] = dequant(W[ids[j], :]); one work-group
// per row, work-items stride over sub-blocks.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void k_qembed(__global const uchar* W, __global const uint* ids, __global float* O,
              int n_ids, int K, int qt, int qk, int bsz, ulong woff, int ids_off, int vocab,
              __global uint* fault) {
    int j = get_group_id(0);
    int t = get_local_id(0);
    if (j >= n_ids) return;
    uint id = ids[ids_off + j];
    if (id >= (uint)vocab) { *fault = 1; return; }
    __global const uchar* row = W + woff + (ulong)id * (ulong)(K / qk) * (ulong)bsz;
    __global float* o = O + (long)j * K;
    int spb = qk / 32;
    int nsub = K / 32;
    float buf[32];
    for (int s = t; s < nsub; s += WG) {
        int b = s / spb;
        int jj = s - b * spb;
        dequant_sub(qt, row + (ulong)b * bsz, jj, buf);
        for (int i = 0; i < 32; i++) o[s * 32 + i] = buf[i];
    }
}
