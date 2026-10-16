// Decode megakernel: one persistent launch runs a whole single-token decode
// step of a dense pre-norm transformer (every layer's RMSNorm, Q/K/V
// projections, Q/K norms, RoPE, KV-cache write, attention, output projection,
// SwiGLU MLP and both residual adds, then the final norm and the LM head).
//
// Why: the op-at-a-time path issues ~30 launches per layer.  For a model
// that fits on the GPU the weights of a small model stream in a few
// milliseconds while launch latency and inter-kernel gaps cost several
// times that.  Here the host submits one kernel per token.
//
// Scheduling.  The step is a fixed list of stages, each split into tasks
// (a block of output rows, one attention head, ...).  Work-groups take task
// tickets in order from an atomic counter and, before running a task, wait
// until every task of the previous stage has signalled completion.  A
// work-group only ever waits on tasks with lower tickets, and a ticket is
// held only by a work-group that is already running, so the lowest
// unfinished ticket can always make progress: the kernel cannot deadlock
// however many work-groups the device keeps resident (no co-residency
// assumption, no root-group barrier).
//
// Numerics.  Each GEMV mirrors the op-at-a-time kernel the unfused path
// would pick for that weight (`k_qgemv_mr` for the 256-element formats it
// takes, `k_qgemv` for the other blocks, `k_gemv_nt` / `k_hgemv` for dense
// weights) lane-for-lane, and the norms mirror `k_rmsnorm`, so their results
// match the unfused path's.  Attention sums each score and context element
// sequentially, as the tiled GEMM does for grouped-query heads.
#pragma once
#include "kernels.hpp"

namespace joshua_sycl {
namespace mega {

// One weight matrix [n, k] (rows contiguous, GGUF block layout for block
// formats).  `cols` > 0 selects the k_qgemv_mr lane mapping with that many
// columns per pass.
struct W {
    const uchar* p;
    int qt, qk, bsz, n, k, cols, pad;
};

struct Layer {
    const float* attn_norm;
    const float* ffn_norm;
    W qkv[3];                // Q, K, V row ranges (views into a fused matrix are fine)
    const float* qkv_b[3];   // optional biases
    const float* q_norm;     // optional [head_dim]
    const float* k_norm;     // optional [head_dim]
    W o;
    const float* o_b;
    W gate, up, down;
};

// One layer's KV cache: [n_kv, cap, head_dim] f32 for keys and values.
struct Kv {
    float* k;
    float* v;
    int cap, pad;
};

// Stages of one layer, in order.
enum { S_NORM1, S_QKV, S_PREP, S_ATTN, S_OPROJ, S_NORM2, S_GATEUP, S_DOWN, S_LAYER };
// Rows one GEMV task produces.
constexpr int ROWS = 8;
// Largest head_dim the per-head stages hold in local memory.
constexpr int MAX_HD = 512;
// Local memory: reduction tree + task results + one head.
constexpr int LOCAL_FLOATS = WG + ROWS + MAX_HD;

struct Model {
    int n_layers, h, n_head, n_kv, hd, n_rot, neox, ffn, vocab;
    float eps, scale;
    const float* cos;        // [max_pos, n_rot / 2]
    const float* sin;
    const float* out_norm;
    W out;
    const float* out_b;
    const Layer* layers;     // device copy, n_layers entries
    // Tasks per stage of one layer, and the prefix sums (`start[S_LAYER]`
    // tasks per layer); then the final norm (1 task) and the head.
    int count[S_LAYER];
    int start[S_LAYER + 1];
    int head_tasks;
};

struct Scratch {
    float* x;       // [h] residual
    float* xn;      // [h] normed input of the current projection
    float* qkv;     // [n_head*hd + 2*n_kv*hd]
    float* q;       // [n_head*hd] normed, rotated queries
    float* ctx;     // [n_head*hd] attention output
    float* act;     // [ffn]
    float* scores;  // [n_head, scores_cap]
    int scores_cap, pad;
    uint* sync;     // [0] ticket, [1 + stage] completed tasks
};

using atomic_u32 = sycl::atomic_ref<uint, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>;

struct Ctx {
    sycl::nd_item<1> it;
    float* local;   // LOCAL_FLOATS
    int t() const { return (int)it.get_local_id(0); }
    void barrier() const { sycl::group_barrier(it.get_group()); }
    float* sh() const { return local; }
    float* res() const { return local + WG; }
    float* head() const { return local + WG + ROWS; }

    // Block until `*ctr` reaches `need`, then make the other work-groups'
    // writes visible to every lane.
    void wait_for(uint* ctr, uint need) const {
        if (t() == 0) {
            atomic_u32 a(*ctr);
            while (a.load(sycl::memory_order::acquire) < need) {}
        }
        barrier();
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
    }
    // Publish this task's writes and count it done.
    void signal(uint* ctr) const {
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
        barrier();
        if (t() == 0) atomic_u32(*ctr).fetch_add(1u, sycl::memory_order::release);
    }

    // Tree-sum `v` over the work-group exactly as the op-at-a-time kernels
    // do (`sh[t] += sh[t + s]`, halving s), returning the total to all lanes.
    float group_sum(float v) const {
        float* s = sh();
        s[t()] = v;
        barrier();
        for (int k = WG / 2; k > 0; k >>= 1) {
            if (t() < k) s[t()] += s[t() + k];
            barrier();
        }
        float r = s[0];
        barrier();
        return r;
    }
    float group_max(float v) const {
        float* s = sh();
        s[t()] = v;
        barrier();
        for (int k = WG / 2; k > 0; k >>= 1) {
            if (t() < k) s[t()] = fmax(s[t()], s[t() + k]);
            barrier();
        }
        float r = s[0];
        barrier();
        return r;
    }

    // k_rmsnorm over one row of `cols`: o = x / sqrt(mean(x^2) + eps) * alpha.
    // `o` may be local memory; it may not alias `x` unless every lane reads
    // only the elements it writes (true: both loops stride by WG from t).
    void rmsnorm(const float* x, const float* alpha, float* o, int cols, float eps) const {
        float ss = 0.0f;
        for (int c = t(); c < cols; c += WG) { float v = x[c]; ss += v * v; }
        float total = group_sum(ss);
        float scale = 1.0f / sqrt(total / (float)cols + eps);
        for (int c = t(); c < cols; c += WG) o[c] = x[c] * scale * alpha[c];
        barrier();
    }

    // One output row of a dense or one-row-kernel weight: k_gemv_nt,
    // k_hgemv or k_qgemv, lane-for-lane.
    float row_dot(const W& w, int n, const float* x) const {
        float acc = 0.0f;
        if (w.qt == QT_F32) {
            const float* r = reinterpret_cast<const float*>(w.p) + (int64_t)n * w.k;
            for (int k = t(); k < w.k; k += WG) acc += x[k] * r[k];
        } else if (w.qt == QT_F16 || w.qt == QT_BF16) {
            const uchar* row = w.p + (uint64_t)n * (uint64_t)w.k * 2ul;
            for (int k = t(); k < w.k; k += WG) {
                float v = w.qt == QT_BF16 ? bf16_at(row + 2 * k) : f16_at(row + 2 * k);
                acc += x[k] * v;
            }
        } else {
            const uchar* row = w.p + (uint64_t)n * (uint64_t)(w.k / w.qk) * (uint64_t)w.bsz;
            int spb = w.qk / 32;
            int nsub = w.k / 32;
            float buf[32];
            for (int s = t(); s < nsub; s += WG) {
                int b = s / spb;
                int jj = s - b * spb;
                dequant_sub(w.qt, row + (uint64_t)b * w.bsz, jj, buf);
                const float* xb = x + s * 32;
                float d = 0.0f;
                for (int i = 0; i < 32; i++) d += xb[i] * buf[i];
                acc += d;
            }
        }
        return group_sum(acc);
    }

    // One k_qgemv_mr pass (M = 1): columns n0 .. n0 + cols, results to
    // res()[base + c].
    void mr_pass(const W& w, int n0, const float* x, int base) const {
        int cols = w.cols;
        int lpc = WG / cols;
        int c = t() / lpc;
        int lane = t() - c * lpc;
        int n = n0 + c;
        int ngroups = w.k / 8;
        float acc = 0.0f;
        if (n < w.n) {
            const uchar* row = w.p + (uint64_t)n * (uint64_t)(w.k / 256) * (uint64_t)w.bsz;
            float wv8[8];
            for (int g = lane; g < ngroups; g += lpc) {
                int b = g >> 5;
                int j = (g >> 2) & 7;
                int l = g & 3;
                dequant8(w.qt, row + (uint64_t)b * w.bsz, j, l, wv8);
                float8 wv = float8(wv8[0], wv8[1], wv8[2], wv8[3], wv8[4], wv8[5], wv8[6], wv8[7]);
                float8 xv = vload8(0, x + g * 8);
                acc += dot(wv.lo(), xv.lo()) + dot(wv.hi(), xv.hi());
            }
        }
        float* s = sh();
        s[t()] = acc;
        barrier();
        for (int k = lpc / 2; k > 0; k >>= 1) {
            if (lane < k) s[t()] += s[t() + k];
            barrier();
        }
        if (lane == 0 && n < w.n && base + c < ROWS) res()[base + c] = s[t()];
        barrier();
    }

    // Rows r0 .. r0 + nr (nr <= ROWS) of `w` @ x into res()[0 .. nr).
    void rows(const W& w, int r0, int nr, const float* x) const {
        if (w.cols > 0) {
            for (int base = 0; base < nr; base += w.cols) mr_pass(w, r0 + base, x, base);
        } else {
            for (int i = 0; i < nr; i++) {
                float v = row_dot(w, r0 + i, x);
                if (t() == 0) res()[i] = v;
            }
            barrier();
        }
    }

    // RoPE on the leading n_rot dims of one head held in `v` (k_rope /
    // k_rope_i for position `pos`), written to `o` (global); the remaining
    // dims pass through.
    void rope_head(const Model& m, const float* v, float* o, int pos) const {
        int hr = m.n_rot / 2;
        const float* cs = m.cos + (int64_t)pos * hr;
        const float* sn = m.sin + (int64_t)pos * hr;
        for (int p = t(); p < hr; p += WG) {
            float c = cs[p], s = sn[p];
            if (m.neox) {
                float x1 = v[p], x2 = v[hr + p];
                o[p] = x1 * c - x2 * s;
                o[hr + p] = x2 * c + x1 * s;
            } else {
                float x1 = v[2 * p], x2 = v[2 * p + 1];
                o[2 * p] = x1 * c - x2 * s;
                o[2 * p + 1] = x1 * s + x2 * c;
            }
        }
        for (int d = m.n_rot + t(); d < m.hd; d += WG) o[d] = v[d];
    }
};

inline void run(const Ctx& c, const Model& m, const Scratch& s, const Kv* kv,
                const float* x_in, float* logits, int pos) {
    const int per_layer = m.start[S_LAYER];
    const int layer_tasks = m.n_layers * per_layer;
    const int total = layer_tasks + 1 + m.head_tasks;
    const int q_dim = m.n_head * m.hd, kv_dim = m.n_kv * m.hd;
    const int seq = pos + 1;
    uint* done = s.sync + 1;
    for (;;) {
        uint ticket = 0;
        if (c.t() == 0) ticket = atomic_u32(s.sync[0]).fetch_add(1u);
        ticket = sycl::group_broadcast(c.it.get_group(), ticket, 0);
        int task = (int)ticket;
        if (task >= total) return;

        int stage, sub, layer;
        int need;  // tasks of the previous stage
        if (task < layer_tasks) {
            layer = task / per_layer;
            int r = task - layer * per_layer;
            int st = 0;
            while (r >= m.start[st + 1]) st++;
            sub = r - m.start[st];
            stage = layer * S_LAYER + st;
            need = st > 0 ? m.count[st - 1] : m.count[S_LAYER - 1];
        } else if (task == layer_tasks) {
            layer = m.n_layers;
            sub = 0;
            stage = m.n_layers * S_LAYER;
            need = m.count[S_LAYER - 1];
        } else {
            layer = m.n_layers;
            sub = task - layer_tasks - 1;
            stage = m.n_layers * S_LAYER + 1;
            need = 1;
        }
        if (stage > 0) c.wait_for(done + stage - 1, (uint)need);

        if (layer == m.n_layers) {
            if (stage == m.n_layers * S_LAYER) {
                c.rmsnorm(s.x, m.out_norm, s.xn, m.h, m.eps);
            } else {
                int r0 = sub * ROWS, nr = sycl::min(ROWS, m.vocab - r0);
                c.rows(m.out, r0, nr, s.xn);
                if (c.t() < nr) logits[r0 + c.t()] = c.res()[c.t()] + (m.out_b ? m.out_b[r0 + c.t()] : 0.0f);
            }
            c.signal(done + stage);
            continue;
        }

        const Layer& L = m.layers[layer];
        const Kv& cache = kv[layer];
        switch (stage - layer * S_LAYER) {
        case S_NORM1: {
            const float* src = layer == 0 ? x_in : s.x;
            if (layer == 0) {
                for (int i = c.t(); i < m.h; i += WG) s.x[i] = x_in[i];
            }
            c.rmsnorm(src, L.attn_norm, s.xn, m.h, m.eps);
            break;
        }
        case S_QKV: {
            // Each of the Q, K and V ranges is split into whole tasks of
            // ROWS rows, so a task never straddles two ranges.
            int tq = (L.qkv[0].n + ROWS - 1) / ROWS, tk = (L.qkv[1].n + ROWS - 1) / ROWS;
            int g = sub < tq ? 0 : sub < tq + tk ? 1 : 2;
            int row = (g == 0 ? sub : g == 1 ? sub - tq : sub - tq - tk) * ROWS;
            const W& w = L.qkv[g];
            int nr = sycl::min(ROWS, w.n - row);
            c.rows(w, row, nr, s.xn);
            if (c.t() < nr) {
                int base = (g == 0 ? 0 : g == 1 ? q_dim : q_dim + kv_dim) + row + c.t();
                const float* b = L.qkv_b[g];
                s.qkv[base] = c.res()[c.t()] + (b ? b[row + c.t()] : 0.0f);
            }
            break;
        }
        case S_PREP: {
            float* hv = c.head();
            if (sub < m.n_head) {
                const float* src = s.qkv + sub * m.hd;
                if (L.q_norm) {
                    c.rmsnorm(src, L.q_norm, hv, m.hd, m.eps);
                } else {
                    for (int d = c.t(); d < m.hd; d += WG) hv[d] = src[d];
                    c.barrier();
                }
                c.rope_head(m, hv, s.q + sub * m.hd, pos);
            } else {
                int kh = sub - m.n_head;
                const float* ks = s.qkv + q_dim + kh * m.hd;
                const float* vs = s.qkv + q_dim + kv_dim + kh * m.hd;
                int64_t row = ((int64_t)kh * cache.cap + pos) * m.hd;
                if (L.k_norm) {
                    c.rmsnorm(ks, L.k_norm, hv, m.hd, m.eps);
                } else {
                    for (int d = c.t(); d < m.hd; d += WG) hv[d] = ks[d];
                    c.barrier();
                }
                c.rope_head(m, hv, cache.k + row, pos);
                for (int d = c.t(); d < m.hd; d += WG) cache.v[row + d] = vs[d];
            }
            break;
        }
        case S_ATTN: {
            int h = sub, kh = h / (m.n_head / m.n_kv);
            const float* q = s.q + h * m.hd;
            const float* kc = cache.k + (int64_t)kh * cache.cap * m.hd;
            const float* vc = cache.v + (int64_t)kh * cache.cap * m.hd;
            float* sc = s.scores + (int64_t)h * s.scores_cap;
            float mx = -std::numeric_limits<float>::infinity();
            for (int p = c.t(); p < seq; p += WG) {
                const float* kr = kc + (int64_t)p * m.hd;
                float acc = 0.0f;
                for (int d = 0; d < m.hd; d++) acc += q[d] * kr[d];
                float v = acc * m.scale;
                sc[p] = v;
                mx = fmax(mx, v);
            }
            mx = c.group_max(mx);
            float sum = 0.0f;
            for (int p = c.t(); p < seq; p += WG) {
                float e = exp(sc[p] - mx);
                sc[p] = e;
                sum += e;
            }
            float inv = 1.0f / c.group_sum(sum);
            for (int p = c.t(); p < seq; p += WG) sc[p] *= inv;
            // Every lane reads every probability below.
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
            c.barrier();
            for (int d = c.t(); d < m.hd; d += WG) {
                float acc = 0.0f;
                for (int p = 0; p < seq; p++) acc += vc[(int64_t)p * m.hd + d] * sc[p];
                s.ctx[h * m.hd + d] = acc;
            }
            break;
        }
        case S_OPROJ: {
            int r0 = sub * ROWS, nr = sycl::min(ROWS, m.h - r0);
            c.rows(L.o, r0, nr, s.ctx);
            if (c.t() < nr) {
                int r = r0 + c.t();
                s.x[r] = s.x[r] + (c.res()[c.t()] + (L.o_b ? L.o_b[r] : 0.0f));
            }
            break;
        }
        case S_NORM2:
            c.rmsnorm(s.x, L.ffn_norm, s.xn, m.h, m.eps);
            break;
        case S_GATEUP: {
            int r0 = sub * ROWS, nr = sycl::min(ROWS, m.ffn - r0);
            float g = 0.0f;
            c.rows(L.gate, r0, nr, s.xn);
            if (c.t() < nr) g = c.res()[c.t()];
            c.barrier();
            c.rows(L.up, r0, nr, s.xn);
            if (c.t() < nr) s.act[r0 + c.t()] = unary_f(g, OP_SILU) * c.res()[c.t()];
            break;
        }
        case S_DOWN: {
            int r0 = sub * ROWS, nr = sycl::min(ROWS, m.h - r0);
            c.rows(L.down, r0, nr, s.act);
            if (c.t() < nr) {
                int r = r0 + c.t();
                s.x[r] = s.x[r] + c.res()[c.t()];
            }
            break;
        }
        }
        c.signal(done + stage);
    }
}

} // namespace mega
} // namespace joshua_sycl
