// C ABI boundary for Rust. All exceptions are caught here; no C++ object or
// standard-library type crosses the boundary. ABI version changes on any
// incompatible change to argument layouts or kernel dispatch.
#include "kernels.hpp"
#include "mega.hpp"
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <type_traits>

#if defined(_WIN32)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

namespace {
thread_local std::string last_error;
// DPC++ also requires serialization across separate device contexts.
std::mutex runtime_mutex;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> int guard(F&& f) noexcept {
    try { std::lock_guard lock(runtime_mutex); f(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); }
    catch (...) { last_error = "unknown SYCL exception"; }
    return -1;
}
struct Context {
    sycl::queue queue;
    // A second in-order queue for weight uploads (the bounded VRAM expert
    // cache's background uploader, #114).  On the shared compute queue a
    // large upload sits between already-enqueued kernels and delays every
    // one of them — with a small-residency cache uploading continuously
    // that serialized decode against the transfer stream.  On its own
    // queue an upload only occupies the PCIe link; the uploader finishes
    // this queue before a slot is published, so kernels never read a
    // half-written expert.
    sycl::queue transfer_queue;
    std::mutex mutex;
    std::vector<std::pair<void*, size_t>> retired;
    size_t retired_bytes = 0;
    size_t pending = 0;
    explicit Context(sycl::device dev) : queue(dev, [](sycl::exception_list errors) {
        for (auto error : errors) std::rethrow_exception(error);
    }, sycl::property::queue::in_order{}), transfer_queue(dev, [](sycl::exception_list errors) {
        for (auto error : errors) std::rethrow_exception(error);
    }, sycl::property::queue::in_order{}) {}
    void wait() {
        queue.wait_and_throw();
        transfer_queue.wait_and_throw();
        for (auto [p, size] : retired) sycl::free(p, queue);
        retired.clear(); retired_bytes = 0; pending = 0;
    }
    template<class F> void submit(F&& f) {
        queue.submit(std::forward<F>(f));
        if (++pending >= 256) wait();
    }
    ~Context() {
        // Never free live device memory, even during unwinding.
        try { wait(); } catch (...) { queue.wait(); transfer_queue.wait_and_throw(); }
        for (auto [p, size] : retired) sycl::free(p, queue);
    }
};
using Handle = std::shared_ptr<Context>;
struct Buffer {
    Handle owner;
    void* data;
    size_t size;
};
struct Arg { const void* data; size_t size; };
Context& context(uintptr_t h) { require(h != 0, "null SYCL context"); return **reinterpret_cast<Handle*>(h); }
Buffer& buffer(uintptr_t h) { require(h != 0, "null SYCL buffer"); return *reinterpret_cast<Buffer*>(h); }
void bounds(const Buffer& b, size_t off, size_t size) {
    require(off <= b.size && size <= b.size - off, "SYCL buffer range out of bounds");
}
template<class T> T arg(const Arg* args, size_t count, size_t index) {
    require(index < count && args[index].size == sizeof(T), "invalid SYCL kernel argument");
    T value;
    std::memcpy(&value, args[index].data, sizeof(T));
    if constexpr (std::is_pointer_v<T>) return static_cast<T>(buffer(reinterpret_cast<uintptr_t>(value)).data);
    else return value;
}
void dispatch(Context& ctx, const std::string& name, const Arg* args, size_t count,
              sycl::range<3> global, sycl::range<3> local_size) {
    using namespace joshua_sycl;
#include "dispatch.inc"
}
}
API uint32_t joshua_sycl_abi_version() noexcept { return 1; }
API const char* joshua_sycl_error() noexcept { return last_error.c_str(); }
API int joshua_sycl_open(size_t ordinal, uintptr_t* out) noexcept {
    return guard([&] {
        // Respect ONEAPI_DEVICE_SELECTOR. Prefer GPUs when present, otherwise
        // allow a real SYCL CPU device for testing on machines without a GPU.
        auto devices = sycl::device::get_devices();
        auto gpu = std::find_if(devices.begin(), devices.end(), [](auto& d) { return d.is_gpu(); });
        if (gpu != devices.end()) devices.erase(std::remove_if(devices.begin(), devices.end(), [](auto& d) { return !d.is_gpu(); }), devices.end());
        // Two very different conditions share one error today: an empty
        // device list (the usual failure - the oneAPI UR adapter's own deps,
        // level-zero loader / UMF / hwloc, are missing so no device is even
        // enumerable) and a genuinely too-large ordinal.  Give the empty
        // case an actionable message pointing at sycl-ls, and include the
        // visible count for the ordinal case.
        if (devices.empty()) {
            throw std::runtime_error(
                "no SYCL devices visible; run `sycl-ls --verbose` and check that the Level Zero "
                "loader (libze_loader.so.1), UMF (libumf.so.1) and hwloc (libhwloc.so.15) load -- "
                "oneAPI bundles them under /opt/intel/oneapi/{umf,tcm}/<ver>/lib");
        }
        if (ordinal >= devices.size()) {
            std::string msg = "SYCL device ordinal out of range: requested " + std::to_string(ordinal)
                + " but only " + std::to_string(devices.size()) + " device(s) visible";
            throw std::runtime_error(msg);
        }
        auto dev = devices[ordinal];
        require(dev.has(sycl::aspect::usm_device_allocations), "SYCL device has no device USM support");
        require(dev.get_info<sycl::info::device::max_work_group_size>() >= 64, "SYCL backend requires workgroups of 64 invocations");
        require(dev.get_info<sycl::info::device::local_mem_size>() >= 16384, "SYCL backend requires 16 KiB of local memory");
        auto sizes = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
        require(sizes[2] >= 64 && sizes[1] >= 16, "SYCL work item limits are too small");
        *out = reinterpret_cast<uintptr_t>(new Handle(std::make_shared<Context>(dev)));
    });
}
API int joshua_sycl_close(uintptr_t h) noexcept {
    return guard([&] { delete reinterpret_cast<Handle*>(h); });
}
API int joshua_sycl_info(uintptr_t h, char* name, size_t len, uint64_t* memory) noexcept {
    return guard([&] {
        auto dev = context(h).queue.get_device();
        auto text = dev.get_info<sycl::info::device::name>();
        require(len > 0, "empty name buffer");
        std::strncpy(name, text.c_str(), len - 1); name[len - 1] = 0;
        *memory = dev.get_info<sycl::info::device::global_mem_size>();
    });
}
API int joshua_sycl_alloc(uintptr_t h, size_t size, uintptr_t* out) noexcept {
    return guard([&] {
        auto owner = *reinterpret_cast<Handle*>(h);
        auto& ctx = *owner;
        std::lock_guard lock(ctx.mutex);
        auto p = sycl::malloc_device(std::max(size, size_t(1)), ctx.queue);
        require(p != nullptr, "SYCL device allocation failed");
        try { *out = reinterpret_cast<uintptr_t>(new Buffer{owner, p, size}); }
        catch (...) { sycl::free(p, ctx.queue); throw; }
    });
}
API int joshua_sycl_free(uintptr_t h) noexcept {
    return guard([&] {
        std::unique_ptr<Buffer> b(reinterpret_cast<Buffer*>(h));
        auto owner = b->owner;
        std::lock_guard lock(owner->mutex);
        // A kernel may still reference a dropped Rust tensor. Retire the USM
        // pointer until the in-order queue completes. Bound deferred memory.
        owner->retired.emplace_back(b->data, b->size);
        owner->retired_bytes += b->size;
        if (owner->retired_bytes >= (64u << 20)) owner->wait();
    });
}
API int joshua_sycl_finish(uintptr_t h) noexcept {
    return guard([&] { auto& ctx = context(h); std::lock_guard lock(ctx.mutex); ctx.wait(); });
}
API int joshua_sycl_write(uintptr_t h, uintptr_t dst, size_t off, size_t size, const void* src) noexcept {
    return guard([&] {
        auto& ctx = context(h); auto& b = buffer(dst); bounds(b, off, size);
        require(b.owner.get() == &ctx, "SYCL buffer belongs to another context");
        std::lock_guard lock(ctx.mutex);
        ctx.queue.memcpy(static_cast<char*>(b.data) + off, src, size); ctx.wait();
    });
}
API int joshua_sycl_transfer_write(uintptr_t h, uintptr_t dst, size_t off, size_t size, const void* src) noexcept {
    return guard([&] {
        auto& ctx = context(h); auto& b = buffer(dst); bounds(b, off, size);
        require(b.owner.get() == &ctx, "SYCL buffer belongs to another context");
        std::lock_guard lock(ctx.mutex);
        // Blocking on the *transfer* queue: the calling thread (the expert
        // uploader) waits for its own memcpy, but no compute-queue kernel
        // is delayed and no kernel can slip ahead of the write on this
        // in-order queue, so the storage is complete when this returns.
        ctx.transfer_queue.memcpy(static_cast<char*>(b.data) + off, src, size);
        ctx.transfer_queue.wait_and_throw();
    });
}
API int joshua_sycl_read(uintptr_t h, uintptr_t src, size_t off, size_t size, void* dst) noexcept {
    return guard([&] {
        auto& ctx = context(h); auto& b = buffer(src); bounds(b, off, size);
        require(b.owner.get() == &ctx, "SYCL buffer belongs to another context");
        std::lock_guard lock(ctx.mutex);
        ctx.queue.memcpy(dst, static_cast<char*>(b.data) + off, size); ctx.wait();
    });
}
API int joshua_sycl_copy(uintptr_t h, uintptr_t src, uintptr_t dst, size_t so, size_t d, size_t size) noexcept {
    return guard([&] {
        auto& ctx = context(h); auto& a = buffer(src); auto& b = buffer(dst);
        bounds(a, so, size); bounds(b, d, size);
        require(a.owner.get() == &ctx && b.owner.get() == &ctx, "SYCL buffer belongs to another context");
        std::lock_guard lock(ctx.mutex);
        ctx.queue.memcpy(static_cast<char*>(b.data) + d, static_cast<char*>(a.data) + so, size);
        if (++ctx.pending >= 256) ctx.wait();
    });
}
API int joshua_sycl_launch(uintptr_t h, const char* name, const Arg* args, size_t count,
                          const size_t* global, const size_t* local) noexcept {
    return guard([&] {
        auto& ctx = context(h); std::lock_guard lock(ctx.mutex);
        for (int i=0; i<3; i++) require(local[i] > 0 && global[i] % local[i] == 0, "invalid SYCL launch dimensions");
        dispatch(ctx, name, args, count, {global[2], global[1], global[0]}, {local[2], local[1], local[0]});
    });
}

// ─── Decode megakernel (mega.hpp) ────────────────────────────────────────────
// The kernel's GEMVs reduce each row over one sub-group; 16 lanes is the
// width every Intel Xe GPU offers.
#ifdef JOSHUA_SYCL_EMU_IMPL
#define MEGA_SG_SIZE
#else
#define MEGA_SG_SIZE [[sycl::reqd_sub_group_size(16)]]
#endif
// Optional entry points: a Rust build that predates them still loads the
// rest of the bridge, and one that knows them reports a bridge without them.
// Descriptors name buffers by handle; the plan resolves them to USM
// pointers once and keeps a device copy of the per-layer table.
namespace {
struct MegaVDesc { uintptr_t buf; uint64_t off; };
struct MegaWDesc { uintptr_t buf; uint64_t off; int32_t qt, qk, bsz, n, k, cols; };
struct MegaLayerDesc {
    MegaVDesc attn_norm, ffn_norm;
    MegaWDesc qkv[3];
    MegaVDesc qkv_b[3];
    MegaVDesc q_norm, k_norm;
    MegaWDesc o;
    MegaVDesc o_b;
    MegaWDesc gate, up, down;
};
struct MegaModelDesc {
    int32_t n_layers, h, n_head, n_kv, hd, n_rot, neox, ffn, vocab, max_pos, groups, pad;
    float eps, scale;
    MegaVDesc cos, sin, out_norm, out_b;
    MegaWDesc out;
};
struct MegaKvDesc { MegaVDesc k, v; int32_t cap, pad; };

struct MegaPlan {
    Handle owner;
    joshua_sycl::mega::Model model{};
    joshua_sycl::mega::Scratch scratch{};
    joshua_sycl::mega::Layer* layers_dev = nullptr;
    joshua_sycl::mega::Kv* kv_dev = nullptr;
    std::vector<joshua_sycl::mega::Kv> kv_host;
    void* block = nullptr;   // x, xn, qkv, q, ctx, act, sync
    size_t sync_bytes = 0;
    size_t groups = 0;
    int max_pos = 0;
    ~MegaPlan() {
        // Callers hold the context mutex and have drained the queue.
        auto& q = owner->queue;
        if (layers_dev) sycl::free(layers_dev, q);
        if (kv_dev) sycl::free(kv_dev, q);
        if (block) sycl::free(block, q);
        if (scratch.scores) sycl::free(scratch.scores, q);
    }
};

const float* mega_vec(Context& ctx, const MegaVDesc& d, size_t elems, bool optional, const char* what) {
    if (d.buf == 0) {
        if (optional) return nullptr;
        throw std::runtime_error(std::string("megakernel: missing ") + what);
    }
    auto& b = buffer(d.buf);
    require(b.owner.get() == &ctx, "megakernel: buffer belongs to another context");
    bounds(b, d.off, elems * 4);
    require(d.off % 4 == 0, "megakernel: misaligned f32 vector");
    return reinterpret_cast<const float*>(static_cast<char*>(b.data) + d.off);
}

joshua_sycl::mega::W mega_w(Context& ctx, const MegaWDesc& d, int n, int k, const char* what) {
    using namespace joshua_sycl;
    if (d.buf == 0) throw std::runtime_error(std::string("megakernel: missing ") + what);
    require(d.n == n && d.k == k, "megakernel: weight shape mismatch");
    require(d.qk > 0 && d.bsz > 0 && k % d.qk == 0, "megakernel: bad weight block size");
    if (d.qt != QT_F32 && d.qt != QT_F16 && d.qt != QT_BF16) {
        require(k % 32 == 0 && d.qk % 32 == 0, "megakernel: block weight inner dim");
    }
    if (d.cols > 0) {
        require(d.qk == 256 && k % 256 == 0 && WG % d.cols == 0 && d.cols <= mega::ROWS && mega::ROWS % d.cols == 0,
                "megakernel: bad multi-row layout");
    }
    auto& b = buffer(d.buf);
    require(b.owner.get() == &ctx, "megakernel: buffer belongs to another context");
    size_t bytes = (size_t)n * (size_t)(k / d.qk) * (size_t)d.bsz;
    bounds(b, d.off, bytes);
    mega::W w{};
    w.p = static_cast<const uchar*>(b.data) + d.off;
    w.qt = d.qt; w.qk = d.qk; w.bsz = d.bsz; w.n = n; w.k = k; w.cols = d.cols;
    return w;
}
}

API int joshua_sycl_mega_new(uintptr_t h, const MegaModelDesc* md, const MegaLayerDesc* ld, uintptr_t* out) noexcept {
    return guard([&] {
        using namespace joshua_sycl;
        auto owner = *reinterpret_cast<Handle*>(h);
        auto& ctx = *owner;
        std::lock_guard lock(ctx.mutex);
        const auto& d = *md;
        require(d.n_layers > 0 && d.h > 0 && d.n_head > 0 && d.n_kv > 0 && d.n_head % d.n_kv == 0,
                "megakernel: bad model shape");
        require(d.hd > 0 && d.hd <= mega::MAX_HD && d.n_rot > 0 && d.n_rot <= d.hd && d.n_rot % 2 == 0,
                "megakernel: unsupported head size");
        require(d.ffn > 0 && d.vocab > 0 && d.max_pos > 0, "megakernel: bad model shape");
        auto plan = std::make_unique<MegaPlan>();
        plan->owner = owner;
        plan->max_pos = d.max_pos;
        auto& m = plan->model;
        m.n_layers = d.n_layers; m.h = d.h; m.n_head = d.n_head; m.n_kv = d.n_kv; m.hd = d.hd;
        m.n_rot = d.n_rot; m.neox = d.neox; m.ffn = d.ffn; m.vocab = d.vocab;
        m.eps = d.eps; m.scale = d.scale;
        size_t pairs = (size_t)d.n_rot / 2;
        m.cos = mega_vec(ctx, d.cos, (size_t)d.max_pos * pairs, false, "rope cos");
        m.sin = mega_vec(ctx, d.sin, (size_t)d.max_pos * pairs, false, "rope sin");
        m.out_norm = mega_vec(ctx, d.out_norm, d.h, false, "output norm");
        m.out = mega_w(ctx, d.out, d.vocab, d.h, "output");
        m.out_b = mega_vec(ctx, d.out_b, d.vocab, true, "output bias");
        int q_dim = d.n_head * d.hd, kv_dim = d.n_kv * d.hd;
        std::vector<mega::Layer> layers(d.n_layers);
        for (int l = 0; l < d.n_layers; l++) {
            const auto& s = ld[l];
            auto& L = layers[l];
            L.attn_norm = mega_vec(ctx, s.attn_norm, d.h, false, "attention norm");
            L.ffn_norm = mega_vec(ctx, s.ffn_norm, d.h, false, "ffn norm");
            int rows[3] = {q_dim, kv_dim, kv_dim};
            for (int g = 0; g < 3; g++) {
                L.qkv[g] = mega_w(ctx, s.qkv[g], rows[g], d.h, "q/k/v");
                L.qkv_b[g] = mega_vec(ctx, s.qkv_b[g], rows[g], true, "q/k/v bias");
            }
            L.q_norm = mega_vec(ctx, s.q_norm, d.hd, true, "q norm");
            L.k_norm = mega_vec(ctx, s.k_norm, d.hd, true, "k norm");
            L.o = mega_w(ctx, s.o, d.h, q_dim, "attention output");
            L.o_b = mega_vec(ctx, s.o_b, d.h, true, "attention output bias");
            L.gate = mega_w(ctx, s.gate, d.ffn, d.h, "ffn gate");
            L.up = mega_w(ctx, s.up, d.ffn, d.h, "ffn up");
            L.down = mega_w(ctx, s.down, d.h, d.ffn, "ffn down");
        }
        auto& q = ctx.queue;
        // One work-group per compute unit by default: more only add
        // spinning waiters (measured on the Arc Pro B50).
        size_t groups = d.groups > 0 ? (size_t)d.groups
            : (size_t)q.get_device().get_info<sycl::info::device::max_compute_units>();
        groups = std::max<size_t>(1, groups);
        // Rows per task: about one task per work-group for each stage.
        auto per_task = [&](int rows) {
            size_t r = ((size_t)rows + groups - 1) / groups;
            return (int)((r + mega::ROWS - 1) / mega::ROWS * mega::ROWS);
        };
        auto tasks = [](int rows, int rpt) { return (rows + rpt - 1) / rpt; };
        int qkv_rpt = per_task(q_dim + 2 * kv_dim);
        for (int i = 0; i < mega::S_LAYER; i++) m.rpt[i] = mega::ROWS;
        m.rpt[mega::S_QKV] = qkv_rpt;
        m.rpt[mega::S_OPROJ] = per_task(d.h);
        m.rpt[mega::S_GATEUP] = per_task(d.ffn);
        m.rpt[mega::S_DOWN] = per_task(d.h);
        m.head_rpt = per_task(d.vocab);
        m.count[mega::S_NORM1] = 1;
        m.count[mega::S_QKV] = tasks(q_dim, qkv_rpt) + 2 * tasks(kv_dim, qkv_rpt);
        m.count[mega::S_PREP] = d.n_head + d.n_kv;
        m.count[mega::S_ATTN] = d.n_head;
        m.count[mega::S_OPROJ] = tasks(d.h, m.rpt[mega::S_OPROJ]);
        m.count[mega::S_NORM2] = 1;
        m.count[mega::S_GATEUP] = tasks(d.ffn, m.rpt[mega::S_GATEUP]);
        m.count[mega::S_DOWN] = tasks(d.h, m.rpt[mega::S_DOWN]);
        m.start[0] = 0;
        for (int i = 0; i < mega::S_LAYER; i++) m.start[i + 1] = m.start[i] + m.count[i];
        m.head_tasks = tasks(d.vocab, m.head_rpt);

        plan->layers_dev = sycl::malloc_device<mega::Layer>(layers.size(), q);
        require(plan->layers_dev != nullptr, "megakernel: device allocation failed");
        q.memcpy(plan->layers_dev, layers.data(), layers.size() * sizeof(mega::Layer)).wait_and_throw();
        m.layers = plan->layers_dev;
        plan->kv_dev = sycl::malloc_device<mega::Kv>(d.n_layers, q);
        require(plan->kv_dev != nullptr, "megakernel: device allocation failed");

        size_t floats = (size_t)d.h * 2 + (size_t)(q_dim + 2 * kv_dim) + (size_t)q_dim * 2 + (size_t)d.ffn;
        size_t stages = (size_t)d.n_layers * mega::S_LAYER + 2;
        plan->sync_bytes = (stages + 1) * mega::SYNC_STRIDE * sizeof(uint32_t);
        // Counters first, so each starts a cache line.
        plan->block = sycl::aligned_alloc_device(256, plan->sync_bytes + floats * 4, q);
        require(plan->block != nullptr, "megakernel: device allocation failed");
        auto& sc = plan->scratch;
        sc.sync = static_cast<uint32_t*>(plan->block);
        float* f = reinterpret_cast<float*>(static_cast<char*>(plan->block) + plan->sync_bytes);
        sc.x = f; f += d.h;
        sc.xn = f; f += d.h;
        sc.qkv = f; f += q_dim + 2 * kv_dim;
        sc.q = f; f += q_dim;
        sc.ctx = f; f += q_dim;
        sc.act = f; f += d.ffn;
        sc.scores = nullptr; sc.scores_cap = 0;

        size_t total = (size_t)d.n_layers * m.start[mega::S_LAYER] + 1 + m.head_tasks;
        plan->groups = std::max<size_t>(1, std::min(groups, total));
        *out = reinterpret_cast<uintptr_t>(plan.release());
    });
}

API int joshua_sycl_mega_free(uintptr_t p) noexcept {
    return guard([&] {
        std::unique_ptr<MegaPlan> plan(reinterpret_cast<MegaPlan*>(p));
        auto owner = plan->owner;
        std::lock_guard lock(owner->mutex);
        owner->wait();
        plan.reset();
    });
}

API int joshua_sycl_mega_groups(uintptr_t p) noexcept {
    return static_cast<int>(reinterpret_cast<MegaPlan*>(p)->groups);
}

API int joshua_sycl_mega_decode(uintptr_t h, uintptr_t p, const MegaKvDesc* kv, size_t n_kv, int32_t pos,
                                MegaVDesc x_in, MegaVDesc logits) noexcept {
    return guard([&] {
        using namespace joshua_sycl;
        auto& ctx = context(h);
        auto& plan = *reinterpret_cast<MegaPlan*>(p);
        require(plan.owner.get() == &ctx, "megakernel: plan belongs to another context");
        std::lock_guard lock(ctx.mutex);
        const auto& m = plan.model;
        require(n_kv == (size_t)m.n_layers, "megakernel: one KV cache per layer");
        require(pos >= 0 && pos < plan.max_pos, "megakernel: position past the RoPE table");
        std::vector<mega::Kv> table(m.n_layers);
        size_t kv_elems_per_row = (size_t)m.n_kv * m.hd;
        for (int l = 0; l < m.n_layers; l++) {
            require(kv[l].cap > pos, "megakernel: KV cache has no room for this position");
            size_t elems = kv_elems_per_row * (size_t)kv[l].cap;
            table[l].k = const_cast<float*>(mega_vec(ctx, kv[l].k, elems, false, "key cache"));
            table[l].v = const_cast<float*>(mega_vec(ctx, kv[l].v, elems, false, "value cache"));
            table[l].cap = kv[l].cap;
        }
        bool same = plan.kv_host.size() == table.size();
        for (size_t l = 0; same && l < table.size(); l++) {
            same = table[l].k == plan.kv_host[l].k && table[l].v == plan.kv_host[l].v && table[l].cap == plan.kv_host[l].cap;
        }
        if (!same) {
            // Only when a cache grew or another session runs: the table is
            // read by kernels already queued, so drain them first.
            ctx.wait();
            ctx.queue.memcpy(plan.kv_dev, table.data(), table.size() * sizeof(mega::Kv)).wait_and_throw();
            plan.kv_host = table;
        }
        if (plan.scratch.scores_cap <= pos) {
            int cap = std::max({pos + 1, plan.scratch.scores_cap * 2, 256});
            ctx.wait();
            if (plan.scratch.scores) sycl::free(plan.scratch.scores, ctx.queue);
            plan.scratch.scores = sycl::malloc_device<float>((size_t)cap * m.n_head, ctx.queue);
            require(plan.scratch.scores != nullptr, "megakernel: device allocation failed");
            plan.scratch.scores_cap = cap;
        }
        const float* xin = mega_vec(ctx, x_in, m.h, false, "input");
        float* out = const_cast<float*>(mega_vec(ctx, logits, m.vocab, false, "logits"));
        ctx.queue.memset(plan.scratch.sync, 0, plan.sync_bytes);
        auto model = m;
        auto scratch = plan.scratch;
        auto* kv_dev = plan.kv_dev;
        size_t groups = plan.groups;
        ctx.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> local(sycl::range<1>(mega::LOCAL_FLOATS), cgh);
            cgh.parallel_for(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) MEGA_SG_SIZE {
                mega::Ctx c{it, local.template get_multi_ptr<sycl::access::decorated::no>().get()};
                mega::run(c, model, scratch, kv_dev, xin, out, pos);
            });
        });
    });
}
