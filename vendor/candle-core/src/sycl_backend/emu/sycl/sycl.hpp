// Host emulation of the slice of SYCL 2020 that the joshua SYCL bridge
// (bridge.cpp, kernels.hpp, mega.hpp) uses, so the bridge can be built with
// a plain C++20 compiler and its kernels run on the CPU:
//
//   c++ -std=c++20 -O2 -fPIC -shared -Iemu bridge.cpp -o libjoshua_sycl.so
//
// A testing aid, not a backend: one "device", USM is host memory, every
// kernel runs synchronously at submit.  Each work-group's work-items are
// real threads (barriers block), and the work-groups of one launch run
// `JOSHUA_SYCL_EMU_GROUPS` at a time (default 3) on a shared atomic ticket,
// so a persistent kernel whose groups wait on one another (mega.hpp) runs
// with genuinely concurrent groups.  Memory ordering is the host's, which is
// stronger than a GPU's: missing fences will not show up here.
#pragma once
#include <algorithm>
#include <atomic>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace sycl {

// ─── Scalars / math ──────────────────────────────────────────────────────────
using half = _Float16;
template<class To, class From> To bit_cast(const From& f) { return std::bit_cast<To>(f); }
using std::exp; using std::log; using std::sin; using std::cos; using std::tanh; using std::sqrt;
using std::erf; using std::fabs; using std::ceil; using std::floor; using std::round; using std::pow;
using std::fmax; using std::fmin; using std::isnan;
inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
template<class T> T min(T a, T b) { return a < b ? a : b; }
template<class T> T max(T a, T b) { return a > b ? a : b; }
namespace native {
inline float exp(float x) { return std::exp(x); }
inline float log(float x) { return std::log(x); }
inline float sin(float x) { return std::sin(x); }
inline float cos(float x) { return std::cos(x); }
inline float sqrt(float x) { return std::sqrt(x); }
inline float recip(float x) { return 1.0f / x; }
}

template<class T, int N> struct vec {
    T v[N];
    vec() = default;
    template<class... A, class = std::enable_if_t<sizeof...(A) == N>>
    vec(A... a) : v{static_cast<T>(a)...} {}
    vec<T, N / 2> lo() const { vec<T, N / 2> r; for (int i = 0; i < N / 2; i++) r.v[i] = v[i]; return r; }
    vec<T, N / 2> hi() const { vec<T, N / 2> r; for (int i = 0; i < N / 2; i++) r.v[i] = v[N / 2 + i]; return r; }
    T x() const { return v[0]; } T y() const { return v[1]; } T z() const { return v[2]; } T w() const { return v[3]; }
    T s0() const { return v[0]; } T s1() const { return v[1]; } T s2() const { return v[2]; } T s3() const { return v[3]; }
    T s4() const { return v[4]; } T s5() const { return v[5]; } T s6() const { return v[6]; } T s7() const { return v[7]; }
};
template<class T, int N> T dot(const vec<T, N>& a, const vec<T, N>& b) {
    T r = 0;
    for (int i = 0; i < N; i++) r += a.v[i] * b.v[i];
    return r;
}

// ─── Memory model ────────────────────────────────────────────────────────────
enum class memory_order { relaxed, acquire, release, acq_rel, seq_cst };
enum class memory_scope { work_item, sub_group, work_group, device, system };
namespace access { enum class address_space { global_space, local_space }; enum class decorated { no, yes }; }
inline std::memory_order to_std(memory_order o) {
    switch (o) {
    case memory_order::relaxed: return std::memory_order_relaxed;
    case memory_order::acquire: return std::memory_order_acquire;
    case memory_order::release: return std::memory_order_release;
    case memory_order::acq_rel: return std::memory_order_acq_rel;
    default: return std::memory_order_seq_cst;
    }
}
inline void atomic_fence(memory_order o, memory_scope) { std::atomic_thread_fence(to_std(o)); }
template<class T, memory_order D, memory_scope S, access::address_space A> struct atomic_ref {
    std::atomic_ref<T> r;
    explicit atomic_ref(T& t) : r(t) {}
    T load(memory_order o = D) const { return r.load(to_std(o)); }
    T fetch_add(T v, memory_order o = D) const { return r.fetch_add(v, to_std(o)); }
};

// ─── Work-items and groups ───────────────────────────────────────────────────
template<int D> struct range {
    size_t r[D];
    template<class... A> range(A... a) : r{static_cast<size_t>(a)...} {}
    size_t operator[](int i) const { return r[i]; }
    size_t size() const { size_t s = 1; for (int i = 0; i < D; i++) s *= r[i]; return s; }
};
template<int D> struct nd_range {
    range<D> global, local;
    nd_range(range<D> g, range<D> l) : global(g), local(l) {}
};
struct group_state {
    std::barrier<>* barrier;
    std::vector<uint64_t>* bcast;
};
template<int D> struct group {
    group_state* st;
};
// Sub-groups of EMU_SG consecutive work-items.  Collectives on them are
// emulated with work-group barriers, so every sub-group of a work-group must
// reach each one (true of the kernels that use them).
constexpr size_t EMU_SG = 16;
struct sub_group {
    group_state* st;
    size_t lin, size;
    size_t get_local_linear_id() const { return lin % EMU_SG; }
    size_t get_local_linear_range() const { return EMU_SG; }
    size_t get_group_linear_id() const { return lin / EMU_SG; }
    size_t get_group_linear_range() const { return (size + EMU_SG - 1) / EMU_SG; }
};
template<class T = void> struct plus { T operator()(T a, T b) const { return a + b; } };
template<int D> struct nd_item {
    size_t gid[D], lid[D], grp[D], lrange[D];
    group_state* st;
    sub_group get_sub_group() const {
        size_t n = 1;
        for (int i = 0; i < D; i++) n *= lrange[i];
        return sub_group{st, get_local_linear_id(), n};
    }
    size_t get_local_id(int d) const { return lid[d]; }
    size_t get_global_id(int d) const { return gid[d]; }
    size_t get_group(int d) const { return grp[d]; }
    group<D> get_group() const { return group<D>{st}; }
    size_t get_local_linear_id() const {
        size_t r = 0;
        for (int i = 0; i < D; i++) r = r * lrange[i] + lid[i];
        return r;
    }
};
extern thread_local size_t emu_linear_id;
template<int D> void group_barrier(group<D> g) { g.st->barrier->arrive_and_wait(); }
template<int D, class T> T group_broadcast(group<D> g, T v, size_t from) {
    // Every work-item stores into its own slot; all read `from`'s.
    std::memcpy(&(*g.st->bcast)[emu_linear_id], &v, sizeof(T));
    g.st->barrier->arrive_and_wait();
    T r;
    std::memcpy(&r, &(*g.st->bcast)[from], sizeof(T));
    g.st->barrier->arrive_and_wait();
    return r;
}

template<class T, class Op> T reduce_over_group(sub_group g, T v, Op op) {
    std::memcpy(&(*g.st->bcast)[emu_linear_id], &v, sizeof(T));
    g.st->barrier->arrive_and_wait();
    size_t base = emu_linear_id / EMU_SG * EMU_SG, end = std::min(base + EMU_SG, g.size);
    T r;
    std::memcpy(&r, &(*g.st->bcast)[base], sizeof(T));
    for (size_t i = base + 1; i < end; i++) {
        T x;
        std::memcpy(&x, &(*g.st->bcast)[i], sizeof(T));
        r = op(r, x);
    }
    g.st->barrier->arrive_and_wait();
    return r;
}

// ─── Devices, queues, USM ────────────────────────────────────────────────────
enum class aspect { usm_device_allocations };
namespace info {
namespace device {
struct name {}; struct global_mem_size {}; struct max_work_group_size {}; struct local_mem_size {};
struct max_compute_units {};
template<int D> struct max_work_item_sizes {};
}
}
template<class P> struct info_ret;
template<> struct info_ret<info::device::name> { using type = std::string; static type get() { return "joshua SYCL host emulator"; } };
template<> struct info_ret<info::device::global_mem_size> { using type = uint64_t; static type get() { return 8ull << 30; } };
template<> struct info_ret<info::device::max_work_group_size> { using type = size_t; static type get() { return 1024; } };
template<> struct info_ret<info::device::local_mem_size> { using type = uint64_t; static type get() { return 64 << 10; } };
template<> struct info_ret<info::device::max_compute_units> { using type = uint32_t; static type get() { return 2; } };
template<> struct info_ret<info::device::max_work_item_sizes<3>> {
    using type = range<3>; static type get() { return range<3>(1024, 1024, 1024); }
};

struct device {
    static std::vector<device> get_devices() { return {device{}}; }
    bool is_gpu() const { return true; }
    bool has(aspect) const { return true; }
    template<class P> typename info_ret<P>::type get_info() const { return info_ret<P>::get(); }
};

struct exception_list {
    std::vector<std::exception_ptr> v;
    auto begin() const { return v.begin(); }
    auto end() const { return v.end(); }
};
namespace property { namespace queue { struct in_order {}; } }

struct event {
    void wait() {}
    void wait_and_throw() {}
};

struct handler;
struct queue {
    device dev;
    template<class H, class P> queue(device d, H, P) : dev(d) {}
    device get_device() const { return dev; }
    template<class F> event submit(F&& f);
    event memcpy(void* d, const void* s, size_t n) { std::memmove(d, s, n); return {}; }
    event memset(void* d, int v, size_t n) { std::memset(d, v, n); return {}; }
    void wait() {}
    void wait_and_throw() {}
};
inline void* malloc_device(size_t n, const queue&) { return std::calloc(1, n ? n : 1); }
template<class T> T* malloc_device(size_t n, const queue& q) { return static_cast<T*>(malloc_device(n * sizeof(T), q)); }
inline void* aligned_alloc_device(size_t align, size_t n, const queue&) { void* p = std::aligned_alloc(align, (n + align - 1) / align * align); if (p) std::memset(p, 0, n); return p; }
inline void free(void* p, const queue&) { std::free(p); }

// ─── Kernels ─────────────────────────────────────────────────────────────────
// Each local_accessor is an index into the running work-group's local
// buffers, which the executor installs per work-item thread.
extern thread_local std::vector<std::vector<uint64_t>>* emu_locals;

template<class T, int D> struct local_accessor {
    size_t index;
    local_accessor(range<1> r, handler& h);
    template<access::decorated> struct mp {
        T* p;
        T* get() const { return p; }
    };
    template<access::decorated Dc> mp<Dc> get_multi_ptr() const {
        return mp<Dc>{reinterpret_cast<T*>((*emu_locals)[index].data())};
    }
};

struct handler {
    std::vector<size_t> local_bytes;
    template<int D, class K> void parallel_for(nd_range<D> r, K k);
};

template<class T, int D> local_accessor<T, D>::local_accessor(range<1> r, handler& h) {
    index = h.local_bytes.size();
    h.local_bytes.push_back(r[0] * sizeof(T));
}

void emu_run(size_t groups, size_t group_size,
             const std::vector<size_t>& local_bytes,
             const std::function<void(size_t group, size_t item, group_state*)>& body);

template<int D, class K> void handler::parallel_for(nd_range<D> r, K k) {
    size_t gsize = r.local.size();
    size_t ngroups = r.global.size() / gsize;
    size_t gcount[D];
    for (int i = 0; i < D; i++) gcount[i] = r.global[i] / r.local[i];
    emu_run(ngroups, gsize, local_bytes, [&](size_t g, size_t li, group_state* st) {
        nd_item<D> it{};
        it.st = st;
        // Row-major decomposition, dimension D-1 fastest (as SYCL linearises).
        size_t gg = g, ll = li;
        for (int i = D - 1; i >= 0; i--) {
            it.grp[i] = gg % gcount[i]; gg /= gcount[i];
            it.lid[i] = ll % r.local[i]; ll /= r.local[i];
            it.lrange[i] = r.local[i];
            it.gid[i] = it.grp[i] * r.local[i] + it.lid[i];
        }
        k(it);
    });
}

template<class F> event queue::submit(F&& f) {
    handler h;
    f(h);
    return {};
}

} // namespace sycl

#ifdef JOSHUA_SYCL_EMU_IMPL
namespace sycl {
thread_local std::vector<std::vector<uint64_t>>* emu_locals = nullptr;
thread_local size_t emu_linear_id = 0;

void emu_run(size_t groups, size_t group_size,
             const std::vector<size_t>& local_bytes,
             const std::function<void(size_t, size_t, group_state*)>& body) {
    static const size_t slots = [] {
        const char* v = std::getenv("JOSHUA_SYCL_EMU_GROUPS");
        long n = v ? std::atol(v) : 3;
        return (size_t)std::max(1L, n);
    }();
    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    std::exception_ptr error;
    std::mutex error_mutex;
    size_t nslots = std::min(slots, groups);
    std::vector<std::unique_ptr<std::barrier<>>> barriers;
    for (size_t s = 0; s < nslots; s++) barriers.push_back(std::make_unique<std::barrier<>>(group_size));
    for (size_t s = 0; s < nslots; s++) {
        // One slot: group_size threads that run work-groups one after another.
        auto shared = std::make_shared<std::vector<std::vector<uint64_t>>>();
        for (size_t b : local_bytes) shared->emplace_back((b + 7) / 8);
        auto bcast = std::make_shared<std::vector<uint64_t>>(group_size);
        auto current = std::make_shared<std::atomic<size_t>>(0);
        auto* bar = barriers[s].get();
        for (size_t li = 0; li < group_size; li++) {
            threads.emplace_back([&, li, shared, bcast, current, bar] {
                emu_locals = shared.get();
                emu_linear_id = li;
                group_state st{bar, bcast.get()};
                for (;;) {
                    if (li == 0) current->store(next.fetch_add(1));
                    bar->arrive_and_wait();
                    size_t g = current->load();
                    bar->arrive_and_wait();
                    if (g >= groups) return;
                    try { body(g, li, &st); }
                    catch (...) { std::lock_guard l(error_mutex); error = std::current_exception(); }
                    bar->arrive_and_wait();
                }
            });
        }
    }
    for (auto& t : threads) t.join();
    if (error) std::rethrow_exception(error);
}
}
#endif
