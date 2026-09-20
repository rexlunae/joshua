// C ABI boundary for Rust. All exceptions are caught here; no C++ object or
// standard-library type crosses the boundary. ABI version changes on any
// incompatible change to argument layouts or kernel dispatch.
#include "kernels.hpp"
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
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> int guard(F&& f) noexcept {
    try { f(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); }
    catch (...) { last_error = "unknown SYCL exception"; }
    return -1;
}
struct Context {
    sycl::queue queue;
    std::mutex mutex;
    std::vector<std::pair<void*, size_t>> retired;
    size_t retired_bytes = 0;
    size_t pending = 0;
    explicit Context(sycl::device dev) : queue(dev, [](sycl::exception_list errors) {
        for (auto error : errors) std::rethrow_exception(error);
    }, sycl::property::queue::in_order{}) {}
    void wait() {
        queue.wait_and_throw();
        for (auto [p, size] : retired) sycl::free(p, queue);
        retired.clear(); retired_bytes = 0; pending = 0;
    }
    template<class F> void submit(F&& f) {
        queue.submit(std::forward<F>(f));
        if (++pending >= 256) wait();
    }
    ~Context() {
        // Never free live device memory, even during unwinding.
        try { wait(); } catch (...) { queue.wait(); }
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
        require(ordinal < devices.size(), "SYCL device ordinal out of range (or no device available)");
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
