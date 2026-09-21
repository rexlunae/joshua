// Typed dispatch for kernels.hpp. The Rust launchers supply arguments in this order.
    if (name == "k_unary_c") {
        require(count == 5, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_unary_c(a0, a1, a2, a3, a4);
            });
        });
        return;
    }
    if (name == "k_unary_s") {
        require(count == 5, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_unary_s(a0, a1, a2, a3, a4);
            });
        });
        return;
    }
    if (name == "k_affine_c") {
        require(count == 6, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<float>(args, count, 4);
        auto a5 = arg<float>(args, count, 5);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_affine_c(a0, a1, a2, a3, a4, a5);
            });
        });
        return;
    }
    if (name == "k_affine_s") {
        require(count == 6, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        auto a4 = arg<float>(args, count, 4);
        auto a5 = arg<float>(args, count, 5);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_affine_s(a0, a1, a2, a3, a4, a5);
            });
        });
        return;
    }
    if (name == "k_powf_s") {
        require(count == 5, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        auto a4 = arg<float>(args, count, 4);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_powf_s(a0, a1, a2, a3, a4);
            });
        });
        return;
    }
    if (name == "k_elu_s") {
        require(count == 5, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        auto a4 = arg<float>(args, count, 4);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_elu_s(a0, a1, a2, a3, a4);
            });
        });
        return;
    }
    if (name == "k_binary_c") {
        require(count == 7, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<const float*>(args, count, 1);
        auto a2 = arg<float*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_binary_c(a0, a1, a2, a3, a4, a5, a6);
            });
        });
        return;
    }
    if (name == "k_binary_s") {
        require(count == 6, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<const float*>(args, count, 1);
        auto a2 = arg<float*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<Idx>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_binary_s(a0, a1, a2, a3, a4, a5);
            });
        });
        return;
    }
    if (name == "k_binary_u32_s") {
        require(count == 6, "wrong kernel argument count");
        auto a0 = arg<const uint*>(args, count, 0);
        auto a1 = arg<const uint*>(args, count, 1);
        auto a2 = arg<uint*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<Idx>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_binary_u32_s(a0, a1, a2, a3, a4, a5);
            });
        });
        return;
    }
    if (name == "k_cmp_f32") {
        require(count == 6, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<const float*>(args, count, 1);
        auto a2 = arg<uchar*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<Idx>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cmp_f32(a0, a1, a2, a3, a4, a5);
            });
        });
        return;
    }
    if (name == "k_cmp_u32") {
        require(count == 6, "wrong kernel argument count");
        auto a0 = arg<const uint*>(args, count, 0);
        auto a1 = arg<const uint*>(args, count, 1);
        auto a2 = arg<uchar*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<Idx>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cmp_u32(a0, a1, a2, a3, a4, a5);
            });
        });
        return;
    }
    if (name == "k_where_4") {
        require(count == 6, "wrong kernel argument count");
        auto a0 = arg<const uchar*>(args, count, 0);
        auto a1 = arg<const uint*>(args, count, 1);
        auto a2 = arg<const uint*>(args, count, 2);
        auto a3 = arg<uint*>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<Idx>(args, count, 5);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_where_4(a0, a1, a2, a3, a4, a5);
            });
        });
        return;
    }
    if (name == "k_where_8") {
        require(count == 6, "wrong kernel argument count");
        auto a0 = arg<const uchar*>(args, count, 0);
        auto a1 = arg<const uint64_t*>(args, count, 1);
        auto a2 = arg<const uint64_t*>(args, count, 2);
        auto a3 = arg<uint64_t*>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<Idx>(args, count, 5);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_where_8(a0, a1, a2, a3, a4, a5);
            });
        });
        return;
    }
    if (name == "k_where_u32c_4") {
        require(count == 6, "wrong kernel argument count");
        auto a0 = arg<const uint*>(args, count, 0);
        auto a1 = arg<const uint*>(args, count, 1);
        auto a2 = arg<const uint*>(args, count, 2);
        auto a3 = arg<uint*>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<Idx>(args, count, 5);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_where_u32c_4(a0, a1, a2, a3, a4, a5);
            });
        });
        return;
    }
    if (name == "k_copy_s1") {
        require(count == 5, "wrong kernel argument count");
        auto a0 = arg<const uchar*>(args, count, 0);
        auto a1 = arg<uchar*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_copy_s1(a0, a1, a2, a3, a4);
            });
        });
        return;
    }
    if (name == "k_copy_s2") {
        require(count == 5, "wrong kernel argument count");
        auto a0 = arg<const ushort*>(args, count, 0);
        auto a1 = arg<ushort*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_copy_s2(a0, a1, a2, a3, a4);
            });
        });
        return;
    }
    if (name == "k_copy_s4") {
        require(count == 5, "wrong kernel argument count");
        auto a0 = arg<const uint*>(args, count, 0);
        auto a1 = arg<uint*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_copy_s4(a0, a1, a2, a3, a4);
            });
        });
        return;
    }
    if (name == "k_copy_s8") {
        require(count == 5, "wrong kernel argument count");
        auto a0 = arg<const uint64_t*>(args, count, 0);
        auto a1 = arg<uint64_t*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_copy_s8(a0, a1, a2, a3, a4);
            });
        });
        return;
    }
    if (name == "k_copy2d_4") {
        require(count == 8, "wrong kernel argument count");
        auto a0 = arg<const uint*>(args, count, 0);
        auto a1 = arg<uint*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_copy2d_4(a0, a1, a2, a3, a4, a5, a6, a7);
            });
        });
        return;
    }
    if (name == "k_copy2d_8") {
        require(count == 8, "wrong kernel argument count");
        auto a0 = arg<const uint64_t*>(args, count, 0);
        auto a1 = arg<uint64_t*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_copy2d_8(a0, a1, a2, a3, a4, a5, a6, a7);
            });
        });
        return;
    }
    if (name == "k_copy2d_1") {
        require(count == 8, "wrong kernel argument count");
        auto a0 = arg<const uchar*>(args, count, 0);
        auto a1 = arg<uchar*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_copy2d_1(a0, a1, a2, a3, a4, a5, a6, a7);
            });
        });
        return;
    }
    if (name == "k_copy2d_2") {
        require(count == 8, "wrong kernel argument count");
        auto a0 = arg<const ushort*>(args, count, 0);
        auto a1 = arg<ushort*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_copy2d_2(a0, a1, a2, a3, a4, a5, a6, a7);
            });
        });
        return;
    }
    if (name == "k_fill_4") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<uint*>(args, count, 0);
        auto a1 = arg<int>(args, count, 1);
        auto a2 = arg<Idx>(args, count, 2);
        auto a3 = arg<uint>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_fill_4(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_fill_8") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<uint64_t*>(args, count, 0);
        auto a1 = arg<int>(args, count, 1);
        auto a2 = arg<Idx>(args, count, 2);
        auto a3 = arg<uint64_t>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_fill_8(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_fill_1") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<uchar*>(args, count, 0);
        auto a1 = arg<int>(args, count, 1);
        auto a2 = arg<Idx>(args, count, 2);
        auto a3 = arg<uchar>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_fill_1(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_fill_2") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<ushort*>(args, count, 0);
        auto a1 = arg<int>(args, count, 1);
        auto a2 = arg<Idx>(args, count, 2);
        auto a3 = arg<ushort>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_fill_2(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_f32_u32") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<uint*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_f32_u32(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_u32_f32") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const uint*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_u32_f32(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_f32_u8") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<uchar*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_f32_u8(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_u8_f32") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const uchar*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_u8_f32(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_f32_i64") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<int64_t*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_f32_i64(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_i64_f32") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const int64_t*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_i64_f32(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_u32_i64") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const uint*>(args, count, 0);
        auto a1 = arg<int64_t*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_u32_i64(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_i64_u32") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const int64_t*>(args, count, 0);
        auto a1 = arg<uint*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_i64_u32(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_ids_i64") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const int64_t*>(args, count, 0);
        auto a1 = arg<uint*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_ids_i64(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_u8_u32") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const uchar*>(args, count, 0);
        auto a1 = arg<uint*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_u8_u32(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_u32_u8") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const uint*>(args, count, 0);
        auto a1 = arg<uchar*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_u32_u8(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_f32_f16") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<half*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_f32_f16(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_f16_f32") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const half*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_f16_f32(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_i32_f32") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const int*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_i32_f32(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_f32_i32") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<int*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_f32_i32(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_u32_i32") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const uint*>(args, count, 0);
        auto a1 = arg<int*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_u32_i32(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_cast_i32_u32") {
        require(count == 4, "wrong kernel argument count");
        auto a0 = arg<const int*>(args, count, 0);
        auto a1 = arg<uint*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_cast_i32_u32(a0, a1, a2, a3);
            });
        });
        return;
    }
    if (name == "k_reduce_last") {
        require(count == 6, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((sizeof(float) * (WG) + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_reduce_last(a0, a1, a2, a3, a4, a5);
            });
        });
        return;
    }
    if (name == "k_reduce_generic") {
        require(count == 7, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<Idx>(args, count, 3);
        auto a4 = arg<Idx>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_reduce_generic(a0, a1, a2, a3, a4, a5, a6);
            });
        });
        return;
    }
    if (name == "k_arg_last") {
        require(count == 6, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<uint*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((sizeof(float) * (WG) + sizeof(uint) * (WG) + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_arg_last(a0, a1, a2, a3, a4, a5);
            });
        });
        return;
    }
    if (name == "k_index_select_u32_4") {
        require(count == 12, "wrong kernel argument count");
        auto a0 = arg<const uint*>(args, count, 0);
        auto a1 = arg<const uint*>(args, count, 1);
        auto a2 = arg<uint*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<uint*>(args, count, 10);
        auto a11 = arg<int>(args, count, 11);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_index_select_u32_4(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            });
        });
        return;
    }
    if (name == "k_index_select_i64_4") {
        require(count == 12, "wrong kernel argument count");
        auto a0 = arg<const uint*>(args, count, 0);
        auto a1 = arg<const int64_t*>(args, count, 1);
        auto a2 = arg<uint*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<uint*>(args, count, 10);
        auto a11 = arg<int>(args, count, 11);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_index_select_i64_4(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            });
        });
        return;
    }
    if (name == "k_index_select_u32_8") {
        require(count == 12, "wrong kernel argument count");
        auto a0 = arg<const uint64_t*>(args, count, 0);
        auto a1 = arg<const uint*>(args, count, 1);
        auto a2 = arg<uint64_t*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<uint*>(args, count, 10);
        auto a11 = arg<int>(args, count, 11);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_index_select_u32_8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            });
        });
        return;
    }
    if (name == "k_gather_4") {
        require(count == 9, "wrong kernel argument count");
        auto a0 = arg<const uint*>(args, count, 0);
        auto a1 = arg<const uint*>(args, count, 1);
        auto a2 = arg<uint*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<Idx>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<uint*>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_gather_4(a0, a1, a2, a3, a4, a5, a6, a7, a8);
            });
        });
        return;
    }
    if (name == "k_scatter_set_4") {
        require(count == 12, "wrong kernel argument count");
        auto a0 = arg<uint*>(args, count, 0);
        auto a1 = arg<const uint*>(args, count, 1);
        auto a2 = arg<const uint*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<Idx>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<uint*>(args, count, 10);
        auto a11 = arg<int>(args, count, 11);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_scatter_set_4(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            });
        });
        return;
    }
    if (name == "k_scatter_add_f32") {
        require(count == 12, "wrong kernel argument count");
        auto a0 = arg<float*>(args, count, 0);
        auto a1 = arg<const uint*>(args, count, 1);
        auto a2 = arg<const float*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<Idx>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<uint*>(args, count, 10);
        auto a11 = arg<int>(args, count, 11);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_scatter_add_f32(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            });
        });
        return;
    }
    if (name == "k_index_add_f32") {
        require(count == 12, "wrong kernel argument count");
        auto a0 = arg<float*>(args, count, 0);
        auto a1 = arg<const uint*>(args, count, 1);
        auto a2 = arg<const float*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<uint*>(args, count, 10);
        auto a11 = arg<int>(args, count, 11);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_index_add_f32(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            });
        });
        return;
    }
    if (name == "k_gemm") {
        require(count == 17, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<const float*>(args, count, 1);
        auto a2 = arg<float*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<int>(args, count, 10);
        auto a11 = arg<int>(args, count, 11);
        auto a12 = arg<int>(args, count, 12);
        auto a13 = arg<int>(args, count, 13);
        auto a14 = arg<int>(args, count, 14);
        auto a15 = arg<int>(args, count, 15);
        auto a16 = arg<int>(args, count, 16);
        ctx.submit([&](sycl::handler& h) {
            // Local tile buffers: As is [TK][TM+1] floats at offset 0, Bs is
            // [TK][TN+1] floats after it.  Allocate the full padded span so Bs
            // does not overlap As (see kernels.hpp k_gemm).
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((sizeof(float) * TK * ((TM + 1) + (TN + 1)) + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_gemm(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16);
            });
        });
        return;
    }
    if (name == "k_gemv_nt") {
        require(count == 12, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<const float*>(args, count, 1);
        auto a2 = arg<float*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<int>(args, count, 10);
        auto a11 = arg<int>(args, count, 11);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((sizeof(float) * (WG) + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_gemv_nt(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            });
        });
        return;
    }
    if (name == "k_gemv_nn") {
        require(count == 12, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<const float*>(args, count, 1);
        auto a2 = arg<float*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<int>(args, count, 10);
        auto a11 = arg<int>(args, count, 11);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_gemv_nn(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            });
        });
        return;
    }
    if (name == "k_softmax_last") {
        require(count == 5, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((sizeof(float) * (WG) + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_softmax_last(a0, a1, a2, a3, a4);
            });
        });
        return;
    }
    if (name == "k_rmsnorm") {
        require(count == 8, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<const float*>(args, count, 1);
        auto a2 = arg<float*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<float>(args, count, 7);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((sizeof(float) * (WG) + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_rmsnorm(a0, a1, a2, a3, a4, a5, a6, a7);
            });
        });
        return;
    }
    if (name == "k_rope_i") {
        require(count == 12, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<const float*>(args, count, 1);
        auto a2 = arg<const float*>(args, count, 2);
        auto a3 = arg<float*>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<int>(args, count, 10);
        auto a11 = arg<int>(args, count, 11);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_rope_i(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            });
        });
        return;
    }
    if (name == "k_rope") {
        require(count == 12, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<const float*>(args, count, 1);
        auto a2 = arg<const float*>(args, count, 2);
        auto a3 = arg<float*>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<int>(args, count, 10);
        auto a11 = arg<int>(args, count, 11);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_rope(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            });
        });
        return;
    }
    if (name == "k_qgemv") {
        require(count == 12, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<const uchar*>(args, count, 1);
        auto a2 = arg<float*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<uint64_t>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<int>(args, count, 10);
        auto a11 = arg<int>(args, count, 11);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((sizeof(float) * (WG) + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_qgemv(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            });
        });
        return;
    }
    if (name == "k_qgemv_mr") {
        require(count == 12, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<const uchar*>(args, count, 1);
        auto a2 = arg<float*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<uint64_t>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<int>(args, count, 10);
        auto a11 = arg<int>(args, count, 11);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((sizeof(float) * (QGEMV_MR_MAX_ROWS * WG) + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_qgemv_mr(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            });
        });
        return;
    }
    if (name == "k_hgemv") {
        require(count == 10, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<const uchar*>(args, count, 1);
        auto a2 = arg<float*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<uint64_t>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((sizeof(float) * (WG) + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_hgemv(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
            });
        });
        return;
    }
    if (name == "k_dequant") {
        require(count == 7, "wrong kernel argument count");
        auto a0 = arg<const uchar*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<uint64_t>(args, count, 6);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_dequant(a0, a1, a2, a3, a4, a5, a6);
            });
        });
        return;
    }
    if (name == "k_dequant_half") {
        require(count == 5, "wrong kernel argument count");
        auto a0 = arg<const uchar*>(args, count, 0);
        auto a1 = arg<float*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<uint64_t>(args, count, 4);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_dequant_half(a0, a1, a2, a3, a4);
            });
        });
        return;
    }
    if (name == "k_hembed") {
        require(count == 11, "wrong kernel argument count");
        auto a0 = arg<const uchar*>(args, count, 0);
        auto a1 = arg<const uint*>(args, count, 1);
        auto a2 = arg<float*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<uint64_t>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<int>(args, count, 8);
        auto a9 = arg<uint*>(args, count, 9);
        auto a10 = arg<int>(args, count, 10);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_hembed(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
            });
        });
        return;
    }
    if (name == "k_qembed") {
        require(count == 13, "wrong kernel argument count");
        auto a0 = arg<const uchar*>(args, count, 0);
        auto a1 = arg<const uint*>(args, count, 1);
        auto a2 = arg<float*>(args, count, 2);
        auto a3 = arg<int>(args, count, 3);
        auto a4 = arg<int>(args, count, 4);
        auto a5 = arg<int>(args, count, 5);
        auto a6 = arg<int>(args, count, 6);
        auto a7 = arg<int>(args, count, 7);
        auto a8 = arg<uint64_t>(args, count, 8);
        auto a9 = arg<int>(args, count, 9);
        auto a10 = arg<int>(args, count, 10);
        auto a11 = arg<uint*>(args, count, 11);
        auto a12 = arg<int>(args, count, 12);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_qembed(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
            });
        });
        return;
    }
    if (name == "k_count_nan") {
        require(count == 3, "wrong kernel argument count");
        auto a0 = arg<const float*>(args, count, 0);
        auto a1 = arg<uint*>(args, count, 1);
        auto a2 = arg<int>(args, count, 2);
        ctx.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> local(sycl::range<1>((1 + 7) / 8), h);
            h.parallel_for(sycl::nd_range<3>(global, local_size), [=](sycl::nd_item<3> item) {
                auto scratch = reinterpret_cast<uchar*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());
                Kernels{item, scratch}.k_count_nan(a0, a1, a2);
            });
        });
        return;
    }
    throw std::runtime_error("unknown SYCL kernel: " + name);
