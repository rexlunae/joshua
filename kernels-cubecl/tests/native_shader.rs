#[test]
fn generated_native_shader_matches_candle_abi() {
    for (format, source) in [
        (
            12,
            include_str!("../../vendor/candle-core/src/vulkan_backend/q4k_cubecl.wgsl"),
        ),
        (
            8,
            include_str!("../../vendor/candle-core/src/vulkan_backend/q8_0_cubecl.wgsl"),
        ),
        (
            10,
            include_str!("../../vendor/candle-core/src/vulkan_backend/q2k_cubecl.wgsl"),
        ),
        (
            14,
            include_str!("../../vendor/candle-core/src/vulkan_backend/q6k_cubecl.wgsl"),
        ),
        (
            108,
            include_str!("../../vendor/candle-core/src/vulkan_backend/grouped_q8_cubecl.wgsl"),
        ),
    ] {
        let workgroup = 256;
        assert_eq!(
            source,
            joshua_kernels_cubecl::native_wgsl_format(workgroup, format)
        );
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let entry = info.get_entry_point(0);
        let mut used_bindings: Vec<_> = module
            .global_variables
            .iter()
            .filter(|(handle, _)| !entry[*handle].is_empty())
            .filter_map(|(_, var)| var.binding.as_ref().map(|binding| binding.binding))
            .collect();
        used_bindings.sort_unstable();
        assert_eq!(
            used_bindings,
            [0, 1, 2, 3],
            "unexpected metadata or temporary buffer binding"
        );
        for (handle, var) in module.global_variables.iter() {
            if matches!(
                var.binding.as_ref().map(|binding| binding.binding),
                Some(0 | 1 | 3)
            ) {
                assert!(
                    !entry[handle].contains(naga::valid::GlobalUse::WRITE),
                    "input buffer is writable"
                );
            }
        }
        assert_eq!(module.entry_points.len(), 1);
        assert_eq!(module.entry_points[0].name, "main");
        assert_eq!(
            module.entry_points[0].workgroup_size,
            [workgroup as u32, 1, 1]
        );
        naga::back::spv::write_vec(
            &module,
            &info,
            &Default::default(),
            Some(&naga::back::spv::PipelineOptions {
                shader_stage: naga::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
        )
        .unwrap();
    }
}
