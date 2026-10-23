fn main() {
    let format = std::env::var("GGML_DTYPE")
        .map(|s| s.parse().unwrap())
        .unwrap_or(12);
    // Match the generic Vulkan lane assignment and reduction tree.
    let workgroup = 256;
    let source = joshua_kernels_cubecl::native_wgsl_format(workgroup, format);
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--check") => assert_eq!(
            source,
            std::fs::read_to_string(&args[2]).unwrap(),
            "generated WGSL is stale"
        ),
        Some("--output") => std::fs::write(&args[2], source).unwrap(),
        None => print!("{source}"),
        _ => panic!("usage: generate [--check|--output PATH]"),
    }
}
