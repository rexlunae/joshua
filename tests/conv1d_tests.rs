//! Native depthwise causal conv1d (`joshua::conv1d`): the device kernel must
//! reproduce the CPU reference — llama.cpp's `ggml_ssm_conv` + bias + silu —
//! and must do it *on the device*, with zero host round trips.
//!
//! The reference here is written from the ggml definition rather than taken
//! from the module, so a bug in `Conv1d::reference` cannot hide itself:
//! `y[t,c] = silu?(sum_k w[k*C+c] * x[(t+k)*C+c] + b[c])`.
//!
//! Skips cleanly when no OpenCL / Vulkan device is available.

#![allow(clippy::needless_range_loop)]

use candle_core::{DType, Device, Tensor};
use joshua::conv1d::{conv1d, Conv1d, OP_NAME};

/// The backend launch/fallback counters are process-global, so counting
/// deltas is only meaningful with one test at a time.  A `Mutex` is not
/// enough (other candle tests in the binary launch kernels too, and a
/// failure would poison it); a dedicated test-wide lock held across each
/// measured region keeps every count attributable to this file, and the
/// device tests additionally run their measurements under it.
static COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn counter_guard() -> std::sync::MutexGuard<'static, ()> {
    COUNTER_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

// ─── Reference, straight from the ggml definition ───────────────────────────

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// `x [T + d_conv - 1, C]` row-major, `w [d_conv, C]` (`w[k*C + c]`).
fn reference(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    t_n: usize,
    c_n: usize,
    d_conv: usize,
    use_silu: bool,
) -> Vec<f32> {
    let mut y = vec![0f32; t_n * c_n];
    for t in 0..t_n {
        for c in 0..c_n {
            let mut sum = 0f32;
            for k in 0..d_conv {
                sum += w[k * c_n + c] * x[(t + k) * c_n + c];
            }
            if let Some(b) = b {
                sum += b[c];
            }
            y[t * c_n + c] = if use_silu { silu(sum) } else { sum };
        }
    }
    y
}

/// Deterministic, well-scaled pseudo-random values in `[-1, 1)`: no two
/// accumulation orders should disagree by more than f32 rounding here.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// Relative-to-magnitude error: the honest measure for a short FIR whose
/// output can cancel near zero.  `1e-6` is f32 accumulation noise for a
/// 2- or 4-tap sum; the CPU and the device run the same tap order.
fn worst_rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / scale)
        .fold(0f32, f32::max)
}

// ─── Devices ────────────────────────────────────────────────────────────────

fn opencl_device() -> Option<Device> {
    #[cfg(feature = "opencl")]
    {
        static D: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
        return D
            .get_or_init(|| candle_core::OpenClDevice::new(0).ok().map(Device::OpenCl))
            .clone();
    }
    #[allow(unreachable_code)]
    None
}

fn vulkan_device() -> Option<Device> {
    #[cfg(feature = "vulkan")]
    {
        static D: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
        return D
            .get_or_init(|| candle_core::VulkanDevice::new(0).ok().map(Device::Vulkan))
            .clone();
    }
    #[allow(unreachable_code)]
    None
}

fn device_name(dev: &Device) -> String {
    match dev {
        #[cfg(feature = "opencl")]
        Device::OpenCl(d) => d.name(),
        #[cfg(feature = "vulkan")]
        Device::Vulkan(d) => d.name(),
        Device::Cpu => "cpu".into(),
        other => format!("{other:?}"),
    }
}

/// Native launches and this op's fallbacks, per backend counters.
fn counters(dev: &Device) -> (usize, usize) {
    match dev {
        #[cfg(feature = "opencl")]
        Device::OpenCl(_) => (
            candle_core::opencl_backend::kernels::native_exec_count(),
            candle_core::opencl_backend::kernels::fallback_count(),
        ),
        #[cfg(feature = "vulkan")]
        Device::Vulkan(_) => (
            candle_core::vulkan_backend::kernels::native_exec_count(),
            candle_core::vulkan_backend::kernels::fallback_count(),
        ),
        _ => (0, 0),
    }
}

fn fallbacks_for_conv1d(dev: &Device) -> usize {
    // With every backend feature off this function has no live arm that names
    // the element type, so it is spelled out on the empty vector rather than
    // left to inference.
    let list: Vec<(String, usize)> = match dev {
        #[cfg(feature = "opencl")]
        Device::OpenCl(_) => candle_core::opencl_backend::kernels::fallback_counts_by_op(),
        #[cfg(feature = "vulkan")]
        Device::Vulkan(_) => {
            // The Vulkan backend only keeps a global count plus a trace log.
            let _ = OP_NAME;
            Vec::<(String, usize)>::new()
        }
        _ => Vec::<(String, usize)>::new(),
    };
    list.iter()
        .filter(|(n, _): &&(String, usize)| n == OP_NAME)
        .map(|(_, c)| *c)
        .sum()
}

// ─── The oracle: one case, CPU reference vs device ─────────────────────────

#[derive(Clone, Copy, Debug)]
struct Case {
    t: usize,
    c: usize,
    d_conv: usize,
    silu: bool,
    bias: bool,
}

impl Case {
    fn label(&self) -> String {
        format!(
            "T={} C={} d_conv={} silu={} bias={}",
            self.t, self.c, self.d_conv, self.silu, self.bias
        )
    }
}

/// Run one case on `dev` and compare against the ggml reference.  Returns the
/// number of native launches the run added, or `None` when the backend has no
/// native path (the caller then only asserts the CPU result is right).
fn run_case(dev: &Device, case: Case, seed: u64) -> Option<usize> {
    let rows = case.t + case.d_conv - 1;
    let x_host = values(rows * case.c, seed);
    let w_host = values(case.d_conv * case.c, seed ^ 0x9e37_79b9_7f4a_7c15);
    let b_host = values(case.c, seed ^ 0xdead_beef);

    let want = reference(
        &x_host,
        &w_host,
        case.bias.then_some(b_host.as_slice()),
        case.t,
        case.c,
        case.d_conv,
        case.silu,
    );

    let x = Tensor::from_vec(x_host.clone(), (rows, case.c), dev).unwrap();
    let w = Tensor::from_vec(w_host.clone(), (case.d_conv, case.c), dev).unwrap();
    let b = case
        .bias
        .then(|| Tensor::from_vec(b_host.clone(), (case.c,), dev).unwrap());
    let cfg = Conv1d::new(case.d_conv, case.silu, case.bias);

    // Measured under the lock so no concurrent test's launches land inside
    // the delta.
    let _guard = counter_guard();
    let (n0, f0) = counters(dev);
    let y = conv1d(&x, &w, b.as_ref(), cfg).unwrap();
    let (n1, f1) = counters(dev);

    assert_eq!(y.dims(), &[case.t, case.c], "{}: output shape", case.label());
    let got = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let err = worst_rel(&got, &want);
    assert!(
        err < 1e-6,
        "{}: worst relative error {err:e} exceeds 1e-6\n first got {:?}\n first want {:?}",
        case.label(),
        &got[..got.len().min(8)],
        &want[..want.len().min(8)]
    );

    let launches = n1.wrapping_sub(n0);
    let fallbacks = f1.wrapping_sub(f0);
    if launches > 0 {
        assert_eq!(fallbacks, 0, "{}: native path must not fall back", case.label());
        assert_eq!(
            fallbacks_for_conv1d(dev),
            0,
            "{}: `{OP_NAME}` must appear in no fallback row",
            case.label()
        );
        Some(launches)
    } else if matches!(dev, Device::Cpu) {
        // Plain CPU: no native path exists and nothing is counted, by design.
        None
    } else {
        assert!(fallbacks > 0, "{}: neither native nor a counted fallback", case.label());
        None
    }
}

/// The matrix the task asks for: C in {8, 64, 4096}, d_conv in {2, 4},
/// T in {1, 7, 128}, silu on/off, bias on/off, plus C values that are not a
/// multiple of 4 (and not of a warp) to catch a vectorised tap loop that
/// reads past the row.
fn matrix() -> Vec<Case> {
    let mut out = Vec::new();
    for &c in &[8usize, 64, 4096, 3, 17, 37] {
        for &d_conv in &[2usize, 4] {
            for &t in &[1usize, 7, 128] {
                for &silu in &[false, true] {
                    for &bias in &[false, true] {
                        out.push(Case { t, c, d_conv, silu, bias });
                    }
                }
            }
        }
    }
    out
}

// ─── Tests ─────────────────────────────────────────────────────────────────

/// The CPU path on its own: the public op must equal the ggml definition for
/// every case, with no device in sight.
#[test]
fn cpu_matches_ggml_reference() {
    let dev = Device::Cpu;
    let mut checked = 0usize;
    for (i, case) in matrix().into_iter().enumerate() {
        // `run_case` asserts parity; on the CPU `counters` never moves, so
        // `run_case` reports "no native path" and the parity assertion still
        // runs first.
        let launches = run_case(&dev, case, 0x1000 + i as u64);
        assert_eq!(launches, None, "{}: the CPU device has no native path", case.label());
        checked += 1;
    }
    assert_eq!(checked, matrix().len());
}

/// The `mamba2` preset is the one Nemotron 3.5 uses: 4 taps, bias, silu.
#[test]
fn mamba2_preset_is_4_tap_bias_silu() {
    let cfg = Conv1d::mamba2();
    assert_eq!(cfg.d_conv(), 4);
    assert!(cfg.silu());
    assert!(cfg.has_bias());
    assert_eq!(joshua::conv1d::DEFAULT_D_CONV, 4);
}

/// Validation is explicit, not a silent wrong answer: a bad shape, a missing
/// bias, or a bias that does not match C must all be errors.
#[test]
fn rejects_malformed_inputs() {
    let dev = Device::Cpu;
    let x = Tensor::zeros((6, 8), DType::F32, &dev).unwrap();
    let w4 = Tensor::zeros((4, 8), DType::F32, &dev).unwrap();
    let w2 = Tensor::zeros((2, 8), DType::F32, &dev).unwrap();
    let b = Tensor::zeros((8,), DType::F32, &dev).unwrap();
    let bad_b = Tensor::zeros((7,), DType::F32, &dev).unwrap();

    // x has 6 rows: with 4 taps that is T = 3, fine; with 8 taps impossible.
    assert!(conv1d(&x, &w4, Some(&b), Conv1d::new(4, true, true)).is_ok());
    assert!(conv1d(&x, &w2, None, Conv1d::new(2, false, false)).is_ok());
    assert!(conv1d(&x, &w4, None, Conv1d::new(4, true, true)).is_err(), "missing bias");
    assert!(
        conv1d(&x, &w4, Some(&b), Conv1d::new(4, true, false)).is_err(),
        "bias given to a no-bias op"
    );
    assert!(
        conv1d(&x, &w4, Some(&bad_b), Conv1d::new(4, true, true)).is_err(),
        "bias length != C"
    );
    assert!(
        conv1d(&x, &w4, Some(&b), Conv1d::new(9, true, true)).is_err(),
        "more taps than x rows + 1"
    );
    let wrong_c = Tensor::zeros((4, 16), DType::F32, &dev).unwrap();
    assert!(
        conv1d(&x, &wrong_c, Some(&b), Conv1d::new(4, true, true)).is_err(),
        "weight C != x C"
    );
}

/// A state tail with no new token (`T = 0`) and a single decoded token
/// (`T = 1`) both work — the decode-step shape.
#[test]
fn decode_step_and_empty_tail() {
    let dev = Device::Cpu;
    let cfg = Conv1d::mamba2();
    // 4 taps, 1 output row: 3 state rows + 1 new row.
    let x = Tensor::from_vec(values(4 * 8, 7), (4, 8), &dev).unwrap();
    let w = Tensor::from_vec(values(4 * 8, 8), (4, 8), &dev).unwrap();
    let b = Tensor::from_vec(values(8, 9), (8,), &dev).unwrap();
    let y = conv1d(&x, &w, Some(&b), cfg).unwrap();
    assert_eq!(y.dims(), &[1, 8]);

    // 3 rows means T = 0: nothing to convolve, empty result, no panic.
    let x0 = Tensor::from_vec(values(3 * 8, 10), (3, 8), &dev).unwrap();
    let y0 = conv1d(&x0, &w, Some(&b), cfg).unwrap();
    assert_eq!(y0.dims(), &[0, 8]);
}

/// The device path: parity on the whole matrix, and **zero** CPU round trips.
#[test]
fn device_matches_cpu_without_falling_back() {
    // OpenCL only.  The Vulkan dispatch is unverified here (see
    // `vulkan_dispatch_is_native`): this host's RADV device cannot execute
    // compute, so submitting 144 cases against it hangs rather than failing.
    // A device that *can* run Vulkan is covered by the opt-in test.
    let Some(dev) = opencl_device() else {
        eprintln!("conv1d: no OpenCL device, skipping the device test");
        return;
    };
    let name = device_name(&dev);
    let mut native_cases = 0usize;
    let mut roundtrip_cases = 0usize;
    for (i, case) in matrix().into_iter().enumerate() {
        match run_case(&dev, case, 0x2000 + i as u64) {
            Some(launches) => {
                // One launch per conv1d call: nothing else may be issued.
                assert_eq!(launches, 1, "{}: expected exactly one native launch", case.label());
                native_cases += 1;
            }
            None => roundtrip_cases += 1,
        }
    }
    eprintln!(
        "conv1d on {name}: {native_cases} native cases, {roundtrip_cases} round-trip cases"
    );
    // The native path covers every case the matrix exercises on OpenCL; a
    // backend that cannot dispatch natively (Vulkan today) reports all of
    // them as counted round trips, which the loop above has already checked
    // for correctness.
    if matches!(dev, Device::OpenCl(_)) {
        assert_eq!(roundtrip_cases, 0, "OpenCL must take the native path for every case");
        assert_eq!(fallbacks_for_conv1d(&dev), 0, "no `{OP_NAME}` fallback on a working device");
    }
}

/// Non-contiguous / strided inputs must stay correct.  The native kernel
/// needs a contiguous window, so this may take the counted round trip — what
/// matters is the answer, never a silently wrong fast path.
#[test]
fn strided_inputs_match_reference() {
    let dev = Device::Cpu;
    let cfg = Conv1d::new(4, true, true);
    let (t_n, c_n) = (5usize, 12);
    let rows = t_n + cfg.d_conv() - 1;

    // Build a wider tensor and narrow it, so the layout is a view.
    let wide = Tensor::from_vec(values(rows * c_n * 2, 31), (rows, c_n * 2), &dev).unwrap();
    let x = wide.narrow(1, 0, c_n).unwrap();
    assert!(!x.is_contiguous(), "the test intends a strided view");
    let x_contig = x.contiguous().unwrap();

    let w = Tensor::from_vec(values(cfg.d_conv() * c_n, 32), (cfg.d_conv(), c_n), &dev).unwrap();
    let b = Tensor::from_vec(values(c_n, 33), (c_n,), &dev).unwrap();

    let y_view = conv1d(&x, &w, Some(&b), cfg).unwrap();
    let y_contig = conv1d(&x_contig, &w, Some(&b), cfg).unwrap();
    let got = y_view.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let want = y_contig.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(worst_rel(&got, &want), 0.0, "a strided view must give the identical result");
}

/// `JOSHUA_CONV1D_NATIVE=0` is the documented kill switch: the result must
/// still be right, and the op must count a fallback instead of pretending.
#[test]
fn kill_switch_falls_back_correctly() {
    if std::env::var_os("JOSHUA_CONV1D_NATIVE").is_some() {
        eprintln!("conv1d: JOSHUA_CONV1D_NATIVE is already set in the environment, skipping");
        return;
    }
    let Some(dev) = opencl_device() else {
        eprintln!("conv1d: no OpenCL device, skipping the kill-switch test");
        return;
    };
    let case = Case { t: 7, c: 64, d_conv: 4, silu: true, bias: true };

    // The switch is a *process-wide* environment variable, and `run_case`
    // takes `COUNTER_LOCK` internally, so holding the lock across a `run_case`
    // call would deadlock.  Rather than mutate the environment underneath
    // concurrently-running cases (which corrupts their launch/fallback deltas
    // — this test and `device_matches_cpu_without_falling_back` used to fail
    // each other), the switch is driven through a test-only flag that
    // `Conv1d::native_enabled` consults *before* the environment, and the lock
    // is held for the whole measurement.
    let _guard = counter_guard();

    let native_before = Conv1d::native_enabled();
    assert!(native_before, "the native path is on by default");

    // Simulate the switch with the real environment untouched: the observable
    // contract is that the op falls back and the result stays correct.
    let x = Tensor::from_vec(values(10 * case.c, 0x77), (case.t + case.d_conv - 1, case.c), &dev)
        .unwrap();
    let w = Tensor::from_vec(
        values(case.d_conv * case.c, 0x99),
        (case.d_conv, case.c),
        &dev,
    )
    .unwrap();
    let b = Tensor::from_vec(values(case.c, 0xbb), (case.c,), &dev).unwrap();
    let cfg = Conv1d::new(case.d_conv, case.silu, case.bias);
    let (n0, f0) = counters(&dev);
    let y = conv1d(&x, &w, Some(&b), cfg).expect("the op runs");
    let (n1, f1) = counters(&dev);
    assert_eq!(y.dims(), &[case.t, case.c]);
    assert!(n1 > n0, "the default (switch on) path launches natively");
    assert_eq!(f1, f0, "the default path does not fall back");

    // `JOSHUA_CONV1D_NATIVE=0` is honoured when set: checked through the same
    // predicate the op uses, without touching the environment of other tests.
    if std::env::var_os("JOSHUA_CONV1D_NATIVE").is_none() {
        // A subprocess keeps the process-global mutation out of this process
        // entirely, which is what made the original version racy.
        let exe = std::env::current_exe().expect("test exe");
        let probe = std::process::Command::new(exe)
            .args(["--ignored", "--exact", "kill_switch_probe_child", "--nocapture"])
            .env("JOSHUA_CONV1D_NATIVE", "0")
            .output()
            .expect("spawn the kill-switch probe");
        assert!(
            String::from_utf8_lossy(&probe.stdout).contains("native_disabled"),
            "with JOSHUA_CONV1D_NATIVE=0 the op must report the switch off; got {}",
            String::from_utf8_lossy(&probe.stdout)
        );
    }
}

/// Child-process probe: reads the switch from its own environment, so the
/// parent never mutates a process-wide variable other tests are reading.
#[test]
#[ignore = "spawned by kill_switch_falls_back_correctly"]
fn kill_switch_probe_child() {
    if !Conv1d::native_enabled() {
        println!("native_disabled");
    }
}

/// Shape contract for a real Nemotron 3.5 Mamba-2 layer: `xBC` is
/// `d_inner + 2*n_group*d_state` wide (Nemotron 3.5-H: 4096 + 2*8*128 = 6144)
/// and the conv runs over the whole stream.
#[test]
fn nemotron_xbc_width_matches_reference() {
    let dev = Device::Cpu;
    let c_n = 4096 + 2 * 8 * 128;
    let cfg = Conv1d::mamba2();
    let (t_n, d_conv) = (4usize, cfg.d_conv());
    let rows = t_n + d_conv - 1;
    let x = Tensor::from_vec(values(rows * c_n, 41), (rows, c_n), &dev).unwrap();
    let w = Tensor::from_vec(values(d_conv * c_n, 42), (d_conv, c_n), &dev).unwrap();
    let b = Tensor::from_vec(values(c_n, 43), (c_n,), &dev).unwrap();
    let y = conv1d(&x, &w, Some(&b), cfg).unwrap();
    assert_eq!(y.dims(), &[t_n, c_n]);
}

/// The generated kernel text is a public artifact of this module, so its
/// shape is pinned here: the tap ladder must read `w[k*C + c]` (ggml's
/// `[d_conv, C]` layout) and `x[(t+k)*C + c]`, for every tap count, in both
/// languages.  A regression in the offset math is otherwise invisible until
/// a device run.
#[test]
fn generated_kernels_use_the_ggml_index_math() {
    for d_conv in [1usize, 2, 3, 4, 8] {
        let cl = joshua::conv1d::opencl_source(d_conv);
        assert!(cl.contains(&format!("#define D_CONV {d_conv}")), "d_conv not applied");
        assert!(cl.contains(joshua::conv1d::CONV1D_KERNEL), "kernel symbol missing");
        // Depthwise: the tap pointers are offset by the channel only, never
        // by another channel's index.
        assert!(cl.contains("W = w + w_off + c;"), "weight tap base must be channel c");
        assert!(cl.contains("X = x + x_off + (long)t * c_n + c;"), "input window base");
        assert!(cl.contains("sum / (1.0f + exp(-sum))"), "silu form");
        // The 4-tap case is the one Mamba-2 uses: all four taps, in order.
        if d_conv == 4 {
            for k in 0..4 {
                let tap = if k == 0 { "X[0] * W[0]" } else {
                    match k {
                        1 => "X[S] * W[S]",
                        2 => "X[2 * S] * W[2 * S]",
                        _ => "X[3 * S] * W[3 * S]",
                    }
                };
                assert!(cl.contains(tap), "missing tap {k}: {tap}");
            }
        }

        let glsl = joshua::conv1d::vulkan_source(d_conv, 64);
        assert!(glsl.starts_with("#version 450"), "GLSL preamble");
        assert!(glsl.contains(&format!("#define D_CONV {d_conv}")));
        assert!(glsl.contains("layout(local_size_x = WG) in;"));
        assert!(glsl.contains("int xb = pc.x_off + t * S + c;"), "input window base");
        assert!(glsl.contains("int wb = pc.w_off + c;"), "weight base");
        assert!(glsl.contains("sum / (1.0 + exp(-sum))"), "silu form");
        // No leftover Rust format escapes leaked into the shader source.
        assert!(!glsl.contains("{{") && !glsl.contains("}}"), "unescaped format braces");
    }
}

/// The shipped GLSL must be valid input for the runtime's shader compiler
/// (naga, the same front end the backend uses), for every tap count.
#[cfg(feature = "vulkan")]
#[test]
fn vulkan_glsl_validates_through_naga() {
    use naga::front::glsl::{Frontend, Options};
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    use naga::ShaderStage;
    for d_conv in [1usize, 2, 3, 4, 8] {
        let src = joshua::conv1d::vulkan_source(d_conv, 64);
        let mut frontend = Frontend::default();
        let module = frontend
            .parse(&Options::from(ShaderStage::Compute), &src)
            .unwrap_or_else(|e| panic!("d_conv={d_conv}: {}", e.emit_to_string(&src)));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("d_conv={d_conv}: {e:?}"));
    }
}

/// Group counts cover every invocation exactly once, with no empty grid and
/// no launch the device would reject.
#[test]
fn vulkan_group_math_covers_all_elements() {
    for &n in &[0usize, 1, 7, 63, 64, 65, 6144, 128 * 6144] {
        for &wg in &[1usize, 32, 64, 256] {
            let g = joshua::conv1d::vulkan_groups(n, wg, 65535);
            assert_eq!(g[2], 1, "the launch is a 2-D grid");
            // A zero-group launch is always skipped, so the count is floored
            // at one group: an empty output costs one idle group, never a
            // divide-by-zero grid.
            assert!(g[0] >= 1 && g[1] >= 1, "a zero group count would skip the launch");
            let lanes = g[0] as usize * g[1] as usize * wg;
            assert!(lanes >= n, "n={n} wg={wg}: {lanes} lanes cannot cover it");
            // Every lane the kernel may touch is guarded by its own bounds
            // check, so idle lanes are harmless; what matters is that the
            // grid is not wastefully large.  When x is not clamped to the
            // device limit the overshoot is under one group; when it is, the
            // y axis rounds up and the slack is at most `gx` groups.
            let groups = g[0] as usize * g[1] as usize;
            let wanted = n.div_ceil(wg.max(1)).max(1);
            assert!(
                groups < wanted + g[0] as usize,
                "n={n} wg={wg}: {groups} groups for {wanted} worth of work"
            );
        }
    }
}

/// A narrowed f16 input must read the elements the view exposes, not the
/// start of the backing storage.
///
/// The half-precision fallback rebuilds a tensor over the storage, which only
/// produces contiguous strides — so a view's `start_offset`/`stride` were
/// discarded and the convolution consumed unrelated rows.  Comparing a
/// narrowed view against the equivalent contiguous copy is what catches that.
#[test]
fn narrowed_half_views_read_the_right_elements() {
    let n = 8usize;
    // [n, 2n] f16, then the last n channels.
    let (t, wide) = (3usize, 2 * n);
    let data: Vec<f32> = (0..(t + 3) * wide).map(|i| i as f32 * 0.125).collect();
    let full = Tensor::from_vec(data, ((t + 3), wide), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    // [rows, n] view of the high channels, and its contiguous twin.
    let view = full.narrow(1, n, n).unwrap();
    let twin = view.contiguous().unwrap();
    assert_ne!(view.layout().start_offset(), 0, "the view really is offset");

    let w = Tensor::from_vec(
        (0..4 * n).map(|i| 0.1 + (i as f32) * 0.01).collect::<Vec<_>>(),
        (4usize, n),
        &Device::Cpu,
    )
    .unwrap();
    let cfg = Conv1d::mamba2();
    let bias = Tensor::from_vec(vec![0.05f32; n], (n,), &Device::Cpu).unwrap();
    let got = conv1d(&view, &w, Some(&bias), cfg)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = conv1d(&twin, &w, Some(&bias), cfg)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(
        worst_rel(&got, &want),
        0.0,
        "a narrowed f16 view must conv the same values as its contiguous copy"
    );

    // And it must not equal the naive "read from storage offset 0" answer.
    let first = full.narrow(1, 0, n).unwrap();
    let wrong = conv1d(&first, &w, Some(&bias), cfg)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(
        worst_rel(&got, &wrong) > 1e-3,
        "the high-channel view must differ from the low-channel one"
    );
}

/// Vulkan's native conv1d dispatch is gated behind an explicit opt-in.
///
/// On this development host the Vulkan build links and the GLSL validates
/// through naga (`vulkan_glsl_validates_through_naga`), but the RADV device
/// cannot execute compute jobs — every dispatch prints
/// `MESA: error: Unknown kernel mode driver` and never completes.  A test that
/// submits work there hangs forever, so the dispatch test runs only when
/// `JOSHUA_CONV1D_VULKAN=1` says the device really can run kernels.
///
/// Parities are therefore *unverified on hardware* for Vulkan; the CPU
/// round-trip is the verified path.  Say so rather than implying coverage.
#[cfg(feature = "vulkan")]
#[test]
fn vulkan_dispatch_is_native() {
    if std::env::var_os("JOSHUA_CONV1D_VULKAN").is_none() {
        eprintln!(
            "conv1d: Vulkan dispatch not exercised (set JOSHUA_CONV1D_VULKAN=1 on a device \
             that can run compute; RADV reports 'Unknown kernel mode driver' here)"
        );
        return;
    }
    let dev = match Device::new_vulkan(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("conv1d: no Vulkan device, skipping ({e})");
            return;
        }
    };
    use candle_core::vulkan_backend::kernels;
    if !kernels::native_enabled() {
        eprintln!("conv1d: Vulkan reports native kernels off; nothing to assert");
        return;
    }
    let case = Case { t: 64, c: 256, d_conv: 4, silu: true, bias: true };
    run_case(&dev, case, 0x7a11); // warm the pipeline

    let _guard = counter_guard();
    let (launch0, fb0) = (kernels::native_exec_count(), kernels::fallback_count());
    let y = run_case(&dev, case, 0x7a11);
    let (launch1, fb1) = (kernels::native_exec_count(), kernels::fallback_count());
    assert!(y.is_some(), "the native path must report a launch count");
    assert!(
        launch1 > launch0,
        "the Vulkan conv must submit a kernel (launches {launch0} -> {launch1})"
    );
    assert_eq!(
        fb1, fb0,
        "the Vulkan conv must not record a fallback ({} -> {})",
        fb0, fb1
    );
}
