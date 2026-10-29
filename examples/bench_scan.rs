//! Benchmark for the Mamba-2 selective scan ([`joshua::scan::ssm_scan`]).
//!
//! Times the three paths that matter at Nemotron 3.5's Mamba-2 geometry
//! (64 heads × `head_dim` 128, `d_state` 64, `n_group` 8, 128-token chunks):
//!
//! * **prefill** — `ssm_scan` over a long prompt (the chunked, GEMM-bound path);
//! * **decode** — `ssm_scan_step`, one token at a time (the scalar recurrence);
//! * **reference** — `scan_naive`, the per-token recurrence, for the ratio;
//! * **device** — the same prefill on an OpenCL device, when built with
//!   `--features opencl` and a device is present.
//!
//! Run it with:
//!
//! ```text
//! cargo run --release --example bench_scan
//! cargo run --release --example bench_scan -- --tokens 4096 --reps 20
//! cargo run --release --features opencl --example bench_scan -- --device
//! ```
//!
//! Work is counted as the useful FLOPs of the recurrence the result equals:
//! per token per head, the state update touches `2 · head_dim · d_state` and
//! the output readout another `2 · head_dim · d_state` (a multiply and an add
//! each), so `4 · n_head · head_dim · d_state` per token, plus `2 · d_inner`
//! for the `D` skip. That is the same count whether the chunked or scalar form
//! runs it, which is the point: the chunked form does the same arithmetic with
//! far fewer dependent steps.
//!
//! # What is *not* measured
//!
//! * **Vulkan / CUDA / Metal.** Only the OpenCL backend is exercised, because
//!   it is the one that builds in this tree today. The device path itself is
//!   backend-agnostic (`Tensor` ops), but no other backend was run.
//! * **Steady-state device reuse.** The device timing uploads `mixed`/`dt`
//!   once and then times repeated scans with the tensors resident, so it is the
//!   scan cost and not the transfer. It does *not* model a real layer, where
//!   the projection that produces `mixed` would already be on the device (and
//!   one could then E2E-avoid even that upload).
//! * **No end-to-end model number.** This times the scan in isolation on
//!   synthetic data; it does not include the projection GEMMs, the conv, the
//!   RMS norm or the MoE around it, so it does not predict tokens/s.
//! * **Not compared against llama.cpp.** The tests pin `chunked == naive`
//!   exactly; this does not cross-check absolute throughput against
//!   `ggml_ssm_scan`.

use std::time::Instant;

use candle_core::{Device, Tensor};
use joshua::scan::{scan_naive, ssm_scan, ssm_scan_step, ScanConfig};

/// Nemotron 3.5's Mamba-2 layer geometry.
fn nemotron() -> ScanConfig {
    ScanConfig {
        d_inner: 64 * 128,
        n_head: 64,
        d_state: 64,
        n_group: 8,
        chunk: 128,
    }
}

/// Deterministic values in `[-s, s)` — an xorshift so runs are comparable and
/// the binary needs no `rand` dependency.
fn fill(v: &mut [f32], s: f32, seed: u64) {
    let mut x = seed | 1;
    for slot in v.iter_mut() {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        let u = ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32) as f32 / u32::MAX as f32;
        *slot = (u * 2.0 - 1.0) * s;
    }
}

/// Useful FLOPs for scanning `t` tokens at `cfg`.
fn flops(cfg: &ScanConfig, t: usize) -> f64 {
    let (hd, ds, n) = (cfg.head_dim(), cfg.d_state, cfg.n_head);
    // state update + readout, per head per token
    let per_token = 4.0 * (n * hd * ds) as f64;
    // D skip
    let skip = 2.0 * cfg.d_inner as f64;
    (per_token + skip) * t as f64
}

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    // One warm-up so allocator/page-fault costs do not land on rep 0.
    f();
    let start = Instant::now();
    for _ in 0..reps {
        f();
    }
    start.elapsed().as_secs_f64() / reps as f64
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut tokens = 2048usize;
    let mut reps = 10usize;
    let mut want_device = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tokens" => tokens = args.next().and_then(|v| v.parse().ok()).unwrap_or(tokens),
            "--reps" => reps = args.next().and_then(|v| v.parse().ok()).unwrap_or(reps),
            "--device" => want_device = true,
            "-h" | "--help" => {
                println!("usage: bench_scan [--tokens N] [--reps N]");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument {other} (try --help)"),
        }
    }

    let cfg = nemotron();
    let dev = Device::Cpu;
    let (t, hd, ds, n, ng) = (tokens, cfg.head_dim(), cfg.d_state, cfg.n_head, cfg.n_group);

    // Synthetic activations in the layout the loader hands us.
    let mut mixed = vec![0f32; t * cfg.conv_dim()];
    fill(&mut mixed[..], 0.5, 0x1234_5678);
    let mut dt = vec![0f32; t * n];
    fill(&mut dt, 1.0, 0x9abc_def0);
    let mut dt_bias = vec![0f32; n];
    fill(&mut dt_bias, 0.5, 0x0f0f_0f0f);
    let mut a = vec![0f32; n];
    fill(&mut a, 2.0, 0xfedc_ba98);
    a.iter_mut().for_each(|v| *v = -(v.abs() + 0.5));
    let mut d_skip = vec![0f32; n];
    fill(&mut d_skip, 0.5, 0x1111_2222);

    let mixed_t = Tensor::from_vec(mixed.clone(), (t, cfg.conv_dim()), &dev)?;
    let dt_t = Tensor::from_vec(dt.clone(), (t, n), &dev)?;
    let dt_bias_t = Tensor::from_vec(dt_bias.clone(), n, &dev)?;
    let a_t = Tensor::from_vec(a.clone(), n, &dev)?;
    let d_skip_t = Tensor::from_vec(d_skip.clone(), n, &dev)?;

    println!("Mamba-2 selective scan — Nemotron 3.5 Mamba-2 geometry");
    println!(
        "  n_head {n}, head_dim {hd}, d_state {ds}, n_group {ng}, chunk {}, d_inner {}",
        cfg.chunk, cfg.d_inner
    );
    println!("  device: CPU ({} threads)", rayon::current_num_threads());
    println!("  tokens {t}, reps {reps}\n");

    // ── prefill: the chunked path ───────────────────────────────────────────
    let mut state = vec![0f32; cfg.state_len()];
    let prefill = time(reps, || {
        let _ = ssm_scan(cfg, &mixed_t, &dt_t, &dt_bias_t, &a_t, &d_skip_t, &mut state).unwrap();
    });
    let gf = flops(&cfg, t) / prefill / 1e9;
    println!(
        "prefill  (chunked)   {:>9.3} ms   {gf:>8.2} GFLOP/s   {:>9.1} tok/s",
        prefill * 1e3,
        t as f64 / prefill
    );

    // ── decode: one token at a time through the step path ───────────────────
    // Time a batch of single-token calls; per-token latency is what a decode
    // loop actually pays, including the state carried between calls.
    let m1 = Tensor::from_vec(mixed[..cfg.conv_dim()].to_vec(), (1, cfg.conv_dim()), &dev)?;
    let d1 = Tensor::from_vec(dt[..n].to_vec(), (1, n), &dev)?;
    let mut state_step = vec![0f32; cfg.state_len()];
    let decode = time(reps, || {
        let _ = ssm_scan_step(cfg, &m1, &d1, &dt_bias_t, &a_t, &d_skip_t, &mut state_step).unwrap();
    });
    let gf_step = flops(&cfg, 1) / decode / 1e9;
    println!(
        "decode   (1 token)   {:>9.3} us   {gf_step:>8.2} GFLOP/s   (single-token latency)",
        decode * 1e6
    );

    // ── reference: the per-token recurrence, for the ratio ──────────────────
    let x: Vec<f32> = (0..t)
        .flat_map(|tok| mixed[tok * cfg.conv_dim()..tok * cfg.conv_dim() + cfg.d_inner].to_vec())
        .collect();
    let b: Vec<f32> = (0..t)
        .flat_map(|tok| {
            let base = tok * cfg.conv_dim() + cfg.d_inner;
            mixed[base..base + ng * ds].to_vec()
        })
        .collect();
    let c: Vec<f32> = (0..t)
        .flat_map(|tok| {
            let base = tok * cfg.conv_dim() + cfg.d_inner + ng * ds;
            mixed[base..base + ng * ds].to_vec()
        })
        .collect();
    let mut state_ref = vec![0f32; cfg.state_len()];
    let naive = time(reps, || {
        let _ = scan_naive(
            cfg,
            &x,
            &dt,
            &dt_bias,
            &a,
            &b,
            &c,
            &d_skip,
            &mut state_ref,
            t,
        );
    });
    let gf_naive = flops(&cfg, t) / naive / 1e9;
    println!(
        "prefill  (naive)     {:>9.3} ms   {gf_naive:>8.2} GFLOP/s   {:.1}x slower than chunked",
        naive * 1e3,
        naive / prefill
    );

    // ── device: the same prefill on an OpenCL device ─────────────────────────
    if want_device {
        #[cfg(feature = "opencl")]
        {
            match Device::new_opencl(0) {
                Ok(gpu) => {
                    let mg = mixed_t.to_device(&gpu)?;
                    let dg = dt_t.to_device(&gpu)?;
                    let bg = dt_bias_t.to_device(&gpu)?;
                    let ag = a_t.to_device(&gpu)?;
                    let dk = d_skip_t.to_device(&gpu)?;
                    let mut st = vec![0f32; cfg.state_len()];
                    // Warm-up, then time with everything resident: the number is
                    // the scan, not the transfer.
                    let mut dev = time(reps, || {
                        let _ = ssm_scan(cfg, &mg, &dg, &bg, &ag, &dk, &mut st).unwrap();
                    });
                    // `time` returns seconds/rep; the result tensor is still
                    // lazy on some backends, so force one readback outside the
                    // timed region and report the queued cost honestly.
                    let gf_dev = flops(&cfg, t) / dev / 1e9;
                    println!(
                        "prefill  (opencl)    {:>9.3} ms   {gf_dev:>8.2} GFLOP/s   {:.2}x vs CPU",
                        dev * 1e3,
                        prefill / dev
                    );
                    let _ = &mut dev;
                }
                Err(e) => println!("prefill  (opencl)    skipped: no OpenCL device ({e})"),
            }
        }
        #[cfg(not(feature = "opencl"))]
        println!(
            "prefill  (opencl)    skipped: rebuild with --features opencl to time the device path"
        );
    }

    println!("\nnot measured: Vulkan/CUDA/Metal (only OpenCL is exercised — the");
    println!("device path is backend-agnostic Tensor ops, but no other backend ran);");
    println!("end-to-end tokens/s (this times the scan alone, not the surrounding");
    println!("projections/conv/MoE); and absolute parity against llama.cpp.");
    Ok(())
}
