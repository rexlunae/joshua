//! Vulkan op probe: find which op diverges from the CPU at long sequence
//! lengths (#81).  Tests softmax_last, RMSNorm, rope (broadcast sin/cos), and
//! a broadcast multiply at graded seq lengths, CPU vs Vulkan, same tolerance
//! policy as the model tests.
//!
//!   cargo run --release --features vulkan --example vk_op_probe
use candle_core::{Device, Tensor, D};

fn spread_tol(cpu: &[f32]) -> f32 {
    let lo = cpu.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (hi - lo).abs().max(1e-3) * 0.08 + 0.05
}

fn check(what: &str, dev: &Tensor, cpu: &Tensor) -> bool {
    let d = dev
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let c = cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let tol = spread_tol(&c);
    let mut worst = 0.0f32;
    let mut wi = 0;
    for (i, (a, b)) in d.iter().zip(&c).enumerate() {
        let diff = (a - b).abs();
        if diff > worst {
            worst = diff;
            wi = i;
        }
    }
    let ok = worst <= tol;
    println!(
        "{what}: {} worst_abs={worst:.5} at {wi} (tol {tol:.5}) dev0={} cpu0={}",
        if ok { "PASS" } else { "FAIL" },
        d.first().copied().unwrap_or(0.),
        c.first().copied().unwrap_or(0.)
    );
    ok
}

fn main() -> candle_core::Result<()> {
    let dev = Device::vulkan_if_available(0)?;
    if dev.is_cpu() {
        println!("SKIP: no vulkan device");
        return Ok(());
    }
    for seq in [8usize, 32, 48, 64, 72, 80, 88, 96, 104, 128, 160, 200] {
        let (b, h, d) = (1usize, 8usize, 64usize);
        let xs_v: Vec<f32> = (0..b * h * seq * d)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.05)
            .collect();

        // 1. softmax_last over attention scores [b, h, seq, seq].
        let scores: Vec<f32> = (0..b * h * seq * seq)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.3)
            .collect();
        let s_cpu = Tensor::from_vec(scores.clone(), (b, h, seq, seq), &Device::Cpu)?;
        let s_dev = Tensor::from_vec(scores, (b, h, seq, seq), &dev)?;
        let sm_dev = candle_nn::ops::softmax_last_dim(&s_dev)?;
        let sm_cpu = candle_nn::ops::softmax_last_dim(&s_cpu)?;
        check(
            &format!("softmax_last [{seq}]"),
            &sm_dev,
            &sm_cpu,
        );
        // Row-level diagnosis: which rows diverge, and the sum ratio.
        if seq == 96 {
            let dv = sm_dev.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            let cv = sm_cpu.flatten_all()?.to_vec1::<f32>()?;
            let cols = seq;
            let rows = dv.len() / cols;
            let mut bad_rows: Vec<usize> = Vec::new();
            for r in 0..rows {
                let dev_sum: f32 = dv[r * cols..(r + 1) * cols].iter().sum();
                if (dev_sum - 1.0).abs() > 1e-2 {
                    bad_rows.push(r);
                }
            }
            let r0 = bad_rows.first().copied().unwrap_or(0);
            let d0 = &dv[r0 * cols..(r0 + 1) * cols];
            let c0 = &cv[r0 * cols..(r0 + 1) * cols];
            println!("  diag [96]: rows={rows} bad={}? sample row {r0}:", bad_rows.len());
            for c in (0..cols).step_by((cols / 12).max(1)) {
                println!("    c={c}: dev={:.5} cpu={:.5} ratio={}", d0[c], c0[c],
                    if c0[c].abs() > 1e-9 { d0[c] / c0[c] } else { f32::NAN });
            }
        }

        // 2. rms_norm over the hidden dim on [b, seq, d].
        let x_cpu = Tensor::from_vec(xs_v.clone(), (b, seq, d), &Device::Cpu)?;
        let x_dev = Tensor::from_vec(xs_v.clone(), (b, seq, d), &dev)?;
        let alpha = Tensor::ones(d, candle_core::DType::F32, &Device::Cpu)?;
        check(
            &format!("rms_norm [{seq}]"),
            &candle_nn::ops::rms_norm(&x_dev, &alpha.to_device(&dev)?, 1e-5)?,
            &candle_nn::ops::rms_norm(&x_cpu, &alpha, 1e-5)?,
        );

        // 3. rope: sin/cos [seq, d/2] broadcast over [b, h, seq, d].
        let half = d / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / half as f32))
            .collect();
        let pos = Tensor::from_vec(
            (0..seq).map(|p| p as f32).collect::<Vec<_>>(),
            (seq, 1),
            &Device::Cpu,
        )?;
        let freqs = pos.matmul(&Tensor::from_vec(inv, (half, 1), &Device::Cpu)?.t()?)?;
        let cos_cpu = freqs.cos()?;
        let sin_cpu = freqs.sin()?;
        let cos_dev = cos_cpu.to_device(&dev)?;
        let sin_dev = sin_cpu.to_device(&dev)?;
        let xr_cpu = Tensor::from_vec(xs_v.clone(), (b, h, seq, d), &Device::Cpu)?;
        let xr_dev = Tensor::from_vec(xs_v.clone(), (b, h, seq, d), &dev)?;
        let r_cpu = candle_nn::rotary_emb::rope(&xr_cpu, &sin_cpu, &cos_cpu)?;
        let r_dev = candle_nn::rotary_emb::rope(&xr_dev, &sin_dev, &cos_dev)?;
        check(&format!("rope [{seq}]"), &r_dev, &r_cpu);

        // 4. plain broadcast multiply: [b, h, seq, half] * [seq, half].
        let m_cpu = (xr_cpu.narrow(D::Minus1, 0, half)? * &sin_cpu.broadcast_as(
            xr_cpu
                .narrow(D::Minus1, 0, half)?
                .shape(),
        )?)?;
        let m_dev = (xr_dev.narrow(D::Minus1, 0, half)? * &sin_dev.broadcast_as(
            xr_dev
                .narrow(D::Minus1, 0, half)?
                .shape(),
        )?)?;
        check(&format!("broadcast mul [{seq}]"), &m_dev, &m_cpu);
    }
    Ok(())
}
