//! Closed-form float64 references for the wide rows used by DeepSeek.
use candle_core::{Device, Result, Tensor};

#[test]
fn weighted_rms_uses_accurate_sum_and_respects_storage_offsets() -> Result<()> {
    for cols in [1280, 5120, 20481] {
        for rows in [2, 33] {
            let mut data = vec![0.1f32; (rows + 1) * cols];
            for row in 1..=rows {
                data[row * cols] = if row % 2 == 0 { -0.6 } else { 0.6 };
            }
            let x = Tensor::from_vec(data, (rows + 1, cols), &Device::Cpu)?.narrow(0, 1, rows)?;
            let weights: Vec<f32> = (0..cols + 1).map(|i| 1. + (i % 31) as f32).collect();
            let alpha =
                Tensor::from_vec(weights.clone(), cols + 1, &Device::Cpu)?.narrow(0, 1, cols)?;
            for eps in [1e-20f32, 1e-6] {
                let a = 0.1f32 as f64;
                let b = 0.6f32 as f64;
                let inv = (((cols - 1) as f64 * a * a + b * b) / cols as f64 + eps as f64)
                    .sqrt()
                    .recip();
                let y = candle_nn::ops::rms_norm(&x, &alpha, eps)?.to_vec2::<f32>()?;
                for (row, values) in y.iter().enumerate() {
                    for (col, &got) in values.iter().enumerate() {
                        let v = if col == 0 {
                            if row % 2 == 0 {
                                b
                            } else {
                                -b
                            }
                        } else {
                            a
                        };
                        let want = (v * inv * weights[col + 1] as f64) as f32;
                        assert_eq!(got, want, "cols={cols} row={row} col={col} eps={eps}");
                    }
                }
            }
        }
    }
    Ok(())
}
