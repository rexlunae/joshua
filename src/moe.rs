//! Shared expert dispatch for models whose experts run independently.

use candle_core::{bail, DType, Result, Tensor};

/// Dispatch routed tokens without materializing intermediate `(token, weight)`
/// pairs. Single-token decode keeps routing weights on the device; prefill
/// moves each expert's index and weight vectors directly into tensors.
pub(crate) fn dispatch(
    x: &Tensor,
    indices: &Tensor,
    weights: &Tensor,
    n_experts: usize,
    forward: impl Fn(usize, &Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let (n_tokens, hidden) = x.dims2()?;
    let (rows, k) = indices.dims2()?;
    if rows != n_tokens || weights.dims() != indices.dims() {
        bail!("MoE: routing shapes must match the input token count and each other");
    }
    let ids: Vec<u32> = indices.flatten_all()?.to_vec1()?;
    let mut counts = if n_tokens == 1 {
        Vec::new()
    } else {
        vec![0; n_experts]
    };
    for (slot, &expert) in ids.iter().enumerate() {
        if expert as usize >= n_experts {
            bail!(
                "MoE: router selected expert {expert} out of {n_experts} (token {}, slot {})",
                slot / k,
                slot % k
            );
        }
        if n_tokens != 1 {
            counts[expert as usize] += 1;
        }
    }

    if n_tokens == 1 {
        let mut outputs = Vec::with_capacity(k);
        for expert in ids {
            outputs.push(forward(expert as usize, x)?);
        }
        let output = Tensor::stack(&outputs, 0)?;
        return output.broadcast_mul(&weights.reshape((k, 1, 1))?)?.sum(0);
    }

    let wts: Vec<f32> = weights.flatten_all()?.to_vec1()?;
    let mut per_expert: Vec<_> = counts
        .into_iter()
        .map(|count| (Vec::with_capacity(count), Vec::with_capacity(count)))
        .collect();
    for (slot, (expert, weight)) in ids.into_iter().zip(wts).enumerate() {
        let (tokens, weights) = &mut per_expert[expert as usize];
        tokens.push((slot / k) as u32);
        weights.push(weight);
    }

    let device = x.device();
    let mut output = Tensor::zeros((n_tokens, hidden), DType::F32, device)?;
    for (expert, (tokens, weights)) in per_expert.into_iter().enumerate() {
        let count = tokens.len();
        if count == 0 {
            continue;
        }
        let indices = Tensor::from_vec(tokens, count, device)?;
        let selected = x.index_select(&indices, 0)?;
        let values = forward(expert, &selected)?;
        let weights = Tensor::from_vec(weights, (count, 1), device)?;
        output = output.index_add(&indices, &values.broadcast_mul(&weights)?, 0)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn prefill_and_decode_match_weighted_expert_sum() -> Result<()> {
        let device = &Device::Cpu;
        let x = Tensor::new(&[[1f32, -2.], [3., 4.], [-5., 6.]], device)?;
        // Includes an unused expert and repeated assignments to expert 2.
        let ids = Tensor::new(&[[2u32, 0], [1, 2], [2, 2]], device)?;
        let weights = Tensor::new(&[[0.25f32, 0.75], [0.5, 0.5], [0.2, 0.8]], device)?;
        let forward = |expert: usize, x: &Tensor| x.affine((expert + 1) as f64, 0.0);
        let expected = vec![vec![1.5, -3.], vec![7.5, 10.], vec![-15., 18.]];
        assert_eq!(
            dispatch(&x, &ids, &weights, 4, forward)?.to_vec2::<f32>()?,
            expected
        );
        for (row, expected) in expected.iter().enumerate() {
            let actual = dispatch(
                &x.narrow(0, row, 1)?,
                &ids.narrow(0, row, 1)?,
                &weights.narrow(0, row, 1)?,
                4,
                forward,
            )?;
            assert_eq!(&actual.to_vec2::<f32>()?[0], expected);
        }
        Ok(())
    }

    #[test]
    fn invalid_expert_is_rejected_before_executing_experts() -> Result<()> {
        for rows in [1, 2] {
            let x = Tensor::zeros((rows, 2), DType::F32, &Device::Cpu)?;
            let ids = Tensor::from_vec(vec![0u32, 4].repeat(rows), (rows, 2), &Device::Cpu)?;
            let weights = Tensor::ones((rows, 2), DType::F32, &Device::Cpu)?;
            let error =
                dispatch(&x, &ids, &weights, 4, |_, _| panic!("invalid routing")).unwrap_err();
            assert!(error.to_string().contains("expert 4 out of 4"));
        }
        Ok(())
    }
}
