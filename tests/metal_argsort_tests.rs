//! Isolate a suspected Metal `arg_sort_last_dim` bug at MoE-router shapes.
//!
//! The real-model prefill crashed with an expert index of 200923413
//! (`len is 128 but the index is 200923413`) — a value that a *correct*
//! argsort can never produce, since its output is a permutation of
//! `0..ncols` no matter what the input values are.  This test exercises the
//! exact shapes the router uses: `[n_tokens, 128]` F32 probs, top-k=8.

use candle_core::{Device, Tensor};

#[test]
fn metal_argsort_router_shapes_stay_in_range() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("no Metal device, skipping");
            return;
        }
    };

    for n_tokens in [1usize, 5, 128, 512] {
        let ncols = 128usize;
        let k = 8usize;
        // Random-ish probs (rows don't need to sum to 1 for argsort).
        let data: Vec<f32> = (0..n_tokens * ncols)
            .map(|i| ((i as u64 * 2654435761u64 % 1000) as f32) / 1000.0)
            .collect();
        let t = Tensor::from_vec(data, (n_tokens, ncols), &dev).unwrap();

        let topk = t
            .arg_sort_last_dim(false)
            .unwrap()
            .narrow(candle_core::D::Minus1, 0, k)
            .unwrap()
            .contiguous()
            .unwrap();
        let idx: Vec<u32> = topk.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(idx.len(), n_tokens * k);
        for (pos, &i) in idx.iter().enumerate() {
            assert!(
                (i as usize) < ncols,
                "n_tokens={n_tokens}: argsort index {i} at pos {pos} out of range (ncols={ncols})"
            );
        }
        // Descending order: first value must be >= last value in each row.
        for row in 0..n_tokens {
            let row_idx = &idx[row * k..(row + 1) * k];
            let row_vals: Vec<f32> = t
                .narrow(0, row, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            for w in row_idx.windows(2) {
                assert!(
                    row_vals[w[0] as usize] >= row_vals[w[1] as usize],
                    "n_tokens={n_tokens}: row {row} not descending at {w:?}"
                );
            }
        }
        eprintln!("n_tokens={n_tokens}: argsort OK ({} indices, all in range, descending)", idx.len());
    }
}
