use crate::{Result, Tensor};
use rayon::prelude::*;

#[cfg(test)]
mod tests {
    use crate::{DType, Device, Result, Tensor};

    #[test]
    fn arg_sort_nan_is_total() -> Result<()> {
        let mut state = 175u32;
        for row in 0..2000 {
            let len = 1 + row % 384;
            let values: Vec<f32> = (0..len).map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                match state % 13 {
                    0 | 1 => f32::NAN,
                    2 => f32::INFINITY,
                    3 => f32::NEG_INFINITY,
                    4 => -0.0,
                    5 => 0.0,
                    _ => (state % 17) as f32 - 8.0,
                }
            }).collect();
            for asc in [true, false] {
                let mut expected: Vec<u32> = (0..len as u32).filter(|&index| !values[index as usize].is_nan()).collect();
                expected.sort_by(|&left, &right| {
                    let order = values[left as usize].partial_cmp(&values[right as usize]).unwrap();
                    if asc { order } else { order.reverse() }
                });
                expected.extend((0..len as u32).filter(|&index| values[index as usize].is_nan()));
                let tensor = Tensor::from_vec(values.clone(), (1, len), &Device::Cpu)?;
                let actual = tensor.arg_sort_last_dim(asc)?.flatten_all()?.to_vec1::<u32>()?;
                assert_eq!(actual, expected, "row {row}, asc={asc}");
                let mut permutation = actual;
                permutation.sort_unstable();
                assert_eq!(permutation, (0..len as u32).collect::<Vec<_>>());
            }
        }
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            let tensor = Tensor::new(&[[f32::NAN, 0.0, -0.0, f32::NAN]], &Device::Cpu)?.to_dtype(dtype)?;
            for asc in [true, false] {
                assert_eq!(tensor.arg_sort_last_dim(asc)?.to_vec2::<u32>()?, vec![vec![1, 2, 0, 3]]);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct ArgSort {
    asc: bool,
    last_dim: usize,
}

fn compare_nan_last<T: PartialOrd>(left: &T, right: &T, asc: bool) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match left.partial_cmp(right) {
        Some(order) => if asc { order } else { order.reverse() },
        None => match (left.partial_cmp(left), right.partial_cmp(right)) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Greater,
            _ => Ordering::Less,
        },
    }
}

impl ArgSort {
    fn asort<T: crate::WithDType>(&self, vs: &[T], layout: &crate::Layout) -> Vec<u32> {
        #[allow(clippy::uninit_vec)]
        // Safety: indexes are set later in the parallelized section.
        let mut sort_indexes = unsafe {
            let el_count = layout.shape().elem_count();
            let mut v = Vec::with_capacity(el_count);
            v.set_len(el_count);
            v
        };
        if self.asc {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&i, &j| {
                        compare_nan_last(&vs[i as usize], &vs[j as usize], true)
                    })
                });
        } else {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&i, &j| {
                        compare_nan_last(&vs[i as usize], &vs[j as usize], false)
                    })
                });
        }
        sort_indexes
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::cuda_backend::cudarc::driver::{
        CudaSlice, DeviceRepr, LaunchConfig, ValidAsZeroBits,
    };
    use crate::cuda_backend::{kernel_name, kernels, CudaStorageSlice as S, WrapErr};
    use crate::{CudaDevice, WithDType};

    impl crate::cuda_backend::Map1Any for ArgSort {
        fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(CudaSlice<T>) -> S>(
            &self,
            src: &CudaSlice<T>,
            dev: &CudaDevice,
            layout: &crate::Layout,
            _wrap: W,
        ) -> Result<S> {
            use cudarc::driver::PushKernelArg;

            let slice = match layout.contiguous_offsets() {
                None => crate::bail!("input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let elem_count = layout.shape().elem_count();
            let dst = unsafe { dev.alloc::<u32>(elem_count)? };
            let func = if self.asc {
                dev.get_or_load_func(&kernel_name::<T>("asort_asc"), &kernels::SORT)?
            } else {
                dev.get_or_load_func(&kernel_name::<T>("asort_desc"), &kernels::SORT)?
            };
            let ncols = self.last_dim;
            let nrows = elem_count / ncols;
            let ncols_pad = next_power_of_2(ncols);
            // Limit block dim to 1024 threads, which is the maximum on modern CUDA gpus.
            let block_dim = ncols_pad.min(1024);
            let cfg = LaunchConfig {
                grid_dim: (nrows as u32, 1, 1),
                block_dim: (block_dim as u32, 1, 1),
                shared_mem_bytes: (ncols_pad * std::mem::size_of::<u32>()) as u32,
            };
            let stream = dev.cuda_stream();
            let mut builder = stream.launch_builder(&func);
            let ncols = ncols as i32;
            let ncols_pad = ncols_pad as i32;
            builder.arg(&slice).arg(&dst).arg(&ncols).arg(&ncols_pad);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(S::U32(dst))
        }
    }
}

impl crate::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "argsort"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, crate::Shape)> {
        let sort_indexes = match storage {
            crate::CpuStorage::U8(vs) => self.asort(vs, layout),
            crate::CpuStorage::U32(vs) => self.asort(vs, layout),
            crate::CpuStorage::I16(vs) => self.asort(vs, layout),
            crate::CpuStorage::I32(vs) => self.asort(vs, layout),
            crate::CpuStorage::I64(vs) => self.asort(vs, layout),
            crate::CpuStorage::BF16(vs) => self.asort(vs, layout),
            crate::CpuStorage::F16(vs) => self.asort(vs, layout),
            crate::CpuStorage::F32(vs) => self.asort(vs, layout),
            crate::CpuStorage::F64(vs) => self.asort(vs, layout),
            crate::CpuStorage::F8E4M3(vs) => self.asort(vs, layout),
            // Dummy types don't support sorting
            crate::CpuStorage::F6E2M3(_) => {
                return Err(
                    crate::Error::UnsupportedDTypeForOp(crate::DType::F6E2M3, "argsort").bt(),
                )
            }
            crate::CpuStorage::F6E3M2(_) => {
                return Err(
                    crate::Error::UnsupportedDTypeForOp(crate::DType::F6E3M2, "argsort").bt(),
                )
            }
            crate::CpuStorage::F4(_) => {
                return Err(crate::Error::UnsupportedDTypeForOp(crate::DType::F4, "argsort").bt())
            }
            crate::CpuStorage::F8E8M0(_) => {
                return Err(
                    crate::Error::UnsupportedDTypeForOp(crate::DType::F8E8M0, "argsort").bt(),
                )
            }
        };
        let sort_indexes = crate::CpuStorage::U32(sort_indexes);
        Ok((sort_indexes, layout.shape().into()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        use crate::cuda_backend::Map1Any;
        let dev = storage.device();
        let slice = self.map(&storage.slice, dev, layout)?;
        let dst = crate::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        use crate::DType;

        let name = {
            if self.asc {
                match storage.dtype() {
                    DType::BF16 => "asort_asc_bf16",
                    DType::F16 => "asort_asc_f16",
                    DType::F32 => "asort_asc_f32",
                    DType::F64 => "asort_asc_f64",
                    DType::U8 => "asort_asc_u8",
                    DType::U32 => "asort_asc_u32",
                    DType::I16 => "asort_asc_i16",
                    DType::I32 => "asort_asc_i32",
                    DType::I64 => "asort_asc_i64",
                    DType::F8E4M3 => crate::bail!("Metal device does not yet support F8E4M3."),
                    DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                        return Err(
                            crate::Error::UnsupportedDTypeForOp(storage.dtype(), "argsort").bt(),
                        )
                    }
                }
            } else {
                match storage.dtype() {
                    DType::BF16 => "asort_desc_bf16",
                    DType::F16 => "asort_desc_f16",
                    DType::F32 => "asort_desc_f32",
                    DType::F64 => "asort_desc_f64",
                    DType::U8 => "asort_desc_u8",
                    DType::U32 => "asort_desc_u32",
                    DType::I16 => "asort_desc_i16",
                    DType::I32 => "asort_desc_i32",
                    DType::I64 => "asort_desc_i64",
                    DType::F8E4M3 => crate::bail!("Metal device does not yet support F8E4M3."),
                    DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                        return Err(
                            crate::Error::UnsupportedDTypeForOp(storage.dtype(), "argsort").bt(),
                        )
                    }
                }
            }
        };
        let device = storage.device();
        let kernels = device.kernels();
        let command_encoder = device.command_encoder()?;
        let el = layout.shape().elem_count();
        let ncols = self.last_dim;
        let nrows = el / ncols;
        let src = crate::metal_backend::buffer_o(storage.buffer(), layout, storage.dtype());
        let dst = device
            .new_buffer_builder()
            .with_size_for(el, DType::U32)
            .with_label("asort")
            .build()?;
        let mut ncols_pad = 1;
        while ncols_pad < ncols {
            ncols_pad *= 2;
        }
        candle_metal_kernels::call_arg_sort(
            device.metal_device(),
            &command_encoder,
            kernels,
            name,
            nrows,
            ncols,
            ncols_pad,
            src,
            &dst,
        )
        .map_err(crate::Error::wrap)?;
        let dst = crate::MetalStorage::new(dst, device.clone(), el, DType::U32);
        Ok((dst, layout.shape().clone()))
    }
}

#[allow(unused)]
fn next_power_of_2(x: usize) -> usize {
    let mut n = 1;
    while n < x {
        n *= 2
    }
    n
}

impl Tensor {
    /// Returns the indices that sort the tensor along the last dimension.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn arg_sort_last_dim(&self, asc: bool) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "arg_sort_last_dim",
            });
        }
        let last_dim = match self.dims().last() {
            None => crate::bail!("empty last-dim in arg-sort"),
            Some(last_dim) => *last_dim,
        };
        // No need for a backward pass for arg sort.
        self.apply_op1_no_bwd(&ArgSort { asc, last_dim })
    }

    /// Sorts the tensor along the last dimension, returns the sorted tensor together with the
    /// sorted indexes.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn sort_last_dim(&self, asc: bool) -> Result<(Tensor, Tensor)> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "sort_last_dim",
            });
        }
        let asort = self.arg_sort_last_dim(asc)?;
        let sorted = self.gather(&asort, crate::D::Minus1)?;
        Ok((sorted, asort))
    }
}
