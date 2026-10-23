//! Replay one operation over identical captured inputs to isolate backend rounding.
//! This is diagnostic: captures/readbacks must never be included in timing runs.
use anyhow::{ensure, Context, Result};
use candle_core::{Device, Module, Tensor};
use clap::{Parser, ValueEnum};
use sha2::{Digest, Sha256};
use std::{fs::File, path::PathBuf};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Op {
    Rms,
    RmsRows,
    Silu,
    Softmax,
    Projection,
    Cos,
    Sin,
    Rope,
    Matmul,
    RmsRowsFused,
    RouterSelection,
}
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    cols: usize,
    #[arg(long)]
    weight: Option<String>,
    #[arg(long, value_enum, default_value = "rms")]
    op: Op,
    #[arg(long, default_value_t = 1e-6)]
    eps: f32,
    #[arg(long)]
    rope_dim: Option<usize>,
    #[arg(long, default_value_t = 10000.)]
    rope_theta: f32,
    #[arg(long, default_value_t = 0)]
    position: usize,
    #[arg(long)]
    rhs: Option<PathBuf>,
    /// First backend is the reference. Keep submissions serial.
    #[arg(long, value_delimiter = ',', default_value = "vulkan,sycl,cpu")]
    backends: Vec<String>,
    #[arg(long)]
    output: PathBuf,
}
fn fallbacks(dev: &Device) -> usize {
    match dev {
        Device::Vulkan(_) => candle_core::vulkan_backend::kernels::fallback_count(),
        Device::Sycl(_) => candle_core::sycl_backend::fallback_count(),
        Device::OpenCl(_) => candle_core::opencl_backend::fallback_count(),
        _ => 0,
    }
}
fn main() -> Result<()> {
    let a = Args::parse();
    std::fs::create_dir_all(&a.output)?;
    let bytes = std::fs::read(&a.input)?;
    ensure!(bytes.len().is_multiple_of(4), "partial f32 input");
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        a.cols > 0 && !data.is_empty() && data.len().is_multiple_of(a.cols),
        "invalid row shape"
    );
    ensure!(!a.backends.is_empty(), "at least one backend required");
    let mut reference = None::<Vec<f32>>;
    let mut results = Vec::new();
    for backend in &a.backends {
        let dev = match backend.as_str() {
            "vulkan" => Device::new_vulkan(0)?,
            "sycl" => Device::new_sycl(0)?,
            "opencl" => Device::new_opencl(0)?,
            "cpu" => Device::Cpu,
            _ => anyhow::bail!("unknown backend {backend}"),
        };
        let x = Tensor::from_vec(data.clone(), (data.len() / a.cols, a.cols), &dev)?;
        let native = match &dev {
            Device::Vulkan(_) => candle_core::vulkan_backend::kernels::native_enabled(),
            Device::Sycl(_) => candle_core::sycl_backend::kernels::native_enabled(),
            Device::OpenCl(_) => candle_core::opencl_backend::native_enabled(),
            _ => true,
        };
        ensure!(native, "{backend}: native kernels are disabled");
        let before_fallbacks = fallbacks(&dev);
        let weight = || -> Result<_> {
            let mut file = File::open(
                a.model
                    .as_ref()
                    .context("--model is required for this operation")?,
            )?;
            let header = joshua::gguf_ext::read_header(&mut file)?;
            Ok(header.to_candle_content()?.tensor(
                &mut file,
                a.weight
                    .as_deref()
                    .context("--weight is required for this operation")?,
                &dev,
            )?)
        };
        let y = match a.op {
            Op::Rms => candle_nn::ops::rms_norm(&x, &weight()?.dequantize(&dev)?, a.eps)?,
            Op::RmsRows => joshua::mhc::rms_rows(&x, a.eps as f64)?,
            Op::RmsRowsFused => candle_nn::ops::rms_norm(
                &x,
                &Tensor::ones(a.cols, candle_core::DType::F32, &dev)?,
                a.eps,
            )?,
            Op::RouterSelection => {
                let scores = joshua::moe::Gating::SqrtSoftplus.scores(&x)?;
                scores.broadcast_add(&weight()?.dequantize(&dev)?.reshape((1, ()))?)?
            }
            Op::Silu => candle_nn::ops::silu(&x)?,
            Op::Softmax => candle_nn::ops::softmax_last_dim(&x)?,
            Op::Cos => x.cos()?,
            Op::Sin => x.sin()?,
            Op::Matmul => {
                let bytes = std::fs::read(a.rhs.as_ref().context("--rhs is required")?)?;
                ensure!(bytes.len() % (4 * a.cols) == 0, "invalid right-hand matrix");
                let data: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let w = Tensor::from_vec(data, (bytes.len() / (4 * a.cols), a.cols), &dev)?;
                x.matmul(&w.t()?)?
            }
            Op::Rope => {
                let rot = a.rope_dim.unwrap_or(a.cols);
                ensure!(
                    rot > 0 && rot <= a.cols && rot % 2 == 0,
                    "invalid rotary width"
                );
                let rope = joshua::attention::Rope::new(
                    rot,
                    a.rope_theta,
                    a.position + 1,
                    joshua::attention::RopeStyle::Interleaved,
                    &dev,
                )?;
                let x = x.reshape((1, data.len() / a.cols, 1, a.cols))?;
                let rotated = rope.apply(&x.narrow(3, a.cols - rot, rot)?, a.position)?;
                if rot == a.cols {
                    rotated
                } else {
                    Tensor::cat(&[&x.narrow(3, 0, a.cols - rot)?, &rotated], 3)?
                }
            }
            Op::Projection => {
                let w = weight()?;
                match joshua::quant_matmul::try_fast_cpu_qmatmul(&w, &x) {
                    Some(y) => y?,
                    None => candle_core::quantized::QMatMul::from_qtensor(w)?.forward(&x)?,
                }
            }
        }
        .flatten_all()?
        .to_vec1::<f32>()?;
        let cpu_fallbacks = fallbacks(&dev) - before_fallbacks;
        ensure!(
            cpu_fallbacks == 0,
            "{backend}: {cpu_fallbacks} CPU fallbacks invalidate GPU arithmetic replay"
        );
        ensure!(
            y.iter().all(|v| v.is_finite()),
            "{backend}: nonfinite output"
        );
        std::fs::write(
            a.output.join(format!("{backend}.f32")),
            y.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        )?;
        let r = reference.get_or_insert_with(|| y.clone());
        ensure!(r.len() == y.len(), "output shape differs between backends");
        let max_abs = r
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let changed = r
            .iter()
            .zip(&y)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "{backend}: max_abs={max_abs:e} changed={changed}/{}",
            y.len()
        );
        results.push(serde_json::json!({"backend":backend,"max_abs":max_abs,"different_bits":changed,"elements":y.len(),"cpu_fallbacks":cpu_fallbacks,"absolute_1e_5":max_abs<=1e-5}));
    }
    let rhs_sha256 = a
        .rhs
        .as_ref()
        .map(std::fs::read)
        .transpose()?
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)));
    let report = serde_json::json!({"op":format!("{:?}",a.op),"model":a.model,"weight":a.weight,"cols":a.cols,"eps":a.eps,"input":a.input,"input_sha256":format!("{:x}",Sha256::digest(&bytes)),"rhs":a.rhs,"rhs_sha256":rhs_sha256,"rope_dim":a.rope_dim,"rope_theta":a.rope_theta,"position":a.position,"reference_backend":a.backends[0],"runs":results});
    serde_json::to_writer_pretty(File::create(a.output.join("comparison.json"))?, &report)?;
    Ok(())
}
