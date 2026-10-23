//! Real-model backend benchmark: one model/device per process, shared weights per
//! session, host logits included, identical teacher-forced decode inputs.
use anyhow::{ensure, Context, Result};
use candle_core::vulkan_backend::kernels::{self, Qgemv};
use candle_core::{Device, Module, Tensor, VulkanDevice};
use clap::Parser;
use joshua::model::QuantizedModel;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Read, Write},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

#[derive(Parser)]
#[command(args_override_self = true)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long, default_value = "production")]
    kernel: String,
    /// cpu, opencl, sycl, or vulkan; Vulkan remains the default for legacy A/B runs.
    #[arg(long, default_value = "vulkan")]
    backend: String,
    /// Use the engine's actual dense/expert/cache placement policy.
    #[arg(long)]
    auto_placement: bool,
    /// Keep dense on CPU while evaluating the engine's automatic cache policy.
    #[arg(long, requires = "auto_placement")]
    auto_dense_cpu: bool,
    /// Explicit CPU dense placement for device-expert-cache experiments.
    #[arg(long, conflicts_with = "auto_placement")]
    dense_cpu: bool,
    /// Compact GPU expert-cache budget (DeepSeek OpenCL/SYCL only).
    #[arg(long, requires = "dense_cpu", conflicts_with = "auto_placement")]
    expert_cache_mib: Option<u64>,
    /// Time one actual packed projection without loading the complete model.
    #[arg(long, conflicts_with = "auto_placement")]
    projection_tensor: Option<String>,
    #[arg(long, default_value_t = 1)]
    projection_rows: usize,
    /// Benchmark the selected diagonal outputs of a grouped projection.
    #[arg(long, requires = "projection_tensor")]
    projection_grouped: bool,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Teacher-force this continuation to exercise diverse routing.
    #[arg(long, conflicts_with = "reference")]
    decode_text_file: Option<PathBuf>,
    #[arg(long, default_value_t = 16)]
    prompt_tokens: usize,
    /// Use this complete prompt instead of truncating the default text.
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Render the GGUF chat template with a user message (thinking disabled).
    #[arg(long)]
    chat: bool,
    /// Inspect tokenization/rendering without loading model weights.
    #[arg(long)]
    print_prompt: bool,
    /// Warm only this prefix; timed sessions always use the full prompt.
    #[arg(long)]
    warm_prompt_tokens: Option<usize>,
    #[arg(long, default_value_t = 1)]
    prefill_chunk: usize,
    #[arg(long, default_value_t = 32)]
    decode: usize,
    #[arg(long, default_value_t = 3)]
    passes: usize,
    /// Decode tokens used to warm pipelines before timed passes.
    #[arg(long, default_value_t = 4)]
    warm_decode: usize,
    /// Maximum KV positions; defaults to the workload plus warmup.
    #[arg(long)]
    context: Option<usize>,
}
#[derive(Serialize, Deserialize)]
struct Trace {
    prompt: Vec<u32>,
    inputs: Vec<u32>,
}

// Pin the executable's identity before timing or loading the model. Results
// from different builds cannot isolate a kernel selector, even with equal IDs.
fn executable_sha256() -> Result<String> {
    let mut file = File::open(std::env::current_exe()?)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
#[derive(Serialize)]
struct MemorySample {
    phase: &'static str,
    pass: usize,
    position: usize,
    reserved_bytes: u64,
    live_bytes: u64,
    anonymous_kib: u64,
    swapped_kib: u64,
    total_anonymous_kib: u64,
    file_kib: u64,
    rss_kib: u64,
    reference_logits_bytes: usize,
    driver_memory_bytes: BTreeMap<String, u64>,
}
fn memory_sample(
    dev: &Device,
    phase: &'static str,
    pass: usize,
    position: usize,
    reference_logits_bytes: usize,
) -> MemorySample {
    let (reserved_bytes, live_bytes) = device_memory(dev);
    let host = host_memory_kib();
    MemorySample {
        phase,
        pass,
        position,
        reserved_bytes,
        live_bytes,
        anonymous_kib: host["RssAnon"],
        swapped_kib: host["VmSwap"],
        total_anonymous_kib: host["RssAnon"] + host["VmSwap"],
        file_kib: host["RssFile"],
        rss_kib: host["VmRSS"],
        reference_logits_bytes,
        driver_memory_bytes: driver_memory_bytes(),
    }
}
// DRM clients may have duplicate descriptors. Count each client once, and
// keep regions separate: system allocations are not necessarily VRAM residency.
fn driver_memory_bytes() -> BTreeMap<String, u64> {
    let mut clients = std::collections::BTreeSet::new();
    let mut totals = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir("/proc/self/fdinfo") else {
        return totals;
    };
    for entry in entries.flatten() {
        let Ok(info) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let fields: BTreeMap<_, _> = info
            .lines()
            .filter_map(|l| l.split_once(':'))
            .map(|(k, v)| (k.trim(), v.trim()))
            .collect();
        if fields.get("drm-driver") != Some(&"xe")
            || fields.get("drm-pdev") != Some(&"0000:03:00.0")
        {
            continue;
        }
        let Some(id) = fields.get("drm-client-id") else {
            continue;
        };
        if !clients.insert(id.to_string()) {
            continue;
        }
        for (key, value) in fields {
            if key.starts_with("drm-total-cycles-") {
                continue;
            }
            if !(key.starts_with("drm-memory-")
                || key.starts_with("drm-resident-")
                || key.starts_with("drm-total-"))
            {
                continue;
            }
            let mut parts = value.split_whitespace();
            if let Some(bytes) = parts.next().and_then(|n| n.parse::<u64>().ok()) {
                let scale = match parts.next() {
                    Some("KiB") => 1024,
                    Some("MiB") => 1024 * 1024,
                    _ => 1,
                };
                *totals.entry(key.to_owned()).or_default() += bytes * scale;
            }
        }
    }
    totals
}
// These are Vulkan allocator statistics only. Other backends expose driver
// samples separately; zeros here must never be treated as a memory-gate pass.
fn device_memory(dev: &Device) -> (u64, u64) {
    match dev {
        Device::Vulkan(vk) => vk.memory_stats(),
        _ => (0, 0),
    }
}
fn fallback_count(dev: &Device) -> usize {
    match dev {
        Device::Vulkan(_) => kernels::fallback_count(),
        #[cfg(feature = "opencl")]
        Device::OpenCl(_) => candle_core::opencl_backend::fallback_count(),
        #[cfg(feature = "sycl")]
        Device::Sycl(_) => candle_core::sycl_backend::fallback_count(),
        _ => 0,
    }
}
fn device_name(dev: &Device) -> String {
    match dev {
        Device::Cpu => "CPU".into(),
        Device::Vulkan(vk) => vk.name().into(),
        #[cfg(feature = "opencl")]
        Device::OpenCl(d) => d.name(),
        #[cfg(feature = "sycl")]
        Device::Sycl(d) => d.name().into(),
        _ => format!("{dev:?}"),
    }
}
fn io_counters() -> BTreeMap<String, u64> {
    std::fs::read_to_string("/proc/self/io")
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            Some((key.to_owned(), value.trim().parse().ok()?))
        })
        .collect()
}
fn host_memory_kib() -> BTreeMap<String, u64> {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let key = fields.next()?.trim_end_matches(':');
            matches!(key, "VmRSS" | "RssAnon" | "RssFile" | "RssShmem" | "VmSwap")
                .then(|| (key.to_owned(), fields.next().unwrap().parse().unwrap()))
        })
        .collect()
}
fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find(|v| v.starts_with("VmRSS:"))
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap()
}
fn argmax(x: &[f32]) -> u32 {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}
fn forward(m: &mut QuantizedModel, dev: &Device, ids: &[u32], offset: usize) -> Result<Vec<f32>> {
    let input = Tensor::new(ids, dev)?.unsqueeze(0)?;
    Ok(m.forward(&input, offset)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?)
}
fn prefill(
    m: &mut QuantizedModel,
    dev: &Device,
    ids: &[u32],
    chunk: usize,
    mut completed: impl FnMut(usize),
) -> Result<Vec<f32>> {
    let mut last = Vec::new();
    for (index, tokens) in ids.chunks(chunk).enumerate() {
        last = forward(m, dev, tokens, index * chunk)?;
        ensure!(
            last.iter().all(|v| v.is_finite()),
            "nonfinite prefill logits at chunk {index}"
        );
        let position = index * chunk + tokens.len();
        completed(position);
        println!(
            "prefill tokens={position}/{} batch={}",
            ids.len(),
            tokens.len()
        );
    }
    Ok(last)
}
fn benchmark_projection(
    args: &Args,
    header: &joshua::gguf_ext::GgufHeader,
    file: &File,
    dev: &Device,
    name: &str,
    executable_sha256: &str,
) -> Result<()> {
    ensure!(
        (1..=16).contains(&args.projection_rows),
        "projection rows must be 1..=16"
    );
    let content = header.to_candle_content()?;
    let weight = content.tensor(&mut &*file, name, dev)?;
    let dtype = format!("{:?}", weight.dtype());
    let (n, k) = weight.shape().dims2()?;
    let weight_bytes = weight.storage_size_in_bytes();
    let matmul = candle_core::quantized::QMatMul::from_qtensor(weight)?;
    let values = (0..args.projection_rows * k)
        .map(|i| ((i % 17) as f32 - 8.) * 0.125)
        .collect::<Vec<_>>();
    let input = Tensor::from_vec(values, (args.projection_rows, k), dev)?;
    let run = || -> Result<Vec<f32>> {
        let out = if args.projection_grouped {
            ensure!(
                n % args.projection_rows == 0,
                "group count must divide output rows"
            );
            joshua::attention::grouped_output_projection(
                &matmul,
                &input.reshape((1, args.projection_rows, k))?,
                args.projection_rows,
                n / args.projection_rows,
            )?
        } else {
            matmul.forward(&input)?
        };
        Ok(out.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?)
    };
    let reference = run()?;
    ensure!(
        reference.iter().all(|v| v.is_finite()),
        "nonfinite projection"
    );
    let mut times = Vec::new();
    let mut driver_after = Vec::new();
    let mut allocator_after = Vec::new();
    let mut repeat_difference = 0f32;
    for pass in 0..args.passes {
        let mut elapsed = 0.;
        for _ in 0..args.decode {
            let start = Instant::now();
            let out = run()?;
            ensure!(out.iter().all(|v| v.is_finite()), "nonfinite projection");
            std::hint::black_box(argmax(&out));
            elapsed += start.elapsed().as_secs_f64() * 1000.;
            for (a, b) in out.iter().zip(&reference) {
                repeat_difference = repeat_difference.max((a - b).abs());
            }
        }
        times.push(elapsed / args.decode as f64);
        dev.synchronize()?;
        driver_after.push(driver_memory_bytes());
        allocator_after.push(device_memory(dev));
        println!(
            "projection={name} pass={} ms/op={:.3}",
            pass + 1,
            times[pass]
        );
    }
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let median = (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.;
    let mut dump = BufWriter::new(File::create(args.output.join("logits.f32"))?);
    for value in &reference {
        dump.write_all(&value.to_le_bytes())?;
    }
    dump.flush()?;
    let summary = serde_json::json!({"mode":"projection", "model":args.model,
        "executable_sha256":executable_sha256,
        "backend":args.backend,"gpu":device_name(dev),"tensor":name,"dtype":dtype,
        "grouped":args.projection_grouped,"kernel":args.kernel,"m":args.projection_rows,"n":n,"k":k,"weight_bytes":weight_bytes,"iterations_per_pass":args.decode,
        "passes":args.passes,"ms_per_op":times,"median_ms_per_op":median,
        "max_repeat_logit_difference":repeat_difference,
        "after_session_driver_memory_bytes":driver_after,"after_session_reserved_live":allocator_after});
    serde_json::to_writer_pretty(File::create(args.output.join("summary.json"))?, &summary)?;
    Ok(())
}
fn main() -> Result<()> {
    let executable_sha256 = executable_sha256()?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    ensure!(
        args.decode > 0 && args.passes > 0 && args.prompt_tokens > 0 && args.prefill_chunk > 0,
        "nonzero workload required"
    );
    ensure!(
        !args.auto_placement || args.kernel == "production",
        "auto placement requires production dispatch"
    );
    ensure!(
        !args.output.join("summary.json").exists(),
        "refusing to overwrite completed run"
    );
    std::fs::create_dir_all(&args.output)?;
    let kernel = match args.kernel.as_str() {
        "generic" => Qgemv::Generic,
        "cubecl" => Qgemv::CubeClExpanded,
        "production" => Qgemv::CubeClProduction,
        "cubecl-packed" => Qgemv::CubeClPacked,
        "cubecl-next" => Qgemv::CubeClNext,
        "cubecl-q6" => Qgemv::CubeClFormat(candle_core::quantized::GgmlDType::Q6K),
        "cubecl-q4" => Qgemv::CubeClFormat(candle_core::quantized::GgmlDType::Q4K),
        "cubecl-q8" => Qgemv::CubeClFormat(candle_core::quantized::GgmlDType::Q8_0),
        "cubecl-q2" => Qgemv::CubeClFormat(candle_core::quantized::GgmlDType::Q2K),
        _ => anyhow::bail!(
            "kernel must be generic, production, cubecl, cubecl-packed, cubecl-next, cubecl-q6, cubecl-q4, cubecl-q8 or cubecl-q2"
        ),
    };
    let tokenizer =
        tokenizers::Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow::anyhow!("{e}"))?;
    let file = Arc::new(File::open(&args.model)?);
    let header = joshua::gguf_ext::read_header(&mut &*file)?;
    let mut rendered_prompt = None;
    let mut trace = if let Some(path) = &args.reference {
        serde_json::from_reader::<_, Trace>(File::open(path)?)?
    } else {
        let mut prompt = if let Some(path) = &args.prompt_file {
            std::fs::read_to_string(path)?
        } else {
            "Explain how a hash table stores and retrieves keys. Discuss collisions, resizing, memory use, and the tradeoffs between open addressing and separate chaining. Give a concrete example and explain why average lookup can be constant time. ".repeat(20)
        };
        if args.chat {
            let template = header
                .metadata
                .get("tokenizer.chat_template")
                .context("GGUF has no chat template")?
                .to_string()?
                .clone();
            let special = |key: &str| -> String {
                match header.metadata.get(key).and_then(|v| v.to_u64().ok()) {
                    Some(id) => tokenizer.decode(&[id as u32], false).unwrap(),
                    None => String::new(),
                }
            };
            let template = joshua::template::ChatTemplate::new(
                template,
                special("tokenizer.ggml.bos_token_id"),
                special("tokenizer.ggml.eos_token_id"),
            );
            prompt = template
                .render(&[joshua::types::ChatMessage::text("user", prompt)], None)
                .map_err(|e| anyhow::anyhow!("chat template: {e}"))?;
        }
        let ids = tokenizer
            .encode(prompt.clone(), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        let count = if args.prompt_file.is_some() {
            ids.len()
        } else {
            args.prompt_tokens
        };
        ensure!(count > 0 && ids.len() >= count, "prompt is too short");
        rendered_prompt = Some(prompt);
        Trace {
            prompt: ids[..count].to_vec(),
            inputs: Vec::new(),
        }
    };
    if let Some(path) = &args.decode_text_file {
        let text = std::fs::read_to_string(path)?;
        let ids = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(
            ids.len() >= args.decode,
            "continuation has fewer than {} tokens",
            args.decode
        );
        trace.inputs = ids.get_ids()[..args.decode].to_vec();
    }
    ensure!(
        trace.inputs.is_empty() || trace.inputs.len() == args.decode,
        "reference decode length differs"
    );
    if args.print_prompt {
        println!(
            "prompt_tokens={}\n{}",
            trace.prompt.len(),
            rendered_prompt.as_deref().unwrap_or("saved token trace")
        );
        return Ok(());
    }
    let mut dev = match args.backend.as_str() {
        "cpu" => Device::Cpu,
        "vulkan" => {
            let vk = if args.kernel == "production" {
                VulkanDevice::new(0)?
            } else {
                VulkanDevice::new_with_qgemv(0, kernel)?
            };
            ensure!(
                vk.limits().qgemv == kernel,
                "production selector overridden"
            );
            Device::Vulkan(vk)
        }
        #[cfg(feature = "opencl")]
        "opencl" => Device::new_opencl(0)?,
        #[cfg(feature = "sycl")]
        "sycl" => Device::new_sycl(0)?,
        _ => anyhow::bail!("backend unavailable: {} (build its feature)", args.backend),
    };
    let gpu_name = device_name(&dev);
    ensure!(
        dev.is_cpu() || (gpu_name.contains("Arc") && gpu_name.contains("B50")),
        "expected B50, found {gpu_name}"
    );
    if let Some(name) = &args.projection_tensor {
        return benchmark_projection(&args, &header, &file, &dev, name, &executable_sha256);
    }
    let mut dtype_elements = BTreeMap::<u32, usize>::new();
    for info in header.tensors.values() {
        *dtype_elements.entry(info.dtype).or_default() += info.elem_count();
    }
    header.validate_layout(file.metadata()?.len())?;
    ensure!(
        dtype_elements.keys().any(|t| matches!(t, 8 | 10 | 12 | 14)),
        "model has no CubeCL-supported Q8_0, Q2_K, Q4_K or Q6_K tensors"
    );
    let architecture = header
        .metadata
        .get("general.architecture")
        .context("missing architecture")?
        .to_string()?
        .clone();
    let cpu_experts = architecture.starts_with("deepseek4");
    let context = args
        .context
        .unwrap_or(trace.prompt.len() + args.decode.max(args.warm_decode));
    ensure!(
        context >= trace.prompt.len() + args.decode.max(args.warm_decode),
        "context must cover prompt, decode and warmup"
    );
    let mut model_dev = dev.clone();
    let mut expert_dev = if cpu_experts {
        Device::Cpu
    } else {
        dev.clone()
    };
    let mut cache_bytes = None;
    if args.auto_placement {
        // Engine construction resolves policy without loading a weight template.
        // Stage symlinks because Engine locates tokenizer.json beside the GGUF.
        let staged = args.output.join("engine-input");
        std::fs::create_dir_all(&staged)?;
        let model_link = staged.join("model.gguf");
        let tokenizer_link = staged.join("tokenizer.json");
        ensure!(
            !model_link.exists() && !tokenizer_link.exists(),
            "fresh output required"
        );
        std::os::unix::fs::symlink(std::fs::canonicalize(&args.model)?, &model_link)?;
        std::os::unix::fs::symlink(std::fs::canonicalize(&args.tokenizer)?, &tokenizer_link)?;
        let backend = match args.backend.as_str() {
            "cpu" => joshua::engine::ComputeBackend::Cpu,
            "vulkan" => joshua::engine::ComputeBackend::Vulkan,
            "opencl" => joshua::engine::ComputeBackend::OpenCl,
            "sycl" => joshua::engine::ComputeBackend::Sycl,
            _ => unreachable!(),
        };
        let engine = joshua::engine::Engine::with_options(
            &model_link,
            joshua::engine::EngineOptions {
                backend,
                dense_placement: if args.auto_dense_cpu {
                    joshua::DensePlacement::Cpu
                } else {
                    joshua::DensePlacement::Auto
                },
                n_ctx: context.try_into()?,
                prefill_chunk: args.prefill_chunk,
                ..Default::default()
            },
        )?;
        ensure!(
            engine.shared_weight_sessions() == 0,
            "policy resolution loaded weights"
        );
        dev = engine.device().clone();
        model_dev = engine.dense_device().clone();
        expert_dev = engine.expert_device().clone();
        cache_bytes = engine.device_expert_cache();
        println!(
            "auto dense={} experts={} cache_bytes={cache_bytes:?}",
            device_name(&model_dev),
            device_name(&expert_dev)
        );
    }
    ensure!(
        !args.auto_placement
            || cache_bytes.is_none()
            || !model_dev.is_cpu()
            || !expert_dev.is_cpu(),
        "automatic accelerator cache resolved to CPU-only placement"
    );
    if args.dense_cpu {
        model_dev = Device::Cpu;
        expert_dev = Device::Cpu;
        if let Some(mib) = args.expert_cache_mib {
            ensure!(
                cpu_experts && (dev.is_opencl() || dev.is_sycl()),
                "device expert cache requires DeepSeek on OpenCL/SYCL"
            );
            ensure!(
                mib > 0 && mib <= 12288,
                "B50 experiment cache budget must be 1..=12288 MiB"
            );
            cache_bytes = Some(mib * 1024 * 1024);
            expert_dev = dev.clone();
        }
    }
    println!(
        "architecture={architecture} context={context} routed_experts={}",
        if cpu_experts {
            "cpu-mmap"
        } else {
            "model-device"
        }
    );
    println!(
        "GPU={} kernel={} file_bytes={} elements_by_ggml_dtype={dtype_elements:?}",
        gpu_name,
        args.kernel,
        file.metadata()?.len()
    );
    // SAFETY: the GGUF is opened read-only and must remain unchanged for this run.
    // The model retains the mapping while its borrowed weight views are live.
    let mapping = Arc::new(unsafe { memmap2::Mmap::map(&*file)? });
    let load_start = Instant::now();
    let model = QuantizedModel::from_gguf_mmap_placed(
        header.to_candle_content()?,
        &mut &*file,
        &model_dev,
        &expert_dev,
        Some(mapping),
        Some(file.clone()),
        context,
        cache_bytes,
    )?;
    dev.synchronize()?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let loaded_memory = device_memory(&dev);
    let loaded_host_memory = host_memory_kib();
    let loaded_io = io_counters();
    println!(
        "loaded in {load_seconds:.2}s, reserved/live={loaded_memory:?}, RSS={} KiB",
        rss_kib()
    );
    if let Some(prompt) = &rendered_prompt {
        std::fs::write(args.output.join("rendered_prompt.txt"), prompt)?;
    }
    let warm_prompt_len = args.warm_prompt_tokens.unwrap_or(trace.prompt.len());
    ensure!(
        warm_prompt_len > 0 && warm_prompt_len <= trace.prompt.len(),
        "invalid warm prompt length"
    );
    println!("prompt_tokens={} prefill_chunk={} decode={} passes={} warm_prompt_tokens={warm_prompt_len}", trace.prompt.len(), args.prefill_chunk, args.decode, args.passes);
    // Warm every decode pipeline and cache shape, independent of timed traces.
    let vocab = {
        let mut session = model
            .new_session()
            .context("model must share weights across sessions")?;
        let prefill = prefill(
            &mut session,
            &model_dev,
            &trace.prompt[..warm_prompt_len],
            args.prefill_chunk,
            |_| {},
        )?;
        ensure!(
            prefill.iter().all(|v| v.is_finite()),
            "nonfinite prefill logits"
        );
        let mut next = argmax(&prefill);
        for step in 0..args.warm_decode {
            let logits = forward(&mut session, &model_dev, &[next], warm_prompt_len + step)?;
            ensure!(
                logits.iter().all(|v| v.is_finite()),
                "nonfinite warm decode logits at step {step}"
            );
            next = argmax(&logits);
        }
        prefill.len()
    };
    // Allocate the complete capture before sampling memory. Both timed passes
    // can now release their temporary host logits instead of retaining only
    // the first pass's allocations and forcing new scratch on the second.
    let mut reference_logits = vec![vec![0f32; vocab]; args.decode + 1];
    if trace.inputs.is_empty() {
        trace.inputs.reserve(args.decode);
    }
    dev.synchronize()?;
    let warm_memory = device_memory(&dev);
    let warm_rss = rss_kib();
    let warm_host_memory = host_memory_kib();
    let warm_io = io_counters();
    let mut pass_ms = Vec::new();
    let mut memory_after = Vec::new();
    let mut rss_after = Vec::new();
    let mut host_after = Vec::new();
    let mut driver_after = Vec::new();
    let mut cache_after = Vec::new();
    let mut io_after = Vec::new();
    let mut peak_memory = warm_memory;
    let mut peak_rss = warm_rss;
    let mut fallback_counts = Vec::new();
    let mut prefill_fallback_counts = Vec::new();
    let mut top1 = Vec::new();
    let mut max_repeat_difference = 0f32;
    let mut samples = Vec::with_capacity(
        args.passes * (args.decode + trace.prompt.len().div_ceil(args.prefill_chunk)),
    );
    let mut prefill_times = Vec::new();
    for pass in 0..args.passes {
        let mut session = model.new_session().context("shared session")?;
        let prefill_before = fallback_count(&dev);
        let prefill_start = Instant::now();
        let prefill = prefill(
            &mut session,
            &model_dev,
            &trace.prompt,
            args.prefill_chunk,
            |position| {
                samples.push(memory_sample(
                    &dev,
                    "prefill",
                    pass + 1,
                    position,
                    reference_logits.iter().map(|v| v.len() * 4).sum(),
                ));
            },
        )?;
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1e3;
        prefill_times.push(prefill_ms);
        prefill_fallback_counts.push(fallback_count(&dev) - prefill_before);
        ensure!(
            prefill.iter().all(|v| v.is_finite()),
            "nonfinite prefill logits"
        );
        let mut next = argmax(&prefill);
        let generating = trace.inputs.is_empty();
        if pass == 0 {
            top1.push(next);
            reference_logits[0].copy_from_slice(&prefill);
        } else {
            for (a, b) in prefill.iter().zip(&reference_logits[0]) {
                max_repeat_difference = max_repeat_difference.max((a - b).abs());
            }
        }
        let before = fallback_count(&dev);
        let mut ms = 0.;
        for step in 0..args.decode {
            let input = if generating {
                trace.inputs.push(next);
                next
            } else {
                trace.inputs[step]
            };
            let start = Instant::now();
            let output = forward(
                &mut session,
                &model_dev,
                &[input],
                trace.prompt.len() + step,
            )?;
            ensure!(output.iter().all(|v| v.is_finite()), "nonfinite logits");
            next = argmax(&output);
            ms += start.elapsed().as_secs_f64() * 1e3;
            if pass == 0 {
                top1.push(next);
                reference_logits[step + 1].copy_from_slice(&output);
            } else {
                for (a, b) in output.iter().zip(&reference_logits[step + 1]) {
                    max_repeat_difference = max_repeat_difference.max((a - b).abs());
                }
            }
            let memory = device_memory(&dev);
            peak_memory.0 = peak_memory.0.max(memory.0);
            peak_memory.1 = peak_memory.1.max(memory.1);
            let sample = memory_sample(
                &dev,
                "decode",
                pass + 1,
                trace.prompt.len() + step + 1,
                reference_logits.iter().map(|v| v.len() * 4).sum(),
            );
            peak_rss = peak_rss.max(sample.rss_kib);
            samples.push(sample);
            if (step + 1) % 8 == 0 {
                println!(
                    "pass={} decoded={} mean_ms/token={:.3}",
                    pass + 1,
                    step + 1,
                    ms / (step + 1) as f64
                );
            }
        }
        pass_ms.push(ms / args.decode as f64);
        fallback_counts.push(fallback_count(&dev) - before);
        drop(session);
        dev.synchronize()?;
        memory_after.push(device_memory(&dev));
        rss_after.push(rss_kib());
        host_after.push(host_memory_kib());
        driver_after.push(driver_memory_bytes());
        io_after.push(io_counters());
        cache_after.push(model.device_expert_cache().map(|r| format!("{r:?}")));
        println!("pass={} prefill_ms={prefill_ms:.2} decode_ms/token={:.3} reserved/live={:?} RSS={} KiB",pass+1,pass_ms[pass],memory_after[pass],rss_after[pass]);
    }
    let mut dump = BufWriter::new(File::create(args.output.join("logits.f32"))?);
    for row in &reference_logits {
        for v in row {
            dump.write_all(&v.to_le_bytes())?;
        }
    }
    dump.flush()?;
    serde_json::to_writer_pretty(File::create(args.output.join("trace.json"))?, &trace)?;
    let mut sorted = pass_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let median = (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.;
    let continuation = tokenizer
        .decode(&top1, true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    for sample in &samples {
        peak_memory.0 = peak_memory.0.max(sample.reserved_bytes);
        peak_memory.1 = peak_memory.1.max(sample.live_bytes);
        peak_rss = peak_rss.max(sample.rss_kib);
    }
    let mut summary = serde_json::json!({"model":args.model,"model_bytes":file.metadata()?.len(),"gpu":gpu_name,"kernel":args.kernel,
        "executable_sha256":executable_sha256,"dispatch":format!("{kernel:?}"),
        "cubecl_formats":match args.kernel.as_str() {"cubecl"=>vec![8,10,12],"production"|"cubecl-packed"|"cubecl-next"=>vec![8,10,12,14],"cubecl-q6"=>vec![14],"cubecl-q4"=>vec![12],"cubecl-q8"=>vec![8],"cubecl-q2"=>vec![10],_=>vec![]},
        "architecture":architecture,"expert_placement":if cpu_experts {"cpu-mmap"}else{"model-device"},
        "context":context,"warm_decode":args.warm_decode,"engram_disabled":std::env::var_os("JOSHUA_NO_ENGRAM").is_some(),
        "elements_by_ggml_dtype":dtype_elements,"prompt_tokens":trace.prompt.len(),"prefill_chunk":args.prefill_chunk,"decode_tokens":args.decode,"passes":args.passes,
        "load_seconds":load_seconds,"loaded_reserved_live":loaded_memory,"warm_reserved_live":warm_memory,"peak_reserved_live":peak_memory,
        "loaded_host_memory_kib":loaded_host_memory,"warm_host_memory_kib":warm_host_memory,"after_session_host_memory_kib":host_after,
        "after_session_reserved_live":memory_after,"warm_rss_kib":warm_rss,"peak_rss_kib":peak_rss,"after_session_rss_kib":rss_after,
        "ms_per_token":pass_ms,"median_ms_per_token":median,"tokens_per_second":1000./median,"decode_cpu_fallbacks":fallback_counts,"prefill_cpu_fallbacks":prefill_fallback_counts,
        "max_repeat_logit_difference":max_repeat_difference,"logit_rows":reference_logits.len(),"vocab":reference_logits[0].len(),"top1":top1,"greedy_text":continuation});
    if !dev.is_vulkan() {
        summary["dispatch"] = "backend-native".into();
        summary["cubecl_formats"] = serde_json::json!([]);
    }
    summary["decode_text_file"] = serde_json::to_value(&args.decode_text_file)?;
    summary["loaded_io_counters"] = serde_json::to_value(loaded_io)?;
    summary["warm_io_counters"] = serde_json::to_value(warm_io)?;
    summary["after_session_io_counters"] = serde_json::to_value(io_after)?;
    summary["explicit_dense_cpu"] = args.dense_cpu.into();
    summary["explicit_expert_cache_mib"] = serde_json::to_value(args.expert_cache_mib)?;
    summary["placement_probe_skipped"] = (!joshua::auto_placement::probe_enabled()).into();
    summary["auto_dense_cpu"] = args.auto_dense_cpu.into();
    summary["auto_placement"] = args.auto_placement.into();
    summary["dense_device"] = device_name(&model_dev).into();
    summary["expert_device"] = device_name(&expert_dev).into();
    summary["expert_cache_budget_bytes"] = serde_json::to_value(cache_bytes)?;
    summary["expert_cache_report"] =
        serde_json::to_value(model.device_expert_cache().map(|r| format!("{r:?}")))?;
    summary["backend"] = args.backend.clone().into();
    summary["allocator_stats_available"] = dev.is_vulkan().into();
    summary["after_session_driver_memory_bytes"] = serde_json::to_value(driver_after)?;
    summary["after_session_expert_cache_reports"] = serde_json::to_value(cache_after)?;
    summary["driver_memory_after"] = serde_json::to_value(driver_memory_bytes())?;
    summary["attention_output_groups"] = serde_json::to_value(
        header
            .metadata
            .get(&format!("{architecture}.attention.output_group_count"))
            .and_then(|v| v.to_u32().map(u64::from).or_else(|_| v.to_u64()).ok()),
    )?;
    summary["attention_output_a_gguf_shapes"] = serde_json::to_value(
        header
            .tensors
            .iter()
            .filter(|(name, _)| name.ends_with(".attn_output_a.weight"))
            .map(|(name, info)| (name, &info.dims))
            .collect::<BTreeMap<_, _>>(),
    )?;
    summary["prompt_file"] = serde_json::to_value(&args.prompt_file)?;
    summary["chat"] = args.chat.into();
    summary["memory_samples"] = serde_json::to_value(&samples)?;
    summary["prefill_ms"] = serde_json::to_value(&prefill_times)?;
    summary["reference_logits_bytes"] = reference_logits
        .iter()
        .map(|v| v.len() * 4)
        .sum::<usize>()
        .into();
    summary["warm_prompt_tokens"] = warm_prompt_len.into();
    serde_json::to_writer_pretty(File::create(args.output.join("summary.json"))?, &summary)?;
    println!(
        "median_ms/token={median:.3} tokens/sec={:.3} repeat_max_abs={max_repeat_difference}",
        1000. / median
    );
    Ok(())
}
