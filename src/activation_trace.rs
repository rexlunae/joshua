//! Opt-in bounded activation capture for backend numerical diagnosis.
use candle_core::{DType, Result, Tensor};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    },
};
static CONFIG: OnceLock<Option<(PathBuf, usize, Option<usize>)>> = OnceLock::new();
static NEXT: AtomicUsize = AtomicUsize::new(0);
thread_local! {
    static POSITION: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}
pub(crate) fn position(offset: usize) {
    POSITION.with(|p| p.set(Some(offset)));
}
// The disabled path returns before inspecting, allocating or reading a tensor.
#[inline]
pub(crate) fn record(label: &str, x: &Tensor) -> Result<()> {
    let Some((root, limit, position)) = CONFIG.get_or_init(|| {
        let root = std::env::var_os("JOSHUA_ACTIVATION_TRACE")?;
        let limit = std::env::var("JOSHUA_ACTIVATION_TRACE_LIMIT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        let position = std::env::var("JOSHUA_ACTIVATION_TRACE_POSITION")
            .ok()
            .and_then(|s| s.parse().ok());
        Some((PathBuf::from(root), limit, position))
    }) else {
        return Ok(());
    };
    if position.is_some() && POSITION.with(|p| p.get()) != *position {
        return Ok(());
    }
    let index = NEXT.fetch_add(1, Ordering::Relaxed);
    if index >= *limit {
        return Ok(());
    }
    std::fs::create_dir_all(root)?;
    let name = format!("{index:05}-{label}");
    let values = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(root.join(format!("{name}.f32")), bytes)?;
    std::fs::write(
        root.join(format!("{name}.json")),
        serde_json::to_vec(&serde_json::json!({"label":label,"shape":x.dims()}))
            .map_err(candle_core::Error::wrap)?,
    )?;
    Ok(())
}
