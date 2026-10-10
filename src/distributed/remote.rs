//! Serve a controller-deployed pipeline through [`crate::Engine`]: each
//! engine generation session is one pipeline session, so chat templates,
//! sampling, streaming and KV reuse work unchanged on remote stages.

use std::{
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
};

use super::pipeline::{Input, Pipeline};
use crate::npu::{NpuBackend, NpuSession};

/// The engine's remote session source. Sessions share the pipeline, which
/// runs one batch at a time.
pub struct PipelineBackend {
    pipeline: Arc<Mutex<Pipeline>>,
    name: String,
}

impl PipelineBackend {
    pub fn new(pipeline: Pipeline) -> Self {
        let name = format!("pipeline({} stages)", pipeline.plan().stages.len());
        Self {
            pipeline: Arc::new(Mutex::new(pipeline)),
            name,
        }
    }

    /// Remote sessions the workers hold at once.
    pub fn capacity(&self) -> usize {
        lock(&self.pipeline).plan().limits.sessions
    }

    /// Largest prompt piece one forward call carries.
    pub fn chunk(&self) -> usize {
        lock(&self.pipeline).plan().limits.chunk
    }

    /// Context limit the workers reserved KV for.
    pub fn context(&self) -> usize {
        lock(&self.pipeline).plan().limits.context
    }

    pub fn plan(&self) -> super::pipeline::Plan {
        lock(&self.pipeline).plan().clone()
    }
}

impl Drop for PipelineBackend {
    fn drop(&mut self) {
        // Unloading the model: release the workers' stages cleanly before
        // returning, so they can take the next model.
        let _ = lock(&self.pipeline).stop();
    }
}

/// A failed pipeline refuses every later call, so a panic while it was held
/// leaves nothing inconsistent to protect.
fn lock(pipeline: &Mutex<Pipeline>) -> MutexGuard<'_, Pipeline> {
    pipeline
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl NpuBackend for PipelineBackend {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn create_session(
        &self,
        _model_path: &Path,
        _n_ctx: u32,
    ) -> std::result::Result<Box<dyn NpuSession>, String> {
        let mut pipeline = lock(&self.pipeline);
        let id = pipeline.open().map_err(|e| format!("{e:#}"))?;
        let plan = pipeline.plan();
        Ok(Box::new(PipelineSession {
            pipeline: Arc::clone(&self.pipeline),
            id: Some(id),
            vocab: plan.vocab,
            chunk: plan.limits.chunk,
        }))
    }
}

struct PipelineSession {
    pipeline: Arc<Mutex<Pipeline>>,
    id: Option<u64>,
    vocab: usize,
    chunk: usize,
}

impl NpuSession for PipelineSession {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn forward(&mut self, tokens: &[u32], pos: usize) -> std::result::Result<Vec<f32>, String> {
        let id = self.id.ok_or("pipeline session was closed")?;
        if tokens.is_empty() {
            return Err("empty forward".into());
        }
        let mut pipeline = lock(&self.pipeline);
        let mut logits = Vec::new();
        for (i, piece) in tokens.chunks(self.chunk).enumerate() {
            let mut out = pipeline
                .forward_batch(&[Input {
                    session: id,
                    offset: pos + i * self.chunk,
                    tokens: piece.to_vec(),
                }])
                .map_err(|e| format!("{e:#}"))?;
            logits = out.pop().ok_or("pipeline returned no logits")?;
        }
        Ok(logits)
    }

    /// Workers keep no way to rewind a session, so a reset replaces it.
    fn reset(&mut self) -> bool {
        let mut pipeline = lock(&self.pipeline);
        if let Some(id) = self.id.take() {
            if pipeline.close(id).is_err() {
                return false;
            }
        }
        match pipeline.open() {
            Ok(id) => {
                self.id = Some(id);
                true
            }
            Err(_) => false,
        }
    }
}

impl Drop for PipelineSession {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            // Best effort: a failed pipeline has already dropped every session.
            let _ = lock(&self.pipeline).close(id);
        }
    }
}
