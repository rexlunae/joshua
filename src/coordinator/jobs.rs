use super::*;
use crate::job_admission::BoundedOutput;
use futures_util::{stream, StreamExt, TryStreamExt};
use serde_json::Value;
use sha2::{Digest, Sha256};

const MAX_INPUTS: usize = 2048;
const PARALLEL_BATCHES: usize = 4;

impl Coordinator {
    pub(super) async fn refresh_completed_audio(&self) {
        let Ok(refresh) = self.audio_refresh.try_lock() else {
            return;
        };
        let stale = self.workers.iter().any(|worker| {
            let status = worker.status();
            status.healthy
                && status.observed_audio_in_flight > 0
                && status.audio_probe_epoch != status.audio_completion_epoch
                && worker.audio_dispatched.load(Ordering::Acquire) == 0
        });
        if stale {
            let _refresh = AudioRefreshGuard {
                lock: Some(refresh),
                released: &self.released,
            };
            self.refresh().await;
        }
    }

    pub(super) fn try_pick_audio(self: &Arc<Self>, exclude: &[usize]) -> Pick {
        let _lock = self
            .pick_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut busy = false;
        let mut tried = false;
        let count = self.workers.len();
        let start = self.rotation.fetch_add(1, Ordering::Relaxed) % count;
        for offset in 0..count {
            let index = (start + offset) % count;
            let worker = &self.workers[index];
            let status = worker.status();
            let Some(_) = status
                .info
                .as_ref()
                .filter(|info| status.healthy && info.capabilities.transcriptions)
            else {
                continue;
            };
            if exclude.contains(&index) {
                tried = true;
                continue;
            }
            let local = worker.audio_dispatched.load(Ordering::Acquire);
            if local.saturating_add(status.observed_audio_in_flight) > 0 {
                busy = true;
                continue;
            }
            if self.config.worker_slots.is_some_and(|cap| {
                worker.dispatched.load(Ordering::Acquire) + status.external_in_flight >= cap.max(1)
            }) {
                busy = true;
                continue;
            }
            worker.audio_dispatched.fetch_add(1, Ordering::AcqRel);
            worker.dispatched.fetch_add(1, Ordering::AcqRel);
            return Pick::Slot(SlotGuard {
                coordinator: self.clone(),
                index,
                model: None,
                work: 0,
                audio: true,
            });
        }
        if busy {
            Pick::Busy
        } else if tried {
            Pick::Reject(Reject::Exhausted)
        } else {
            Pick::Reject(Reject::Unavailable)
        }
    }

    pub(super) async fn embedding_jobs(
        self: &Arc<Self>,
        headers: &HeaderMap,
        body: &Bytes,
    ) -> Option<Response> {
        let request: crate::types::EmbeddingRequest = serde_json::from_slice(body).ok()?;
        let texts = request.input.into_vec();
        if texts.is_empty() {
            return None;
        }
        let model = request.model;
        let excluded: Vec<_> = self
            .workers
            .iter()
            .enumerate()
            .filter_map(|(index, worker)| {
                let status = worker.status();
                (!(status.healthy
                    && status
                        .info
                        .as_ref()
                        .is_some_and(|info| embedding_owner(info, &model).is_some())))
                .then_some(index)
            })
            .collect();
        if excluded.len() == self.workers.len() {
            return None;
        }
        let request_id = request_id_from(headers);
        let result = self
            .run_embedding_jobs(body, &model, texts, &excluded, &request_id)
            .await;
        Some(with_request_id(
            result.unwrap_or_else(|response| response),
            &request_id,
        ))
    }

    async fn run_embedding_jobs(
        self: &Arc<Self>,
        body: &Bytes,
        model: &str,
        texts: Vec<String>,
        excluded: &[usize],
        request_id: &str,
    ) -> Result<Response, Response> {
        if texts.len() > MAX_INPUTS {
            return Err(job_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "too many embedding inputs",
                "job_input_limit",
            ));
        }
        let work = RequestWork {
            context: None,
            units: body.len() as u64,
        };
        let waiting = Instant::now();
        let first = self
            .acquire("/v1/embeddings", Some(model), &work, excluded)
            .await
            .map_err(reject_response)?;
        let first_wait = waiting.elapsed();
        let anchor = first.index;
        let info = self.workers[anchor]
            .status()
            .info
            .clone()
            .ok_or_else(owner_changed)?;
        let identity = embedding_owner(&info, model)
            .cloned()
            .ok_or_else(owner_changed)?;
        let owners: Vec<_> = self
            .workers
            .iter()
            .enumerate()
            .map(|(index, worker)| {
                let status = worker.status();
                if !status.healthy || excluded.contains(&index) {
                    return None;
                }
                let candidate = status.info.as_ref()?;
                let job = embedding_owner(candidate, model)?;
                let compatible = index == anchor
                    || (identity.revision.is_some()
                        && job.revision == identity.revision
                        && job.backend == identity.backend
                        && job.input_format == identity.input_format);
                compatible.then_some(job.clone())
            })
            .collect();
        let max_inputs = owners
            .iter()
            .flatten()
            .map(|job| job.max_inputs)
            .min()
            .unwrap_or(1)
            .clamp(1, 32);
        let max_bytes = owners
            .iter()
            .flatten()
            .map(|job| job.max_input_bytes)
            .min()
            .unwrap_or(1)
            .clamp(1, 256 << 10);
        let mut envelope: Value = serde_json::from_slice(body).map_err(|_| owner_changed())?;
        let batches = partition(&mut envelope, texts, max_inputs, max_bytes)?;
        let count: usize = batches.iter().map(|batch| batch.1).sum();
        let started = Instant::now();
        let job_request_id = Uuid::new_v4();
        let mut first = Some(first);
        let jobs = batches
            .into_iter()
            .enumerate()
            .map(|(ordinal, (offset, length, inputs))| {
                let slot = first.take().map(|slot| (slot, first_wait));
                let owners = &owners;
                let envelope = &envelope;
                async move {
                    let mut batch = envelope.clone();
                    batch["input"] = Value::Array(inputs);
                    let payload =
                        Bytes::from(serde_json::to_vec(&batch).map_err(|_| invalid_result())?);
                    drop(batch);
                    let job_id = format!(
                        "job-{:x}",
                        Sha256::digest(format!("{job_request_id}:embedding:{ordinal}"))
                    );
                    let response = self
                        .embedding_batch(model, owners, slot, payload, &job_id)
                        .await?;
                    if !response.status().is_success() {
                        return Err(response);
                    }
                    let bytes =
                        axum::body::to_bytes(response.into_body(), self.config.max_response_bytes)
                            .await
                            .map_err(|_| invalid_result())?;
                    let result: Value =
                        serde_json::from_slice(&bytes).map_err(|_| invalid_result())?;
                    Ok((offset, length, bytes.len(), result))
                }
            });
        let parallelism = PARALLEL_BATCHES.min(self.config.queue_depth.saturating_add(1));
        let mut pending = stream::iter(jobs).buffer_unordered(parallelism);
        let mut data = vec![Value::Null; count];
        let mut prompt_tokens = 0u32;
        let mut buffered = 0usize;
        let mut dimensions = None;
        while let Some((offset, length, bytes, mut result)) = pending.try_next().await? {
            buffered = buffered
                .checked_add(bytes)
                .filter(|total| *total <= self.config.max_response_bytes.min(8 << 20))
                .ok_or_else(invalid_result)?;
            if result["model"] != model || result["object"] != "list" {
                return Err(invalid_result());
            }
            let tokens = result["usage"]["prompt_tokens"]
                .as_u64()
                .and_then(|tokens| u32::try_from(tokens).ok())
                .ok_or_else(invalid_result)?;
            if result["usage"]["total_tokens"].as_u64() != Some(tokens as u64)
                || result["usage"]["completion_tokens"].as_u64().unwrap_or(0) != 0
            {
                return Err(invalid_result());
            }
            prompt_tokens = prompt_tokens
                .checked_add(tokens)
                .ok_or_else(invalid_result)?;
            let rows = result["data"].as_array_mut().ok_or_else(invalid_result)?;
            if rows.len() != length {
                return Err(invalid_result());
            }
            for mut row in rows.drain(..) {
                let local = row["index"]
                    .as_u64()
                    .and_then(|index| usize::try_from(index).ok())
                    .filter(|index| *index < length)
                    .ok_or_else(invalid_result)?;
                if !data[offset + local].is_null() || row["object"] != "embedding" {
                    return Err(invalid_result());
                }
                let shape = match &row["embedding"] {
                    Value::Array(vector) if vector.iter().all(Value::is_number) => {
                        (false, vector.len())
                    }
                    Value::String(vector) => (true, vector.len()),
                    _ => return Err(invalid_result()),
                };
                if dimensions.is_some_and(|expected| expected != shape) {
                    return Err(invalid_result());
                }
                dimensions = Some(shape);
                row["index"] = json!(offset + local);
                data[offset + local] = row;
            }
        }
        let mut result = json!({"object": "list", "model": model,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens}});
        result["data"] = Value::Array(data);
        let mut output = BoundedOutput {
            bytes: Vec::new(),
            limit: self.config.max_response_bytes.min(8 << 20),
        };
        serde_json::to_writer(&mut output, &result).map_err(|_| invalid_result())?;
        tracing::info!(
            request_id,
            inputs = count,
            response_bytes = output.bytes.len(),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "embedding jobs completed"
        );
        Ok(([(header::CONTENT_TYPE, "application/json")], output.bytes).into_response())
    }

    async fn embedding_batch(
        self: &Arc<Self>,
        model: &str,
        owners: &[Option<JobModel>],
        mut reserved: Option<(SlotGuard, Duration)>,
        payload: Bytes,
        job_id: &str,
    ) -> Result<Response, Response> {
        let mut excluded: Vec<_> = owners
            .iter()
            .enumerate()
            .filter_map(|(index, owner)| owner.is_none().then_some(index))
            .collect();
        let work = RequestWork {
            context: None,
            units: payload.len() as u64,
        };
        let mut queue_wait = Duration::ZERO;
        let mut attempt = 0;
        loop {
            let waiting = Instant::now();
            let slot = match reserved.take() {
                Some((slot, elapsed)) => {
                    queue_wait += elapsed;
                    slot
                }
                None => {
                    let slot = self
                        .acquire("/v1/embeddings", Some(model), &work, &excluded)
                        .await
                        .map_err(reject_response)?;
                    queue_wait += waiting.elapsed();
                    slot
                }
            };
            attempt += 1;
            let index = slot.index;
            let owner = owners[index].as_ref().ok_or_else(owner_changed)?;
            let current = self.workers[index].status().info.clone();
            if !current
                .as_ref()
                .is_some_and(|info| info.embedding_jobs.contains(owner))
            {
                return Err(owner_changed());
            }
            let log = RequestLog {
                owner: Some(owner.owner.clone()),
                response_limit: owner.max_output_bytes.min(2 << 20),
                request_id: job_id.to_string(),
                worker: self.workers[index].endpoint.base.clone(),
                path: "/v1/embeddings",
                attempt,
                queue_ms: queue_wait.as_millis() as u64,
                started: Instant::now(),
            };
            match self
                .forward(slot, "/v1/embeddings", payload.clone(), None, log)
                .await
            {
                Ok(response) | Err(Failure::Final(response)) => return Ok(response),
                Err(Failure::Retry { safe: true, .. }) => excluded.push(index),
                Err(Failure::Retry { .. }) => return Err(reject_response(Reject::Exhausted)),
            }
        }
    }
}

fn embedding_owner<'info>(info: &'info WorkerInfo, model: &str) -> Option<&'info JobModel> {
    if !info.capabilities.embeddings
        || !info.model_ids().any(|id| id == model)
        || info
            .slots_of(Some(model))
            .is_some_and(|slots| !slots.embeddings)
    {
        return None;
    }
    info.embedding_jobs.iter().find(|job| {
        job.model == model
            && job.input_format == "text"
            && job.max_inputs > 0
            && job.max_input_bytes > 0
            && job.max_output_bytes > 0
    })
}

fn partition(
    envelope: &mut Value,
    texts: Vec<String>,
    max_inputs: usize,
    max_bytes: usize,
) -> Result<Vec<(usize, usize, Vec<Value>)>, Response> {
    envelope["input"] = json!([]);
    let overhead = serde_json::to_vec(envelope)
        .map_err(|_| invalid_result())?
        .len();
    let mut batches = Vec::new();
    let mut inputs = Vec::new();
    let mut bytes = overhead;
    let mut offset = 0;
    for text in texts {
        let size = serde_json::to_vec(&text)
            .map_err(|_| invalid_result())?
            .len()
            + 1;
        if overhead.saturating_add(size) > max_bytes {
            return Err(job_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "embedding input exceeds subbatch byte limit",
                "job_input_limit",
            ));
        }
        if inputs.len() == max_inputs || bytes + size > max_bytes {
            let length = inputs.len();
            batches.push((offset, length, std::mem::take(&mut inputs)));
            offset += length;
            bytes = overhead;
        }
        inputs.push(Value::String(text));
        bytes += size;
    }
    if !inputs.is_empty() {
        let length = inputs.len();
        batches.push((offset, length, inputs));
    }
    Ok(batches)
}

fn job_error(status: StatusCode, message: &str, code: &str) -> Response {
    error_response(status, message, "server_error", Some(code))
}

fn invalid_result() -> Response {
    job_error(
        StatusCode::BAD_GATEWAY,
        "invalid or oversized embedding job response",
        "invalid_job_result",
    )
}

fn owner_changed() -> Response {
    job_error(
        StatusCode::CONFLICT,
        "embedding model ownership changed; retry the whole request",
        "job_owner_changed",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_bounded_partition_preserves_escaped_text_and_unknown_options() {
        let mut envelope =
            json!({"model": "m", "input": [], "encoding_format": "base64", "normalize": false});
        let texts: Vec<_> = ["\"\n", "日本語", "", "fourth"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let expected = texts.clone();
        let batches =
            partition(&mut envelope, texts, 2, 120).unwrap_or_else(|_| panic!("valid partition"));
        let mut actual = Vec::new();
        for (offset, length, inputs) in batches {
            assert_eq!(offset, actual.len());
            assert!(length <= 2);
            envelope["input"] = Value::Array(inputs.clone());
            assert!(serde_json::to_vec(&envelope).unwrap().len() <= 120);
            assert_eq!(envelope["normalize"], false);
            actual.extend(
                inputs
                    .into_iter()
                    .map(|input| input.as_str().unwrap().to_string()),
            );
        }
        assert_eq!(actual, expected);
    }
}
