//! Ollama's model fetching, against joshua's model directory:
//! `POST /api/pull`, `HEAD`/`POST /api/blobs/{digest}` and `POST
//! /api/create` with `files`.
//!
//! **Pull** reads a model name the way Ollama does,
//! `[host/][namespace/]model[:tag]` (registry.ollama.ai, `library` and
//! `latest` by default; `hf.co/<org>/<repo>:<quant>` works too, as Hugging
//! Face speaks the same protocol), fetches its manifest and downloads the
//! manifest's model layer, checking it against its SHA256 digest.  The GGUF
//! lands in the model directory as `<model>.gguf`, the name joshua serves
//! it under (the tag and namespace are dropped, as everywhere else on the
//! Ollama routes).  The manifest's template, parameters and system layers
//! are not used, as joshua renders the GGUF's own template; projector and
//! adapter layers are skipped with a status line saying so.  An `https://`
//! URL to a `.gguf` file is downloaded as it is.  A plain `http://` registry
//! or URL needs `insecure: true`, as in Ollama.  Progress streams as
//! Ollama's newline-delimited status objects.
//!
//! **Blobs** are kept in `<model dir>/.blobs/sha256-<hex>`, which the
//! inventory does not list.  `POST /api/blobs/{digest}` stores an upload
//! whose SHA256 matches its digest; **create** with `files` then puts the
//! one GGUF the files name into the directory as `<model>.gguf` (a hard
//! link where the filesystem allows, else a copy).  Modelfile settings
//! that would change the model (`template`, `system`, `parameters`,
//! `messages`, `adapters`, `quantize`) are refused rather than ignored.

// Without the `pull` feature the download half is compiled out and the
// route answers 501.
#![cfg_attr(not(feature = "pull"), allow(dead_code, unused_imports))]

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use axum::{
    body::Body,
    extract::{Path as UrlPath, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::StreamExt;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::server::{ApiError, AppState, CancelOnDrop};

/// The registry a bare model name is pulled from.
const DEFAULT_REGISTRY: &str = "registry.ollama.ai";
/// A manifest's GGUF layer.
const MODEL_LAYER: &str = "application/vnd.ollama.image.model";
/// How often a download reports its progress.
const PROGRESS_EVERY: Duration = Duration::from_millis(250);

/// Where a pull fetches from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullSource {
    /// A registry speaking Ollama's (Docker distribution) protocol.
    Registry {
        /// `https://host` (or `http://` when insecure).
        base: String,
        /// `namespace/model`.
        repository: String,
        tag: String,
        /// The name the model is served under: the model's own.
        stem: String,
    },
    /// A GGUF at a URL.
    Url { url: String, stem: String },
}

impl PullSource {
    /// Read a pull's model name.
    pub fn parse(name: &str, insecure: bool) -> Result<Self, String> {
        let name = name.trim();
        for (scheme, secure) in [("https://", true), ("http://", false)] {
            let Some(rest) = name.strip_prefix(scheme) else {
                continue;
            };
            if !secure && !insecure {
                return Err(format!("`{name}` is plain http: pass `insecure: true` to pull it"));
            }
            let path = rest.split(['?', '#']).next().unwrap_or_default();
            let file = path.rsplit('/').next().unwrap_or_default();
            let stem = file
                .strip_suffix(".gguf")
                .or_else(|| file.strip_suffix(".GGUF"))
                .filter(|stem| valid_stem(stem))
                .ok_or_else(|| format!("`{name}` does not name a .gguf file"))?;
            return Ok(Self::Url {
                url: name.to_string(),
                stem: stem.to_string(),
            });
        }
        let (path, tag) = match name.rsplit_once(':') {
            // A colon in the last path segment is the tag; one before it is a
            // host's port.
            Some((path, tag)) if !tag.contains('/') => (path, tag),
            _ => (name, "latest"),
        };
        let mut parts: Vec<&str> = path.split('/').collect();
        let host = match parts.first() {
            Some(first)
                if parts.len() > 1
                    && (first.contains('.') || first.contains(':') || *first == "localhost") =>
            {
                parts.remove(0)
            }
            _ => DEFAULT_REGISTRY,
        };
        let repository = match parts.as_slice() {
            [model] => format!("library/{model}"),
            _ => parts.join("/"),
        };
        let stem = parts.last().copied().unwrap_or_default();
        let segment_ok = |s: &str| {
            !s.is_empty()
                && s != "."
                && s != ".."
                && s.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
        };
        let host_ok = !host.is_empty()
            && !host.starts_with(['.', '-', ':'])
            && !host.contains("..")
            && host.chars().all(|c| c.is_ascii_alphanumeric() || ".-:".contains(c));
        if !host_ok
            || !valid_stem(stem)
            || !parts.iter().all(|p| segment_ok(p))
            || !tag.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
            || tag.is_empty()
        {
            return Err(format!("invalid model name `{name}`"));
        }
        let scheme = if insecure { "http" } else { "https" };
        Ok(Self::Registry {
            base: format!("{scheme}://{host}"),
            repository,
            tag: tag.to_string(),
            stem: stem.to_string(),
        })
    }

    fn stem(&self) -> &str {
        match self {
            Self::Registry { stem, .. } | Self::Url { stem, .. } => stem,
        }
    }
}

/// A name usable as `<stem>.gguf` in the model directory.
fn valid_stem(stem: &str) -> bool {
    !stem.is_empty()
        && !stem.starts_with('.')
        && !stem.contains(['/', '\\'])
        && !stem.contains("..")
}

/// A blob digest's hex, when it is `sha256:<64 hex digits>`.
fn digest_hex(digest: &str) -> Option<&str> {
    let hex = digest.strip_prefix("sha256:").or_else(|| digest.strip_prefix("sha256-"))?;
    (hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit())).then_some(hex)
}

/// Where the blob store keeps `digest`.
fn blob_path(model_dir: &Path, digest: &str) -> Result<PathBuf, String> {
    let hex = digest_hex(digest)
        .ok_or_else(|| format!("invalid digest `{digest}`: expected sha256:<64 hex digits>"))?;
    Ok(model_dir.join(".blobs").join(format!("sha256-{}", hex.to_ascii_lowercase())))
}

/// The model directory, or the refusal a server without one gives.
fn model_dir(state: &AppState) -> Result<PathBuf, ApiError> {
    state
        .manager
        .as_ref()
        .map(|manager| manager.model_dir.clone())
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::FORBIDDEN,
                "model management is disabled on this server",
                "management_disabled",
            )
        })
}

/// Whether `path` starts with GGUF's magic.
fn is_gguf(path: &Path) -> bool {
    let mut magic = [0u8; 4];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .is_ok_and(|()| &magic == b"GGUF")
}

/// Put `source` (a blob) into the model directory as `<stem>.gguf`: a hard
/// link where the filesystem allows, else a copy, through a temporary name
/// so a reader never sees half a file.
fn place(source: &Path, model_dir: &Path, stem: &str) -> std::io::Result<PathBuf> {
    let target = model_dir.join(format!("{stem}.gguf"));
    let temp = model_dir.join(format!(".{stem}.gguf.placing"));
    let _ = std::fs::remove_file(&temp);
    if std::fs::hard_link(source, &temp).is_err() {
        std::fs::copy(source, &temp)?;
    }
    std::fs::rename(&temp, &target)?;
    Ok(target)
}

// ─── pull ───────────────────────────────────────────────────────────────────

/// Why a pull stopped.
#[derive(Debug)]
enum PullError {
    Cancelled,
    Failed(String),
}

impl From<String> for PullError {
    fn from(message: String) -> Self {
        Self::Failed(message)
    }
}

/// Copy `reader` into `file`, hashing it and reporting progress as
/// `status` with `digest` (when given), stopping if `cancel` is set.
/// Returns the byte count and the SHA256 hex.
#[allow(clippy::too_many_arguments)]
fn download(
    mut reader: impl Read,
    file: &mut File,
    status: &str,
    digest: Option<&str>,
    total: Option<u64>,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(Value),
) -> Result<(u64, String), PullError> {
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut completed = 0u64;
    let mut reported = Instant::now() - PROGRESS_EVERY;
    let report = |completed: u64, progress: &mut dyn FnMut(Value)| {
        let mut line = json!({ "status": status, "completed": completed });
        if let Some(digest) = digest {
            line["digest"] = Value::from(digest);
        }
        if let Some(total) = total {
            line["total"] = Value::from(total);
        }
        progress(line);
    };
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(PullError::Cancelled);
        }
        let n = reader
            .read(&mut buffer)
            .map_err(|e| format!("download failed: {e}"))?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        file.write_all(&buffer[..n])
            .map_err(|e| format!("writing the download: {e}"))?;
        completed += n as u64;
        if reported.elapsed() >= PROGRESS_EVERY {
            report(completed, progress);
            reported = Instant::now();
        }
    }
    report(completed, progress);
    file.flush().map_err(|e| format!("writing the download: {e}"))?;
    Ok((completed, format!("{:x}", hash.finalize())))
}

#[cfg(feature = "pull")]
fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(300))
        .user_agent(concat!("joshua/", env!("CARGO_PKG_VERSION")))
        .build()
}

#[cfg(feature = "pull")]
fn get(agent: &ureq::Agent, url: &str, accept: Option<&str>) -> Result<ureq::Response, String> {
    let mut request = agent.get(url);
    if let Some(accept) = accept {
        request = request.set("Accept", accept);
    }
    request.call().map_err(|e| match e {
        ureq::Error::Status(404, _) => format!("{url}: not found"),
        ureq::Error::Status(code, _) => format!("{url}: HTTP {code}"),
        other => format!("{url}: {other}"),
    })
}

/// Pull `source` into `model_dir`, reporting progress; returns the file
/// written.
#[cfg(feature = "pull")]
fn pull_blocking(
    model_dir: &Path,
    source: &PullSource,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(Value),
) -> Result<PathBuf, PullError> {
    let agent = agent();
    match source {
        PullSource::Registry { base, repository, tag, stem } => {
            progress(json!({ "status": "pulling manifest" }));
            let reply = get(
                &agent,
                &format!("{base}/v2/{repository}/manifests/{tag}"),
                Some("application/vnd.docker.distribution.manifest.v2+json"),
            )?;
            // A manifest is a few kilobytes; a reply far larger is not one.
            let manifest: Value = serde_json::from_reader(reply.into_reader().take(4 << 20))
                .map_err(|e| format!("reading the manifest: {e}"))?;
            let layers = manifest["layers"].as_array().cloned().unwrap_or_default();
            let mut model = None;
            for layer in &layers {
                let kind = layer["mediaType"].as_str().unwrap_or_default();
                match kind {
                    MODEL_LAYER if model.is_none() => model = Some(layer),
                    MODEL_LAYER => return Err("the manifest has more than one model layer".to_string().into()),
                    "application/vnd.ollama.image.projector" | "application/vnd.ollama.image.adapter" => {
                        let what = kind.rsplit('.').next().unwrap_or(kind);
                        progress(json!({
                            "status": format!("skipping the {what} layer: joshua does not load it from a pull"),
                        }));
                    }
                    _ => {}
                }
            }
            let layer = model.ok_or_else(|| "the manifest has no model layer".to_string())?;
            let digest = layer["digest"].as_str().unwrap_or_default();
            let hex = digest_hex(digest)
                .ok_or_else(|| format!("the manifest's model layer has an invalid digest `{digest}`"))?
                .to_ascii_lowercase();
            let size = layer["size"].as_u64();
            let blob = blob_path(model_dir, digest)?;
            let status = format!("pulling {}", &hex[..12]);
            let have = std::fs::metadata(&blob).ok().map(|m| m.len());
            if have.is_some() && have == size {
                let size = size.unwrap_or_default();
                progress(json!({ "status": status, "digest": digest, "total": size, "completed": size }));
            } else {
                std::fs::create_dir_all(blob.parent().unwrap_or(model_dir))
                    .map_err(|e| format!("creating the blob store: {e}"))?;
                let partial = blob.with_extension("partial");
                let mut file =
                    File::create(&partial).map_err(|e| format!("creating {}: {e}", partial.display()))?;
                let response = get(&agent, &format!("{base}/v2/{repository}/blobs/{digest}"), None)?;
                let fetched = download(
                    response.into_reader(),
                    &mut file,
                    &status,
                    Some(digest),
                    size,
                    cancel,
                    progress,
                );
                drop(file);
                let (bytes, got) = match fetched {
                    Ok(done) => done,
                    Err(e) => {
                        let _ = std::fs::remove_file(&partial);
                        return Err(e);
                    }
                };
                progress(json!({ "status": "verifying sha256 digest" }));
                if got != hex || size.is_some_and(|size| size != bytes) {
                    let _ = std::fs::remove_file(&partial);
                    return Err(format!("digest mismatch: expected {digest}, got sha256:{got}").into());
                }
                std::fs::rename(&partial, &blob).map_err(|e| format!("storing the blob: {e}"))?;
            }
            if !is_gguf(&blob) {
                return Err("the model layer is not a GGUF file".to_string().into());
            }
            progress(json!({ "status": "writing manifest" }));
            Ok(place(&blob, model_dir, stem).map_err(|e| format!("placing the model: {e}"))?)
        }
        PullSource::Url { url, stem } => {
            let response = get(&agent, url, None)?;
            let total = response
                .header("Content-Length")
                .and_then(|n| n.parse::<u64>().ok());
            let target = model_dir.join(format!("{stem}.gguf"));
            let partial = model_dir.join(format!(".{stem}.gguf.partial"));
            let mut file =
                File::create(&partial).map_err(|e| format!("creating {}: {e}", partial.display()))?;
            let status = format!("downloading {stem}.gguf");
            let fetched = download(response.into_reader(), &mut file, &status, None, total, cancel, progress);
            drop(file);
            if let Err(e) = fetched {
                let _ = std::fs::remove_file(&partial);
                return Err(e);
            }
            if !is_gguf(&partial) {
                let _ = std::fs::remove_file(&partial);
                return Err(format!("{url} is not a GGUF file").into());
            }
            std::fs::rename(&partial, &target).map_err(|e| format!("placing the model: {e}"))?;
            Ok(target)
        }
    }
}

#[cfg(not(feature = "pull"))]
fn pull_blocking(
    _model_dir: &Path,
    _source: &PullSource,
    _cancel: &AtomicBool,
    _progress: &mut dyn FnMut(Value),
) -> Result<PathBuf, PullError> {
    unreachable!("the pull route answers 501 without the `pull` feature")
}

/// `POST /api/pull`: fetch a model from a registry or an https URL into the
/// model directory, streaming Ollama's status lines unless `stream` is
/// false.
pub async fn pull(State(state): State<AppState>, Json(body): Json<Value>) -> Result<Response, ApiError> {
    let name = body
        .get("model")
        .or_else(|| body.get("name"))
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("missing 'model'"))?;
    let insecure = body.get("insecure").and_then(Value::as_bool).unwrap_or(false);
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(true);
    let source = PullSource::parse(name, insecure).map_err(ApiError::bad_request)?;
    let dir = model_dir(&state)?;
    if !cfg!(feature = "pull") {
        return Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "this server was built without the `pull` feature; place the .gguf in the model \
             directory and POST /v1/models/load",
            "pull_not_implemented",
        ));
    }
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    let cancel = CancelOnDrop::new();
    let flag = cancel.flag();
    let task = tokio::task::spawn_blocking(move || {
        let mut progress = |line: Value| {
            let _ = tx.send(line);
        };
        let result = pull_blocking(&dir, &source, &flag, &mut progress);
        match &result {
            Ok(_) => progress(json!({ "status": "success" })),
            Err(PullError::Failed(message)) => progress(json!({ "error": message })),
            Err(PullError::Cancelled) => {}
        }
        result.map(|_| source.stem().to_string())
    });
    if stream {
        let lines = futures_util::stream::unfold((rx, cancel), |(mut rx, cancel)| async move {
            rx.recv()
                .await
                .map(|line| (Ok::<String, std::convert::Infallible>(format!("{line}\n")), (rx, cancel)))
        });
        let mut response = Body::from_stream(lines).into_response();
        response
            .headers_mut()
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/x-ndjson"));
        return Ok(response);
    }
    // The whole pull, answered once it is done.
    let drain = async { while rx.recv().await.is_some() {} };
    let (result, ()) = tokio::join!(task, drain);
    drop(cancel);
    match result.map_err(|e| ApiError::internal(e.to_string()))? {
        Ok(_) => Ok(Json(json!({ "status": "success" })).into_response()),
        // The reason is the client's to read: what failed upstream, not a
        // fault of this server's to hide.
        Err(PullError::Failed(message)) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("pull failed: {message}"),
            "server_error",
        )),
        Err(PullError::Cancelled) => Err(ApiError::internal("pull cancelled")),
    }
}

// ─── blobs ──────────────────────────────────────────────────────────────────

/// `HEAD /api/blobs/{digest}`: 200 when the blob store holds it, else 404.
pub async fn blob_exists(
    State(state): State<AppState>,
    UrlPath(digest): UrlPath<String>,
) -> Result<StatusCode, ApiError> {
    let path = blob_path(&model_dir(&state)?, &digest).map_err(ApiError::bad_request)?;
    Ok(if path.is_file() { StatusCode::OK } else { StatusCode::NOT_FOUND })
}

/// `POST /api/blobs/{digest}`: store the body as that blob, if its SHA256
/// is the digest.  201 when stored.
pub async fn upload_blob(
    State(state): State<AppState>,
    UrlPath(digest): UrlPath<String>,
    body: Body,
) -> Result<StatusCode, ApiError> {
    let dir = model_dir(&state)?;
    let path = blob_path(&dir, &digest).map_err(ApiError::bad_request)?;
    let expected = digest_hex(&digest).unwrap_or_default().to_ascii_lowercase();
    if path.is_file() {
        return Ok(StatusCode::OK);
    }
    let store = path.parent().unwrap_or(&dir).to_path_buf();
    tokio::fs::create_dir_all(&store)
        .await
        .map_err(|e| ApiError::internal(format!("creating the blob store: {e}")))?;
    let partial = store.join(format!(".upload-{}", uuid::Uuid::new_v4().simple()));
    let written = async {
        use tokio::io::AsyncWriteExt;
        let mut file = tokio::fs::File::create(&partial)
            .await
            .map_err(|e| ApiError::internal(format!("creating the blob: {e}")))?;
        let mut hash = Sha256::new();
        let mut chunks = body.into_data_stream();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.map_err(|e| ApiError::bad_request(format!("reading the upload: {e}")))?;
            hash.update(&chunk);
            file.write_all(&chunk)
                .await
                .map_err(|e| ApiError::internal(format!("writing the blob: {e}")))?;
        }
        file.flush()
            .await
            .map_err(|e| ApiError::internal(format!("writing the blob: {e}")))?;
        Ok::<_, ApiError>(format!("{:x}", hash.finalize()))
    }
    .await;
    let got = match written {
        Ok(got) => got,
        Err(e) => {
            let _ = tokio::fs::remove_file(&partial).await;
            return Err(e);
        }
    };
    if got != expected {
        let _ = tokio::fs::remove_file(&partial).await;
        return Err(ApiError::bad_request(format!(
            "digest mismatch: the upload is sha256:{got}, not {digest}"
        )));
    }
    tokio::fs::rename(&partial, &path)
        .await
        .map_err(|e| ApiError::internal(format!("storing the blob: {e}")))?;
    Ok(StatusCode::CREATED)
}

// ─── create with files ──────────────────────────────────────────────────────

/// `POST /api/create` with `files`: put the GGUF blob the files name into
/// the model directory as `<model>.gguf`.  Returns the status lines.
pub fn create_from_files(state: &AppState, stem: &str, body: &Value) -> Result<Vec<Value>, ApiError> {
    for field in ["template", "system", "parameters", "messages", "adapters", "quantize"] {
        let set = match body.get(field) {
            None | Some(Value::Null) => false,
            Some(Value::String(s)) => !s.is_empty(),
            Some(Value::Object(o)) => !o.is_empty(),
            Some(Value::Array(a)) => !a.is_empty(),
            Some(_) => true,
        };
        if set {
            return Err(ApiError::bad_request(format!(
                "`{field}` is not supported by create: joshua serves a GGUF with its own template \
                 and settings, so a Modelfile may hold only FROM"
            )));
        }
    }
    if body.get("from").is_some_and(|v| !v.is_null()) {
        return Err(ApiError::bad_request("give `files` or `from`, not both"));
    }
    let files = body
        .get("files")
        .and_then(Value::as_object)
        .ok_or_else(|| ApiError::bad_request("`files` must map file names to blob digests"))?;
    let ggufs: Vec<(&String, &str)> = files
        .iter()
        .map(|(name, digest)| (name, digest.as_str().unwrap_or_default()))
        .collect();
    let [(name, digest)] = ggufs.as_slice() else {
        return Err(ApiError::bad_request(
            "`files` must name exactly one .gguf file: joshua does not load split models, \
             projectors or safetensors through create",
        ));
    };
    if !name.to_ascii_lowercase().ends_with(".gguf") {
        return Err(ApiError::bad_request(format!(
            "`{name}` is not a .gguf file: joshua serves GGUF models only"
        )));
    }
    let dir = model_dir(state)?;
    let blob = blob_path(&dir, digest).map_err(ApiError::bad_request)?;
    if !blob.is_file() {
        return Err(ApiError::bad_request(format!(
            "blob {digest} not found: upload it first with POST /api/blobs/{digest}"
        )));
    }
    if !is_gguf(&blob) {
        return Err(ApiError::bad_request(format!("blob {digest} is not a GGUF file")));
    }
    place(&blob, &dir, stem).map_err(|e| ApiError::internal(format!("placing the model: {e}")))?;
    Ok(vec![
        json!({ "status": format!("using existing layer {digest}") }),
        json!({ "status": "writing manifest" }),
        json!({ "status": "success" }),
    ])
}

/// Ollama's create answer: the status lines, streamed as newline-delimited
/// JSON unless `stream` is false, when only the last is sent.
pub fn status_response(lines: Vec<Value>, stream: bool) -> Response {
    if !stream {
        let last = lines.into_iter().last().unwrap_or_else(|| json!({ "status": "success" }));
        return Json(last).into_response();
    }
    let body: String = lines.iter().map(|line| format!("{line}\n")).collect();
    let mut response = body.into_response();
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/x-ndjson"));
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry(base: &str, repository: &str, tag: &str, stem: &str) -> PullSource {
        PullSource::Registry {
            base: base.into(),
            repository: repository.into(),
            tag: tag.into(),
            stem: stem.into(),
        }
    }

    #[test]
    fn pull_names_read_as_ollama_reads_them() {
        let parse = |name| PullSource::parse(name, false).unwrap();
        assert_eq!(
            parse("llama3.2"),
            registry("https://registry.ollama.ai", "library/llama3.2", "latest", "llama3.2")
        );
        assert_eq!(
            parse("llama3.2:3b"),
            registry("https://registry.ollama.ai", "library/llama3.2", "3b", "llama3.2")
        );
        assert_eq!(
            parse("someone/model:q4"),
            registry("https://registry.ollama.ai", "someone/model", "q4", "model")
        );
        assert_eq!(
            parse("hf.co/org/Repo-GGUF:Q4_K_M"),
            registry("https://hf.co", "org/Repo-GGUF", "Q4_K_M", "Repo-GGUF")
        );
        assert_eq!(
            PullSource::parse("localhost:5000/library/tiny", true).unwrap(),
            registry("http://localhost:5000", "library/tiny", "latest", "tiny")
        );
        assert_eq!(
            parse("https://example.com/models/tiny-q4.gguf?download=1"),
            PullSource::Url {
                url: "https://example.com/models/tiny-q4.gguf?download=1".into(),
                stem: "tiny-q4".into()
            }
        );
        for bad in ["", "../x", "a/../b", "x:a/b:", "https://example.com/model.bin", "http://e.com/m.gguf"] {
            assert!(PullSource::parse(bad, false).is_err(), "{bad}");
        }
    }

    #[test]
    fn digests_name_blobs_inside_the_store() {
        let dir = Path::new("/models");
        let hex = "a".repeat(64);
        assert_eq!(
            blob_path(dir, &format!("sha256:{hex}")).unwrap(),
            dir.join(".blobs").join(format!("sha256-{hex}"))
        );
        assert!(blob_path(dir, "sha256:../../etc").is_err());
        assert!(blob_path(dir, &format!("md5:{hex}")).is_err());
    }
}
