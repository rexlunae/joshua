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
//! link where the filesystem allows, else a copy).  **Create** with `from`
//! and `adapters` (a Modelfile's `FROM` + `ADAPTER`) keeps one LoRA adapter
//! as a model of its own name on its base, in `<model dir>/.adapters/<name>/`
//! (see [`AdapterModel`]).  Other Modelfile settings that would change the
//! model (`template`, `system`, `parameters`, `messages`, `quantize`) are
//! refused rather than ignored.

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
    http::StatusCode,
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
        // `insecure` is for a registry of one's own: the public ones are
        // always spoken to over https.
        let public = matches!(host, DEFAULT_REGISTRY | "hf.co" | "huggingface.co");
        let scheme = if insecure && !public { "http" } else { "https" };
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

/// A file being written under a temporary name of its own (so concurrent
/// requests never share one), removed when dropped unless it was moved
/// into place: a failed, cancelled or abandoned request leaves nothing
/// behind.
struct Staged {
    path: PathBuf,
    kept: bool,
}

impl Staged {
    fn new(dir: &Path, name: &str) -> Self {
        Self {
            path: dir.join(format!(".{name}.{}.partial", uuid::Uuid::new_v4().simple())),
            kept: false,
        }
    }

    /// Move the file to `to`, replacing what is there.
    fn persist(mut self, to: &Path) -> std::io::Result<()> {
        std::fs::rename(&self.path, to)?;
        self.kept = true;
        Ok(())
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        if !self.kept {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Put `source` (a blob) into the model directory as `<stem>.gguf`: a hard
/// link where the filesystem allows, else a copy, through a temporary name
/// so a reader never sees half a file.
fn place(source: &Path, model_dir: &Path, stem: &str) -> std::io::Result<PathBuf> {
    let target = model_dir.join(format!("{stem}.gguf"));
    let staged = Staged::new(model_dir, &format!("{stem}.gguf"));
    if std::fs::hard_link(source, &staged.path).is_err() {
        std::fs::copy(source, &staged.path)?;
    }
    staged.persist(&target)?;
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

/// Whether `ip` is an address on the public internet, as opposed to this
/// host, its private networks or link-local ones: where a pull may connect
/// unless the server allows private hosts.
fn is_public(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, ..] = v4.octets();
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_multicast()
                || a == 0
                // Carrier-grade NAT, 100.64.0.0/10.
                || (a == 100 && (64..128).contains(&b))
                // IETF protocol assignments, 192.0.0.0/24.
                || (a == 192 && b == 0 && v4.octets()[2] == 0)
                // Benchmarking, 198.18.0.0/15, often routed internally.
                || (a == 198 && (b & 0xfe) == 18)
                // Reserved, 240.0.0.0/4.
                || a >= 240)
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_public(IpAddr::V4(v4));
            }
            let first = v6.segments()[0];
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                // Unique local fc00::/7, link-local fe80::/10 and the
                // deprecated site-local fec0::/10.
                || (first & 0xfe00) == 0xfc00
                || (first & 0xffc0) == 0xfe80
                || (first & 0xffc0) == 0xfec0)
        }
    }
}

/// Whether `ip` is one of this host's own addresses (a public one
/// included), where services the firewall keeps from other hosts may
/// listen.  A UDP socket "connected" to the address (no packet is sent)
/// takes the source address the kernel's route picks, which is the address
/// itself only when the route is a local one; unlike a test bind, this is
/// not fooled by `ip_nonlocal_bind`.
fn is_own_address(ip: std::net::IpAddr) -> bool {
    use std::net::{Ipv4Addr, Ipv6Addr, UdpSocket};
    let any: std::net::IpAddr = if ip.is_ipv4() {
        Ipv4Addr::UNSPECIFIED.into()
    } else {
        Ipv6Addr::UNSPECIFIED.into()
    };
    UdpSocket::bind((any, 0))
        .and_then(|socket| {
            socket.connect((ip, 9))?;
            socket.local_addr()
        })
        .is_ok_and(|local| local.ip() == ip)
}

/// The HTTP client a pull uses.  Unless `allow_private`, names are
/// resolved to public addresses only, every connection (redirects
/// included) checked at the address it actually dials, so a pull cannot be
/// pointed at services on this host or its network.
#[cfg(feature = "pull")]
fn agent(allow_private: bool) -> ureq::Agent {
    let mut builder = ureq::AgentBuilder::new();
    if !allow_private {
        builder = builder.resolver(|netloc: &str| -> std::io::Result<Vec<std::net::SocketAddr>> {
            use std::net::ToSocketAddrs;
            let addrs: Vec<_> = netloc.to_socket_addrs()?.collect();
            let public: Vec<_> = addrs
                .iter()
                .copied()
                .filter(|a| is_public(a.ip()) && !is_own_address(a.ip()))
                .collect();
            if public.is_empty() && !addrs.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    format!(
                        "{netloc} is a private address; start the server with --allow-private-pull \
                         to pull from it"
                    ),
                ));
            }
            Ok(public)
        });
    }
    builder
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
    allow_private: bool,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(Value),
) -> Result<PathBuf, PullError> {
    let agent = agent(allow_private);
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
                let partial = Staged::new(blob.parent().unwrap_or(model_dir), &format!("sha256-{hex}"));
                let mut file = File::create(&partial.path)
                    .map_err(|e| format!("creating {}: {e}", partial.path.display()))?;
                let response = get(&agent, &format!("{base}/v2/{repository}/blobs/{digest}"), None)?;
                let (bytes, got) = download(
                    response.into_reader(),
                    &mut file,
                    &status,
                    Some(digest),
                    size,
                    cancel,
                    progress,
                )?;
                drop(file);
                progress(json!({ "status": "verifying sha256 digest" }));
                if got != hex || size.is_some_and(|size| size != bytes) {
                    return Err(format!("digest mismatch: expected {digest}, got sha256:{got}").into());
                }
                partial.persist(&blob).map_err(|e| format!("storing the blob: {e}"))?;
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
            let partial = Staged::new(model_dir, &format!("{stem}.gguf"));
            let mut file = File::create(&partial.path)
                .map_err(|e| format!("creating {}: {e}", partial.path.display()))?;
            let status = format!("downloading {stem}.gguf");
            download(response.into_reader(), &mut file, &status, None, total, cancel, progress)?;
            drop(file);
            if !is_gguf(&partial.path) {
                return Err(format!("{url} is not a GGUF file").into());
            }
            partial.persist(&target).map_err(|e| format!("placing the model: {e}"))?;
            Ok(target)
        }
    }
}

#[cfg(not(feature = "pull"))]
fn pull_blocking(
    _model_dir: &Path,
    _source: &PullSource,
    _allow_private: bool,
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
    let allow_private = state.manager.as_ref().is_some_and(|m| m.allow_private_pull);
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
        let result = pull_blocking(&dir, &source, allow_private, &flag, &mut progress);
        match &result {
            Ok(_) => progress(json!({ "status": "success" })),
            Err(PullError::Failed(message)) => progress(json!({ "error": message })),
            Err(PullError::Cancelled) => {}
        }
        result.map(|_| source.stem().to_string())
    });
    if stream {
        return Ok(crate::ollama::ndjson(rx, cancel));
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
    // Removed when the request ends without storing it, the client going
    // away mid-upload included (the handler's future is then dropped).
    let partial = Staged::new(&store, "upload");
    let written = async {
        use tokio::io::AsyncWriteExt;
        let mut file = tokio::fs::File::create(&partial.path)
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
    let got = written?;
    if got != expected {
        return Err(ApiError::bad_request(format!(
            "digest mismatch: the upload is sha256:{got}, not {digest}"
        )));
    }
    partial
        .persist(&path)
        .map_err(|e| ApiError::internal(format!("storing the blob: {e}")))?;
    Ok(StatusCode::CREATED)
}

// ─── create with files ──────────────────────────────────────────────────────

/// `POST /api/create` with `files`: put the GGUF blob the files name into
/// the model directory as `<model>.gguf`.  Returns the status lines.
pub fn create_from_files(state: &AppState, stem: &str, body: &Value) -> Result<Vec<Value>, ApiError> {
    refuse_settings(body, &[])?;
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

/// Whether a create request sets `field` (to anything but an empty value).
pub fn sets(body: &Value, field: &str) -> bool {
    match body.get(field) {
        None | Some(Value::Null) => false,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(_) => true,
    }
}

/// Refuse the Modelfile settings that would change the model, but those in
/// `allowed`, rather than ignore them.
pub fn refuse_settings(body: &Value, allowed: &[&str]) -> Result<(), ApiError> {
    for field in ["template", "system", "parameters", "messages", "adapters", "quantize"] {
        if !allowed.contains(&field) && sets(body, field) {
            return Err(ApiError::bad_request(format!(
                "`{field}` is not supported by create: joshua serves a GGUF with its own template \
                 and settings, so a Modelfile may hold only FROM (and ADAPTER, with FROM naming \
                 a model)"
            )));
        }
    }
    Ok(())
}

// ─── create with adapters ───────────────────────────────────────────────────

/// Where created adapter models are kept, inside the model directory.
const ADAPTERS: &str = ".adapters";
/// The file in an adapter model's directory naming its base and adapter.
const ADAPTER_MANIFEST: &str = "joshua-adapter.json";

/// A model `create` made from a base model and a LoRA adapter (Ollama's
/// `FROM` + `ADAPTER`), kept as `<model dir>/.adapters/<name>/`.  It is
/// served under its name as an adapter-named model on its base (see
/// [`crate::server::select_generation`]), the adapter loaded on first use.
#[derive(Debug, Clone)]
pub struct AdapterModel {
    pub name: String,
    /// The base model's id.
    pub base: String,
    /// The adapter: a GGUF file, or a PEFT adapter's directory.
    pub path: PathBuf,
    /// The model's own directory.
    pub dir: PathBuf,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct AdapterManifest {
    from: String,
    /// The adapter, relative to the model's directory (`.` for PEFT).
    adapter: String,
}

/// A name a created adapter model may have: a bare name, as a model file's
/// stem is.
fn adapter_name_ok(name: &str) -> bool {
    !name.is_empty() && !name.starts_with('.') && !name.contains(['/', '\\']) && !name.contains("..")
}

/// The adapter model `create` made as `name` in `model_dir`, if any.
pub fn adapter_model(model_dir: &Path, name: &str) -> Option<AdapterModel> {
    if !adapter_name_ok(name) {
        return None;
    }
    let dir = model_dir.join(ADAPTERS).join(name);
    let manifest: AdapterManifest =
        serde_json::from_slice(&std::fs::read(dir.join(ADAPTER_MANIFEST)).ok()?).ok()?;
    if manifest.adapter.contains(['/', '\\']) || (manifest.adapter != "." && manifest.adapter.contains("..")) {
        return None;
    }
    Some(AdapterModel {
        name: name.to_string(),
        base: manifest.from,
        path: dir.join(&manifest.adapter),
        dir,
    })
}

/// Every adapter model `create` made in `model_dir`, by name.
pub fn adapter_models(model_dir: &Path) -> Vec<AdapterModel> {
    let Ok(entries) = std::fs::read_dir(model_dir.join(ADAPTERS)) else {
        return Vec::new();
    };
    let mut out: Vec<AdapterModel> = entries
        .flatten()
        .filter_map(|entry| adapter_model(model_dir, entry.file_name().to_str()?))
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// A directory being filled under a temporary name, removed when dropped
/// unless it was moved into place.
struct StagedDir(Option<PathBuf>);

impl StagedDir {
    fn new(root: &Path, name: &str) -> std::io::Result<Self> {
        let path = root.join(format!(".{name}.{}.partial", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&path)?;
        Ok(Self(Some(path)))
    }

    fn path(&self) -> &Path {
        self.0.as_deref().expect("a staged directory until persisted")
    }

    /// Move the directory to `to`, replacing what is there.
    fn persist(mut self, to: &Path) -> std::io::Result<()> {
        let from = self.0.take().expect("persisted once");
        if to.exists() {
            std::fs::remove_dir_all(to)?;
        }
        std::fs::rename(&from, to).inspect_err(|_| {
            let _ = std::fs::remove_dir_all(&from);
        })
    }
}

impl Drop for StagedDir {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

/// A hard link to `source` at `to` where the filesystem allows, else a copy.
fn link_or_copy(source: &Path, to: &Path) -> std::io::Result<()> {
    if std::fs::hard_link(source, to).is_err() {
        std::fs::copy(source, to)?;
    }
    Ok(())
}

/// `POST /api/create` with `from` and `adapters`: keep the adapter the
/// files name as model `stem` on the model `base` (which the caller checked
/// is one).  The adapter is one GGUF LoRA adapter, or a Hugging Face PEFT
/// adapter's `adapter_model.safetensors` with its `adapter_config.json`.
/// It is checked against the base when it is first loaded.  Returns the
/// status lines.
pub fn create_adapter(state: &AppState, stem: &str, base: &str, body: &Value) -> Result<Vec<Value>, ApiError> {
    refuse_settings(body, &["adapters"])?;
    if !adapter_name_ok(stem) {
        return Err(ApiError::bad_request(format!("invalid model name `{stem}`")));
    }
    let adapters = body
        .get("adapters")
        .and_then(Value::as_object)
        .ok_or_else(|| ApiError::bad_request("`adapters` must map file names to blob digests"))?;
    let dir = model_dir(state)?;
    let blob = |digest: &str| -> Result<PathBuf, ApiError> {
        let path = blob_path(&dir, digest).map_err(ApiError::bad_request)?;
        if !path.is_file() {
            return Err(ApiError::bad_request(format!(
                "blob {digest} not found: upload it first with POST /api/blobs/{digest}"
            )));
        }
        Ok(path)
    };
    let file_name = |name: &str| name.rsplit(['/', '\\']).next().unwrap_or(name).to_ascii_lowercase();
    let named = |test: &dyn Fn(&str) -> bool| -> Vec<(&String, &str)> {
        adapters
            .iter()
            .filter(|(name, _)| test(&file_name(name)))
            .map(|(name, digest)| (name, digest.as_str().unwrap_or_default()))
            .collect()
    };
    let ggufs = named(&|n| n.ends_with(".gguf"));
    let weights = named(&|n| n.ends_with(".safetensors"));
    let configs = named(&|n| n == "adapter_config.json");
    let (files, adapter): (Vec<(PathBuf, &str)>, &str) =
        match (ggufs.as_slice(), weights.as_slice(), configs.as_slice()) {
            ([(name, digest)], [], _) => {
                let path = blob(digest)?;
                let facts = is_gguf(&path)
                    .then(|| crate::model_info::GgufFacts::read(&path).ok())
                    .flatten();
                if facts.and_then(|f| f.kind).as_deref() != Some("adapter") {
                    return Err(ApiError::bad_request(format!(
                        "`{name}` is not a LoRA adapter GGUF (its general.type is not `adapter`)"
                    )));
                }
                (vec![(path, "adapter")], "adapter")
            }
            ([], [(_, w)], [(_, c)]) => (
                vec![
                    (blob(w)?, crate::lora_peft::WEIGHTS),
                    (blob(c)?, crate::lora_peft::CONFIG),
                ],
                ".",
            ),
            _ => {
                return Err(ApiError::bad_request(
                    "`adapters` must name one LoRA adapter: a GGUF adapter, or a Hugging Face \
                     PEFT adapter's adapter_model.safetensors with its adapter_config.json",
                ))
            }
        };
    let internal = |e: std::io::Error| ApiError::internal(format!("keeping the adapter: {e}"));
    let root = dir.join(ADAPTERS);
    std::fs::create_dir_all(&root).map_err(internal)?;
    let staged = StagedDir::new(&root, stem).map_err(internal)?;
    for (source, name) in &files {
        link_or_copy(source, &staged.path().join(name)).map_err(internal)?;
    }
    let manifest = AdapterManifest { from: base.to_string(), adapter: adapter.to_string() };
    std::fs::write(
        staged.path().join(ADAPTER_MANIFEST),
        serde_json::to_vec_pretty(&manifest).expect("a manifest serializes"),
    )
    .map_err(internal)?;
    staged.persist(&root.join(stem)).map_err(internal)?;
    let mut lines: Vec<Value> = adapters
        .values()
        .map(|digest| json!({ "status": format!("using existing layer {}", digest.as_str().unwrap_or_default()) }))
        .collect();
    lines.push(json!({ "status": "writing manifest" }));
    lines.push(json!({ "status": "success" }));
    Ok(lines)
}

/// Remove the adapter model `name`; `false` when there is none.
pub fn delete_adapter(model_dir: &Path, name: &str) -> std::io::Result<bool> {
    match adapter_model(model_dir, name) {
        Some(model) => std::fs::remove_dir_all(&model.dir).map(|()| true),
        None => Ok(false),
    }
}

/// Copy the adapter model `from` as `to`; `false` when there is no `from`.
pub fn copy_adapter(model_dir: &Path, from: &str, to: &str) -> std::io::Result<bool> {
    let Some(model) = adapter_model(model_dir, from) else {
        return Ok(false);
    };
    if !adapter_name_ok(to) {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("invalid model name `{to}`")));
    }
    let root = model_dir.join(ADAPTERS);
    let staged = StagedDir::new(&root, to)?;
    for entry in std::fs::read_dir(&model.dir)?.flatten() {
        link_or_copy(&entry.path(), &staged.path().join(entry.file_name()))?;
    }
    staged.persist(&root.join(to))?;
    Ok(true)
}

/// Ollama's create answer: the status lines, streamed as newline-delimited
/// JSON unless `stream` is false, when only the last is sent.
pub fn status_response(lines: Vec<Value>, stream: bool) -> Response {
    if !stream {
        let last = lines.into_iter().last().unwrap_or_else(|| json!({ "status": "success" }));
        return Json(last).into_response();
    }
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    for line in lines {
        let _ = tx.send(line);
    }
    crate::ollama::ndjson(rx, ())
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
        // `insecure` never takes a public registry off https.
        assert_eq!(
            PullSource::parse("llama3.2", true).unwrap(),
            registry("https://registry.ollama.ai", "library/llama3.2", "latest", "llama3.2")
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
    fn only_public_addresses_are_public() {
        for ip in ["127.0.0.1", "10.1.2.3", "192.168.0.9", "172.20.0.1", "169.254.169.254",
                   "100.100.0.1", "0.0.0.0", "::1", "fd00::1", "fe80::1", "::ffff:10.0.0.1",
                   "198.18.0.1", "198.19.255.1", "192.0.0.8", "240.0.0.1", "fec0::1"] {
            assert!(!is_public(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["1.1.1.1", "104.21.0.1", "198.20.0.1", "2606:4700::1111"] {
            assert!(is_public(ip.parse().unwrap()), "{ip}");
        }
        assert!(is_own_address("127.0.0.1".parse().unwrap()));
        assert!(!is_own_address("1.1.1.1".parse().unwrap()));
        // This host's address on its outbound route (not loopback, the
        // case a public address of its own takes) is its own; skipped on
        // a host with no route out.
        let outbound = std::net::UdpSocket::bind(("0.0.0.0", 0))
            .and_then(|s| s.connect(("1.1.1.1", 9)).and_then(|()| s.local_addr()));
        if let Ok(local) = outbound {
            assert!(!local.ip().is_loopback(), "{local}");
            assert!(is_own_address(local.ip()), "{local}");
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
