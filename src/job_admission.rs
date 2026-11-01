use std::sync::Arc;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub(crate) const EMBEDDING_BYTES: usize = 2 << 20;
pub(crate) const AUDIO_BYTES: usize = 25 << 20;

#[derive(Clone)]
pub(crate) struct Admission {
    embeddings: Arc<Semaphore>,
    audio: Arc<Semaphore>,
}

impl Default for Admission {
    fn default() -> Self {
        Self {
            embeddings: Arc::new(Semaphore::new(32)),
            audio: Arc::new(Semaphore::new(2)),
        }
    }
}

pub(crate) struct JobLease {
    _permit: OwnedSemaphorePermit,
}

pub(crate) struct BoundedOutput {
    pub(crate) bytes: Vec<u8>,
    pub(crate) limit: usize,
}

impl std::io::Write for BoundedOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other("job response limit exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(crate) fn json_response(
    value: &impl serde::Serialize,
    limit: usize,
) -> Result<Response, serde_json::Error> {
    let mut output = BoundedOutput {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut output, value)?;
    Ok((
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        output.bytes,
    )
        .into_response())
}

pub(crate) async fn admit(
    State(admission): State<Admission>,
    mut request: Request,
    next: Next,
) -> Response {
    let (slots, limit) = match request.uri().path() {
        "/v1/embeddings" => (admission.embeddings, EMBEDDING_BYTES),
        "/v1/audio/transcriptions" => (admission.audio, AUDIO_BYTES),
        _ => return next.run(request).await,
    };
    let Ok(permit) = slots.try_acquire_owned() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [("retry-after", "1")],
            axum::Json(serde_json::json!({"error": {
                "message": "job payload budget is full; retry shortly",
                "type": "overloaded", "code": "job_budget_full", "param": null
            }})),
        )
            .into_response();
    };
    let lease = Arc::new(JobLease { _permit: permit });
    request.extensions_mut().insert(lease.clone());
    let limit = if request.uri().path() == "/v1/embeddings"
        && request
            .headers()
            .contains_key(crate::coordinator::JOB_OWNER_HEADER)
    {
        limit.min(256 << 10)
    } else {
        limit
    };
    DefaultBodyLimit::max(limit).apply(&mut request);
    let response = next.run(request).await;
    let (parts, body) = response.into_parts();
    let stream = futures_util::stream::unfold(
        (body.into_data_stream(), lease),
        |(mut body, lease)| async move { body.next().await.map(|chunk| (chunk, (body, lease))) },
    );
    Response::from_parts(parts, Body::from_stream(stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{middleware, routing::post, Router};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn app(admission: Admission) -> Router {
        Router::new()
            .route(
                "/v1/embeddings",
                post(|body: axum::body::Bytes| async move { body }),
            )
            .route(
                "/v1/audio/transcriptions",
                post(|body: axum::body::Bytes| async move { body }),
            )
            .layer(middleware::from_fn_with_state(admission, admit))
    }

    #[tokio::test]
    async fn budgets_precede_body_polling_and_audio_is_independent() {
        let admission = Admission::default();
        let audio = admission.audio.clone().acquire_many_owned(2).await.unwrap();
        let request = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .body(Body::from_stream(futures_util::stream::pending::<
                Result<axum::body::Bytes, std::io::Error>,
            >()))
            .unwrap();
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            app(admission.clone()).oneshot(request),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let response = app(admission.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/embeddings")
                    .body(Body::from("x"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(admission.embeddings.available_permits(), 31);
        drop(response);
        assert_eq!(admission.embeddings.available_permits(), 32);
        drop(audio);
        assert_eq!(admission.audio.available_permits(), 2);
    }

    #[tokio::test]
    async fn chunked_upload_limits_and_response_lifetime_release_leases() {
        let admission = Admission::default();
        let chunks = vec![
            Ok::<_, std::io::Error>(axum::body::Bytes::from(vec![0; EMBEDDING_BYTES])),
            Ok(axum::body::Bytes::from_static(b"x")),
        ];
        let response = app(admission.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/embeddings")
                    .body(Body::from_stream(futures_util::stream::iter(chunks)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        response.into_body().collect().await.unwrap();
        assert_eq!(admission.embeddings.available_permits(), 32);
        let response = app(admission.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/transcriptions")
                    .body(Body::from(vec![0; AUDIO_BYTES + 1]))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        drop(response);
        assert_eq!(admission.audio.available_permits(), 2);
    }
}
