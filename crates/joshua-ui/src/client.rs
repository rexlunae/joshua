//! HTTP transport for [`crate::api`], built on `reqwest` (browser `fetch`
//! when compiled to `wasm32`).

use futures_util::StreamExt;
use web_time::Instant;

pub use crate::api::StreamSummary;
use crate::api::{
    bearer_header, endpoint_url, error_from_status, parse_chat_response, parse_health,
    parse_models, ApiError, ChatRequest, ChatResponse, ChatStream, Endpoint, Health, ModelList,
};

/// A connection to one Joshua server.
#[derive(Clone, Debug)]
pub struct JoshuaClient {
    base: String,
    api_key: Option<String>,
    http: reqwest::Client,
}

impl JoshuaClient {
    /// `base` must already be normalised with [`crate::api::normalize_base_url`].
    pub fn new(base: String, api_key: Option<String>) -> Self {
        Self {
            base,
            api_key,
            http: reqwest::Client::new(),
        }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    fn request(&self, method: reqwest::Method, endpoint: Endpoint) -> reqwest::RequestBuilder {
        let mut req = self
            .http
            .request(method, endpoint_url(&self.base, endpoint));
        if endpoint.requires_auth() {
            if let Some(auth) = bearer_header(self.api_key.as_deref()) {
                req = req.header(reqwest::header::AUTHORIZATION, auth);
            }
        }
        req
    }

    async fn send_text(req: reqwest::RequestBuilder) -> Result<String, ApiError> {
        let resp = req
            .send()
            .await
            .map_err(|e| ApiError::Transport(e.to_string()))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| ApiError::Transport(e.to_string()))?;
        if status.is_success() {
            Ok(body)
        } else {
            Err(error_from_status(status.as_u16(), &body))
        }
    }

    /// `GET /health`, with the round-trip time in milliseconds.
    pub async fn health(&self) -> Result<(Health, f64), ApiError> {
        let start = Instant::now();
        let body = Self::send_text(self.request(reqwest::Method::GET, Endpoint::Health)).await?;
        let rtt = start.elapsed().as_secs_f64() * 1000.0;
        Ok((parse_health(&body)?, rtt))
    }

    /// `GET /v1/models`.
    pub async fn models(&self) -> Result<ModelList, ApiError> {
        let body = Self::send_text(self.request(reqwest::Method::GET, Endpoint::Models)).await?;
        parse_models(&body)
    }

    /// Non-streaming `POST /v1/chat/completions`.
    pub async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, ApiError> {
        let body = Self::send_text(self.chat_request(req)).await?;
        parse_chat_response(&body)
    }

    fn chat_request(&self, req: &ChatRequest) -> reqwest::RequestBuilder {
        self.request(reqwest::Method::POST, Endpoint::ChatCompletions)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(req.to_json())
    }

    /// Streaming `POST /v1/chat/completions`; `on_delta` sees each piece of
    /// content as it arrives.
    pub async fn chat_stream(
        &self,
        req: &ChatRequest,
        mut on_delta: impl FnMut(&str),
    ) -> Result<StreamSummary, ApiError> {
        let resp = self
            .chat_request(req)
            .send()
            .await
            .map_err(|e| ApiError::Transport(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(error_from_status(status.as_u16(), &body));
        }
        let mut stream = ChatStream::new();
        let mut bytes = resp.bytes_stream();
        while let Some(chunk) = bytes.next().await {
            let chunk = chunk.map_err(|e| ApiError::Transport(e.to_string()))?;
            if stream.push(&chunk, &mut on_delta)? {
                break;
            }
        }
        stream.finish(&mut on_delta)
    }
}
