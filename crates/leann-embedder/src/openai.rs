//! OpenAI-compatible embeddings API client.
//!
//! Works with anything speaking `POST {base_url}/embeddings`:
//! OpenAI, Azure OpenAI (with deployed-path base URL), Ollama
//! (`http://localhost:11434/v1`), vLLM, LM Studio, SiliconFlow, Together,
//! DeepSeek-compatible gateways, etc.
//!
//! Features:
//! - request batching (`batch_size`, default 128)
//! - bounded parallel in-flight requests (`concurrency`, default 4)
//! - exponential backoff retries on 429/5xx/network (default 3)
//! - response ordering by `index` field
//! - `text-embedding-3-*` dimension truncation via `dimensions` parameter

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use leann_core::cache::EmbeddingProvider;
use leann_core::{LeannError, Result};

#[derive(Debug)]
pub struct OpenAiEmbedder {
    agent: ureq::Agent,
    base_url: String,
    api_key: String,
    model: String,
    dim: usize,
    batch_size: usize,
    concurrency: usize,
    max_retries: u32,
    /// Send the `dimensions` parameter (only supported by some models).
    send_dimensions: bool,
    requests: AtomicUsize,
    retried: AtomicUsize,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    #[allow(dead_code)]
    model: Option<String>,
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    index: usize,
    embedding: Vec<f32>,
}

impl OpenAiEmbedder {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        dim: usize,
    ) -> Self {
        Self::with_options(base_url, api_key, model, dim, 128, 4, 3)
    }

    pub fn with_options(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        dim: usize,
        batch_size: usize,
        concurrency: usize,
        max_retries: u32,
    ) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(120))
            .redirects(4)
            .build();
        Self {
            agent,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: model.into(),
            dim,
            batch_size: batch_size.clamp(1, 2048),
            concurrency: concurrency.clamp(1, 32),
            max_retries,
            send_dimensions: false,
            requests: AtomicUsize::new(0),
            retried: AtomicUsize::new(0),
        }
    }

    /// Enable sending `dimensions` (for text-embedding-3-* models).
    pub fn with_dimensions_param(mut self, on: bool) -> Self {
        self.send_dimensions = on;
        self
    }

    fn embed_batch_inner(&self, input: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = EmbedRequest {
            model: &self.model,
            input,
            dimensions: if self.send_dimensions {
                Some(self.dim)
            } else {
                None
            },
        };
        let url = format!("{}/embeddings", self.base_url);
        let mut attempt = 0u32;
        loop {
            let req = self
                .agent
                .post(&url)
                .set("Content-Type", "application/json");
            let req = if self.api_key.is_empty() {
                // local servers (ollama / vllm) accept no-auth requests
                req
            } else {
                req.set("Authorization", &format!("Bearer {}", self.api_key))
            };
            match req.send_json(serde_json::to_value(&body)?) {
                Ok(resp) => {
                    let parsed: EmbedResponse = serde_json::from_reader(resp.into_reader())
                        .map_err(|e| LeannError::Embedder(format!("bad embeddings response: {e}")))?;
                    self.requests.fetch_add(1, Ordering::Relaxed);
                    let mut out = parsed.data;
                    out.sort_by_key(|d| d.index);
                    if out.len() != input.len() {
                        return Err(LeannError::Embedder(format!(
                            "embeddings count mismatch: got {} want {}",
                            out.len(),
                            input.len()
                        )));
                    }
                    let mut vecs: Vec<Vec<f32>> = out.into_iter().map(|d| d.embedding).collect();
                    for v in vecs.iter() {
                        if v.len() != self.dim {
                            return Err(LeannError::Embedder(format!(
                                "embedding dim mismatch: got {} want {} (model {})",
                                v.len(), self.dim, self.model
                            )));
                        }
                    }
                    vecs.truncate(input.len());
                    return Ok(vecs);
                }
                Err(ureq::Error::Status(code, resp)) => {
                    let msg = resp.into_string().unwrap_or_default();
                    let retryable = code == 429 || code >= 500;
                    if retryable && attempt < self.max_retries {
                        attempt += 1;
                        self.retried.fetch_add(1, Ordering::Relaxed);
                        let backoff = 250u64 << attempt.min(4);
                        std::thread::sleep(Duration::from_millis(backoff));
                        continue;
                    }
                    return Err(LeannError::Embedder(format!(
                        "embeddings API {code}: {}",
                        truncate(&msg, 400)
                    )));
                }
                Err(e) => {
                    if attempt < self.max_retries {
                        attempt += 1;
                        self.retried.fetch_add(1, Ordering::Relaxed);
                        std::thread::sleep(Duration::from_millis(250 << attempt.min(4)));
                        continue;
                    }
                    return Err(LeannError::Embedder(format!("embeddings request failed: {e}")));
                }
            }
        }
    }

    pub fn stats(&self) -> (usize, usize) {
        (
            self.requests.load(Ordering::Relaxed),
            self.retried.load(Ordering::Relaxed),
        )
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        let mut end = n;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

impl EmbeddingProvider for OpenAiEmbedder {
    fn name(&self) -> &str {
        "openai-compatible"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let batches: Vec<&[&str]> = texts.chunks(self.batch_size).collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.concurrency)
            .build()
            .map_err(|e| LeannError::Embedder(format!("thread pool: {e}")))?;
        let results: Result<Vec<Vec<Vec<f32>>>> =
            pool.install(|| batches.par_iter().map(|b| self.embed_batch_inner(b)).collect());
        let flattened: Vec<Vec<f32>> = results?.into_iter().flatten().collect();
        if flattened.len() != texts.len() {
            return Err(LeannError::Embedder("batch length mismatch".into()));
        }
        Ok(flattened)
    }
}

#[allow(dead_code)]
pub fn openai_embedder(
    base_url: &str,
    api_key: &str,
    model: &str,
    dim: usize,
) -> Arc<dyn EmbeddingProvider> {
    Arc::new(OpenAiEmbedder::new(base_url, api_key, model, dim))
}
