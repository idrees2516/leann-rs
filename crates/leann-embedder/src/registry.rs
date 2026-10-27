//! Embedder registry: build providers from serializable config (used by the
//! Node bindings and CLI so users never touch Rust types).

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use leann_core::cache::EmbeddingProvider;
use leann_core::{LeannError, Result};

use crate::{LocalHashEmbedder, OpenAiEmbedder};

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum EmbedderKind {
    /// OpenAI-compatible /embeddings endpoint.
    #[default]
    OpenAi,
    /// Zero-dependency local feature-hashing fallback.
    LocalHash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedderConfig {
    pub kind: EmbedderKind,
    /// Base URL without `/embeddings`; default `https://api.openai.com/v1`.
    /// Reads env `OPENAI_BASE_URL` / `LEANN_EMBED_BASE_URL` when absent.
    pub base_url: Option<String>,
    /// Reads env `OPENAI_API_KEY` / `LEANN_EMBED_API_KEY` when absent.
    pub api_key: Option<String>,
    /// Default `text-embedding-3-small` (or env `LEANN_EMBED_MODEL`).
    pub model: Option<String>,
    /// Embedding dimensionality (must match the model).
    pub dim: usize,
    /// Texts per API request (default 128).
    pub batch_size: usize,
    /// Parallel in-flight requests (default 4).
    pub concurrency: usize,
    /// Retries on 429/5xx (default 3).
    pub max_retries: u32,
    /// Send the `dimensions` request parameter (text-embedding-3-*).
    pub send_dimensions_param: bool,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            kind: EmbedderKind::OpenAi,
            base_url: None,
            api_key: None,
            model: None,
            dim: 1536,
            batch_size: 128,
            concurrency: 4,
            max_retries: 3,
            send_dimensions_param: false,
        }
    }
}

/// Create a provider from config, applying env-var fallbacks.
pub fn create_embedder(cfg: &EmbedderConfig) -> Result<Arc<dyn EmbeddingProvider>> {
    match cfg.kind {
        EmbedderKind::LocalHash => Ok(Arc::new(LocalHashEmbedder::new(
            if cfg.dim == 1536 { 384 } else { cfg.dim }, // sane local default
        ))),
        EmbedderKind::OpenAi => {
            let base_url = cfg
                .base_url
                .clone()
                .or_else(|| env_first(&["LEANN_EMBED_BASE_URL", "OPENAI_BASE_URL"]))
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
            let api_key = cfg
                .api_key
                .clone()
                .or_else(|| env_first(&["LEANN_EMBED_API_KEY", "OPENAI_API_KEY"]))
                .unwrap_or_default();
            let model = cfg
                .model
                .clone()
                .or_else(|| std::env::var("LEANN_EMBED_MODEL").ok())
                .unwrap_or_else(|| "text-embedding-3-small".to_string());
            if api_key.is_empty() && !base_url.contains("localhost") && !base_url.contains("127.0.0.1") {
                return Err(LeannError::Embedder(
                    "no API key: set api_key or OPENAI_API_KEY (or use kind=local-hash)".into(),
                ));
            }
            if cfg.dim == 0 {
                return Err(LeannError::Invalid("dim must be > 0".into()));
            }
            let mut e = OpenAiEmbedder::with_options(
                base_url,
                api_key,
                model,
                cfg.dim,
                cfg.batch_size,
                cfg.concurrency,
                cfg.max_retries,
            )
            .map_err(LeannError::Invalid)?;
            if cfg.send_dimensions_param {
                e = e.with_dimensions_param(true);
            }
            Ok(Arc::new(e))
        }
    }
}

fn env_first(names: &[&str]) -> Option<String> {
    for n in names {
        if let Ok(v) = std::env::var(n) {
            if !v.trim().is_empty() {
                return Some(v);
            }
        }
    }
    None
}
