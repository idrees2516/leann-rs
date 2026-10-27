//! # leann-embedder
//!
//! Embedding providers for LEANN:
//!
//! - [`OpenAiEmbedder`] — any OpenAI-compatible `/embeddings` endpoint
//!   (OpenAI, Azure, SiliconFlow, Ollama, vLLM, LM Studio, DeepSeek, ...).
//!   Batches requests, retries on 429/5xx with backoff, parallel in-flight
//!   requests via rayon, and normalizes when the index metric is cosine.
//! - [`LocalHashEmbedder`] — deterministic feature-hashing embedder
//!   (word unigrams + char trigrams). Zero dependencies, zero network,
//!   instant — the offline fallback (same role as memvid's hash-based
//!   "no API key" mode, but with n-gram semantics).
//!
//! Providers implement `leann_core::EmbeddingProvider`, so they plug
//! directly into `LeannIndex::build / open / add`.

mod local;
mod openai;
mod registry;

pub use local::LocalHashEmbedder;
pub use openai::OpenAiEmbedder;
pub use registry::{create_embedder, EmbedderConfig, EmbedderKind};

#[cfg(test)]
mod tests {
    use super::*;
    use leann_core::cache::EmbeddingProvider;

    #[test]
    fn local_hash_is_deterministic_and_normalized() {
        let e = LocalHashEmbedder::new(256);
        let a = e.embed_batch(&["hello world", "vector search"]).unwrap();
        let b = e.embed_batch(&["hello world", "vector search"]).unwrap();
        assert_eq!(a[0], b[0]);
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].len(), 256);
        let n: f32 = a[0].iter().map(|x| x * x).sum();
        assert!((n - 1.0).abs() < 1e-4, "should be L2 normalized, got {n}");
    }

    #[test]
    fn local_hash_semantic_grouping() {
        // same-topic texts should be closer than cross-topic
        let e = LocalHashEmbedder::new(512);
        let t1 = "rust borrow checker memory safety";
        let t2 = "rust ownership and borrow checking lifetimes";
        let t3 = "chocolate cake recipe with sugar and flour";
        let v = e.embed_batch(&[t1, t2, t3]).unwrap();
        let dot = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b).map(|(x, y)| x * y).sum()
        };
        assert!(dot(&v[0], &v[1]) > dot(&v[0], &v[2]));
    }

    #[test]
    fn registry_creates_local() {
        let cfg = EmbedderConfig {
            kind: EmbedderKind::LocalHash,
            dim: 128,
            ..Default::default()
        };
        let e = create_embedder(&cfg).unwrap();
        assert_eq!(e.dim(), 128);
        let v = e.embed_batch(&["x"]).unwrap();
        assert_eq!(v[0].len(), 128);
    }

    #[test]
    fn openai_config_defaults() {
        let cfg = EmbedderConfig::default();
        assert!(matches!(cfg.kind, EmbedderKind::OpenAi));
        let e = OpenAiEmbedder::new(
            "https://api.openai.com/v1",
            "test-key",
            "text-embedding-3-small",
            1536,
        );
        assert_eq!(e.expect("valid default url").dim(), 1536);
    }
}
