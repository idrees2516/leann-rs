//! # leann-core
//!
//! Storage-efficient vector search engine implementing the LEANN approach
//! (Low-storage vector search via graph pruning and PQ filtering), extended
//! with BM25 hybrid retrieval, metadata filtering, context expansion, and
//! dynamic updates for agentic RAG workloads.
//!
//! Key ideas (see the LEANN paper, arXiv:2506.08276):
//! - **No stored embeddings.** Raw embeddings are discarded after build;
//!   at query time they are recomputed on the fly for a small candidate set.
//! - **Product Quantization (PQ)** codes (~100x compression) provide cheap
//!   approximate distances that guide graph traversal.
//! - **High-degree preserving graph pruning** shrinks HNSW graph metadata:
//!   a small fraction of hub nodes (top β%) keep up to M edges while all
//!   other nodes are limited to m = M/5.
//! - **Two-level search** interleaves approximate (PQ) and exact (recomputed)
//!   distances: an Approximate Queue (AQ) holds PQ estimates for every node
//!   ever seen, while the Exact Queue (EQ) drives traversal; only the top
//!   α fraction of the AQ is promoted to exact recomputation per step.
//! - **Dynamic batching** aggregates recomputations across search hops into
//!   embedder-friendly batches.
//!
//! On top of that, this crate ships a BM25 inverted index, reciprocal-rank
//! fusion hybrid search, JSON metadata filters, and neighbor context
//! expansion — everything an agentic RAG pipeline needs.

pub mod batch;
pub mod bm25;
pub mod cache;
pub mod context;
pub mod crypto;
pub mod dedup;
pub mod export;
pub mod filter;
pub mod format;
pub mod graph;
pub mod hybrid;
pub mod index;
pub mod metric;
pub mod pq;
pub mod search;
pub mod simd;
pub mod validate;

pub use batch::{DynamicBatcher, BatcherStats};
pub use bm25::{Bm25Index, Bm25Params};
pub use cache::{EmbedCache, EmbedSource, EmbeddingProvider, TextSource};
pub use context::{expand_context, ContextExpansionOptions, ExpandedContext};
pub use filter::{Filter, Op};
pub use format::{IndexFile, FORMAT_VERSION, MAGIC};
pub use graph::{GraphBuildConfig, HnswGraph, HubPruneConfig};
pub use hybrid::{fusion, hybrid_search, FusionMode, HybridWeights};
pub use index::{
    HybridOptions, HybridResultHit, HybridSearchResult, IndexConfig, IndexStats, LeannIndex,
    SearchDefaults, SearchHit, SearchOptions, SearchReport, SearchResult,
};
pub use metric::Metric;
pub use simd::{dot, l2_sq, SimdLevel};
pub use dedup::{DedupMode, Deduper};
pub use export::{
    alive_chunks, export_csv, export_json, export_markdown, export_string, import_json,
    ExportFormat, ImportedChunk,
};
pub use validate::{check_filter_limits, safe_component, validate_base_url};
pub use pq::{PqCodec, PqConfig};

/// Common error type for the whole crate.
pub mod error {
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum LeannError {
        #[error("I/O error: {0}")]
        Io(#[from] std::io::Error),
        #[error("serialization error: {0}")]
        Serde(#[from] serde_json::Error),
        #[error("corrupt index file: {0}")]
        Corrupt(String),
        #[error("unsupported index format version {found}, expected {expected}")]
        Version { found: u32, expected: u32 },
        #[error("dimension mismatch: index has {index_dim}, got {got_dim}")]
    DimMismatch { index_dim: usize, got_dim: usize },
        #[error("empty index: {0}")]
        Empty(String),
        #[error("invalid argument: {0}")]
    Invalid(String),
        #[error("embedder error: {0}")]
    Embedder(String),
        #[error("crypto error: {0}")]
    Crypto(String),
    }
    pub type Result<T> = std::result::Result<T, LeannError>;
}

pub use error::{LeannError, Result};
