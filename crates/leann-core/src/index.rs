//! The `LeannIndex` orchestrator: build → search → hybrid → add/delete →
//! persist. This is the type every higher layer (CLI, Node bindings, chat)
//! talks to.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::bm25::{Bm25Index, Bm25Params};
use crate::cache::{EmbedSource, EmbeddingProvider};
use crate::context::{expand_context, ChunkPlacement, ContextExpansionOptions, ExpandedContext};
use crate::filter::Filter;
use crate::format::{read_index, write_index, IndexFile, TextStore};
use crate::graph::{DistOracle, GraphBuildConfig, HnswGraph, HubPruneConfig, SliceOracle};
use crate::hybrid::{fusion, FusionMode, HybridWeights};
use crate::metric::Metric;
use crate::pq::{kmeans_centroids, PqCodec, PqConfig};
use crate::search::{brute_force_search, two_level_search_tls, SearchParams, SearchStats};
use crate::dedup::{DedupMode, Deduper};
use crate::format::{read_index_with_passphrase, write_index_encrypted};
use crate::{LeannError, Result};

/// Default search tunables (see `SearchParams` for meaning).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SearchDefaults {
    pub ef: usize,
    pub alpha: f32,
    pub min_batch: usize,
    pub max_batch: usize,
    pub relax: f32,
}

impl Default for SearchDefaults {
    fn default() -> Self {
        Self {
            ef: 96,
            alpha: 0.08,
            min_batch: 4,
            max_batch: 64,
            relax: 1.0,
        }
    }
}

/// Full index configuration (persisted in the file header).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub metric: Metric,
    pub pq: PqConfig,
    pub graph: GraphBuildConfig,
    pub hub: HubPruneConfig,
    pub search: SearchDefaults,
    pub cache_capacity: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            metric: Metric::Cosine,
            pq: PqConfig::default(),
            graph: GraphBuildConfig::default(),
            hub: HubPruneConfig::default(),
            search: SearchDefaults::default(),
            cache_capacity: 8192,
        }
    }
}

/// Per-call search options.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub k: usize,
    pub ef: Option<usize>,
    pub alpha: Option<f32>,
    pub min_batch: Option<usize>,
    pub max_batch: Option<usize>,
    pub relax: Option<f32>,
    pub filter: Option<Filter>,
    /// Brute-force exact scan (testing / small corpora).
    pub exact: bool,
    pub with_text: bool,
    pub with_metadata: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            k: 10,
            ef: None,
            alpha: None,
            min_batch: None,
            max_batch: None,
            relax: None,
            filter: None,
            exact: false,
            with_text: true,
            with_metadata: true,
        }
    }
}

/// Hybrid (vector + BM25 + fusion) options.
#[derive(Debug, Clone)]
pub struct HybridOptions {
    pub base: SearchOptions,
    pub fusion: FusionMode,
    pub weights: HybridWeights,
    /// Lexual candidate count; default = base.k * 3.
    pub lexical_k: Option<usize>,
    /// Vector over-fetch multiplier; default 3.0.
    pub vector_overfetch: f32,
    /// Optional neighbor-context expansion.
    pub expand: Option<ContextExpansionOptions>,
}

impl Default for HybridOptions {
    fn default() -> Self {
        Self {
            base: SearchOptions::default(),
            fusion: FusionMode::default(),
            weights: HybridWeights::default(),
            lexical_k: None,
            vector_overfetch: 3.0,
            expand: None,
        }
    }
}

/// One retrieval hit.
#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub id: u32,
    pub distance: f32,
    pub similarity: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// Result of a pure vector search.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub hits: Vec<SearchHit>,
    pub took_ms: f64,
    pub stats: SearchReport,
}

/// One hybrid hit.
#[derive(Debug, Clone, Serialize)]
pub struct HybridResultHit {
    pub id: u32,
    pub fused_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_rank: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_rank: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_similarity: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// Result of a hybrid search.
#[derive(Debug, Clone, Serialize)]
pub struct HybridSearchResult {
    pub hits: Vec<HybridResultHit>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub expanded: Vec<ExpandedContext>,
    pub took_ms: f64,
    pub stats: SearchReport,
}

/// JSON-friendly stats block.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SearchReport {
    pub hops: usize,
    pub pq_scored: usize,
    pub promoted: usize,
    pub embed_calls: usize,
    pub vectors_embedded: usize,
    pub embed_secs: f64,
    pub total_secs: f64,
    pub lexical_hits: usize,
    pub vector_hits: usize,
}

impl From<SearchStats> for SearchReport {
    fn from(s: SearchStats) -> Self {
        Self {
            hops: s.hops,
            pq_scored: s.pq_scored,
            promoted: s.promoted,
            embed_calls: s.embed_calls,
            vectors_embedded: s.vectors_embedded,
            embed_secs: s.embed_secs,
            total_secs: s.total_secs,
            ..Default::default()
        }
    }
}

/// Index-level statistics.
#[derive(Debug, Clone, Serialize)]
pub struct IndexStats {
    pub chunks: usize,
    pub alive: usize,
    pub deleted: usize,
    pub dim: usize,
    pub metric: Metric,
    pub pq_m: usize,
    pub pq_compression: f32,
    pub text_bytes: u64,
    pub codes_bytes: u64,
    pub codebook_bytes: u64,
    pub graph_edges: usize,
    pub graph_bytes: u64,
    pub avg_degree: f32,
    pub bm25_vocab: usize,
    pub embedder: String,
    pub file_bytes: Option<u64>,
    /// Storage a conventional index would need for raw embeddings.
    pub conventional_vector_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// A LEANN index over a set of text chunks.
pub struct LeannIndex {
    cfg: IndexConfig,
    texts: TextStore,
    metadatas: Vec<Value>,
    codec: PqCodec,
    codes: Vec<u8>,
    graph: HnswGraph,
    tombstones: Vec<u64>,
    bm25: Bm25Index,
    source: EmbedSource,
    embedder_name: String,
    placement: ChunkPlacement,
    file_bytes: Option<u64>,
}

impl LeannIndex {
    /// Empty in-memory index (fill via `add`).
    pub fn empty(cfg: IndexConfig, provider: Arc<dyn EmbeddingProvider>) -> Self {
        let source = EmbedSource::new(provider, cfg.metric, cfg.cache_capacity);
        Self {
            cfg,
            texts: TextStore::Owned(Vec::new()),
            metadatas: Vec::new(),
            codec: PqCodec {
                m: 0,
                ksub: 0,
                dim: 0,
                subdim: 0,
                codebook: Vec::new(),
                metric: Metric::Cosine,
            },
            codes: Vec::new(),
            graph: HnswGraph::default(),
            tombstones: Vec::new(),
            bm25: Bm25Index::new(Bm25Params::default()),
            source,
            embedder_name: "unset".into(),
            placement: ChunkPlacement::default(),
            file_bytes: None,
        }
    }

    /// Build a full index in one pass (all embeddings transiently in RAM).
    /// This is the fastest build path and the right default for API-backed
    /// embedders (exactly one embedding call per chunk).
    pub fn build(
        texts: &[String],
        metadatas: Option<Vec<Value>>,
        provider: Arc<dyn EmbeddingProvider>,
        cfg: IndexConfig,
    ) -> Result<Self> {
        let n = texts.len();
        if n == 0 {
            return Err(LeannError::Empty("cannot build an index over 0 chunks".into()));
        }
        let metas = metadatas.unwrap_or_else(|| vec![Value::Null; n]);
        if metas.len() != n {
            return Err(LeannError::Invalid("metadatas length mismatch".into()));
        }

        let source = EmbedSource::new(provider, cfg.metric, cfg.cache_capacity);
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embs = source.embed_texts(&refs)?;
        if embs.len() != n || embs[0].len() == 0 {
            return Err(LeannError::Embedder("empty embedding response".into()));
        }

        // PQ train + encode
        let codec = PqCodec::train(&embs, &cfg.pq, cfg.metric)?;
        let mut codes = Vec::with_capacity(n * codec.m);
        for e in &embs {
            codes.extend_from_slice(&codec.encode(e));
        }

        // graph build + hub prune
        let oracle = SliceOracle {
            vecs: &embs,
            metric: cfg.metric,
        };
        let mut graph = HnswGraph::build_local(n, &oracle, &cfg.graph);
        graph.hub_prune(&cfg.hub, cfg.graph.m, &oracle);
        graph.connect_orphans();

        let bm25 = Bm25Index::build(&refs, Bm25Params::default());
        let embedder_name = source.provider().name().to_string();
        let placement = Self::build_placement(&metas, n);

        Ok(Self {
            cfg,
            texts: TextStore::Owned(texts.to_vec()),
            metadatas: metas,
            codec,
            codes,
            graph,
            tombstones: vec![0; (n + 63) / 64],
            bm25,
            source,
            embedder_name,
            placement,
            file_bytes: None,
        })
    }

    /// Storage-efficient sharded build (paper §6): k-means soft-assigns every
    /// chunk to its two nearest shard centroids; each shard builds its own
    /// graph from recomputed embeddings; shard graphs are merged. Peak
    /// embedding memory ≈ one shard, at the cost of ~3 embedding passes
    /// total (assignment + two shard memberships). Prefer `build` for
    /// pay-per-token API embedders.
    pub fn build_sharded(
        texts: &[String],
        metadatas: Option<Vec<Value>>,
        provider: Arc<dyn EmbeddingProvider>,
        cfg: IndexConfig,
        shard_size: usize,
    ) -> Result<Self> {
        let n = texts.len();
        if n == 0 {
            return Err(LeannError::Empty("cannot build an index over 0 chunks".into()));
        }
        let shard_size = shard_size.clamp(1000, 200_000);
        let metas = metadatas.unwrap_or_else(|| vec![Value::Null; n]);
        if metas.len() != n {
            return Err(LeannError::Invalid("metadatas length mismatch".into()));
        }
        let source = EmbedSource::new(provider, cfg.metric, cfg.cache_capacity);

        // 1) sample embeddings -> shard centroids + PQ codec
        let sample_n = n.min(20_000).max(256).min(n);
        let sample_idx: Vec<usize> = (0..n).step_by((n / sample_n).max(1)).take(sample_n).collect();
        let sample_refs: Vec<&str> = sample_idx.iter().map(|&i| texts[i].as_str()).collect();
        let sample_embs = source.embed_texts(&sample_refs)?;
        let k = ((n as f32 / shard_size as f32).ceil() as usize).clamp(1, 256).min(n);
        let centroids = kmeans_centroids(&sample_embs, k, 8, cfg.graph.seed, cfg.metric.needs_normalization());
        let codec = PqCodec::train(&sample_embs, &cfg.pq, cfg.metric)?;

        // 2) pass 1: embed windows, assign to 2 shards, PQ-encode, discard
        let mut shard_members: Vec<Vec<u32>> = vec![Vec::new(); k];
        let mut codes: Vec<u8> = Vec::with_capacity(n * codec.m);
        let window = 512;
        for w_start in (0..n).step_by(window) {
            let w_end = (w_start + window).min(n);
            let wrefs: Vec<&str> = texts[w_start..w_end].iter().map(|s| s.as_str()).collect();
            let wembs = source.embed_texts(&wrefs)?;
            for (j, emb) in wembs.iter().enumerate() {
                let doc = (w_start + j) as u32;
                // two nearest centroids
                let mut d: Vec<(f32, usize)> = centroids
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| (cfg.metric.distance(emb, c), ci))
                    .collect();
                d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                shard_members[d[0].1].push(doc);
                if k > 1 {
                    shard_members[d[1].1].push(doc);
                }
                codes.extend_from_slice(&codec.encode(emb));
            }
        }

        // 3) per-shard graph construction + merge
        let mut acc = HnswGraph::with_capacity(n);
        let mut have_any = false;
        for members in &shard_members {
            if members.len() < 2 {
                continue;
            }
            let refs: Vec<&str> = members.iter().map(|&d| texts[d as usize].as_str()).collect();
            let embs = source.embed_texts(&refs)?;
            let oracle = SliceOracle {
                vecs: &embs,
                metric: cfg.metric,
            };
            let mut g = HnswGraph::build_local(members.len(), &oracle, &cfg.graph);
            g.hub_prune(&cfg.hub, cfg.graph.m, &oracle);
            let map: Vec<u32> = members.to_vec();
            g.remap(&map, n);
            if have_any {
                acc.merge_union(&g, &cfg.hub, cfg.graph.m);
            } else {
                acc = g;
                have_any = true;
            }
        }
        if !have_any {
            // degenerate: fall back to single-shard build
            return Self::build(texts, Some(metas), source.provider().clone(), cfg);
        }
        acc.connect_orphans();

        let all_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let bm25 = Bm25Index::build(&all_refs, Bm25Params::default());
        let placement = Self::build_placement(&metas, n);
        let embedder_name = source.provider().name().to_string();

        Ok(Self {
            cfg,
            texts: TextStore::Owned(texts.to_vec()),
            metadatas: metas,
            codec,
            codes,
            graph: acc,
            tombstones: vec![0; (n + 63) / 64],
            bm25,
            source,
            embedder_name,
            placement,
            file_bytes: None,
        })
    }

    /// Open a persisted index. BM25 and placement maps are rebuilt from the
    /// stored texts (cheap, keeps the file minimal).
    pub fn open(path: &Path, provider: Arc<dyn EmbeddingProvider>) -> Result<Self> {
        let loaded = read_index(path)?;
        if provider.dim() != loaded.codec.dim {
            return Err(LeannError::DimMismatch {
                index_dim: loaded.codec.dim,
                got_dim: provider.dim(),
            });
        }
        Self::from_loaded(loaded, provider)
    }

    fn from_loaded(loaded: crate::format::LoadedIndex, provider: Arc<dyn EmbeddingProvider>) -> Result<Self> {
        let n = loaded.texts.len();
        let refs: Vec<&str> = (0..n as u32).map(|i| loaded.texts.text(i)).collect();
        let bm25 = Bm25Index::build(&refs, Bm25Params::default());
        let placement = Self::build_placement(&loaded.metadatas, n);
        let source = EmbedSource::new(provider, loaded.config.metric, loaded.config.cache_capacity);
        Ok(Self {
            cfg: loaded.config,
            texts: loaded.texts,
            metadatas: loaded.metadatas,
            codec: loaded.codec,
            codes: loaded.codes,
            graph: loaded.graph,
            tombstones: loaded.tombstones,
            bm25,
            source,
            embedder_name: loaded.embedder_name,
            placement,
            file_bytes: Some(loaded.file_size),
        })
    }

    /// Open an index that may be encrypted at rest (see [`Self::save_encrypted`]).
    pub fn open_with_passphrase(
        path: &Path,
        passphrase: &str,
        provider: Arc<dyn EmbeddingProvider>,
    ) -> Result<Self> {
        let loaded = read_index_with_passphrase(path, Some(passphrase))?;
        if provider.dim() != loaded.codec.dim {
            return Err(LeannError::DimMismatch {
                index_dim: loaded.codec.dim,
                got_dim: provider.dim(),
            });
        }
        Self::from_loaded(loaded, provider)
    }

    /// Persist (atomic tmp + rename).
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        write_index(
            path,
            IndexFile {
                config: &self.cfg,
                texts: &self.texts,
                metadatas: &self.metadatas,
                codec: &self.codec,
                codes: &self.codes,
                graph: &self.graph,
                tombstones: &self.tombstones,
                embedder_name: &self.embedder_name,
            },
        )
    }

    /// Persist encrypted at rest: ChaCha20-Poly1305 AEAD over the whole
    /// serialized payload, key derived from `passphrase` with Argon2id.
    /// A stolen file reveals nothing (texts, codes, graph, metadata are all
    /// inside the seal) and any tampering fails authentication on open.
    pub fn save_encrypted(&self, path: &Path, passphrase: &str) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        write_index_encrypted(
            path,
            IndexFile {
                config: &self.cfg,
                texts: &self.texts,
                metadatas: &self.metadatas,
                codec: &self.codec,
                codes: &self.codes,
                graph: &self.graph,
                tombstones: &self.tombstones,
                embedder_name: &self.embedder_name,
            },
            passphrase,
        )
    }

    // ---------------- retrieval ----------------

    /// Vector search.
    pub fn search(&self, query: &str, opts: &SearchOptions) -> Result<SearchResult> {
        let q = self.source.embed_query(query)?;
        self.search_by_vector(&q, opts)
    }

    /// Vector search with a pre-computed query vector.
    pub fn search_by_vector(&self, q: &[f32], opts: &SearchOptions) -> Result<SearchResult> {
        let t0 = Instant::now();
        if self.graph.n == 0 {
            return Ok(SearchResult {
                hits: Vec::new(),
                took_ms: 0.0,
                stats: SearchReport::default(),
            });
        }
        let k = opts.k.max(1);
        // over-fetch when filtering so post-filter results still fill k
        let fetch_k = if opts.filter.is_some() { (k * 3).max(k + 10) } else { k };
        let params = SearchParams {
            k: fetch_k,
            ef: opts.ef.unwrap_or(self.cfg.search.ef).max(fetch_k),
            alpha: opts.alpha.unwrap_or(self.cfg.search.alpha),
            min_batch: opts.min_batch.unwrap_or(self.cfg.search.min_batch),
            max_batch: opts.max_batch.unwrap_or(self.cfg.search.max_batch),
            relax: opts.relax.unwrap_or(self.cfg.search.relax),
        };
        let (raw, stats) = if opts.exact {
            let alive = |id: u32| self.is_alive(id);
            brute_force_search(self.graph.n, &alive, &self.source, &self.texts, q, fetch_k)?
        } else {
            two_level_search_tls(&self.graph, &self.codec, &self.codes, &self.source, &self.texts, q, &params)?
        };

        let mut report: SearchReport = stats.into();
        report.vector_hits = raw.len();
        let mut hits = Vec::with_capacity(k);
        for (id, d) in raw {
            if !self.is_alive(id) {
                continue;
            }
            if let Some(f) = &opts.filter {
                if !f.eval(self.metadata(id)) {
                    continue;
                }
            }
            hits.push(SearchHit {
                id,
                distance: d,
                similarity: similarity(self.cfg.metric, d),
                text: if opts.with_text {
                    Some(self.texts.text(id).to_string())
                } else {
                    None
                },
                metadata: if opts.with_metadata {
                    Some(self.metadata(id).clone())
                } else {
                    None
                },
            });
            if hits.len() >= k {
                break;
            }
        }
        let took_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(SearchResult {
            hits,
            took_ms,
            stats: report,
        })
    }

    /// Batch vector search: embeds all queries in **one** provider call, then
    /// searches in parallel across rayon workers (per-thread scratch + embed
    /// cache shards make the steady state allocation-free). Results are
    /// returned in input order.
    pub fn search_batch(&self, queries: &[&str], opts: &SearchOptions) -> Result<Vec<SearchResult>> {
        let qs = self.source.embed_texts(queries)?;
        self.search_by_vector_batch(&qs, opts)
    }

    /// Batch search with pre-computed query vectors (parallel).
    pub fn search_by_vector_batch(
        &self,
        qs: &[Vec<f32>],
        opts: &SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        use rayon::prelude::*;
        qs.par_iter().map(|q| self.search_by_vector(q, opts)).collect()
    }

    /// Hybrid search: LEANN vectors + BM25 + fusion (+ optional context
    /// expansion).
    pub fn hybrid_search(&self, query: &str, opts: &HybridOptions) -> Result<HybridSearchResult> {
        let t0 = Instant::now();
        let k = opts.base.k.max(1);
        let v_k = ((k as f32 * opts.vector_overfetch.max(1.0)).ceil() as usize).max(k);
        let l_k = opts.lexical_k.unwrap_or(v_k.max(10));

        // vector leg (with filter applied inside)
        let v_opts = SearchOptions {
            k: v_k,
            ..opts.base.clone()
        };
        let q = self.source.embed_query(query)?;
        let vres = self.search_by_vector(&q, &v_opts)?;
        let mut report = vres.stats;
        report.vector_hits = vres.hits.len();

        // lexical leg
        let alive = |id: u32| self.is_alive(id);
        let lex_all = self.bm25.search(query, l_k, &alive);
        let lex: Vec<(u32, f32)> = lex_all
            .into_iter()
            .filter(|(id, _)| {
                opts.base
                    .filter
                    .as_ref()
                    .map(|f| f.eval(self.metadata(*id)))
                    .unwrap_or(true)
            })
            .take(l_k)
            .collect();
        report.lexical_hits = lex.len();

        // fusion
        let vlist: Vec<(u32, f32)> =
            vres.hits.iter().map(|h| (h.id, h.similarity)).collect();
        let fused = fusion(&vlist, &lex, opts.fusion, opts.weights, k);

        let hits: Vec<HybridResultHit> = fused
            .iter()
            .map(|f| HybridResultHit {
                id: f.id,
                fused_score: f.fused_score,
                vector_rank: f.vector_rank,
                lexical_rank: f.lexical_rank,
                vector_similarity: f.vector_score,
                lexical_score: f.lexical_score,
                text: if opts.base.with_text {
                    Some(self.texts.text(f.id).to_string())
                } else {
                    None
                },
                metadata: if opts.base.with_metadata {
                    Some(self.metadata(f.id).clone())
                } else {
                    None
                },
            })
            .collect();

        let expanded = match &opts.expand {
            Some(eo) => {
                let ids: Vec<u32> = hits.iter().map(|h| h.id).collect();
                expand_context(&ids, &self.placement, *eo)
            }
            None => Vec::new(),
        };

        let took_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(HybridSearchResult {
            hits,
            expanded,
            took_ms,
            stats: report,
        })
    }

    /// Expand context around chunk ids.
    pub fn expand(
        &self,
        ids: &[u32],
        opts: ContextExpansionOptions,
    ) -> Vec<ExpandedContext> {
        expand_context(ids, &self.placement, opts)
    }

    // ---------------- updates ----------------

    /// Dynamically add chunks (paper Algorithm 4 with distance caching via
    /// the embed cache). Returns the assigned chunk ids.
    pub fn add(
        &mut self,
        texts: &[String],
        metadatas: Option<Vec<Value>>,
    ) -> Result<Vec<u32>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let n_existing = self.texts.len();
        let mut metas = metadatas.unwrap_or_else(|| vec![Value::Null; texts.len()]);
        if metas.len() != texts.len() {
            return Err(LeannError::Invalid("metadatas length mismatch".into()));
        }
        normalize_placement_metas(&mut metas);
        if n_existing == 0 {
            // first data into an empty index: do a full build
            let provider = self.source.provider().clone();
            let cfg = self.cfg.clone();
            let built = Self::build(texts, Some(metas), provider, cfg)?;
            *self = built;
            return Ok((0..texts.len() as u32).collect());
        }

        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embs = self.source.embed_texts(&refs)?;
        if embs[0].len() != self.codec.dim {
            return Err(LeannError::DimMismatch {
                index_dim: self.codec.dim,
                got_dim: embs[0].len(),
            });
        }

        let start = n_existing as u32;
        // extend storage
        for (t, m) in texts.iter().zip(metas.into_iter()) {
            self.texts.push(t.clone());
            self.metadatas.push(m);
        }
        for (i, t) in texts.iter().enumerate() {
            self.bm25.add_doc(start + i as u32, t);
        }
        for e in &embs {
            self.codes.extend_from_slice(&self.codec.encode(e));
        }
        let new_n = self.texts.len();
        self.graph.grow_to(new_n);
        self.tombstones.resize((new_n + 63) / 64, 0);
        self.rebuild_placement();

        // Runtime oracle: new embeddings in a side map, old nodes resolved
        // through the embed cache (pre-warmed by the batched searches below).
        let mut new_embs: HashMap<u32, Arc<Vec<f32>>> = HashMap::new();
        for (i, e) in embs.iter().enumerate() {
            new_embs.insert(start + i as u32, Arc::new(e.clone()));
        }

        // Pre-warm the cache with one batched two-level search per new node
        // (this is what keeps `add` from degenerating into size-1 embed calls).
        let prewarm_params = SearchParams {
            k: self.cfg.graph.ef_construction.min(new_n - 1),
            ef: self.cfg.graph.ef_construction,
            alpha: 0.25,
            ..SearchParams::default()
        };
        for e in embs.iter() {
            let _ = crate::search::two_level_search_tls(
                &self.graph,
                &self.codec,
                &self.codes,
                &self.source,
                &self.texts,
                e,
                &prewarm_params,
            )?;
        }

        // Insert each node (Algorithm 4): search neighbors with ef_C,
        // RNG-shrink, connect forward + reverse edges with degree caps.
        for i in 0..texts.len() as u32 {
            let id = start + i;
            let level = HnswGraph::sample_level(
                self.cfg.graph.seed.wrapping_add(id as u64),
                self.cfg.graph.m,
            );
            let oracle = RuntimeOracle {
                source: &self.source,
                texts: &self.texts,
                new_embs: &new_embs,
                metric: self.cfg.metric,
            };
            self.graph.insert(id, level, &oracle, &self.cfg.graph);
        }
        Ok((start..start + texts.len() as u32).collect())
    }

    /// Delete chunks by id (tombstones; graph edges remain for navigation).
    pub fn delete(&mut self, ids: &[u32]) -> usize {
        let mut n = 0;
        for &id in ids {
            if self.set_tombstone(id) {
                n += 1;
            }
        }
        n
    }

    /// Delete chunks matching a filter.
    pub fn delete_where(&mut self, filter: &Filter) -> usize {
        let victims: Vec<u32> = (0..self.texts.len() as u32)
            .filter(|&id| self.is_alive(id) && filter.eval(self.metadata(id)))
            .collect();
        self.delete(&victims)
    }

    // ---------------- accessors ----------------

    pub fn len(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    pub fn text(&self, id: u32) -> &str {
        self.texts.text(id)
    }

    pub fn metadata(&self, id: u32) -> &Value {
        &self.metadatas[id as usize]
    }

    pub fn config(&self) -> &IndexConfig {
        &self.cfg
    }

    pub fn source(&self) -> &EmbedSource {
        &self.source
    }

    /// Read-only graph access (diagnostics, tests).
    pub fn graph_ref(&self) -> &crate::graph::HnswGraph {
        &self.graph
    }

    pub fn is_alive(&self, id: u32) -> bool {
        let (w, b) = ((id / 64) as usize, (id % 64) as usize);
        match self.tombstones.get(w) {
            Some(word) => (word >> b) & 1 == 0,
            None => true,
        }
    }

    fn set_tombstone(&mut self, id: u32) -> bool {
        if (id as usize) >= self.texts.len() || !self.is_alive(id) {
            return false;
        }
        let (w, b) = ((id / 64) as usize, (id % 64) as usize);
        if self.tombstones.len() <= w {
            self.tombstones.resize(w + 1, 0);
        }
        self.tombstones[w] |= 1u64 << b;
        true
    }

    pub fn alive_count(&self) -> usize {
        (0..self.texts.len() as u32).filter(|&i| self.is_alive(i)).count()
    }

    pub fn deleted_count(&self) -> usize {
        self.texts.len() - self.alive_count()
    }

    pub fn stats(&self) -> IndexStats {
        let n = self.texts.len();
        let graph_edges = self.graph.edge_count();
        let (ch, cm) = self.source.cache().stats();
        IndexStats {
            chunks: n,
            alive: self.alive_count(),
            deleted: self.deleted_count(),
            dim: self.codec.dim,
            metric: self.cfg.metric,
            pq_m: self.codec.m,
            pq_compression: if self.codec.m > 0 {
                (self.codec.dim * 4) as f32 / self.codec.m as f32
            } else {
                0.0
            },
            text_bytes: self.texts.total_bytes() as u64,
            codes_bytes: self.codes.len() as u64,
            codebook_bytes: (self.codec.codebook.len() * self.codec.ksub * self.codec.subdim * 4) as u64,
            graph_edges,
            graph_bytes: (graph_edges * 4) as u64,
            avg_degree: if n > 0 { graph_edges as f32 / n as f32 } else { 0.0 },
            bm25_vocab: self.bm25.vocab_size(),
            embedder: self.embedder_name.clone(),
            file_bytes: self.file_bytes,
            conventional_vector_bytes: (n * self.codec.dim * 4) as u64,
            cache_hits: ch,
            cache_misses: cm,
        }
    }

    /// Summary as JSON (handy for bindings / CLI).
    pub fn stats_json(&self) -> Value {
        serde_json::to_value(self.stats()).unwrap_or(Value::Null)
    }

    pub fn doc_info(&self, id: u32) -> Value {
        json!({
            "id": id,
            "text": self.texts.text(id),
            "metadata": self.metadata(id),
            "alive": self.is_alive(id),
        })
    }

    /// Garbage-collect deleted chunks: drops tombstoned entries, remaps the
    /// graph in place (no re-embedding), rebuilds BM25/placement, and clears
    /// the embed cache (ids shift). Returns the number of chunks removed.
    ///
    /// Ids of surviving chunks change: the `old -> new` mapping is returned
    /// so callers can translate any references they hold.
    pub fn compact(&mut self) -> Result<Vec<u32>> {
        let old_n = self.texts.len();
        let alive: Vec<u32> = (0..old_n as u32).filter(|&i| self.is_alive(i)).collect();
        if alive.len() == old_n {
            return Ok((0..old_n as u32).collect()); // nothing to do
        }

        let new_n = alive.len();
        // old -> new map; u32::MAX marks deleted
        let mut old_to_new = vec![u32::MAX; old_n];
        for (new_id, &old_id) in alive.iter().enumerate() {
            old_to_new[old_id as usize] = new_id as u32;
        }

        // remap data sections
        let mut new_texts: Vec<String> = Vec::with_capacity(new_n);
        let mut new_metas: Vec<Value> = Vec::with_capacity(new_n);
        let m_bytes = self.codec.m;
        let mut new_codes = vec![0u8; new_n * m_bytes];
        for (new_id, &old_id) in alive.iter().enumerate() {
            new_texts.push(self.texts.text(old_id).to_string());
            new_metas.push(self.metadatas[old_id as usize].clone());
            let src = &self.codes[old_id as usize * m_bytes..(old_id as usize + 1) * m_bytes];
            new_codes[new_id * m_bytes..(new_id + 1) * m_bytes].copy_from_slice(src);
        }

        // remap graph: filter dead neighbors and renumber
        let mut new_graph = crate::graph::HnswGraph::with_capacity(new_n);
        let mut max_level = 0u8;
        let mut entry = 0u32;
        for (new_id, &old_id) in alive.iter().enumerate() {
            let lvl = self.graph.levels[old_id as usize];
            new_graph.levels[new_id] = lvl; // with_capacity pre-fills: assign, don't push
            if lvl > 0 {
                while new_graph.upper[new_id].len() < lvl as usize {
                    new_graph.upper[new_id].push(Vec::new());
                }
                for (li, lst) in self.graph.upper[old_id as usize].iter().enumerate() {
                    new_graph.upper[new_id][li] = lst
                        .iter()
                        .filter_map(|&n| old_to_new.get(n as usize).copied())
                        .filter(|&n| n != u32::MAX)
                        .collect();
                }
            }
            if lvl > max_level {
                max_level = lvl;
                entry = new_id as u32;
            }
            new_graph.adj0[new_id] = self.graph.adj0[old_id as usize]
                .iter()
                .filter_map(|&n| old_to_new.get(n as usize).copied())
                .filter(|&n| n != u32::MAX)
                .collect();
        }
        new_graph.max_level = max_level;
        new_graph.entry = entry;
        new_graph.mark_complete(new_n);

        // rebuild derived state
        let refs: Vec<&str> = new_texts.iter().map(|s| s.as_str()).collect();
        let bm25 = Bm25Index::build(&refs, Bm25Params::default());
        let placement = Self::build_placement(&new_metas, new_n);

        // ids shift: cached embeddings would be wrong — drop them all
        self.source.cache().clear();

        self.texts = TextStore::Owned(new_texts);
        self.metadatas = new_metas;
        self.codes = new_codes;
        self.graph = new_graph;
        self.tombstones = Vec::new();
        self.bm25 = bm25;
        self.placement = placement;
        self.file_bytes = None;
        Ok(old_to_new)
    }

    /// Add chunks with duplicate suppression. Returns `(added_ids, skipped)`
    /// where `skipped` counts texts judged duplicates of *existing index
    /// content or earlier texts in this call* (exact match, or SimHash
    /// Hamming distance <= 3 in [`DedupMode::Near`]).
    pub fn add_with_dedup(
        &mut self,
        texts: &[String],
        metadatas: Option<Vec<Value>>,
        mode: DedupMode,
    ) -> Result<(Vec<u32>, usize)> {
        if mode == DedupMode::Off {
            let ids = self.add(texts, metadatas)?;
            return Ok((ids, 0));
        }
        // hash existing corpus once
        let mut deduper = Deduper::new(mode);
        if self.texts.len() > 0 {
            for i in 0..self.texts.len() as u32 {
                deduper.register(self.texts.text(i));
            }
        }
        let mut keep_texts = Vec::new();
        let mut keep_metas = Vec::new();
        let metas = metadatas.unwrap_or_else(|| vec![Value::Null; texts.len()]);
        let mut skipped = 0usize;
        for (t, m) in texts.iter().zip(metas.into_iter()) {
            if deduper.register(t) {
                skipped += 1;
            } else {
                keep_texts.push(t.clone());
                keep_metas.push(m);
            }
        }
        let ids = if keep_texts.is_empty() {
            Vec::new()
        } else {
            self.add(&keep_texts, Some(keep_metas))?
        };
        Ok((ids, skipped))
    }

    fn build_placement(metas: &[Value], n: usize) -> ChunkPlacement {
        let doc_key_of = |id: u32| -> Option<u64> {
            let m = &metas[id as usize];
            m.get("doc_id")
                .and_then(|v| v.as_str())
                .map(|s| fxhash64(s.as_bytes()))
        };
        let ordinal_of = |id: u32| -> Option<u32> {
            let m = &metas[id as usize];
            m.get("chunk_index").and_then(|v| v.as_u64()).map(|x| x as u32)
        };
        ChunkPlacement::from_records(n, doc_key_of, ordinal_of)
    }
    fn rebuild_placement(&mut self) {
        let n = self.texts.len();
        let metas = std::mem::take(&mut self.metadatas);
        self.placement = Self::build_placement(&metas, n);
        self.metadatas = metas;
    }
}

fn fxhash64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0x517c_c1b7_2722_0a95;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Fill in `doc_id` / `chunk_index` when callers supply raw metadata (or
/// `null`) without placement fields, so context expansion works for every
/// add path. One `add` call becomes one synthetic document; explicit values
/// from the ingest pipeline are always preserved.
fn normalize_placement_metas(metas: &mut [Value]) {
    let missing = |m: &Value| {
        m.get("doc_id").and_then(|v| v.as_str()).is_none()
            || m.get("chunk_index").and_then(|v| v.as_u64()).is_none()
    };
    if !metas.iter().any(missing) {
        return;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let doc_id = format!("add-{nanos:x}-{}", metas.len());
    for (i, m) in metas.iter_mut().enumerate() {
        if !m.is_object() {
            if m.is_null() {
                *m = serde_json::json!({});
            } else {
                continue; // scalar metadata: leave untouched
            }
        }
        let obj = m.as_object_mut().expect("checked above");
        obj.entry("doc_id")
            .or_insert_with(|| Value::String(doc_id.clone()));
        obj.entry("chunk_index")
            .or_insert_with(|| Value::from(i as u64));
    }
}

fn similarity(metric: Metric, distance: f32) -> f32 {
    match metric {
        Metric::L2 => 1.0 / (1.0 + distance),
        Metric::Cosine => 1.0 - distance,
        Metric::InnerProduct => -distance,
    }
}

/// DistOracle for runtime `add`: resolves new nodes from a side table and
/// existing nodes through the (pre-warmed) embedding cache.
struct RuntimeOracle<'a> {
    source: &'a EmbedSource,
    texts: &'a TextStore,
    new_embs: &'a HashMap<u32, Arc<Vec<f32>>>,
    metric: Metric,
}

impl<'a> DistOracle for RuntimeOracle<'a> {
    fn dist(&self, a: u32, b: u32) -> f32 {
        if a == b {
            return 0.0;
        }
        match (self.new_embs.get(&a), self.new_embs.get(&b)) {
            (Some(x), Some(y)) => self.metric.distance(x, y),
            (Some(x), None) => match self.source.batch_by_id(&[b], self.texts) {
                Ok(mut vs) => self.metric.distance(x, &vs.remove(0)),
                Err(_) => f32::INFINITY,
            },
            (None, Some(y)) => match self.source.batch_by_id(&[a], self.texts) {
                Ok(mut vs) => self.metric.distance(&vs.remove(0), y),
                Err(_) => f32::INFINITY,
            },
            (None, None) => {
                let vs = self
                    .source
                    .batch_by_id(&[a, b], self.texts)
                    .unwrap_or_default();
                if vs.len() == 2 {
                    self.metric.distance(&vs[0], &vs[1])
                } else {
                    f32::INFINITY
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // deterministic n-gram embedder for tests
    struct NgramProvider {
        dim: usize,
    }
    impl EmbeddingProvider for NgramProvider {
        fn name(&self) -> &str {
            "test-ngram"
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| ngram_embed(t, self.dim)).collect())
        }
    }

    fn ngram_embed(text: &str, dim: usize) -> Vec<f32> {
        use crate::metric::normalize;
        let mut v = vec![0f32; dim];
        let chars: Vec<char> = text.chars().collect();
        for w in text.split_whitespace() {
            let mut h = 5381u64;
            for b in w.bytes() {
                h = h.wrapping_mul(33) ^ b as u64;
            }
            v[(h as usize) % dim] += 1.0;
        }
        for i in 0..chars.len().saturating_sub(2) {
            let g: String = chars[i..i + 3].iter().collect();
            let mut h = 5381u64;
            for b in g.bytes() {
                h = h.wrapping_mul(33) ^ b as u64;
            }
            v[(h as usize) % dim] += 0.5;
        }
        normalize(&mut v);
        v
    }

    fn corpus(n: usize) -> Vec<String> {
        let topics = [
            "machine learning gradient descent neural network training loss",
            "rust ownership borrow checker lifetimes memory safety zero cost",
            "vector database index approximate nearest neighbor search embedding",
            "coffee brewing espresso roast temperature grind size extraction",
            "planetary orbit gravity telescope astronomy star light years",
            "recipe pasta tomato garlic basil olive oil cooking instructions",
            "quarterly revenue profit margin sales forecast business report",
            "hiking mountain trail forest wilderness camping backpack gear",
        ];
        (0..n)
            .map(|i| {
                format!(
                    "document marker{x} number {i} discusses {}",
                    topics[i % topics.len()],
                    x = format_args!("tok{i:05}")
                )
            })
            .collect()
    }

    fn metas(n: usize) -> Vec<Value> {
        (0..n)
            .map(|i| {
                json!({
                    "doc_id": format!("doc-{}", i / 5),
                    "chunk_index": i % 5,
                    "format": if i % 2 == 0 { "txt" } else { "pdf" },
                    "page": i,
                })
            })
            .collect()
    }

    fn provider() -> Arc<dyn EmbeddingProvider> {
        Arc::new(NgramProvider { dim: 128 })
    }

    fn test_cfg() -> IndexConfig {
        IndexConfig {
            metric: Metric::Cosine,
            pq: PqConfig {
                m: 16,
                train_iters: 6,
                ..Default::default()
            },
            graph: GraphBuildConfig {
                m: 12,
                m0: 24,
                ef_construction: 80,
                seed: 7,
            },
            hub: HubPruneConfig::default(),
            search: SearchDefaults {
                ef: 48,
                alpha: 0.12,
                ..SearchDefaults::default()
            },
            cache_capacity: 4096,
        }
    }

    #[test]
    fn build_search_roundtrip() {
        let texts = corpus(300);
        let metas = metas(300);
        let idx = LeannIndex::build(&texts, Some(metas), provider(), test_cfg()).unwrap();
        assert_eq!(idx.len(), 300);

        let res = idx
            .search(
                "rust borrow checker memory safety",
                &SearchOptions {
                    k: 5,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(res.hits.len(), 5);
        // top hits should be rust docs (ids ≡ 1 mod 8)
        let rust_hits = res
            .hits
            .iter()
            .filter(|h| h.id % 8 == 1)
            .count();
        assert!(rust_hits >= 3, "expected rust-topic hits, got {:?}", res.hits.iter().map(|h| h.id).collect::<Vec<_>>());

        // exact mode agreement
        let exact = idx
            .search(
                "rust borrow checker memory safety",
                &SearchOptions {
                    k: 5,
                    exact: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let a: Vec<u32> = res.hits.iter().map(|h| h.id).collect();
        let b: Vec<u32> = exact.hits.iter().map(|h| h.id).collect();
        let inter = a.iter().filter(|x| b.contains(x)).count();
        assert!(inter >= 3, "recall vs exact too low: {a:?} vs {b:?}");
    }

    #[test]
    fn hybrid_finds_keywords() {
        let texts = corpus(300);
        let idx = LeannIndex::build(&texts, None, provider(), test_cfg()).unwrap();
        // rare exact keyword that embeddings may blur
        let res = idx
            .hybrid_search(
                "espresso grind temperature 42",
                &HybridOptions {
                    base: SearchOptions {
                        k: 5,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(!res.hits.is_empty());
        let coffee = res
            .hits
            .iter()
            .filter(|h| h.id % 8 == 3)
            .count();
        assert!(coffee >= 2, "expected coffee-topic hits: {:?}", res.hits.iter().map(|h| h.id).collect::<Vec<_>>());
    }

    #[test]
    fn add_delete_dynamically() {
        let texts = corpus(200);
        let mut idx = LeannIndex::build(&texts, Some(metas(200)), provider(), test_cfg()).unwrap();

        // add
        let new_texts: Vec<String> = (0..20)
            .map(|i| format!("zqx unique new chunk {i} about quantum computing qubits"))
            .collect();
        let new_metas: Vec<Value> = (0..20)
            .map(|i| json!({ "doc_id": "new-doc", "chunk_index": i }))
            .collect();
        let ids = idx.add(&new_texts, Some(new_metas)).unwrap();
        assert_eq!(ids.len(), 20);
        assert_eq!(idx.len(), 220);

        let res = idx
            .search("quantum computing qubits", &SearchOptions { k: 5, ..Default::default() })
            .unwrap();
        assert!(!res.hits.is_empty());
        let new_hits = res.hits.iter().filter(|h| h.id >= 200).count();
        assert!(new_hits >= 1, "new chunks must be retrievable: {:?}", res.hits.iter().map(|h| h.id).collect::<Vec<_>>());

        // delete the new ones
        let deleted = idx.delete(&ids);
        assert_eq!(deleted, 20);
        assert_eq!(idx.alive_count(), 200);
        let res2 = idx
            .search("quantum computing qubits", &SearchOptions { k: 5, ..Default::default() })
            .unwrap();
        assert!(res2.hits.iter().all(|h| h.id < 200), "deleted chunks must not be returned");
    }

    #[test]
    fn filter_and_context() {
        let texts = corpus(100);
        let idx = LeannIndex::build(&texts, Some(metas(100)), provider(), test_cfg()).unwrap();

        let f = crate::filter::parse_filter(r#"{"op":"eq","field":"format","value":"pdf"}"#).unwrap();
        let res = idx
            .search(
                "machine learning neural network",
                &SearchOptions {
                    k: 10,
                    filter: Some(f),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(!res.hits.is_empty());
        for h in &res.hits {
            assert_eq!(idx.metadata(h.id)["format"], json!("pdf"));
        }

        // context expansion: doc_id groups of 5
        let exp = idx.expand(
            &[7],
            ContextExpansionOptions {
                before: 2,
                after: 2,
            },
        );
        assert_eq!(exp.len(), 1);
        // chunk 7 is in doc-1 (chunks 5..9)
        assert_eq!(exp[0].neighbors.len(), 4);
    }

    #[test]
    fn raw_add_gets_synthetic_placement() {
        // Regression: add() with caller-supplied metadata lacking
        // doc_id/chunk_index must still support context expansion.
        let texts = corpus(60);
        let mut idx = LeannIndex::build(&texts, None, provider(), test_cfg()).unwrap();

        // incremental add with custom (placement-less) metadata
        let extra: Vec<String> = (0..5).map(|i| format!("synthetic chunk {i} about zebras")).collect();
        let metas: Vec<Value> = (0..5)
            .map(|i| json!({ "topic": format!("t{i}") }))
            .collect();
        let ids = idx.add(&extra, Some(metas)).unwrap();
        assert_eq!(ids.len(), 5);

        // the middle synthetic chunk must expand to its siblings
        let mid = ids[2];
        let exp = idx.expand(
            &[mid],
            ContextExpansionOptions {
                before: 1,
                after: 1,
            },
        );
        assert_eq!(exp.len(), 1);
        assert_eq!(exp[0].neighbors.len(), 2, "expected +/-1 neighbors");
        let (n_before, off_before) = exp[0].neighbors[0];
        let (n_after, off_after) = exp[0].neighbors[1];
        assert_eq!(off_before, -1);
        assert_eq!(off_after, 1);
        assert_eq!(n_before, ids[1]);
        assert_eq!(n_after, ids[3]);
        // metadata got the synthetic fields added (original fields kept)
        let m = idx.metadata(mid);
        assert_eq!(m["topic"], json!("t2"));
        assert!(m["doc_id"].as_str().unwrap().starts_with("add-"));
        assert_eq!(m["chunk_index"], json!(2));
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = std::env::temp_dir().join("leann-test-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("idx.leann");
        let texts = corpus(150);
        let metas = metas(150);
        let idx = LeannIndex::build(&texts, Some(metas), provider(), test_cfg()).unwrap();
        idx.save(&path).unwrap();

        let loaded = LeannIndex::open(&path, provider()).unwrap();
        assert_eq!(loaded.len(), 150);

        let r1 = idx.search("vector database search", &SearchOptions::default()).unwrap();
        let r2 = loaded.search("vector database search", &SearchOptions::default()).unwrap();
        let a: Vec<u32> = r1.hits.iter().map(|h| h.id).collect();
        let b: Vec<u32> = r2.hits.iter().map(|h| h.id).collect();
        assert_eq!(a, b, "loaded index must produce identical results");

        // file should be lean: no raw embeddings
        let size = std::fs::metadata(&path).unwrap().len() as f64;
        let conventional = 150.0 * 128.0 * 4.0;
        assert!(
            size < conventional * 0.25 + 200_000.0, // header+texts+json slack
            "index file too large: {size} bytes"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sharded_build_matches_single() {
        let texts = corpus(400);
        let p = provider();
        let single = LeannIndex::build(&texts, None, p.clone(), test_cfg()).unwrap();
        let sharded =
            LeannIndex::build_sharded(&texts, None, p.clone(), test_cfg(), 100).unwrap();
        assert_eq!(sharded.len(), 400);

        let q = "hiking mountain trail wilderness";
        let r1 = single.search(q, &SearchOptions { k: 5, ..Default::default() }).unwrap();
        let r2 = sharded.search(q, &SearchOptions { k: 5, ..Default::default() }).unwrap();
        let a: Vec<u32> = r1.hits.iter().map(|h| h.id).collect();
        let b: Vec<u32> = r2.hits.iter().map(|h| h.id).collect();
        let inter = a.iter().filter(|x| b.contains(x)).count();
        assert!(inter >= 3, "sharded recall too different: {a:?} vs {b:?}");
    }

    #[test]
    fn compact_reclaims_tombstones() {
        let texts = corpus(120);
        let mut idx = LeannIndex::build(&texts, None, provider(), test_cfg()).unwrap();
        // delete every 3rd chunk
        let doomed: Vec<u32> = (0..120).step_by(3).collect();
        idx.delete(&doomed);
        assert_eq!(idx.stats().deleted, 40);

        let map = idx.compact().unwrap();
        assert_eq!(map.len(), 120);
        let s = idx.stats();
        assert_eq!(s.chunks, 80, "compacted chunk count");
        assert_eq!(s.deleted, 0, "tombstones cleared");

        // search still works and returns only alive chunks
        let res = idx
            .search("machine learning", &SearchOptions { k: 10, ..Default::default() })
            .unwrap();
        assert!(!res.hits.is_empty());
        for h in &res.hits {
            assert!(h.id < 80, "ids must be remapped into [0,80)");
        }
        // old->new mapping is a permutation onto alive slots
        let mapped: std::collections::HashSet<u32> = map
            .iter()
            .filter(|&&n| n != u32::MAX)
            .copied()
            .collect();
        assert_eq!(mapped.len(), 80);

        // regression: compact + save + load must roundtrip (levels were
        // double-pushed when with_capacity pre-filled them)
        let dir = std::env::temp_dir().join("leann-test-compact");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.leann");
        idx.save(&path).unwrap();
        let re = LeannIndex::open(&path, provider()).unwrap();
        assert_eq!(re.stats().chunks, 80);
        let res2 = re
            .search("machine learning", &SearchOptions { k: 5, ..Default::default() })
            .unwrap();
        assert!(!res2.hits.is_empty());
    }

    #[test]
    fn encrypted_roundtrip() {
        let dir = std::env::temp_dir().join("leann-test-enc");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("secret.leann");
        let texts = corpus(80);
        let idx = LeannIndex::build(&texts, None, provider(), test_cfg()).unwrap();
        idx.save_encrypted(&path, "open sesame").unwrap();

        // file must NOT contain the plaintext magic or any raw text
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..8], b"LEANNENC");
        assert!(!bytes.windows(16).any(|w| w == b"machine learning"));

        // wrong passphrase rejected
        assert!(LeannIndex::open_with_passphrase(&path, "wrong", provider()).is_err());
        // no passphrase rejected
        assert!(LeannIndex::open(&path, provider()).is_err());
        // correct passphrase works and searches
        let opened = LeannIndex::open_with_passphrase(&path, "open sesame", provider()).unwrap();
        let res = opened
            .search("vector search", &SearchOptions { k: 5, ..Default::default() })
            .unwrap();
        assert!(!res.hits.is_empty());
    }

    #[test]
    fn batch_search_matches_single() {
        let texts = corpus(200);
        let idx = LeannIndex::build(&texts, None, provider(), test_cfg()).unwrap();
        let queries: Vec<String> = (0..8)
            .map(|i| format!("query number {i} about {}", ["machine learning", "rust ownership", "vector search"][i % 3]))
            .collect();
        let refs: Vec<&str> = queries.iter().map(|s| s.as_str()).collect();
        let opts = SearchOptions { k: 5, ..Default::default() };

        let batch = idx.search_batch(&refs, &opts).unwrap();
        assert_eq!(batch.len(), 8);
        for (i, r) in batch.iter().enumerate() {
            let single = idx.search(&queries[i], &opts).unwrap();
            let bid: Vec<u32> = r.hits.iter().map(|h| h.id).collect();
            let sid: Vec<u32> = single.hits.iter().map(|h| h.id).collect();
            // identical when the embed cache is warm for both
            assert_eq!(bid, sid, "query {i} differs between batch and single");
        }
    }

    #[test]
    fn dedup_add_suppresses_duplicates() {
        let texts = corpus(60);
        let mut idx = LeannIndex::build(&texts, None, provider(), test_cfg()).unwrap();
        let base = "The quarterly report shows revenue growth of 12 percent driven by enterprise sales in the EMEA region";
        let dupes: Vec<String> = vec![
            base.to_string(),
            base.to_string(),                                  // exact
            "The quarterly report shows revenue growth of 13 percent driven by enterprise sales in the EMEA region".to_string(), // near
            "A completely novel sentence about zebras crossing the river at dawn while crocodiles sleep".to_string(),          // keep
        ];
        let (ids, skipped) = idx.add_with_dedup(&dupes, None, DedupMode::Near).unwrap();
        assert_eq!(ids.len(), 2, "2 of 4 kept");
        assert_eq!(skipped, 2);
        // exact mode keeps the near-dup
        let more: Vec<String> = vec![base.to_string()];
        let (ids2, skipped2) = idx.add_with_dedup(&more, None, DedupMode::Exact).unwrap();
        assert_eq!(ids2.len(), 0);
        assert_eq!(skipped2, 1);
    }
}
