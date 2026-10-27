//! # leann-node
//!
//! TypeScript / Node.js bindings for LEANN via napi-rs.
//!
//! ```ts
//! import { Leann } from "leann-node";
//!
//! const leann = new Leann({ embedder: { kind: "local-hash", dim: 384 } });
//! await leann.addFiles(["paper.pdf", "notes.md", "src/main.rs"]);
//! const res = await leann.hybridSearch("graph pruning storage", { k: 5 });
//! const chat = await leann.chat("How does LEANN reduce storage?");
//! ```

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;

use leann_chat::provider::{ChatMessage, LlmProvider};
use leann_chat::rag::{RagAnswer, RagConfig, SourceRef};
use leann_chat::{create_llm, LlmConfig, LlmKind};
use leann_guard::{Guardian as CoreGuardian, OpenAiJudge};
use leann_core::context::ContextExpansionOptions;
use leann_core::filter::Filter;
use leann_core::graph::GraphBuildConfig;
use leann_core::index::{
    HybridOptions, IndexConfig, LeannIndex, SearchOptions,
};
use leann_core::metric::Metric;
use leann_embedder::{create_embedder, EmbedderConfig, EmbedderKind};
use leann_ingest::{ingest_files, IngestOptions, OpenAiTranscriber, Transcriber};

// ---------------------------------------------------------------------------
// option objects (TS-friendly)
// ---------------------------------------------------------------------------

#[napi(object)]
pub struct EmbedderOptions {
    /// "openai" (default) or "local-hash"
    pub kind: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub dim: Option<u32>,
    pub batch_size: Option<u32>,
    pub concurrency: Option<u32>,
    pub max_retries: Option<u32>,
    pub send_dimensions_param: Option<bool>,
}

#[napi(object)]
pub struct LlmOptions {
    /// "openai" (default) or "anthropic"
    pub kind: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
}

#[napi(object)]
pub struct TranscriberOptions {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub language: Option<String>,
}

#[napi(object)]
pub struct ChunkerOptions {
    pub chunk_tokens: Option<u32>,
    pub overlap_tokens: Option<u32>,
    pub min_tokens: Option<u32>,
    pub max_tokens: Option<u32>,
}

#[napi(object)]
pub struct LeannOptions {
    pub embedder: EmbedderOptions,
    pub llm: Option<LlmOptions>,
    pub transcriber: Option<TranscriberOptions>,
    /// "cosine" (default) | "l2" | "ip"
    pub metric: Option<String>,
    pub chunker: Option<ChunkerOptions>,
    /// Embedding cache size (per index)
    pub cache_capacity: Option<u32>,
    pub extra_metadata: Option<Value>,
    /// IBM Granite Guardian judge (enables answer verification)
    pub guardian: Option<GuardianOptions>,
}

#[napi(object)]
pub struct SearchArgs {
    pub k: Option<u32>,
    pub ef: Option<u32>,
    pub alpha: Option<f64>,
    pub min_batch: Option<u32>,
    pub max_batch: Option<u32>,
    pub relax: Option<f64>,
    /// Brute-force exact scan (slow, small corpora)
    pub exact: Option<bool>,
    pub with_text: Option<bool>,
    pub with_metadata: Option<bool>,
    /// JSON filter, e.g. { op: "eq", field: "format", value: "pdf" }
    pub filter: Option<Value>,
}

#[napi(object)]
pub struct HybridArgs {
    pub k: Option<u32>,
    /// "rrf" (default) or "weighted"
    pub fusion: Option<String>,
    pub vector_weight: Option<f64>,
    pub lexical_weight: Option<f64>,
    pub vector_overfetch: Option<f64>,
    pub lexical_k: Option<u32>,
    pub expand_before: Option<u32>,
    pub expand_after: Option<u32>,
    pub filter: Option<Value>,
    pub with_text: Option<bool>,
}

#[napi(object)]
pub struct ChatArgs {
    pub guard: Option<GuardArgs>,
    pub top_k: Option<u32>,
    pub use_hybrid: Option<bool>,
    pub expand_before: Option<u32>,
    pub expand_after: Option<u32>,
    pub filter: Option<Value>,
    pub system_prompt: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub max_context_chars: Option<u32>,
    /// Reset conversation memory before this question
    pub fresh: Option<bool>,
}

#[napi(object)]
pub struct IngestArgs {
    pub chunker: Option<ChunkerOptions>,
    pub extra_metadata: Option<Value>,
}

/// IBM Granite Guardian verification options (leann-guard crate).
#[napi(object)]
pub struct GuardArgs {
    /// Minimum groundedness confidence (0-1) to accept; default 0.7
    pub groundedness_threshold: Option<f64>,
    /// Repair attempts when checks fail; default 1
    pub repair_attempts: Option<u32>,
    /// Run harm checks; default true
    pub harm_check: Option<bool>,
    /// Policy: "abstain" (default) | "repair" | "annotate" | "log"
    pub policy: Option<String>,
    /// Sentence-level audit (default true) vs whole-answer verdict
    pub sentence_level: Option<bool>,
}

/// Options for the standalone Guardian judge.
#[napi(object)]
pub struct GuardianOptions {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    /// e.g. "ibm-granite/granite-guardian-3.3-8b" (served) or "gpt-4o-mini"
    pub model: Option<String>,
    /// Think mode: reasoning traces before scores (default false)
    pub think: Option<bool>,
}

// ---------------------------------------------------------------------------
// result objects
// ---------------------------------------------------------------------------

#[napi(object)]
pub struct Hit {
    pub id: u32,
    pub distance: f64,
    pub similarity: f64,
    pub text: Option<String>,
    pub metadata: Option<Value>,
}

#[napi(object)]
pub struct SearchStatsJs {
    pub hops: u32,
    pub pq_scored: u32,
    pub promoted: u32,
    pub embed_calls: u32,
    pub vectors_embedded: u32,
    pub embed_secs: f64,
    pub took_ms: f64,
}

#[napi(object)]
pub struct SearchResultJs {
    pub hits: Vec<Hit>,
    pub stats: SearchStatsJs,
}

#[napi(object)]
pub struct HybridHit {
    pub id: u32,
    pub fused_score: f64,
    pub vector_rank: Option<u32>,
    pub lexical_rank: Option<u32>,
    pub vector_similarity: Option<f64>,
    pub lexical_score: Option<f64>,
    pub text: Option<String>,
    pub metadata: Option<Value>,
}

#[napi(object)]
pub struct ExpansionJs {
    pub primary: u32,
    /// neighbor chunk ids in reading order
    pub neighbors: Vec<u32>,
    /// offsets from primary (negative = before)
    pub offsets: Vec<i32>,
}

#[napi(object)]
pub struct HybridResultJs {
    pub hits: Vec<HybridHit>,
    pub expanded: Vec<ExpansionJs>,
    pub took_ms: f64,
    pub stats: SearchStatsJs,
}

#[napi(object)]
pub struct SourceRefJs {
    pub chunk_id: u32,
    pub source: Option<String>,
    pub score: Option<f64>,
    pub text: Option<String>,
    pub metadata: Option<Value>,
}

#[napi(object)]
pub struct ChatResultJs {
    pub answer: String,
    pub sources: Vec<SourceRefJs>,
    pub query: String,
    pub guard: Option<GuardReportJs>,
}

/// Guard report (subset, JSON-friendly) for chat results.
#[napi(object)]
pub struct GuardReportJs {
    pub risk_triggered: bool,
    pub risks: Vec<String>,
    pub action: String,
    pub repaired: bool,
    pub groundedness_confidence: Option<f64>,
    pub ungrounded_sentences: u32,
    pub total_sentences: u32,
}

/// One sentence's grounding verdict.
#[napi(object)]
pub struct VerifiedSentenceJs {
    pub text: String,
    pub grounded: bool,
    pub confidence: f64,
    pub supporting: Vec<u32>,
}

/// One guardian verdict.
#[napi(object)]
pub struct GuardianVerdictJs {
    pub risk_id: String,
    pub triggered: bool,
    pub confidence: f64,
    pub trace: Option<String>,
    pub explanation: Option<String>,
}

#[napi(object)]
pub struct AddResult {
    pub chunk_ids: Vec<u32>,
    pub chunks_added: u32,
    pub total_chunks: u32,
}

#[napi(object)]
pub struct StatsJs {
    pub chunks: u32,
    pub alive: u32,
    pub deleted: u32,
    pub dim: u32,
    pub metric: String,
    pub pq_m: u32,
    pub pq_compression: f64,
    pub text_bytes: f64,
    pub codes_bytes: f64,
    pub graph_edges: u32,
    pub avg_degree: f64,
    /// Bytes a conventional vector index would need for raw embeddings
    pub conventional_vector_bytes: f64,
    pub storage_saved_percent: f64,
    pub bm25_vocab: u32,
    pub embedder: String,
    pub file_bytes: Option<f64>,
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn to_napi_err<E: std::fmt::Display>(e: E) -> Error {
    Error::new(Status::GenericFailure, format!("{e}"))
}

fn parse_filter(v: Option<Value>) -> Result<Option<Filter>> {
    match v {
        None => Ok(None),
        Some(val) if val.is_null() => Ok(None),
        Some(val) => serde_json::from_value(val)
            .map(Some)
            .map_err(|e| Error::new(Status::InvalidArg, format!("invalid filter: {e}"))),
    }
}

fn metric_from_str(s: Option<&str>) -> Metric {
    match s.map(|x| x.to_lowercase()).as_deref() {
        Some("l2") | Some("euclidean") => Metric::L2,
        Some("ip") | Some("inner") | Some("innerproduct") => Metric::InnerProduct,
        _ => Metric::Cosine,
    }
}

fn build_index_config(opts: &LeannOptions) -> IndexConfig {
    let metric = metric_from_str(opts.metric.as_deref());
    let mut cfg = IndexConfig {
        metric,
        ..Default::default()
    };
    if let Some(c) = &opts.chunker {
        if let Some(t) = c.chunk_tokens {
            cfg.graph = GraphBuildConfig::default(); // graph params stay default
            let _ = t;
        }
    }
    if let Some(cap) = opts.cache_capacity {
        cfg.cache_capacity = cap as usize;
    }
    cfg
}

fn ingest_opts(args: Option<&IngestArgs>, base: Option<&Value>) -> IngestOptions {
    let mut io = IngestOptions::default();
    if let Some(b) = base {
        io.extra_metadata = b.clone();
    }
    if let Some(a) = args {
        if let Some(c) = &a.chunker {
            if let Some(t) = c.chunk_tokens {
                io.chunker.chunk_tokens = t as usize;
            }
            if let Some(t) = c.overlap_tokens {
                io.chunker.overlap_tokens = t as usize;
            }
            if let Some(t) = c.min_tokens {
                io.chunker.min_tokens = t as usize;
            }
            if let Some(t) = c.max_tokens {
                io.chunker.max_tokens = t as usize;
            }
        }
        if let Some(m) = &a.extra_metadata {
            io.extra_metadata = m.clone();
        }
    }
    io
}

// ---------------------------------------------------------------------------
// the Leann class
// ---------------------------------------------------------------------------

#[napi]
pub struct Leann {
    inner: Arc<RwLock<LeannIndex>>,
    llm: Option<Arc<dyn LlmProvider>>,
    transcriber: Option<Arc<dyn Transcriber>>,
    base_metadata: Option<Value>,
    rag: Arc<std::sync::Mutex<RagState>>,
    guardian: Option<CoreGuardian>,
}

struct RagState {
    history: Vec<ChatMessage>,
}

impl Leann {
    fn build_llm(o: &Option<LlmOptions>) -> Result<Option<Arc<dyn LlmProvider>>> {
        match o {
            None => Ok(None),
            Some(llm) => {
                let cfg = LlmConfig {
                    kind: match llm.kind.as_deref() {
                        Some("anthropic") | Some("claude") => LlmKind::Anthropic,
                        _ => LlmKind::OpenAi,
                    },
                    base_url: llm.base_url.clone(),
                    api_key: llm.api_key.clone(),
                    model: llm.model.clone(),
                    temperature: llm.temperature.map(|t| t as f32),
                    max_tokens: llm.max_tokens,
                };
                create_llm(&cfg).map(Some).map_err(to_napi_err)
            }
        }
    }

    fn embedder(o: &EmbedderOptions) -> Result<Arc<dyn leann_core::cache::EmbeddingProvider>> {
        let kind = match o.kind.as_deref() {
            Some("local-hash") | Some("local") | Some("hash") => EmbedderKind::LocalHash,
            _ => EmbedderKind::OpenAi,
        };
        let dim = o.dim.map(|d| d as usize).unwrap_or(if matches!(kind, EmbedderKind::LocalHash) { 384 } else { 1536 });
        let cfg = EmbedderConfig {
            kind,
            base_url: o.base_url.clone(),
            api_key: o.api_key.clone(),
            model: o.model.clone(),
            dim,
            batch_size: o.batch_size.map(|b| b as usize).unwrap_or(128),
            concurrency: o.concurrency.map(|c| c as usize).unwrap_or(4),
            max_retries: o.max_retries.unwrap_or(3),
            send_dimensions_param: o.send_dimensions_param.unwrap_or(false),
        };
        create_embedder(&cfg).map_err(to_napi_err)
    }

    fn build_guardian(o: &Option<GuardianOptions>) -> Option<CoreGuardian> {
        o.as_ref().map(|g| {
            let judge = OpenAiJudge::new(
                g.base_url
                    .clone()
                    .unwrap_or_else(|| "http://localhost:11434/v1".into()),
                g.api_key.clone().unwrap_or_default(),
                g.model
                    .clone()
                    .unwrap_or_else(|| "ibm-granite/granite-guardian-3.3-8b".into()),
            )
            .with_think(g.think.unwrap_or(false));
            CoreGuardian::with_judge(judge)
        })
    }

    fn make_transcriber(o: &Option<TranscriberOptions>) -> Option<Arc<dyn Transcriber>> {
        o.as_ref().map(|t| {
            let mut tr = OpenAiTranscriber::with_base(
                t.base_url.as_deref().unwrap_or("https://api.openai.com/v1"),
                t.api_key.clone().unwrap_or_default(),
                t.model.as_deref().unwrap_or("whisper-1"),
            );
            if let Some(lang) = &t.language {
                tr = tr.with_language(Some(lang.clone()));
            }
            Arc::new(tr) as Arc<dyn Transcriber>
        })
    }

    fn guard_settings_from(a: &ChatArgs) -> Option<leann_chat::rag::GuardSettings> {
        a.guard.as_ref().map(|g| leann_chat::rag::GuardSettings {
            policy: match g.policy.as_deref() {
                Some("abstain") => leann_guard::GuardPolicy::Abstain,
                Some("repair") => leann_guard::GuardPolicy::Repair,
                Some("annotate") => leann_guard::GuardPolicy::Annotate,
                Some("log") | Some("logonly") => leann_guard::GuardPolicy::LogOnly,
                _ => leann_guard::GuardPolicy::Repair,
            },
            repair_attempts: g.repair_attempts.unwrap_or(1) as usize,
            groundedness_threshold: g.groundedness_threshold.unwrap_or(0.7) as f32,
            harm_check: g.harm_check.unwrap_or(true),
            sentence_level: g.sentence_level.unwrap_or(true),
        })
    }

    fn guard_report_js(r: &leann_guard::GuardReport) -> GuardReportJs {
        let g = r.groundedness();
        GuardReportJs {
            risk_triggered: r.risk_triggered,
            risks: r.risks.iter().map(|x| x.as_str().to_string()).collect(),
            action: format!("{:?}", r.action).to_lowercase(),
            repaired: r.repaired,
            groundedness_confidence: g.map(|v| v.confidence as f64),
            ungrounded_sentences: r.sentences.iter().filter(|s| !s.keep).count() as u32,
            total_sentences: r.sentences.len() as u32,
        }
    }

    fn rag_cfg_from(args: Option<&ChatArgs>) -> RagConfig {
        let mut cfg = RagConfig::default();
        if let Some(a) = args {
            cfg.guard = Self::guard_settings_from(a);
            if let Some(k) = a.top_k {
                cfg.top_k = k as usize;
            }
            if let Some(h) = a.use_hybrid {
                cfg.use_hybrid = h;
            }
            if a.expand_before.is_some() || a.expand_after.is_some() {
                cfg.context_expansion = Some(ContextExpansionOptions {
                    before: a.expand_before.unwrap_or(0) as usize,
                    after: a.expand_after.unwrap_or(0) as usize,
                });
            }
            if let Some(sp) = &a.system_prompt {
                cfg.system_prompt = Some(sp.clone());
            }
            cfg.temperature = a.temperature.map(|t| t as f32);
            cfg.max_tokens = a.max_tokens;
            if let Some(m) = a.max_context_chars {
                cfg.max_context_chars = m as usize;
            }
        }
        cfg
    }

    fn source_ref_js(s: &SourceRef) -> SourceRefJs {
        SourceRefJs {
            chunk_id: s.chunk_id,
            source: s.source.clone(),
            score: s.score.map(|f| f as f64),
            text: s.text.clone(),
            metadata: s.metadata.clone(),
        }
    }

    fn answer_js(a: &RagAnswer) -> ChatResultJs {
        ChatResultJs {
            answer: a.answer.clone(),
            sources: a.sources.iter().map(Leann::source_ref_js).collect(),
            query: a.query.clone(),
            guard: a.guard.as_ref().map(Leann::guard_report_js),
        }
    }
}

#[napi]
impl Leann {
    /// Create an in-memory index. Use `Leann.open` to load a saved one.
    #[napi(constructor)]
    pub fn new(opts: LeannOptions) -> Result<Self> {
        let provider = Self::embedder(&opts.embedder)?;
        let cfg = build_index_config(&opts);
        let index = LeannIndex::empty(cfg, provider);
        let llm = Self::build_llm(&opts.llm)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(index)),
            llm,
            transcriber: Self::make_transcriber(&opts.transcriber),
            base_metadata: opts.extra_metadata.clone(),
            rag: Arc::new(std::sync::Mutex::new(RagState { history: Vec::new() })),
            guardian: Self::build_guardian(&opts.guardian),
        })
    }

    /// Open a saved index file (single-file portable `.leann`).
    #[napi]
    pub fn open(path: String, opts: LeannOptions) -> Result<Self> {
        let provider = Self::embedder(&opts.embedder)?;
        let index = LeannIndex::open(&PathBuf::from(path), provider).map_err(to_napi_err)?;
        let llm = Self::build_llm(&opts.llm)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(index)),
            llm,
            transcriber: Self::make_transcriber(&opts.transcriber),
            base_metadata: opts.extra_metadata.clone(),
            rag: Arc::new(std::sync::Mutex::new(RagState { history: Vec::new() })),
            guardian: None,
        })
    }

    /// Ingest files (pdf, txt, md, html, docx, epub, json, code, audio,
    /// video — audio/video require transcriber options).
    #[napi]
    pub async fn add_files(
        &self,
        paths: Vec<String>,
        args: Option<IngestArgs>,
    ) -> Result<AddResult> {
        let io = ingest_opts(args.as_ref(), self.base_metadata.as_ref());
        let pb: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
        let transcriber = self.transcriber.clone();
        let chunks = tokio::task::spawn_blocking(move || {
            ingest_files(&pb, &io, transcriber)
        })
        .await
        .map_err(|e| to_napi_err(e))?
        .map_err(to_napi_err)?;

        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let metas: Vec<Value> = chunks.iter().map(|c| c.metadata.clone()).collect();
        self.add_texts_internal(texts, Some(metas)).await
    }

    /// Add raw text chunks (chat memory, pre-processed content).
    #[napi]
    pub async fn add_texts(
        &self,
        texts: Vec<String>,
        metadatas: Option<Vec<Value>>,
    ) -> Result<AddResult> {
        self.add_texts_internal(texts, metadatas).await
    }

    async fn add_texts_internal(
        &self,
        texts: Vec<String>,
        metadatas: Option<Vec<Value>>,
    ) -> Result<AddResult> {
        if texts.is_empty() {
            return Ok(AddResult {
                chunk_ids: Vec::new(),
                chunks_added: 0,
                total_chunks: self.inner.read().unwrap().len() as u32,
            });
        }
        let inner = self.inner.clone();
        let ids = tokio::task::spawn_blocking(move || {
            let mut idx = inner.write().unwrap();
            idx.add(&texts, metadatas)
        })
        .await
        .map_err(to_napi_err)?
        .map_err(to_napi_err)?;
        let total = self.inner.read().unwrap().len() as u32;
        Ok(AddResult {
            chunks_added: ids.len() as u32,
            chunk_ids: ids,
            total_chunks: total,
        })
    }

    /// Pure vector search (LEANN two-level).
    #[napi]
    pub async fn search(&self, query: String, args: Option<SearchArgs>) -> Result<SearchResultJs> {
        let opts = self.search_opts(&args)?;
        let inner = self.inner.clone();
        let q = query.clone();
        let res = tokio::task::spawn_blocking(move || {
            let idx = inner.read().unwrap();
            idx.search(&q, &opts)
        })
        .await
        .map_err(to_napi_err)?
        .map_err(to_napi_err)?;
        Ok(SearchResultJs {
            hits: res
                .hits
                .iter()
                .map(|h| Hit {
                    id: h.id,
                    distance: h.distance as f64,
                    similarity: h.similarity as f64,
                    text: h.text.clone(),
                    metadata: h.metadata.clone(),
                })
                .collect(),
            stats: SearchStatsJs {
                hops: res.stats.hops as u32,
                pq_scored: res.stats.pq_scored as u32,
                promoted: res.stats.promoted as u32,
                embed_calls: res.stats.embed_calls as u32,
                vectors_embedded: res.stats.vectors_embedded as u32,
                embed_secs: res.stats.embed_secs,
                took_ms: res.took_ms,
            },
        })
    }

    /// Hybrid search: LEANN vectors + BM25 + reciprocal-rank fusion,
    /// optional neighbor-context expansion.
    #[napi]
    pub async fn hybrid_search(
        &self,
        query: String,
        args: Option<HybridArgs>,
    ) -> Result<HybridResultJs> {
        let filter = args.as_ref().and_then(|a| a.filter.clone());
        let filter = parse_filter(filter)?;
        let mut opts = HybridOptions::default();
        if let Some(a) = &args {
            opts.base.k = a.k.map(|k| k as usize).unwrap_or(10);
            if let Some(f) = &a.fusion {
                if f == "weighted" {
                    opts.fusion = leann_core::hybrid::FusionMode::Weighted;
                }
            }
            if let Some(w) = a.vector_weight {
                opts.weights.vector = w as f32;
            }
            if let Some(w) = a.lexical_weight {
                opts.weights.lexical = w as f32;
            }
            if let Some(v) = a.vector_overfetch {
                opts.vector_overfetch = v as f32;
            }
            if let Some(lk) = a.lexical_k {
                opts.lexical_k = Some(lk as usize);
            }
            if a.expand_before.is_some() || a.expand_after.is_some() {
                opts.expand = Some(ContextExpansionOptions {
                    before: a.expand_before.unwrap_or(0) as usize,
                    after: a.expand_after.unwrap_or(0) as usize,
                });
            }
            if let Some(wt) = a.with_text {
                opts.base.with_text = wt;
            }
        }
        opts.base.filter = filter;

        let inner = self.inner.clone();
        let q = query;
        let res = tokio::task::spawn_blocking(move || {
            let idx = inner.read().unwrap();
            idx.hybrid_search(&q, &opts)
        })
        .await
        .map_err(to_napi_err)?
        .map_err(to_napi_err)?;

        Ok(HybridResultJs {
            hits: res
                .hits
                .iter()
                .map(|h| HybridHit {
                    id: h.id,
                    fused_score: h.fused_score as f64,
                    vector_rank: h.vector_rank,
                    lexical_rank: h.lexical_rank,
                    vector_similarity: h.vector_similarity.map(|f| f as f64),
                    lexical_score: h.lexical_score.map(|f| f as f64),
                    text: h.text.clone(),
                    metadata: h.metadata.clone(),
                })
                .collect(),
            expanded: res
                .expanded
                .iter()
                .map(|e| ExpansionJs {
                    primary: e.primary,
                    neighbors: e.neighbors.iter().map(|(id, _)| *id).collect(),
                    offsets: e.neighbors.iter().map(|(_, off)| *off).collect(),
                })
                .collect(),
            took_ms: res.took_ms,
            stats: SearchStatsJs {
                hops: res.stats.hops as u32,
                pq_scored: res.stats.pq_scored as u32,
                promoted: res.stats.promoted as u32,
                embed_calls: res.stats.embed_calls as u32,
                vectors_embedded: res.stats.vectors_embedded as u32,
                embed_secs: res.stats.embed_secs,
                took_ms: res.took_ms,
            },
        })
    }

    /// RAG chat: retrieve → cited context → LLM. Maintains conversation
    /// memory across calls (clear with `clearHistory` or `fresh: true`).
    #[napi]
    pub async fn chat(&self, query: String, args: Option<ChatArgs>) -> Result<ChatResultJs> {
        let llm = self
            .llm
            .clone()
            .ok_or_else(|| Error::new(Status::GenericFailure, "no LLM configured: pass llm options to the constructor"))?;
        let filter = args.as_ref().and_then(|a| a.filter.clone());
        let filter = parse_filter(filter)?;
        let mut cfg = Self::rag_cfg_from(args.as_ref());
        cfg.filter = filter;

        let inner = self.inner.clone();
        let rag_state = self.rag.clone();
        let fresh = args.as_ref().and_then(|a| a.fresh).unwrap_or(false);
        let guard_guardian = self.guardian.as_ref().map(|g| g.clone_guardian());

        let answer = tokio::task::spawn_blocking(move || -> Result<RagAnswer> {
            let idx = inner.read().unwrap();
            // retrieve context (hybrid + optional expansion + filter)
            let mut hits: Vec<(u32, f64)> = Vec::new();
            if cfg.use_hybrid {
                let mut ho = HybridOptions {
                    base: SearchOptions {
                        k: cfg.top_k,
                        filter: cfg.filter.clone(),
                        ..Default::default()
                    },
                    expand: cfg.context_expansion,
                    ..Default::default()
                };
                if ho.base.k < cfg.top_k {
                    ho.base.k = cfg.top_k;
                }
                if let Ok(res) = idx.hybrid_search(&query, &ho) {
                    hits = res.hits.iter().map(|h| (h.id, h.fused_score as f64)).collect();
                }
            } else if let Ok(res) = idx.search(&query, &SearchOptions {
                k: cfg.top_k,
                filter: cfg.filter.clone(),
                ..Default::default()
            }) {
                hits = res.hits.iter().map(|h| (h.id, h.similarity as f64)).collect();
            }
            let context: Vec<(u32, String, Value)> = hits
                .iter()
                .map(|&(id, _)| {
                    (
                        id,
                        idx.text(id).to_string(),
                        idx.metadata(id).clone(),
                    )
                })
                .collect();

            let history = if fresh {
                Vec::new()
            } else {
                rag_state.lock().unwrap().history.clone()
            };
            let ans = leann_chat::answer_with_context_guarded(
                llm.as_ref(),
                &query,
                &context,
                &history,
                &cfg,
                guard_guardian.as_ref(),
            )
            .map_err(to_napi_err)?;

            let new_history: Vec<ChatMessage> = if fresh {
                vec![
                    ChatMessage::user(query.clone()),
                    ChatMessage::assistant(ans.answer.clone()),
                ]
            } else {
                let mut h = history;
                h.push(ChatMessage::user(query.clone()));
                h.push(ChatMessage::assistant(ans.answer.clone()));
                h
            };
            let mut st = rag_state.lock().unwrap();
            st.history = new_history;
            if st.history.len() > 20 {
                let drop = st.history.len() - 20;
                st.history.drain(0..drop);
            }
            Ok(ans)
        })
        .await
        .map_err(to_napi_err)??;

        Ok(Self::answer_js(&answer))
    }

    /// Clear RAG conversation memory.
    #[napi]
    pub fn clear_history(&self) {
        self.rag.lock().unwrap().history.clear();
    }

    /// Expand context around chunk ids (neighbors within the same document).
    #[napi]
    pub fn expand_context(
        &self,
        ids: Vec<u32>,
        before: Option<u32>,
        after: Option<u32>,
    ) -> Result<Vec<ExpansionJs>> {
        let idx = self.inner.read().unwrap();
        let opts = ContextExpansionOptions {
            before: before.unwrap_or(1) as usize,
            after: after.unwrap_or(1) as usize,
        };
        let exp = idx.expand(&ids, opts);
        Ok(exp
            .iter()
            .map(|e| ExpansionJs {
                primary: e.primary,
                neighbors: e.neighbors.iter().map(|(id, _)| *id).collect(),
                offsets: e.neighbors.iter().map(|(_, off)| *off).collect(),
            })
            .collect())
    }

    /// Delete chunks by id.
    #[napi]
    pub fn delete_chunks(&self, ids: Vec<u32>) -> u32 {
        let mut idx = self.inner.write().unwrap();
        idx.delete(&ids) as u32
    }

    /// Delete chunks matching a JSON filter.
    #[napi]
    pub fn delete_where(&self, filter: Value) -> Result<u32> {
        let f: Filter = serde_json::from_value(filter)
            .map_err(|e| Error::new(Status::InvalidArg, format!("invalid filter: {e}")))?;
        let mut idx = self.inner.write().unwrap();
        Ok(idx.delete_where(&f) as u32)
    }

    /// Fetch one chunk (text + metadata).
    #[napi]
    pub fn get_chunk(&self, id: u32) -> Result<Value> {
        let idx = self.inner.read().unwrap();
        if id as usize >= idx.len() {
            return Err(Error::new(Status::InvalidArg, "chunk id out of range"));
        }
        Ok(idx.doc_info(id))
    }

    /// Save the index to a single portable `.leann` file.
    #[napi]
    pub fn save(&self, path: String) -> Result<()> {
        let idx = self.inner.read().unwrap();
        idx.save(&PathBuf::from(path)).map_err(to_napi_err)
    }

    /// Index statistics.
    #[napi]
    pub fn stats(&self) -> StatsJs {
        let idx = self.inner.read().unwrap();
        let s = idx.stats();
        let saved = if s.conventional_vector_bytes > 0 {
            100.0 * (1.0 - (s.codes_bytes + s.graph_bytes + s.codebook_bytes) as f64
                / s.conventional_vector_bytes as f64)
        } else {
            0.0
        };
        StatsJs {
            chunks: s.chunks as u32,
            alive: s.alive as u32,
            deleted: s.deleted as u32,
            dim: s.dim as u32,
            metric: format!("{:?}", s.metric).to_lowercase(),
            pq_m: s.pq_m as u32,
            pq_compression: s.pq_compression as f64,
            text_bytes: s.text_bytes as f64,
            codes_bytes: s.codes_bytes as f64,
            graph_edges: s.graph_edges as u32,
            avg_degree: s.avg_degree as f64,
            conventional_vector_bytes: s.conventional_vector_bytes as f64,
            storage_saved_percent: saved,
            bm25_vocab: s.bm25_vocab as u32,
            embedder: s.embedder,
            file_bytes: s.file_bytes.map(|f| f as f64),
        }
    }

    /// Attach or replace the IBM Granite Guardian judge at runtime.
    #[napi]
    pub fn set_guardian(&mut self, opts: GuardianOptions) {
        self.guardian = Self::build_guardian(&Some(opts));
    }

    /// Sentence-level grounding audit of any answer against context strings
    /// (IBM Granite Guardian method; local pre-filters + judge calls).
    #[napi]
    pub async fn verify_answer(
        &self,
        answer: String,
        context: Vec<String>,
    ) -> Result<Vec<VerifiedSentenceJs>> {
        let guardian = self
            .guardian
            .as_ref()
            .map(|g| g.clone_guardian())
            .ok_or_else(|| Error::new(Status::GenericFailure, "no guardian configured: pass guardian options to the constructor or call setGuardian"))?;
        tokio::task::spawn_blocking(move || -> Result<Vec<VerifiedSentenceJs>> {
            let attrs = guardian
                .audit_sentences(&answer, &context)
                .map_err(to_napi_err)?;
            Ok(attrs
                .iter()
                .map(|a| VerifiedSentenceJs {
                    text: a.text.clone(),
                    grounded: a.keep,
                    confidence: a.judge_confidence.unwrap_or(1.0) as f64,
                    supporting: a.supporting.iter().map(|&i| i as u32).collect(),
                })
                .collect())
        })
        .await
        .map_err(to_napi_err)?
    }

    fn search_opts(&self, args: &Option<SearchArgs>) -> Result<SearchOptions> {
        let mut o = SearchOptions::default();
        if let Some(a) = args {
            if let Some(k) = a.k {
                o.k = k as usize;
            }
            if let Some(ef) = a.ef {
                o.ef = Some(ef as usize);
            }
            if let Some(al) = a.alpha {
                o.alpha = Some(al as f32);
            }
            if let Some(mb) = a.min_batch {
                o.min_batch = Some(mb as usize);
            }
            if let Some(mb) = a.max_batch {
                o.max_batch = Some(mb as usize);
            }
            if let Some(r) = a.relax {
                o.relax = Some(r as f32);
            }
            if let Some(e) = a.exact {
                o.exact = e;
            }
            if let Some(wt) = a.with_text {
                o.with_text = wt;
            }
            if let Some(wm) = a.with_metadata {
                o.with_metadata = wm;
            }
            o.filter = parse_filter(a.filter.clone())?;
        }
        Ok(o)
    }
}

/// Standalone IBM Granite Guardian judge (leann-guard): works with any
/// OpenAI-compatible endpoint, including served granite-guardian models.
#[napi]
pub struct Guardian {
    inner: CoreGuardian,
}

#[napi]
impl Guardian {
    #[napi(constructor)]
    pub fn new(opts: Option<GuardianOptions>) -> Self {
        let inner = match opts {
            Some(o) => {
                let judge = OpenAiJudge::new(
                    o.base_url.unwrap_or_else(|| "http://localhost:11434/v1".into()),
                    o.api_key.unwrap_or_default(),
                    o.model
                        .unwrap_or_else(|| "ibm-granite/granite-guardian-3.3-8b".into()),
                )
                .with_think(o.think.unwrap_or(false));
                CoreGuardian::with_judge(judge)
            }
            None => CoreGuardian::new(),
        };
        Self { inner }
    }

    fn verdict_js(v: &leann_guard::Verdict) -> GuardianVerdictJs {
        GuardianVerdictJs {
            risk_id: v.risk_id.as_str().to_string(),
            triggered: v.triggered,
            confidence: v.confidence as f64,
            trace: v.trace.clone(),
            explanation: v.explanation.clone(),
        }
    }

    /// Is `response` ungrounded w.r.t. `context`? (triggered = hallucinated)
    #[napi]
    pub async fn check_groundedness(
        &self,
        response: String,
        context: Vec<String>,
    ) -> Result<GuardianVerdictJs> {
        let g = self.inner.clone_guardian();
        tokio::task::spawn_blocking(move || {
            g.check_groundedness(&response, &context)
                .map_err(to_napi_err)
                .map(|v| Self::verdict_js(&v))
        })
        .await
        .map_err(to_napi_err)?
    }

    /// Is the retrieved context irrelevant to `query`?
    #[napi]
    pub async fn check_context_relevance(
        &self,
        query: String,
        context: Vec<String>,
    ) -> Result<GuardianVerdictJs> {
        let g = self.inner.clone_guardian();
        tokio::task::spawn_blocking(move || {
            g.check_context_relevance(&query, &context)
                .map_err(to_napi_err)
                .map(|v| Self::verdict_js(&v))
        })
        .await
        .map_err(to_napi_err)?
    }

    /// Does `response` fail to address `query`?
    #[napi]
    pub async fn check_answer_relevance(
        &self,
        query: String,
        response: String,
    ) -> Result<GuardianVerdictJs> {
        let g = self.inner.clone_guardian();
        tokio::task::spawn_blocking(move || {
            g.check_answer_relevance(&query, &response)
                .map_err(to_napi_err)
                .map(|v| Self::verdict_js(&v))
        })
        .await
        .map_err(to_napi_err)?
    }

    /// Harm screen (bias / profanity / violence / sexual / unethical).
    #[napi]
    pub async fn check_harm(&self, text: String) -> Result<GuardianVerdictJs> {
        let g = self.inner.clone_guardian();
        tokio::task::spawn_blocking(move || {
            g.check_harm(&text)
                .map_err(to_napi_err)
                .map(|v| Self::verdict_js(&v))
        })
        .await
        .map_err(to_napi_err)?
    }

    /// Jailbreak / prompt-injection detection for prompts.
    #[napi]
    pub async fn check_jailbreak(&self, prompt: String) -> Result<GuardianVerdictJs> {
        let g = self.inner.clone_guardian();
        tokio::task::spawn_blocking(move || {
            g.check_jailbreak(&prompt)
                .map_err(to_napi_err)
                .map(|v| Self::verdict_js(&v))
        })
        .await
        .map_err(to_napi_err)?
    }

    /// Bring-your-own-criteria check.
    #[napi]
    pub async fn check_custom(
        &self,
        text: String,
        criteria: String,
        context: Option<Vec<String>>,
    ) -> Result<GuardianVerdictJs> {
        let g = self.inner.clone_guardian();
        tokio::task::spawn_blocking(move || {
            g.check_custom(&text, &criteria, &context.unwrap_or_default())
                .map_err(to_napi_err)
                .map(|v| Self::verdict_js(&v))
        })
        .await
        .map_err(to_napi_err)?
    }
}

/// Collections helper: per-user / per-collection index management on disk.
#[napi]
pub struct Collections {
    root: PathBuf,
}

#[napi]
impl Collections {
    /// Create a collections manager rooted at a directory.
    #[napi(constructor)]
    pub fn new(root: String) -> Result<Self> {
        let root = PathBuf::from(root);
        std::fs::create_dir_all(&root).map_err(to_napi_err)?;
        Ok(Self { root })
    }

    fn path_for(&self, user: &str, name: &str) -> PathBuf {
        let safe = |s: &str| s.replace(['/', '\\', ':'], "_");
        self.root.join(safe(user)).join(format!("{}.leann", safe(name)))
    }

    /// Open (or create) a user's collection.
    #[napi]
    pub fn open(&self, user: String, name: String, opts: LeannOptions) -> Result<Leann> {
        let path = self.path_for(&user, &name);
        if path.exists() {
            Leann::open(path.to_string_lossy().to_string(), opts)
        } else {
            Ok(Leann::new(opts)?)
        }
    }

    /// List collection names for a user.
    #[napi]
    pub fn list(&self, user: String) -> Vec<String> {
        let dir = self.root.join(user.replace(['/', '\\', ':'], "_"));
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                if let Some(name) = e.path().file_stem().and_then(|s| s.to_str()) {
                    if e.path().extension().map(|x| x == "leann").unwrap_or(false) {
                        out.push(name.to_string());
                    }
                }
            }
        }
        out.sort();
        out
    }

    /// Persist a collection for a user.
    #[napi]
    pub fn save(&self, user: String, name: String, leann: &Leann) -> Result<()> {
        let path = self.path_for(&user, &name);
        leann.save(path.to_string_lossy().to_string())
    }

    /// Delete a collection file.
    #[napi]
    pub fn delete(&self, user: String, name: String) -> bool {
        let path = self.path_for(&user, &name);
        std::fs::remove_file(path).is_ok()
    }
}

/// Package version string.
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Quick self-test: build a tiny local index and search it. Returns true on
/// success — handy for smoke-testing the native module.
#[napi]
pub async fn self_test() -> Result<bool> {
    let opts = LeannOptions {
        embedder: EmbedderOptions {
            kind: Some("local-hash".into()),
            dim: Some(256),
            base_url: None,
            api_key: None,
            model: None,
            batch_size: None,
            concurrency: None,
            max_retries: None,
            send_dimensions_param: None,
        },
        llm: None,
        transcriber: None,
        metric: Some("cosine".into()),
        chunker: None,
        cache_capacity: None,
        extra_metadata: None,
        guardian: None,
    };
    let leann = Leann::new(opts)?;
    let texts: Vec<String> = (0..40)
        .map(|i| {
            format!(
                "chunk marker{i:03} about {}",
                ["rust vectors", "pdf parsing", "quantum qubits", "market reports"][i % 4]
            )
        })
        .collect();
    leann.add_texts(texts, None).await?;
    let res = leann
        .search(
            "quantum qubits".into(),
            Some(SearchArgs {
                k: Some(3),
                ef: None,
                alpha: None,
                min_batch: None,
                max_batch: None,
                relax: None,
                exact: None,
                with_text: None,
                with_metadata: None,
                filter: None,
            }),
        )
        .await?;
    Ok(!res.hits.is_empty())
}

// ---------------------------------------------------------------------------
// offline (deterministic) grounding checks — no judge server required
// ---------------------------------------------------------------------------

/// Result of the offline (deterministic) grounding check from `leann-guard`.
#[napi(object)]
pub struct LocalGroundingJs {
    /// Fraction of answer numbers found in the context (0-1).
    pub number_coverage: f64,
    /// Fraction of answer content terms covered by the context (0-1).
    pub term_coverage: f64,
    /// Overall pass at the crate's default thresholds.
    pub grounded: bool,
    /// Numbers present in the answer but missing from context.
    pub missing_numbers: Vec<String>,
    /// Salient terms present in the answer but missing from context.
    pub missing_terms: Vec<String>,
}

/// One sentence's offline grounding verdict.
#[napi(object)]
pub struct LocalSentenceJs {
    pub index: u32,
    pub text: String,
    pub grounded: bool,
    pub term_coverage: f64,
    pub number_coverage: f64,
    pub missing_terms: Vec<String>,
}

impl From<leann_guard::LocalGrounding> for LocalGroundingJs {
    fn from(g: leann_guard::LocalGrounding) -> Self {
        LocalGroundingJs {
            number_coverage: g.number_coverage as f64,
            term_coverage: g.term_coverage as f64,
            grounded: g.grounded,
            missing_numbers: g.missing_numbers,
            missing_terms: g.missing_terms,
        }
    }
}

/// Offline (no LLM, no server) grounding check of `answer` against `contexts`.
///
/// Deterministic term/number-coverage analysis from the IBM Granite Guardian
/// crate's local pre-filter layer. Fast enough to run on every generation;
/// use it to gate or flag answers before (or instead of) a judge call.
#[napi]
pub fn local_grounding(answer: String, contexts: Vec<String>) -> LocalGroundingJs {
    let ctx = contexts.join("\n");
    leann_guard::local_groundedness(&answer, &ctx).into()
}

/// Offline sentence-level audit: splits `answer` into sentences and checks
/// each against `contexts` with deterministic grounding. Returns per-sentence
/// verdicts; sentences with `grounded == false` should be flagged or removed.
#[napi]
pub fn local_verify(answer: String, contexts: Vec<String>) -> Vec<LocalSentenceJs> {
    let ctx = contexts.join("\n");
    leann_guard::split_sentences(&answer)
        .into_iter()
        .enumerate()
        .map(|(index, text)| {
            let g = leann_guard::local_groundedness(&text, &ctx);
            LocalSentenceJs {
                index: index as u32,
                grounded: g.grounded,
                term_coverage: g.term_coverage as f64,
                number_coverage: g.number_coverage as f64,
                missing_terms: g.missing_terms,
                text,
            }
        })
        .collect()
}

/// Offline grounded rewrite: the answer with ungrounded sentences removed.
/// The cheapest hallucination-elimination pass (deterministic, zero-cost).
#[napi]
pub fn local_rewrite(answer: String, contexts: Vec<String>) -> String {
    let ctx = contexts.join("\n");
    leann_guard::split_sentences(&answer)
        .into_iter()
        .filter(|s| leann_guard::local_groundedness(s, &ctx).grounded)
        .collect::<Vec<_>>()
        .join(" ")
}
