//! # leann-server — LEANN as a real-world REST service
//!
//! Wraps the full LEANN stack (index, embedder, ingest, chat, guard, visual
//! memory) in a production-ready HTTP API:
//!
//! * **Multi-collection** — every collection is a directory under the data
//!   dir holding its own `index.leann` (+ optional QR-video memory).
//! * **Security** — optional bearer/X-API-Key auth, per-client token-bucket
//!   rate limiting, strict collection-name validation (no path traversal),
//!   configurable body limit, path-scoped file ingestion, and CORS allow-list.
//! * **Endpoints** — search, hybrid search, RAG chat, hallucination guard,
//!   add texts/files, delete-by-filter, export/import (JSON/CSV/Markdown),
//!   visual memory build + search, stats.
//! * **Zero blocking on the runtime** — every index/LLM/embedder call runs
//!   inside `spawn_blocking` (embedders and LLM clients are sync/ureq).
//! * **Web playground** — a single-file UI at `GET /` for instant manual
//!   exploration (no build step, no npm).
//!
//! ```no_run
//! # fn main() {
//! let cfg = leann_server::ServerConfig {
//!     data_dir: "./leann-data".into(),
//!     ..Default::default()
//! };
//! // serve on 0.0.0.0:8080 (set LEANN_API_KEY to require auth)
//! leann_server::serve("0.0.0.0:8080", cfg);
//! # }
//! ```

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{HeaderValue, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Value};

use leann_chat::LlmConfig;
use leann_core::filter::parse_filter_checked;
use leann_core::index::{HybridOptions, IndexConfig, LeannIndex, SearchOptions};
use leann_core::Filter;
use leann_embedder::{create_embedder, EmbedderConfig, EmbedderKind};
use leann_guard::{GuardPolicy, Guardian, RagGuard};
use leann_visual::{VideoFormat, VisualOptions, VisualRetriever};

mod playground;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Server-wide configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Root directory holding one sub-directory per collection.
    pub data_dir: PathBuf,
    /// Require `Authorization: Bearer <key>` or `X-API-Key: <key>`.
    pub api_key: Option<String>,
    /// Requests per client per minute (0 disables limiting).
    pub rate_limit_per_min: u32,
    /// Maximum request body size.
    pub body_limit_bytes: usize,
    /// CORS `Access-Control-Allow-Origin` values.
    pub allow_origins: Vec<String>,
    /// OpenAI-compatible base URL for RAG chat (None disables chat).
    pub llm_base_url: Option<String>,
    /// API key for the LLM (defaults to OPENAI_API_KEY env).
    pub llm_api_key: Option<String>,
    /// Default chat model.
    pub llm_model: String,
    /// When set, file ingestion is restricted to paths under this root.
    pub ingest_root: Option<PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./leann-data"),
            api_key: None,
            rate_limit_per_min: 240,
            body_limit_bytes: 64 * 1024 * 1024,
            allow_origins: vec!["*".into()],
            llm_base_url: std::env::var("LEANN_LLM_BASE_URL").ok().or(Some("https://api.openai.com/v1".into())),
            llm_api_key: std::env::var("OPENAI_API_KEY").ok(),
            llm_model: std::env::var("LEANN_LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into()),
            ingest_root: std::env::var("LEANN_INGEST_ROOT").ok().map(PathBuf::from),
        }
    }
}

/// A loaded collection: in-memory index + its on-disk location.
pub struct CollectionHandle {
    pub name: String,
    pub dir: PathBuf,
    pub index: RwLock<LeannIndex>,
}

/// Token-bucket rate limiter keyed by client identity.
struct RateLimiter {
    per_min: u32,
    map: Mutex<HashMap<String, (Instant, f64)>>,
}

impl RateLimiter {
    fn new(per_min: u32) -> Self {
        Self { per_min, map: Mutex::new(HashMap::new()) }
    }

    fn allow(&self, key: &str) -> bool {
        if self.per_min == 0 {
            return true;
        }
        let mut map = self.map.lock();
        let now = Instant::now();
        let rate = self.per_min as f64 / 60.0;
        let entry = map.entry(key.to_string()).or_insert((now, self.per_min as f64));
        let (last, tokens) = *entry;
        let refill = tokens + now.duration_since(last).as_secs_f64() * rate;
        let tokens = (refill + 1e-9).min(self.per_min as f64);
        if tokens >= 1.0 {
            *entry = (now, tokens - 1.0);
            true
        } else {
            *entry = (now, tokens);
            false
        }
    }
}

/// Shared server state.
pub struct AppState {
    pub cfg: ServerConfig,
    pub collections: RwLock<HashMap<String, Arc<CollectionHandle>>>,
    pub embedder: Arc<dyn leann_core::cache::EmbeddingProvider>,
    pub rate: RateLimiter,
    pub started: Instant,
    pub requests_served: std::sync::atomic::AtomicU64,
}

pub type SharedState = Arc<AppState>;

impl AppState {
    pub fn new(cfg: ServerConfig) -> Result<Self, String> {
        let embedder = embedder_from_env();
        std::fs::create_dir_all(&cfg.data_dir).map_err(|e| format!("cannot create data dir: {e}"))?;
        let state = Self {
            collections: RwLock::new(HashMap::new()),
            embedder,
            rate: RateLimiter::new(cfg.rate_limit_per_min),
            cfg,
            started: Instant::now(),
            requests_served: std::sync::atomic::AtomicU64::new(0),
        };
        state.reload_collections()?;
        Ok(state)
    }

    /// Scan the data dir and lazily open every existing collection.
    fn reload_collections(&self) -> Result<(), String> {
        let mut loaded = 0;
        for entry in std::fs::read_dir(&self.cfg.data_dir).map_err(|e| e.to_string())?.flatten() {
            let dir = entry.path();
            let index_path = dir.join("index.leann");
            if !index_path.is_file() {
                continue;
            }
            let name = match entry.file_name().to_string_lossy().into_owned().parse::<CollectionName>() {
                Ok(n) => n.0,
                Err(_) => {
                    eprintln!("leann-server: skipping collection with invalid name: {}", dir.display());
                    continue;
                }
            };
            match LeannIndex::open(&index_path, self.embedder.clone()) {
                Ok(idx) => {
                    self.collections.write().insert(name.clone(), Arc::new(CollectionHandle {
                        name,
                        dir,
                        index: RwLock::new(idx),
                    }));
                    loaded += 1;
                }
                Err(e) => eprintln!("leann-server: cannot open {}: {e}", index_path.display()),
            }
        }
        if loaded > 0 {
            eprintln!("leann-server: loaded {loaded} collection(s) from {}", self.cfg.data_dir.display());
        }
        Ok(())
    }

    fn collection(&self, name: &str) -> Result<Arc<CollectionHandle>, ApiError> {
        self.collections
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("collection '{name}' not found")))
    }
}

/// Pick the embedder: `LEANN_EMBEDDER=local` forces the zero-dependency
/// local-hash embedder; otherwise OpenAI (needs OPENAI_API_KEY); when no key
/// is present we fall back to local-hash so the server always boots.
fn embedder_from_env() -> Arc<dyn leann_core::cache::EmbeddingProvider> {
    let cfg = if std::env::var("LEANN_EMBEDDER").map(|v| v == "local" || v == "local-hash").unwrap_or(false) {
        EmbedderConfig { kind: EmbedderKind::LocalHash, dim: 384, ..Default::default() }
    } else if std::env::var("OPENAI_API_KEY").is_ok() {
        EmbedderConfig::default()
    } else {
        eprintln!("leann-server: no OPENAI_API_KEY — using offline local-hash embedder");
        EmbedderConfig { kind: EmbedderKind::LocalHash, dim: 384, ..Default::default() }
    };
    create_embedder(&cfg).expect("embedder construction cannot fail")
}

/// Validated collection name (also used for directory names).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionName(String);

impl std::str::FromStr for CollectionName {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let ok = !s.is_empty()
            && s.len() <= 64
            && s.chars().next().unwrap().is_ascii_alphanumeric()
            && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        if ok {
            Ok(Self(s.to_string()))
        } else {
            Err(format!("invalid collection name '{s}': 1-64 chars, [a-zA-Z0-9_-], must start alnum"))
        }
    }
}

/// Uniform JSON error payload.
pub struct ApiError(pub StatusCode, pub String);

impl ApiError {
    fn bad(msg: impl Into<String>) -> Self {
        Self(StatusCode::BAD_REQUEST, msg.into())
    }
}

impl From<String> for ApiError {
    fn from(e: String) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

fn ok_json(v: Value) -> Response {
    Json(v).into_response()
}

// ---------------------------------------------------------------- middleware

async fn auth_mw(State(state): State<SharedState>, req: Request<axum::body::Body>, next: Next) -> Response {
    let key = state.cfg.api_key.clone();
    if let Some(expected) = key {
        let provided = req
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(str::to_string)
            .or_else(|| {
                req.headers()
                    .get("x-api-key")
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string)
            });
        match provided {
            Some(p) if constant_time_eq(p.as_bytes(), expected.as_bytes()) => {}
            _ => {
                return ApiError(StatusCode::UNAUTHORIZED, "missing or invalid API key".into())
                    .into_response()
            }
        }
    }
    state.requests_served.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    next.run(req).await
}

/// Constant-time byte-slice comparison (avoids leaking key length/prefix timing).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

async fn rate_mw(State(state): State<SharedState>, req: Request<axum::body::Body>, next: Next) -> Response {
    let key = req
        .extensions()
        .get::<axum::extract::ConnectInfo<SocketAddr>>()
        .map(|c| c.0.ip().to_string())
        .unwrap_or_else(|| "local".into());
    if state.rate.allow(&key) {
        next.run(req).await
    } else {
        ApiError(StatusCode::TOO_MANY_REQUESTS, "rate limit exceeded".into()).into_response()
    }
}

// ---------------------------------------------------------------- handlers

async fn health(State(state): State<SharedState>) -> Response {
    ok_json(json!({
        "status": "ok",
        "version": VERSION,
        "uptime_secs": state.started.elapsed().as_secs(),
        "collections": state.collections.read().len(),
        "auth": if state.cfg.api_key.is_some() { "bearer" } else { "disabled" },
    }))
}

async fn list_collections(State(state): State<SharedState>) -> Response {
    let cols = state.collections.read();
    let mut items: Vec<Value> = cols
        .values()
        .map(|c| {
            let idx = c.index.read();
            let s = idx.stats();
            json!({
                "name": c.name,
                "chunks": s.chunks,
                "alive": s.alive,
                "deleted": s.deleted,
                "file_bytes": std::fs::metadata(c.dir.join("index.leann")).map(|m| m.len()).ok(),
            })
        })
        .collect();
    items.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    ok_json(json!({ "collections": items }))
}

async fn create_collection(
    State(state): State<SharedState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    let name = body
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad("field 'name' (string) is required"))?
        .parse::<CollectionName>()
        .map_err(ApiError::bad)?;
    {
        let cols = state.collections.read();
        if cols.contains_key(&name.0) {
            return Err(ApiError(StatusCode::CONFLICT, format!("collection '{}' already exists", name.0)));
        }
    }
    let dir = state.cfg.data_dir.join(&name.0);
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create dir: {e}"))?;
    let idx = LeannIndex::empty(IndexConfig::default(), state.embedder.clone());
    idx.save(&dir.join("index.leann")).map_err(|e| e.to_string())?;
    state.collections.write().insert(
        name.0.clone(),
        Arc::new(CollectionHandle { name: name.0.clone(), dir, index: RwLock::new(idx) }),
    );
    Ok(ok_json(json!({ "created": name.0 })))
}

async fn delete_collection(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> Result<Response, ApiError> {
    let handle = state.collection(&name)?;
    state.collections.write().remove(&name);
    // Only ever delete inside the data dir (defense in depth).
    let canon_root = state
        .cfg
        .data_dir
        .canonicalize()
        .map_err(|e| format!("bad data dir: {e}"))?;
    let canon = handle.dir.canonicalize().map_err(|e| format!("bad collection dir: {e}"))?;
    if canon.starts_with(&canon_root) {
        std::fs::remove_dir_all(&canon).map_err(|e| format!("delete failed: {e}"))?;
    }
    Ok(ok_json(json!({ "deleted": name })))
}

async fn collection_stats(State(state): State<SharedState>, Path(name): Path<String>) -> Result<Response, ApiError> {
    let handle = state.collection(&name)?;
    let idx = handle.index.read();
    Ok(ok_json(idx.stats_json()))
}

#[derive(serde::Deserialize)]
struct TextsBody {
    texts: Vec<String>,
    #[serde(default)]
    metadatas: Option<Vec<Value>>,
    #[serde(default)]
    dedup: Option<bool>,
}

async fn add_texts(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    body: Result<Json<TextsBody>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    if body.texts.is_empty() {
        return Err(ApiError::bad("'texts' must be a non-empty array"));
    }
    if body.texts.len() > 100_000 {
        return Err(ApiError::bad("at most 100000 texts per request"));
    }
    let handle = state.collection(&name)?;
    let dir = handle.dir.clone();
    let texts = body.texts;
    let metas = body.metadatas;
    let dedup = body.dedup.unwrap_or(false);
    let (added, skipped) = tokio::task::spawn_blocking(move || -> Result<(usize, usize), String> {
        let metas: Option<Vec<Value>> = metas.map(|m| {
            let mut m = m;
            while m.len() < texts.len() {
                m.push(Value::Null);
            }
            m
        });
        let mut idx = handle.index.write();
        if dedup {
            let (_, skipped) = idx
                .add_with_dedup(&texts, metas, leann_core::dedup::DedupMode::Near)
                .map_err(|e| e.to_string())?;
            idx.save(&dir.join("index.leann")).map_err(|e| e.to_string())?;
            Ok((texts.len() - skipped, skipped))
        } else {
            idx.add(&texts, metas).map_err(|e| e.to_string())?;
            idx.save(&dir.join("index.leann")).map_err(|e| e.to_string())?;
            Ok((texts.len(), 0))
        }
    })
    .await
    .map_err(|e| format!("task panicked: {e}"))?
    .map_err(ApiError::from)?;
    Ok(ok_json(json!({
        "added": added,
        "skipped_duplicates": skipped,
        "total": state.collection(&name)?.index.read().len(),
    })))
}

#[derive(serde::Deserialize)]
struct FilesBody {
    paths: Vec<String>,
    #[serde(default)]
    chunk_tokens: Option<usize>,
}

async fn add_files(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    body: Result<Json<FilesBody>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    if body.paths.is_empty() {
        return Err(ApiError::bad("'paths' must be a non-empty array"));
    }
    if body.paths.len() > 10_000 {
        return Err(ApiError::bad("at most 10000 files per request"));
    }
    // Path validation: must exist, be files, and stay inside ingest_root when set.
    let root = state.cfg.ingest_root.clone();
    let mut validated = Vec::with_capacity(body.paths.len());
    for p in &body.paths {
        let path = PathBuf::from(p);
        if path.is_absolute() || p.contains("..") {
            if let Some(root) = &root {
                let canon_p = path.canonicalize().map_err(|e| ApiError::bad(format!("{p}: {e}")))?;
                let canon_root = root.canonicalize().map_err(|e| ApiError::bad(format!("ingest root: {e}")))?;
                if !canon_p.starts_with(&canon_root) {
                    return Err(ApiError(StatusCode::FORBIDDEN, format!("{p}: outside LEANN_INGEST_ROOT")));
                }
            } else {
                return Err(ApiError(StatusCode::FORBIDDEN, format!(
                    "{p}: absolute paths / '..' require LEANN_INGEST_ROOT to be configured"
                )));
            }
        }
        if !path.is_file() {
            return Err(ApiError::bad(format!("{p}: not a file")));
        }
        validated.push(path);
    }
    let handle = state.collection(&name)?;
    let dir = handle.dir.clone();
    let chunk_tokens = body.chunk_tokens;
    let n = tokio::task::spawn_blocking(move || -> Result<usize, String> {
        let opts = leann_ingest::IngestOptions {
            chunker: leann_ingest::chunker::ChunkerConfig {
                chunk_tokens: chunk_tokens.unwrap_or(512),
                ..Default::default()
            },
            ..Default::default()
        };
        let chunks = leann_ingest::ingest_files(&validated, &opts, None).map_err(|e| e.to_string())?;
        if chunks.is_empty() {
            return Err("no text extracted from the given files".into());
        }
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let metas: Vec<Value> = chunks.iter().map(|c| c.metadata.clone()).collect();
        let mut idx = handle.index.write();
        idx.add(&texts, Some(metas)).map_err(|e| e.to_string())?;
        idx.save(&dir.join("index.leann")).map_err(|e| e.to_string())?;
        Ok(texts.len())
    })
    .await
    .map_err(|e| format!("task panicked: {e}"))?
    .map_err(ApiError::from)?;
    Ok(ok_json(json!({ "added_chunks": n, "files": body.paths.len() })))
}

fn parse_filter_value(v: Option<&Value>) -> Result<Option<Filter>, ApiError> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(f) => {
            let s = serde_json::to_string(f).map_err(|e| ApiError::bad(format!("bad filter: {e}")))?;
            parse_filter_checked(&s).map(Some).map_err(|e| ApiError::bad(format!("bad filter: {e}")))
        }
    }
}

#[derive(serde::Deserialize)]
struct SearchBody {
    query: String,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    filter: Option<Value>,
    #[serde(default)]
    expand: bool,
}

fn default_k() -> usize {
    10
}

async fn search(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    body: Result<Json<SearchBody>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    let filter = parse_filter_value(body.filter.as_ref())?;
    let handle = state.collection(&name)?;
    let res = tokio::task::spawn_blocking(move || -> Result<Value, String> {
        let idx = handle.index.read();
        let opts = SearchOptions { k: body.k, filter, ..Default::default() };
        let res = idx.search(&body.query, &opts).map_err(|e| e.to_string())?;
        Ok(serde_json::to_value(&res).unwrap_or(Value::Null))
    })
    .await
    .map_err(|e| format!("task panicked: {e}"))?
    .map_err(ApiError::from)?;
    Ok(ok_json(res))
}

#[derive(serde::Deserialize)]
struct HybridBody {
    query: String,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    filter: Option<Value>,
    #[serde(default)]
    expand: bool,
}

async fn hybrid_search(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    body: Result<Json<HybridBody>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    let filter = parse_filter_value(body.filter.as_ref())?;
    let handle = state.collection(&name)?;
    let res = tokio::task::spawn_blocking(move || -> Result<Value, String> {
        let idx = handle.index.read();
        let mut opts = HybridOptions {
            base: SearchOptions { k: body.k, filter, ..Default::default() },
            ..Default::default()
        };
        if body.expand {
            opts.expand = Some(leann_core::context::ContextExpansionOptions { before: 1, after: 1 });
        }
        let res = idx.hybrid_search(&body.query, &opts).map_err(|e| e.to_string())?;
        Ok(serde_json::to_value(&res).unwrap_or(Value::Null))
    })
    .await
    .map_err(|e| format!("task panicked: {e}"))?
    .map_err(ApiError::from)?;
    Ok(ok_json(res))
}

#[derive(serde::Deserialize)]
struct ChatBody {
    query: String,
    #[serde(default = "default_chat_k")]
    k: usize,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    guard: bool,
}

fn default_chat_k() -> usize {
    6
}

async fn chat(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    body: Result<Json<ChatBody>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    let base_url = state.cfg.llm_base_url.clone();
    let api_key = state
        .cfg
        .llm_api_key
        .clone()
        .filter(|k| !k.is_empty())
        .ok_or_else(|| {
            ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "no LLM configured: set OPENAI_API_KEY (or LEANN_LLM_BASE_URL for local models)".into(),
            )
        })?;
    let model = body.model.unwrap_or_else(|| state.cfg.llm_model.clone());
    let handle = state.collection(&name)?;
    let out = tokio::task::spawn_blocking(move || -> Result<Value, String> {
        let llm = leann_chat::create_llm(&LlmConfig {
            kind: leann_chat::LlmKind::OpenAi,
            base_url: base_url.clone(),
            api_key: Some(api_key),
            model: Some(model),
            ..Default::default()
        })
        .map_err(|e| e.to_string())?;
        let (hits, idx_len) = {
            let idx = handle.index.read();
            let res = idx
                .search(&body.query, &SearchOptions { k: body.k, ..Default::default() })
                .map_err(|e| e.to_string())?;
            (res.hits, idx.len())
        };
        let context: Vec<(u32, String, Value)> = hits
            .iter()
            .map(|h| (h.id, h.text.clone().unwrap_or_default(), h.metadata.clone().unwrap_or(Value::Null)))
            .collect();
        let cfg = leann_chat::rag::RagConfig::default();
        let guardian = if body.guard { Some(Guardian::new()) } else { None };
        let ans = leann_chat::rag::answer_with_context_guarded(
            llm.as_ref(),
            &body.query,
            &context,
            &[],
            &cfg,
            guardian.as_ref(),
        )
        .map_err(|e| e.to_string())?;
        Ok(json!({
            "answer": ans.answer,
            "sources": context.iter().map(|(id, t, m)| json!({"id": id, "text": t, "metadata": m})).collect::<Vec<_>>(),
            "index_size": idx_len,
            "guard": ans.guard,
        }))
    })
    .await
    .map_err(|e| format!("task panicked: {e}"))?
    .map_err(ApiError::from)?;
    Ok(ok_json(out))
}

#[derive(serde::Deserialize)]
struct GuardBody {
    query: String,
    answer: String,
    #[serde(default = "default_chat_k")]
    k: usize,
}

/// Deterministic offline audit: retrieve context for `query`, then check the
/// provided `answer` for groundedness (IBM Granite Guardian local mode).
async fn guard(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    body: Result<Json<GuardBody>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    let handle = state.collection(&name)?;
    let out = tokio::task::spawn_blocking(move || -> Result<Value, String> {
        let docs = {
            let idx = handle.index.read();
            let res = idx
                .search(&body.query, &SearchOptions { k: body.k, ..Default::default() })
                .map_err(|e| e.to_string())?;
            res.hits.iter().map(|h| h.text.clone().unwrap_or_default()).collect::<Vec<_>>()
        };
        let grounding = leann_guard::local_groundedness(&body.answer, &docs.join("\n\n"));
        let decision = RagGuard::new(Guardian::new())
            .with_policy(GuardPolicy::Annotate)
            .check_rag_answer(&body.query, &body.answer, &docs)
            .map_err(|e| e.to_string())?;
        Ok(json!({
            "groundedness": {
                "grounded": grounding.grounded,
                "number_coverage": grounding.number_coverage,
                "term_coverage": grounding.term_coverage,
                "missing_numbers": grounding.missing_numbers,
                "missing_terms": grounding.missing_terms,
            },
            "decision": decision,
            "context_chunks": docs.len(),
        }))
    })
    .await
    .map_err(|e| format!("task panicked: {e}"))?
    .map_err(ApiError::from)?;
    Ok(ok_json(out))
}

#[derive(serde::Deserialize)]
struct DeleteBody {
    filter: Value,
}

async fn delete_where(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    body: Result<Json<DeleteBody>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    let filter = parse_filter_value(Some(&body.filter))?
        .ok_or_else(|| ApiError::bad("'filter' must be a filter object"))?;
    let handle = state.collection(&name)?;
    let dir = handle.dir.clone();
    let n = tokio::task::spawn_blocking(move || -> Result<usize, String> {
        let mut idx = handle.index.write();
        let n = idx.delete_where(&filter);
        idx.save(&dir.join("index.leann")).map_err(|e| e.to_string())?;
        Ok(n)
    })
    .await
    .map_err(|e| format!("task panicked: {e}"))?
    .map_err(ApiError::from)?;
    Ok(ok_json(json!({ "deleted": n })))
}

async fn export_collection(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Query(q): Query<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let format = q.get("format").map(String::as_str).unwrap_or("json");
    let fmt = leann_core::export::ExportFormat::parse(format)
        .ok_or_else(|| ApiError::bad("format must be one of json|csv|md"))?;
    let handle = state.collection(&name)?;
    let body = tokio::task::spawn_blocking(move || -> Result<String, String> {
        let idx = handle.index.read();
        Ok(leann_core::export::export_string(&idx, fmt))
    })
    .await
    .map_err(|e| format!("task panicked: {e}"))?
    .map_err(ApiError::from)?;
    let (ctype, ext) = match fmt {
        leann_core::export::ExportFormat::Json => ("application/json", "json"),
        leann_core::export::ExportFormat::Csv => ("text/csv", "csv"),
        leann_core::export::ExportFormat::Markdown => ("text/markdown", "md"),
    };
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_str(ctype).unwrap());
    headers.insert(
        "content-disposition",
        HeaderValue::from_str(&format!("attachment; filename=\"{name}.{ext}\"")).unwrap(),
    );
    Ok((StatusCode::OK, headers, body).into_response())
}

#[derive(serde::Deserialize)]
struct ImportBody {
    data: Value,
}

async fn import_collection(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    body: Result<Json<ImportBody>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    let data_str = serde_json::to_string(&body.data).map_err(|e| ApiError::bad(e.to_string()))?;
    let imported = leann_core::export::import_json(&data_str).map_err(ApiError::bad)?;
    if imported.is_empty() {
        return Err(ApiError::bad("nothing to import"));
    }
    let handle = state.collection(&name)?;
    let dir = handle.dir.clone();
    let n = tokio::task::spawn_blocking(move || -> Result<usize, String> {
        let mut idx = handle.index.write();
        let texts: Vec<String> = imported.iter().map(|c| c.text.clone()).collect();
        let metas: Vec<Value> = imported.iter().map(|c| c.metadata.clone()).collect();
        idx.add(&texts, Some(metas)).map_err(|e| e.to_string())?;
        idx.save(&dir.join("index.leann")).map_err(|e| e.to_string())?;
        Ok(texts.len())
    })
    .await
    .map_err(|e| format!("task panicked: {e}"))?
    .map_err(ApiError::from)?;
    Ok(ok_json(json!({ "imported": n })))
}

#[derive(serde::Deserialize)]
struct VisualBody {
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    scale: Option<u32>,
}

/// Build a QR-code video memory for the collection (memvid-style).
async fn visual_build(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    body: Result<Json<VisualBody>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    let format = body
        .format
        .as_deref()
        .and_then(VideoFormat::parse)
        .unwrap_or(VideoFormat::Auto);
    let scale = body.scale.unwrap_or(8);
    let handle = state.collection(&name)?;
    let video_path = handle.dir.join("memory.mp4");
    let out = tokio::task::spawn_blocking(move || -> Result<Value, String> {
        let idx = handle.index.read();
        let opts = VisualOptions { scale, format, ..Default::default() };
        let stats = leann_visual::build_visual_from_index(&idx, &video_path, opts)
            .map_err(|e| e.to_string())?;
        Ok(json!({
            "chunks": stats.chunks,
            "container": stats.container,
            "video_bytes": stats.video_bytes,
            "text_bytes": stats.text_bytes,
            "bytes_per_text_byte": stats.bytes_per_text_byte,
            "video": stats.video_path,
            "sidecar": stats.sidecar_path,
        }))
    })
    .await
    .map_err(|e| format!("task panicked: {e}"))?
    .map_err(ApiError::from)?;
    Ok(ok_json(out))
}

#[derive(serde::Deserialize)]
struct VisualSearchBody {
    query: String,
    #[serde(default = "default_k")]
    k: usize,
}

/// Semantic search whose hit text is decoded out of the QR video.
async fn visual_search(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    body: Result<Json<VisualSearchBody>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    let handle = state.collection(&name)?;
    let video_path = handle.dir.join("memory.mp4");
    let out = tokio::task::spawn_blocking(move || -> Result<Value, String> {
        let idx = handle.index.read();
        let retriever = VisualRetriever::open(&video_path).map_err(|e| e.to_string())?;
        let hits = retriever
            .search(&idx, &body.query, body.k)
            .map_err(|e| e.to_string())?;
        Ok(json!({
            "container": retriever.container(),
            "hits": hits,
        }))
    })
    .await
    .map_err(|e| format!("task panicked: {e}"))?
    .map_err(|e| ApiError(StatusCode::NOT_FOUND, format!("visual memory not built: {e}")))?;
    Ok(ok_json(out))
}

async fn playground() -> Response {
    (
        StatusCode::OK,
        [("content-type", HeaderValue::from_static("text/html; charset=utf-8"))],
        playground::HTML,
    )
        .into_response()
}

// ---------------------------------------------------------------- router

/// Build the full application router (useful for tests and embedding).
pub fn build_router(state: SharedState) -> Router {
    let cors = if state.cfg.allow_origins.iter().any(|o| o == "*") {
        tower_http::cors::CorsLayer::permissive()
    } else {
        let origins = state
            .cfg
            .allow_origins
            .iter()
            .filter_map(|o| o.parse::<HeaderValue>().ok())
            .collect::<Vec<_>>();
        tower_http::cors::CorsLayer::new().allow_origin(origins)
    };
    Router::new()
        .route("/", get(playground))
        .route("/health", get(health))
        .route("/stats", get(health))
        .route("/collections", get(list_collections).post(create_collection))
        .route(
            "/collections/{name}",
            get(collection_stats).delete(delete_collection),
        )
        .route("/collections/{name}/stats", get(collection_stats))
        .route("/collections/{name}/texts", post(add_texts))
        .route("/collections/{name}/files", post(add_files))
        .route("/collections/{name}/search", post(search))
        .route("/collections/{name}/hybrid", post(hybrid_search))
        .route("/collections/{name}/chat", post(chat))
        .route("/collections/{name}/guard", post(guard))
        .route("/collections/{name}/delete", post(delete_where))
        .route("/collections/{name}/export", get(export_collection))
        .route("/collections/{name}/import", post(import_collection))
        .route("/collections/{name}/visual", post(visual_build))
        .route("/collections/{name}/visual-search", post(visual_search))
        .layer(middleware::from_fn_with_state(state.clone(), auth_mw))
        .layer(middleware::from_fn_with_state(state.clone(), rate_mw))
        .layer(DefaultBodyLimit::max(state.cfg.body_limit_bytes))
        .layer(cors)
        .with_state(state)
}

/// Bind and serve forever (graceful ctrl-c).
pub async fn serve_async(addr: &str, cfg: ServerConfig) -> Result<(), String> {
    let state = Arc::new(AppState::new(cfg)?);
    let app = build_router(state).into_make_service_with_connect_info::<SocketAddr>();
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| format!("bind {addr}: {e}"))?;
    eprintln!("leann-server v{VERSION} listening on http://{addr}");
    eprintln!("  playground:  http://{addr}/");
    eprintln!("  collections: {}", listener.local_addr().map(|a| a.to_string()).unwrap_or_default());
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("shutting down");
        })
        .await
        .map_err(|e| e.to_string())
}

/// Blocking serve helper for CLIs.
pub fn serve(addr: &str, cfg: ServerConfig) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    if let Err(e) = rt.block_on(serve_async(addr, cfg)) {
        eprintln!("leann-server: {e}");
        std::process::exit(1);
    }
}
