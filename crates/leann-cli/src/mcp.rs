//! MCP (Model Context Protocol) server over stdio.
//!
//! Turns LEANN into an agent-native tool server: Claude Desktop, Cursor and
//! any MCP client can point at `leann mcp` and get first-class retrieval,
//! ingestion, RAG chat and hallucination auditing as tools.
//!
//! Protocol: JSON-RPC 2.0, one request per line on stdin, responses on
//! stdout (notifications produce no output). No extra dependencies — the
//! whole server is ~400 lines of serde_json plumbing.
//!
//! Tools exposed:
//! - `leann_search`     — vector or hybrid search (+ filters, context expansion)
//! - `leann_add_files`  — ingest documents (PDF/TXT/MD/HTML/DOCX/EPUB/code/audio/video)
//! - `leann_add_texts`  — ingest raw text chunks
//! - `leann_chat`       — RAG answer with citations (optional IBM Granite Guardian)
//! - `leann_guard`      — offline audit of an answer against context strings
//! - `leann_stats`      — index statistics
//! - `leann_delete_where` — bulk delete by metadata filter
//! - `leann_compact`    — garbage-collect deleted chunks

use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use leann_core::filter::parse_filter_checked;
use leann_core::index::{HybridOptions, LeannIndex, SearchOptions};
use leann_embedder::{create_embedder, EmbedderConfig, EmbedderKind};

const SERVER_NAME: &str = "leann";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
const PROTOCOL_VERSION: &str = "2024-11-05";

/// Tool schema used by `leann mcp`.
fn tools_schema() -> Value {
    json!([
        {
            "name": "leann_search",
            "description": "Semantic search over the index. Hybrid mode fuses vector search with BM25 keyword search (better for exact terms, ids, names). Returns chunk ids, scores, text and metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "natural-language query"},
                    "k": {"type": "integer", "default": 8, "description": "top-k results"},
                    "hybrid": {"type": "boolean", "default": false, "description": "fuse vector + BM25 (RRF)"},
                    "filter": {"type": "string", "description": "JSON metadata filter, e.g. {\"op\":\"eq\",\"field\":\"format\",\"value\":\"pdf\"}"},
                    "expand": {"type": "boolean", "default": false, "description": "include neighboring chunks (reading order)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "leann_add_files",
            "description": "Ingest documents into the index: PDF, TXT, MD, HTML, DOCX, EPUB, JSON, 30+ code formats, audio and video (transcribed). Index is auto-saved.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "file paths to ingest"}
                },
                "required": ["paths"]
            }
        },
        {
            "name": "leann_add_texts",
            "description": "Add raw text chunks (e.g. conversation memory, scraped content). Index is auto-saved.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "texts": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object", "description": "optional metadata merged into every chunk"}
                },
                "required": ["texts"]
            }
        },
        {
            "name": "leann_chat",
            "description": "RAG: retrieve relevant chunks and answer with an LLM, with citations. Requires LLM env vars (OPENAI_API_KEY / OPENAI_BASE_URL) or a configured endpoint.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 6}
                },
                "required": ["query"]
            }
        },
        {
            "name": "leann_guard",
            "description": "Audit an answer for hallucinations against context strings (IBM Granite Guardian method, offline deterministic mode: no LLM needed). Returns per-sentence grounding verdicts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "contexts": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["answer", "contexts"]
            }
        },
        {
            "name": "leann_stats",
            "description": "Index statistics: chunk counts, PQ compression ratio, storage saved vs conventional vector index.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "leann_delete_where",
            "description": "Bulk-delete chunks matching a JSON metadata filter, e.g. {\"op\":\"eq\",\"field\":\"source\",\"value\":\"old.pdf\"}. Auto-saves.",
            "inputSchema": {
                "type": "object",
                "properties": {"filter": {"type": "string"}},
                "required": ["filter"]
            }
        },
        {
            "name": "leann_compact",
            "description": "Garbage-collect deleted chunks (reclaims space, renumbers ids). Auto-saves.",
            "inputSchema": {"type": "object", "properties": {}}
        }
    ])
}

fn text_response(id: Value, text: String) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": false
        }
    })
}

fn error_response(id: Value, code: i64, message: String) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "content": [{"type": "text", "text": format!("error: {message}")}],
            "isError": true
        },
        "_code": code
    })
}

struct McpState {
    index_path: PathBuf,
    index: RwLock<Option<LeannIndex>>,
    provider: Arc<dyn leann_core::cache::EmbeddingProvider>,
}

impl McpState {
    fn with_index<T>(
        &self,
        f: impl FnOnce(&mut LeannIndex) -> std::result::Result<T, String>,
    ) -> std::result::Result<T, String> {
        let mut guard = self.index.write().map_err(|e| e.to_string())?;
        if guard.is_none() {
            let idx = if self.index_path.exists() {
                LeannIndex::open(&self.index_path, self.provider.clone())
                    .map_err(|e| format!("open {}: {e}", self.index_path.display()))?
            } else {
                LeannIndex::empty(Default::default(), self.provider.clone())
            };
            *guard = Some(idx);
        }
        let idx = guard.as_mut().expect("just initialized");
        let out = f(idx)?;
        idx.save(&self.index_path).map_err(|e| format!("save: {e}"))?;
        Ok(out)
    }

    fn read_only<T>(
        &self,
        f: impl FnOnce(&LeannIndex) -> std::result::Result<T, String>,
    ) -> std::result::Result<T, String> {
        let mut guard = self.index.write().map_err(|e| e.to_string())?;
        if guard.is_none() {
            if !self.index_path.exists() {
                return Err(format!(
                    "index {} does not exist yet — call leann_add_files first",
                    self.index_path.display()
                ));
            }
            let idx = LeannIndex::open(&self.index_path, self.provider.clone())
                .map_err(|e| format!("open {}: {e}", self.index_path.display()))?;
            *guard = Some(idx);
        }
        f(guard.as_ref().expect("just initialized"))
    }
}

fn call_tool(state: &McpState, name: &str, args: &Value) -> Value {
    match name {
        "leann_search" => {
            let query = args["query"].as_str().unwrap_or_default().to_string();
            let k = args["k"].as_u64().unwrap_or(8) as usize;
            let hybrid = args["hybrid"].as_bool().unwrap_or(false);
            let expand = args["expand"].as_bool().unwrap_or(false);
            if query.is_empty() {
                return json!({"error": "query is required"});
            }
            state
                .read_only(|idx| {
                    let result = if hybrid {
                        let opts = HybridOptions {
                            base: SearchOptions {
                                k,
                                with_text: true,
                                with_metadata: true,
                                ..Default::default()
                            },
                            expand: if expand {
                                Some(leann_core::context::ContextExpansionOptions {
                                    before: 1,
                                    after: 1,
                                })
                            } else {
                                None
                            },
                            ..Default::default()
                        };
                        let r = idx.hybrid_search(&query, &opts).map_err(|e| e.to_string())?;
                        json!({
                            "took_ms": r.took_ms,
                            "hits": r.hits.iter().map(|h| json!({
                                "id": h.id,
                                "score": h.fused_score,
                                "vector_rank": h.vector_rank,
                                "lexical_rank": h.lexical_rank,
                                "text": h.text,
                                "metadata": h.metadata,
                            })).collect::<Vec<_>>()
                        })
                    } else {
                        let filter = args["filter"]
                            .as_str()
                            .map(parse_filter_checked)
                            .transpose()?;
                        let opts = SearchOptions {
                            k,
                            filter,
                            with_text: true,
                            with_metadata: true,
                            ..Default::default()
                        };
                        let r = idx.search(&query, &opts).map_err(|e| e.to_string())?;
                        json!({
                            "took_ms": r.took_ms,
                            "hits": r.hits.iter().map(|h| json!({
                                "id": h.id,
                                "similarity": h.similarity,
                                "text": h.text,
                                "metadata": h.metadata,
                            })).collect::<Vec<_>>()
                        })
                    };
                    serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
                })
                .unwrap_or_else(|e| format!("error: {e}"))
                .into()
        }
        "leann_add_files" => {
            let paths: Vec<String> = args["paths"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            if paths.is_empty() {
                return json!({"error": "paths is required"}).into();
            }
            state
                .with_index(|idx| {
                    let pb: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
                    let chunks = leann_ingest::ingest_files(&pb, &Default::default(), None)
                        .map_err(|e| format!("ingest: {e}"))?;
                    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
                    let metas: Vec<Value> =
                        chunks.iter().map(|c| c.metadata.clone()).collect();
                    let ids = idx.add(&texts, Some(metas)).map_err(|e| e.to_string())?;
                    Ok(format!("added {} chunks from {} files (ids {:?}..)", ids.len(), pb.len(), ids.first()))
                })
                .unwrap_or_else(|e| format!("error: {e}"))
                .into()
        }
        "leann_add_texts" => {
            let texts: Vec<String> = args["texts"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            if texts.is_empty() {
                return json!({"error": "texts is required"}).into();
            }
            let extra = args["metadata"].clone();
            state
                .with_index(|idx| {
                    let metas: Vec<Value> = texts.iter().map(|_| extra.clone()).collect();
                    let ids = idx.add(&texts, Some(metas)).map_err(|e| e.to_string())?;
                    Ok(format!("added {} chunks", ids.len()))
                })
                .unwrap_or_else(|e| format!("error: {e}"))
                .into()
        }
        "leann_chat" => {
            let query = args["query"].as_str().unwrap_or_default().to_string();
            let k = args["k"].as_u64().unwrap_or(6) as usize;
            if query.is_empty() {
                return json!({"error": "query is required"}).into();
            }
            let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
            let base_url =
                std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into());
            let model = std::env::var("LEANN_LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
            let llm = match leann_chat::create_llm(&leann_chat::LlmConfig {
                kind: leann_chat::LlmKind::OpenAi,
                base_url: Some(base_url),
                api_key: Some(api_key),
                model: Some(model),
                ..Default::default()
            }) {
                Ok(l) => l,
                Err(e) => return format!("error: {e}").into(),
            };
            state
                .read_only(|idx| {
                    let res = idx.search(&query, &SearchOptions { k, with_text: true, ..Default::default() }).map_err(|e| e.to_string())?;
                    let hits: Vec<(u32, String, Value)> = res
                        .hits
                        .iter()
                        .map(|h| (h.id, h.text.clone().unwrap_or_default(), h.metadata.clone().unwrap_or(Value::Null)))
                        .collect();
                    let ans = leann_chat::answer_with_context(llm.as_ref(), &query, &hits, &[], &Default::default())
                        .map_err(|e| format!("llm: {e}"))?;
                    let mut out = format!("{}\n\nsources:", ans.answer);
                    for s in &ans.sources {
                        out.push_str(&format!(
                            "\n  [{}] {}",
                            s.chunk_id,
                            s.text.as_deref().unwrap_or("").chars().take(80).collect::<String>()
                        ));
                    }
                    Ok(out)
                })
                .unwrap_or_else(|e| format!("error: {e}"))
                .into()
        }
        "leann_guard" => {
            let answer = args["answer"].as_str().unwrap_or_default();
            let contexts: Vec<String> = args["contexts"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let sentences = leann_guard::sentence_attribution_offline(answer, &contexts);
            let report = serde_json::to_string_pretty(&json!({
                "answer_length": answer.len(),
                "sentences": sentences.iter().map(|s| json!({
                    "text": s.text,
                    "grounded": s.local.grounded,
                    "term_coverage": s.local.term_coverage,
                    "number_coverage": s.local.number_coverage,
                    "missing_terms": s.local.missing_terms,
                })).collect::<Vec<_>>(),
                "all_grounded": sentences.iter().all(|s| s.local.grounded),
            }))
            .unwrap_or_default();
            Value::String(report)
        }
        "leann_stats" => state
            .read_only(|idx| {
                let s = idx.stats_json();
                serde_json::to_string_pretty(&s).map_err(|e| e.to_string())
            })
            .unwrap_or_else(|e| format!("error: {e}"))
            .into(),
        "leann_delete_where" => {
            let filter = args["filter"].as_str().unwrap_or_default();
            state
                .with_index(|idx| {
                    let f = parse_filter_checked(filter)?;
                    let n = idx.delete_where(&f);
                    Ok(format!("deleted {n} chunks"))
                })
                .unwrap_or_else(|e| format!("error: {e}"))
                .into()
        }
        "leann_compact" => state
            .with_index(|idx| {
                let before = idx.stats().chunks;
                idx.compact().map_err(|e| e.to_string())?;
                Ok(format!("compacted: {} chunks remain", before))
            })
            .unwrap_or_else(|e| format!("error: {e}"))
            .into(),
        _ => format!("error: unknown tool {name}").into(),
    }
}

/// Run the MCP stdio server loop.
pub fn run(index_path: PathBuf) -> i32 {
    let provider: Arc<dyn leann_core::cache::EmbeddingProvider> =
        match std::env::var("LEANN_EMBEDDER").as_deref() {
            Ok("local") | Ok("local-hash") => create_embedder(&EmbedderConfig {
                kind: EmbedderKind::LocalHash,
                ..Default::default()
            })
            .expect("local embedder cannot fail"),
            _ => match create_embedder(&EmbedderConfig::default()) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("leann mcp: embedder error: {e}");
                    eprintln!("hint: set OPENAI_API_KEY, or LEANN_EMBEDDER=local for offline mode");
                    return 2;
                }
            },
        };

    let state = McpState {
        index_path,
        index: RwLock::new(None),
        provider,
    };

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(
                    out,
                    "{}",
                    error_response(Value::Null, -32700, format!("parse error: {e}"))
                );
                let _ = out.flush();
                continue;
            }
        };
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req["method"].as_str().unwrap_or_default().to_string();

        // notifications (no id) get no response
        if id.is_null() {
            continue;
        }

        let resp = match method.as_str() {
            "initialize" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": false}
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION
                    }
                }
            }),
            "tools/list" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {"tools": tools_schema()}
            }),
            "tools/call" => {
                let name = req["params"]["name"].as_str().unwrap_or_default().to_string();
                let args = if req["params"]["arguments"].is_null() { json!({}) } else { req["params"]["arguments"].clone() };
                let result = call_tool(&state, &name, &args);
                if let Some(err) = result.get("error") {
                    error_response(id, 400, err.as_str().unwrap_or("bad request").to_string())
                } else {
                    text_response(id, to_text(&result))
                }
            }
            "ping" => json!({"jsonrpc": "2.0", "id": id, "result": {}}),
            other => error_response(id, -32601, format!("method not found: {other}")),
        };
        let _ = writeln!(out, "{resp}");
        let _ = out.flush();
    }
    0
}

fn to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_is_valid() {
        let tools = tools_schema();
        let arr = tools.as_array().expect("tools must be an array");
        assert!(arr.len() >= 8, "expected at least 8 tools");
        for t in arr {
            assert!(t["name"].is_string());
            assert!(t["description"].is_string());
            assert!(t["inputSchema"]["type"].is_string());
        }
        let names: Vec<&str> = arr.iter().filter_map(|t| t["name"].as_str()).collect();
        assert!(names.contains(&"leann_search"));
        assert!(names.contains(&"leann_guard"));
    }
}
