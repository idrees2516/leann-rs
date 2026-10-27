//! RAG pipeline: hybrid retrieval → cited context → chat with memory.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

use leann_core::context::ContextExpansionOptions;
use leann_core::index::{HybridOptions, LeannIndex, SearchOptions};
use leann_core::filter::Filter;
use leann_guard::{GuardPolicy, Guardian, RagGuard, RepairConfig};

use crate::provider::{ChatMessage, LlmProvider};
use crate::Result;

/// RAG behavior knobs.
#[derive(Debug, Clone)]
pub struct RagConfig {
    pub top_k: usize,
    /// Use hybrid (vector + BM25) retrieval.
    pub use_hybrid: bool,
    /// Pull neighboring chunks around each hit.
    pub context_expansion: Option<ContextExpansionOptions>,
    /// Optional metadata filter for retrieval.
    pub filter: Option<Filter>,
    pub system_prompt: Option<String>,
    /// Hard cap on assembled context characters.
    pub max_context_chars: usize,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// Include source citations [n] in the prompt.
    pub cite_sources: bool,
    /// Optional hallucination guard (IBM Granite Guardian method).
    /// When set, every answer is verified before it is returned.
    pub guard: Option<GuardSettings>,
}

/// Configuration for the IBM Granite Guardian integration.
#[derive(Debug, Clone)]
pub struct GuardSettings {
    pub policy: GuardPolicy,
    pub repair_attempts: usize,
    pub groundedness_threshold: f32,
    pub harm_check: bool,
    pub sentence_level: bool,
}

impl Default for GuardSettings {
    fn default() -> Self {
        Self {
            policy: GuardPolicy::Repair,
            repair_attempts: 1,
            groundedness_threshold: 0.7,
            harm_check: true,
            sentence_level: true,
        }
    }
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            top_k: 6,
            use_hybrid: true,
            context_expansion: None,
            filter: None,
            system_prompt: None,
            max_context_chars: 24_000,
            temperature: None,
            max_tokens: None,
            cite_sources: true,
            guard: None,
        }
    }
}

/// One retrieved source attached to an answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRef {
    pub chunk_id: u32,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub score: Option<f32>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

/// A RAG answer with provenance (and optional guard verification).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagAnswer {
    pub answer: String,
    pub sources: Vec<SourceRef>,
    pub query: String,
    /// Present when a guard was configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard: Option<leann_guard::GuardReport>,
}

pub const DEFAULT_RAG_SYSTEM_PROMPT: &str = "You are a helpful assistant answering questions over a private document collection. \
Answer using the provided context. When the context is relevant, cite it inline like [1], [2]. \
If the context does not contain the answer, say so plainly and answer from general knowledge only if you are confident, \
marking that part as uncited.";

/// One-shot: answer a question over retrieved context.
pub fn answer_with_context(
    provider: &dyn LlmProvider,
    query: &str,
    context: &[(u32, String, Value)], // (chunk_id, text, metadata)
    history: &[ChatMessage],
    cfg: &RagConfig,
) -> Result<RagAnswer> {
    answer_with_context_guarded(provider, query, context, history, cfg, None)
}

/// Like [`answer_with_context`] but with a guardian verifying the answer
/// (IBM Granite Guardian elimination pipeline).
pub fn answer_with_context_guarded(
    provider: &dyn LlmProvider,
    query: &str,
    context: &[(u32, String, Value)],
    history: &[ChatMessage],
    cfg: &RagConfig,
    guardian: Option<&Guardian>,
) -> Result<RagAnswer> {
    let mut ctx = String::new();
    let mut budget = cfg.max_context_chars;
    for (i, (id, text, meta)) in context.iter().enumerate() {
        let header = source_header(*id, meta);
        let piece = format!("[{n}] {header}\n{text}\n\n", n = i + 1, header = header);
        if piece.len() > budget {
            break;
        }
        budget -= piece.len();
        ctx.push_str(&piece);
    }
    if ctx.is_empty() {
        ctx = "(no relevant context found)".to_string();
    }

    let sys = cfg.system_prompt.clone().unwrap_or_else(|| DEFAULT_RAG_SYSTEM_PROMPT.to_string());
    let mut messages = vec![ChatMessage::system(sys)];
    messages.extend(history.iter().cloned());
    let user = if cfg.cite_sources {
        format!("Context:\n{ctx}\n\nQuestion: {query}")
    } else {
        format!("Context:\n{ctx}\n\nQuestion: {query}")
    };
    messages.push(ChatMessage::user(user));

    let req = crate::provider::ChatRequest {
        messages,
        temperature: cfg.temperature,
        max_tokens: cfg.max_tokens,
    };
    let mut answer = provider.chat(&req)?;
    let mut guard_report = None;

    // ---- IBM Granite Guardian elimination pass ----
    if let (Some(g), Some(gs)) = (guardian, cfg.guard.as_ref()) {
        let docs: Vec<String> = context.iter().map(|(_, t, _)| t.clone()).collect();
        let rag_guard = RagGuard::new(g.clone_guardian()).with_config(RepairConfig {
            max_attempts: gs.repair_attempts,
            groundedness_threshold: gs.groundedness_threshold,
            harm_check: gs.harm_check,
            policy: gs.policy,
            sentence_level: gs.sentence_level,
            ..Default::default()
        });
        let decision = rag_guard
            .check_rag_answer(query, &answer, &docs)
            .map_err(|e| crate::ChatError::Api(format!("guard failed: {e}")))?;
        answer = decision.answer;
        guard_report = Some(decision.report);
    }

    Ok(RagAnswer {
        answer,
        sources: context
            .iter()
            .map(|(id, text, meta)| SourceRef {
                chunk_id: *id,
                source: meta.get("source").and_then(|s| s.as_str()).map(String::from),
                score: None,
                text: Some(text.clone()),
                metadata: Some(meta.clone()),
            })
            .collect(),
        query: query.to_string(),
        guard: guard_report,
    })
}

fn source_header(id: u32, meta: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(s) = meta.get("source").and_then(|v| v.as_str()) {
        parts.push(s.to_string());
    }
    if let Some(h) = meta.get("heading").and_then(|v| v.as_str()) {
        parts.push(h.to_string());
    }
    if let Some(p) = meta.get("page").and_then(|v| v.as_u64()) {
        parts.push(format!("p.{p}"));
    }
    if let Some(l) = meta.get("language").and_then(|v| v.as_str()) {
        parts.push(l.to_string());
    }
    if parts.is_empty() {
        format!("(chunk {id})")
    } else {
        format!("(chunk {id}, {})", parts.join(" · "))
    }
}

/// Stateful RAG session over one index: retrieval + rolling chat memory.
pub struct RagSession {
    index: Arc<LeannIndex>,
    provider: Arc<dyn LlmProvider>,
    pub cfg: RagConfig,
    pub history: Vec<ChatMessage>,
    guardian: Option<Guardian>,
}

impl RagSession {
    pub fn new(index: Arc<LeannIndex>, provider: Arc<dyn LlmProvider>, cfg: RagConfig) -> Self {
        Self {
            index,
            provider,
            cfg,
            history: Vec::new(),
            guardian: None,
        }
    }

    /// Attach a guardian (IBM Granite Guardian checks run on every answer).
    pub fn with_guardian(mut self, guardian: Guardian) -> Self {
        self.guardian = Some(guardian);
        self
    }

    /// Ask a question; history is maintained across calls.
    pub fn chat(&mut self, query: &str) -> Result<RagAnswer> {
        let context = self.retrieve(query);
        let answer = answer_with_context_guarded(
            self.provider.as_ref(),
            query,
            &context,
            &self.history,
            &self.cfg,
            self.guardian.as_ref(),
        )?;
        self.history.push(ChatMessage::user(query.to_string()));
        self.history
            .push(ChatMessage::assistant(answer.answer.clone()));
        // keep memory bounded
        if self.history.len() > 20 {
            let drop = self.history.len() - 20;
            self.history.drain(0..drop);
        }
        Ok(answer)
    }

    /// Retrieve context chunks for a query (hybrid + expansion, filtered).
    pub fn retrieve(&self, query: &str) -> Vec<(u32, String, Value)> {
        let k = self.cfg.top_k;
        let expand = self.cfg.context_expansion;
        let mut hits: Vec<(u32, f32)> = Vec::new();

        if self.cfg.use_hybrid {
            let opts = HybridOptions {
                base: SearchOptions {
                    k,
                    filter: self.cfg.filter.clone(),
                    ..Default::default()
                },
                expand,
                ..Default::default()
            };
            if let Ok(res) = self.index.hybrid_search(query, &opts) {
                hits = res
                    .hits
                    .iter()
                    .map(|h| (h.id, h.fused_score))
                    .collect();
            }
        } else if let Ok(res) = self.index.search(
            query,
            &SearchOptions {
                k,
                filter: self.cfg.filter.clone(),
                ..Default::default()
            },
        ) {
            hits = res.hits.iter().map(|h| (h.id, h.similarity)).collect();
        }

        let mut out: Vec<(u32, String, Value)> = hits
            .iter()
            .map(|&(id, _score)| {
                (
                    id,
                    self.index.text(id).to_string(),
                    self.index.metadata(id).clone(),
                )
            })
            .collect();
        // annotate scores
        for ((_, _, meta), (_, score)) in out.iter_mut().zip(hits.iter()) {
            if let Value::Object(m) = meta {
                m.insert("_score".into(), Value::from(*score));
            }
        }
        out
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}
