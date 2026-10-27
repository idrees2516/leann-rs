//! # leann-guard
//!
//! **Hallucination elimination for RAG, implementing IBM's Granite Guardian**
//! (Padhi et al., *"Granite Guardian: A Suite of Safeguards for Risk
//! Detection in LLM Prompts and Responses"*, arXiv:2412.07724, IBM Research,
//! Dec 2024 — https://github.com/ibm-granite/granite-guardian).
//!
//! The paper's core artifact is a **guardian block**: a judging protocol in
//! which a verifier model receives (a) the text under review, (b) the
//! evidence (retrieved documents / tool schemas / conversation), and (c) a
//! *risk criterion*, then answers a strict binary question inside
//! `<score>yes</score>` / `<score>no</score>` tags, optionally preceded by an
//! `IMD` reasoning trace (think mode) and a verbalized confidence.
//!
//! This crate re-implements the full protocol **and builds an elimination
//! pipeline on top of it**:
//!
//! | Layer | What it does | Where |
//! |-------|--------------|-------|
//! | [`criteria`] | The paper's complete risk catalog: **RAG hallucination risks** (context relevance, groundedness, answer relevance), **harm risks** (bias, profanity, violence, sexual content, unethical behavior), **jailbreak**, **function-call hallucination**, and BYOC (bring your own criteria) | any app |
//! | [`block`] | Guardian-block construction: think/no-think modes, documents, tools, scoring schema | any app |
//! | [`parse`] | Response parsing: `<score>` tags, `IMD` traces, verbalized confidence | any app |
//! | [`judge`] | [`Judge`] trait + [`OpenAiJudge`] (works with the actual granite-guardian model served via vLLM/Ollama, or any OpenAI-compatible LLM as judge) + [`MockJudge`] for tests | any app |
//! | [`checks`] | Typed one-shot checks: groundedness, context relevance, answer relevance, harm, jailbreak, custom | any app |
//! | [`local`] | Deterministic pre-filters (no LLM): term-coverage grounding, number consistency, citation validation | any app |
//! | [`sentence`] | Sentence-level attribution: per-sentence grounding verdicts + **grounded rewrite** (strip/flag ungrounded sentences = literal elimination) | any app |
//! | [`pipeline`] | [`RagGuard`] composite gate + policies (Abstain / Annotate / Repair / LogOnly) + [`GuardedChat`] generate→guard→**repair** loop | RAG apps |
//!
//! The crate is deliberately **standalone** (no leann-* dependencies): embed
//! it next to any LLM application. Higher layers (`leann-chat`,
//! `leann-node`) provide thin adapters.
//!
//! ## Example
//!
//! ```no_run
//! use leann_guard::{OpenAiJudge, Guardian, RagGuard, GuardPolicy};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Point at a served granite-guardian model (or any capable judge LLM).
//! let judge = OpenAiJudge::new(
//!     "http://localhost:11434/v1", "", // ollama: no key needed
//!     "ibm-granite/granite-guardian-3.3-8b",
//! );
//! let guardian = Guardian::with_judge(judge);
//!
//! // One-shot groundedness check (paper's headline RAG risk).
//! let verdict = guardian.check_groundedness(
//!     "The film Eat premiered on July 16, 1964.",
//!     &["Eat (1964) ... first shown by Jonas Mekas on July 16, 1964 ...".to_string()],
//! )?;
//! assert!(!verdict.triggered); // not ungrounded
//!
//! // Full gate for a RAG answer: context relevance -> groundedness ->
//! // answer relevance -> harm, with abstention policy.
//! let guard = RagGuard::new(guardian).with_policy(GuardPolicy::Abstain);
//! let query = "When was the film Eat first shown?";
//! let answer = "The film Eat was first shown on July 16, 1964.";
//! let contexts = vec!["Eat (1964) ... first shown on July 16, 1964 ...".to_string()];
//! let decision = guard.check_rag_answer(query, answer, &contexts)?;
//! if decision.safe() { /* use answer */ } else { /* use decision.answer (refusal) */ }
//! # Ok(())
//! # }
//! ```

pub mod block;
pub mod checks;
pub mod criteria;
pub mod judge;
pub mod local;
pub mod parse;
pub mod pipeline;
pub mod sentence;

pub use block::{build_guardian_block, ChatRole, JudgeMessage};
pub use checks::Guardian;
pub use criteria::{Criteria, RiskId};
pub use judge::{Judge, MockJudge, OpenAiJudge};
pub use local::{local_groundedness, number_consistency, LocalGrounding};
pub use parse::{parse_guardian_response, ParsedVerdict};
pub use pipeline::{GuardDecision, GuardPolicy, GuardReport, GuardedChat, RagGuard, RepairConfig};
pub use sentence::{
    grounded_rewrite, sentence_attribution, sentence_attribution_offline, split_sentences,
    SentenceAttribution,
};

/// Common error type.
#[derive(thiserror::Error, Debug)]
pub enum GuardError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("judge/backend error: {0}")]
    Judge(String),
    #[error("invalid argument: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, GuardError>;

/// A single guardian verdict for one risk check.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Verdict {
    /// Which risk was checked.
    pub risk_id: RiskId,
    /// True = the risk IS present (paper semantics: "yes" = risk detected).
    pub triggered: bool,
    /// 0.0-1.0 verbalized/derived confidence in the judgment.
    pub confidence: f32,
    /// Reasoning trace (think mode), when the judge provides one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<String>,
    /// Short human-readable explanation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
}

impl Verdict {
    pub fn triggered(risk: RiskId, confidence: f32) -> Self {
        Self {
            risk_id: risk,
            triggered: true,
            confidence,
            trace: None,
            explanation: None,
        }
    }
    pub fn clear(risk: RiskId, confidence: f32) -> Self {
        Self {
            risk_id: risk,
            triggered: false,
            confidence,
            trace: None,
            explanation: None,
        }
    }
    /// The paper's "safe" semantics: risk not triggered, or triggered with
    /// confidence below `min_confidence`.
    pub fn is_safe(&self, min_confidence: f32) -> bool {
        !self.triggered || self.confidence < min_confidence
    }
}
