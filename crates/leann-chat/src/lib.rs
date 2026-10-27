//! # leann-chat
//!
//! LLM providers and the RAG layer for LEANN:
//!
//! - [`LlmProvider`] trait with two batteries-included implementations:
//!   [`OpenAiLlm`] (any OpenAI-compatible `/chat/completions` endpoint —
//!   OpenAI, Ollama, vLLM, LM Studio, DeepSeek, Groq, SiliconFlow...) and
//!   [`AnthropicLlm`] (Messages API).
//! - Vision support (`describe_image`) for multimodal RAG over video frames.
//! - [`RagSession`]: hybrid retrieval → context assembly with source
//!   citations → chat with rolling conversation memory.

pub mod anthropic;
pub mod openai;
pub mod provider;
pub mod rag;
pub mod registry;

pub use anthropic::AnthropicLlm;
pub use openai::OpenAiLlm;
pub use provider::{ChatMessage, ChatRequest, LlmProvider, Role};
pub use rag::{answer_with_context, answer_with_context_guarded, RagAnswer, RagConfig, RagSession, SourceRef, GuardSettings};
pub use registry::{create_llm, LlmConfig, LlmKind};

pub mod error {
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum ChatError {
        #[error("I/O error: {0}")]
        Io(#[from] std::io::Error),
        #[error("serialization error: {0}")]
        Serde(#[from] serde_json::Error),
        #[error("LLM API error: {0}")]
        Api(String),
        #[error("invalid argument: {0}")]
        Invalid(String),
    }
    pub type Result<T> = std::result::Result<T, ChatError>;
}

pub use error::{ChatError, Result};
