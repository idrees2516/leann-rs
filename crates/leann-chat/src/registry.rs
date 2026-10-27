//! LLM registry: providers from serializable config.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::anthropic::AnthropicLlm;
use crate::openai::OpenAiLlm;
use crate::provider::LlmProvider;
use crate::{ChatError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum LlmKind {
    #[default]
    OpenAi,
    Anthropic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub kind: LlmKind,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            kind: LlmKind::OpenAi,
            base_url: None,
            api_key: None,
            model: None,
            temperature: None,
            max_tokens: None,
        }
    }
}

pub fn create_llm(cfg: &LlmConfig) -> Result<Arc<dyn LlmProvider>> {
    match cfg.kind {
        LlmKind::OpenAi => {
            let base_url = cfg
                .base_url
                .clone()
                .or_else(|| env_first(&["LEANN_LLM_BASE_URL", "OPENAI_BASE_URL"]))
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
            let api_key = cfg
                .api_key
                .clone()
                .or_else(|| env_first(&["LEANN_LLM_API_KEY", "OPENAI_API_KEY"]))
                .unwrap_or_default();
            let model = cfg
                .model
                .clone()
                .or_else(|| std::env::var("LEANN_LLM_MODEL").ok())
                .unwrap_or_else(|| "gpt-4o-mini".to_string());
            Ok(Arc::new(OpenAiLlm::new(base_url, api_key, model)))
        }
        LlmKind::Anthropic => {
            let api_key = cfg
                .api_key
                .clone()
                .or_else(|| env_first(&["ANTHROPIC_API_KEY"]))
                .ok_or_else(|| ChatError::Invalid("anthropic requires api_key or ANTHROPIC_API_KEY".into()))?;
            let model = cfg
                .model
                .clone()
                .unwrap_or_else(|| "claude-sonnet-4-20250514".to_string());
            Ok(Arc::new(AnthropicLlm::new(api_key, model)))
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
