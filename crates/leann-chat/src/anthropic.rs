//! Anthropic Messages API client.

use std::time::Duration;

use serde_json::{json, Value};

use crate::provider::{ChatRequest, ChatMessage, LlmProvider, Role};
use crate::{ChatError, Result};

#[derive(Debug, Clone)]
pub struct AnthropicLlm {
    agent: ureq::Agent,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub version: String,
}

impl AnthropicLlm {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(300))
                .build(),
            base_url: "https://api.anthropic.com".into(),
            api_key: api_key.into(),
            model: model.into(),
            version: "2023-06-01".into(),
        }
    }

    pub fn with_base_url(mut self, base: impl Into<String>) -> Self {
        self.base_url = base.into().trim_end_matches('/').to_string();
        self
    }
}

impl LlmProvider for AnthropicLlm {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn chat(&self, req: &ChatRequest) -> Result<String> {
        // Anthropic: system prompt is a top-level field, not a message
        let system: Vec<&ChatMessage> = req
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .collect();
        let convo: Vec<Value> = req
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| json!({"role": m.role.as_str(), "content": m.content}))
            .collect();
        let mut body = json!({
            "model": self.model,
            "max_tokens": req.max_tokens.unwrap_or(2048),
            "messages": convo,
        });
        if !system.is_empty() {
            body["system"] = json!(system
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"));
        }
        if let Some(t) = req.temperature {
            body["temperature"] = json!(t);
        }

        let url = format!("{}/v1/messages", self.base_url);
        let resp = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json")
            .set("x-api-key", &self.api_key)
            .set("anthropic-version", &self.version)
            .send_json(body)
            .map_err(|e| ChatError::Api(format!("anthropic request failed: {e}")))?;
        let parsed: Value = serde_json::from_reader(resp.into_reader())
            .map_err(|e| ChatError::Api(format!("bad anthropic response: {e}")))?;
        let text = parsed["content"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter(|b| b["type"] == "text")
                    .filter_map(|b| b["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();
        if text.is_empty() {
            return Err(ChatError::Api(format!(
                "no content in anthropic response: {}",
                parsed.to_string().chars().take(300).collect::<String>()
            )));
        }
        Ok(text)
    }
}
