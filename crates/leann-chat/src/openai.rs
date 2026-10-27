//! OpenAI-compatible chat completions client, with vision (image input)
//! support for multimodal RAG.

use std::path::Path;
use std::time::Duration;

use serde_json::{json, Value};

use crate::provider::{ChatRequest, LlmProvider};
use crate::{ChatError, Result};

#[derive(Debug, Clone)]
pub struct OpenAiLlm {
    agent: ureq::Agent,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl OpenAiLlm {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(300))
                .build(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    fn post_chat(&self, body: Value) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut req = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json");
        if !self.api_key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.api_key));
        }
        let resp = req
            .send_json(body)
            .map_err(|e| ChatError::Api(format!("chat request failed: {e}")))?;
        let parsed: Value = serde_json::from_reader(resp.into_reader())
            .map_err(|e| ChatError::Api(format!("bad chat response: {e}")))?;
        let content = parsed["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| ChatError::Api(format!("no content in response: {}", truncate(&parsed.to_string(), 300))))?
            .to_string();
        Ok(content)
    }
}

impl LlmProvider for OpenAiLlm {
    fn name(&self) -> &str {
        "openai-compatible"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn chat(&self, req: &ChatRequest) -> Result<String> {
        let messages: Vec<Value> = req
            .messages
            .iter()
            .map(|m| json!({"role": m.role.as_str(), "content": m.content}))
            .collect();
        let mut body = json!({
            "model": self.model,
            "messages": messages,
        });
        if let Some(t) = req.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(mt) = req.max_tokens {
            body["max_tokens"] = json!(mt);
        }
        self.post_chat(body)
    }
}

/// Describe an image with a vision-capable model (e.g. gpt-4o-mini).
pub fn describe_image(
    base_url: &str,
    api_key: &str,
    model: &str,
    image_path: &Path,
    prompt: &str,
) -> Result<String> {
    let bytes = std::fs::read(image_path)
        .map_err(|e| ChatError::Api(format!("read image failed: {e}")))?;
    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);
    let mime = match image_path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        _ => "image/jpeg",
    };
    let client = OpenAiLlm::new(base_url, api_key, model);
    let body = json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{b64}")}}
            ]
        }],
        "max_tokens": 300
    });
    client.post_chat(body)
}

/// Adapter implementing `leann_ingest::FrameDescriber` via an
/// OpenAI-compatible vision endpoint.
pub struct OpenAiVisionDescriber {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub prompt: String,
}

impl Default for OpenAiVisionDescriber {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            api_key: String::new(),
            model: "gpt-4o-mini".into(),
            prompt: "Describe this video frame concisely: the scene, visible text, and any notable objects or actions.".into(),
        }
    }
}

impl leann_ingest::parsers::media::FrameDescriber for OpenAiVisionDescriber {
    fn describe(&self, image_path: &Path, _timestamp_sec: f64) -> leann_ingest::Result<String> {
        describe_image(&self.base_url, &self.api_key, &self.model, image_path, &self.prompt)
            .map_err(|e| leann_ingest::LeannIngestError::Media(e.to_string()))
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}...", &s[..n.min(s.len())])
    }
}
