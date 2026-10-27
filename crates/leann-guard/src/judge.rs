//! Judge backends: the trait plus an OpenAI-compatible client that talks to
//! served Granite Guardian models (vLLM / Ollama / watsonx) or any capable
//! judge LLM, and a deterministic mock for tests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use crate::block::{build_guardian_block, ChatRole, GuardianRequest, JudgeMessage};
use crate::parse::{parse_guardian_response, to_verdict};
use crate::criteria::Criteria;
use crate::{GuardError, Result, Verdict};

/// A backend that can judge guardian blocks.
pub trait Judge: Send + Sync {
    /// Model name (diagnostics).
    fn model(&self) -> &str;
    /// Run a guardian check: messages in, raw judge text out.
    fn judge(&self, messages: &[JudgeMessage], temperature: f32) -> Result<String>;

    /// Run a full guardian request end-to-end (block build -> judge -> parse).
    fn check(
        &self,
        req: &GuardianRequest,
        criteria: &Criteria,
        temperature: f32,
    ) -> Result<Verdict> {
        let messages = build_guardian_block(req);
        let raw = self.judge(&messages, temperature)?;
        let parsed = parse_guardian_response(&raw);
        Ok(to_verdict(&parsed, criteria.risk_id, 1.0))
    }
}

/// OpenAI-compatible judge (`POST {base}/chat/completions`).
///
/// Works with:
/// - **The actual Granite Guardian models** served via vLLM, Ollama, TGI
///   (`ibm-granite/granite-guardian-3.3-8b`, `granite-guardian-3.2-5b`, ...)
///   — the server applies the guardian chat template to our messages.
/// - Any OpenAI-compatible chat model used as an LLM judge
///   (`gpt-4o-mini`, `claude` via proxy, local `llama3.1`...).
#[derive(Debug, Clone)]
pub struct OpenAiJudge {
    agent: ureq::Agent,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    /// Reasoning (think) mode — see [`crate::block::JUDGE_THINK`].
    pub think: bool,
    pub max_retries: u32,
    calls: Arc<AtomicUsize>,
}

impl OpenAiJudge {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(120))
                .build(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: model.into(),
            think: false,
            max_retries: 3,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Enable/disable think mode (reasoning traces before the score).
    pub fn with_think(mut self, think: bool) -> Self {
        self.think = think;
        self
    }

    /// Number of judge calls made so far (telemetry).
    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl Judge for OpenAiJudge {
    fn model(&self) -> &str {
        &self.model
    }

    fn judge(&self, messages: &[JudgeMessage], temperature: f32) -> Result<String> {
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| json!({"role": m.role.as_str(), "content": m.content}))
            .collect();
        // guardian models are scored with temperature 0 (paper protocol)
        let body = json!({
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": 1024,
        });
        let url = format!("{}/chat/completions", self.base_url);
        let mut attempt = 0u32;
        loop {
            let mut req = self
                .agent
                .post(&url)
                .set("Content-Type", "application/json");
            if !self.api_key.is_empty() {
                req = req.set("Authorization", &format!("Bearer {}", self.api_key));
            }
            match req.send_json(serde_json::to_value(&body)?) {
                Ok(resp) => {
                    self.calls.fetch_add(1, Ordering::Relaxed);
                    let parsed: Value = serde_json::from_reader(resp.into_reader())
                        .map_err(|e| GuardError::Judge(format!("bad judge response: {e}")))?;
                    let content = parsed["choices"][0]["message"]["content"]
                        .as_str()
                        .ok_or_else(|| {
                            GuardError::Judge(format!(
                                "judge returned no content: {}",
                                parsed.to_string().chars().take(300).collect::<String>()
                            ))
                        })?
                        .to_string();
                    return Ok(content);
                }
                Err(ureq::Error::Status(code, resp)) => {
                    let msg = resp.into_string().unwrap_or_default();
                    if (code == 429 || code >= 500) && attempt < self.max_retries {
                        attempt += 1;
                        std::thread::sleep(Duration::from_millis(250u64 << attempt.min(4)));
                        continue;
                    }
                    return Err(GuardError::Judge(format!(
                        "judge API {code}: {}",
                        msg.chars().take(300).collect::<String>()
                    )));
                }
                Err(e) => {
                    if attempt < self.max_retries {
                        attempt += 1;
                        std::thread::sleep(Duration::from_millis(250u64 << attempt.min(4)));
                        continue;
                    }
                    return Err(GuardError::Judge(format!("judge request failed: {e}")));
                }
            }
        }
    }
}

/// Deterministic mock judge for tests and offline development.
///
/// The mock implements a *cheap lexical grounding heuristic*: it triggers
/// groundedness when the text contains numbers or capitalized entities that
/// appear nowhere in the documents. For all other risks it returns
/// `triggered = false` unless configured otherwise.
pub struct MockJudge {
    /// Force these risks to trigger (test scenarios).
    pub force_trigger: Mutex<Vec<crate::criteria::RiskId>>,
    /// Confidence to attach to verdicts.
    pub confidence: f32,
    pub think: bool,
    pub call_log: Mutex<Vec<Vec<JudgeMessage>>>,
}

impl Default for MockJudge {
    fn default() -> Self {
        Self {
            force_trigger: Mutex::new(Vec::new()),
            confidence: 0.9,
            think: false,
            call_log: Mutex::new(Vec::new()),
        }
    }
}

impl MockJudge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn forcing(risk: crate::criteria::RiskId) -> Self {
        Self {
            force_trigger: Mutex::new(vec![risk]),
            confidence: 0.95,
            think: false,
            call_log: Mutex::new(Vec::new()),
        }
    }

    /// Inspect the messages the mock received (assert prompt structure).
    pub fn requests(&self) -> Vec<Vec<JudgeMessage>> {
        self.call_log.lock().unwrap().clone()
    }
}

impl Judge for MockJudge {
    fn model(&self) -> &str {
        "mock-judge"
    }

    fn judge(&self, messages: &[JudgeMessage], _temperature: f32) -> Result<String> {
        self.call_log.lock().unwrap().push(messages.to_vec());

        // which risk are we judging? infer from the criteria text
        let block = messages
            .iter()
            .rev()
            .find(|m| m.role == ChatRole::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();
        let is_groundedness = block.contains("ungrounded or unfaithful");

        // extract the text under review (assistant message before the block)
        let text_under_review = messages
            .iter()
            .rev()
            .find(|m| m.role == ChatRole::Assistant)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        let mut triggered = false;
        if !self.force_trigger.lock().unwrap().is_empty() {
            // honor forced risks only when the criteria matches one of them
            if is_groundedness {
                triggered = self
                    .force_trigger
                    .lock()
                    .unwrap()
                    .contains(&crate::criteria::RiskId::Groundedness);
            }
        } else if is_groundedness {
            // lexical heuristic: numbers in the answer must exist in context
            let sys = messages
                .iter()
                .find(|m| m.role == ChatRole::System)
                .map(|m| m.content.clone())
                .unwrap_or_default();
            triggered = !crate::local::number_consistency(&text_under_review, &sys);
        }

        let score = if triggered { "yes" } else { "no" };
        Ok(format!(
            "{}<score>{score}</score> Confidence: {:.0}%",
            if self.think { "IMD quick reasoning IMD_END" } else { "" },
            self.confidence * 100.0
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::criteria::{criteria_for, RiskId};

    fn request<'a>(text: &'a str, docs: &'a [String], criteria: &'a str) -> GuardianRequest<'a> {
        GuardianRequest {
            text,
            documents: docs,
            query: None,
            tools: &[],
            criteria,
            think: false,
        }
    }

    #[test]
    fn mock_judge_grounds_numbers() {
        let judge = MockJudge::new();
        let docs = vec!["The film premiered on July 16, 1964.".to_string()];
        // consistent date -> not triggered
        let v = judge
            .check(&request("It premiered on July 16, 1964.", &docs, &criteria_for(RiskId::Groundedness).definition), &criteria_for(RiskId::Groundedness), 0.0)
            .unwrap();
        assert!(!v.triggered);
        // invented date -> triggered
        let v2 = judge
            .check(&request("It premiered on July 16, 1999.", &docs, &criteria_for(RiskId::Groundedness).definition), &criteria_for(RiskId::Groundedness), 0.0)
            .unwrap();
        assert!(v2.triggered);
        assert_eq!(v2.confidence, 0.9);
    }

    #[test]
    fn mock_judge_prompt_structure() {
        let judge = MockJudge::new();
        let docs = vec!["doc text".to_string()];
        let _ = judge
            .check(&request("answer", &docs, &criteria_for(RiskId::Groundedness).definition), &criteria_for(RiskId::Groundedness), 0.0)
            .unwrap();
        let reqs = judge.requests();
        assert_eq!(reqs.len(), 1);
        let m = &reqs[0];
        assert_eq!(m[0].role, ChatRole::System);
        assert!(m[0].content.contains("doc text"));
        assert_eq!(m[1].role, ChatRole::Assistant);
        assert_eq!(m[2].role, ChatRole::User);
        assert!(m[2].content.starts_with("<guardian>"));
    }

    #[test]
    fn forced_trigger() {
        let judge = MockJudge::forcing(RiskId::Groundedness);
        let v = judge
            .check(&request("anything", &[], &criteria_for(RiskId::Groundedness).definition), &criteria_for(RiskId::Groundedness), 0.0)
            .unwrap();
        assert!(v.triggered);
        assert!(v.confidence > 0.9);
    }
}
