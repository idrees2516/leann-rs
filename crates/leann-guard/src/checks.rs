//! [`Guardian`] — the high-level typed check API over any [`Judge`].

use std::sync::Arc;

use crate::block::GuardianRequest;
use crate::criteria::{criteria_for, custom_criteria, Criteria, RiskId};
use crate::judge::Judge;
use crate::{local, parse, Result, Verdict};

/// A configured guardian: a judge backend + options.
pub struct Guardian {
    judge: Arc<dyn Judge>,
    /// Global think-mode toggle (per-call overrides in requests).
    pub think: bool,
    /// Judge temperature (paper protocol uses 0).
    pub temperature: f32,
    /// Judge every sentence (vs local pre-filtering) in attribution passes.
    pub judge_every_sentence: bool,
}

impl Guardian {
    pub fn with_judge(judge: impl Judge + 'static) -> Self {
        Self {
            judge: Arc::new(judge),
            think: false,
            temperature: 0.0,
            judge_every_sentence: false,
        }
    }

    pub fn new() -> Self {
        // offline default: mock judge (deterministic local checks only)
        Self::with_judge(crate::judge::MockJudge::new())
    }

    pub fn from_arc(judge: Arc<dyn Judge>) -> Self {
        Self {
            judge,
            think: false,
            temperature: 0.0,
            judge_every_sentence: false,
        }
    }

    pub fn with_think(mut self, think: bool) -> Self {
        self.think = think;
        self
    }

    pub fn judge(&self) -> &Arc<dyn Judge> {
        &self.judge
    }

    /// Cheap clone (the judge backend is shared).
    pub fn clone_guardian(&self) -> Guardian {
        Guardian {
            judge: self.judge.clone(),
            think: self.think,
            temperature: self.temperature,
            judge_every_sentence: self.judge_every_sentence,
        }
    }

    fn check(
        &self,
        text: &str,
        documents: &[String],
        query: Option<&str>,
        tools: &[String],
        criteria: &Criteria,
    ) -> Result<Verdict> {
        let req = GuardianRequest {
            text,
            documents,
            query,
            tools,
            criteria: &criteria.definition,
            think: self.think,
        };
        let messages = crate::block::build_guardian_block(&req);
        let raw = self.judge.judge(&messages, self.temperature)?;
        let parsed = parse::parse_guardian_response(&raw);
        Ok(parse::to_verdict(&parsed, criteria.risk_id, 1.0))
    }

    // ---------------- RAG hallucination risks ----------------

    /// **Groundedness** — is `response` ungrounded w.r.t. `documents`?
    /// `triggered = true` means the response hallucinated.
    pub fn check_groundedness(&self, response: &str, documents: &[String]) -> Result<Verdict> {
        self.check(response, documents, None, &[], &criteria_for(RiskId::Groundedness))
    }

    /// **Context relevance** — is the retrieved `documents` set irrelevant
    /// to `query`? `triggered = true` means the context is irrelevant.
    pub fn check_context_relevance(&self, query: &str, documents: &[String]) -> Result<Verdict> {
        // judge the *documents* as the text under review (paper framing)
        let context_text = documents
            .iter()
            .enumerate()
            .map(|(i, d)| format!("[Doc {i}] {}", d.trim()))
            .collect::<Vec<_>>()
            .join("\n\n");
        self.check(
            &context_text,
            &[],
            Some(query),
            &[],
            &criteria_for(RiskId::ContextRelevance),
        )
    }

    /// **Answer relevance** — does `response` fail to address `query`?
    /// `triggered = true` means the response is irrelevant.
    pub fn check_answer_relevance(&self, query: &str, response: &str) -> Result<Verdict> {
        self.check(response, &[], Some(query), &[], &criteria_for(RiskId::AnswerRelevance))
    }

    // ---------------- harm / security risks ----------------

    /// **Harm umbrella** — social bias, profanity, violence, sexual content,
    /// unethical behavior. Returns the first triggering verdict (most
    /// severe) or the last clear one.
    pub fn check_harm(&self, text: &str) -> Result<Verdict> {
        let risks = [
            RiskId::SocialBias,
            RiskId::Profanity,
            RiskId::Violence,
            RiskId::SexualContent,
            RiskId::UnethicalBehavior,
        ];
        let mut last = Verdict::clear(RiskId::SocialBias, 1.0);
        for r in risks {
            let v = self.check(text, &[], None, &[], &criteria_for(r))?;
            if v.triggered {
                return Ok(v);
            }
            last = v;
        }
        Ok(last)
    }

    /// **Jailbreak** — is `prompt` an attempt to bypass safety guidelines?
    pub fn check_jailbreak(&self, prompt: &str) -> Result<Verdict> {
        self.check(prompt, &[], None, &[], &criteria_for(RiskId::Jailbreak))
    }

    // ---------------- agentic risks ----------------

    /// **Function-call hallucination** — does `response` (a tool call)
    /// misuse `tools` (JSON schemas)?
    pub fn check_function_call(&self, response: &str, tools: &[String]) -> Result<Verdict> {
        self.check(response, &[], None, tools, &criteria_for(RiskId::FunctionCallHallucination))
    }

    // ---------------- BYOC ----------------

    /// **Bring your own criteria** — judge `text` against a custom rule.
    pub fn check_custom(
        &self,
        text: &str,
        criteria: &str,
        documents: &[String],
    ) -> Result<Verdict> {
        let c = custom_criteria(criteria, 0.7);
        self.check(text, documents, None, &[], &c)
    }

    // ---------------- cheap local helpers ----------------

    /// Deterministic numeric grounding check (no judge call).
    pub fn local_numbers_ok(&self, answer: &str, context: &str) -> bool {
        local::number_consistency(answer, context)
    }

    /// Invalid `[n]` citations in an answer (no judge call).
    pub fn invalid_citations(&self, answer: &str, doc_count: usize) -> Vec<usize> {
        local::validate_citations(answer, doc_count)
    }

    /// Sentence-level attribution audit (see [`crate::sentence`]).
    pub fn audit_sentences(
        &self,
        answer: &str,
        documents: &[String],
    ) -> Result<Vec<crate::sentence::SentenceAttribution>> {
        crate::sentence::sentence_attribution(
            self.judge.as_ref(),
            answer,
            documents,
            self.judge_every_sentence,
        )
    }

    /// Grounded rewrite: the answer with ungrounded sentences removed.
    pub fn grounded_rewrite(&self, answer: &str, documents: &[String]) -> Result<String> {
        let attrs = self.audit_sentences(answer, documents)?;
        Ok(crate::sentence::grounded_rewrite(&attrs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::MockJudge;

    #[test]
    fn groundedness_via_guardian() {
        let g = Guardian::new();
        let docs = vec!["premiered on July 16, 1964".to_string()];
        let ok = g.check_groundedness("premiered on July 16, 1964", &docs).unwrap();
        assert!(!ok.triggered);
        let bad = g.check_groundedness("premiered on July 16, 1999", &docs).unwrap();
        assert!(bad.triggered);
    }

    #[test]
    fn custom_criteria_path() {
        let g = Guardian::new();
        let v = g
            .check_custom("Answer is longer than 3 words", "The answer must be at most 3 words long.", &[])
            .unwrap();
        assert_eq!(v.risk_id, RiskId::Custom);
        assert!(!v.triggered, "mock judge only implements groundedness heuristics");
    }

    #[test]
    fn citations_and_numbers_helpers() {
        let g = Guardian::new();
        assert!(g.local_numbers_ok("16 1964", "16 1964 present"));
        assert!(!g.local_numbers_ok("1999", "1964"));
        assert!(g.invalid_citations("see [1]", 2).is_empty());
        assert_eq!(g.invalid_citations("see [7]", 2), vec![7]);
    }
}
