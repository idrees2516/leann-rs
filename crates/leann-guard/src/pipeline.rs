//! The elimination pipeline: composite RAG gate + guarded generation with
//! repair loops.
//!
//! [`RagGuard::check_rag_answer`] runs the paper's three RAG checks in the
//! order that maximizes elimination per judge call:
//!
//! 1. **Context relevance** — if the retrieved context is irrelevant, no
//!    amount of answer checking helps: abstain immediately.
//! 2. **Groundedness** — sentence-level attribution with local pre-filters;
//!    depending on policy, rewrite (drop ungrounded sentences) or flag.
//! 3. **Answer relevance** — the answer must still address the question
//!    after rewriting.
//! 4. **Harm** — final safety screen.
//!
//! Policies:
//! - **Abstain**: on any triggering risk, return a refusal + the guard
//!   report (safest).
//! - **Repair**: when groundedness fails, *rewrite* the answer to its
//!   grounded subset and re-verify; optionally call back into the
//!   generator LLM with feedback ([`GuardedChat`]).
//! - **Annotate**: pass the answer through with inline risk annotations.
//! - **LogOnly**: ship everything, record the report (telemetry mode).

use serde::{Deserialize, Serialize};

use crate::checks::Guardian;
use crate::criteria::RiskId;
use crate::{Result, Verdict};

/// What to do when a check triggers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GuardPolicy {
    /// Refuse to return risky output (default).
    #[default]
    Abstain,
    /// Drop ungrounded sentences and re-verify, then abstain if still bad.
    Repair,
    /// Return the answer with an attached risk report.
    Annotate,
    /// Return the answer unchanged; report only.
    LogOnly,
}

/// Repair-loop configuration for [`GuardedChat`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairConfig {
    /// Max generator retries with judge feedback.
    pub max_attempts: usize,
    /// Minimum groundedness confidence to accept (0-1).
    pub groundedness_threshold: f32,
    /// Minimum context-relevance confidence to proceed answering.
    pub context_threshold: f32,
    /// Run harm checks on prompt and answer.
    pub harm_check: bool,
    pub policy: GuardPolicy,
    /// Sentence-level audit (vs whole-answer single verdict).
    pub sentence_level: bool,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            max_attempts: 1,
            groundedness_threshold: 0.7,
            context_threshold: 0.7,
            harm_check: true,
            policy: GuardPolicy::Abstain,
            sentence_level: true,
        }
    }
}

/// Full report of a guard pass.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GuardReport {
    pub checks: Vec<Verdict>,
    /// Overall: any risk triggered above threshold?
    pub risk_triggered: bool,
    /// Triggering risk ids.
    pub risks: Vec<RiskId>,
    /// Sentence-level details when sentence audit ran.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sentences: Vec<crate::sentence::SentenceAttribution>,
    /// Which action was taken.
    pub action: GuardAction,
    /// Whether a repair/rewrite was applied.
    pub repaired: bool,
}

impl GuardReport {
    pub fn verdict(&self, risk: RiskId) -> Option<&Verdict> {
        self.checks.iter().find(|v| v.risk_id == risk)
    }
    pub fn groundedness(&self) -> Option<&Verdict> {
        self.verdict(RiskId::Groundedness)
    }
    pub fn context_relevance(&self) -> Option<&Verdict> {
        self.verdict(RiskId::ContextRelevance)
    }
    pub fn answer_relevance(&self) -> Option<&Verdict> {
        self.verdict(RiskId::AnswerRelevance)
    }
}

/// The action a policy decided on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GuardAction {
    /// Answer passed all checks.
    #[default]
    Passed,
    /// Answer was rewritten to its grounded subset.
    Rewritten,
    /// Answer was rejected (abstain).
    Rejected,
    /// Answer returned with annotations.
    Annotated,
    /// Logged only.
    Logged,
}

/// The result of guarding one RAG answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardDecision {
    /// The answer text to use (possibly rewritten / refusal).
    pub answer: String,
    /// Full report.
    pub report: GuardReport,
}

impl GuardDecision {
    /// Whether the answer is safe to surface as-is.
    pub fn safe(&self) -> bool {
        !self.report.risk_triggered
    }
    /// The grounded rewrite when present, else the original answer.
    pub fn alternative(&self) -> &str {
        &self.answer
    }
}

/// Composite RAG guard.
pub struct RagGuard {
    guardian: Guardian,
    cfg: RepairConfig,
}

impl RagGuard {
    pub fn new(guardian: Guardian) -> Self {
        Self {
            guardian,
            cfg: RepairConfig::default(),
        }
    }

    pub fn with_policy(mut self, policy: GuardPolicy) -> Self {
        self.cfg.policy = policy;
        self
    }

    pub fn with_config(mut self, cfg: RepairConfig) -> Self {
        self.cfg = cfg;
        self
    }

    pub fn config(&self) -> &RepairConfig {
        &self.cfg
    }

    pub fn guardian(&self) -> &Guardian {
        &self.guardian
    }

    /// Run the full elimination pipeline over one RAG answer.
    pub fn check_rag_answer(
        &self,
        query: &str,
        answer: &str,
        documents: &[String],
    ) -> Result<GuardDecision> {
        let mut report = GuardReport::default();
        let mut answer = answer.to_string();

        // ---- 1. context relevance (gate before wasting more calls) ----
        if !documents.is_empty() {
            let cr = self.guardian.check_context_relevance(query, documents)?;
            if cr.triggered && cr.confidence >= self.cfg.context_threshold {
                report.checks.push(cr.clone());
                report.risk_triggered = true;
                report.risks.push(RiskId::ContextRelevance);
                report.action = GuardAction::Rejected;
                return Ok(GuardDecision {
                    answer: abstention_text(query, "the retrieved context is not relevant to the question"),
                    report,
                });
            }
            report.checks.push(cr);
        }

        // ---- 2. groundedness (sentence-level elimination) ----
        if !documents.is_empty() {
            if self.cfg.sentence_level {
                let attrs = self
                    .guardian
                    .audit_sentences(&answer, documents)?;
                let ungrounded = attrs.iter().filter(|a| !a.keep).count();
                if ungrounded > 0 {
                    let rewritten = crate::sentence::grounded_rewrite(&attrs);
                    match self.cfg.policy {
                        GuardPolicy::Abstain | GuardPolicy::Repair => {
                            if rewritten.trim().is_empty() {
                                report.risk_triggered = true;
                                report.risks.push(RiskId::Groundedness);
                                report.action = GuardAction::Rejected;
                                report.sentences = attrs.clone();
                                return Ok(GuardDecision {
                                    answer: abstention_text(
                                        query,
                                        "no part of the generated answer is supported by the retrieved context",
                                    ),
                                    report,
                                });
                            }
                            answer = rewritten;
                            report.repaired = true;
                            report.sentences = attrs;
                        }
                        GuardPolicy::Annotate | GuardPolicy::LogOnly => {
                            report.sentences = attrs;
                            report.risk_triggered = true;
                            report.risks.push(RiskId::Groundedness);
                        }
                    }
                } else {
                    report.sentences = attrs;
                }
                report.checks.push(Verdict {
                    risk_id: RiskId::Groundedness,
                    triggered: ungrounded > 0,
                    confidence: if ungrounded == 0 { 1.0 } else { 0.9 },
                    trace: None,
                    explanation: Some(format!(
                        "{ungrounded} of {} sentences ungrounded",
                        report.sentences.len()
                    )),
                });
            } else {
                let g = self.guardian.check_groundedness(&answer, documents)?;
                if g.triggered && g.confidence >= self.cfg.groundedness_threshold {
                    if self.cfg.policy == GuardPolicy::Abstain {
                        report.checks.push(g);
                        report.risk_triggered = true;
                        report.risks.push(RiskId::Groundedness);
                        report.action = GuardAction::Rejected;
                        return Ok(GuardDecision {
                            answer: abstention_text(
                                query,
                                "the answer is not sufficiently grounded in the retrieved context",
                            ),
                            report,
                        });
                    }
                }
                report.checks.push(g);
            }
        }

        // ---- 3. answer relevance ----
        let ar = self.guardian.check_answer_relevance(query, &answer)?;
        if ar.triggered {
            report.risk_triggered = true;
            report.risks.push(RiskId::AnswerRelevance);
            if self.cfg.policy == GuardPolicy::Abstain {
                report.checks.push(ar);
                report.action = GuardAction::Rejected;
                return Ok(GuardDecision {
                    answer: abstention_text(query, "the generated answer does not address the question"),
                    report,
                });
            }
        }
        report.checks.push(ar);

        // ---- 4. harm ----
        if self.cfg.harm_check {
            let h = self.guardian.check_harm(&answer)?;
            if h.triggered {
                report.risk_triggered = true;
                report.risks.push(h.risk_id);
                if self.cfg.policy == GuardPolicy::Abstain {
                    report.checks.push(h);
                    report.action = GuardAction::Rejected;
                    return Ok(GuardDecision {
                        answer: "I can't provide that response because it was flagged by the safety guard."
                            .into(),
                        report,
                    });
                }
            }
            report.checks.push(h);
        }

        // ---- final action ----
        report.action = if report.repaired {
            GuardAction::Rewritten
        } else if report.risk_triggered && self.cfg.policy == GuardPolicy::Annotate {
            GuardAction::Annotated
        } else if report.risk_triggered {
            GuardAction::Logged
        } else {
            GuardAction::Passed
        };

        Ok(GuardDecision { answer, report })
    }
}

fn abstention_text(query: &str, reason: &str) -> String {
    format!("I can't answer \"{query}\" confidently: {reason}.")
}

/// A minimal generation trait so the repair loop works with any generator.
pub trait Generator: Send + Sync {
    /// Generate an answer given a (system, user) prompt pair.
    fn generate(&self, system: &str, user: &str) -> Result<String>;
}

/// Generate → guard → repair loop.
///
/// 1. `generate` produces a candidate answer from the RAG prompt.
/// 2. [`RagGuard`] verifies it.
/// 3. On failure (policy permitting), regenerates with judge feedback
///    appended to the prompt, up to `max_attempts`.
/// 4. Falls back to the guard's decision (rewrite/abstain).
pub struct GuardedChat {
    guard: RagGuard,
}

impl GuardedChat {
    pub fn new(guard: RagGuard) -> Self {
        Self { guard }
    }

    /// Run one guarded generation.
    pub fn answer(
        &self,
        generator: &dyn Generator,
        query: &str,
        documents: &[String],
        base_system_prompt: &str,
    ) -> Result<GuardDecision> {
        let mut system = base_system_prompt.to_string();
        let ctx = documents
            .iter()
            .enumerate()
            .map(|(i, d)| format!("[Doc {i}] {}", d.trim()))
            .collect::<Vec<_>>()
            .join("\n\n");

        for attempt in 0..=self.guard.cfg.max_attempts {
            let user = if attempt == 0 {
                format!("Context:\n{ctx}\n\nQuestion: {query}")
            } else {
                format!(
                    "Context:\n{ctx}\n\nQuestion: {query}\n\n\
                     IMPORTANT: your previous answer failed verification. \
                     Only state facts that are explicitly supported by the \
                     context above. Do not add outside information, numbers, \
                     names, or dates that do not appear in the context."
                )
            };
            let candidate = generator.generate(&system, &user)?;
            let decision = self.guard.check_rag_answer(query, &candidate, documents)?;
            if decision.safe() || self.guard.cfg.policy != GuardPolicy::Repair {
                return Ok(decision);
            }
            // repair policy: retry with feedback
            system = format!(
                "{base_system_prompt}\n\nVerification feedback: {} ungrounded sentences were removed from your last answer. Stricter grounding required.",
                decision.report.sentences.iter().filter(|s| !s.keep).count()
            );
        }
        // last resort: guard an explicit refusal
        Ok(GuardDecision {
            answer: abstention_text(query, "verification could not be satisfied"),
            report: GuardReport {
                action: GuardAction::Rejected,
                risk_triggered: true,
                risks: vec![RiskId::Groundedness],
                ..Default::default()
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::MockJudge;

    fn docs() -> Vec<String> {
        vec!["Eat (1964) is a 45-minute film by Andy Warhol. It was first shown on July 16, 1964."
            .to_string()]
    }

    #[test]
    fn clean_answer_passes() {
        let guard = RagGuard::new(Guardian::new()).with_policy(GuardPolicy::Abstain);
        let answer = "Eat is a 45-minute film by Andy Warhol, first shown on July 16, 1964.";
        let d = guard.check_rag_answer("Tell me about Eat", &answer, &docs()).unwrap();
        assert!(d.safe());
        assert_eq!(d.report.action, GuardAction::Passed);
        assert!(!d.report.repaired);
        assert!(!d.report.sentences.is_empty());
    }

    #[test]
    fn fabricated_answer_is_eliminated() {
        let guard = RagGuard::new(Guardian::new()).with_policy(GuardPolicy::Abstain);
        let answer = "Eat is a 45-minute film by Andy Warhol. It won three Oscars in 2001.";
        let d = guard.check_rag_answer("Tell me about Eat", &answer, &docs()).unwrap();
        // sentence 2 fabricated -> dropped; sentence 1 survives -> rewritten, not rejected
        assert!(d.report.repaired);
        assert!(d.answer.contains("45-minute"));
        assert!(!d.answer.contains("Oscars"));
        assert_eq!(d.report.action, GuardAction::Rewritten);
    }

    #[test]
    fn fully_fabricated_answer_abstains() {
        let guard = RagGuard::new(Guardian::new()).with_policy(GuardPolicy::Abstain);
        let answer = "Kubrick directed Eat in 1972 and it won five Academy Awards in 1973.";
        let d = guard.check_rag_answer("Tell me about Eat", &answer, &docs()).unwrap();
        assert!(!d.safe());
        assert_eq!(d.report.action, GuardAction::Rejected);
        assert!(d.answer.contains("can't answer"));
        assert!(d.report.risks.contains(&RiskId::Groundedness));
    }

    #[test]
    fn annotate_policy_keeps_text() {
        let guard = RagGuard::new(Guardian::new()).with_policy(GuardPolicy::Annotate);
        let answer = "Eat is a 45-minute film by Andy Warhol. It won three Oscars in 2001.";
        let d = guard.check_rag_answer("Tell me about Eat", &answer, &docs()).unwrap();
        assert_eq!(d.answer, answer, "annotate must not modify the text");
        assert!(d.report.risk_triggered);
        assert_eq!(d.report.action, GuardAction::Annotated);
    }

    #[test]
    fn log_policy_ships_everything() {
        let guard = RagGuard::new(Guardian::new()).with_policy(GuardPolicy::LogOnly);
        let answer = "Kubrick directed Eat in 1972.";
        let d = guard.check_rag_answer("q", &answer, &docs()).unwrap();
        assert_eq!(d.answer, answer);
        assert_eq!(d.report.action, GuardAction::Logged);
    }

    struct FixedGenerator(&'static str);
    impl Generator for FixedGenerator {
        fn generate(&self, _system: &str, _user: &str) -> Result<String> {
            Ok(self.0.to_string())
        }
    }

    #[test]
    fn guarded_chat_repair_loop() {
        let guard = RagGuard::new(Guardian::new()).with_config(RepairConfig {
            policy: GuardPolicy::Repair,
            max_attempts: 2,
            ..Default::default()
        });
        let chat = GuardedChat::new(guard);
        // generator always fabricates -> repair exhausts -> refusal
        let d = chat
            .answer(
                &FixedGenerator("Kubrick directed Eat in 1972 and it won five awards."),
                "Tell me about Eat",
                &docs(),
                "You are a helpful assistant.",
            )
            .unwrap();
        assert!(!d.safe());
        assert!(d.answer.contains("can't answer"));
    }

    #[test]
    fn guarded_chat_good_generator() {
        let guard = RagGuard::new(Guardian::new()).with_config(RepairConfig {
            policy: GuardPolicy::Repair,
            ..Default::default()
        });
        let chat = GuardedChat::new(guard);
        let d = chat
            .answer(
                &FixedGenerator("Eat is a 45-minute film by Andy Warhol, first shown on July 16, 1964."),
                "Tell me about Eat",
                &docs(),
                "You are a helpful assistant.",
            )
            .unwrap();
        assert!(d.safe());
        assert_eq!(d.report.action, GuardAction::Passed);
    }

    #[test]
    fn mock_judge_interop() {
        // guardian over an explicit mock judge with forced grounding failures
        let g = Guardian::with_judge(MockJudge::forcing(RiskId::Groundedness));
        let v = g.check_groundedness("any text", &["any doc".to_string()]).unwrap();
        assert!(v.triggered);
    }
}
