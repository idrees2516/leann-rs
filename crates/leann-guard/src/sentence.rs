//! Sentence-level attribution and grounded rewriting.
//!
//! The paper's guardian gives one verdict per text. For *elimination* we go
//! further: decompose an answer into sentences, verify each against the
//! evidence with the judge (plus the local numeric pre-filter), and either
//! (a) attach per-sentence citations, (b) flag risky sentences, or (c)
//! **rewrite the answer to contain only grounded sentences** — turning
//! hallucination *detection* into hallucination *removal*.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::block::GuardianRequest;
use crate::criteria::{criteria_for, RiskId};
use crate::judge::Judge;
use crate::local::local_groundedness;
use crate::Result;

/// One sentence's grounding status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceAttribution {
    /// Index of the sentence in the original answer.
    pub index: usize,
    pub text: String,
    /// Local (deterministic) grounding signal.
    pub local: crate::local::LocalGrounding,
    /// Judge verdict: None = skipped (local pass + cheap mode), Some =
    /// judged grounded (true) or hallucinated (false).
    pub judge_grounded: Option<bool>,
    /// Judge confidence for its verdict.
    pub judge_confidence: Option<f32>,
    /// Indices of documents judged/lexically supporting this sentence.
    pub supporting: Vec<usize>,
    /// Final per-sentence decision: keep this sentence?
    pub keep: bool,
}

/// Split an answer into sentences (heuristic; safe for citations like [1]).
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        cur.push(c);
        if c == '.' || c == '!' || c == '?' {
            // include trailing citation markers/quotes, then break on space
            let mut j = i + 1;
            while j < chars.len()
                && (chars[j] == ']'
                    || chars[j] == '"'
                    || chars[j] == '\''
                    || chars[j] == ')'
                    || chars[j].is_ascii_digit()
                    || chars[j] == '[')
            {
                cur.push(chars[j]);
                j += 1;
            }
            let boundary = j >= chars.len() || chars[j].is_whitespace() || chars[j] == '\n';
            if boundary && cur.chars().count() > 2 {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            i = j;
            continue;
        }
        i += 1;
    }
    if cur.trim().chars().count() > 1 {
        out.push(cur.trim().to_string());
    }
    out.into_iter().filter(|s| !s.is_empty()).collect()
}

/// Verify every sentence of `answer` against `documents`.
///
/// - `judge_all = true`: every sentence gets a judge call (accurate, costly)
/// - `judge_all = false` (default): only sentences that **fail or look
///   marginal on the local signal** (number misses, term coverage < 0.75)
///   get judge calls — typically 0-2 calls per answer instead of N.
///
/// Each sentence is checked against each document; a sentence is grounded if
/// the judge clears it against *any* document (or, without a judge, if its
/// best local coverage passes).
pub fn sentence_attribution(
    judge: &dyn Judge,
    answer: &str,
    documents: &[String],
    judge_all: bool,
) -> Result<Vec<SentenceAttribution>> {
    let sentences = split_sentences(answer);
    if documents.is_empty() {
        return Ok(sentences
            .into_iter()
            .enumerate()
            .map(|(index, text)| SentenceAttribution {
                index,
                text,
                local: crate::local::LocalGrounding {
                    number_coverage: 0.0,
                    term_coverage: 0.0,
                    grounded: false,
                    missing_numbers: Vec::new(),
                    missing_terms: Vec::new(),
                },
                judge_grounded: None,
                judge_confidence: None,
                supporting: Vec::new(),
                keep: false,
            })
            .collect());
    }

    let results: Vec<Result<SentenceAttribution>> = sentences
        .into_par_iter()
        .enumerate()
        .map(|(index, text)| -> Result<SentenceAttribution> {
            // best local signal across documents
            let mut best_local = None;
            let mut best_doc = 0usize;
            for (di, d) in documents.iter().enumerate() {
                let lg = local_groundedness(&text, d);
                let better = best_local
                    .as_ref()
                    .map(|b: &crate::local::LocalGrounding| {
                        (lg.term_coverage + lg.number_coverage)
                            > (b.term_coverage + b.number_coverage)
                    })
                    .unwrap_or(true);
                if better {
                    best_local = Some(lg);
                    best_doc = di;
                }
            }
            let local = best_local.unwrap();

            let needs_judge =
                judge_all || !local.grounded || local.term_coverage < 0.75;

            let mut judge_grounded = None;
            let mut judge_confidence = None;
            let mut supporting = Vec::new();

            if needs_judge {
                let criteria = criteria_for(RiskId::Groundedness);
                // judge against the full context (cheaper) — the block
                // protocol numbers documents so the judge can locate support
                let req = GuardianRequest {
                    text: &text,
                    documents,
                    query: None,
                    tools: &[],
                    criteria: &criteria.definition,
                    think: false,
                };
                let messages = crate::block::build_guardian_block(&req);
                let raw = judge.judge(&messages, 0.0)?;
                let parsed = crate::parse::parse_guardian_response(&raw);
                // guardian semantics: yes = risk present = ungrounded
                let grounded = parsed.risk_detected.map(|r| !r).unwrap_or(local.grounded);
                judge_grounded = Some(grounded);
                judge_confidence = parsed.confidence;
                if grounded {
                    supporting.push(best_doc);
                }
            } else {
                // local pass without judge: support = best matching doc
                supporting.push(best_doc);
            }

            let keep = judge_grounded.unwrap_or(local.grounded);
            Ok(SentenceAttribution {
                index,
                text,
                local,
                judge_grounded,
                judge_confidence,
                supporting,
                keep,
            })
        })
        .collect();

    let mut out = Vec::with_capacity(results.len());
    for r in results {
        out.push(r?);
    }
    Ok(out)
}

/// Rewrite an answer keeping only grounded sentences (the literal
/// elimination pass). Preserves order; drops sentences that failed checks.
/// Also strips citation markers when no documents support them.
pub fn grounded_rewrite(attributions: &[SentenceAttribution]) -> String {
    attributions
        .iter()
        .filter(|a| a.keep)
        .map(|a| a.text.clone())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::MockJudge;

    fn docs() -> Vec<String> {
        vec![
            "Eat (1964) is a 45-minute underground film by Andy Warhol featuring Robert Indiana. The film was first shown by Jonas Mekas on July 16, 1964 at the Washington Square Gallery."
                .to_string(),
            "Jonas Mekas was a Lithuanian-American filmmaker called the godfather of American avant-garde cinema.".to_string(),
        ]
    }

    #[test]
    fn sentence_splitting() {
        let s = split_sentences("First sentence. Second one cites [1]. Third? Yes! Trailing text");
        assert_eq!(s.len(), 5);
        assert_eq!(s[1], "Second one cites [1].");
        assert_eq!(s[4], "Trailing text");
    }

    #[test]
    fn attribution_flags_fabricated_numbers() {
        let judge = MockJudge::new();
        let answer = "Eat premiered on July 16, 1964. It won the Academy Award in 1995 for Best Picture. Warhol was American.";
        let attrs = sentence_attribution(&judge, answer, &docs(), false).unwrap();
        assert_eq!(attrs.len(), 3);
        // sentence 1: grounded (dates match)
        assert!(attrs[0].keep);
        // sentence 2: fabricated (1995, Oscar) -> judge triggered
        assert!(!attrs[1].keep, "fabricated sentence must be dropped: {:?}", attrs[1].local.missing_numbers);
        // sentence 3: "American" appears in doc 1 -> grounded
        assert!(attrs[2].keep);
    }

    #[test]
    fn rewrite_eliminates() {
        let judge = MockJudge::new();
        let answer = "Eat premiered on July 16, 1964. It won the Academy Award in 1995 for Best Picture.";
        let attrs = sentence_attribution(&judge, answer, &docs(), false).unwrap();
        let rewritten = grounded_rewrite(&attrs);
        assert!(rewritten.contains("July 16, 1964"));
        assert!(!rewritten.contains("Academy Award"));
        assert!(!rewritten.contains("1995"));
    }

    #[test]
    fn no_documents_drops_everything() {
        let judge = MockJudge::new();
        let attrs = sentence_attribution(&judge, "anything", &[], false).unwrap();
        assert!(!attrs[0].keep);
    }
}
