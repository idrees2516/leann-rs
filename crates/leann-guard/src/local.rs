//! Deterministic local (no-LLM) grounding checks.
//!
//! Cheap first-pass filters that catch the most common hallucination
//! classes before spending a judge call:
//! - **Number consistency**: every numeral in the answer must appear in the
//!   context (dates, quantities, versions — the classic fabrication vector).
//! - **Term-coverage grounding**: content words of the answer should be
//!   mostly covered by the context (an NLI-lite lexical entailment signal).
//! - **Citation validation**: `[n]` citations must reference existing docs.

use serde::{Deserialize, Serialize};

/// Result of a lexical grounding check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalGrounding {
    /// Fraction of answer numbers found in the context (0-1).
    pub number_coverage: f32,
    /// Fraction of answer content terms covered by the context (0-1).
    pub term_coverage: f32,
    /// Overall pass at the given thresholds.
    pub grounded: bool,
    /// The specific numbers missing from context (diagnostics).
    pub missing_numbers: Vec<String>,
    /// The specific salient terms missing from context.
    pub missing_terms: Vec<String>,
}

const STOP: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "is", "are", "was",
    "were", "be", "been", "it", "this", "that", "with", "as", "at", "by", "from", "not", "no",
    "yes", "if", "then", "than", "so", "such", "can", "could", "will", "would", "should", "may",
    "might", "shall", "do", "does", "did", "have", "has", "had", "i", "you", "he", "she", "we",
    "they", "there", "here", "what", "which", "who", "whom", "when", "where", "why", "how",
    "also", "into", "over", "under", "about", "after", "before", "between", "during", "through",
    "its", "their", "our", "your", "his", "her", "them", "us", "me", "him",
];

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1 && !STOP.contains(t))
        .map(String::from)
        .collect()
}

/// Extract numbers (integers, decimals, years, percentages, versions) from
/// text in normalized form.
fn extract_numbers(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_digit() || (ch == '.' && cur.contains(|c: char| c.is_ascii_digit())) {
            cur.push(ch);
        } else if !cur.is_empty() {
            push_number(&mut out, &cur);
            cur.clear();
        }
    }
    if !cur.is_empty() {
        push_number(&mut out, &cur);
    }
    // small standalone numbers (1..12) are usually list indices / ordinals —
    // keep them only when they look like years or quantities attached above
    out
}

fn push_number(out: &mut Vec<String>, s: &str) {
    let t = s.trim_matches('.');
    if t.is_empty() {
        return;
    }
    // skip trivial enumerations like "[1]" handled by caller-side filtering
    if t.len() >= 2 || t.parse::<i64>().map(|v| v > 31).unwrap_or(false) {
        out.push(normalize_number(t));
    }
}

fn normalize_number(s: &str) -> String {
    // strip leading zeros: "1964", "0.85", "16"
    if let Some(stripped) = s.strip_prefix("0.") {
        return format!("0.{stripped}");
    }
    let trimmed = s.trim_start_matches('0');
    if trimmed.is_empty() {
        "0".into()
    } else {
        trimmed.into()
    }
}

/// Check that every number in `answer` appears in `context`.
/// Returns false when any is missing.
pub fn number_consistency(answer: &str, context: &str) -> bool {
    let ctx: Vec<String> = extract_numbers(context);
    let ans: Vec<String> = extract_numbers(answer);
    ans.iter().all(|n| ctx.contains(n))
}

/// Full local grounding analysis of an answer against its context.
pub fn local_groundedness(answer: &str, context: &str) -> LocalGrounding {
    let ans_terms = tokenize(answer);
    let ctx_terms = tokenize(context);
    let ctx_set: std::collections::HashSet<&String> = ctx_terms.iter().collect();

    let mut missing_terms: Vec<String> = Vec::new();
    let mut covered = 0usize;
    for t in &ans_terms {
        if ctx_set.contains(t) {
            covered += 1;
        } else {
            missing_terms.push(t.clone());
        }
    }
    let term_coverage = if ans_terms.is_empty() {
        1.0
    } else {
        covered as f32 / ans_terms.len() as f32
    };

    let ans_nums = extract_numbers(answer);
    let ctx_nums: Vec<String> = extract_numbers(context);
    let missing_numbers: Vec<String> = ans_nums
        .iter()
        .filter(|n| !ctx_nums.contains(n))
        .cloned()
        .collect();
    let number_coverage = if ans_nums.is_empty() {
        1.0
    } else {
        1.0 - missing_numbers.len() as f32 / ans_nums.len() as f32
    };

    LocalGrounding {
        number_coverage,
        term_coverage,
        // numbers are hard evidence: any miss fails; terms are soft:
        // ≥55% coverage passes (answers legitimately paraphrase)
        grounded: number_coverage >= 1.0 && term_coverage >= 0.55,
        missing_numbers,
        missing_terms: missing_terms.into_iter().take(12).collect(),
    }
}

/// Validate that `[n]` citation markers in `answer` reference documents in
/// range. Returns the list of invalid citations.
pub fn validate_citations(answer: &str, doc_count: usize) -> Vec<usize> {
    let mut bad = Vec::new();
    let chars: Vec<char> = answer.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '[' {
            let mut j = i + 1;
            let mut num = String::new();
            while j < chars.len() && (chars[j].is_ascii_digit()) {
                num.push(chars[j]);
                j += 1;
            }
            if j < chars.len() && chars[j] == ']' && !num.is_empty() {
                if let Ok(n) = num.parse::<usize>() {
                    if n == 0 || n > doc_count {
                        bad.push(n);
                    }
                }
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    bad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_detected() {
        let n = extract_numbers("July 16, 1964 at 530 West Broadway, 45-minute film");
        let s: Vec<String> = n.into_iter().collect();
        assert!(s.contains(&"16".to_string()));
        assert!(s.contains(&"1964".to_string()));
        assert!(s.contains(&"530".to_string()));
        assert!(s.contains(&"45".to_string()));
    }

    #[test]
    fn number_consistency_works() {
        assert!(number_consistency(
            "premiered on July 16, 1964",
            "first shown on July 16, 1964"
        ));
        assert!(!number_consistency(
            "premiered on July 16, 1999",
            "first shown on July 16, 1964"
        ));
        assert!(number_consistency("no numbers here", "none there"));
    }

    #[test]
    fn local_grounding_signal() {
        let ctx = "Eat (1964) is a 45-minute film by Andy Warhol featuring Robert Indiana, first shown by Jonas Mekas on July 16, 1964.";
        let good = local_groundedness(
            "Eat is a 45-minute film by Andy Warhol, first shown on July 16, 1964.",
            ctx,
        );
        assert!(good.grounded);
        assert_eq!(good.number_coverage, 1.0);

        let bad = local_groundedness(
            "Eat is a 90-minute film by Stanley Kubrick released in 1972 that won three Oscars.",
            ctx,
        );
        assert!(!bad.grounded);
        assert!(bad.number_coverage < 1.0);
        assert!(!bad.missing_numbers.is_empty());
    }

    #[test]
    fn citation_validation() {
        assert!(validate_citations("supported by [1] and [2]", 3).is_empty());
        assert_eq!(validate_citations("bad [0] and [9]", 3), vec![0, 9]);
    }
}
