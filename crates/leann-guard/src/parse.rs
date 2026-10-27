//! Guardian response parsing: `<score>` tags, `IMD` reasoning traces, and
//! verbalized confidence (Granite Guardian 3.2+ feature).

use crate::Verdict;

/// Raw parsed output of one guardian call.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedVerdict {
    /// Some(true) = "yes", Some(false) = "no", None = unparseable.
    pub risk_detected: Option<bool>,
    /// Reasoning trace (think mode).
    pub trace: Option<String>,
    /// Verbalized confidence in [0,1] when present (e.g. "Confidence: 85%").
    pub confidence: Option<f32>,
    /// Full raw response (diagnostics).
    pub raw: String,
}

/// Parse a guardian-model response.
///
/// Accepted score forms (in priority order):
/// - `<score>yes</score>` / `<score>no</score>` (canonical)
/// - a bare trailing `yes` / `no` line (degenerate outputs)
/// - JSON `{"score": "yes"}` or `{"risk_detected": true}`
pub fn parse_guardian_response(raw: &str) -> ParsedVerdict {
    let mut out = ParsedVerdict {
        risk_detected: None,
        trace: None,
        confidence: None,
        raw: raw.to_string(),
    };

    // 1) reasoning trace (last IMD block wins)
    let trace = extract_tag(raw, "IMD", "IMD_END").or_else(|| {
        raw.lines()
            .find(|l| l.trim().starts_with("IMD"))
            .map(|l| l.trim().to_string())
    });
    out.trace = trace.filter(|t| !t.trim().is_empty());

    // 2) score tag(s) — take the LAST occurrence
    out.risk_detected = extract_tag(raw, "<score>", "</score>")
        .map(|s| parse_yes_no(&s))
        .flatten();

    // 3) fallbacks
    if out.risk_detected.is_none() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw.trim()) {
            if let Some(s) = v.get("score").and_then(|s| s.as_str()) {
                out.risk_detected = parse_yes_no(s);
            } else if let Some(b) = v.get("risk_detected").and_then(|b| b.as_bool()) {
                out.risk_detected = Some(b);
            }
        }
    }
    if out.risk_detected.is_none() {
        // last non-empty line, yes/no only
        if let Some(line) = raw.lines().rev().find(|l| !l.trim().is_empty()) {
            out.risk_detected = parse_yes_no(line.trim());
        }
    }

    // 4) verbalized confidence: "Confidence: 85%" / "confidence 0.85"
    if let Some(c) = parse_confidence(raw) {
        out.confidence = Some(c);
    }

    out
}

fn extract_tag(raw: &str, open: &str, close: &str) -> Option<String> {
    let mut best: Option<String> = None;
    let mut rest = raw;
    while let Some(s) = rest.find(open) {
        let after = &rest[s + open.len()..];
        if let Some(e) = after.find(close) {
            best = Some(after[..e].trim().to_string());
            rest = &after[e + close.len()..];
        } else {
            // unterminated: take everything after the opener
            best = Some(after.trim().to_string());
            break;
        }
    }
    best
}

fn parse_yes_no(s: &str) -> Option<bool> {
    let t = s.trim().trim_end_matches('.').to_ascii_lowercase();
    match t.as_str() {
        "yes" | "'yes'" | "\"yes\"" | "y" | "true" => Some(true),
        "no" | "'no'" | "\"no\"" | "n" | "false" => Some(false),
        _ => {
            // tolerate "Yes, because..." style answers
            if t.starts_with("yes") {
                Some(true)
            } else if t.starts_with("no") {
                Some(false)
            } else {
                None
            }
        }
    }
}

fn parse_confidence(raw: &str) -> Option<f32> {
    let lower = raw.to_ascii_lowercase();
    let idx = lower.find("confidence")?;
    let window = &lower[idx..(idx + 40).min(lower.len())];
    // percent form
    if let Some(p) = window.find('%') {
        let digits: String = window[..p]
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ' ')
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        if let Ok(v) = digits.trim().parse::<f32>() {
            return Some((v / 100.0).clamp(0.0, 1.0));
        }
    }
    // 0-1 form
    let mut best: Option<f32> = None;
    for tok in window.split_whitespace() {
        if let Ok(v) = tok.trim_matches(&[':', ',', '.'][..]).parse::<f32>() {
            if (0.0..=1.0).contains(&v) {
                best = Some(v);
            }
        }
    }
    best
}

/// Convert a parsed verdict into a typed [`Verdict`] for a risk.
///
/// When the judge did not verbalize a confidence, `default_confidence`
/// (1.0 = trust the judgment fully) is used.
pub fn to_verdict(parsed: &ParsedVerdict, risk: crate::RiskId, default_confidence: f32) -> Verdict {
    let triggered = parsed.risk_detected.unwrap_or(false);
    let confidence = parsed.confidence.unwrap_or(default_confidence);
    Verdict {
        risk_id: risk,
        triggered,
        confidence,
        trace: parsed.trace.clone(),
        explanation: if parsed.risk_detected.is_none() {
            Some("unparseable judge output".to_string())
        } else {
            None
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::criteria::RiskId;

    #[test]
    fn parses_canonical_score() {
        let p = parse_guardian_response("IMD The answer cites Doc 0 correctly. IMD_END<score>no</score>");
        assert_eq!(p.risk_detected, Some(false));
        assert!(p.trace.unwrap().contains("cites Doc 0"));
    }

    #[test]
    fn parses_yes_with_confidence() {
        let p = parse_guardian_response("IMD check IMD_END<score>yes</score> Confidence: 85%");
        assert_eq!(p.risk_detected, Some(true));
        assert_eq!(p.confidence, Some(0.85));
    }

    #[test]
    fn parses_bare_trailing_yes() {
        let p = parse_guardian_response("no think\n\nyes");
        assert_eq!(p.risk_detected, Some(true));
    }

    #[test]
    fn parses_json_fallback() {
        let p = parse_guardian_response(r#"{"score": "yes"}"#);
        assert_eq!(p.risk_detected, Some(true));
        let p2 = parse_guardian_response(r#"{"risk_detected": false}"#);
        assert_eq!(p2.risk_detected, Some(false));
    }

    #[test]
    fn confidence_forms() {
        assert_eq!(parse_confidence("Confidence: 70%"), Some(0.7));
        assert_eq!(parse_confidence("confidence 0.9"), Some(0.9));
        assert_eq!(parse_confidence("no confidence here"), None);
    }

    #[test]
    fn to_verdict_defaults() {
        let p = parse_guardian_response("<score>yes</score>");
        let v = to_verdict(&p, RiskId::Groundedness, 1.0);
        assert!(v.triggered);
        assert_eq!(v.confidence, 1.0);
    }
}
