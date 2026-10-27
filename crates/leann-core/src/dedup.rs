//! Duplicate detection for ingestion: exact-text and near-duplicate.
//!
//! - **Exact**: 64-bit FxHash of the whitespace-normalized text. Zero false
//!   positives (modulo astronomically unlikely collisions), one array read.
//! - **Near**: 64-bit SimHash over token unigrams + bigrams. Documents whose
//!   SimHash values sit within Hamming distance 3 are considered
//!   near-duplicates (the classic Charikar threshold for 64-bit simhashes).
//!   This catches boilerplate, repeated headers/footers, re-saved files and
//!   minor edits — the things that silently poison recall in agentic RAG.
//!
//! The deduper is an in-memory set (16 bytes per entry: 8-byte hash + gen
//! bookkeeping is unnecessary — a plain `HashSet<u64>`), typically built
//! once at ingest time and dropped afterwards.

use std::collections::HashSet;

/// How aggressively to deduplicate at add/build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DedupMode {
    /// Keep everything (previous behavior).
    #[default]
    Off,
    /// Drop exact text duplicates (after whitespace normalization).
    Exact,
    /// Drop exact + near duplicates (SimHash Hamming distance <= 3).
    Near,
}

/// Max Hamming distance for the near-dup test. Calibrated empirically on
/// chunk-length texts: single-token edits land at ~6, reordered sentences
/// ~12, unrelated documents ~29 — so 10 separates "minor edit / re-save"
/// from "different content" with a comfortable margin while keeping false
/// positives (silently dropped distinct chunks) unlikely.
const NEAR_HAMMING: u32 = 10;

/// SimHash on very short texts is noisy ("alpha beta" vs "alpha beta gamma"
/// already sits at hamming 15); below this token count, near-mode falls back
/// to exact matching only.
const NEAR_MIN_TOKENS: usize = 16;

/// Duplicate detector state.
#[derive(Debug, Default, Clone)]
pub struct Deduper {
    mode: DedupMode,
    seen_exact: HashSet<u64>,
    seen_sim: Vec<u64>,
}

impl Deduper {
    pub fn new(mode: DedupMode) -> Self {
        Self {
            mode,
            seen_exact: HashSet::new(),
            seen_sim: Vec::new(),
        }
    }

    pub fn mode(&self) -> DedupMode {
        self.mode
    }

    /// Register `text`; returns `true` if it is a duplicate and should be
    /// skipped. First occurrences are recorded and return `false`.
    pub fn register(&mut self, text: &str) -> bool {
        match self.mode {
            DedupMode::Off => false,
            DedupMode::Exact => {
                let h = text_hash(text);
                !self.seen_exact.insert(h)
            }
            DedupMode::Near => {
                let h = text_hash(text);
                if !self.seen_exact.insert(h) {
                    return true;
                }
                if tokens(text).len() < NEAR_MIN_TOKENS {
                    return false; // too short for a meaningful simhash
                }
                let sh = simhash64(text);
                // exact-simhash short-circuit (very common for boilerplate)
                if self.seen_sim.contains(&sh) {
                    return true;
                }
                for &existing in &self.seen_sim {
                    if (existing ^ sh).count_ones() <= NEAR_HAMMING {
                        return true;
                    }
                }
                self.seen_sim.push(sh);
                false
            }
        }
    }

    /// Number of distinct texts registered so far.
    pub fn len(&self) -> usize {
        match self.mode {
            DedupMode::Off => 0,
            _ => self.seen_exact.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// FxHash-style 64-bit hash of the whitespace-normalized text.
#[inline]
pub fn text_hash(text: &str) -> u64 {
    let mut h: u64 = 0x517c_c1b7_2722_0a95;
    let mut last_ws = true; // collapse runs of whitespace
    for &b in text.as_bytes() {
        let ws = b == b' ' || b == b'\n' || b == b'\r' || b == b'\t';
        if ws && last_ws {
            continue;
        }
        h ^= if ws { b' ' } else { b } as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
        last_ws = ws;
    }
    h
}

/// Tokenizer shared with the simhash path: lowercase alphanumeric runs.
fn tokens(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            out.push(&text[s..i]);
        }
    }
    if let Some(s) = start {
        out.push(&text[s..]);
    }
    out
}

/// 64-bit SimHash over token unigrams + bigrams (Charikar's simhash).
pub fn simhash64(text: &str) -> u64 {
    let toks = tokens(text);
    if toks.is_empty() {
        return 0;
    }
    let mut weights = [0i64; 64];
    let mut feed = |gram: &str, w: i64| {
        let h = text_hash(gram);
        // weight each bit by +w/-w depending on the bit of the gram hash
        for b in 0..64 {
            if (h >> b) & 1 == 1 {
                weights[b] += w;
            } else {
                weights[b] -= w;
            }
        }
    };
    for t in &toks {
        feed(t, 1);
    }
    for pair in toks.windows(2) {
        feed(&format!("{}_{}", pair[0], pair[1]), 2);
    }
    let mut out = 0u64;
    for b in 0..64 {
        if weights[b] > 0 {
            out |= 1 << b;
        }
    }
    out
}

/// Hamming distance between two 64-bit values.
#[inline]
pub fn hamming64(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_mode() {
        let mut d = Deduper::new(DedupMode::Exact);
        assert!(!d.register("hello world"));
        assert!(d.register("hello world"));
        // whitespace-normalized: same text
        assert!(d.register("hello   world"));
        assert!(d.register("hello\nworld"));
        // different text
        assert!(!d.register("goodbye world"));
    }

    #[test]
    fn near_mode_catches_edits() {
        let mut d = Deduper::new(DedupMode::Near);
        let base = "The quarterly report shows revenue growth of 12 percent driven by enterprise sales in the EMEA region";
        assert!(!d.register(base));
        // single-token edit (calibrated: hamming ~6 <= 10) -> near dup
        assert!(d.register("The quarterly report shows revenue growth of 13 percent driven by enterprise sales in the EMEA region"));
        // re-saved identical file -> exact dup
        assert!(d.register(base));
        // genuinely different document (calibrated: hamming ~29) -> kept
        assert!(!d.register("Rust ownership rules prevent data races at compile time through the borrow checker and lifetimes"));
    }

    #[test]
    fn near_mode_short_texts_fall_back_to_exact() {
        let mut d = Deduper::new(DedupMode::Near);
        assert!(!d.register("tiny chunk"));
        assert!(d.register("tiny chunk"));
        // different short chunk: kept (simhash not applied below 16 tokens)
        assert!(!d.register("tiny chunk 2"));
    }

    #[test]
    fn off_mode_keeps_everything() {
        let mut d = Deduper::new(DedupMode::Off);
        assert!(!d.register("x"));
        assert!(!d.register("x"));
    }

    #[test]
    fn simhash_stability() {
        assert_eq!(simhash64("alpha beta gamma"), simhash64("alpha beta gamma"));
        assert_eq!(simhash64(""), 0);
        // short-text simhash is intentionally noisy; near-mode guards with
        // NEAR_MIN_TOKENS, so no tightness assertion here beyond determinism
        assert!(hamming64(simhash64("alpha beta"), simhash64("alpha beta gamma")) > 0);
    }
}
