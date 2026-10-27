//! BM25 inverted index for lexical (keyword) retrieval.
//!
//! LEANN's vector search handles semantics; BM25 handles exact keyword
//! matches (IDs, function names, rare terms) that embeddings often blur.
//! Fusing the two (see `hybrid.rs`) gives the agentic-RAG sweet spot that
//! plain vector stores miss.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// BM25 hyperparameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
    /// Strip a small English stop-word list. Default off (keeps precision for
    /// code identifiers like `if`/`for` intact is more important than a tiny
    /// index-size win).
    pub stop_words: bool,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            stop_words: false,
        }
    }
}

#[derive(Debug, Clone)]
struct Posting {
    doc: u32,
    tf: u16,
}

/// Inverted index over chunk texts.
#[derive(Debug, Default, Clone)]
pub struct Bm25Index {
    params: Bm25Params,
    n_docs: usize,
    avg_len: f32,
    doc_len: Vec<u32>,
    /// term -> postings (sorted by doc id)
    postings: HashMap<Box<str>, Vec<Posting>>,
}

const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were",
    "be", "been", "it", "this", "that", "with", "as", "at", "by", "from", "but", "not",
];

/// Unicode-alphanumeric tokenizer, lowercased.
pub fn tokenize(text: &str) -> Vec<Box<str>> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            cur.extend(ch.to_lowercase());
        } else if !cur.is_empty() {
            out.push(cur.as_str().into());
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur.as_str().into());
    }
    out
}

impl Bm25Index {
    pub fn new(params: Bm25Params) -> Self {
        Self {
            params,
            ..Default::default()
        }
    }

    /// Build from chunk texts in parallel.
    pub fn build(texts: &[&str], params: Bm25Params) -> Self {
        let per_doc: Vec<(u32, Vec<Box<str>>)> = texts
            .par_iter()
            .enumerate()
            .map(|(i, t)| (i as u32, tokenize(t)))
            .collect();

        let total_len: u64 = per_doc.iter().map(|(_, t)| t.len() as u64).sum();
        let n = per_doc.len();
        let mut idx = Self {
            params,
            n_docs: n,
            avg_len: if n > 0 { total_len as f32 / n as f32 } else { 0.0 },
            doc_len: per_doc.iter().map(|(_, t)| t.len() as u32).collect(),
            postings: HashMap::new(),
        };

        // merge per-doc term frequencies into postings
        for (doc, toks) in &per_doc {
            let mut tf: HashMap<&str, u16> = HashMap::new();
            for t in toks {
                *tf.entry(t).or_insert(0) += 1;
            }
            for (term, f) in tf {
                idx.postings
                    .entry(term.into())
                    .or_default()
                    .push(Posting { doc: *doc, tf: f });
            }
        }
        for list in idx.postings.values_mut() {
            list.sort_by_key(|p| p.doc);
        }
        idx
    }

    pub fn len(&self) -> usize {
        self.n_docs
    }

    pub fn is_empty(&self) -> bool {
        self.n_docs == 0
    }

    pub fn vocab_size(&self) -> usize {
        self.postings.len()
    }

    /// Append one document (for `add`); does not recompute avg_len exactly
    /// (running average).
    pub fn add_doc(&mut self, doc: u32, text: &str) {
        let toks = tokenize(text);
        self.doc_len.push(toks.len() as u32);
        self.avg_len = if self.n_docs == 0 {
            toks.len() as f32
        } else {
            (self.avg_len * self.n_docs as f32 + toks.len() as f32)
                / (self.n_docs as f32 + 1.0)
        };
        self.n_docs += 1;
        let mut tf: HashMap<&str, u16> = HashMap::new();
        for t in &toks {
            *tf.entry(&**t).or_insert(0) += 1;
        }
        for (term, f) in tf {
            self.postings
                .entry(term.into())
                .or_default()
                .push(Posting { doc, tf: f });
        }
    }

    /// Score documents against a query. `alive` filters tombstoned docs.
    /// Returns (doc, score) descending.
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        alive: &dyn Fn(u32) -> bool,
    ) -> Vec<(u32, f32)> {
        if self.n_docs == 0 {
            return Vec::new();
        }
        let toks = tokenize(query);
        let toks: Vec<&str> = if self.params.stop_words {
            toks.iter().map(|t| &**t).filter(|t| !STOP_WORDS.contains(t)).collect()
        } else {
            toks.iter().map(|t| &**t).collect()
        };
        if toks.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<u32, f32> = HashMap::new();
        let k1 = self.params.k1;
        let b = self.params.b;
        let n = self.n_docs as f32;

        for term in toks {
            let plist = match self.postings.get(term) {
                Some(p) => p,
                None => continue,
            };
            let df = plist.len() as f32;
            // BM25 idf (Lucene variant, always positive)
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for p in plist {
                if !alive(p.doc) {
                    continue;
                }
                let dl = self.doc_len.get(p.doc as usize).copied().unwrap_or(1) as f32;
                let tf = p.tf as f32;
                let norm = tf + k1 * (1.0 - b + b * dl / self.avg_len.max(1.0));
                let s = idf * (tf * (k1 + 1.0)) / norm;
                *scores.entry(p.doc).or_insert(0.0) += s;
            }
        }

        let mut out: Vec<(u32, f32)> = scores.into_iter().collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        out.truncate(top_k);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_ranks_keyword_hits() {
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "rust vector search engine implementation details",
            "quick sort algorithm in rust with benchmarks",
            "deep learning gradient descent optimization",
        ];
        let idx = Bm25Index::build(&texts, Bm25Params::default());
        let alive = |_| true;
        let hits = idx.search("rust vector", 3, &alive);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, 1, "doc 1 should rank first for 'rust vector'");
        let hits2 = idx.search("quick", 2, &alive);
        let ids: Vec<u32> = hits2.iter().map(|(d, _)| *d).collect();
        assert!(ids.contains(&0) && ids.contains(&2));
    }

    #[test]
    fn add_doc_extends() {
        let mut idx = Bm25Index::build(&["alpha beta"], Bm25Params::default());
        idx.add_doc(1, "gamma delta alpha");
        let alive = |_| true;
        let hits = idx.search("gamma", 5, &alive);
        assert_eq!(hits[0].0, 1);
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn tokenizer_unicode() {
        let toks = tokenize("Hello, world! café_42 naïve");
        let strs: Vec<&str> = toks.iter().map(|t| &**t).collect();
        assert!(strs.contains(&"hello"));
        assert!(strs.contains(&"café"));
        assert!(strs.contains(&"42"));
    }
}
