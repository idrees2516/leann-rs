//! Deterministic feature-hashing embedder.
//!
//! Text -> sparse features -> hashed dense vector -> L2 normalize.
//! Features: word unigrams (weight 1.0) + char trigrams (weight 0.5) +
//! word bigrams (weight 0.7). Captures lexical similarity surprisingly
//! well for a zero-dependency fallback and pairs nicely with BM25 hybrid
//! search. Not a semantic model — point API/embedding-model users at
//! [`crate::OpenAiEmbedder`] instead.

use rayon::prelude::*;
use std::sync::Arc;

use leann_core::cache::EmbeddingProvider;
use leann_core::metric::normalize;
use leann_core::{LeannError, Result};

pub struct LocalHashEmbedder {
    dim: usize,
}

impl LocalHashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self {
            dim: dim.clamp(32, 8192),
        }
    }

    pub fn embed_one(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        let words: Vec<&str> = text.split_whitespace().collect();

        let add = |feature: &str, weight: f32, v: &mut Vec<f32>| {
            let h = fxhash(feature);
            v[(h as usize) % self.dim] += weight
                * if (h >> 63) & 1 == 0 { 1.0 } else { -1.0 }; // signed hashing
        };

        for w in &words {
            add(&w.to_lowercase(), 1.0, &mut v);
        }
        for pair in words.windows(2) {
            add(&format!("{}{}", pair[0], pair[1]).to_lowercase(), 0.7, &mut v);
        }
        let chars: Vec<char> = text.to_lowercase().chars().collect();
        for g in chars.windows(3) {
            let s: String = g.iter().collect();
            add(&s, 0.5, &mut v);
        }
        normalize(&mut v);
        v
    }
}

fn fxhash(s: &str) -> u64 {
    let mut h: u64 = 0x517c_c1b7_2722_0a95;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

impl EmbeddingProvider for LocalHashEmbedder {
    fn name(&self) -> &str {
        "local-hash"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        Ok(texts.par_iter().map(|t| self.embed_one(t)).collect())
    }
}

/// Convenience constructor matching registry use.
#[allow(dead_code)]
pub fn local_hash(dim: usize) -> Arc<dyn EmbeddingProvider> {
    Arc::new(LocalHashEmbedder::new(dim))
}

#[allow(dead_code)]
fn _unused(_: LeannError) {}
