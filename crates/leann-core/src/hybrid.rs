//! Hybrid retrieval: fuse LEANN vector search with BM25 lexical search.
//!
//! Two fusion modes:
//! - **RRF** (reciprocal rank fusion, default): `score(d) = Σ 1/(k + rank_i(d))`
//!   — rank-based, robust to incomparable score scales.
//! - **Weighted**: min-max normalize each list, then apply per-source weights.

use serde::{Deserialize, Serialize};

/// Weighted-fusion weights (should sum to ~1.0).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HybridWeights {
    pub vector: f32,
    pub lexical: f32,
}

impl Default for HybridWeights {
    fn default() -> Self {
        Self {
            vector: 0.6,
            lexical: 0.4,
        }
    }
}

/// Fusion strategy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FusionMode {
    /// Reciprocal rank fusion (k = 60). Default — robust and tunable.
    #[default]
    Rrf,
    /// Min-max normalized weighted sum.
    Weighted,
}

/// One fused result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridHit {
    pub id: u32,
    pub fused_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_rank: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_rank: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_score: Option<f32>,
}

/// RRF constant.
const RRF_K: u32 = 60;

/// Fuse two ranked lists (each: (id, score), best first) into `top_k` results.
pub fn fusion(
    vector: &[(u32, f32)],
    lexical: &[(u32, f32)],
    mode: FusionMode,
    weights: HybridWeights,
    top_k: usize,
) -> Vec<HybridHit> {
    let mut out: Vec<HybridHit> = Vec::new();
    match mode {
        FusionMode::Rrf => {
            for (rank, &(id, score)) in vector.iter().enumerate() {
                out.push(HybridHit {
                    id,
                    fused_score: 1.0 / (RRF_K + rank as u32 + 1) as f32,
                    vector_rank: Some(rank as u32 + 1),
                    vector_score: Some(score),
                    lexical_rank: None,
                    lexical_score: None,
                });
            }
            for (rank, &(id, score)) in lexical.iter().enumerate() {
                let rrf = 1.0 / (RRF_K + rank as u32 + 1) as f32;
                if let Some(hit) = out.iter_mut().find(|h| h.id == id) {
                    hit.fused_score += rrf;
                    hit.lexical_rank = Some(rank as u32 + 1);
                    hit.lexical_score = Some(score);
                } else {
                    out.push(HybridHit {
                        id,
                        fused_score: rrf,
                        vector_rank: None,
                        vector_score: None,
                        lexical_rank: Some(rank as u32 + 1),
                        lexical_score: Some(score),
                    });
                }
            }
        }
        FusionMode::Weighted => {
            let norm = |list: &[(u32, f32)]| -> Vec<(u32, f32)> {
                if list.is_empty() {
                    return Vec::new();
                }
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                for &(_, s) in list {
                    lo = lo.min(s);
                    hi = hi.max(s);
                }
                let span = (hi - lo).max(1e-9);
                // note: for distances lower is better; caller passes similarity
                // or we invert here based on ordering assumption — we assume
                // scores are "higher is better" (similarity-like)
                list.iter().map(|&(id, s)| (id, (s - lo) / span)).collect()
            };
            let v = norm(vector);
            let l = norm(lexical);
            for &(id, s) in &v {
                out.push(HybridHit {
                    id,
                    fused_score: weights.vector * s,
                    vector_rank: None,
                    vector_score: Some(s),
                    lexical_rank: None,
                    lexical_score: None,
                });
            }
            for (rank, &(id, s)) in l.iter().enumerate() {
                if let Some(hit) = out.iter_mut().find(|h| h.id == id) {
                    hit.fused_score += weights.lexical * s;
                    hit.lexical_score = Some(s);
                    hit.lexical_rank = Some(rank as u32 + 1);
                } else {
                    out.push(HybridHit {
                        id,
                        fused_score: weights.lexical * s,
                        vector_rank: None,
                        vector_score: None,
                        lexical_rank: Some(rank as u32 + 1),
                        lexical_score: Some(s),
                    });
                }
            }
            // fill ranks for vector-only entries
            for (rank, hit) in out.iter_mut().enumerate() {
                if hit.vector_rank.is_none() && hit.vector_score.is_some() {
                    hit.vector_rank = Some(rank as u32 + 1);
                }
            }
        }
    }
    out.sort_by(|a, b| b.fused_score.partial_cmp(&a.fused_score).unwrap());
    out.truncate(top_k);
    out
}

/// Convenience wrapper matching `LeannIndex::hybrid_search`.
pub fn hybrid_search(
    vector: &[(u32, f32)],
    lexical: &[(u32, f32)],
    top_k: usize,
) -> Vec<HybridHit> {
    fusion(vector, lexical, FusionMode::default(), HybridWeights::default(), top_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_prefers_dual_hits() {
        let v = vec![(1u32, 0.9), (2, 0.8), (3, 0.7)];
        let l = vec![(2u32, 12.0), (4, 9.0), (1, 8.0)];
        let hits = hybrid_search(&v, &l, 3);
        assert_eq!(hits.len(), 3);
        // ids 1 and 2 appear in both -> top-2
        let top2: Vec<u32> = hits.iter().take(2).map(|h| h.id).collect();
        assert!(top2.contains(&1) && top2.contains(&2));
        assert!(hits[0].fused_score >= hits[1].fused_score);
    }

    #[test]
    fn weighted_mode() {
        let v = vec![(1u32, 1.0), (2, 0.5)];
        let l = vec![(2u32, 1.0), (1, 0.0)];
        let hits = fusion(
            &v,
            &l,
            FusionMode::Weighted,
            HybridWeights {
                vector: 0.5,
                lexical: 0.5,
            },
            2,
        );
        // both tie at 0.5
        assert!((hits[0].fused_score - 0.5).abs() < 1e-6);
        assert_eq!(hits.len(), 2);
    }
}
