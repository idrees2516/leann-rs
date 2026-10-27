//! Neighbor context expansion.
//!
//! Chunking splits documents into overlapping pieces; a hit on chunk *i*
//! often means chunks *i-1* / *i+1* carry the surrounding reasoning. This
//! module reconstructs those neighbors from chunk ordering metadata
//! (`doc_id` + `chunk_index`), the feature memvid exposes as
//! "get context around a retrieved chunk".

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// How many chunks before/after each hit to pull in.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct ContextExpansionOptions {
    pub before: usize,
    pub after: usize,
}

/// One expanded hit: the primary chunk id plus its neighbors in order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandedContext {
    pub primary: u32,
    /// (chunk_id, offset from primary; negative = before)
    pub neighbors: Vec<(u32, i32)>,
}

/// Chunk placement info supplied by the index.
#[derive(Debug, Clone, Default)]
pub struct ChunkPlacement {
    /// doc key -> ordered chunk ids
    pub doc_chunks: HashMap<u64, Vec<u32>>,
    /// chunk id -> (doc key, ordinal)
    pub placement: HashMap<u32, (u64, u32)>,
}

impl ChunkPlacement {
    pub fn from_records(
        n: usize,
        doc_key_of: impl Fn(u32) -> Option<u64>,
        ordinal_of: impl Fn(u32) -> Option<u32>,
    ) -> Self {
        let mut p = Self::default();
        for id in 0..n as u32 {
            if let (Some(dk), Some(ord)) = (doc_key_of(id), ordinal_of(id)) {
                p.placement.insert(id, (dk, ord));
                p.doc_chunks.entry(dk).or_default().push(id);
            }
        }
        for list in p.doc_chunks.values_mut() {
            list.sort_by_key(|id| p.placement[id].1);
        }
        p
    }

    pub fn locate(&self, id: u32) -> Option<(u64, u32)> {
        self.placement.get(&id).copied()
    }
}

/// Expand hits with their neighboring chunks.
pub fn expand_context(
    hits: &[u32],
    placement: &ChunkPlacement,
    opts: ContextExpansionOptions,
) -> Vec<ExpandedContext> {
    hits.iter()
        .map(|&h| {
            let mut neighbors = Vec::new();
            if let Some(&(dk, _ord)) = placement.placement.get(&h) {
                if let Some(list) = placement.doc_chunks.get(&dk) {
                    if let Some(pos) = list.iter().position(|&x| x == h) {
                        for back in 1..=opts.before {
                            if pos >= back {
                                neighbors.push((list[pos - back], -(back as i32)));
                            }
                        }
                        neighbors.reverse();
                        for fwd in 1..=opts.after {
                            if pos + fwd < list.len() {
                                neighbors.push((list[pos + fwd], fwd as i32));
                            }
                        }
                    }
                }
            }
            ExpandedContext {
                primary: h,
                neighbors,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_both_directions() {
        let mut p = ChunkPlacement::default();
        p.doc_chunks.insert(7, vec![10, 11, 12, 13, 14]);
        for (ord, id) in [10, 11, 12, 13, 14].into_iter().enumerate() {
            p.placement.insert(id, (7, ord as u32));
        }
        let out = expand_context(
            &[12],
            &p,
            ContextExpansionOptions {
                before: 2,
                after: 1,
            },
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].neighbors, vec![(10, -2), (11, -1), (13, 1)]);

        // edge: first chunk
        let out2 = expand_context(
            &[10],
            &p,
            ContextExpansionOptions {
                before: 3,
                after: 1,
            },
        );
        assert_eq!(out2[0].neighbors, vec![(11, 1)]);
    }
}
