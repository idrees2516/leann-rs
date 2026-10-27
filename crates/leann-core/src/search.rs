//! Two-level search with interleaved approximate/exact distances (paper §4.1)
//! plus dynamic batching (§4.2).
//!
//! The searcher maintains:
//! - **AQ** (approximate queue): PQ-ADC distances for every node ever
//!   encountered — cheap, no recompute.
//! - **EQ** (exact queue): exact distances for a small promoted subset,
//!   obtained by recomputing embeddings through the provider.
//!
//! Each hop expands the closest unvisited EQ node, pushes its neighbors into
//! the AQ, then promotes the top α fraction of the AQ (excluding already-exact
//! nodes) into a *pending* set. Pending nodes are flushed as one batched
//! provider call once the [`DynamicBatcher`] target is reached — possibly
//! spanning several hops. This is exactly the paper's "aggregate across hops,
//! trade slight staleness for GPU utilization" strategy.
//!
//! Upper HNSW layers are descended greedily using cheap ADC distances only.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;

use crate::batch::DynamicBatcher;
use crate::cache::{EmbedSource, TextSource};
use crate::graph::{F32Ord, HnswGraph};
use crate::pq::PqCodec;
use crate::Result;

/// Tunables for one search.
#[derive(Debug, Clone)]
pub struct SearchParams {
    pub k: usize,
    /// Beam width: max size of the exact candidate queue.
    pub ef: usize,
    /// Promotion ratio α: fraction of the AQ promoted to exact per hop.
    pub alpha: f32,
    pub min_batch: usize,
    pub max_batch: usize,
    /// Stopping relaxation: stop when the closest unvisited exact candidate is
    /// `relax` times worse than the k-th result (1.0 = strict HNSW rule).
    pub relax: f32,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            k: 10,
            ef: 96,
            alpha: 0.08,
            min_batch: 4,
            max_batch: 64,
            relax: 1.0,
        }
    }
}

/// Instrumentation returned from a search.
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub hops: usize,
    pub pq_scored: usize,
    pub promoted: usize,
    pub embed_calls: usize,
    pub vectors_embedded: usize,
    pub embed_secs: f64,
    pub total_secs: f64,
}

/// Two-level graph search. Returns `(id, exact_distance)` ascending.
pub fn two_level_search(
    graph: &HnswGraph,
    codec: &PqCodec,
    codes: &[u8],
    source: &EmbedSource,
    texts: &dyn TextSource,
    q: &[f32],
    params: &SearchParams,
) -> Result<(Vec<(u32, f32)>, SearchStats)> {
    let t0 = Instant::now();
    let n = graph.n;
    if n == 0 || codec.m == 0 {
        return Ok((Vec::new(), SearchStats::default()));
    }
    let m = codec.m;
    debug_assert_eq!(codes.len(), n * m);

    let table = codec.adc_table(q);
    let adc = |id: u32| -> f32 {
        let code = &codes[id as usize * m..(id as usize + 1) * m];
        codec.adc_distance(&table, code)
    };

    let k = params.k.clamp(1, n);
    let ef = params.ef.max(k).max(1);

    // ---- upper-layer greedy descent with ADC distances (cheap) ----
    let mut ep = graph.entry;
    for layer in (1..=graph.max_level).rev() {
        let mut best = adc(ep);
        let mut improved = true;
        while improved {
            improved = false;
            for &nbr in graph.neighbors(ep, layer) {
                let d = adc(nbr);
                if d < best {
                    best = d;
                    ep = nbr;
                    improved = true;
                }
            }
        }
    }

    // ---- layer-0 two-level beam search ----
    let mut stats = SearchStats::default();
    let mut batcher = DynamicBatcher::new(params.min_batch, params.max_batch);

    let mut eq: BinaryHeap<std::cmp::Reverse<(F32Ord, u32)>> = BinaryHeap::new(); // min-heap
    let mut results: BinaryHeap<(F32Ord, u32)> = BinaryHeap::new(); // max-heap, cap k
    let mut aq: HashMap<u32, f32> = HashMap::new();
    let mut exact: HashSet<u32> = HashSet::new();
    let mut pending: Vec<u32> = Vec::new();
    let mut pending_set: HashSet<u32> = HashSet::new();
    let mut visited: HashSet<u32> = HashSet::new();

    let metric = source.metric();

    macro_rules! kth_best {
        () => {
            results.peek().map(|(d, _)| d.get()).unwrap_or(f32::INFINITY)
        };
    }

    // Flush the pending recompute set as one batched provider call.
    // Implemented as a macro to avoid closure/batcher borrow conflicts.
    macro_rules! flush {
        ($pending:expr, $pending_set:expr, $eq:expr, $results:expr, $stats:expr, $batcher:expr) => {{
            if !$pending.is_empty() {
                let ids: Vec<u32> = std::mem::take(&mut $pending);
                $pending_set.clear();
                let t = Instant::now();
                let embs = match source.batch_by_id(&ids, texts) {
                    Ok(e) => e,
                    Err(e) => return Err(e),
                };
                let elapsed = t.elapsed();
                $stats.embed_calls += 1;
                $stats.vectors_embedded += ids.len();
                $stats.embed_secs += elapsed.as_secs_f64();
                $batcher.record(ids.len(), elapsed);
                for (id, emb) in ids.into_iter().zip(embs.into_iter()) {
                    let d = metric.distance(q, &emb);
                    $eq.push(std::cmp::Reverse((F32Ord::new(d), id)));
                    $results.push((F32Ord::new(d), id));
                    if $results.len() > ef {
                        $results.pop();
                    }
                }
                // keep EQ bounded
                if $eq.len() > ef * 2 {
                    let mut all: Vec<_> = $eq.drain().collect();
                    all.sort();
                    all.truncate(ef);
                    $eq = all.into_iter().collect();
                }
            }
        }};
    }

    // seed entry point
    aq.insert(ep, adc(ep));
    exact.insert(ep);
    pending.push(ep);
    pending_set.insert(ep);
    flush!(pending, pending_set, eq, results, stats, batcher);

    loop {
        // pop closest unvisited EQ node
        let mut current = None;
        while let Some(std::cmp::Reverse((d, id))) = eq.pop() {
            if !visited.contains(&id) {
                current = Some((d.get(), id));
                break;
            }
        }
        let (cd, cid) = match current {
            Some(x) => x,
            None => {
                if pending.is_empty() {
                    break;
                }
                flush!(pending, pending_set, eq, results, stats, batcher);
                continue;
            }
        };

        // stopping rule (HNSW semantics: stop only when the closest unvisited
        // exact candidate is worse than the worst of the ef-sized result set;
        // pending promotions may still rescue the search)
        if results.len() >= ef && cd > kth_best!() * params.relax {
            if pending.is_empty() {
                break;
            }
            // put the popped node back: pending may contain closer nodes,
            // and `cid` should still be expanded afterwards
            eq.push(std::cmp::Reverse((F32Ord::new(cd), cid)));
            flush!(pending, pending_set, eq, results, stats, batcher);
            continue;
        }

        visited.insert(cid);
        stats.hops += 1;

        // expand neighbors into AQ
        for &nbr in graph.neighbors(cid, 0) {
            if !aq.contains_key(&nbr) && !exact.contains(&nbr) {
                aq.insert(nbr, adc(nbr));
                stats.pq_scored += 1;
            }
        }

        // promote top-α of AQ (excluding exact & pending) into pending
        let take = ((params.alpha * aq.len() as f32).ceil() as usize)
            .clamp(1, params.max_batch);
        let mut cands: Vec<(f32, u32)> = aq
            .iter()
            .filter(|(id, _)| !exact.contains(id))
            .map(|(&id, &d)| (d, id))
            .collect();
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for (_, id) in cands.into_iter().take(take) {
            if pending_set.insert(id) {
                exact.insert(id);
                pending.push(id);
                stats.promoted += 1;
            }
        }

        if batcher.should_flush(pending.len()) {
            flush!(pending, pending_set, eq, results, stats, batcher);
        }
    }

    // final flush (leftovers)
    #[allow(unused_assignments)]
    {
        flush!(pending, pending_set, eq, results, stats, batcher);
    }

    let mut out: Vec<(u32, f32)> = results
        .into_iter()
        .map(|(d, id)| (id, d.get()))
        .collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    out.truncate(k);
    stats.total_secs = t0.elapsed().as_secs_f64();
    Ok((out, stats))
}

/// Brute-force exact search (embeds every alive chunk; for testing / exact
/// mode on small corpora). Uses the shared cache, so repeated calls are cheap.
pub fn brute_force_search(
    n: usize,
    alive: &dyn Fn(u32) -> bool,
    source: &EmbedSource,
    texts: &dyn TextSource,
    q: &[f32],
    k: usize,
) -> Result<(Vec<(u32, f32)>, SearchStats)> {
    let t0 = Instant::now();
    let ids: Vec<u32> = (0..n as u32).filter(|&i| alive(i)).collect();
    let mut stats = SearchStats::default();
    if ids.is_empty() {
        return Ok((Vec::new(), stats));
    }
    let t = Instant::now();
    let embs = source.batch_by_id(&ids, texts)?;
    stats.embed_calls = 1;
    stats.vectors_embedded = ids.len();
    stats.embed_secs = t.elapsed().as_secs_f64();
    let metric = source.metric();
    let mut scored: Vec<(u32, f32)> = ids
        .into_iter()
        .zip(embs.into_iter())
        .map(|(id, e)| (id, metric.distance(q, &e)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.truncate(k);
    stats.total_secs = t0.elapsed().as_secs_f64();
    Ok((scored, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::EmbeddingProvider;
    use crate::graph::{GraphBuildConfig, HubPruneConfig};
    use crate::metric::{normalize, Metric};
    use crate::pq::{PqCodec, PqConfig};
    use std::sync::Arc;

    /// Deterministic n-gram hashing embedder (same idea as leann-embedder's
    /// local fallback, kept tiny for tests).
    struct NgramProvider {
        dim: usize,
    }
    impl EmbeddingProvider for NgramProvider {
        fn name(&self) -> &str {
            "test-ngram"
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| ngram_embed(t, self.dim)).collect())
        }
    }

    pub fn ngram_embed(text: &str, dim: usize) -> Vec<f32> {
        let mut v = vec![0f32; dim];
        let chars: Vec<char> = text.chars().collect();
        for w in text.split_whitespace() {
            let h = fxhash(w);
            v[(h as usize) % dim] += 1.0;
        }
        for i in 0..chars.len().saturating_sub(2) {
            let g: String = chars[i..i + 3].iter().collect();
            let h = fxhash(&g);
            v[(h as usize) % dim] += 0.5;
        }
        normalize(&mut v);
        v
    }

    fn fxhash(s: &str) -> u64 {
        let mut h: u64 = 0x517c_c1b7_2722_0a95;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        h
    }

    fn corpus(n: usize) -> Vec<String> {
        let topics = [
            "machine learning gradient descent neural network training",
            "rust ownership borrow checker lifetimes memory safety",
            "vector database index search retrieval embedding",
            "coffee brewing espresso roast temperature grind",
            "planetary orbit gravity telescope astronomy star",
            "recipe pasta tomato garlic basil olive oil",
            "quarterly revenue profit margin sales forecast",
            "hiking mountain trail forest wilderness camp",
        ];
        (0..n)
            .map(|i| format!("doc marker{i:05} {i} about {}", topics[i % topics.len()]))
            .collect()
    }

    #[test]
    fn two_level_recall_vs_bruteforce() {
        let texts = corpus(800);
        let provider = Arc::new(NgramProvider { dim: 128 });
        let source = EmbedSource::new(provider, Metric::Cosine, 4096);
        let vecs = source.embed_texts(&texts.iter().map(|s| s.as_str()).collect::<Vec<_>>()).unwrap();

        // build graph + PQ
        let gcfg = GraphBuildConfig {
            m: 16,
            m0: 32,
            ef_construction: 120,
            seed: 3,
        };
        let oracle = crate::graph::SliceOracle {
            vecs: &vecs,
            metric: Metric::Cosine,
        };
        let mut graph = HnswGraph::build_local(texts.len(), &oracle, &gcfg);
        graph.hub_prune(&HubPruneConfig::default(), gcfg.m, &oracle);

        let pqcfg = PqConfig {
            m: 16,
            train_iters: 8,
            ..Default::default()
        };
        let codec = PqCodec::train(&vecs, &pqcfg, Metric::Cosine).unwrap();
        let codes: Vec<u8> = codec.encode_batch(&vecs).into_iter().flatten().collect();

        let alive = |_id: u32| true;

        let mut recalls = 0f32;
        let probes = 20;
        for qi in 0..probes {
            let qid = (qi * 37 + 5) % texts.len() as u32;
            let q = vecs[qid as usize].clone();
            let params = SearchParams {
                k: 5,
                ef: 48,
                alpha: 0.12,
                ..Default::default()
            };
            let (hits, stats) = two_level_search(
                &graph,
                &codec,
                &codes,
                &source,
                &texts,
                &q,
                &params,
            )
            .unwrap();
            assert!(!hits.is_empty());

            // ground truth via brute force
            let (truth, _) =
                brute_force_search(texts.len(), &alive, &source, &texts, &q, 5).unwrap();
            let got: HashSet<u32> = hits.iter().take(5).map(|(i, _)| *i).collect();
            let want: HashSet<u32> = truth.iter().take(5).map(|(i, _)| *i).collect();
            let inter = got.intersection(&want).count();
            recalls += inter as f32 / 5.0;

            // dynamic batching must have batched *some* work
            assert!(stats.vectors_embedded > 0);
        }
        let avg = recalls / probes as f32;
        assert!(avg > 0.8, "two-level recall too low: {avg}");
    }
}
