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
//!
//! ## Scratch reuse (allocation-free steady state)
//!
//! [`SearchScratch`] replaces the per-call `HashMap`/`HashSet` bookkeeping
//! with **generation-stamped flat arrays** indexed by node id: membership
//! tests are a single indexed load + compare (no hashing, no rehashing, no
//! allocator traffic), and clearing between queries is `O(1)` — just bump
//! the generation counter. Batch/parallel searchers reuse one scratch per
//! thread, so steady-state search performs **zero heap allocations** outside
//! the result vector and the embedder batch buffers.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
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

/// Reusable per-thread search state: generation-stamped membership arrays
/// plus pre-allocated heaps. Clearing is O(1) (generation bump).
pub struct SearchScratch {
    gen: u64,
    /// generation stamp for "id is in the AQ"
    aq_gen: Vec<u64>,
    /// ADC distance when `aq_gen[id] == gen`
    aq_dist: Vec<f32>,
    /// ids ever inserted into the AQ this query (may include promoted ones)
    aq_list: Vec<u32>,
    /// generation stamp for "id has an exact distance (seeded or promoted)"
    exact_gen: Vec<u64>,
    /// generation stamp for "id visited (expanded)"
    visited_gen: Vec<u64>,
    /// generation stamp for "id currently pending recompute"
    pending_gen: Vec<u64>,
    eq: BinaryHeap<Reverse<(F32Ord, u32)>>,
    results: BinaryHeap<(F32Ord, u32)>,
    pending: Vec<u32>,
    /// scratch for the promotion candidate sort
    cands: Vec<(f32, u32)>,
}

impl SearchScratch {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            gen: 0,
            aq_gen: vec![0; n],
            aq_dist: vec![0.0; n],
            aq_list: Vec::new(),
            exact_gen: vec![0; n],
            visited_gen: vec![0; n],
            pending_gen: vec![0; n],
            eq: BinaryHeap::with_capacity(128),
            results: BinaryHeap::with_capacity(64),
            pending: Vec::with_capacity(64),
            cands: Vec::new(),
        }
    }

    /// Reset for a new query over `n` nodes. O(1) amortized: arrays only
    /// grow when the graph grew.
    #[inline]
    pub fn clear(&mut self, n: usize) {
        if self.aq_gen.len() < n {
            self.aq_gen.resize(n, 0);
            self.aq_dist.resize(n, 0.0);
            self.exact_gen.resize(n, 0);
            self.visited_gen.resize(n, 0);
            self.pending_gen.resize(n, 0);
        }
        self.gen = self.gen.wrapping_add(1);
        self.aq_list.clear();
        self.eq.clear();
        self.results.clear();
        self.pending.clear();
        self.cands.clear();
    }

    #[inline]
    fn is_visited(&self, id: u32) -> bool {
        self.visited_gen[id as usize] == self.gen
    }
}

impl Default for SearchScratch {
    fn default() -> Self {
        Self::new()
    }
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
    let mut scratch = SearchScratch::new();
    two_level_search_with_scratch(graph, codec, codes, source, texts, q, params, &mut scratch)
}

thread_local! {
    static TLS_SCRATCH: std::cell::RefCell<SearchScratch> =
        std::cell::RefCell::new(SearchScratch::new());
}

/// Two-level search using a thread-local scratch — identical semantics to
/// [`two_level_search`] but with per-thread scratch reuse, so steady-state
/// queries allocate nothing. Safe to call from rayon workers.
pub fn two_level_search_tls(
    graph: &HnswGraph,
    codec: &PqCodec,
    codes: &[u8],
    source: &EmbedSource,
    texts: &dyn TextSource,
    q: &[f32],
    params: &SearchParams,
) -> Result<(Vec<(u32, f32)>, SearchStats)> {
    TLS_SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        two_level_search_with_scratch(graph, codec, codes, source, texts, q, params, &mut s)
    })
}

/// Two-level search reusing a caller-provided [`SearchScratch`] — the
/// allocation-free steady-state entry point (one scratch per thread).
#[allow(clippy::too_many_arguments)]
pub fn two_level_search_with_scratch(
    graph: &HnswGraph,
    codec: &PqCodec,
    codes: &[u8],
    source: &EmbedSource,
    texts: &dyn TextSource,
    q: &[f32],
    params: &SearchParams,
    s: &mut SearchScratch,
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

    // ---- layer-0 two-level beam search (allocation-free bookkeeping) ----
    s.clear(n);
    let gen = s.gen;
    let mut stats = SearchStats::default();
    let mut batcher = DynamicBatcher::new(params.min_batch, params.max_batch);
    let metric = source.metric();

    macro_rules! kth_best {
        () => {
            s.results.peek().map(|(d, _)| d.get()).unwrap_or(f32::INFINITY)
        };
    }

    // Flush the pending recompute set as one batched provider call.
    macro_rules! flush {
        ($s:expr, $stats:expr, $batcher:expr) => {{
            if !$s.pending.is_empty() {
                let ids: Vec<u32> = std::mem::take(&mut $s.pending);
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
                    $s.eq.push(Reverse((F32Ord::new(d), id)));
                    $s.results.push((F32Ord::new(d), id));
                    if $s.results.len() > ef {
                        $s.results.pop();
                    }
                }
                // keep EQ bounded
                if $s.eq.len() > ef * 2 {
                    let mut all: Vec<_> = $s.eq.drain().collect();
                    all.sort_unstable();
                    all.truncate(ef);
                    $s.eq = all.into_iter().collect();
                }
            }
        }};
    }

    // seed entry point
    s.aq_gen[ep as usize] = gen;
    s.aq_dist[ep as usize] = adc(ep);
    s.aq_list.push(ep);
    s.exact_gen[ep as usize] = gen;
    s.pending_gen[ep as usize] = gen;
    s.pending.push(ep);
    flush!(s, stats, batcher);

    loop {
        // pop closest unvisited EQ node
        let mut current = None;
        while let Some(Reverse((d, id))) = s.eq.pop() {
            if !s.is_visited(id) {
                current = Some((d.get(), id));
                break;
            }
        }
        let (cd, cid) = match current {
            Some(x) => x,
            None => {
                if s.pending.is_empty() {
                    break;
                }
                flush!(s, stats, batcher);
                continue;
            }
        };

        // stopping rule (HNSW semantics: stop only when the closest unvisited
        // exact candidate is worse than the worst of the ef-sized result set;
        // pending promotions may still rescue the search)
        if s.results.len() >= ef && cd > kth_best!() * params.relax {
            if s.pending.is_empty() {
                break;
            }
            // put the popped node back: pending may contain closer nodes,
            // and `cid` should still be expanded afterwards
            s.eq.push(Reverse((F32Ord::new(cd), cid)));
            flush!(s, stats, batcher);
            continue;
        }

        s.visited_gen[cid as usize] = gen;
        stats.hops += 1;

        // expand neighbors into AQ
        for &nbr in graph.neighbors(cid, 0) {
            let nu = nbr as usize;
            if s.aq_gen[nu] != gen && s.exact_gen[nu] != gen {
                s.aq_gen[nu] = gen;
                s.aq_dist[nu] = adc(nbr);
                s.aq_list.push(nbr);
                stats.pq_scored += 1;
            }
        }

        // promote top-α of AQ (excluding exact & pending) into pending
        let take = ((params.alpha * s.aq_list.len() as f32).ceil() as usize)
            .clamp(1, params.max_batch);
        s.cands.clear();
        for &id in s.aq_list.iter() {
            if s.exact_gen[id as usize] != gen {
                s.cands.push((s.aq_dist[id as usize], id));
            }
        }
        s.cands.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let promoted = take.min(s.cands.len());
        for &(_, id) in s.cands.iter().take(promoted) {
            let iu = id as usize;
            if s.pending_gen[iu] != gen {
                s.pending_gen[iu] = gen;
                s.exact_gen[iu] = gen;
                s.pending.push(id);
                stats.promoted += 1;
            }
        }

        if batcher.should_flush(s.pending.len()) {
            flush!(s, stats, batcher);
        }
    }

    // final flush (leftovers)
    flush!(s, stats, batcher);

    let mut out: Vec<(u32, f32)> = s
        .results
        .drain()
        .map(|(d, id)| (id, d.get()))
        .collect();
    // distance ascending; id as tie-breaker so ordering is deterministic
    out.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
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
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
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
            let got: std::collections::HashSet<u32> = hits.iter().take(5).map(|(i, _)| *i).collect();
            let want: std::collections::HashSet<u32> = truth.iter().take(5).map(|(i, _)| *i).collect();
            let inter = got.intersection(&want).count();
            recalls += inter as f32 / 5.0;

            // dynamic batching must have batched *some* work
            assert!(stats.vectors_embedded > 0);
        }
        let avg = recalls / probes as f32;
        assert!(avg > 0.8, "two-level recall too low: {avg}");
    }

    #[test]
    fn scratch_reuse_matches_fresh_search() {
        // Reused scratch must produce identical results to a fresh one
        // across queries of different lengths (generation-stamp correctness).
        let texts = corpus(400);
        let provider = Arc::new(NgramProvider { dim: 96 });
        let source = EmbedSource::new(provider, Metric::Cosine, 4096);
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let vecs = source.embed_texts(&refs).unwrap();

        let gcfg = GraphBuildConfig {
            m: 12,
            m0: 24,
            ef_construction: 80,
            seed: 9,
        };
        let oracle = crate::graph::SliceOracle {
            vecs: &vecs,
            metric: Metric::Cosine,
        };
        let mut graph = HnswGraph::build_local(texts.len(), &oracle, &gcfg);
        graph.hub_prune(&HubPruneConfig::default(), gcfg.m, &oracle);
        let codec = PqCodec::train(&vecs, &PqConfig { m: 8, train_iters: 6, ..Default::default() }, Metric::Cosine).unwrap();
        let codes: Vec<u8> = codec.encode_batch_flat(&vecs);

        let params = SearchParams {
            k: 7,
            ef: 40,
            alpha: 0.15,
            ..Default::default()
        };
        // Pre-warm the embed cache so every search sees identical (cache-hit)
        // timing; the dynamic batcher adapts to measured latency, so cold vs
        // warm caches legitimately produce different batching patterns.
        let all_ids: Vec<u32> = (0..texts.len() as u32).collect();
        let _ = source.batch_by_id(&all_ids, &texts);

        let mut scratch = SearchScratch::new();
        for qi in 0..12u32 {
            let qid = (qi * 71 + 3) % texts.len() as u32;
            let q = vecs[qid as usize].clone();
            let (with_reuse, _) = two_level_search_with_scratch(
                &graph, &codec, &codes, &source, &texts, &q, &params, &mut scratch,
            )
            .unwrap();
            let (fresh, _) = two_level_search(
                &graph, &codec, &codes, &source, &texts, &q, &params,
            )
            .unwrap();
            assert_eq!(with_reuse, fresh, "qi={qi}");
        }
    }
}
