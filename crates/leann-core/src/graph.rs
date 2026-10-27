//! HNSW-style proximity graph with LEANN's high-degree preserving pruning.
//!
//! Build pipeline (mirrors the paper):
//! 1. Incremental HNSW insertion with `ef_construction` beam search and
//!    RNG-based neighbor shrinking (Algorithm 1 / FAISS `shrink_neighbor_list`).
//! 2. **High-degree preserving pruning** (Algorithm 3): the top β% of nodes
//!    by degree are designated hubs and may keep up to `M` out-edges; every
//!    other node is cut to `m = M / 5`. Edges pointing *at* hubs are always
//!    retained so low-degree nodes keep their navigation backbone.
//! 3. Optional sharded merge build: k-means soft-assigns each document to two
//!    shards, each shard builds its own graph, then the shard graphs are
//!    unioned — global connectivity through the two-shard membership.
//!
//! The resulting graph is stored in CSR form on disk; exact embeddings are
//! never persisted.

use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

use crate::metric::Metric;

/// Construction hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphBuildConfig {
    /// Max out-degree per node per layer during construction (paper default 32).
    pub m: usize,
    /// Max degree at layer 0 (HNSW convention: 2*m).
    pub m0: usize,
    /// Beam width during construction (paper default 200).
    pub ef_construction: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for GraphBuildConfig {
    fn default() -> Self {
        let m = 32;
        Self {
            m,
            m0: 2 * m,
            ef_construction: 200,
            seed: 42,
        }
    }
}

/// Hub pruning hyperparameters (paper Algorithm 3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubPruneConfig {
    /// Fraction (percent) of nodes designated as hubs. Paper default: 2.0.
    pub beta_percent: f32,
    /// m = M / divisor for non-hub nodes. Paper default: 5.
    pub m_divisor: usize,
}

impl Default for HubPruneConfig {
    fn default() -> Self {
        Self {
            beta_percent: 2.0,
            m_divisor: 5,
        }
    }
}

/// Distance oracle: exact distance between two node ids.
/// Implementations may hit an embedding cache or recompute embeddings.
pub trait DistOracle: Sync {
    fn dist(&self, a: u32, b: u32) -> f32;
}

/// Oracle over an in-memory embedding table (build time).
pub struct SliceOracle<'a> {
    pub vecs: &'a [Vec<f32>],
    pub metric: Metric,
}

impl<'a> DistOracle for SliceOracle<'a> {
    #[inline]
    fn dist(&self, a: u32, b: u32) -> f32 {
        self.metric.distance(&self.vecs[a as usize], &self.vecs[b as usize])
    }
}

/// Total-order wrapper for f32 so it can live in a BinaryHeap.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct F32Ord(u32);

impl F32Ord {
    #[inline]
    pub(crate) fn new(d: f32) -> Self {
        Self(d.to_bits())
    }
    #[inline]
    pub(crate) fn get(self) -> f32 {
        f32::from_bits(self.0)
    }
}

impl PartialOrd for F32Ord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for F32Ord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

/// The proximity graph.
#[derive(Debug, Default, Clone)]
pub struct HnswGraph {
    pub n: usize,
    pub max_level: u8,
    pub entry: u32,
    /// Top layer of each node.
    pub levels: Vec<u8>,
    /// Layer-0 adjacency (CSR on disk, Vec in memory).
    pub adj0: Vec<Vec<u32>>,
    /// Upper-layer adjacency: `upper[node][l-1]` = neighbors at layer `l`.
    /// Empty for the vast majority of nodes.
    pub upper: Vec<Vec<Vec<u32>>>,
    /// Number of nodes actually inserted (== n for loaded graphs).
    inserted: usize,
}

impl HnswGraph {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            n,
            max_level: 0,
            entry: 0,
            levels: vec![0; n],
            adj0: vec![Vec::new(); n],
            upper: vec![Vec::new(); n],
            inserted: 0,
        }
    }

    #[inline]
    pub fn neighbors(&self, node: u32, layer: u8) -> &[u32] {
        if layer == 0 {
            &self.adj0[node as usize]
        } else if (layer as usize) <= self.upper[node as usize].len() {
            &self.upper[node as usize][(layer - 1) as usize]
        } else {
            &[]
        }
    }

    #[inline]
    pub fn degree0(&self, node: u32) -> usize {
        self.adj0[node as usize].len()
    }

    /// Total stored out-edge count across all layers.
    pub fn edge_count(&self) -> usize {
        let up: usize = self
            .upper
            .iter()
            .map(|u| u.iter().map(|l| l.len()).sum::<usize>())
            .sum();
        self.adj0.iter().map(|a| a.len()).sum::<usize>() + up
    }

    fn random_level(rng: &mut impl Rng, m: usize) -> u8 {
        let ml = 1.0 / (m.max(2) as f32).ln();
        let u: f32 = rng.gen::<f32>().clamp(1e-9, 1.0);
        ((-u.ln() * ml).floor() as i32).clamp(0, 24) as u8
    }

    /// Public level sampling (used by runtime `add`).
    pub fn sample_level(seed: u64, m: usize) -> u8 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self::random_level(&mut rng, m)
    }

    /// Grow storage to `new_n` nodes (new slots empty, uninserted).
    pub fn grow_to(&mut self, new_n: usize) {
        if new_n > self.n {
            self.levels.resize(new_n, 0);
            self.adj0.resize(new_n, Vec::new());
            self.upper.resize(new_n, Vec::new());
            self.n = new_n;
        }
    }

    /// Mark the graph as fully inserted (after loading from disk).
    pub fn mark_complete(&mut self, n: usize) {
        self.inserted = n;
    }

    /// Link nodes with zero layer-0 edges to the entry point (post-merge fix).
    pub fn connect_orphans(&mut self) {
        if self.n <= 1 {
            return;
        }
        let entry = self.entry;
        for i in 0..self.n as u32 {
            if i != entry && self.adj0[i as usize].is_empty() {
                self.adj0[i as usize].push(entry);
                if !self.adj0[entry as usize].contains(&i) {
                    self.adj0[entry as usize].push(i);
                }
            }
        }
    }

    /// Build a graph over `n` local ids (0..n) using exact distances from
    /// `oracle`. Used both for whole-dataset builds and per-shard builds.
    pub fn build_local(n: usize, oracle: &dyn DistOracle, cfg: &GraphBuildConfig) -> Self {
        let mut g = Self::with_capacity(n);
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        for id in 0..n as u32 {
            let level = Self::random_level(&mut rng, cfg.m);
            g.insert(id, level, oracle, cfg);
        }
        g
    }

    /// Incremental HNSW insert of `id` at `level`.
    pub fn insert(&mut self, id: u32, level: u8, oracle: &dyn DistOracle, cfg: &GraphBuildConfig) {
        debug_assert!((id as usize) < self.n);
        self.levels[id as usize] = level;
        if level > 0 {
            if self.upper[id as usize].len() < level as usize {
                self.upper[id as usize].resize(level as usize, Vec::new());
            }
        }

        if self.inserted == 0 {
            self.entry = id;
            self.max_level = level;
            self.inserted = 1;
            return;
        }
        self.inserted += 1;

        // 1) greedy descent through layers above `level`
        let mut ep = self.entry;
        for l in ((level + 1)..=self.max_level).rev() {
            ep = self.greedy_step(ep, l, id, oracle);
        }

        // 2) beam search + connect on each layer from min(level, max_level)..0
        let top = level.min(self.max_level);
        for l in (0..=top).rev() {
            let cands = self.search_layer(id, &[ep], l, cfg.ef_construction, oracle);
            let mmax = if l == 0 { cfg.m0 } else { cfg.m };
            let selected = Self::select_rng(cands, cfg.m, oracle);

            self.layer_mut(id, l)
                .extend(selected.iter().map(|(_, x)| *x));

            for &(_d, nbr) in &selected {
                let nbr_top = self.levels[nbr as usize];
                if l > nbr_top {
                    continue;
                }
                let nbr_list = self.layer_mut(nbr, l);
                if !nbr_list.contains(&id) {
                    nbr_list.push(id);
                }
                if nbr_list.len() > mmax {
                    let mut c: Vec<(f32, u32)> =
                        nbr_list.iter().map(|&x| (oracle.dist(nbr, x), x)).collect();
                    c.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    let shrunk = Self::select_rng(c, mmax, oracle);
                    let ids: Vec<u32> = shrunk.iter().map(|(_, x)| *x).collect();
                    *self.layer_mut(nbr, l) = ids;
                }
            }
            ep = selected.first().map(|(_, x)| *x).unwrap_or(ep);
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = id;
        }
    }

    fn layer_mut(&mut self, node: u32, layer: u8) -> &mut Vec<u32> {
        if layer == 0 {
            &mut self.adj0[node as usize]
        } else {
            &mut self.upper[node as usize][(layer - 1) as usize]
        }
    }

    fn greedy_step(&self, mut cur: u32, layer: u8, q: u32, oracle: &dyn DistOracle) -> u32 {
        let mut improved = true;
        let mut best = oracle.dist(q, cur);
        while improved {
            improved = false;
            for &nbr in self.neighbors(cur, layer) {
                let d = oracle.dist(q, nbr);
                if d < best {
                    best = d;
                    cur = nbr;
                    improved = true;
                }
            }
        }
        cur
    }

    /// Beam search on one layer. Returns (dist, node) sorted ascending.
    pub fn search_layer(
        &self,
        q: u32,
        eps: &[u32],
        layer: u8,
        ef: usize,
        oracle: &dyn DistOracle,
    ) -> Vec<(f32, u32)> {
        let ef = ef.max(1);
        let mut visited = vec![false; self.n];
        let mut cand: BinaryHeap<std::cmp::Reverse<(F32Ord, u32)>> = BinaryHeap::new();
        let mut result: BinaryHeap<(F32Ord, u32)> = BinaryHeap::new();

        for &e in eps {
            if e as usize >= self.n || visited[e as usize] {
                continue;
            }
            visited[e as usize] = true;
            let d = oracle.dist(q, e);
            cand.push(std::cmp::Reverse((F32Ord::new(d), e)));
            result.push((F32Ord::new(d), e));
            if result.len() > ef {
                result.pop();
            }
        }

        while let Some(std::cmp::Reverse((cd, c))) = cand.pop() {
            if result.len() >= ef {
                if let Some(&(wd, _)) = result.peek() {
                    if cd.get() > wd.get() {
                        break;
                    }
                }
            }
            for &nbr in self.neighbors(c, layer) {
                if visited[nbr as usize] {
                    continue;
                }
                visited[nbr as usize] = true;
                let d = oracle.dist(q, nbr);
                let accept = match result.peek() {
                    Some(&(wd, _)) => d < wd.get() || result.len() < ef,
                    None => true,
                };
                if accept {
                    cand.push(std::cmp::Reverse((F32Ord::new(d), nbr)));
                    result.push((F32Ord::new(d), nbr));
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }

        let mut out: Vec<(f32, u32)> = result.into_iter().map(|(d, n)| (d.get(), n)).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out
    }

    /// RNG-based neighbor selection (Algorithm 1, line 6): keep candidate `x`
    /// unless an already-kept neighbor `a` is closer to `x` than `v` is.
    pub fn select_rng(
        mut cands: Vec<(f32, u32)>,
        m: usize,
        oracle: &dyn DistOracle,
    ) -> Vec<(f32, u32)> {
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cands.dedup_by(|a, b| a.1 == b.1);
        let mut kept: Vec<(f32, u32)> = Vec::with_capacity(m);
        for (d, x) in cands {
            if kept.len() >= m {
                break;
            }
            let dominated = kept.iter().any(|&(_, a)| oracle.dist(a, x) < d);
            if !dominated {
                kept.push((d, x));
            }
        }
        kept
    }

    /// LEANN Algorithm 3: high-degree preserving pruning.
    ///
    /// - Hubs (top β% by degree) keep up to `m_graph` edges.
    /// - Everyone else keeps up to `m = m_graph / divisor` edges.
    /// - Edges whose *target* is a hub are always kept (navigation backbone).
    pub fn hub_prune(&mut self, hub: &HubPruneConfig, m_graph: usize, oracle: &dyn DistOracle) {
        if self.n == 0 {
            return;
        }
        let mut order: Vec<u32> = (0..self.n as u32).collect();
        order.sort_by_key(|&a| std::cmp::Reverse(self.adj0[a as usize].len()));

        let hub_count = ((hub.beta_percent / 100.0) * self.n as f32).ceil() as usize;
        let hub_count = hub_count.clamp(1, self.n);
        let mut is_hub = vec![false; self.n];
        for &id in &order[..hub_count] {
            is_hub[id as usize] = true;
        }

        let m_low = (m_graph / hub.m_divisor.max(1)).max(2);
        let m_hub = m_graph.max(m_low);

        let old = std::mem::take(&mut self.adj0);
        let mut new_adj: Vec<Vec<u32>> = Vec::with_capacity(self.n);
        for (v, nbrs) in old.iter().enumerate() {
            let cap = if is_hub[v] { m_hub } else { m_low };
            if nbrs.len() <= cap {
                new_adj.push(nbrs.clone());
                continue;
            }
            let mut scored: Vec<(f32, u32)> =
                nbrs.iter().map(|&x| (oracle.dist(v as u32, x), x)).collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut kept: Vec<u32> = Vec::with_capacity(cap);
            for (_, x) in scored {
                if kept.len() >= cap {
                    break;
                }
                kept.push(x);
            }
            // re-add cut edges that point at hubs (paper's bidirectional
            // allowance keeps every node connected to the navigation backbone)
            for &x in nbrs {
                if kept.len() >= m_hub {
                    break;
                }
                if is_hub[x as usize] && !kept.contains(&x) {
                    kept.push(x);
                }
            }
            new_adj.push(kept);
        }
        self.adj0 = new_adj;
    }

    /// Remap local ids to global ids (used after shard-local builds).
    pub fn remap(&mut self, map: &[u32], global_n: usize) {
        let mut g = HnswGraph::with_capacity(global_n);
        g.inserted = global_n;
        for (local, &global) in map.iter().enumerate() {
            g.levels[global as usize] = self.levels[local];
            let lv = self.levels[local] as usize;
            if lv > 0 {
                g.upper[global as usize] = vec![Vec::new(); lv];
                for (li, lst) in self.upper[local].iter().enumerate() {
                    g.upper[global as usize][li] = lst.iter().map(|&l| map[l as usize]).collect();
                }
            }
        }
        for (local, lst) in self.adj0.iter().enumerate() {
            let global = map[local] as usize;
            g.adj0[global] = lst.iter().map(|&l| map[l as usize]).collect();
        }
        g.max_level = self.max_level;
        g.entry = map[self.entry as usize];
        *self = g;
    }

    /// Union another (already globally-remapped) graph into `self`,
    /// deduplicating edges and capping degrees (hubs keep more).
    pub fn merge_union(&mut self, other: &HnswGraph, hub: &HubPruneConfig, m_graph: usize) {
        if other.n == 0 || self.n == 0 {
            return;
        }
        debug_assert_eq!(self.n, other.n);
        let m_low = (m_graph / hub.m_divisor.max(1)).max(2);
        let m_hub = m_graph.max(m_low);

        let mut order: Vec<u32> = (0..self.n as u32).collect();
        order.sort_by_key(|&a| {
            std::cmp::Reverse(self.adj0[a as usize].len() + other.adj0[a as usize].len())
        });
        let hub_count = ((hub.beta_percent / 100.0) * self.n as f32).ceil() as usize;
        let mut is_hub = vec![false; self.n];
        for &id in order.iter().take(hub_count.min(self.n)) {
            is_hub[id as usize] = true;
        }

        for i in 0..self.n {
            let mut seen = std::collections::HashSet::new();
            let mut merged: Vec<u32> = Vec::with_capacity(self.adj0[i].len() + other.adj0[i].len());
            for &x in self.adj0[i].iter().chain(other.adj0[i].iter()) {
                if x as usize != i && seen.insert(x) {
                    merged.push(x);
                }
            }
            let cap = if is_hub[i] { m_hub } else { m_low };
            if merged.len() > cap {
                // keep hub targets first (stable), then closest-seen order
                merged.sort_by_key(|&x| !is_hub[x as usize]);
                merged.truncate(cap);
            }
            self.adj0[i] = merged;

            let lmax = self.levels[i].max(other.levels[i]);
            if lmax > 0 && self.upper[i].len() < lmax as usize {
                self.upper[i].resize(lmax as usize, Vec::new());
            }
            for (li, lst) in other.upper[i].iter().enumerate() {
                while self.upper[i].len() <= li {
                    self.upper[i].push(Vec::new());
                }
                for &x in lst {
                    if !self.upper[i][li].contains(&x) {
                        self.upper[i][li].push(x);
                    }
                }
            }
            self.levels[i] = lmax;
        }
        if other.max_level > self.max_level {
            self.max_level = other.max_level;
            self.entry = other.entry;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::normalize;

    fn cluster_data(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &centers[i % clusters];
                let mut v: Vec<f32> =
                    c.iter().map(|x| x + (rng.gen::<f32>() - 0.5) * 0.12).collect();
                normalize(&mut v);
                v
            })
            .collect()
    }

    #[test]
    fn build_and_search_recall() {
        let n = 1200;
        let data = cluster_data(n, 64, 40, 3);
        let oracle = SliceOracle {
            vecs: &data,
            metric: Metric::Cosine,
        };
        let cfg = GraphBuildConfig {
            m: 16,
            m0: 32,
            ef_construction: 120,
            seed: 1,
        };
        let g = HnswGraph::build_local(n, &oracle, &cfg);
        assert_eq!(g.n, n);
        assert!(g.edge_count() > 0);

        let mut recalls = 0f32;
        let probes = 50;
        for qi in 0..probes {
            let q = (qi * 7 + 13) % n;
            let hits = g.search_layer(q as u32, &[g.entry], 0, 128, &oracle);
            let got: std::collections::HashSet<u32> =
                hits.iter().take(10).map(|(_, x)| *x).collect();

            let mut truth: Vec<(f32, u32)> = (0..n as u32)
                .filter(|&i| i != q as u32)
                .map(|i| (oracle.dist(q as u32, i), i))
                .collect();
            truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let truth: std::collections::HashSet<u32> =
                truth.iter().take(10).map(|(_, i)| *i).collect();
            recalls += got.intersection(&truth).count() as f32 / 10.0;
        }
        let avg = recalls / probes as f32;
        assert!(avg > 0.85, "graph recall too low: {avg}");
    }

    #[test]
    fn hub_prune_keeps_hubs_and_connectivity() {
        let n = 500;
        let data = cluster_data(n, 32, 10, 5);
        let oracle = SliceOracle {
            vecs: &data,
            metric: Metric::Cosine,
        };
        let cfg = GraphBuildConfig {
            m: 16,
            m0: 32,
            ef_construction: 100,
            seed: 2,
        };
        let mut g = HnswGraph::build_local(n, &oracle, &cfg);
        let before = g.edge_count();
        g.hub_prune(&HubPruneConfig::default(), 32, &oracle);
        assert!(g.edge_count() <= before, "pruning must not add edges");

        let mut recalls = 0f32;
        for qi in 0..20 {
            let q = (qi * 11 + 3) % n;
            let hits = g.search_layer(q as u32, &[g.entry], 0, 200, &oracle);
            let got: std::collections::HashSet<u32> =
                hits.iter().take(5).map(|(_, x)| *x).collect();
            let mut truth: Vec<(f32, u32)> = (0..n as u32)
                .filter(|&i| i != q as u32)
                .map(|i| (oracle.dist(q as u32, i), i))
                .collect();
            truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let truth: std::collections::HashSet<u32> =
                truth.iter().take(5).map(|(_, i)| *i).collect();
            recalls += got.intersection(&truth).count() as f32 / 5.0;
        }
        let avg = recalls / 20.0;
        assert!(avg > 0.7, "post-prune recall too low: {avg}");
    }

    #[test]
    fn remap_and_merge() {
        let data = cluster_data(300, 16, 6, 9);
        let oracle = SliceOracle {
            vecs: &data,
            metric: Metric::Cosine,
        };
        let cfg = GraphBuildConfig {
            m: 8,
            m0: 16,
            ef_construction: 64,
            seed: 4,
        };
        let g1 = HnswGraph::build_local(150, &oracle, &cfg);
        let map1: Vec<u32> = (0..150).map(|i| (i * 2) as u32).collect();
        let mut g1 = g1;
        g1.remap(&map1, 300);

        let g2 = HnswGraph::build_local(150, &oracle, &cfg);
        let map2: Vec<u32> = (0..150).map(|i| (i * 2 + 1) as u32).collect();
        let mut g2 = g2;
        g2.remap(&map2, 300);

        g1.merge_union(&g2, &HubPruneConfig::default(), 8);
        assert_eq!(g1.n, 300);
        let linked = g1.adj0.iter().filter(|a| !a.is_empty()).count();
        assert!(linked > 200, "merged graph should connect most nodes, got {linked}");
    }

    #[test]
    fn insert_into_existing() {
        let data = cluster_data(200, 16, 8, 11);
        let oracle = SliceOracle {
            vecs: &data,
            metric: Metric::Cosine,
        };
        let cfg = GraphBuildConfig {
            m: 8,
            m0: 16,
            ef_construction: 64,
            seed: 6,
        };
        let mut g = HnswGraph::build_local(199, &oracle, &cfg);
        // node 199 exists in data but not yet in graph; insert it
        g.levels.push(0);
        g.adj0.push(Vec::new());
        g.upper.push(Vec::new());
        g.n += 1;
        g.insert(199, 0, &oracle, &cfg);
        assert!(!g.adj0[199].is_empty(), "new node must gain neighbors");
        // its neighbors should link back
        let back = g.adj0[199].iter().any(|&x| g.adj0[x as usize].contains(&199));
        assert!(back, "bidirectional links expected");
    }
}
