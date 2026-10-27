//! Product Quantization codec.
//!
//! Each vector is split into `m` sub-vectors of `dim / m` floats; each
//! sub-vector is encoded as the index (1 byte) of its nearest centroid out of
//! `k = 256` centroids learned per subspace. Encoding a full vector therefore
//! costs `m` bytes — for `dim = 768` and `m = 32` that is a ~96x compression
//! versus the original 3072-byte `f32` vector, matching LEANN's ~100x target.
//!
//! Asymmetric distance computation (ADC): given a query, we precompute a
//! `m x 256` table of squared distances between each query sub-vector and
//! every centroid. The approximate distance of a coded point is then the sum
//! of `m` table lookups — a handful of ns per point.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::metric::{dot, l2_sq, normalize, Metric};
use crate::Result;

/// PQ hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqConfig {
    /// Number of subquantizers (bytes per code).
    pub m: usize,
    /// Number of centroids per subquantizer (fixed to 256 to fit a byte).
    pub ksub: usize,
    /// k-means iterations for codebook training.
    pub train_iters: usize,
    /// Max training samples (training is sampled for speed).
    pub train_samples: usize,
    /// Seed for reproducible training.
    pub seed: u64,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            m: 0, // 0 = auto: ~100x compression of a 4-byte-per-dim f32 vector
            ksub: 256,
            train_iters: 12,
            train_samples: 65_536,
            seed: 42,
        }
    }
}

impl PqConfig {
    /// Auto-pick `m` for a target ~100x compression (`m ~= 4*dim/100`).
    pub fn auto_m(dim: usize) -> usize {
        let m = (dim * 4 + 99) / 100; // ceil
        m.clamp(4, 64).next_power_of_two().min(64)
    }
}

/// A trained product quantizer: `m` codebooks of `ksub x subdim` centroids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqCodec {
    pub m: usize,
    pub ksub: usize,
    pub dim: usize,
    /// subdim per subspace (dim / m, requires dim % m == 0)
    pub subdim: usize,
    /// codebook[j][c] = centroid c of subspace j (len subdim)
    pub codebook: Vec<Vec<f32>>, // flattened as j*ksub*subdim
    #[serde(skip)]
    pub metric: Metric,
}

impl PqCodec {
    /// Train a codec from raw (already normalized if cosine) vectors.
    pub fn train(vectors: &[Vec<f32>], cfg: &PqConfig, metric: Metric) -> Result<Self> {
        if vectors.is_empty() {
            return Err(crate::LeannError::Empty("no vectors to train PQ".into()));
        }
        let dim = vectors[0].len();
        if dim == 0 {
            return Err(crate::LeannError::Invalid("zero-dim vectors".into()));
        }
        let m = if cfg.m == 0 { PqConfig::auto_m(dim) } else { cfg.m };
        if dim % m != 0 {
            // fall back to the largest divisor of dim <= m from powers of two
            let mut mm = m;
            while mm > 1 && dim % mm != 0 {
                mm /= 2;
            }
            if mm == 1 && dim > 1 {
                return Err(crate::LeannError::Invalid(format!(
                    "dim {dim} not divisible by any power-of-two <= {m}"
                )));
            }
            if mm != m {
                log::warn_if_enabled(format_args!(
                    "PQ m adjusted {m} -> {mm} (dim divisibility)"
                ));
            }
            return Self::train_with_m(vectors, mm, cfg, metric);
        }
        Self::train_with_m(vectors, m, cfg, metric)
    }

    fn train_with_m(vectors: &[Vec<f32>], m: usize, cfg: &PqConfig, metric: Metric) -> Result<Self> {
        let dim = vectors[0].len();
        let subdim = dim / m;
        let ksub = cfg.ksub.min(vectors.len().max(1));

        // Deterministic sampling of training vectors.
        let n_train = vectors.len().min(cfg.train_samples.max(ksub * 4));
        let idx: Vec<usize> = if vectors.len() <= n_train {
            (0..vectors.len()).collect()
        } else {
            use rand::seq::SliceRandom;
        use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
            let mut all: Vec<usize> = (0..vectors.len()).collect();
            all.shuffle(&mut rng);
            all.truncate(n_train);
            all
        };

        // Train each subspace in parallel (rayon over subspaces).
        let codebook: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|j| {
                let sub_data: Vec<Vec<f32>> = idx
                    .iter()
                    .map(|&i| vectors[i][j * subdim..(j + 1) * subdim].to_vec())
                    .collect();
                kmeans(
                    &sub_data,
                    ksub,
                    cfg.train_iters,
                    cfg.seed + j as u64,
                    metric.needs_normalization(),
                )
            })
            .collect();

        Ok(Self {
            m,
            ksub,
            dim,
            subdim,
            codebook,
            metric,
        })
    }

    /// Encode one vector into `m` bytes.
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        debug_assert_eq!(v.len(), self.dim);
        let mut code = vec![0u8; self.m];
        for j in 0..self.m {
            let sub = &v[j * self.subdim..(j + 1) * self.subdim];
            let cbslab = &self.codebook[j];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..self.ksub {
                let centroid = &cbslab[c * self.subdim..(c + 1) * self.subdim];
                let d = match self.metric {
                    Metric::L2 => l2_sq(sub, centroid),
                    Metric::Cosine => 1.0 - dot(sub, centroid),
                    Metric::InnerProduct => -dot(sub, centroid),
                };
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            code[j] = best as u8;
        }
        code
    }

    /// Encode a batch in parallel.
    pub fn encode_batch(&self, vs: &[Vec<f32>]) -> Vec<Vec<u8>> {
        vs.par_iter().map(|v| self.encode(v)).collect()
    }

    /// Encode a batch into one flat, contiguous code buffer (`n * m` bytes)
    /// with zero per-vector allocations — the build-time hot path.
    pub fn encode_batch_flat(&self, vs: &[Vec<f32>]) -> Vec<u8> {
        let mut out = vec![0u8; vs.len() * self.m];
        out.par_chunks_mut(self.m)
            .zip(vs.par_iter())
            .for_each(|(dst, v)| {
                debug_assert_eq!(v.len(), self.dim);
                self.encode_into(v, dst);
            });
        out
    }

    /// Encode one vector into a caller-provided `m`-byte buffer (no alloc).
    pub fn encode_into(&self, v: &[f32], out: &mut [u8]) {
        debug_assert_eq!(v.len(), self.dim);
        debug_assert!(out.len() >= self.m);
        for j in 0..self.m {
            let sub = &v[j * self.subdim..(j + 1) * self.subdim];
            let cbslab = &self.codebook[j];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..self.ksub {
                let centroid = &cbslab[c * self.subdim..(c + 1) * self.subdim];
                let d = match self.metric {
                    Metric::L2 => l2_sq(sub, centroid),
                    Metric::Cosine => 1.0 - dot(sub, centroid),
                    Metric::InnerProduct => -dot(sub, centroid),
                };
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            out[j] = best as u8;
        }
    }

    /// Build the ADC lookup table for a query: `table[j * ksub + c]` is the
    /// distance between the query's `j`-th sub-vector and centroid `c`.
    pub fn adc_table(&self, q: &[f32]) -> Vec<f32> {
        debug_assert_eq!(q.len(), self.dim);
        let mut table = vec![0f32; self.m * self.ksub];
        for j in 0..self.m {
            let sub = &q[j * self.subdim..(j + 1) * self.subdim];
            let cbslab = &self.codebook[j];
            let off = j * self.ksub;
            for c in 0..self.ksub {
                let centroid = &cbslab[c * self.subdim..(c + 1) * self.subdim];
                table[off + c] = match self.metric {
                    Metric::L2 => l2_sq(sub, centroid),
                    Metric::Cosine => 1.0 - dot(sub, centroid),
                    Metric::InnerProduct => -dot(sub, centroid),
                };
            }
        }
        table
    }

    /// Approximate distance between a query (via ADC table) and a code.
    #[inline]
    pub fn adc_distance(&self, table: &[f32], code: &[u8]) -> f32 {
        let mut s = 0f32;
        for (j, &c) in code.iter().enumerate().take(self.m) {
            s += table[j * self.ksub + c as usize];
        }
        s
    }

    /// Reconstruct an approximate vector from a code (useful for debugging).
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        for j in 0..self.m {
            let c = code[j] as usize;
            v[j * self.subdim..(j + 1) * self.subdim].copy_from_slice(
                &self.codebook[j][c * self.subdim..(c + 1) * self.subdim],
            );
        }
        v
    }

    /// Serialized byte size of `n` codes.
    pub fn code_bytes(&self, n: usize) -> usize {
        n * self.m
    }
}

/// Plain Lloyd's k-means over `ksub` clusters with k-means++-style init.
fn kmeans(data: &[Vec<f32>], k: usize, iters: usize, seed: u64, normalize_centroids: bool) -> Vec<f32> {
    let n = data.len();
    let d = if n > 0 { data[0].len() } else { return Vec::new() };
    let k = k.min(n).max(1);
    let flat: Vec<f32> = data.iter().flatten().copied().collect();

    // --- init: spread initial centroids across sampled points deterministically
    use rand::seq::SliceRandom;
        use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut rng);
    let mut centroids: Vec<f32> = Vec::with_capacity(k * d);
    for &i in order.iter().take(k) {
        centroids.extend_from_slice(&data[i]);
    }

    let mut assignments = vec![0u32; n];
    for _ in 0..iters {
        // assignment step (parallel over blocks)
        assignments
            .par_chunks_mut(1024)
            .enumerate()
            .for_each(|(blk, asgn)| {
                let base = blk * 1024;
                for (ii, a) in asgn.iter_mut().enumerate() {
                    let i = base + ii;
                    let x = &flat[i * d..(i + 1) * d];
                    let mut best = 0usize;
                    let mut best_d = f32::INFINITY;
                    for c in 0..k {
                        let cd = l2_sq(x, &centroids[c * d..(c + 1) * d]);
                        if cd < best_d {
                            best_d = cd;
                            best = c;
                        }
                    }
                    *a = best as u32;
                }
            });

        // update step: per-cluster sums (parallel with per-thread partials)
        let mut sums = vec![0f32; k * d];
        let mut counts = vec![0u32; k];
        {
            // single-thread accumulation is fine for small data; use rayon fold
            // via chunks to keep it snappy for large training sets
            use std::sync::Mutex;
            let acc = Mutex::new((vec![0f32; k * d], vec![0u32; k]));
            flat.par_chunks(d * 512).enumerate().for_each(|(blk, chunk)| {
                let base = blk * 512;
                let mut local_s = vec![0f32; k * d];
                let mut local_c = vec![0u32; k];
                for (ii, x) in chunk.chunks_exact(d).enumerate() {
                    let i = base + ii;
                    let c = assignments[i] as usize;
                    local_c[c] += 1;
                    for j in 0..d {
                        local_s[c * d + j] += x[j];
                    }
                }
                let mut g = acc.lock().unwrap();
                for (gs, ls) in g.0.iter_mut().zip(local_s.iter()) {
                    *gs += ls;
                }
                for (gc, lc) in g.1.iter_mut().zip(local_c.iter()) {
                    *gc += lc;
                }
            });
            let (s, c) = acc.into_inner().unwrap();
            sums = s;
            counts = c;
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..d {
                    centroids[c * d + j] = sums[c * d + j] * inv;
                }
            }
        }
    }
    // For cosine, centroids live on the unit sphere so ADC tables stay
    // consistent with (normalized) data. For L2 they must remain cluster means.
    if normalize_centroids {
        for c in 0..k {
            normalize(&mut centroids[c * d..(c + 1) * d]);
        }
    }
    centroids
}

/// Convenience: wrap raw codebook bytes for zero-copy serialization.
pub struct PqCodebookView<'a> {
    pub m: usize,
    pub ksub: usize,
    pub subdim: usize,
    pub data: &'a [u8],
}

/// Shared handle type for codecs stored inside an index.
pub type SharedPqCodec = Arc<PqCodec>;

/// Public k-means over whole vectors (used by the sharded-build pipeline to
/// compute shard centroids). Returns `k` centroids.
pub fn kmeans_centroids(
    data: &[Vec<f32>],
    k: usize,
    iters: usize,
    seed: u64,
    normalize_centroids: bool,
) -> Vec<Vec<f32>> {
    if data.is_empty() {
        return Vec::new();
    }
    let d = data[0].len();
    let flat = kmeans(data, k, iters, seed, normalize_centroids);
    flat.chunks(d).map(|c| c.to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect()
    }

    use rand::SeedableRng;

    #[test]
    fn pq_roundtrip_and_recall() {
        let dim = 64;
        let data = gen(2000, dim, 7);
        for v in &data {
            let mut vv = v.clone();
            normalize(&mut vv);
        }
        let cfg = PqConfig {
            m: 8,
            train_iters: 8,
            ..Default::default()
        };
        let codec = PqCodec::train(&data, &cfg, Metric::L2).unwrap();
        assert_eq!(codec.m, 8);
        let codes = codec.encode_batch(&data);

        // ADC distances should rank the true nearest neighbor highly.
        use crate::metric::l2_sq;
        let mut top1_hits = 0;
        let probes = 100;
        for qi in 0..probes {
            let q = &data[qi];
            let table = codec.adc_table(q);
            let mut best_adc = (f32::INFINITY, 0usize);
            for (i, c) in codes.iter().enumerate() {
                if i == qi {
                    continue; // exclude self
                }
                let d = codec.adc_distance(&table, c);
                if d < best_adc.0 {
                    best_adc = (d, i);
                }
            }
            // true top-1 (excluding self)
            let mut truth = (f32::INFINITY, 0usize);
            for (i, v) in data.iter().enumerate() {
                if i == qi {
                    continue;
                }
                let d = l2_sq(q, v);
                if d < truth.0 {
                    truth = (d, i);
                }
            }
            // accept if ADC's best is within true top-10
            let mut ranked: Vec<(f32, usize)> = data
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != qi)
                .map(|(i, v)| (l2_sq(q, v), i))
                .collect();
            ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let top10: Vec<usize> = ranked.iter().take(10).map(|(_, i)| *i).collect();
            if top10.contains(&best_adc.1) || best_adc.1 == truth.1 {
                top1_hits += 1;
            }
        }
        assert!(
            top1_hits as f32 / probes as f32 > 0.6,
            "PQ recall@10-in-top1 too low: {}",
            top1_hits
        );
    }

    #[test]
    fn auto_m_reasonable() {
        assert_eq!(PqConfig::auto_m(768), 32);
        assert_eq!(PqConfig::auto_m(384), 16);
        assert!(PqConfig::auto_m(614) >= 8);
    }
}

// tiny internal logging shim (no log crate dependency)
mod log {
    #[allow(dead_code)]
    pub fn warn_if_enabled(_args: std::fmt::Arguments) {}
}
