//! Embedding provider abstraction + sharded LRU cache.
//!
//! leann-core never stores raw embeddings — it *recomputes* them through an
//! [`EmbeddingProvider`] on demand. Providers are implemented in
//! `leann-embedder` (OpenAI-compatible HTTP embedders, local hashing fallback)
//! and injected here as trait objects.
//!
//! [`EmbedCache`] is a lock-sharded LRU keyed by chunk id. During a query,
//! recomputed embeddings are cached so that repeated traversal through hub
//! nodes and follow-up queries avoid redundant provider calls.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::format::TextStore;
use crate::metric::normalize;
use crate::metric::Metric;
use crate::Result;

/// Source of chunk texts by id (avoids closure-lifetime gymnastics).
pub trait TextSource {
    fn text(&self, id: u32) -> &str;
}

impl TextSource for TextStore {
    #[inline]
    fn text(&self, id: u32) -> &str {
        TextStore::text(self, id)
    }
}

impl TextSource for Vec<String> {
    #[inline]
    fn text(&self, id: u32) -> &str {
        &self[id as usize]
    }
}

impl<T: TextSource + ?Sized> TextSource for &T {
    #[inline]
    fn text(&self, id: u32) -> &str {
        (*self).text(id)
    }
}

/// Anything that can turn texts into vectors.
pub trait EmbeddingProvider: Send + Sync {
    fn name(&self) -> &str;
    fn dim(&self) -> usize;
    /// Embed a batch of texts. Should batch efficiently internally.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

const SHARDS: usize = 16;

struct Shard {
    map: HashMap<u32, (Arc<Vec<f32>>, u64)>,
    order: VecDeque<u32>,
    cap: usize,
}

/// Lock-sharded LRU cache of embeddings keyed by chunk id.
pub struct EmbedCache {
    shards: Vec<Mutex<Shard>>,
    tick: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl EmbedCache {
    pub fn new(capacity: usize) -> Self {
        let per = (capacity / SHARDS).max(4);
        Self {
            shards: (0..SHARDS)
                .map(|_| {
                    Mutex::new(Shard {
                        map: HashMap::new(),
                        order: VecDeque::new(),
                        cap: per,
                    })
                })
                .collect(),
            tick: AtomicU64::new(1),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn get(&self, id: u32) -> Option<Arc<Vec<f32>>> {
        let t = self.tick.fetch_add(1, Ordering::Relaxed);
        let mut s = self.shards[(id as usize) % SHARDS].lock().unwrap();
        if let Some((v, ts)) = s.map.get_mut(&id) {
            *ts = t;
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(v.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn put(&self, id: u32, vec: Arc<Vec<f32>>) {
        let t = self.tick.fetch_add(1, Ordering::Relaxed);
        let mut s = self.shards[(id as usize) % SHARDS].lock().unwrap();
        if s.map.contains_key(&id) {
            if let Some(e) = s.map.get_mut(&id) {
                *e = (vec, t);
            }
            return;
        }
        if s.order.len() >= s.cap {
            if let Some(evict) = s.order.pop_front() {
                s.map.remove(&evict);
            }
        }
        s.order.push_back(id);
        s.map.insert(id, (vec, t));
    }

    pub fn invalidate(&self, id: u32) {
        let mut s = self.shards[(id as usize) % SHARDS].lock().unwrap();
        s.map.remove(&id);
        s.order.retain(|&x| x != id);
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.hits.load(Ordering::Relaxed), self.misses.load(Ordering::Relaxed))
    }

    pub fn clear(&self) {
        for s in &self.shards {
            let mut s = s.lock().unwrap();
            s.map.clear();
            s.order.clear();
        }
    }
}

/// Provider + cache + metric glue. One per index.
pub struct EmbedSource {
    provider: Arc<dyn EmbeddingProvider>,
    cache: EmbedCache,
    metric: Metric,
}

impl EmbedSource {
    pub fn new(provider: Arc<dyn EmbeddingProvider>, metric: Metric, cache_capacity: usize) -> Self {
        Self {
            provider,
            cache: EmbedCache::new(cache_capacity),
            metric,
        }
    }

    pub fn provider(&self) -> &Arc<dyn EmbeddingProvider> {
        &self.provider
    }

    pub fn cache(&self) -> &EmbedCache {
        &self.cache
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Embed the query text (normalized when the metric requires it).
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let mut v = self.provider.embed_batch(&[query])?.remove(0);
        if self.metric.needs_normalization() {
            normalize(&mut v);
        }
        Ok(v)
    }

    /// Embed fresh texts (build/add paths). Returns normalized vectors.
    pub fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut vs = self.provider.embed_batch(texts)?;
        if self.metric.needs_normalization() {
            for v in vs.iter_mut() {
                normalize(v);
            }
        }
        Ok(vs)
    }

    /// Batch-resolve embeddings for chunk ids, cache-first. Cache misses are
    /// embedded in a single provider batch (this is where LEANN's dynamic
    /// batching pays off: many misses across graph hops collapse into one
    /// provider call).
    pub fn batch_by_id(&self, ids: &[u32], texts: &dyn TextSource) -> Result<Vec<Arc<Vec<f32>>>> {
        let mut out: Vec<Option<Arc<Vec<f32>>>> = Vec::with_capacity(ids.len());
        let mut missing: Vec<(usize, u32)> = Vec::new();
        for &id in ids {
            if let Some(v) = self.cache.get(id) {
                out.push(Some(v));
            } else {
                out.push(None);
                missing.push((out.len() - 1, id));
            }
        }
        if !missing.is_empty() {
            let refs: Vec<&str> = missing.iter().map(|&(_, id)| texts.text(id)).collect();
            let mut vs = self.provider.embed_batch(&refs)?;
            if self.metric.needs_normalization() {
                for v in vs.iter_mut() {
                    normalize(v);
                }
            }
            for ((slot, id), v) in missing.into_iter().zip(vs.into_iter()) {
                let arc = Arc::new(v);
                self.cache.put(id, arc.clone());
                out[slot] = Some(arc);
            }
        }
        Ok(out.into_iter().map(|o| o.expect("filled")).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstProvider;
    impl EmbeddingProvider for ConstProvider {
        fn name(&self) -> &str {
            "const"
        }
        fn dim(&self) -> usize {
            3
        }
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let h = t.len() as f32;
                    vec![h, h * 2.0, h * 3.0]
                })
                .collect())
        }
    }

    #[test]
    fn cache_hit_miss() {
        let c = EmbedCache::new(64);
        assert!(c.get(1).is_none());
        c.put(1, Arc::new(vec![1.0, 2.0]));
        assert!(c.get(1).is_some());
        let (h, m) = c.stats();
        assert_eq!((h, m), (1, 1));
        c.invalidate(1);
        assert!(c.get(1).is_none());
    }

    #[test]
    fn source_batch_by_id() {
        let src = EmbedSource::new(Arc::new(ConstProvider), Metric::Cosine, 32);
        let texts: Vec<String> = ["a", "bb", "ccc"].iter().map(|s| s.to_string()).collect();
        let v1 = src.batch_by_id(&[0, 1, 2], &texts).unwrap();
        assert_eq!(v1.len(), 3);
        // normalized
        let n = crate::metric::dot(&v1[0], &v1[0]);
        assert!((n - 1.0).abs() < 1e-5);
        // second call served from cache
        let v2 = src.batch_by_id(&[0, 1], &texts).unwrap();
        assert_eq!(v2.len(), 2);
        let (h, m) = src.cache().stats();
        assert_eq!(m, 3);
        assert_eq!(h, 2);
    }
}
