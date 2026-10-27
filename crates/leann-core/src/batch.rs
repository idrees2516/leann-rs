//! Dynamic batching for embedding recomputation (paper §4.2).
//!
//! Rather than recomputing embeddings one hop at a time, the searcher
//! aggregates promotion candidates across graph hops into a pending set and
//! flushes it as a single provider batch once it is large enough. The target
//! batch size adapts to the measured provider throughput: if bigger batches
//! improved vectors/second, grow the target; otherwise shrink it. This trades
//! a little staleness in the exact queue for dramatically better batching,
//! which matters most for GPU / HTTP embedding backends.

use std::time::Instant;

/// Adaptive batch-size controller.
#[derive(Debug, Clone)]
pub struct DynamicBatcher {
    pub min: usize,
    pub max: usize,
    target: usize,
    best_rate: f32,
    pub stats: BatcherStats,
}

#[derive(Debug, Clone, Default)]
pub struct BatcherStats {
    pub flushes: usize,
    pub vectors_embedded: usize,
    pub total_embed_secs: f64,
}

impl DynamicBatcher {
    pub fn new(min: usize, max: usize) -> Self {
        let min = min.clamp(1, 1024);
        let max = max.clamp(min, 4096);
        Self {
            min,
            max,
            target: min,
            best_rate: 0.0,
            stats: BatcherStats::default(),
        }
    }

    /// Current target batch size.
    pub fn target(&self) -> usize {
        self.target
    }

    /// Whether a pending set of `pending` items should be flushed now.
    #[inline]
    pub fn should_flush(&self, pending: usize) -> bool {
        pending >= self.target
    }

    /// Record one flush; adapt the target size from the observed rate.
    pub fn record(&mut self, n: usize, elapsed: std::time::Duration) {
        let secs = elapsed.as_secs_f64().max(1e-9);
        self.stats.flushes += 1;
        self.stats.vectors_embedded += n;
        self.stats.total_embed_secs += secs;
        if n == 0 {
            return;
        }
        let rate = n as f32 / secs as f32;
        if rate > self.best_rate * 1.05 {
            // throughput still improving with bigger batches — grow
            self.best_rate = self.best_rate.max(rate);
            self.target = ((self.target as f32 * 1.5) as usize).clamp(self.min, self.max);
        } else if rate < self.best_rate * 0.85 {
            // bigger batches stopped paying off — shrink toward min
            self.target = ((self.target as f32 * 0.75) as usize).clamp(self.min, self.max);
            self.best_rate = self.best_rate.max(rate);
        } else {
            self.best_rate = self.best_rate.max(rate);
        }
        self.target = self.target.clamp(self.min, self.max);
    }

    /// Convenience: time a flush closure.
    pub fn timed_flush<F: FnOnce() -> usize>(&mut self, f: F) -> usize {
        let t0 = Instant::now();
        let n = f();
        self.record(n, t0.elapsed());
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapts_within_bounds() {
        let mut b = DynamicBatcher::new(4, 128);
        assert_eq!(b.target(), 4);
        assert!(b.should_flush(4));
        assert!(!b.should_flush(3));

        // simulate improving rates
        b.record(8, std::time::Duration::from_secs_f64(0.01));
        assert!(b.target() >= 4);
        for _ in 0..20 {
            b.record(64, std::time::Duration::from_secs_f64(0.001));
        }
        assert!(b.target() <= 128);
        assert!(b.target() >= 4);
    }
}
