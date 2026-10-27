//! Distance metrics.
//!
//! All kernels dispatch through [`crate::simd`], which picks the best
//! CPU-supported implementation (AVX-512F / AVX2+FMA / SSE4.1 / portable)
//! once per process. For cosine we pre-normalize vectors at embed time so
//! the metric collapses to a dot product plus a subtract — a single fused
//! pass.

use serde::{Deserialize, Serialize};

pub use crate::simd::{l2_sq, dot};

/// Supported distance metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    /// Squared Euclidean distance. (`dist(a,b) = ||a-b||^2`)
    #[default]
    L2,
    /// Cosine distance (`1 - cos`). Vectors are normalized up front so this
    /// is computed as `1 - dot(a,b)`.
    Cosine,
    /// Inner product distance (`-dot(a,b)`), for max-similarity retrieval.
    InnerProduct,
}

impl Metric {
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => l2_sq(a, b),
            Metric::Cosine => 1.0 - dot(a, b),
            Metric::InnerProduct => -dot(a, b),
        }
    }

    /// Whether vectors should be L2-normalized before use (cosine only).
    #[inline]
    pub fn needs_normalization(self) -> bool {
        matches!(self, Metric::Cosine)
    }
}

/// L2-normalize a vector in place. Zero vectors are left untouched.
#[inline]
pub fn normalize(v: &mut [f32]) {
    let n2 = dot(v, v).max(1e-30);
    if n2 <= 1e-30 || (n2 - 1.0).abs() < 1e-6 {
        return;
    }
    let inv = 1.0 / n2.sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_basic() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        assert!((l2_sq(&a, &b) - 2.0).abs() < 1e-6);
        assert!(dot(&a, &a) - 1.0 < 1e-6);
        assert!((Metric::Cosine.distance(&a, &b) - 1.0).abs() < 1e-6);
        assert!(Metric::Cosine.distance(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn normalize_works() {
        let mut v = vec![3.0f32, 4.0];
        normalize(&mut v);
        assert!((dot(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn odd_lengths() {
        let a: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..7).map(|i| (i + 1) as f32).collect();
        let d = dot(&a, &b);
        let expected: f32 = (0..7).map(|i| i as f32 * (i + 1) as f32).sum();
        assert!((d - expected).abs() < 1e-4);
    }
}
