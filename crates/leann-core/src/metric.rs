//! Distance metrics.
//!
//! All hot kernels are written as plain `f32` loops over slices so that
//! `rustc` auto-vectorizes them into AVX2/SSE code paths. For cosine we
//! pre-normalize vectors at embed time so the metric collapses to a dot
//! product plus a subtract — a single fused pass.

use serde::{Deserialize, Serialize};

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

/// Dot product. Auto-vectorizes to a single fused-multiply-add pass.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 4];
    let n = a.len();
    let chunks = n / 4;
    let rem = n % 4;
    let (a4, ar) = a.split_at(chunks * 4);
    let (b4, br) = b.split_at(chunks * 4);
    for i in 0..chunks {
        acc[0] += a4[i * 4] * b4[i * 4];
        acc[1] += a4[i * 4 + 1] * b4[i * 4 + 1];
        acc[2] += a4[i * 4 + 2] * b4[i * 4 + 2];
        acc[3] += a4[i * 4 + 3] * b4[i * 4 + 3];
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for i in 0..rem {
        s += ar[i] * br[i];
    }
    s
}

/// Squared L2 distance. Auto-vectorizes similarly.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 4];
    let n = a.len();
    let chunks = n / 4;
    let rem = n % 4;
    let (a4, ar) = a.split_at(chunks * 4);
    let (b4, br) = b.split_at(chunks * 4);
    for i in 0..chunks {
        let d0 = a4[i * 4] - b4[i * 4];
        let d1 = a4[i * 4 + 1] - b4[i * 4 + 1];
        let d2 = a4[i * 4 + 2] - b4[i * 4 + 2];
        let d3 = a4[i * 4 + 3] - b4[i * 4 + 3];
        acc[0] += d0 * d0;
        acc[1] += d1 * d1;
        acc[2] += d2 * d2;
        acc[3] += d3 * d3;
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for i in 0..rem {
        let d = ar[i] - br[i];
        s += d * d;
    }
    s
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
