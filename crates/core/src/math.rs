/// Shared vector math utilities for Memory Caching (MC) operations.
///
/// These are used across crates for cosine similarity, pooling operations,
/// and softmax normalization — the core numerical primitives behind MC's
/// Gated Residual Memory, Sparse Selective Caching, and checkpoint
/// embedding computation.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn softmax(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_sum: f32 = scores.iter().map(|s| (s - max_score).exp()).sum();
    if exp_sum < 1e-10 {
        let uniform = 1.0 / scores.len() as f32;
        scores.iter_mut().for_each(|s| *s = uniform);
        return;
    }
    for s in scores.iter_mut() {
        *s = (*s - max_score).exp() / exp_sum;
    }
}

pub fn mean_pool(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let dims = vectors[0].len();
    let mut result = vec![0.0f32; dims];
    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            if i < dims {
                result[i] += val;
            }
        }
    }
    let count = vectors.len() as f32;
    for val in result.iter_mut() {
        *val /= count;
    }
    result
}

pub fn weighted_mean_pool(vectors: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
    if vectors.is_empty() || vectors.len() != weights.len() {
        return mean_pool(vectors);
    }
    let dims = vectors[0].len();
    let mut result = vec![0.0f32; dims];
    let mut total_weight = 0.0f32;
    for (v, &w) in vectors.iter().zip(weights.iter()) {
        total_weight += w;
        for (i, &val) in v.iter().enumerate() {
            if i < dims {
                result[i] += val * w;
            }
        }
    }
    if total_weight < 1e-10 {
        return mean_pool(vectors);
    }
    for val in result.iter_mut() {
        *val /= total_weight;
    }
    result
}

pub fn max_pool(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let dims = vectors[0].len();
    let mut result = vec![f32::NEG_INFINITY; dims];
    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            if i < dims && val > result[i] {
                result[i] = val;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched_dims() {
        let a = vec![1.0];
        let b = vec![1.0, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_mean_pool_basic() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = mean_pool(&vectors);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_empty() {
        let result: Vec<Vec<f32>> = Vec::new();
        let pooled = mean_pool(&result);
        assert!(pooled.is_empty());
    }

    #[test]
    fn test_weighted_mean_pool() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = vec![0.75, 0.25];
        let result = weighted_mean_pool(&vectors, &weights);
        assert_eq!(result.len(), 2);
        let expected_0 = (1.0 * 0.75 + 3.0 * 0.25) / 1.0;
        let expected_1 = (2.0 * 0.75 + 4.0 * 0.25) / 1.0;
        assert!((result[0] - expected_0).abs() < 1e-6);
        assert!((result[1] - expected_1).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_mean_pool_fallback_on_mismatch() {
        let vectors = vec![vec![1.0, 2.0]];
        let weights = vec![1.0, 2.0]; // mismatched lengths
        let result = weighted_mean_pool(&vectors, &weights);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_pool_basic() {
        let vectors = vec![vec![1.0, -2.0], vec![3.0, 0.5]];
        let result = max_pool(&vectors);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let mut scores = vec![1.0, 2.0, 3.0];
        softmax(&mut scores);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(scores[2] > scores[1]);
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let mut scores: Vec<f32> = Vec::new();
        softmax(&mut scores);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_softmax_all_equal() {
        let mut scores = vec![2.0, 2.0, 2.0];
        softmax(&mut scores);
        for &s in &scores {
            assert!((s - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        let norm = l2_norm(&v);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dp = dot_product(&a, &b);
        assert!((dp - 32.0).abs() < 1e-6);
    }
}
