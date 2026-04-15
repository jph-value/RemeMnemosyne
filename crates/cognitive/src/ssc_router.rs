use chrono::{DateTime, Utc};
use dashmap::DashMap;
use rememnemosyne_core::{cosine_similarity, Importance, PalaceLocation};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Configuration for the Sparse Selective Caching (SSC) router.
///
/// Implements MC's SSC mechanism (arXiv:2602.24281, Section 3.3):
/// a MoE-style router that selects the Top-k most relevant cached
/// segments for efficient aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSCRouterConfig {
    /// Number of segments to select in Top-k routing (default: 5).
    /// This is N in MC's O(NL) complexity — the number of cached
    /// segments the router selects for fine-grained expansion.
    pub top_k: usize,
    /// Minimum relevance score for a segment to be expanded (default: 0.3).
    /// Below this threshold, a checkpoint's summary is used without
    /// expanding into individual memories.
    pub expansion_threshold: f32,
}

impl Default for SSCRouterConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            expansion_threshold: 0.3,
        }
    }
}

/// Profile of a cached segment, used by the SSC router for relevance scoring.
///
/// MC's formulation: given query x_t and segment S^(i), compute
/// relevance r_t^(i) = ⟨u_t, MeanPool(S^(i))⟩ (Eq 16).
/// This struct stores the precomputed MeanPool(S^(i)) so that
/// scoring is a single dot product per segment.
pub struct SegmentProfile {
    pub id: Uuid,
    /// Mean-pooled embedding of all memories in this segment.
    /// Used as a fallback when importance_weighted_embedding is empty.
    pub mean_embedding: Vec<f32>,
    /// Importance-weighted embedding of all memories in this segment.
    /// High-importance memories dominate the fingerprint, preventing
    /// bland checkpoints that all look similar after averaging.
    pub importance_weighted_embedding: Vec<f32>,
    /// Number of memories in this segment.
    pub memory_count: usize,
    /// Palace location of the dominant room in this segment.
    pub palace_location: Option<PalaceLocation>,
    /// When this segment was last accessed by a query.
    pub last_accessed: DateTime<Utc>,
    /// Highest importance level in this segment.
    pub importance_ceiling: Importance,
}

/// Sparse Selective Caching (SSC) router for Memory Caching.
///
/// Implements arXiv:2602.24281, Section 3.3:
/// For each query, the router computes relevance scores against all
/// known segment profiles and selects the Top-k most relevant ones.
/// This provides sub-linear retrieval cost while preserving recall quality.
///
/// The router uses importance_weighted_embedding for scoring by default
/// (more discriminative than plain mean-pooling). Falls back to
/// mean_embedding if the weighted version is unavailable.
pub struct SSCRouter {
    segments: DashMap<Uuid, SegmentProfile>,
    config: SSCRouterConfig,
}

impl SSCRouter {
    pub fn new(config: SSCRouterConfig) -> Self {
        Self {
            segments: DashMap::new(),
            config,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(SSCRouterConfig::default())
    }

    pub fn config(&self) -> &SSCRouterConfig {
        &self.config
    }

    /// Register a checkpoint as a segment profile for SSC routing.
    ///
    /// The checkpoint's summary_embedding is stored as both the mean and
    /// importance-weighted embedding (since it's already weighted at creation
    /// time based on CheckpointEmbeddingMethod).
    pub fn register_checkpoint(&self, checkpoint: &rememnemosyne_core::MemoryCheckpoint) {
        let profile = SegmentProfile {
            id: checkpoint.id,
            mean_embedding: checkpoint.summary_embedding.clone(),
            importance_weighted_embedding: checkpoint.summary_embedding.clone(),
            memory_count: checkpoint.memory_count,
            palace_location: checkpoint.palace_location.clone(),
            last_accessed: Utc::now(),
            importance_ceiling: checkpoint.importance_ceiling,
        };
        self.segments.insert(checkpoint.id, profile);
    }

    /// Remove a segment profile (e.g., when a checkpoint is evicted).
    pub fn deregister(&self, id: &Uuid) {
        self.segments.remove(id);
    }

    /// Score segments by relevance to a query (MC Eq 16).
    ///
    /// r_t^(i) = ⟨u_t, MeanPool(S^(i))⟩
    ///
    /// Uses importance_weighted_embedding for scoring (more discriminative).
    /// Falls back to mean_embedding if weighted version is empty.
    pub fn score_segments(
        &self,
        query_embedding: &[f32],
        candidate_ids: &[Uuid],
    ) -> Vec<(Uuid, f32)> {
        if query_embedding.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(Uuid, f32)> = candidate_ids
            .iter()
            .filter_map(|id| {
                self.segments.get(id).map(|profile| {
                    let emb = if profile.importance_weighted_embedding.is_empty() {
                        &profile.mean_embedding
                    } else {
                        &profile.importance_weighted_embedding
                    };

                    let score = if emb.is_empty() {
                        0.0
                    } else {
                        cosine_similarity(query_embedding, emb)
                    };

                    (*id, score)
                })
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Route a query to the Top-k most relevant segments (MC Eq 17).
    ///
    /// Returns the IDs of the k most relevant segments.
    pub fn route(&self, query_embedding: &[f32], candidate_ids: &[Uuid]) -> Vec<Uuid> {
        self.score_segments(query_embedding, candidate_ids)
            .into_iter()
            .take(self.config.top_k)
            .map(|(id, _)| id)
            .collect()
    }

    /// Route with both IDs and scores (for soft filtering in MemoryRouter).
    pub fn route_with_scores(
        &self,
        query_embedding: &[f32],
        candidate_ids: &[Uuid],
    ) -> Vec<(Uuid, f32)> {
        let routed = self.score_segments(query_embedding, candidate_ids);
        routed.into_iter().take(self.config.top_k).collect()
    }

    /// Get the number of registered segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Check if the router has any segments.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// List all registered segment IDs.
    pub fn list_segment_ids(&self) -> Vec<Uuid> {
        self.segments.iter().map(|entry| *entry.key()).collect()
    }

    /// Update the last_accessed timestamp for a segment (for eviction decisions).
    pub fn mark_accessed(&self, id: &Uuid) {
        if let Some(mut profile) = self.segments.get_mut(id) {
            profile.last_accessed = Utc::now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssc_router_creation() {
        let router = SSCRouter::with_defaults();
        assert!(router.is_empty());
        assert_eq!(router.config().top_k, 5);
    }

    #[test]
    fn test_ssc_router_register_and_route() {
        use rememnemosyne_core::{CheckpointEmbeddingMethod, MemoryCheckpoint};

        let router = SSCRouter::with_defaults();

        let cp1 = MemoryCheckpoint::new(
            Utc::now() - chrono::Duration::minutes(10),
            Utc::now(),
            vec![0.9, 0.1, 0.0], // Similar to this query
            "Test checkpoint 1".to_string(),
            5,
            vec![Uuid::new_v4(); 5],
            CheckpointEmbeddingMethod::MeanPool,
        );

        let cp2 = MemoryCheckpoint::new(
            Utc::now() - chrono::Duration::minutes(20),
            Utc::now() - chrono::Duration::minutes(10),
            vec![0.0, 0.1, 0.9], // Dissimilar to this query
            "Test checkpoint 2".to_string(),
            3,
            vec![Uuid::new_v4(); 3],
            CheckpointEmbeddingMethod::MeanPool,
        );

        router.register_checkpoint(&cp1);
        router.register_checkpoint(&cp2);

        assert_eq!(router.len(), 2);

        let query = vec![0.95, 0.05, 0.0]; // Should match cp1
        let ids = router.list_segment_ids();
        let routed = router.route(&query, &ids);

        assert_eq!(routed.len(), 2); // Both returned since top_k=5
        assert_eq!(routed[0], cp1.id); // cp1 should be first (higher similarity)
    }

    #[test]
    fn test_ssc_router_empty_query() {
        let router = SSCRouter::with_defaults();
        let ids = vec![Uuid::new_v4()];
        let result = router.route(&[], &ids);
        assert!(result.is_empty());
    }
}
