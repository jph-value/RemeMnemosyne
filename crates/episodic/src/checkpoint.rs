use chrono::{DateTime, Utc};
use dashmap::DashMap;
use rememnemosyne_core::{
    cosine_similarity, max_pool, mean_pool, weighted_mean_pool, CheckpointEmbeddingMethod,
    Importance, MemoryArtifact, MemoryCheckpoint, MemoryId, PalaceLocation, SessionId,
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;
use uuid::Uuid;

/// Configuration for the checkpoint store.
///
/// Controls when checkpoints are created and how many are retained.
/// The dual-trigger mechanism (count + time) ensures checkpoints are
/// created both during high-activity bursts and during steady-state
/// long sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Create a checkpoint every N memories stored (default: 50).
    pub memory_threshold: usize,
    /// Create a checkpoint at least every T seconds (default: 1800 = 30 min).
    pub time_threshold_secs: u64,
    /// Maximum number of checkpoints before eviction (default: 200).
    pub max_checkpoints: usize,
    /// Method for computing checkpoint summary embeddings (default: ImportanceWeightedPool).
    /// ImportanceWeightedPool prevents bland checkpoints that all look similar.
    pub embedding_method: CheckpointEmbeddingMethod,
    /// Minimum cosine similarity to expand a checkpoint's individual memories (default: 0.3).
    /// This is the MC analogue of γ in Eq 9 — the gate threshold for whether
    /// to drill into a checkpoint's window or just use its summary.
    pub expansion_threshold: f32,
    /// Number of checkpoints to retrieve in coarse search (default: 5).
    /// This is MC's Top-k (N in the paper) — the number of cached segments
    /// the SSC router selects for fine-grained expansion.
    pub top_k_routing: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            memory_threshold: 50,
            time_threshold_secs: 1800,
            max_checkpoints: 200,
            embedding_method: CheckpointEmbeddingMethod::ImportanceWeightedPool,
            expansion_threshold: 0.3,
            top_k_routing: 5,
        }
    }
}

/// A compressed checkpoint of a temporal segment of memories.
///
/// Implements the Memory Caching concept from arXiv:2602.24281.
/// When a sequence of memories is segmented into windows, each window's
/// compressed state is cached as a checkpoint. At query time, the SSC
/// router first selects top-k checkpoints (O(N)), then expands relevant
/// checkpoints into individual memories for fine-grained retrieval.
///
/// This provides the MC benefit: effective memory capacity grows with
/// experience without O(L²) cost.
pub struct CheckpointStore {
    checkpoints: DashMap<Uuid, MemoryCheckpoint>,
    /// Ordered list of checkpoint IDs by creation time (oldest first).
    checkpoint_order: RwLock<Vec<Uuid>>,
    /// Number of memories stored since the last checkpoint was created.
    memory_threshold_counter: AtomicUsize,
    /// Timestamp of the last checkpoint creation.
    last_checkpoint_time: RwLock<DateTime<Utc>>,
    config: CheckpointConfig,
}

impl CheckpointStore {
    pub fn new(config: CheckpointConfig) -> Self {
        let now = Utc::now();
        Self {
            checkpoints: DashMap::new(),
            checkpoint_order: RwLock::new(Vec::new()),
            memory_threshold_counter: AtomicUsize::new(0),
            last_checkpoint_time: RwLock::new(now),
            config,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(CheckpointConfig::default())
    }

    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }

    /// Check whether a checkpoint should be created.
    ///
    /// Dual trigger: create checkpoint when either the memory count threshold
    /// OR the time threshold is met. This ensures checkpoints are created
    /// during high-activity bursts (count) and during steady-state
    /// long sessions (time).
    pub fn should_checkpoint(&self, additional_count: usize, now: DateTime<Utc>) -> bool {
        let count_triggered = self.memory_threshold_counter.load(Ordering::Relaxed)
            + additional_count
            >= self.config.memory_threshold;

        let last_time = *self.last_checkpoint_time.read().unwrap();
        let time_triggered =
            (now - last_time).num_seconds() as u64 >= self.config.time_threshold_secs;

        count_triggered || time_triggered
    }

    /// Increment the memory counter. Call this each time a memory is stored.
    pub fn increment_memory_counter(&self) {
        self.memory_threshold_counter
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Reset the memory counter (called after checkpoint creation).
    pub fn reset_memory_counter(&self) {
        self.memory_threshold_counter.store(0, Ordering::Relaxed);
    }

    /// Create a checkpoint from a window of memories.
    ///
    /// Computes the summary embedding using the configured method
    /// (ImportanceWeightedPool by default), extracts key info from
    /// the memories, and stores the checkpoint.
    pub fn create_checkpoint(
        &self,
        memories: &[MemoryArtifact],
        session_id: Option<SessionId>,
    ) -> MemoryCheckpoint {
        if memories.is_empty() {
            // No memories — don't create empty checkpoints
            panic!("Cannot create checkpoint from empty memory window");
        }

        let now = Utc::now();
        let time_window_start = memories.iter().map(|m| m.timestamp).min().unwrap_or(now);
        let time_window_end = memories.iter().map(|m| m.timestamp).max().unwrap_or(now);

        // Compute summary embedding using the configured method
        let summary_embedding = self.compute_summary_embedding(memories);

        // Extract summary text from memory content
        let summary_text = self.compute_summary_text(memories);

        // Collect memory IDs
        let memory_ids: Vec<MemoryId> = memories.iter().map(|m| m.id).collect();

        // Find the highest importance
        let importance_ceiling = memories
            .iter()
            .map(|m| m.importance)
            .max()
            .unwrap_or(Importance::Medium);

        // Determine dominant palace location
        let palace_location = self.dominant_palace_location(memories);

        let mut checkpoint = MemoryCheckpoint::new(
            time_window_start,
            time_window_end,
            summary_embedding,
            summary_text,
            memories.len(),
            memory_ids,
            self.config.embedding_method,
        );

        checkpoint = checkpoint.with_importance_ceiling(importance_ceiling);

        if let Some(location) = palace_location {
            checkpoint = checkpoint.with_palace_location(location);
        }

        if let Some(sid) = session_id {
            checkpoint = checkpoint.with_session(sid);
        }

        // Store the checkpoint
        let id = checkpoint.id;
        self.checkpoints.insert(id, checkpoint.clone());
        self.checkpoint_order.write().unwrap().push(id);

        // Update last checkpoint time
        *self.last_checkpoint_time.write().unwrap() = now;
        self.reset_memory_counter();

        // Evict oldest if exceeding max
        self.evict_if_needed();

        checkpoint
    }

    /// Search checkpoints by query embedding similarity.
    ///
    /// Returns top-k checkpoints sorted by cosine similarity.
    /// This is the coarse O(N) search before fine-grained memory retrieval.
    pub fn search_checkpoints(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Vec<(MemoryCheckpoint, f32)> {
        if query_embedding.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(MemoryCheckpoint, f32)> = self
            .checkpoints
            .iter()
            .filter_map(|entry| {
                let cp = entry.value();
                if cp.summary_embedding.is_empty() {
                    return None;
                }
                let score = cosine_similarity(query_embedding, &cp.summary_embedding);
                Some((cp.clone(), score))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Expand a checkpoint into its constituent memory IDs.
    ///
    /// After coarse checkpoint selection, this returns the individual
    /// memory IDs for fine-grained HNSW retrieval within the checkpoint's
    /// time window.
    pub fn expand_checkpoint(&self, id: Uuid) -> Vec<MemoryId> {
        self.checkpoints
            .get(&id)
            .map(|cp| cp.memory_ids.clone())
            .unwrap_or_default()
    }

    /// Get a checkpoint by ID.
    pub fn get_checkpoint(&self, id: &Uuid) -> Option<MemoryCheckpoint> {
        self.checkpoints.get(id).map(|cp| cp.clone())
    }

    /// List all checkpoint IDs in creation order.
    pub fn list_checkpoint_ids(&self) -> Vec<Uuid> {
        self.checkpoint_order.read().unwrap().clone()
    }

    /// Get the number of stored checkpoints.
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }

    // ========================================================================
    // Private methods
    // ========================================================================

    fn compute_summary_embedding(&self, memories: &[MemoryArtifact]) -> Vec<f32> {
        let embeddings: Vec<Vec<f32>> = memories
            .iter()
            .filter(|m| !m.embedding.is_empty())
            .map(|m| m.embedding.clone())
            .collect();

        if embeddings.is_empty() {
            return Vec::new();
        }

        match self.config.embedding_method {
            CheckpointEmbeddingMethod::MeanPool => mean_pool(&embeddings),
            CheckpointEmbeddingMethod::ImportanceWeightedPool => {
                let weights: Vec<f32> = memories
                    .iter()
                    .filter(|m| !m.embedding.is_empty())
                    .map(|m| m.importance as u8 as f32 * 0.25) // Low=0.25, Med=0.5, High=0.75, Crit=1.0
                    .collect();
                weighted_mean_pool(&embeddings, &weights)
            }
            CheckpointEmbeddingMethod::MaxPool => max_pool(&embeddings),
        }
    }

    fn compute_summary_text(&self, memories: &[MemoryArtifact]) -> String {
        if memories.is_empty() {
            return String::new();
        }

        // Use the highest-importance memory's summary as the base
        let mut sorted = memories.to_vec();
        sorted.sort_by(|a, b| b.importance.cmp(&a.importance));

        let base_summary = sorted[0].summary.clone();
        let tag_list: Vec<String> = sorted
            .iter()
            .take(5)
            .flat_map(|m| m.tags.iter().take(2).cloned())
            .collect();

        format!(
            "{} [{} memories, tags: {}]",
            base_summary,
            memories.len(),
            tag_list.join(", ")
        )
    }

    fn dominant_palace_location(&self, memories: &[MemoryArtifact]) -> Option<PalaceLocation> {
        let locations: Vec<&PalaceLocation> = memories
            .iter()
            .filter_map(|m| m.palace_location.as_ref())
            .collect();

        if locations.is_empty() {
            return None;
        }

        // Return the most frequent location
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for loc in &locations {
            *counts.entry(loc.path()).or_insert(0) += 1;
        }

        let best_path = counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(path, _)| path)?;

        // Reconstruct PalaceLocation from the path
        let parts: Vec<&str> = best_path.split('/').collect();
        let wing = parts.first().unwrap_or(&"").to_string();
        let hall = parts.get(1).unwrap_or(&"").to_string();
        let room = parts.get(2).unwrap_or(&"").to_string();

        Some(PalaceLocation::new(wing, hall, room))
    }

    fn evict_if_needed(&self) {
        if self.len() <= self.config.max_checkpoints {
            return;
        }

        // Evict oldest checkpoints
        let mut order = self.checkpoint_order.write().unwrap();
        while order.len() > self.config.max_checkpoints {
            if let Some(oldest_id) = order.first() {
                self.checkpoints.remove(oldest_id);
                order.remove(0);
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rememnemosyne_core::{MemoryTrigger, MemoryType};

    fn make_memory(id: Uuid, importance: Importance, embedding: Vec<f32>) -> MemoryArtifact {
        let mut m = MemoryArtifact::new(
            MemoryType::Semantic,
            format!("Test memory {}", id),
            format!("Content for {}", id),
            embedding,
            MemoryTrigger::UserInput,
        )
        .with_importance(importance);
        m.id = id;
        m
    }

    #[test]
    fn test_checkpoint_config_defaults() {
        let config = CheckpointConfig::default();
        assert_eq!(config.memory_threshold, 50);
        assert_eq!(config.time_threshold_secs, 1800);
        assert_eq!(config.max_checkpoints, 200);
        assert_eq!(
            config.embedding_method,
            CheckpointEmbeddingMethod::ImportanceWeightedPool
        );
        assert!((config.expansion_threshold - 0.3).abs() < 1e-6);
        assert_eq!(config.top_k_routing, 5);
    }

    #[test]
    fn test_should_checkpoint_count_trigger() {
        let store = CheckpointStore::with_defaults();
        let now = Utc::now();

        // Below threshold
        assert!(!store.should_checkpoint(30, now));

        // At threshold
        assert!(store.should_checkpoint(50, now));

        // Above threshold
        assert!(store.should_checkpoint(100, now));
    }

    #[test]
    fn test_should_checkpoint_time_trigger() {
        let mut store = CheckpointStore::with_defaults();
        let now = Utc::now();

        // Recent last checkpoint — no time trigger
        *store.last_checkpoint_time.write().unwrap() = now;
        assert!(!store.should_checkpoint(0, now));

        // Old last checkpoint — time trigger
        let old_time = now - chrono::Duration::seconds(2000);
        *store.last_checkpoint_time.write().unwrap() = old_time;
        assert!(store.should_checkpoint(0, now));
    }

    #[test]
    fn test_create_checkpoint_basic() {
        let store = CheckpointStore::with_defaults();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let mut m1 = make_memory(id1, Importance::High, vec![0.1, 0.2, 0.3]);
        let mut m2 = make_memory(id2, Importance::Low, vec![0.4, 0.5, 0.6]);
        // Override the auto-generated IDs to match our expected IDs
        m1.id = id1;
        m2.id = id2;

        let memories = vec![m1, m2];

        let checkpoint = store.create_checkpoint(&memories, None);

        assert_eq!(checkpoint.memory_count, 2);
        assert!(checkpoint.memory_ids.contains(&id1));
        assert!(checkpoint.memory_ids.contains(&id2));
        assert_eq!(checkpoint.importance_ceiling, Importance::High);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_search_checkpoints_by_similarity() {
        let store = CheckpointStore::with_defaults();

        let checkpoint_memories_1 = vec![
            make_memory(Uuid::new_v4(), Importance::Medium, vec![0.9, 0.1, 0.0]),
            make_memory(Uuid::new_v4(), Importance::Medium, vec![0.85, 0.15, 0.05]),
        ];

        let checkpoint_memories_2 = vec![
            make_memory(Uuid::new_v4(), Importance::Medium, vec![0.0, 0.1, 0.9]),
            make_memory(Uuid::new_v4(), Importance::Medium, vec![0.05, 0.15, 0.85]),
        ];

        store.create_checkpoint(&checkpoint_memories_1, None);
        store.create_checkpoint(&checkpoint_memories_2, None);

        // Query similar to checkpoint 1
        let query = vec![0.95, 0.05, 0.0];
        let results = store.search_checkpoints(&query, 2);

        assert_eq!(results.len(), 2);
        assert!(results[0].1 > results[1].1); // First result closer to query
    }

    #[test]
    fn test_expand_checkpoint() {
        let store = CheckpointStore::with_defaults();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let memories = vec![
            make_memory(id1, Importance::Medium, vec![0.1, 0.2]),
            make_memory(id2, Importance::Medium, vec![0.3, 0.4]),
        ];

        let checkpoint = store.create_checkpoint(&memories, None);
        let expanded = store.expand_checkpoint(checkpoint.id);

        assert_eq!(expanded.len(), 2);
        assert!(expanded.contains(&id1));
        assert!(expanded.contains(&id2));
    }

    #[test]
    fn test_eviction() {
        let config = CheckpointConfig {
            max_checkpoints: 3,
            ..Default::default()
        };
        let store = CheckpointStore::new(config);

        for i in 0..5 {
            let memories = vec![make_memory(
                Uuid::new_v4(),
                Importance::Medium,
                vec![0.1 * i as f32, 0.2],
            )];
            store.create_checkpoint(&memories, None);
        }

        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_embedding_methods_produce_different_results() {
        let memories = vec![
            make_memory(Uuid::new_v4(), Importance::High, vec![1.0, 0.0]),
            make_memory(Uuid::new_v4(), Importance::Low, vec![0.0, 1.0]),
        ];

        let config_mean = CheckpointConfig {
            embedding_method: CheckpointEmbeddingMethod::MeanPool,
            ..Default::default()
        };
        let config_weighted = CheckpointConfig {
            embedding_method: CheckpointEmbeddingMethod::ImportanceWeightedPool,
            ..Default::default()
        };
        let config_max = CheckpointConfig {
            embedding_method: CheckpointEmbeddingMethod::MaxPool,
            ..Default::default()
        };

        let store_mean = CheckpointStore::new(config_mean);
        let store_weighted = CheckpointStore::new(config_weighted);
        let store_max = CheckpointStore::new(config_max);

        let cp_mean = store_mean.create_checkpoint(&memories, None);
        let cp_weighted = store_weighted.create_checkpoint(&memories, None);
        let cp_max = store_max.create_checkpoint(&memories, None);

        // All should produce different embeddings
        assert_ne!(cp_mean.summary_embedding, cp_weighted.summary_embedding);
        assert_ne!(cp_mean.summary_embedding, cp_max.summary_embedding);
        assert_ne!(cp_weighted.summary_embedding, cp_max.summary_embedding);

        // Importance-weighted should lean toward the High-importance memory
        assert!(cp_weighted.summary_embedding[0] > cp_mean.summary_embedding[0]);
    }
}
