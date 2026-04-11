/// Memory sharding by entity type
///
/// This module provides sharding capabilities to split memories by entity type,
/// enabling better scalability and parallel processing.
/// Enabled with the `sharding` feature flag.
use dashmap::DashMap;
use parking_lot::RwLock;
use rememnemosyne_core::{
    EntityType, MemoryArtifact, MemoryError, MemoryId, MemoryQuery, MemoryType, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Sharding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingConfig {
    /// Enable sharding
    pub enabled: bool,
    /// Number of shards per entity type
    pub shards_per_type: usize,
    /// Maximum memories per shard
    pub max_memories_per_shard: usize,
    /// Enable automatic rebalancing
    pub auto_rebalance: bool,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            shards_per_type: 4,
            max_memories_per_shard: 10000,
            auto_rebalance: true,
        }
    }
}

/// Shard identifier
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardId {
    pub entity_type: EntityType,
    pub shard_index: usize,
}

impl std::fmt::Display for ShardId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}_{}", self.entity_type, self.shard_index)
    }
}

/// Individual shard
pub struct MemoryShard {
    pub id: ShardId,
    pub memories: HashMap<MemoryId, MemoryArtifact>,
    pub max_capacity: usize,
}

impl MemoryShard {
    /// Create a new shard
    pub fn new(id: ShardId, max_capacity: usize) -> Self {
        Self {
            id,
            memories: HashMap::new(),
            max_capacity,
        }
    }

    /// Check if shard has capacity
    pub fn has_capacity(&self) -> bool {
        self.memories.len() < self.max_capacity
    }

    /// Get memory count
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    /// Check if shard is empty
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }
}

/// Sharded memory store
#[cfg(feature = "sharding")]
pub struct ShardedMemoryStore {
    config: ShardingConfig,
    /// Shards organized by entity type
    shards: DashMap<ShardId, Arc<RwLock<MemoryShard>>>,
    /// Quick lookup from memory ID to shard ID
    memory_to_shard: DashMap<MemoryId, ShardId>,
}

#[cfg(feature = "sharding")]
impl ShardedMemoryStore {
    /// Create a new sharded store
    pub fn new(config: ShardingConfig) -> Self {
        let shards = DashMap::new();

        // Pre-create shards for each entity type
        if config.enabled {
            for entity_type in Self::entity_types() {
                for i in 0..config.shards_per_type {
                    let shard_id = ShardId {
                        entity_type: entity_type.clone(),
                        shard_index: i,
                    };
                    let shard = MemoryShard::new(shard_id.clone(), config.max_memories_per_shard);
                    shards.insert(shard_id, Arc::new(RwLock::new(shard)));
                }
            }
        }

        Self {
            config,
            shards,
            memory_to_shard: DashMap::new(),
        }
    }

    /// Create with default config
    pub fn default_store() -> Self {
        Self::new(ShardingConfig::default())
    }

    /// Store a memory artifact
    pub async fn store(&self, artifact: MemoryArtifact) -> Result<MemoryId> {
        if !self.config.enabled {
            // If sharding disabled, just use a single shard
            return Err(MemoryError::Storage("Sharding not enabled".to_string()));
        }

        let id = artifact.id;
        let memory_type = artifact.memory_type;

        // Determine which shard to use
        let shard_id = self.route_memory(memory_type, &id);

        if let Some(shard) = self.shards.get(&shard_id) {
            let mut shard = shard.write();

            if !shard.has_capacity() {
                // Shard is full
                if self.config.auto_rebalance {
                    // TODO: Implement rebalancing
                    tracing::warn!(shard_id = %shard_id, "Shard full, rebalancing not yet implemented");
                }
                return Err(MemoryError::CapacityExceeded(format!(
                    "Shard {} is full",
                    shard_id
                )));
            }

            shard.memories.insert(id, artifact);
        } else {
            return Err(MemoryError::NotFound(format!(
                "Shard {} not found",
                shard_id
            )));
        }

        // Cache the mapping
        self.memory_to_shard.insert(id, shard_id);

        Ok(id)
    }

    /// Get a memory artifact by ID
    pub async fn get(&self, id: &MemoryId) -> Result<Option<MemoryArtifact>> {
        if let Some(shard_id) = self.memory_to_shard.get(id) {
            if let Some(shard) = self.shards.get(shard_id.value()) {
                let shard = shard.read();
                return Ok(shard.memories.get(id).cloned());
            }
        }
        Ok(None)
    }

    /// Delete a memory artifact
    pub async fn delete(&self, id: &MemoryId) -> Result<bool> {
        if let Some(shard_id) = self.memory_to_shard.get(id) {
            let shard_id = shard_id.value().clone();
            if let Some(shard) = self.shards.get(&shard_id) {
                let mut shard = shard.write();
                if shard.memories.remove(id).is_some() {
                    self.memory_to_shard.remove(id);
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Query memories
    pub async fn query(&self, query: &MemoryQuery) -> Result<Vec<MemoryArtifact>> {
        let mut results = Vec::new();

        // Search across all shards
        for entry in self.shards.iter() {
            let shard = entry.value().read();

            for (_id, artifact) in &shard.memories {
                // Apply filters
                if self.matches_query(artifact, query) {
                    results.push(artifact.clone());
                }
            }
        }

        // Sort by relevance
        results.sort_by(|a, b| {
            let score_a = a.compute_relevance();
            let score_b = b.compute_relevance();
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Get shard statistics
    pub fn get_shard_stats(&self) -> HashMap<ShardId, usize> {
        self.shards
            .iter()
            .map(|entry| {
                let shard = entry.value().read();
                (entry.key().clone(), shard.len())
            })
            .collect()
    }

    /// Get total memory count
    pub fn len(&self) -> usize {
        self.memory_to_shard.len()
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.memory_to_shard.is_empty()
    }

    /// Clear all shards
    pub fn clear(&self) -> Result<()> {
        for entry in self.shards.iter() {
            let mut shard = entry.value().write();
            shard.memories.clear();
        }
        self.memory_to_shard.clear();
        Ok(())
    }

    // Private helper methods

    /// Route a memory to the appropriate shard
    fn route_memory(&self, memory_type: MemoryType, id: &MemoryId) -> ShardId {
        // Use hash to distribute across shards
        let hash = self.hash_memory(id);
        let shard_index = hash % self.config.shards_per_type;

        // Map memory type to entity type for sharding
        let entity_type = self.memory_type_to_entity_type(memory_type);

        ShardId {
            entity_type,
            shard_index,
        }
    }

    /// Hash a memory ID to a shard index
    fn hash_memory(&self, id: &MemoryId) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        hasher.finish() as usize
    }

    /// Map memory type to entity type for sharding
    fn memory_type_to_entity_type(&self, memory_type: MemoryType) -> EntityType {
        match memory_type {
            MemoryType::Semantic => EntityType::Concept,
            MemoryType::Episodic => EntityType::Event,
            MemoryType::Graph => EntityType::Concept,
            MemoryType::Temporal => EntityType::Event,
            MemoryType::EventClassification
            | MemoryType::InfrastructureGap
            | MemoryType::GapDocumentation
            | MemoryType::NarrativeThread
            | MemoryType::EvidenceChain
            | MemoryType::CounterNarrative => EntityType::Concept,
        }
    }

    /// Get all entity types
    fn entity_types() -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Concept,
            EntityType::Technology,
            EntityType::Project,
            EntityType::Location,
            EntityType::Event,
            EntityType::Document,
            EntityType::Code,
            EntityType::Data,
            EntityType::Custom("shard_default".to_string()),
        ]
    }

    /// Check if an artifact matches a query
    fn matches_query(&self, artifact: &MemoryArtifact, query: &MemoryQuery) -> bool {
        // Apply type filter
        if let Some(ref memory_type) = query.memory_type {
            if &artifact.memory_type != memory_type {
                return false;
            }
        }

        // Apply importance filter
        if let Some(min_importance) = query.min_importance {
            if artifact.importance < min_importance {
                return false;
            }
        }

        // Apply tag filter
        if let Some(ref tags) = query.tags {
            if !tags.iter().any(|tag| artifact.tags.contains(tag)) {
                return false;
            }
        }

        // Apply session filter
        if let Some(session_id) = query.session_id {
            if artifact.session_id != Some(session_id) {
                return false;
            }
        }

        true
    }
}

/// Stub implementation when feature is not enabled
#[cfg(not(feature = "sharding"))]
pub struct ShardedMemoryStore;

#[cfg(not(feature = "sharding"))]
impl ShardedMemoryStore {
    pub fn new(_config: ShardingConfig) -> Self {
        Self
    }

    pub fn default_store() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "sharding")]
    #[test]
    fn test_sharded_store_creation() {
        let store = ShardedMemoryStore::default_store();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[cfg(feature = "sharding")]
    #[tokio::test]
    async fn test_sharded_store_store_get() {
        let store = ShardedMemoryStore::default_store();

        let artifact = MemoryArtifact::new(
            MemoryType::Semantic,
            "Test",
            "Test content",
            vec![0.1; 128],
            rememnemosyne_core::MemoryTrigger::UserInput,
        );

        let id = artifact.id;
        store.store(artifact.clone()).await.unwrap();

        let retrieved = store.get(&id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, id);
    }

    #[cfg(not(feature = "sharding"))]
    #[test]
    fn test_sharding_not_enabled() {
        let store = ShardedMemoryStore::default_store();
        // Should just be a stub
    }
}
