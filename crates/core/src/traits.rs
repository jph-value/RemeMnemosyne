use async_trait::async_trait;

use crate::types::*;
use crate::error::Result;
use crate::query::*;

/// Core trait that all memory stores must implement
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Store a memory artifact and return its ID
    async fn store(&self, artifact: MemoryArtifact) -> Result<MemoryId>;

    /// Retrieve a memory by ID
    async fn get(&self, id: &MemoryId) -> Result<Option<MemoryArtifact>>;

    /// Query memories based on a query
    async fn query(&self, query: &MemoryQuery) -> Result<Vec<MemoryArtifact>>;

    /// Delete a memory by ID
    async fn delete(&self, id: &MemoryId) -> Result<bool>;

    /// Update a memory artifact
    async fn update(&self, artifact: MemoryArtifact) -> Result<()>;

    /// Get the count of stored memories
    async fn count(&self) -> Result<usize>;

    /// Clear all memories
    async fn clear(&self) -> Result<()>;

    /// Get all memory IDs
    async fn list_ids(&self) -> Result<Vec<MemoryId>>;
}

/// Trait for vector-based memory stores with quantization
#[async_trait]
pub trait VectorMemoryStore: MemoryStore {
    /// Store with pre-computed embedding
    async fn store_with_embedding(
        &self,
        artifact: MemoryArtifact,
        embedding: Vec<f32>,
    ) -> Result<MemoryId>;

    /// Search by vector similarity
    async fn search_similar(
        &self,
        query_vector: &[f32],
        k: usize,
        threshold: f32,
    ) -> Result<Vec<(MemoryArtifact, f32)>>;

    /// Search with quantized vectors (faster, approximate)
    async fn search_quantized(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<(MemoryArtifact, f32)>>;

    /// Get the quantizer configuration
    fn quantizer_config(&self) -> QuantizerConfig;
}

/// Trait for graph-based memory
#[async_trait]
pub trait GraphMemoryStore: Send + Sync {
    /// Add an entity to the graph
    async fn add_entity(&self, entity: Entity) -> Result<EntityId>;

    /// Get an entity by ID
    async fn get_entity(&self, id: &EntityId) -> Result<Option<Entity>>;

    /// Add a relationship between entities
    async fn add_relationship(&self, relationship: Relationship) -> Result<Uuid>;

    /// Find entities related to a given entity
    async fn find_related(
        &self,
        entity_id: &EntityId,
        max_depth: usize,
    ) -> Result<Vec<(Entity, RelationshipType, f32)>>;

    /// Find entities by name or description
    async fn search_entities(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Entity>>;

    /// Get the entity graph as adjacency list
    async fn get_adjacency(&self, entity_id: &EntityId) -> Result<Vec<(EntityId, RelationshipType)>>;
}

/// Trait for temporal memory operations
#[async_trait]
pub trait TemporalMemoryStore: Send + Sync {
    /// Record a memory event
    async fn record_event(&self, event: MemoryEvent) -> Result<Uuid>;

    /// Get events for an entity within a time range
    async fn get_events_for_entity(
        &self,
        entity_id: &EntityId,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<Vec<MemoryEvent>>;

    /// Get events for a memory artifact
    async fn get_events_for_memory(
        &self,
        memory_id: &MemoryId,
    ) -> Result<Vec<MemoryEvent>>;

    /// Get timeline of events
    async fn get_timeline(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<MemoryEvent>>;
}

/// Trait for cognitive/micro-embedding operations
#[async_trait]
pub trait CognitiveEngine: Send + Sync {
    /// Generate micro-embedding for text
    async fn micro_embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Detect intent from text
    async fn detect_intent(&self, text: &str) -> Result<Vec<(String, f32)>>;

    /// Extract entities from text
    async fn extract_entities(&self, text: &str) -> Result<Vec<EntityRef>>;

    /// Predict relevant memories based on current context
    async fn predict_relevance(
        &self,
        context: &[String],
        candidate_ids: &[MemoryId],
    ) -> Result<Vec<(MemoryId, f32)>>;

    /// Prefetch memories based on query
    async fn prefetch(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryId>>;
}

/// Trait for context building
#[async_trait]
pub trait ContextBuilder: Send + Sync {
    /// Build context bundle from query
    async fn build_context(
        &self,
        query: &str,
        max_tokens: usize,
    ) -> Result<ContextBundle>;

    /// Prune context to fit token limit
    fn prune_context(&self, bundle: &mut ContextBundle, max_tokens: usize);

    /// Merge multiple context bundles
    fn merge_contexts(&self, bundles: Vec<ContextBundle>) -> ContextBundle;
}

/// Trait for storage persistence
#[async_trait]
pub trait PersistentStorage: Send + Sync {
    /// Save state to disk
    async fn save(&self, path: &std::path::Path) -> Result<()>;

    /// Load state from disk
    async fn load(&self, path: &std::path::Path) -> Result<()>;

    /// Create a snapshot
    async fn snapshot(&self, name: &str) -> Result<()>;

    /// Restore from snapshot
    async fn restore_snapshot(&self, name: &str) -> Result<()>;

    /// List available snapshots
    async fn list_snapshots(&self) -> Result<Vec<String>>;
}

/// Configuration for quantizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizerConfig {
    pub dimensions: usize,
    pub bits: u8,
    pub subquantizers: usize,
    pub seed: u64,
}

use chrono::DateTime;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
