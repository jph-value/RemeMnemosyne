use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::{EntityId, Importance, MemoryId, MemoryType};

/// Memory query for retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuery {
    /// Text query for semantic search
    pub text: Option<String>,

    /// Embedding vector for similarity search
    pub embedding: Option<Vec<f32>>,

    /// Filter by memory type
    pub memory_type: Option<MemoryType>,

    /// Filter by minimum importance
    pub min_importance: Option<Importance>,

    /// Filter by time range
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,

    /// Filter by session ID
    pub session_id: Option<uuid::Uuid>,

    /// Filter by tags
    pub tags: Option<Vec<String>>,

    /// Filter by entity IDs
    pub entity_ids: Option<Vec<EntityId>>,

    /// Filter by memory IDs
    pub memory_ids: Option<Vec<MemoryId>>,

    /// Search limit
    pub limit: Option<usize>,

    /// Minimum relevance score
    pub min_relevance: Option<f32>,

    /// Custom filters
    pub filters: HashMap<String, serde_json::Value>,

    // RISC.OSINT namespace filtering
    /// Filter by agent namespace (e.g. "categorizer", "narrative")
    pub namespace: Option<String>,
    /// Minimum confidence threshold
    pub min_confidence: Option<f32>,
    /// Filter by agent ID
    pub agent_id: Option<String>,
    /// Filter by tier
    pub tier: Option<u8>,
}

impl MemoryQuery {
    pub fn new() -> Self {
        Self {
            text: None,
            embedding: None,
            memory_type: None,
            min_importance: None,
            time_range: None,
            session_id: None,
            tags: None,
            entity_ids: None,
            memory_ids: None,
            limit: None,
            min_relevance: None,
            filters: HashMap::new(),
            namespace: None,
            min_confidence: None,
            agent_id: None,
            tier: None,
        }
    }

    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = Some(memory_type);
        self
    }

    pub fn with_importance(mut self, importance: Importance) -> Self {
        self.min_importance = Some(importance);
        self
    }

    pub fn with_time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.time_range = Some((start, end));
        self
    }

    pub fn with_session(mut self, session_id: uuid::Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn with_min_relevance(mut self, relevance: f32) -> Self {
        self.min_relevance = Some(relevance);
        self
    }

    pub fn with_filter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.filters.insert(key.into(), value);
        self
    }

    /// RISC.OSINT: Filter by agent namespace
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// RISC.OSINT: Filter by minimum confidence
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence = Some(confidence);
        self
    }

    /// RISC.OSINT: Filter by agent ID
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// RISC.OSINT: Filter by tier
    pub fn with_tier(mut self, tier: u8) -> Self {
        self.tier = Some(tier);
        self
    }
}

impl Default for MemoryQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph query for entity traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQuery {
    pub start_entity: Option<EntityId>,
    pub max_depth: usize,
    pub relationship_types: Option<Vec<String>>,
    pub entity_types: Option<Vec<String>>,
    pub limit: usize,
}

impl Default for GraphQuery {
    fn default() -> Self {
        Self {
            start_entity: None,
            max_depth: 2,
            relationship_types: None,
            entity_types: None,
            limit: 50,
        }
    }
}

/// Temporal query for event retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalQuery {
    pub entity_id: Option<EntityId>,
    pub memory_id: Option<MemoryId>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub event_types: Option<Vec<String>>,
    pub limit: usize,
}

impl Default for TemporalQuery {
    fn default() -> Self {
        Self {
            entity_id: None,
            memory_id: None,
            start_time: None,
            end_time: None,
            event_types: None,
            limit: 100,
        }
    }
}
