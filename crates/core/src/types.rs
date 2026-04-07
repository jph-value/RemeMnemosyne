use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for memory artifacts
pub type MemoryId = Uuid;

/// Unique identifier for entities in graph memory
pub type EntityId = Uuid;

/// Unique identifier for memory sessions
pub type SessionId = Uuid;

/// Memory type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    Semantic,
    Episodic,
    Graph,
    Temporal,
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryType::Semantic => write!(f, "semantic"),
            MemoryType::Episodic => write!(f, "episodic"),
            MemoryType::Graph => write!(f, "graph"),
            MemoryType::Temporal => write!(f, "temporal"),
        }
    }
}

/// Importance level for memory prioritization
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub enum Importance {
    Low = 1,
    #[default]
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Trigger types for memory creation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryTrigger {
    UserInput,
    SystemOutput,
    Decision,
    DesignChange,
    Error,
    Insight,
    Question,
    Answer,
    TaskStart,
    TaskComplete,
    Custom(String),
}

/// Entity reference for graph connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRef {
    pub id: EntityId,
    pub name: String,
    pub entity_type: EntityType,
    pub relevance: f32,
}

/// Entity type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Organization,
    Concept,
    Technology,
    Project,
    Location,
    Event,
    Document,
    Code,
    Data,
    Custom(String),
}

/// Relationship type between entities
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationshipType {
    Uses,
    DependsOn,
    References,
    CreatedBy,
    ModifiedBy,
    Contains,
    PartOf,
    Related,
    CausedBy,
    LeadsTo,
    Custom(String),
}

/// The core memory artifact stored across all memory types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryArtifact {
    pub id: MemoryId,
    pub memory_type: MemoryType,
    pub summary: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub entities: Vec<EntityRef>,
    pub timestamp: DateTime<Utc>,
    pub session_id: Option<SessionId>,
    pub importance: Importance,
    pub trigger: MemoryTrigger,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub parent_id: Option<MemoryId>,
    pub access_count: u64,
    pub last_accessed: Option<DateTime<Utc>>,
}

impl MemoryArtifact {
    pub fn new(
        memory_type: MemoryType,
        summary: impl Into<String>,
        content: impl Into<String>,
        embedding: Vec<f32>,
        trigger: MemoryTrigger,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            memory_type,
            summary: summary.into(),
            content: content.into(),
            embedding,
            entities: Vec::new(),
            timestamp: now,
            session_id: None,
            importance: Importance::default(),
            trigger,
            tags: Vec::new(),
            metadata: HashMap::new(),
            parent_id: None,
            access_count: 0,
            last_accessed: None,
        }
    }

    pub fn with_importance(mut self, importance: Importance) -> Self {
        self.importance = importance;
        self
    }

    pub fn with_session(mut self, session_id: SessionId) -> Self {
        self.session_id = Some(session_id);
        self
    }

    pub fn with_entities(mut self, entities: Vec<EntityRef>) -> Self {
        self.entities = entities;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
        self.last_accessed = Some(Utc::now());
    }

    #[inline]
    pub fn compute_relevance(&self) -> f32 {
        let age_hours = (Utc::now() - self.timestamp).num_hours() as f32;

        // Exponential temporal decay: half-life of 168 hours (1 week)
        let temporal_decay = (-age_hours / 168.0_f32).exp();

        // Access frequency boost: log-scaled, capped at 0.5
        let access_boost = if self.access_count > 0 {
            ((self.access_count as f32).ln() * 0.15).min(0.5)
        } else {
            0.0
        };

        // Importance weight: 0.25 per level
        let importance_weight = self.importance as u8 as f32 * 0.25;

        // Recency boost: if accessed within last 24h, extra weight
        let recency_boost = if let Some(last) = self.last_accessed {
            let hours_since_access = (Utc::now() - last).num_hours() as f32;
            if hours_since_access < 24.0 {
                0.1 * (1.0 - hours_since_access / 24.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        (temporal_decay * 0.5 + access_boost + importance_weight + recency_boost).min(1.0)
    }
}

/// Memory event for temporal tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub id: Uuid,
    pub entity_id: EntityId,
    pub memory_id: MemoryId,
    pub event_type: EventType,
    pub timestamp: DateTime<Utc>,
    pub description: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    Created,
    Updated,
    Accessed,
    Related,
    Merged,
    Archived,
    Deleted,
    Custom(String),
}

/// Graph relationship between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub id: Uuid,
    pub source: EntityId,
    pub target: EntityId,
    pub relationship_type: RelationshipType,
    pub strength: f32,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Relationship {
    pub fn new(
        source: EntityId,
        target: EntityId,
        relationship_type: RelationshipType,
        strength: f32,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            source,
            target,
            relationship_type,
            strength,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// Entity in the graph memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub name: String,
    pub entity_type: EntityType,
    pub description: String,
    pub embedding: Vec<f32>,
    pub attributes: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub mention_count: u64,
}

impl Entity {
    pub fn new(
        name: impl Into<String>,
        entity_type: EntityType,
        description: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            entity_type,
            description: description.into(),
            embedding,
            attributes: HashMap::new(),
            created_at: now,
            updated_at: now,
            mention_count: 1,
        }
    }
}

/// Context bundle for LLM consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBundle {
    pub summaries: Vec<String>,
    pub entities: Vec<Entity>,
    pub memories: Vec<MemoryArtifact>,
    pub relationships: Vec<Relationship>,
    pub temporal_events: Vec<MemoryEvent>,
    pub relevance_scores: HashMap<MemoryId, f32>,
    pub total_tokens_estimate: usize,
}

impl ContextBundle {
    pub fn new() -> Self {
        Self {
            summaries: Vec::new(),
            entities: Vec::new(),
            memories: Vec::new(),
            relationships: Vec::new(),
            temporal_events: Vec::new(),
            relevance_scores: HashMap::new(),
            total_tokens_estimate: 0,
        }
    }

    pub fn add_memory(&mut self, memory: MemoryArtifact, relevance: f32) {
        self.total_tokens_estimate += memory.summary.len() / 4; // Rough token estimate
        self.relevance_scores.insert(memory.id, relevance);
        self.memories.push(memory);
    }

    pub fn is_empty(&self) -> bool {
        self.memories.is_empty() && self.entities.is_empty()
    }

    pub fn merge(&mut self, other: ContextBundle) {
        self.summaries.extend(other.summaries);
        self.entities.extend(other.entities);
        self.memories.extend(other.memories);
        self.relationships.extend(other.relationships);
        self.temporal_events.extend(other.temporal_events);
        self.relevance_scores.extend(other.relevance_scores);
        self.total_tokens_estimate += other.total_tokens_estimate;
    }

    pub fn truncate_to_token_limit(&mut self, max_tokens: usize) {
        // Sort by relevance and keep only what fits
        self.memories.sort_by(|a, b| {
            let score_a = self.relevance_scores.get(&a.id).unwrap_or(&0.0);
            let score_b = self.relevance_scores.get(&b.id).unwrap_or(&0.0);
            score_b
                .partial_cmp(score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut current_tokens = 0;
        self.memories.retain(|m| {
            let tokens = m.summary.len() / 4;
            if current_tokens + tokens <= max_tokens {
                current_tokens += tokens;
                true
            } else {
                false
            }
        });

        self.total_tokens_estimate = current_tokens;
    }
}

impl Default for ContextBundle {
    fn default() -> Self {
        Self::new()
    }
}
