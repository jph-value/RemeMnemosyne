use chrono::{DateTime, Utc};
use rememnemosyne_core::{EntityId, EntityType, MemoryId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Extended entity for graph memory with rich attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEntity {
    pub id: EntityId,
    pub name: String,
    pub entity_type: EntityType,
    pub description: String,
    pub embedding: Vec<f32>,
    pub attributes: HashMap<String, serde_json::Value>,
    pub aliases: Vec<String>,
    pub memory_ids: Vec<MemoryId>,
    pub mention_count: u64,
    pub importance_score: f32,
    pub centrality_score: f32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl GraphEntity {
    pub fn new(
        name: impl Into<String>,
        entity_type: EntityType,
        description: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: EntityId::new_v4(),
            name: name.into(),
            entity_type,
            description: description.into(),
            embedding,
            attributes: HashMap::new(),
            aliases: Vec::new(),
            memory_ids: Vec::new(),
            mention_count: 1,
            importance_score: 0.5,
            centrality_score: 0.0,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }

    pub fn with_attribute(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.attributes.insert(key.into(), value);
        self
    }

    pub fn with_memory(mut self, memory_id: MemoryId) -> Self {
        self.memory_ids.push(memory_id);
        self
    }

    pub fn increment_mention(&mut self) {
        self.mention_count += 1;
        self.updated_at = Utc::now();
        // Recalculate importance
        self.importance_score = self.compute_importance();
    }

    pub fn compute_importance(&self) -> f32 {
        let mention_factor = (self.mention_count as f32).log10() * 0.2;
        let connection_factor = self.centrality_score * 0.3;
        let type_factor = match self.entity_type {
            EntityType::Person | EntityType::Organization => 0.2,
            EntityType::Project | EntityType::Technology => 0.15,
            _ => 0.1,
        };

        (mention_factor + connection_factor + type_factor).min(1.0)
    }

    pub fn matches_name(&self, query: &str) -> bool {
        let query_lower = query.to_lowercase();
        self.name.to_lowercase().contains(&query_lower)
            || self
                .aliases
                .iter()
                .any(|a| a.to_lowercase().contains(&query_lower))
    }

    pub fn similarity(&self, other: &GraphEntity) -> f32 {
        if self.embedding.is_empty() || other.embedding.is_empty() {
            return 0.0;
        }

        let dot: f32 = self
            .embedding
            .iter()
            .zip(other.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = self.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

/// Entity cluster for grouping related entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityCluster {
    pub id: uuid::Uuid,
    pub name: String,
    pub entities: Vec<EntityId>,
    pub centroid: Vec<f32>,
    pub coherence: f32,
    pub created_at: DateTime<Utc>,
}

impl EntityCluster {
    pub fn new(name: impl Into<String>, entities: Vec<EntityId>, centroid: Vec<f32>) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            name: name.into(),
            entities,
            centroid,
            coherence: 0.0,
            created_at: Utc::now(),
        }
    }

    pub fn compute_coherence(&mut self, entity_map: &HashMap<EntityId, GraphEntity>) {
        if self.entities.len() < 2 {
            self.coherence = 1.0;
            return;
        }

        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..self.entities.len() {
            for j in (i + 1)..self.entities.len() {
                if let (Some(e1), Some(e2)) = (
                    entity_map.get(&self.entities[i]),
                    entity_map.get(&self.entities[j]),
                ) {
                    total_sim += e1.similarity(e2);
                    count += 1;
                }
            }
        }

        self.coherence = if count > 0 {
            total_sim / count as f32
        } else {
            0.0
        };
    }
}
