use chrono::{DateTime, Utc};
use mnemosyne_core::{EntityId, RelationshipType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Extended relationship with temporal tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRelationship {
    pub id: Uuid,
    pub source: EntityId,
    pub target: EntityId,
    pub relationship_type: RelationshipType,
    pub strength: f32,
    pub confidence: f32,
    pub evidence: Vec<RelationshipEvidence>,
    pub first_seen: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GraphRelationship {
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
            confidence: 0.5,
            evidence: Vec::new(),
            first_seen: now,
            last_updated: now,
            metadata: HashMap::new(),
        }
    }

    pub fn with_evidence(mut self, evidence: RelationshipEvidence) -> Self {
        self.evidence.push(evidence);
        self.update_confidence();
        self
    }

    pub fn strengthen(&mut self, amount: f32) {
        self.strength = (self.strength + amount).min(1.0);
        self.last_updated = Utc::now();
    }

    pub fn weaken(&mut self, amount: f32) {
        self.strength = (self.strength - amount).max(0.0);
        self.last_updated = Utc::now();
    }

    fn update_confidence(&mut self) {
        // Confidence based on evidence count and recency
        let evidence_count = self.evidence.len() as f32;
        let recency_factor = if let Some(latest) = self.evidence.last() {
            let days_old = (Utc::now() - latest.timestamp).num_days() as f32;
            1.0 / (1.0 + days_old / 30.0)
        } else {
            0.5
        };

        self.confidence = ((evidence_count * 0.2 + recency_factor) / 2.0).min(1.0);
    }

    pub fn is_bidirectional(&self) -> bool {
        matches!(
            self.relationship_type,
            RelationshipType::Related | RelationshipType::Contains | RelationshipType::PartOf
        )
    }
}

/// Evidence supporting a relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipEvidence {
    pub id: Uuid,
    pub source: EvidenceSource,
    pub description: String,
    pub timestamp: DateTime<Utc>,
    pub confidence: f32,
}

impl RelationshipEvidence {
    pub fn new(source: EvidenceSource, description: impl Into<String>, confidence: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            source,
            description: description.into(),
            timestamp: Utc::now(),
            confidence,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceSource {
    ExplicitMention,
    Inference,
    External,
    UserConfirmation,
    PatternMatch,
}

/// Relationship path between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipPath {
    pub entities: Vec<EntityId>,
    pub relationships: Vec<GraphRelationship>,
    pub total_strength: f32,
    pub hop_count: usize,
}

impl RelationshipPath {
    pub fn new(entities: Vec<EntityId>, relationships: Vec<GraphRelationship>) -> Self {
        let total_strength = relationships.iter().map(|r| r.strength).sum::<f32>()
            / relationships.len().max(1) as f32;
        let hop_count = relationships.len();

        Self {
            entities,
            relationships,
            total_strength,
            hop_count,
        }
    }

    pub fn start(&self) -> Option<&EntityId> {
        self.entities.first()
    }

    pub fn end(&self) -> Option<&EntityId> {
        self.entities.last()
    }

    pub fn description(&self, entity_names: &HashMap<EntityId, String>) -> String {
        let names: Vec<String> = self
            .entities
            .iter()
            .map(|id| {
                entity_names
                    .get(id)
                    .map(|s| s.clone())
                    .unwrap_or_else(|| "unknown".to_string())
            })
            .collect();

        let rel_types: Vec<String> = self
            .relationships
            .iter()
            .map(|r| format!("{:?}", r.relationship_type))
            .collect();

        let mut parts = Vec::new();
        for i in 0..names.len().saturating_sub(1) {
            parts.push(names[i].to_string());
            if i < rel_types.len() {
                parts.push(format!(" --[{}]--> ", rel_types[i]));
            }
        }
        if let Some(last) = names.last() {
            parts.push(last.to_string());
        }

        parts.join("")
    }
}
