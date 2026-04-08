use chrono::{DateTime, Utc};
use rememnemosyne_core::{EntityId, RelationshipType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Temporal validity window for relationships
/// Intelligence data has expiration dates - relationships become stale
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidityWindow {
    /// When this relationship becomes valid
    pub valid_from: DateTime<Utc>,
    /// When this relationship expires (None = no expiration)
    pub valid_until: Option<DateTime<Utc>>,
    /// Whether this relationship has been invalidated
    pub invalidated: bool,
    /// Reason for invalidation (if applicable)
    pub invalidation_reason: Option<String>,
    /// Who invalidated this relationship
    pub invalidated_by: Option<String>,
}

impl ValidityWindow {
    /// Create a new validity window with optional expiration
    pub fn new(valid_from: DateTime<Utc>, valid_until: Option<DateTime<Utc>>) -> Self {
        Self {
            valid_from,
            valid_until,
            invalidated: false,
            invalidation_reason: None,
            invalidated_by: None,
        }
    }

    /// Create with default (now, no expiration)
    pub fn indefinite() -> Self {
        Self::new(Utc::now(), None)
    }

    /// Create with specific expiration
    pub fn expires_at(expiration: DateTime<Utc>) -> Self {
        Self::new(Utc::now(), Some(expiration))
    }

    /// Create with duration from now
    pub fn expires_in(duration: chrono::Duration) -> Self {
        Self::new(Utc::now(), Some(Utc::now() + duration))
    }

    /// Check if currently valid
    pub fn is_valid(&self) -> bool {
        if self.invalidated {
            return false;
        }

        let now = Utc::now();

        // Check valid_from
        if now < self.valid_from {
            return false;
        }

        // Check valid_until
        if let Some(until) = self.valid_until {
            if now > until {
                return false;
            }
        }

        true
    }

    /// Check if expired (but not necessarily invalidated)
    pub fn is_expired(&self) -> bool {
        if let Some(until) = self.valid_until {
            Utc::now() > until
        } else {
            false
        }
    }

    /// Invalidate this relationship
    pub fn invalidate(&mut self, reason: impl Into<String>, by: impl Into<String>) {
        self.invalidated = true;
        self.invalidation_reason = Some(reason.into());
        self.invalidated_by = Some(by.into());
    }

    /// Reactivate this relationship
    pub fn reactivate(&mut self) {
        self.invalidated = false;
        self.invalidation_reason = None;
        self.invalidated_by = None;
    }

    /// Get remaining validity time (if expiring)
    pub fn time_remaining(&self) -> Option<chrono::Duration> {
        self.valid_until.map(|until| until - Utc::now())
    }

    /// Get days until expiration
    pub fn days_until_expiration(&self) -> Option<i64> {
        self.time_remaining().map(|d| d.num_days())
    }
}

/// Extended relationship with temporal tracking and validity windows
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
    // ========================================================================
    // Temporal Validity (mempalace-style temporal graph)
    // ========================================================================
    /// Validity window for this relationship
    pub validity: ValidityWindow,
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
            validity: ValidityWindow::indefinite(),
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

    // ========================================================================
    // Temporal Validity Methods
    // ========================================================================

    /// Set validity window with expiration
    pub fn with_expiration(mut self, duration: chrono::Duration) -> Self {
        self.validity = ValidityWindow::expires_in(duration);
        self
    }

    /// Set specific expiration time
    pub fn with_expiration_at(mut self, expiration: DateTime<Utc>) -> Self {
        self.validity = ValidityWindow::expires_at(expiration);
        self
    }

    /// Set indefinite validity (no expiration)
    pub fn with_indefinite_validity(mut self) -> Self {
        self.validity = ValidityWindow::indefinite();
        self
    }

    /// Check if this relationship is currently valid
    pub fn is_valid(&self) -> bool {
        self.validity.is_valid()
    }

    /// Check if this relationship has expired
    pub fn is_expired(&self) -> bool {
        self.validity.is_expired()
    }

    /// Invalidate this relationship
    pub fn invalidate(&mut self, reason: impl Into<String>, by: impl Into<String>) {
        self.validity.invalidate(reason, by);
        self.last_updated = Utc::now();
    }

    /// Reactivate this relationship
    pub fn reactivate(&mut self) {
        self.validity.reactivate();
        self.last_updated = Utc::now();
    }

    /// Get days until expiration
    pub fn days_until_expiration(&self) -> Option<i64> {
        self.validity.days_until_expiration()
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
                    .cloned()
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
