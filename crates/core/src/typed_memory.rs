/// Typed Intelligence Memory
///
/// This module provides specialized memory types for intelligence analysis,
/// going beyond generic notes to support structured intelligence memory.
///
/// Memory types:
/// - EventMemory - Discrete events with temporal/spatial context
/// - NarrativeMemory - Connected storylines and evolving narratives  
/// - RiskNodeMemory - Risk entities with threat assessments
/// - EvidenceMemory - Evidence with source attribution
/// - SimulationMemory - Scenario simulations and projections
///
/// This enables RISC-OSINT and other intelligence platforms to use
/// Mnemosyne as a proper intelligence memory system.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::{EntityId, Importance, MemoryId};

// ============================================================================
// Base Typed Memory
// ============================================================================

/// Base typed memory that all specialized memories extend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedMemoryBase {
    /// Unique identifier
    pub id: MemoryId,
    /// Memory type
    pub memory_type: IntelligenceMemoryType,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Importance level
    pub importance: Importance,
    /// Creation timestamp
    pub timestamp: DateTime<Utc>,
    /// Last accessed
    pub last_accessed: Option<DateTime<Utc>>,
    /// Access count
    pub access_count: u64,
    /// Linked entities
    pub entity_links: Vec<EntityId>,
    /// Source of this memory
    pub source: Option<String>,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Custom metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Tags
    pub tags: Vec<String>,
}

impl TypedMemoryBase {
    pub fn new(memory_type: IntelligenceMemoryType, embedding: Vec<f32>) -> Self {
        Self {
            id: MemoryId::new_v4(),
            memory_type,
            embedding,
            importance: Importance::Medium,
            timestamp: Utc::now(),
            last_accessed: None,
            access_count: 0,
            entity_links: Vec::new(),
            source: None,
            confidence: 0.5,
            metadata: HashMap::new(),
            tags: Vec::new(),
        }
    }

    /// Mark this memory as accessed
    pub fn mark_accessed(&mut self) {
        self.last_accessed = Some(Utc::now());
        self.access_count += 1;
    }

    /// Add entity link
    pub fn with_entity_link(mut self, entity_id: EntityId) -> Self {
        self.entity_links.push(entity_id);
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

// ============================================================================
// Intelligence Memory Types
// ============================================================================

/// Types of intelligence memories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntelligenceMemoryType {
    /// Event memory - discrete occurrences
    Event,
    /// Narrative memory - connected storylines
    Narrative,
    /// Risk node memory - threat/risk entities
    RiskNode,
    /// Evidence memory - attributed evidence
    Evidence,
    /// Simulation memory - scenario projections
    Simulation,
    /// Custom memory type
    Custom(String),
}

// ============================================================================
// EventMemory
// ============================================================================

/// Event memory for discrete occurrences with temporal/spatial context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMemory {
    /// Base memory fields
    pub base: TypedMemoryBase,
    /// Event title
    pub title: String,
    /// Event description
    pub description: String,
    /// Event timestamp (when it occurred)
    pub event_timestamp: DateTime<Utc>,
    /// Location (if applicable)
    pub location: Option<String>,
    /// Entities involved
    pub involved_entities: Vec<EntityId>,
    /// Event category
    pub category: Option<String>,
    /// Severity (1-10)
    pub severity: Option<u8>,
    /// Related events
    pub related_events: Vec<MemoryId>,
}

impl EventMemory {
    pub fn new(
        title: impl Into<String>,
        description: impl Into<String>,
        event_timestamp: DateTime<Utc>,
        embedding: Vec<f32>,
    ) -> Self {
        let mut base = TypedMemoryBase::new(IntelligenceMemoryType::Event, embedding);
        base.importance = Importance::Medium;

        Self {
            base,
            title: title.into(),
            description: description.into(),
            event_timestamp,
            location: None,
            involved_entities: Vec::new(),
            category: None,
            severity: None,
            related_events: Vec::new(),
        }
    }

    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    pub fn with_severity(mut self, severity: u8) -> Self {
        self.severity = Some(severity.min(10));
        self
    }

    pub fn with_involved_entity(mut self, entity_id: EntityId) -> Self {
        self.involved_entities.push(entity_id);
        self.base.entity_links.push(entity_id);
        self
    }

    pub fn with_related_event(mut self, event_id: MemoryId) -> Self {
        self.related_events.push(event_id);
        self
    }
}

// ============================================================================
// NarrativeMemory
// ============================================================================

/// Narrative memory for connected storylines and evolving narratives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeMemory {
    /// Base memory fields
    pub base: TypedMemoryBase,
    /// Narrative title
    pub title: String,
    /// Narrative summary
    pub summary: String,
    /// Full narrative text
    pub narrative: String,
    /// Key entities in narrative
    pub key_entities: Vec<EntityId>,
    /// Narrative arc (beginning, middle, end)
    pub arc_stage: NarrativeArcStage,
    /// Supporting evidence memories
    pub evidence_memories: Vec<MemoryId>,
    /// Related narratives
    pub related_narratives: Vec<MemoryId>,
    /// Confidence in narrative accuracy
    pub narrative_confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NarrativeArcStage {
    Beginning,
    Developing,
    Climax,
    Resolving,
    Concluded,
}

impl NarrativeMemory {
    pub fn new(
        title: impl Into<String>,
        summary: impl Into<String>,
        narrative: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Self {
        let mut base = TypedMemoryBase::new(IntelligenceMemoryType::Narrative, embedding);
        base.importance = Importance::High;

        Self {
            base,
            title: title.into(),
            summary: summary.into(),
            narrative: narrative.into(),
            key_entities: Vec::new(),
            arc_stage: NarrativeArcStage::Beginning,
            evidence_memories: Vec::new(),
            related_narratives: Vec::new(),
            narrative_confidence: 0.5,
        }
    }

    pub fn with_key_entity(mut self, entity_id: EntityId) -> Self {
        self.key_entities.push(entity_id);
        self.base.entity_links.push(entity_id);
        self
    }

    pub fn with_arc_stage(mut self, stage: NarrativeArcStage) -> Self {
        self.arc_stage = stage;
        self
    }

    pub fn with_evidence(mut self, memory_id: MemoryId) -> Self {
        self.evidence_memories.push(memory_id);
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.narrative_confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

// ============================================================================
// RiskNodeMemory
// ============================================================================

/// Risk node memory for threat/risk entity tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskNodeMemory {
    /// Base memory fields
    pub base: TypedMemoryBase,
    /// Risk node name
    pub name: String,
    /// Risk description
    pub description: String,
    /// Risk type (cyber, physical, financial, etc.)
    pub risk_type: RiskType,
    /// Threat level (1-10)
    pub threat_level: u8,
    /// Vulnerability score (1-10)
    pub vulnerability_score: Option<u8>,
    /// Impact assessment (1-10)
    pub impact_score: Option<u8>,
    /// Associated indicators
    pub indicators: Vec<String>,
    /// Mitigation status
    pub mitigation_status: MitigationStatus,
    /// Related risk nodes
    pub related_risks: Vec<MemoryId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskType {
    Cyber,
    Physical,
    Financial,
    Reputational,
    Operational,
    Strategic,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MitigationStatus {
    Unmitigated,
    Partial,
    Mitigated,
    Monitoring,
    Transferred,
}

impl RiskNodeMemory {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        risk_type: RiskType,
        threat_level: u8,
        embedding: Vec<f32>,
    ) -> Self {
        let mut base = TypedMemoryBase::new(IntelligenceMemoryType::RiskNode, embedding);
        base.importance = Importance::High;

        Self {
            base,
            name: name.into(),
            description: description.into(),
            risk_type,
            threat_level: threat_level.min(10),
            vulnerability_score: None,
            impact_score: None,
            indicators: Vec::new(),
            mitigation_status: MitigationStatus::Unmitigated,
            related_risks: Vec::new(),
        }
    }

    pub fn with_vulnerability(mut self, score: u8) -> Self {
        self.vulnerability_score = Some(score.min(10));
        self
    }

    pub fn with_impact(mut self, score: u8) -> Self {
        self.impact_score = Some(score.min(10));
        self
    }

    pub fn with_indicator(mut self, indicator: impl Into<String>) -> Self {
        self.indicators.push(indicator.into());
        self
    }

    pub fn with_mitigation_status(mut self, status: MitigationStatus) -> Self {
        self.mitigation_status = status;
        self
    }

    pub fn with_related_risk(mut self, risk_id: MemoryId) -> Self {
        self.related_risks.push(risk_id);
        self
    }

    /// Calculate composite risk score
    pub fn composite_risk_score(&self) -> f32 {
        let threat = self.threat_level as f32 / 10.0;
        let vulnerability = self.vulnerability_score.unwrap_or(5) as f32 / 10.0;
        let impact = self.impact_score.unwrap_or(5) as f32 / 10.0;

        // Weighted: threat 40%, vulnerability 30%, impact 30%
        threat * 0.4 + vulnerability * 0.3 + impact * 0.3
    }
}

// ============================================================================
// EvidenceMemory
// ============================================================================

/// Evidence memory for attributed evidence tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceMemory {
    /// Base memory fields
    pub base: TypedMemoryBase,
    /// Evidence content
    pub content: String,
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Source of evidence
    pub source: String,
    /// Source reliability (1-10)
    pub source_reliability: u8,
    /// Supporting documents/URLs
    pub supporting_materials: Vec<String>,
    /// Related evidence
    pub related_evidence: Vec<MemoryId>,
    /// Verification status
    pub verified: bool,
    /// Verification notes
    pub verification_notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceType {
    Document,
    Image,
    Signal,
    Human,
    OpenSource,
    Technical,
    Financial,
    Custom(String),
}

impl EvidenceMemory {
    pub fn new(
        content: impl Into<String>,
        evidence_type: EvidenceType,
        source: impl Into<String>,
        source_reliability: u8,
        embedding: Vec<f32>,
    ) -> Self {
        let mut base = TypedMemoryBase::new(IntelligenceMemoryType::Evidence, embedding);
        base.confidence = source_reliability as f32 / 10.0;

        Self {
            base,
            content: content.into(),
            evidence_type,
            source: source.into(),
            source_reliability: source_reliability.min(10),
            supporting_materials: Vec::new(),
            related_evidence: Vec::new(),
            verified: false,
            verification_notes: None,
        }
    }

    pub fn with_supporting_material(mut self, material: impl Into<String>) -> Self {
        self.supporting_materials.push(material.into());
        self
    }

    pub fn with_related_evidence(mut self, evidence_id: MemoryId) -> Self {
        self.related_evidence.push(evidence_id);
        self
    }

    pub fn mark_verified(mut self) -> Self {
        self.verified = true;
        self
    }

    pub fn with_verification_notes(mut self, notes: impl Into<String>) -> Self {
        self.verification_notes = Some(notes.into());
        self
    }
}

// ============================================================================
// SimulationMemory
// ============================================================================

/// Simulation memory for scenario projections and what-if analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationMemory {
    /// Base memory fields
    pub base: TypedMemoryBase,
    /// Simulation title
    pub title: String,
    /// Scenario description
    pub scenario: String,
    /// Simulation parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Simulation results
    pub results: Option<String>,
    /// Outcome probability distribution
    pub outcomes: Vec<SimulationOutcome>,
    /// Related simulations
    pub related_simulations: Vec<MemoryId>,
    /// Simulation status
    pub status: SimulationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationOutcome {
    pub description: String,
    pub probability: f32,
    pub impact_description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimulationStatus {
    Draft,
    Running,
    Complete,
    Superseded,
}

impl SimulationMemory {
    pub fn new(title: impl Into<String>, scenario: impl Into<String>, embedding: Vec<f32>) -> Self {
        let base = TypedMemoryBase::new(IntelligenceMemoryType::Simulation, embedding);

        Self {
            base,
            title: title.into(),
            scenario: scenario.into(),
            parameters: HashMap::new(),
            results: None,
            outcomes: Vec::new(),
            related_simulations: Vec::new(),
            status: SimulationStatus::Draft,
        }
    }

    pub fn with_parameter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }

    pub fn with_results(mut self, results: impl Into<String>) -> Self {
        self.results = Some(results.into());
        self
    }

    pub fn with_outcome(mut self, outcome: SimulationOutcome) -> Self {
        self.outcomes.push(outcome);
        self
    }

    pub fn with_status(mut self, status: SimulationStatus) -> Self {
        self.status = status;
        self
    }

    pub fn with_related_simulation(mut self, sim_id: MemoryId) -> Self {
        self.related_simulations.push(sim_id);
        self
    }
}

// ============================================================================
// Unified Typed Memory Enum
// ============================================================================

/// Unified enum for all intelligence memory types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypedIntelligenceMemory {
    Event(EventMemory),
    Narrative(NarrativeMemory),
    RiskNode(RiskNodeMemory),
    Evidence(EvidenceMemory),
    Simulation(SimulationMemory),
}

impl TypedIntelligenceMemory {
    /// Get base memory fields
    pub fn base(&self) -> &TypedMemoryBase {
        match self {
            Self::Event(m) => &m.base,
            Self::Narrative(m) => &m.base,
            Self::RiskNode(m) => &m.base,
            Self::Evidence(m) => &m.base,
            Self::Simulation(m) => &m.base,
        }
    }

    /// Get embedding
    pub fn embedding(&self) -> &[f32] {
        &self.base().embedding
    }

    /// Get memory type
    pub fn memory_type(&self) -> &IntelligenceMemoryType {
        &self.base().memory_type
    }

    /// Get entity links
    pub fn entity_links(&self) -> &[EntityId] {
        &self.base().entity_links
    }

    /// Mark as accessed
    pub fn mark_accessed(&mut self) {
        match self {
            Self::Event(m) => m.base.mark_accessed(),
            Self::Narrative(m) => m.base.mark_accessed(),
            Self::RiskNode(m) => m.base.mark_accessed(),
            Self::Evidence(m) => m.base.mark_accessed(),
            Self::Simulation(m) => m.base.mark_accessed(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_memory_creation() {
        let event = EventMemory::new(
            "Test Event",
            "A test event occurred",
            Utc::now(),
            vec![0.1; 128],
        )
        .with_location("Test Location")
        .with_severity(7);

        assert_eq!(event.title, "Test Event");
        assert_eq!(event.severity, Some(7));
        assert_eq!(event.base.memory_type, IntelligenceMemoryType::Event);
    }

    #[test]
    fn test_narrative_memory_creation() {
        let narrative = NarrativeMemory::new(
            "Test Narrative",
            "A developing story",
            "Full narrative text",
            vec![0.1; 128],
        )
        .with_arc_stage(NarrativeArcStage::Developing);

        assert_eq!(narrative.title, "Test Narrative");
        assert_eq!(narrative.arc_stage, NarrativeArcStage::Developing);
    }

    #[test]
    fn test_risk_node_composite_score() {
        let risk = RiskNodeMemory::new(
            "Cyber Threat",
            "Active cyber threat detected",
            RiskType::Cyber,
            8,
            vec![0.1; 128],
        )
        .with_vulnerability(7)
        .with_impact(9);

        let score = risk.composite_risk_score();
        // 0.8*0.4 + 0.7*0.3 + 0.9*0.3 = 0.32 + 0.21 + 0.27 = 0.80
        assert!((score - 0.80).abs() < 0.01);
    }

    #[test]
    fn test_evidence_memory_creation() {
        let evidence = EvidenceMemory::new(
            "Evidence content",
            EvidenceType::OpenSource,
            "Reliable Source",
            8,
            vec![0.1; 128],
        )
        .mark_verified();

        assert_eq!(evidence.source_reliability, 8);
        assert!(evidence.verified);
    }

    #[test]
    fn test_simulation_memory_creation() {
        let simulation =
            SimulationMemory::new("Test Simulation", "What if scenario", vec![0.1; 128])
                .with_status(SimulationStatus::Complete);

        assert_eq!(simulation.title, "Test Simulation");
        assert_eq!(simulation.status, SimulationStatus::Complete);
    }
}
