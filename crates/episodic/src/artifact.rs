use chrono::{DateTime, Utc};
use rememnemosyne_core::{EntityRef, MemoryId, SessionId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Episode - a coherent sequence of events/exchanges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: Uuid,
    pub session_id: SessionId,
    pub title: String,
    pub summary: String,
    pub detailed_summary: String,
    pub exchanges: Vec<Exchange>,
    pub key_decisions: Vec<Decision>,
    pub entities_mentioned: Vec<EntityRef>,
    pub topics: Vec<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub importance: f32,
    pub emotional_valence: f32,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Episode {
    pub fn new(session_id: SessionId, title: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            session_id,
            title: title.into(),
            summary: String::new(),
            detailed_summary: String::new(),
            exchanges: Vec::new(),
            key_decisions: Vec::new(),
            entities_mentioned: Vec::new(),
            topics: Vec::new(),
            start_time: now,
            end_time: now,
            importance: 0.5,
            emotional_valence: 0.0,
            metadata: HashMap::new(),
        }
    }

    pub fn add_exchange(&mut self, exchange: Exchange) {
        self.end_time = exchange.timestamp;
        self.exchanges.push(exchange);
    }

    pub fn add_decision(&mut self, decision: Decision) {
        self.key_decisions.push(decision);
    }

    pub fn compute_duration_seconds(&self) -> i64 {
        (self.end_time - self.start_time).num_seconds()
    }

    pub fn compute_engagement_score(&self) -> f32 {
        if self.exchanges.is_empty() {
            return 0.0;
        }

        let avg_length: f32 = self
            .exchanges
            .iter()
            .map(|e| (e.content.len() + e.response.as_ref().map(|r| r.len()).unwrap_or(0)) as f32)
            .sum::<f32>()
            / self.exchanges.len() as f32;

        let decision_factor = self.key_decisions.len() as f32 * 0.1;
        let entity_factor = self.entities_mentioned.len() as f32 * 0.05;

        (avg_length / 1000.0 + decision_factor + entity_factor).min(1.0)
    }
}

/// A single exchange in an episode (user input + system response)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exchange {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub role: ExchangeRole,
    pub content: String,
    pub response: Option<String>,
    pub intent: Option<String>,
    pub entities: Vec<EntityRef>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Exchange {
    pub fn new(role: ExchangeRole, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            role,
            content: content.into(),
            response: None,
            intent: None,
            entities: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_response(mut self, response: impl Into<String>) -> Self {
        self.response = Some(response.into());
        self
    }

    pub fn with_intent(mut self, intent: impl Into<String>) -> Self {
        self.intent = Some(intent.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExchangeRole {
    User,
    Assistant,
    System,
    Tool,
}

/// A decision point captured in an episode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub description: String,
    pub context: String,
    pub alternatives: Vec<String>,
    pub chosen_option: String,
    pub rationale: Option<String>,
    pub outcome: Option<DecisionOutcome>,
    pub entities_involved: Vec<EntityRef>,
}

impl Decision {
    pub fn new(
        description: impl Into<String>,
        context: impl Into<String>,
        chosen_option: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            description: description.into(),
            context: context.into(),
            alternatives: Vec::new(),
            chosen_option: chosen_option.into(),
            rationale: None,
            outcome: None,
            entities_involved: Vec::new(),
        }
    }

    pub fn with_alternatives(mut self, alternatives: Vec<String>) -> Self {
        self.alternatives = alternatives;
        self
    }

    pub fn with_rationale(mut self, rationale: impl Into<String>) -> Self {
        self.rationale = Some(rationale.into());
        self
    }

    pub fn with_outcome(mut self, outcome: DecisionOutcome) -> Self {
        self.outcome = Some(outcome);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionOutcome {
    Successful,
    Failed,
    PartiallySuccessful,
    Pending,
    Reversed,
}

/// Memory artifact specific to episodic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicArtifact {
    pub memory_id: MemoryId,
    pub episode_id: Uuid,
    pub artifact_type: EpisodicArtifactType,
    pub content: String,
    pub embedding: Vec<f32>,
    pub timestamp: DateTime<Utc>,
    pub importance: f32,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpisodicArtifactType {
    Exchange,
    Decision,
    Insight,
    Error,
    Summary,
    Milestone,
}

/// Conversation context for better summarization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationContext {
    pub session_id: SessionId,
    pub episode_id: Option<Uuid>,
    pub previous_exchanges: Vec<Exchange>,
    pub current_topic: Option<String>,
    pub active_entities: Vec<EntityRef>,
}

impl ConversationContext {
    pub fn new(session_id: SessionId) -> Self {
        Self {
            session_id,
            episode_id: None,
            previous_exchanges: Vec::new(),
            current_topic: None,
            active_entities: Vec::new(),
        }
    }

    pub fn add_exchange(&mut self, exchange: Exchange) {
        self.previous_exchanges.push(exchange);
        // Keep only recent exchanges
        if self.previous_exchanges.len() > 20 {
            self.previous_exchanges.remove(0);
        }
    }

    pub fn get_recent_context(&self, n: usize) -> Vec<&Exchange> {
        self.previous_exchanges.iter().rev().take(n).rev().collect()
    }
}
