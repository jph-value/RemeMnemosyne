use chrono::{DateTime, Utc};
use rememnemosyne_core::{EntityId, MemoryId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Extended event for temporal memory with rich metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEvent {
    pub id: Uuid,
    pub entity_id: EntityId,
    pub memory_id: MemoryId,
    pub event_type: TemporalEventType,
    pub timestamp: DateTime<Utc>,
    pub description: String,
    pub details: EventDetails,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub related_events: Vec<Uuid>,
    pub importance: f32,
}

impl TemporalEvent {
    pub fn new(
        entity_id: EntityId,
        memory_id: MemoryId,
        event_type: TemporalEventType,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            entity_id,
            memory_id,
            event_type,
            timestamp: Utc::now(),
            description: description.into(),
            details: EventDetails::default(),
            tags: Vec::new(),
            metadata: HashMap::new(),
            related_events: Vec::new(),
            importance: 0.5,
        }
    }

    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    pub fn with_details(mut self, details: EventDetails) -> Self {
        self.details = details;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }

    pub fn add_related_event(&mut self, event_id: Uuid) {
        if !self.related_events.contains(&event_id) {
            self.related_events.push(event_id);
        }
    }

    pub fn is_recent(&self, hours: i64) -> bool {
        let cutoff = Utc::now() - chrono::Duration::hours(hours);
        self.timestamp > cutoff
    }

    pub fn time_since(&self) -> chrono::Duration {
        Utc::now() - self.timestamp
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalEventType {
    // Entity lifecycle
    Created,
    Updated,
    Deleted,
    Archived,

    // Relationship events
    RelationshipEstablished,
    RelationshipBroken,
    RelationshipStrengthened,

    // Memory events
    MemoryStored,
    MemoryRetrieved,
    MemoryMerged,

    // Firehose/external events (RISC.OSINT)
    ExternalEvent,

    // Agent events
    DecisionMade,
    TaskStarted,
    TaskCompleted,
    ErrorOccurred,
    InsightGained,

    // Custom
    Custom(String),
}

/// Additional details about an event
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventDetails {
    pub previous_value: Option<String>,
    pub new_value: Option<String>,
    pub actor: Option<String>,
    pub source: Option<String>,
    pub confidence: f32,
    pub context: Option<String>,
}

impl EventDetails {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_change(mut self, previous: impl Into<String>, new: impl Into<String>) -> Self {
        self.previous_value = Some(previous.into());
        self.new_value = Some(new.into());
        self
    }

    pub fn with_actor(mut self, actor: impl Into<String>) -> Self {
        self.actor = Some(actor.into());
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// Time window for event filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl TimeWindow {
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self { start, end }
    }

    pub fn contains(&self, timestamp: &DateTime<Utc>) -> bool {
        *timestamp >= self.start && *timestamp <= self.end
    }

    pub fn duration(&self) -> chrono::Duration {
        self.end - self.start
    }

    pub fn last_hours(hours: i64) -> Self {
        let end = Utc::now();
        let start = end - chrono::Duration::hours(hours);
        Self { start, end }
    }

    pub fn last_days(days: i64) -> Self {
        let end = Utc::now();
        let start = end - chrono::Duration::days(days);
        Self { start, end }
    }

    pub fn last_weeks(weeks: i64) -> Self {
        Self::last_days(weeks * 7)
    }

    pub fn from_now(duration: chrono::Duration) -> Self {
        let start = Utc::now();
        let end = start + duration;
        Self { start, end }
    }
}
