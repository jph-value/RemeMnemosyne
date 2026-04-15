use chrono::{DateTime, Utc};
use rememnemosyne_core::EntityId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::event::{TemporalEvent, TemporalEventType, TimeWindow};

/// Timeline - a sequence of events organized chronologically
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub entity_id: Option<EntityId>,
    pub events: Vec<TemporalEvent>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Timeline {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: String::new(),
            entity_id: None,
            events: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    pub fn for_entity(name: impl Into<String>, entity_id: EntityId) -> Self {
        let mut timeline = Self::new(name);
        timeline.entity_id = Some(entity_id);
        timeline
    }

    pub fn add_event(&mut self, event: TemporalEvent) {
        // Insert in chronological order
        let pos = self
            .events
            .binary_search_by(|e| e.timestamp.cmp(&event.timestamp))
            .unwrap_or_else(|pos| pos);
        self.events.insert(pos, event);
        self.updated_at = Utc::now();
    }

    pub fn get_events_in_window(&self, window: &TimeWindow) -> Vec<&TemporalEvent> {
        self.events
            .iter()
            .filter(|e| window.contains(&e.timestamp))
            .collect()
    }

    pub fn get_events_by_type(&self, event_type: &TemporalEventType) -> Vec<&TemporalEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == *event_type)
            .collect()
    }

    pub fn get_recent_events(&self, hours: i64) -> Vec<&TemporalEvent> {
        let cutoff = Utc::now() - chrono::Duration::hours(hours);
        self.events
            .iter()
            .filter(|e| e.timestamp > cutoff)
            .collect()
    }

    pub fn get_first_event(&self) -> Option<&TemporalEvent> {
        self.events.first()
    }

    pub fn get_last_event(&self) -> Option<&TemporalEvent> {
        self.events.last()
    }

    pub fn get_duration(&self) -> Option<chrono::Duration> {
        let first = self.events.first()?;
        let last = self.events.last()?;
        Some(last.timestamp - first.timestamp)
    }

    pub fn get_event_count(&self) -> usize {
        self.events.len()
    }

    pub fn get_event_frequency(&self) -> HashMap<String, usize> {
        let mut frequency = HashMap::new();
        for event in &self.events {
            let type_name = format!("{:?}", event.event_type);
            *frequency.entry(type_name).or_insert(0) += 1;
        }
        frequency
    }

    pub fn compress(&mut self, max_events: usize) {
        if self.events.len() <= max_events {
            return;
        }

        // Keep first, last, and most important events
        let mut indexed: Vec<(usize, &TemporalEvent)> = self.events.iter().enumerate().collect();
        indexed.sort_by(|a, b| {
            b.1.importance
                .partial_cmp(&a.1.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep = std::collections::HashSet::new();
        keep.insert(0); // First
        keep.insert(self.events.len() - 1); // Last

        for (idx, _) in indexed.iter().take(max_events - 2) {
            keep.insert(*idx);
        }

        let mut new_events = Vec::new();
        for (idx, event) in self.events.iter().enumerate() {
            if keep.contains(&idx) {
                new_events.push(event.clone());
            }
        }

        self.events = new_events;
        self.updated_at = Utc::now();
    }

    /// Generate a summary of the timeline
    pub fn generate_summary(&self) -> TimelineSummary {
        let mut summary = TimelineSummary::new(self.id, self.name.clone());

        if self.events.is_empty() {
            return summary;
        }

        summary.start_time = self.events.first().map(|e| e.timestamp);
        summary.end_time = self.events.last().map(|e| e.timestamp);
        summary.event_count = self.events.len();

        // Count by type
        let mut type_counts = HashMap::new();
        for event in &self.events {
            *type_counts
                .entry(format!("{:?}", event.event_type))
                .or_insert(0) += 1;
        }
        summary.event_type_distribution = type_counts;

        // High importance events
        summary.key_events = self
            .events
            .iter()
            .filter(|e| e.importance >= 0.7)
            .cloned()
            .collect();

        summary
    }
}

/// Timeline summary for quick overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineSummary {
    pub timeline_id: Uuid,
    pub name: String,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub event_count: usize,
    pub event_type_distribution: HashMap<String, usize>,
    pub key_events: Vec<TemporalEvent>,
}

impl TimelineSummary {
    pub fn new(timeline_id: Uuid, name: String) -> Self {
        Self {
            timeline_id,
            name,
            start_time: None,
            end_time: None,
            event_count: 0,
            event_type_distribution: HashMap::new(),
            key_events: Vec::new(),
        }
    }

    pub fn duration(&self) -> Option<chrono::Duration> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => Some(end - start),
            _ => None,
        }
    }
}

/// Timeline manager for handling multiple timelines
pub struct TimelineManager {
    timelines: HashMap<Uuid, Timeline>,
    entity_timelines: HashMap<EntityId, Uuid>,
}

impl TimelineManager {
    pub fn new() -> Self {
        Self {
            timelines: HashMap::new(),
            entity_timelines: HashMap::new(),
        }
    }

    pub fn create_timeline(&mut self, name: impl Into<String>) -> Uuid {
        let timeline = Timeline::new(name);
        let id = timeline.id;
        self.timelines.insert(id, timeline);
        id
    }

    pub fn create_entity_timeline(&mut self, entity_id: EntityId, name: impl Into<String>) -> Uuid {
        let timeline = Timeline::for_entity(name, entity_id);
        let id = timeline.id;
        self.entity_timelines.insert(entity_id, id);
        self.timelines.insert(id, timeline);
        id
    }

    pub fn get_timeline(&self, id: &Uuid) -> Option<&Timeline> {
        self.timelines.get(id)
    }

    pub fn get_timeline_mut(&mut self, id: &Uuid) -> Option<&mut Timeline> {
        self.timelines.get_mut(id)
    }

    pub fn get_entity_timeline(&self, entity_id: &EntityId) -> Option<&Timeline> {
        self.entity_timelines
            .get(entity_id)
            .and_then(|id| self.timelines.get(id))
    }

    pub fn get_entity_timeline_mut(&mut self, entity_id: &EntityId) -> Option<&mut Timeline> {
        self.entity_timelines
            .get(entity_id)
            .copied()
            .and_then(|id| self.timelines.get_mut(&id))
    }

    pub fn add_event_to_entity(
        &mut self,
        entity_id: &EntityId,
        event: TemporalEvent,
    ) -> Result<(), String> {
        let timeline_id = *self
            .entity_timelines
            .get(entity_id)
            .ok_or_else(|| "No timeline for entity".to_string())?;

        if let Some(timeline) = self.timelines.get_mut(&timeline_id) {
            timeline.add_event(event);
            Ok(())
        } else {
            Err("Timeline not found".to_string())
        }
    }

    pub fn get_all_timelines(&self) -> Vec<&Timeline> {
        self.timelines.values().collect()
    }

    pub fn search_timelines(&self, query: &str) -> Vec<&Timeline> {
        let query_lower = query.to_lowercase();
        self.timelines
            .values()
            .filter(|t| {
                t.name.to_lowercase().contains(&query_lower)
                    || t.description.to_lowercase().contains(&query_lower)
            })
            .collect()
    }
}

impl Default for TimelineManager {
    fn default() -> Self {
        Self::new()
    }
}
