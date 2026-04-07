use chrono::{DateTime, Utc};
use dashmap::DashMap;
use rememnemosyne_core::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::event::{TemporalEvent, TimeWindow, TemporalEventType};
use crate::timeline::{Timeline, TimelineManager};

/// Configuration for temporal memory store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMemoryConfig {
    pub max_events_per_entity: usize,
    pub auto_create_timelines: bool,
    pub event_retention_days: i64,
    pub compress_old_timelines: bool,
    pub compress_threshold: usize,
}

impl Default for TemporalMemoryConfig {
    fn default() -> Self {
        Self {
            max_events_per_entity: 10000,
            auto_create_timelines: true,
            event_retention_days: 365,
            compress_old_timelines: true,
            compress_threshold: 1000,
        }
    }
}

/// Temporal memory store for timeline-based queries
pub struct TemporalMemoryStore {
    config: TemporalMemoryConfig,
    /// All events indexed by ID
    events: Arc<DashMap<uuid::Uuid, TemporalEvent>>,
    /// Events by entity
    entity_events: Arc<DashMap<EntityId, Vec<uuid::Uuid>>>,
    /// Events by memory
    memory_events: Arc<DashMap<MemoryId, Vec<uuid::Uuid>>>,
    /// Timeline manager
    timeline_manager: Arc<parking_lot::RwLock<TimelineManager>>,
    /// Event index by time (for fast range queries)
    time_index: Arc<parking_lot::RwLock<Vec<(chrono::DateTime<Utc>, uuid::Uuid)>>>,
}

impl TemporalMemoryStore {
    pub fn new(config: TemporalMemoryConfig) -> Self {
        Self {
            config,
            events: Arc::new(DashMap::new()),
            entity_events: Arc::new(DashMap::new()),
            memory_events: Arc::new(DashMap::new()),
            timeline_manager: Arc::new(parking_lot::RwLock::new(TimelineManager::new())),
            time_index: Arc::new(parking_lot::RwLock::new(Vec::new())),
        }
    }

    /// Record an event
    pub async fn record_event(&self, event: TemporalEvent) -> Result<uuid::Uuid> {
        let event_id = event.id;

        // Create entity timeline if needed
        if self.config.auto_create_timelines {
            let mut manager = self.timeline_manager.write();
            if manager.get_entity_timeline(&event.entity_id).is_none() {
                manager.create_entity_timeline(
                    event.entity_id,
                    format!("Timeline for entity {}", event.entity_id),
                );
            }

            // Add to timeline
            if let Err(_) = manager.add_event_to_entity(&event.entity_id, event.clone()) {
                return Err(MemoryError::Storage("Failed to add event to timeline".into()));
            }
        }

        // Store event
        self.events.insert(event_id, event.clone());

        // Index by entity
        self.entity_events
            .entry(event.entity_id)
            .or_insert_with(Vec::new)
            .push(event_id);

        // Index by memory
        self.memory_events
            .entry(event.memory_id)
            .or_insert_with(Vec::new)
            .push(event_id);

        // Update time index
        {
            let mut time_index = self.time_index.write();
            time_index.push((event.timestamp, event_id));
            // Keep sorted
            time_index.sort_by(|a, b| a.0.cmp(&b.0));
        }

        Ok(event_id)
    }

    /// Get events for an entity within a time range
    pub async fn get_events_for_entity(
        &self,
        entity_id: &EntityId,
        window: Option<&TimeWindow>,
    ) -> Result<Vec<TemporalEvent>> {
        let event_ids = self.entity_events.get(entity_id)
            .map(|ids| ids.clone())
            .unwrap_or_default();

        let mut events = Vec::new();
        for event_id in event_ids {
            if let Some(event) = self.events.get(&event_id) {
                if let Some(window) = window {
                    if !window.contains(&event.timestamp) {
                        continue;
                    }
                }
                events.push(event.clone());
            }
        }

        // Sort by timestamp
        events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(events)
    }

    /// Get events for a memory
    pub async fn get_events_for_memory(
        &self,
        memory_id: &MemoryId,
    ) -> Result<Vec<TemporalEvent>> {
        let event_ids = self.memory_events.get(memory_id)
            .map(|ids| ids.clone())
            .unwrap_or_default();

        let mut events = Vec::new();
        for event_id in event_ids {
            if let Some(event) = self.events.get(&event_id) {
                events.push(event.clone());
            }
        }

        events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(events)
    }

    /// Get timeline of events
    pub async fn get_timeline(
        &self,
        window: Option<&TimeWindow>,
        limit: usize,
    ) -> Result<Vec<TemporalEvent>> {
        let time_index = self.time_index.read();
        
        let mut events = Vec::new();
        for (_, event_id) in time_index.iter().rev() {
            if let Some(event) = self.events.get(event_id) {
                if let Some(window) = window {
                    if !window.contains(&event.timestamp) {
                        continue;
                    }
                }
                events.push(event.clone());
                if events.len() >= limit {
                    break;
                }
            }
        }

        Ok(events)
    }

    /// Get events by type
    pub async fn get_events_by_type(
        &self,
        event_type: &TemporalEventType,
        limit: usize,
    ) -> Result<Vec<TemporalEvent>> {
        let mut events: Vec<TemporalEvent> = self.events
            .iter()
            .filter(|e| e.event_type == *event_type)
            .map(|e| e.clone())
            .collect();

        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        events.truncate(limit);
        Ok(events)
    }

    /// Get entity timeline
    pub async fn get_entity_timeline(&self, entity_id: &EntityId) -> Option<Timeline> {
        let manager = self.timeline_manager.read();
        manager.get_entity_timeline(entity_id).cloned()
    }

    /// Search events by description
    pub async fn search_events(
        &self,
        query: &str,
        limit: usize,
    ) -> Vec<TemporalEvent> {
        let query_lower = query.to_lowercase();
        
        let mut events: Vec<TemporalEvent> = self.events
            .iter()
            .filter(|e| e.description.to_lowercase().contains(&query_lower))
            .map(|e| e.clone())
            .collect();

        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        events.truncate(limit);
        events
    }

    /// Get events before/after a timestamp
    pub async fn get_events_around(
        &self,
        timestamp: chrono::DateTime<Utc>,
        before: usize,
        after: usize,
    ) -> (Vec<TemporalEvent>, Vec<TemporalEvent>) {
        let time_index = self.time_index.read();
        
        // Find position in sorted index
        let pos = time_index
            .binary_search_by(|(t, _)| t.cmp(&timestamp))
            .unwrap_or_else(|pos| pos);

        // Get before
        let mut before_events = Vec::new();
        for i in (0..pos).rev().take(before) {
            if let Some(event) = self.events.get(&time_index[i].1) {
                before_events.push(event.clone());
            }
        }

        // Get after
        let mut after_events = Vec::new();
        for i in pos..time_index.len().min(pos + after) {
            if let Some(event) = self.events.get(&time_index[i].1) {
                after_events.push(event.clone());
            }
        }

        (before_events, after_events)
    }

    /// Get event statistics
    pub async fn get_statistics(&self) -> TemporalStatistics {
        let mut stats = TemporalStatistics::default();
        stats.total_events = self.events.len();
        stats.unique_entities = self.entity_events.len();

        // Count by type
        for event in self.events.iter() {
            let type_name = format!("{:?}", event.event_type);
            *stats.events_by_type.entry(type_name).or_insert(0) += 1;
        }

        // Time range
        if let Some(first) = self.time_index.read().first() {
            stats.earliest_event = Some(first.0);
        }
        if let Some(last) = self.time_index.read().last() {
            stats.latest_event = Some(last.0);
        }

        stats
    }

    /// Clean up old events
    pub async fn cleanup_old_events(&self) -> Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::days(self.config.event_retention_days);
        
        let to_remove: Vec<uuid::Uuid> = self.events
            .iter()
            .filter(|e| e.timestamp < cutoff)
            .map(|e| *e.key())
            .collect();

        let removed = to_remove.len();

        for event_id in to_remove {
            if let Some((_, event)) = self.events.remove(&event_id) {
                // Remove from entity index
                if let Some(mut ids) = self.entity_events.get_mut(&event.entity_id) {
                    ids.retain(|id| *id != event_id);
                }
                // Remove from memory index
                if let Some(mut ids) = self.memory_events.get_mut(&event.memory_id) {
                    ids.retain(|id| *id != event_id);
                }
                // Remove from time index
                {
                    let mut time_index = self.time_index.write();
                    time_index.retain(|(_, id)| *id != event_id);
                }
            }
        }

        Ok(removed)
    }
}

// Note: TemporalMemoryStore trait from core is not implemented here
// due to naming conflict. The struct provides its own methods directly.

#[allow(dead_code)]
fn convert_to_memory_event(event: TemporalEvent) -> MemoryEvent {
    MemoryEvent {
        id: event.id,
        entity_id: event.entity_id,
        memory_id: event.memory_id,
        event_type: match event.event_type {
            TemporalEventType::Created => EventType::Created,
            TemporalEventType::Updated => EventType::Updated,
            TemporalEventType::MemoryRetrieved => EventType::Accessed,
            TemporalEventType::RelationshipEstablished => EventType::Related,
            TemporalEventType::MemoryMerged => EventType::Merged,
            TemporalEventType::Archived => EventType::Archived,
            TemporalEventType::Deleted => EventType::Deleted,
            other => EventType::Custom(format!("{:?}", other)),
        },
        timestamp: event.timestamp,
        description: event.description,
        metadata: event.metadata,
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalStatistics {
    pub total_events: usize,
    pub unique_entities: usize,
    pub events_by_type: HashMap<String, usize>,
    pub earliest_event: Option<DateTime<Utc>>,
    pub latest_event: Option<DateTime<Utc>>,
}
