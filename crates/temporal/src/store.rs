use chrono::{DateTime, Utc};
use dashmap::DashMap;
use rememnemosyne_core::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::event::{TemporalEvent, TemporalEventType, TimeWindow};
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

type TimeIndexEntry = (DateTime<Utc>, uuid::Uuid);

/// Temporal memory store for timeline-based queries
pub struct TemporalMemoryStore {
    config: TemporalMemoryConfig,
    events: Arc<DashMap<uuid::Uuid, TemporalEvent>>,
    entity_events: Arc<DashMap<EntityId, Vec<uuid::Uuid>>>,
    memory_events: Arc<DashMap<MemoryId, Vec<uuid::Uuid>>>,
    timeline_manager: Arc<parking_lot::RwLock<TimelineManager>>,
    time_index: Arc<parking_lot::RwLock<Vec<TimeIndexEntry>>>,
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

    pub async fn record_event(&self, event: TemporalEvent) -> Result<uuid::Uuid> {
        let event_id = event.id;
        if self.config.auto_create_timelines {
            let mut manager = self.timeline_manager.write();
            if manager.get_entity_timeline(&event.entity_id).is_none() {
                manager.create_entity_timeline(
                    event.entity_id,
                    format!("Timeline for entity {}", event.entity_id),
                );
            }
            if manager.add_event_to_entity(&event.entity_id, event.clone()).is_err() {
                return Err(MemoryError::Storage("Failed to add event to timeline".into()));
            }
        }
        self.events.insert(event_id, event.clone());
        self.entity_events.entry(event.entity_id).or_default().push(event_id);
        self.memory_events.entry(event.memory_id).or_default().push(event_id);
        {
            let mut time_index = self.time_index.write();
            time_index.push((event.timestamp, event_id));
            time_index.sort_by(|a, b| a.0.cmp(&b.0));
        }
        Ok(event_id)
    }

    pub async fn get_events_for_entity(
        &self,
        entity_id: &EntityId,
        window: Option<&TimeWindow>,
    ) -> Result<Vec<TemporalEvent>> {
        let event_ids = self.entity_events.get(entity_id).map(|ids| ids.clone()).unwrap_or_default();
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
        events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(events)
    }

    pub async fn get_events_for_memory(&self, memory_id: &MemoryId) -> Result<Vec<TemporalEvent>> {
        let event_ids = self.memory_events.get(memory_id).map(|ids| ids.clone()).unwrap_or_default();
        let mut events = Vec::new();
        for event_id in event_ids {
            if let Some(event) = self.events.get(&event_id) {
                events.push(event.clone());
            }
        }
        events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(events)
    }

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

    pub async fn get_events_by_type(
        &self,
        event_type: &TemporalEventType,
        limit: usize,
    ) -> Result<Vec<TemporalEvent>> {
        let mut events: Vec<TemporalEvent> = self
            .events
            .iter()
            .filter(|e| e.event_type == *event_type)
            .map(|e| e.clone())
            .collect();
        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        events.truncate(limit);
        Ok(events)
    }

    pub async fn get_entity_timeline(&self, entity_id: &EntityId) -> Option<Timeline> {
        let manager = self.timeline_manager.read();
        manager.get_entity_timeline(entity_id).cloned()
    }

    pub async fn search_events(&self, query: &str, limit: usize) -> Vec<TemporalEvent> {
        let query_lower = query.to_lowercase();
        let mut events: Vec<TemporalEvent> = self
            .events
            .iter()
            .filter(|e| e.description.to_lowercase().contains(&query_lower))
            .map(|e| e.clone())
            .collect();
        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        events.truncate(limit);
        events
    }

    pub async fn get_events_around(
        &self,
        timestamp: chrono::DateTime<Utc>,
        before: usize,
        after: usize,
    ) -> (Vec<TemporalEvent>, Vec<TemporalEvent>) {
        let time_index = self.time_index.read();
        let pos = time_index
            .binary_search_by(|(t, _)| t.cmp(&timestamp))
            .unwrap_or_else(|pos| pos);
        let mut before_events = Vec::new();
        for i in (0..pos).rev().take(before) {
            if let Some(event) = self.events.get(&time_index[i].1) {
                before_events.push(event.clone());
            }
        }
        let mut after_events = Vec::new();
        for i in pos..time_index.len().min(pos + after) {
            if let Some(event) = self.events.get(&time_index[i].1) {
                after_events.push(event.clone());
            }
        }
        (before_events, after_events)
    }

    pub async fn get_statistics(&self) -> TemporalStatistics {
        let time_index = self.time_index.read();
        TemporalStatistics {
            total_events: self.events.len(),
            unique_entities: self.entity_events.len(),
            events_by_type: self.events.iter().map(|e| format!("{:?}", e.event_type)).fold(HashMap::new(), |mut acc, t| {
                *acc.entry(t).or_insert(0) += 1;
                acc
            }),
            earliest_event: time_index.first().map(|(t, _)| *t),
            latest_event: time_index.last().map(|(t, _)| *t),
        }
    }

    pub async fn cleanup_old_events(&self) -> Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::days(self.config.event_retention_days);
        let to_remove: Vec<uuid::Uuid> = self
            .events
            .iter()
            .filter(|e| e.timestamp < cutoff)
            .map(|e| *e.key())
            .collect();
        let removed = to_remove.len();
        for event_id in to_remove {
            if let Some((_, event)) = self.events.remove(&event_id) {
                if let Some(mut ids) = self.entity_events.get_mut(&event.entity_id) {
                    ids.retain(|id| *id != event_id);
                }
                if let Some(mut ids) = self.memory_events.get_mut(&event.memory_id) {
                    ids.retain(|id| *id != event_id);
                }
                {
                    let mut time_index = self.time_index.write();
                    time_index.retain(|(_, id)| *id != event_id);
                }
            }
        }
        Ok(removed)
    }

    /// Delete all events associated with a given memory ID (cascade delete)
    pub async fn delete_events_by_memory_id(&self, memory_id: &MemoryId) {
        let event_ids: Vec<uuid::Uuid> = self
            .memory_events
            .get(memory_id)
            .map(|ids| ids.clone())
            .unwrap_or_default();
        for event_id in event_ids {
            if let Some((_, event)) = self.events.remove(&event_id) {
                if let Some(mut ids) = self.entity_events.get_mut(&event.entity_id) {
                    ids.retain(|id| *id != event_id);
                }
                {
                    let mut time_index = self.time_index.write();
                    time_index.retain(|(_, id)| *id != event_id);
                }
            }
        }
        self.memory_events.remove(memory_id);
    }

    /// Record a firehose event (RISC.OSINT integration)
    pub async fn record_firehose_event(
        &self,
        entity_id: EntityId,
        memory_id: MemoryId,
        description: impl Into<String>,
    ) -> Result<uuid::Uuid> {
        let event = TemporalEvent::new(
            entity_id,
            memory_id,
            TemporalEventType::ExternalEvent,
            description,
        );
        self.record_event(event).await
    }

    /// Get event density for a time window (heatmap support)
    pub async fn get_event_density(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        bucket_count: usize,
    ) -> Vec<usize> {
        let time_index = self.time_index.read();
        let mut buckets = vec![0usize; bucket_count];
        let total_secs = (end - start).num_seconds() as f64;
        if total_secs <= 0.0 {
            return buckets;
        }
        let bucket_secs = total_secs / bucket_count as f64;
        for (timestamp, _) in time_index.iter() {
            if *timestamp >= start && *timestamp <= end {
                let offset = (*timestamp - start).num_seconds() as f64;
                let bucket = (offset / bucket_secs) as usize;
                if bucket < bucket_count {
                    buckets[bucket] += 1;
                }
            }
        }
        buckets
    }

    /// Detect temporal patterns: escalation or stabilization signals
    pub async fn detect_temporal_pattern(
        &self,
        window_hours: i64,
        comparison_window_hours: i64,
    ) -> TemporalPattern {
        let now = Utc::now();
        let current_start = now - chrono::Duration::hours(window_hours);
        let previous_start = now - chrono::Duration::hours(window_hours + comparison_window_hours);
        let previous_end = now - chrono::Duration::hours(comparison_window_hours);
        let time_index = self.time_index.read();
        let current_count = time_index.iter().filter(|(t, _)| *t >= current_start && *t <= now).count();
        let previous_count = time_index.iter().filter(|(t, _)| *t >= previous_start && *t <= previous_end).count();
        if previous_count == 0 {
            return TemporalPattern::Stable { current_count, previous_count: 0 };
        }
        let ratio = current_count as f32 / previous_count as f32;
        if ratio > 2.0 {
            TemporalPattern::Escalation { current_count, previous_count, ratio }
        } else if ratio < 0.5 {
            TemporalPattern::Stabilization { current_count, previous_count, ratio }
        } else {
            TemporalPattern::Stable { current_count, previous_count }
        }
    }
}

/// Detected temporal pattern
#[derive(Debug, Clone)]
pub enum TemporalPattern {
    Escalation { current_count: usize, previous_count: usize, ratio: f32 },
    Stabilization { current_count: usize, previous_count: usize, ratio: f32 },
    Stable { current_count: usize, previous_count: usize },
}

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
            TemporalEventType::ExternalEvent => EventType::Custom("external_event".into()),
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
