//! Prometheus metrics for monitoring RemeMnemosyne engine
//!
//! This module provides comprehensive metrics for monitoring the memory engine,
//! including memory counts, operation latencies, cache hit rates, and more.
//! Enabled with the `metrics` feature flag.
use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::{
    counter::Counter, family::Family, gauge::Gauge, histogram::Histogram,
};
use prometheus_client::registry::Registry;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// RemeMnemosyne metrics collector
#[derive(Clone)]
pub struct RememnosyneMetrics {
    registry: Arc<parking_lot::RwLock<Registry>>,
    total_memories: Family<MemoryLabels, Gauge>,
    semantic_memories: Gauge,
    episodic_memories: Gauge,
    graph_entities: Gauge,
    graph_relationships: Gauge,
    remember_ops_total: Counter,
    recall_ops_total: Counter,
    delete_ops_total: Counter,
    remember_latency_seconds: Histogram,
    recall_latency_seconds: Histogram,
    cache_hits: Counter,
    cache_misses: Counter,
    errors_total: Family<ErrorLabels, Counter>,
}

/// Label set for memory metrics
#[derive(
    Clone,
    Debug,
    Hash,
    PartialEq,
    Eq,
    prometheus_client::encoding::EncodeLabelSet,
    Serialize,
    Deserialize,
)]
pub struct MemoryLabels {
    pub memory_type: MemoryTypeLabel,
    pub trigger: TriggerLabel,
}

#[derive(
    Clone,
    Debug,
    Hash,
    PartialEq,
    Eq,
    prometheus_client::encoding::EncodeLabelValue,
    Serialize,
    Deserialize,
)]
pub enum MemoryTypeLabel {
    Semantic,
    Episodic,
    Graph,
    Temporal,
    EventClassification,
    InfrastructureGap,
    GapDocumentation,
    NarrativeThread,
    EvidenceChain,
    CounterNarrative,
    Checkpoint,
}

#[derive(
    Clone,
    Debug,
    Hash,
    PartialEq,
    Eq,
    prometheus_client::encoding::EncodeLabelValue,
    Serialize,
    Deserialize,
)]
pub enum TriggerLabel {
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
    Custom,
}

/// Error type labels
#[derive(
    Clone,
    Debug,
    Hash,
    PartialEq,
    Eq,
    prometheus_client::encoding::EncodeLabelSet,
    Serialize,
    Deserialize,
)]
pub struct ErrorLabels {
    pub error_type: ErrorTypeLabel,
}

#[derive(
    Clone,
    Debug,
    Hash,
    PartialEq,
    Eq,
    prometheus_client::encoding::EncodeLabelValue,
    Serialize,
    Deserialize,
)]
pub enum ErrorTypeLabel {
    Storage,
    Quantization,
    Index,
    NotFound,
    InvalidQuery,
    Serialization,
    Graph,
    Cognitive,
    Timeout,
    CapacityExceeded,
}

impl RememnosyneMetrics {
    /// Create new metrics collector
    pub fn new() -> Self {
        let mut registry = Registry::default();

        let total_memories = Family::default();
        let semantic_memories = Gauge::default();
        let episodic_memories = Gauge::default();
        let graph_entities = Gauge::default();
        let graph_relationships = Gauge::default();

        let remember_ops_total = Counter::default();
        let recall_ops_total = Counter::default();
        let delete_ops_total = Counter::default();

        let remember_latency_seconds = Histogram::new(
            vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]
            .into_iter(),
        );
        let recall_latency_seconds = Histogram::new(
            vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]
            .into_iter(),
        );

        let cache_hits = Counter::default();
        let cache_misses = Counter::default();

        let errors_total = Family::default();

        registry.register(
            "total_memories",
            "Total number of memories stored",
            total_memories.clone(),
        );
        registry.register(
            "semantic_memories",
            "Number of semantic memories",
            semantic_memories.clone(),
        );
        registry.register(
            "episodic_memories",
            "Number of episodic memories",
            episodic_memories.clone(),
        );
        registry.register(
            "graph_entities",
            "Number of graph entities",
            graph_entities.clone(),
        );
        registry.register(
            "graph_relationships",
            "Number of graph relationships",
            graph_relationships.clone(),
        );
        registry.register(
            "remember_ops_total",
            "Total remember operations",
            remember_ops_total.clone(),
        );
        registry.register(
            "recall_ops_total",
            "Total recall operations",
            recall_ops_total.clone(),
        );
        registry.register(
            "delete_ops_total",
            "Total delete operations",
            delete_ops_total.clone(),
        );
        registry.register(
            "remember_latency_seconds",
            "Remember operation latency",
            remember_latency_seconds.clone(),
        );
        registry.register(
            "recall_latency_seconds",
            "Recall operation latency",
            recall_latency_seconds.clone(),
        );
        registry.register("cache_hits_total", "Total cache hits", cache_hits.clone());
        registry.register(
            "cache_misses_total",
            "Total cache misses",
            cache_misses.clone(),
        );
        registry.register("errors_total", "Total errors by type", errors_total.clone());

        Self {
            registry: Arc::new(parking_lot::RwLock::new(registry)),
            total_memories,
            semantic_memories,
            episodic_memories,
            graph_entities,
            graph_relationships,
            remember_ops_total,
            recall_ops_total,
            delete_ops_total,
            remember_latency_seconds,
            recall_latency_seconds,
            cache_hits,
            cache_misses,
            errors_total,
        }
    }

    /// Create with default registry (convenience)
    pub fn default_metrics() -> Self {
        Self::new()
    }

    /// Encode metrics to Prometheus text format
    pub fn encode(&self) -> String {
        let mut buffer = String::new();
        let reg = self.registry.read();
        encode(&mut buffer, &reg).unwrap_or_default();
        buffer
    }

    /// Record a remember operation
    pub fn record_remember(
        &self,
        memory_type: MemoryTypeLabel,
        trigger: TriggerLabel,
        duration: std::time::Duration,
    ) {
        self.remember_ops_total.inc();
        self.remember_latency_seconds
            .observe(duration.as_secs_f64());

        self.total_memories
            .get_or_create(&MemoryLabels {
                memory_type: memory_type.clone(),
                trigger,
            })
            .inc();

        match memory_type {
            MemoryTypeLabel::Semantic => {
                self.semantic_memories.inc();
            }
            MemoryTypeLabel::Episodic => {
                self.episodic_memories.inc();
            }
            MemoryTypeLabel::Graph => {
                self.graph_entities.inc();
            }
            MemoryTypeLabel::Temporal
            | MemoryTypeLabel::EventClassification
            | MemoryTypeLabel::InfrastructureGap
            | MemoryTypeLabel::GapDocumentation
            | MemoryTypeLabel::NarrativeThread
            | MemoryTypeLabel::EvidenceChain
            | MemoryTypeLabel::CounterNarrative
            | MemoryTypeLabel::Checkpoint => {}
        }
    }

    /// Record a recall operation
    pub fn record_recall(&self, duration: std::time::Duration, _results_count: usize) {
        self.recall_ops_total.inc();
        self.recall_latency_seconds.observe(duration.as_secs_f64());
    }

    /// Record a delete operation
    pub fn record_delete(&self) {
        self.delete_ops_total.inc();
    }

    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.inc();
    }

    /// Record a cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.inc();
    }

    /// Record an error
    pub fn record_error(&self, error_type: ErrorTypeLabel) {
        self.errors_total
            .get_or_create(&ErrorLabels { error_type })
            .inc();
    }

    /// Update entity count
    pub fn update_entity_count(&self, count: usize) {
        self.graph_entities.set(count as i64);
    }

    /// Update relationship count
    pub fn update_relationship_count(&self, count: usize) {
        self.graph_relationships.set(count as i64);
    }
}

/// Stub implementation when feature is not enabled
#[cfg(not(feature = "metrics"))]
#[derive(Clone)]
pub struct RememnosyneMetrics;

#[cfg(not(feature = "metrics"))]
impl RememnosyneMetrics {
    pub fn new() -> Self {
        Self
    }

    pub fn default_metrics() -> Self {
        Self
    }

    pub fn encode(&self) -> String {
        String::new()
    }

    pub fn record_remember(&self, _: MemoryTypeLabel, _: TriggerLabel, _: std::time::Duration) {}
    pub fn record_recall(&self, _: std::time::Duration, _: usize) {}
    pub fn record_delete(&self) {}
    pub fn record_cache_hit(&self) {}
    pub fn record_cache_miss(&self) {}
    pub fn record_error(&self, _: ErrorTypeLabel) {}
    pub fn update_entity_count(&self, _: usize) {}
    pub fn update_relationship_count(&self, _: usize) {}
}

// Re-export labels for use in engine - they're already public in this module

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "metrics")]
    #[test]
    fn test_metrics_creation() {
        let metrics = RememnosyneMetrics::default_metrics();
        let encoded = metrics.encode();
        assert!(!encoded.is_empty());
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_metrics_recording() {
        let metrics = RememnosyneMetrics::default_metrics();
        metrics.record_remember(
            MemoryTypeLabel::Semantic,
            TriggerLabel::UserInput,
            std::time::Duration::from_millis(10),
        );

        let encoded = metrics.encode();
        assert!(encoded.contains("remember_ops_total"));
    }

    #[cfg(not(feature = "metrics"))]
    #[test]
    fn test_metrics_not_enabled() {
        let metrics = RememnosyneMetrics::default_metrics();
        assert_eq!(metrics.encode(), "");
    }
}
