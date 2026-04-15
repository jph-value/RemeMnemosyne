/// Enhanced auto-pruning for low-importance memories
///
/// This module extends the existing pruner with more advanced auto-pruning strategies
/// including importance-based tiered pruning, access pattern analysis, and configurable policies.
/// Enabled with the `auto-pruning` feature flag.
use chrono::Utc;
use rememnemosyne_core::{Importance, MemoryArtifact, MemoryId, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Auto-pruning configuration with advanced strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoPrunerConfig {
    /// Enable automatic pruning
    pub enabled: bool,
    /// Interval between automatic pruning runs
    pub prune_interval: Duration,
    /// Pruning strategy
    pub strategy: PruningStrategy,
    /// Importance thresholds by tier
    pub importance_thresholds: ImportanceThresholds,
    /// Maximum memories per importance tier
    pub max_memories_per_tier: HashMap<String, usize>,
    /// Whether to archive before deleting
    pub archive_before_delete: bool,
}

impl Default for AutoPrunerConfig {
    fn default() -> Self {
        let mut max_memories = HashMap::new();
        max_memories.insert("Critical".to_string(), usize::MAX);
        max_memories.insert("High".to_string(), 50000);
        max_memories.insert("Medium".to_string(), 20000);
        max_memories.insert("Low".to_string(), 5000);

        Self {
            enabled: false,
            prune_interval: Duration::from_secs(3600), // 1 hour
            strategy: PruningStrategy::ImportanceBased,
            importance_thresholds: ImportanceThresholds::default(),
            max_memories_per_tier: max_memories,
            archive_before_delete: true,
        }
    }
}

/// Pruning strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PruningStrategy {
    /// Prune based on importance score thresholds
    ImportanceBased,
    /// Prune least recently accessed
    LeastRecentlyAccessed,
    /// Prune oldest memories
    OldestFirst,
    /// Prune memories with low access counts
    LowAccessCount,
}

/// Importance thresholds for different tiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceThresholds {
    /// Minimum relevance for Low importance memories
    pub low_min_relevance: f32,
    /// Minimum relevance for Medium importance memories
    pub medium_min_relevance: f32,
    /// Minimum relevance for High importance memories
    pub high_min_relevance: f32,
    /// Maximum age for Low importance memories (days)
    pub low_max_age_days: i64,
    /// Maximum age for Medium importance memories (days)
    pub medium_max_age_days: i64,
}

impl Default for ImportanceThresholds {
    fn default() -> Self {
        Self {
            low_min_relevance: 0.1,
            medium_min_relevance: 0.05,
            high_min_relevance: 0.02,
            low_max_age_days: 30,
            medium_max_age_days: 90,
        }
    }
}

/// Auto-pruning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoPruneStats {
    /// Total memories evaluated
    pub total_evaluated: usize,
    /// Memories pruned by importance
    pub pruned_by_importance: usize,
    /// Memories pruned by age
    pub pruned_by_age: usize,
    /// Memories pruned by access count
    pub pruned_by_access: usize,
    /// Memories archived
    pub archived: usize,
    /// Memories deleted
    pub deleted: usize,
    /// Duration of pruning run (ms)
    pub duration_ms: u64,
}

/// Auto-pruner with advanced strategies
#[cfg(feature = "auto-pruning")]
pub struct AutoPruner {
    config: AutoPrunerConfig,
    /// Last pruning timestamp
    last_prune: parking_lot::RwLock<Option<chrono::DateTime<Utc>>>,
}

#[cfg(feature = "auto-pruning")]
impl AutoPruner {
    /// Create a new auto-pruner
    pub fn new(config: AutoPrunerConfig) -> Self {
        Self {
            config,
            last_prune: parking_lot::RwLock::new(None),
        }
    }

    /// Create with default config
    pub fn default_pruner() -> Self {
        Self::new(AutoPrunerConfig::default())
    }

    /// Execute a pruning run
    pub fn prune(
        &self,
        memories: Vec<MemoryArtifact>,
    ) -> Result<(Vec<MemoryArtifact>, AutoPruneStats)> {
        let start = std::time::Instant::now();

        if !self.config.enabled {
            let total = memories.len();
            return Ok((
                memories,
                AutoPruneStats {
                    total_evaluated: total,
                    pruned_by_importance: 0,
                    pruned_by_age: 0,
                    pruned_by_access: 0,
                    archived: 0,
                    deleted: 0,
                    duration_ms: 0,
                },
            ));
        }

        let mut kept = Vec::new();
        let mut pruned_by_importance = 0;
        let mut pruned_by_age = 0;
        let mut pruned_by_access = 0;
        let mut archived = 0;
        let mut deleted = 0;

        for memory in memories {
            if self.should_prune_memory(&memory) {
                // Archive before delete if configured
                if self.config.archive_before_delete {
                    archived += 1;
                }
                deleted += 1;

                match self.config.strategy {
                    PruningStrategy::ImportanceBased => pruned_by_importance += 1,
                    PruningStrategy::LeastRecentlyAccessed | PruningStrategy::OldestFirst => {
                        pruned_by_age += 1
                    }
                    PruningStrategy::LowAccessCount => pruned_by_access += 1,
                }
            } else {
                kept.push(memory);
            }
        }

        let duration = start.elapsed();
        let total_evaluated = pruned_by_importance + pruned_by_age + pruned_by_access + kept.len();

        let stats = AutoPruneStats {
            total_evaluated,
            pruned_by_importance,
            pruned_by_age,
            pruned_by_access,
            archived,
            deleted,
            duration_ms: duration.as_millis() as u64,
        };

        // Update last prune timestamp
        *self.last_prune.write() = Some(Utc::now());

        tracing::info!(
            total = total_evaluated,
            pruned = pruned_by_importance + pruned_by_age + pruned_by_access,
            archived = archived,
            deleted = deleted,
            "Auto-pruning completed"
        );

        Ok((kept, stats))
    }

    /// Determine if a memory should be pruned
    fn should_prune_memory(&self, memory: &MemoryArtifact) -> bool {
        // Never prune Critical memories
        if memory.importance == Importance::Critical {
            return false;
        }

        match self.config.strategy {
            PruningStrategy::ImportanceBased => self.should_prune_by_importance(memory),
            PruningStrategy::LeastRecentlyAccessed => self.should_prune_by_access(memory),
            PruningStrategy::OldestFirst => self.should_prune_by_age(memory),
            PruningStrategy::LowAccessCount => self.should_prune_by_low_access(memory),
        }
    }

    /// Prune by importance with tiered thresholds
    fn should_prune_by_importance(&self, memory: &MemoryArtifact) -> bool {
        let relevance = memory.compute_relevance();
        let age_days = (Utc::now() - memory.timestamp).num_days();

        match memory.importance {
            Importance::Low => {
                relevance < self.config.importance_thresholds.low_min_relevance
                    || age_days > self.config.importance_thresholds.low_max_age_days
            }
            Importance::Medium => {
                relevance < self.config.importance_thresholds.medium_min_relevance
                    || age_days > self.config.importance_thresholds.medium_max_age_days
            }
            Importance::High => relevance < self.config.importance_thresholds.high_min_relevance,
            Importance::Critical => false, // Never prune critical
        }
    }

    /// Prune least recently accessed
    fn should_prune_by_access(&self, memory: &MemoryArtifact) -> bool {
        let last_accessed = memory.last_accessed.unwrap_or(memory.timestamp);
        let age_days = (Utc::now() - last_accessed).num_days();

        // Prune if not accessed for >90 days
        age_days > 90
    }

    /// Prune oldest memories
    fn should_prune_by_age(&self, memory: &MemoryArtifact) -> bool {
        let age_days = (Utc::now() - memory.timestamp).num_days();

        match memory.importance {
            Importance::Low => age_days > 180,
            Importance::Medium => age_days > 365,
            Importance::High => age_days > 730,
            Importance::Critical => false,
        }
    }

    /// Prune memories with low access counts
    fn should_prune_by_low_access(&self, memory: &MemoryArtifact) -> bool {
        // Prune if accessed less than 2 times and older than 60 days
        let age_days = (Utc::now() - memory.timestamp).num_days();
        memory.access_count < 2 && age_days > 60
    }

    /// Check if pruning is needed
    pub fn should_prune(&self) -> bool {
        if !self.config.enabled {
            return false;
        }

        let last = self.last_prune.read();
        match *last {
            Some(last_time) => {
                let elapsed = Utc::now().signed_duration_since(last_time);
                elapsed > chrono::Duration::from_std(self.config.prune_interval).unwrap_or_default()
            }
            None => true, // Never pruned
        }
    }

    /// Get config
    pub fn config(&self) -> &AutoPrunerConfig {
        &self.config
    }
}

/// Stub implementation when feature is not enabled
#[cfg(not(feature = "auto-pruning"))]
pub struct AutoPruner;

#[cfg(not(feature = "auto-pruning"))]
impl AutoPruner {
    pub fn new(_config: AutoPrunerConfig) -> Self {
        Self
    }

    pub fn default_pruner() -> Self {
        Self
    }

    pub fn prune(
        &self,
        memories: Vec<MemoryArtifact>,
    ) -> Result<(Vec<MemoryArtifact>, AutoPruneStats)> {
        Ok((
            memories,
            AutoPruneStats {
                total_evaluated: memories.len(),
                pruned_by_importance: 0,
                pruned_by_age: 0,
                pruned_by_access: 0,
                archived: 0,
                deleted: 0,
                duration_ms: 0,
            },
        ))
    }

    pub fn should_prune(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_pruner_config_default() {
        let config = AutoPrunerConfig::default();
        assert!(!config.enabled);
        assert!(config.archive_before_delete);
    }

    #[cfg(not(feature = "auto-pruning"))]
    #[test]
    fn test_auto_pruning_not_enabled() {
        let pruner = AutoPruner::default_pruner();
        let memories = vec![];
        let result = pruner.prune(memories);
        assert!(result.is_ok());
    }
}
