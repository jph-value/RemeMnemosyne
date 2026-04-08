/// Memory compaction for optimizing storage
///
/// This module provides memory compaction capabilities to merge old memories,
/// optimize storage layout, and reduce fragmentation.
/// Enabled with the `compaction` feature flag.
use chrono::{DateTime, Utc};
use rememnemosyne_core::{MemoryArtifact, MemoryError, MemoryId, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// Compaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Enable automatic compaction
    pub auto_compact: bool,
    /// Interval between automatic compactions
    pub compaction_interval: Duration,
    /// Minimum age of memories to compact
    pub min_age: Duration,
    /// Merge memories with same entity
    pub merge_same_entity: bool,
    /// Maximum memories to merge at once
    pub max_merge_count: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            auto_compact: false,
            compaction_interval: Duration::from_secs(3600), // 1 hour
            min_age: Duration::from_secs(86400),            // 1 day
            merge_same_entity: true,
            max_merge_count: 10,
        }
    }
}

/// Compaction statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionStats {
    /// Memories before compaction
    pub memories_before: usize,
    /// Memories after compaction
    pub memories_after: usize,
    /// Memories merged
    pub memories_merged: usize,
    /// Memories deleted
    pub memories_deleted: usize,
    /// Space reclaimed (bytes, approximate)
    pub space_reclaimed: usize,
    /// Duration of compaction
    pub duration_ms: u64,
}

/// Memory compaction manager
#[cfg(feature = "compaction")]
pub struct MemoryCompactor {
    config: CompactionConfig,
    /// Last compaction timestamp
    last_compaction: parking_lot::RwLock<Option<DateTime<Utc>>>,
}

#[cfg(feature = "compaction")]
impl MemoryCompactor {
    /// Create a new compactor
    pub fn new(config: CompactionConfig) -> Self {
        Self {
            config,
            last_compaction: parking_lot::RwLock::new(None),
        }
    }

    /// Create with default config
    pub fn default_compactor() -> Self {
        Self::new(CompactionConfig::default())
    }

    /// Compact memories
    pub fn compact(
        &self,
        memories: Vec<MemoryArtifact>,
    ) -> Result<(Vec<MemoryArtifact>, CompactionStats)> {
        let start = std::time::Instant::now();
        let memories_before = memories.len();

        let mut compacted = Vec::new();
        let mut merged = 0;
        let mut deleted = 0;
        let mut space_reclaimed = 0;

        if self.config.merge_same_entity {
            // Group by entities
            let mut entity_groups: std::collections::HashMap<Vec<String>, Vec<MemoryArtifact>> =
                std::collections::HashMap::new();

            for memory in &memories {
                let entity_key = memory.entities.clone();
                entity_groups
                    .entry(entity_key)
                    .or_default()
                    .push(memory.clone());
            }

            // Merge groups
            for (entity_key, mut group) in entity_groups {
                if group.len() > 1 && group.len() <= self.config.max_merge_count {
                    // Check if memories are old enough
                    let now = Utc::now();
                    let should_merge = group.iter().all(|m| {
                        now.signed_duration_since(m.timestamp)
                            > chrono::Duration::from_std(self.config.min_age).unwrap_or_default()
                    });

                    if should_merge {
                        // Merge memories
                        let mut merged_memory = self.merge_memories(&group)?;
                        compacted.push(merged_memory);
                        merged += group.len() - 1;
                        deleted += group.len() - 1;
                    } else {
                        compacted.extend(group);
                    }
                } else {
                    compacted.extend(group);
                }
            }
        } else {
            // No merging, just filter old memories
            compacted = memories;
        }

        let memories_after = compacted.len();
        let duration = start.elapsed();

        let stats = CompactionStats {
            memories_before,
            memories_after,
            memories_merged: merged,
            memories_deleted: deleted,
            space_reclaimed,
            duration_ms: duration.as_millis() as u64,
        };

        // Update last compaction time
        *self.last_compaction.write() = Some(Utc::now());

        tracing::info!(
            memories_before = memories_before,
            memories_after = memories_after,
            merged = merged,
            deleted = deleted,
            "Memory compaction completed"
        );

        Ok((compacted, stats))
    }

    /// Merge multiple memories into one
    fn merge_memories(&self, memories: &[MemoryArtifact]) -> Result<MemoryArtifact> {
        if memories.is_empty() {
            return Err(MemoryError::InvalidQuery(
                "Cannot merge empty memory list".to_string(),
            ));
        }

        if memories.len() == 1 {
            return Ok(memories[0].clone());
        }

        // Take the first memory as base
        let mut base = memories[0].clone();

        // Merge content
        let mut merged_content = base.content.clone();
        let mut merged_summary = base.summary.clone();
        let mut merged_entities = base.entities.clone();
        let mut merged_tags = base.tags.clone();
        let mut merged_metadata = base.metadata.clone();

        for memory in &memories[1..] {
            merged_content.push_str("\n\n");
            merged_content.push_str(&memory.content);

            // Merge entities
            for entity in &memory.entities {
                if !merged_entities.contains(entity) {
                    merged_entities.push(entity.clone());
                }
            }

            // Merge tags
            for tag in &memory.tags {
                if !merged_tags.contains(tag) {
                    merged_tags.push(tag.clone());
                }
            }

            // Merge metadata
            for (key, value) in &memory.metadata {
                merged_metadata.insert(key.clone(), value.clone());
            }
        }

        base.content = merged_content;
        base.summary = format!("Merged from {} memories", memories.len());
        base.entities = merged_entities;
        base.tags = merged_tags;
        base.metadata = merged_metadata;

        Ok(base)
    }

    /// Check if compaction is needed
    pub fn should_compact(&self) -> bool {
        if !self.config.auto_compact {
            return false;
        }

        let last = self.last_compaction.read();
        match *last {
            Some(last_time) => {
                let elapsed = Utc::now().signed_duration_since(last_time);
                elapsed
                    > chrono::Duration::from_std(self.config.compaction_interval)
                        .unwrap_or_default()
            }
            None => true, // Never compacted
        }
    }

    /// Get compaction config
    pub fn config(&self) -> &CompactionConfig {
        &self.config
    }
}

/// Stub implementation when feature is not enabled
#[cfg(not(feature = "compaction"))]
pub struct MemoryCompactor;

#[cfg(not(feature = "compaction"))]
impl MemoryCompactor {
    pub fn new(_config: CompactionConfig) -> Self {
        Self
    }

    pub fn default_compactor() -> Self {
        Self
    }

    pub fn compact(
        &self,
        memories: Vec<MemoryArtifact>,
    ) -> Result<(Vec<MemoryArtifact>, CompactionStats)> {
        Ok((
            memories,
            CompactionStats {
                memories_before: memories.len(),
                memories_after: memories.len(),
                memories_merged: 0,
                memories_deleted: 0,
                space_reclaimed: 0,
                duration_ms: 0,
            },
        ))
    }

    pub fn should_compact(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compaction_config_default() {
        let config = CompactionConfig::default();
        assert!(!config.auto_compact);
        assert!(config.merge_same_entity);
    }

    #[cfg(not(feature = "compaction"))]
    #[test]
    fn test_compaction_not_enabled() {
        let compactor = MemoryCompactor::default_compactor();
        let memories = vec![];
        let result = compactor.compact(memories);
        assert!(result.is_ok());
    }
}
