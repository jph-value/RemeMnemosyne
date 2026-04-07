/// Memory lifecycle management: importance decay, pruning, deduplication.
/// Runs periodically to keep the memory system healthy at scale.
use chrono::Utc;
#[cfg(test)]
use chrono::Duration;
use rememnemosyne_core::{Importance, MemoryArtifact, MemoryId};
use rememnemosyne_storage::archive::{ArchiveCatalog, ArchiveConfig, ArchiveStats, MemoryArchive};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for memory lifecycle management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrunerConfig {
    /// Importance threshold below which memories are pruned
    pub min_importance: f32,
    /// Maximum age in days before low-importance memories are pruned
    pub max_age_days: i64,
    /// Maximum number of memories to keep (0 = unlimited)
    pub max_memories: usize,
    /// Decay factor applied to importance each run (0.0-1.0)
    pub decay_factor: f32,
    /// Minimum access count to survive pruning
    pub min_access_count: u64,
    /// Whether to archive (soft-delete) vs hard-delete
    pub archive_mode: bool,
}

impl Default for PrunerConfig {
    fn default() -> Self {
        Self {
            min_importance: 0.05,
            max_age_days: 90,
            max_memories: 100_000,
            decay_factor: 0.995,
            min_access_count: 0,
            archive_mode: true,
        }
    }
}

/// Memory pruner that manages the lifecycle of stored memories
pub struct MemoryPruner {
    config: PrunerConfig,
    archive: Option<MemoryArchive>,
}

/// Result of a pruning run
#[derive(Debug, Clone)]
pub struct PruneResult {
    pub total_memories: usize,
    pub pruned_count: usize,
    pub decayed_count: usize,
    pub archived_count: usize,
    pub archive_stats: Option<ArchiveRunStats>,
    pub elapsed_ms: f64,
}

/// Archive statistics from a prune run
#[derive(Debug, Clone)]
pub struct ArchiveRunStats {
    pub memories_archived: usize,
    pub original_bytes: u64,
    pub compressed_bytes: u64,
    pub compression_ratio: f64,
}

impl MemoryPruner {
    pub fn new(config: PrunerConfig) -> Self {
        // Open archive if archive mode is enabled
        let archive = if config.archive_mode {
            let archive_config = ArchiveConfig {
                archive_dir: PathBuf::from("./rememnemosyne_data/archive"),
                ..Default::default()
            };
            MemoryArchive::open(archive_config).ok()
        } else {
            None
        };

        Self { config, archive }
    }

    pub fn with_archive_dir(config: PrunerConfig, archive_dir: PathBuf) -> Self {
        let archive = if config.archive_mode {
            let archive_config = ArchiveConfig {
                archive_dir,
                ..Default::default()
            };
            MemoryArchive::open(archive_config).ok()
        } else {
            None
        };

        Self { config, archive }
    }

    pub fn default() -> Self {
        Self::new(PrunerConfig::default())
    }

    /// Evaluate a single memory for pruning.
    /// Returns true if the memory should be kept.
    pub fn should_keep(&self, memory: &MemoryArtifact) -> bool {
        // Always keep critical memories
        if memory.importance == Importance::Critical {
            return true;
        }

        // Check access count
        if memory.access_count < self.config.min_access_count {
            return false;
        }

        // Check age for low-importance memories
        let age_days = (Utc::now() - memory.timestamp).num_days();
        if age_days > self.config.max_age_days {
            let computed_relevance = memory.compute_relevance();
            if computed_relevance < self.config.min_importance {
                return false;
            }
        }

        true
    }

    /// Compute decayed importance for a memory
    pub fn decay_importance(&self, memory: &mut MemoryArtifact) {
        let age_hours = (Utc::now() - memory.timestamp).num_hours() as f32;
        let decay_cycles = age_hours / 24.0; // One cycle per day
        let decay = self.config.decay_factor.powf(decay_cycles);

        // Update metadata with decayed relevance
        memory.metadata.insert(
            "decayed_relevance".to_string(),
            serde_json::json!(memory.compute_relevance() * decay),
        );
    }

    /// Check for duplicate memories based on embedding similarity
    pub fn find_duplicates(
        &self,
        memories: &[MemoryArtifact],
        similarity_threshold: f32,
    ) -> Vec<(MemoryId, MemoryId)> {
        let mut duplicates = Vec::new();

        for i in 0..memories.len() {
            if memories[i].embedding.is_empty() {
                continue;
            }
            for j in (i + 1)..memories.len() {
                if memories[j].embedding.is_empty() {
                    continue;
                }
                let sim = cosine_similarity(&memories[i].embedding, &memories[j].embedding);
                if sim > similarity_threshold {
                    duplicates.push((memories[i].id, memories[j].id));
                }
            }
        }

        duplicates
    }

    /// Get configuration
    pub fn config(&self) -> &PrunerConfig {
        &self.config
    }

    /// Archive a memory that is being pruned.
    /// Stores it with zstd compression in the archive directory.
    pub fn archive_memory(&mut self, memory: &MemoryArtifact) -> std::io::Result<bool> {
        if let Some(ref mut archive) = self.archive {
            archive.archive_memory(memory)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Archive a batch of memories
    pub fn archive_batch(
        &mut self,
        memories: &[MemoryArtifact],
    ) -> std::io::Result<Option<ArchiveStats>> {
        if let Some(ref mut archive) = self.archive {
            let stats = archive.archive_batch(memories)?;
            Ok(Some(stats))
        } else {
            Ok(None)
        }
    }

    /// Decompress a single archived memory by ID (selective decompression)
    pub fn decompress_archived(&self, id: &MemoryId) -> std::io::Result<Option<MemoryArtifact>> {
        if let Some(ref archive) = self.archive {
            archive.decompress_memory(id)
        } else {
            Ok(None)
        }
    }

    /// Search archived memories by metadata WITHOUT decompressing data
    pub fn search_archived(
        &self,
        query: &str,
        tags: Option<&[String]>,
        min_importance: Option<Importance>,
    ) -> Vec<&rememnemosyne_storage::archive::ArchiveEntry> {
        if let Some(ref archive) = self.archive {
            archive.search_by_metadata(query, tags, min_importance)
        } else {
            Vec::new()
        }
    }

    /// Check if archive is available
    pub fn has_archive(&self) -> bool {
        self.archive.is_some()
    }

    /// Get archive statistics
    pub fn archive_stats(&self) -> Option<&ArchiveCatalog> {
        self.archive.as_ref().map(|a| a.stats())
    }
}

/// Simple cosine similarity for duplicate detection
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rememnemosyne_core::MemoryTrigger;

    #[test]
    fn test_should_keep_critical() {
        let pruner = MemoryPruner::default();
        let memory = MemoryArtifact::new(
            rememnemosyne_core::MemoryType::Semantic,
            "test",
            "test content",
            vec![0.1; 1536],
            MemoryTrigger::Insight,
        )
        .with_importance(Importance::Critical);

        assert!(pruner.should_keep(&memory));
    }

    #[test]
    fn test_should_prune_old_low_importance() {
        let mut pruner = MemoryPruner::default();
        pruner.config.max_age_days = 1;
        pruner.config.min_access_count = 0;
        pruner.config.min_importance = 0.5;

        let mut memory = MemoryArtifact::new(
            rememnemosyne_core::MemoryType::Semantic,
            "old",
            "old content",
            vec![0.01; 1536], // Low relevance embedding
            MemoryTrigger::UserInput,
        )
        .with_importance(Importance::Low);
        // Simulate old timestamp
        memory.timestamp = Utc::now() - Duration::days(100);

        assert!(!pruner.should_keep(&memory));
    }

    #[test]
    fn test_find_duplicates() {
        let pruner = MemoryPruner::default();
        let emb = vec![0.1; 1536];
        let memories = vec![
            MemoryArtifact::new(
                rememnemosyne_core::MemoryType::Semantic,
                "a",
                "content a",
                emb.clone(),
                MemoryTrigger::Insight,
            ),
            MemoryArtifact::new(
                rememnemosyne_core::MemoryType::Semantic,
                "b",
                "content b",
                emb.clone(),
                MemoryTrigger::Insight,
            ),
        ];

        let dups = pruner.find_duplicates(&memories, 0.9);
        assert_eq!(dups.len(), 1);
    }
}
