/// JSON backup/export and import functionality
///
/// This module provides comprehensive JSON-based backup and export capabilities
/// for all memory data. It enables interoperability and data migration.
/// Enabled with the `backup-export` feature flag.
use chrono::{DateTime, Utc};
use rememnemosyne_core::{EntityId, MemoryArtifact, MemoryError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Complete backup of all memory data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBackup {
    /// Backup metadata
    pub metadata: BackupMetadata,
    /// All memory artifacts
    pub memories: Vec<MemoryArtifact>,
    /// All entities (from graph memory)
    pub entities: Vec<serde_json::Value>,
    /// All relationships (from graph memory)
    pub relationships: Vec<serde_json::Value>,
}

/// Backup metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// Backup creation timestamp
    pub created_at: DateTime<Utc>,
    /// Version of backup format
    pub version: String,
    /// Total number of memories
    pub memory_count: usize,
    /// Total number of entities
    pub entity_count: usize,
    /// Total number of relationships
    pub relationship_count: usize,
    /// Optional description
    pub description: Option<String>,
    /// Source system identifier
    pub source: String,
}

impl Default for BackupMetadata {
    fn default() -> Self {
        Self {
            created_at: Utc::now(),
            version: "1.0.0".to_string(),
            memory_count: 0,
            entity_count: 0,
            relationship_count: 0,
            description: None,
            source: "rememnemosyne".to_string(),
        }
    }
}

/// Backup manager for export/import operations
pub struct BackupManager {
    /// Optional backup directory
    backup_dir: Option<PathBuf>,
}

impl BackupManager {
    /// Create a new backup manager
    pub fn new(backup_dir: Option<PathBuf>) -> Self {
        Self { backup_dir }
    }

    /// Create with default (no backup directory)
    pub fn default_manager() -> Self {
        Self::new(None)
    }

    /// Export all memories to JSON backup structure
    pub fn create_backup(
        &self,
        memories: Vec<MemoryArtifact>,
        entities: Vec<serde_json::Value>,
        relationships: Vec<serde_json::Value>,
    ) -> MemoryBackup {
        let metadata = BackupMetadata {
            memory_count: memories.len(),
            entity_count: entities.len(),
            relationship_count: relationships.len(),
            ..Default::default()
        };

        MemoryBackup {
            metadata,
            memories,
            entities,
            relationships,
        }
    }

    /// Export to JSON file
    pub async fn export_to_file(&self, backup: &MemoryBackup, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(backup).map_err(|e| {
            MemoryError::Serialization(format!("Failed to serialize backup: {}", e))
        })?;

        tokio::fs::write(path, json)
            .await
            .map_err(|e| MemoryError::Io(e))?;

        tracing::info!(
            path = ?path,
            memory_count = backup.metadata.memory_count,
            "Backup exported to file"
        );

        Ok(())
    }

    /// Import from JSON string
    pub fn import_from_json(&self, json: &str) -> Result<MemoryBackup> {
        let backup: MemoryBackup = serde_json::from_str(json).map_err(|e| {
            MemoryError::Serialization(format!("Failed to deserialize backup: {}", e))
        })?;

        tracing::info!(
            memory_count = backup.metadata.memory_count,
            entity_count = backup.metadata.entity_count,
            "Backup imported successfully"
        );

        Ok(backup)
    }

    /// Import from JSON file
    pub async fn import_from_file(&self, path: &Path) -> Result<MemoryBackup> {
        let json = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| MemoryError::Io(e))?;

        self.import_from_json(&json)
    }

    /// Export memories as NDJSON (newline-delimited JSON) for streaming
    pub fn export_to_ndjson(&self, memories: &[MemoryArtifact]) -> Result<String> {
        let mut ndjson = String::new();

        for memory in memories {
            let line = serde_json::to_string(memory).map_err(|e| {
                MemoryError::Serialization(format!("Failed to serialize memory: {}", e))
            })?;
            ndjson.push_str(&line);
            ndjson.push('\n');
        }

        Ok(ndjson)
    }

    /// Import from NDJSON
    pub fn import_from_ndjson(&self, ndjson: &str) -> Result<Vec<MemoryArtifact>> {
        let mut memories = Vec::new();

        for line in ndjson.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let memory: MemoryArtifact = serde_json::from_str(line).map_err(|e| {
                MemoryError::Serialization(format!("Failed to deserialize memory: {}", e))
            })?;

            memories.push(memory);
        }

        Ok(memories)
    }

    /// Get backup directory
    pub fn backup_dir(&self) -> Option<&PathBuf> {
        self.backup_dir.as_ref()
    }

    /// Set backup directory
    pub fn set_backup_dir(&mut self, path: PathBuf) {
        self.backup_dir = Some(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backup_metadata_creation() {
        let metadata = BackupMetadata::default();
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.memory_count, 0);
    }

    #[test]
    fn test_backup_manager_creation() {
        let manager = BackupManager::default_manager();
        assert!(manager.backup_dir().is_none());
    }

    #[test]
    fn test_ndjson_roundtrip() {
        let manager = BackupManager::default_manager();

        // Test that NDJSON export/import works
        let ndjson = manager.export_to_ndjson(&[]).unwrap();
        let memories = manager.import_from_ndjson(&ndjson).unwrap();
        assert!(memories.is_empty());
    }
}
