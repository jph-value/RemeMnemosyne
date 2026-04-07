//! Snapshot management for storage backends.

use rememnemosyne_core::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use super::backend::StorageBackend;

/// Generic snapshot manager that works with any storage backend.
pub struct SnapshotManager {
    snapshots_dir: PathBuf,
}

impl SnapshotManager {
    /// Create a new snapshot manager.
    pub fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        let snapshots_dir = base_path.as_ref().join("snapshots");
        std::fs::create_dir_all(&snapshots_dir).map_err(|e| {
            rememnemosyne_core::MemoryError::Storage(format!(
                "Failed to create snapshots dir: {}",
                e
            ))
        })?;

        Ok(Self { snapshots_dir })
    }

    /// Save storage state to a snapshot.
    pub fn save_snapshot(&self, name: &str, backend: &dyn StorageBackend) -> Result<()> {
        let snapshot_path = self.snapshots_dir.join(format!("{}.bin", name));
        let meta_path = self.snapshots_dir.join(format!("{}.meta.json", name));

        // Export all data
        let data = export_backend(backend)?;
        let serialized = bincode::serialize(&data)
            .map_err(|e| rememnemosyne_core::MemoryError::Serialization(e.to_string()))?;

        // Write data
        std::fs::write(&snapshot_path, &serialized).map_err(|e| {
            rememnemosyne_core::MemoryError::Storage(format!("Write failed: {}", e))
        })?;

        // Write metadata
        let meta = SnapshotMeta {
            name: name.to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            size_bytes: serialized.len(),
            entry_count: data.len(),
        };
        let meta_json = serde_json::to_string_pretty(&meta)
            .map_err(|e| rememnemosyne_core::MemoryError::Serialization(e.to_string()))?;
        std::fs::write(&meta_path, meta_json).map_err(|e| {
            rememnemosyne_core::MemoryError::Storage(format!("Meta write failed: {}", e))
        })?;

        Ok(())
    }

    /// Restore storage from a snapshot.
    pub fn restore_snapshot(&self, name: &str, backend: &dyn StorageBackend) -> Result<()> {
        let snapshot_path = self.snapshots_dir.join(format!("{}.bin", name));

        let data = std::fs::read(&snapshot_path)
            .map_err(|e| rememnemosyne_core::MemoryError::Storage(format!("Read failed: {}", e)))?;

        let entries: HashMap<Vec<u8>, Vec<u8>> = bincode::deserialize(&data)
            .map_err(|e| rememnemosyne_core::MemoryError::Serialization(e.to_string()))?;

        // Clear existing data
        backend.clear()?;

        // Restore all entries
        for (key, value) in entries {
            backend.put(&key, &value)?;
        }

        backend.flush()?;
        Ok(())
    }

    /// List available snapshots.
    pub fn list_snapshots(&self) -> Result<Vec<SnapshotInfo>> {
        let mut snapshots = Vec::new();

        let entries = std::fs::read_dir(&self.snapshots_dir).map_err(|e| {
            rememnemosyne_core::MemoryError::Storage(format!("Dir read failed: {}", e))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                rememnemosyne_core::MemoryError::Storage(format!("Entry error: {}", e))
            })?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("meta") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(meta) = serde_json::from_str::<SnapshotMeta>(&content) {
                        snapshots.push(SnapshotInfo {
                            name: meta.name,
                            created_at: meta.created_at,
                            size_bytes: meta.size_bytes,
                            entry_count: meta.entry_count,
                        });
                    }
                }
            }
        }

        snapshots.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(snapshots)
    }

    /// Delete a snapshot.
    pub fn delete_snapshot(&self, name: &str) -> Result<()> {
        let data_path = self.snapshots_dir.join(format!("{}.bin", name));
        let meta_path = self.snapshots_dir.join(format!("{}.meta.json", name));

        if data_path.exists() {
            std::fs::remove_file(&data_path).map_err(|e| {
                rememnemosyne_core::MemoryError::Storage(format!("Remove failed: {}", e))
            })?;
        }
        if meta_path.exists() {
            std::fs::remove_file(&meta_path).map_err(|e| {
                rememnemosyne_core::MemoryError::Storage(format!("Remove failed: {}", e))
            })?;
        }

        Ok(())
    }
}

use std::collections::HashMap;

/// Export all data from a backend.
fn export_backend(backend: &dyn StorageBackend) -> Result<HashMap<Vec<u8>, Vec<u8>>> {
    let mut data = HashMap::new();
    for key in backend.keys()? {
        if let Some(value) = backend.get(&key)? {
            data.insert(key, value);
        }
    }
    Ok(data)
}

/// Snapshot metadata.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct SnapshotMeta {
    pub name: String,
    pub created_at: String,
    pub size_bytes: usize,
    pub entry_count: usize,
}

/// Snapshot information (without path).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotInfo {
    pub name: String,
    pub created_at: String,
    pub size_bytes: usize,
    pub entry_count: usize,
}
