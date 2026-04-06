use chrono::{DateTime, Utc};
use mnemosyne_core::{MemoryError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Snapshot manager for memory state persistence
pub struct SnapshotManager {
    base_path: PathBuf,
    snapshots: HashMap<String, SnapshotInfo>,
}

impl SnapshotManager {
    pub fn new(base_path: impl Into<PathBuf>) -> Result<Self> {
        let base_path = base_path.into();

        if !base_path.exists() {
            std::fs::create_dir_all(&base_path).map_err(|e| {
                MemoryError::Storage(format!("Failed to create snapshot dir: {}", e))
            })?;
        }

        let snapshots = Self::load_snapshot_list(&base_path)?;

        Ok(Self {
            base_path,
            snapshots,
        })
    }

    /// Create a new snapshot
    pub fn create_snapshot(&mut self, name: &str, data: &impl Serialize) -> Result<SnapshotInfo> {
        let snapshot_path = self.base_path.join(format!("{}.bin", name));
        let meta_path = self.base_path.join(format!("{}.meta.json", name));

        // Serialize data
        let serialized = bincode::serialize(data)
            .map_err(|e| MemoryError::Serialization(format!("Failed to serialize: {}", e)))?;

        // Write data file
        std::fs::write(&snapshot_path, &serialized)
            .map_err(|e| MemoryError::Storage(format!("Failed to write snapshot: {}", e)))?;

        // Create metadata
        let info = SnapshotInfo {
            name: name.to_string(),
            created_at: Utc::now(),
            size_bytes: serialized.len(),
            data_path: snapshot_path.to_string_lossy().to_string(),
        };

        // Write metadata file
        let meta_json = serde_json::to_string(&info).map_err(|e| {
            MemoryError::Serialization(format!("Failed to serialize metadata: {}", e))
        })?;

        std::fs::write(&meta_path, meta_json)
            .map_err(|e| MemoryError::Storage(format!("Failed to write metadata: {}", e)))?;

        self.snapshots.insert(name.to_string(), info.clone());
        Ok(info)
    }

    /// Load a snapshot
    pub fn load_snapshot<T: for<'de> Deserialize<'de>>(&self, name: &str) -> Result<T> {
        let info = self
            .snapshots
            .get(name)
            .ok_or_else(|| MemoryError::NotFound(format!("Snapshot not found: {}", name)))?;

        let data = std::fs::read(&info.data_path)
            .map_err(|e| MemoryError::Storage(format!("Failed to read snapshot: {}", e)))?;

        let deserialized: T = bincode::deserialize(&data)
            .map_err(|e| MemoryError::Serialization(format!("Failed to deserialize: {}", e)))?;

        Ok(deserialized)
    }

    /// Delete a snapshot
    pub fn delete_snapshot(&mut self, name: &str) -> Result<()> {
        let info = self
            .snapshots
            .remove(name)
            .ok_or_else(|| MemoryError::NotFound(format!("Snapshot not found: {}", name)))?;

        // Remove data file
        let data_path = Path::new(&info.data_path);
        if data_path.exists() {
            std::fs::remove_file(data_path)
                .map_err(|e| MemoryError::Storage(format!("Failed to remove data file: {}", e)))?;
        }

        // Remove metadata file
        let meta_path = self.base_path.join(format!("{}.meta.json", name));
        if meta_path.exists() {
            std::fs::remove_file(meta_path).map_err(|e| {
                MemoryError::Storage(format!("Failed to remove metadata file: {}", e))
            })?;
        }

        Ok(())
    }

    /// List all snapshots
    pub fn list_snapshots(&self) -> Vec<&SnapshotInfo> {
        self.snapshots.values().collect()
    }

    /// Get snapshot info
    pub fn get_snapshot_info(&self, name: &str) -> Option<&SnapshotInfo> {
        self.snapshots.get(name)
    }

    /// Export snapshot to a different format
    pub fn export_snapshot(&self, name: &str, export_path: &Path) -> Result<()> {
        let info = self
            .snapshots
            .get(name)
            .ok_or_else(|| MemoryError::NotFound(format!("Snapshot not found: {}", name)))?;

        let data = std::fs::read(&info.data_path)
            .map_err(|e| MemoryError::Storage(format!("Failed to read snapshot: {}", e)))?;

        std::fs::write(export_path, &data)
            .map_err(|e| MemoryError::Storage(format!("Failed to write export: {}", e)))?;

        Ok(())
    }

    /// Import snapshot from file
    pub fn import_snapshot(&mut self, name: &str, import_path: &Path) -> Result<()> {
        let data = std::fs::read(import_path)
            .map_err(|e| MemoryError::Storage(format!("Failed to read import: {}", e)))?;

        let snapshot_path = self.base_path.join(format!("{}.bin", name));
        std::fs::write(&snapshot_path, &data)
            .map_err(|e| MemoryError::Storage(format!("Failed to write snapshot: {}", e)))?;

        let info = SnapshotInfo {
            name: name.to_string(),
            created_at: Utc::now(),
            size_bytes: data.len(),
            data_path: snapshot_path.to_string_lossy().to_string(),
        };

        // Write metadata
        let meta_path = self.base_path.join(format!("{}.meta.json", name));
        let meta_json = serde_json::to_string(&info).map_err(|e| {
            MemoryError::Serialization(format!("Failed to serialize metadata: {}", e))
        })?;
        std::fs::write(&meta_path, meta_json)
            .map_err(|e| MemoryError::Storage(format!("Failed to write metadata: {}", e)))?;

        self.snapshots.insert(name.to_string(), info);
        Ok(())
    }

    fn load_snapshot_list(base_path: &Path) -> Result<HashMap<String, SnapshotInfo>> {
        let mut snapshots = HashMap::new();

        if !base_path.exists() {
            return Ok(snapshots);
        }

        for entry in std::fs::read_dir(base_path)
            .map_err(|e| MemoryError::Storage(format!("Failed to read dir: {}", e)))?
        {
            let entry =
                entry.map_err(|e| MemoryError::Storage(format!("Dir entry error: {}", e)))?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("meta.json") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(info) = serde_json::from_str::<SnapshotInfo>(&content) {
                        snapshots.insert(info.name.clone(), info);
                    }
                }
            }
        }

        Ok(snapshots)
    }
}

/// Snapshot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotInfo {
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub size_bytes: usize,
    pub data_path: String,
}
