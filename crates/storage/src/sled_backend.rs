//! Pure Rust sled storage backend.
//!
//! This is the default backend - no C/C++ dependencies required.

use rememnemosyne_core::{MemoryError, Result};
use std::path::Path;

use super::backend::StorageBackend;

/// Sled-based storage backend (pure Rust).
///
/// Features:
/// - ACID transactions
/// - Zero-copy reads
/// - Crash recovery
/// - Lock-free reads
#[derive(Debug)]
pub struct SledStorage {
    db: sled::Db,
    path: String,
}

impl SledStorage {
    /// Open or create a sled database at the given path.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let db = sled::open(path)
            .map_err(|e| MemoryError::Storage(format!("Failed to open sled: {}", e)))?;

        Ok(Self { db, path: path_str })
    }

    /// Open with custom configuration.
    pub fn with_config(path: impl AsRef<Path>, config: sled::Config) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let db = config
            .open()
            .map_err(|e| MemoryError::Storage(format!("Failed to open sled: {}", e)))?;

        Ok(Self { db, path: path_str })
    }

    /// Get the path of the database.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get the underlying sled database.
    pub fn inner(&self) -> &sled::Db {
        &self.db
    }

    /// Get database stats.
    pub fn stats(&self) -> Result<SledStats> {
        let mut key_count = 0;
        for _ in self.db.iter() {
            key_count += 1;
        }
        Ok(SledStats {
            len: self.db.len(),
            keys: key_count,
            storage_id: self.db.generate_id().unwrap_or(0),
        })
    }

    /// Perform maintenance operations.
    pub fn maintenance(&self) -> Result<()> {
        self.db
            .flush()
            .map_err(|e| MemoryError::Storage(format!("Flush failed: {}", e)))?;
        Ok(())
    }

    /// Export all data as a HashMap.
    pub fn export_all(&self) -> Result<HashMap<Vec<u8>, Vec<u8>>> {
        let mut map = HashMap::new();
        for entry in self.db.iter() {
            let (key, value) =
                entry.map_err(|e| MemoryError::Storage(format!("Iter error: {}", e)))?;
            map.insert(key.to_vec(), value.to_vec());
        }
        Ok(map)
    }

    /// Import data from a HashMap.
    pub fn import_all(&self, data: HashMap<Vec<u8>, Vec<u8>>) -> Result<()> {
        for (key, value) in data {
            self.db
                .insert(key, value)
                .map_err(|e| MemoryError::Storage(format!("Insert error: {}", e)))?;
        }
        self.db
            .flush()
            .map_err(|e| MemoryError::Storage(format!("Flush error: {}", e)))?;
        Ok(())
    }
}

use std::collections::HashMap;

impl StorageBackend for SledStorage {
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db
            .insert(key, value)
            .map_err(|e| MemoryError::Storage(format!("Put failed: {}", e)))?;
        Ok(())
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let result = self
            .db
            .get(key)
            .map_err(|e| MemoryError::Storage(format!("Get failed: {}", e)))?;
        Ok(result.map(|v| v.to_vec()))
    }

    fn delete(&self, key: &[u8]) -> Result<()> {
        self.db
            .remove(key)
            .map_err(|e| MemoryError::Storage(format!("Delete failed: {}", e)))?;
        Ok(())
    }

    fn exists(&self, key: &[u8]) -> Result<bool> {
        let result = self
            .db
            .contains_key(key)
            .map_err(|e| MemoryError::Storage(format!("Exists check failed: {}", e)))?;
        Ok(result)
    }

    fn flush(&self) -> Result<()> {
        self.db
            .flush()
            .map_err(|e| MemoryError::Storage(format!("Flush failed: {}", e)))?;
        Ok(())
    }

    fn keys(&self) -> Result<Vec<Vec<u8>>> {
        let mut keys = Vec::new();
        for item in self.db.iter() {
            let (key, _) = item.map_err(|e| MemoryError::Storage(format!("Iter error: {}", e)))?;
            keys.push(key.to_vec());
        }
        Ok(keys)
    }

    fn len(&self) -> Result<usize> {
        Ok(self.db.len())
    }

    fn clear(&self) -> Result<()> {
        self.db
            .clear()
            .map_err(|e| MemoryError::Storage(format!("Clear failed: {}", e)))?;
        Ok(())
    }

    fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let results: Vec<(Vec<u8>, Vec<u8>)> = self
            .db
            .scan_prefix(prefix)
            .map(|r| r.map(|(k, v)| (k.to_vec(), v.to_vec())))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| MemoryError::Storage(format!("Scan failed: {}", e)))?;
        Ok(results)
    }

    fn compact(&self) -> Result<()> {
        // Sled handles compaction internally
        self.flush()
    }
}

/// Sled database statistics.
#[derive(Debug, Clone)]
pub struct SledStats {
    pub len: usize,
    pub keys: usize,
    pub storage_id: u64,
}

/// Thread-safe wrapper for concurrent access.
#[derive(Debug, Clone)]
pub struct SharedSledStorage {
    inner: std::sync::Arc<SledStorage>,
}

impl SharedSledStorage {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let storage = SledStorage::new(path)?;
        Ok(Self {
            inner: std::sync::Arc::new(storage),
        })
    }
}

impl StorageBackend for SharedSledStorage {
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.inner.put(key, value)
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.inner.get(key)
    }

    fn delete(&self, key: &[u8]) -> Result<()> {
        self.inner.delete(key)
    }

    fn exists(&self, key: &[u8]) -> Result<bool> {
        self.inner.exists(key)
    }

    fn flush(&self) -> Result<()> {
        self.inner.flush()
    }

    fn keys(&self) -> Result<Vec<Vec<u8>>> {
        self.inner.keys()
    }

    fn len(&self) -> Result<usize> {
        self.inner.len()
    }

    fn clear(&self) -> Result<()> {
        self.inner.clear()
    }

    fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.inner.scan_prefix(prefix)
    }
}
