//! RocksDB storage backend (optional, requires C++ toolchain).
//!
//! Enable with feature: `persistence`

use rememnemosyne_core::{MemoryError, Result};

use super::backend::StorageBackend;

/// RocksDB-based storage backend.
///
/// # Requirements
///
/// This backend requires:
/// - C++ compiler (gcc/clang)
/// - CMake
/// - The rocksdb feature enabled
///
/// # When to Use
///
/// Use RocksDB when:
/// - Write-heavy workload
/// - Need for SSTable compaction
/// - High throughput requirements
/// - Already have C++ toolchain available
///
/// Otherwise, prefer the pure Rust sled backend.
#[derive(Debug)]
pub struct RocksStorage {
    db: rocksdb::DB,
    path: String,
}

impl RocksStorage {
    /// Open or create a RocksDB database.
    pub fn new(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Performance tuning
        opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB
        opts.set_max_write_buffer_number(3);
        opts.set_target_file_size_base(64 * 1024 * 1024);

        let db = rocksdb::DB::open(&opts, path.as_ref())
            .map_err(|e| MemoryError::Storage(format!("Failed to open RocksDB: {}", e)))?;

        Ok(Self { db, path: path_str })
    }

    /// Get the path of the database.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Compact the database.
    pub fn compact_range(&self, start: Option<&[u8]>, end: Option<&[u8]>) {
        self.db.compact_range(start, end);
    }

    /// Get database properties.
    pub fn property(&self, name: &str) -> Result<String> {
        self.db
            .property_value(name)
            .map_err(|e| MemoryError::Storage(format!("Property query failed: {}", e)))
            .map(|v| v.unwrap_or_default())
    }
}

impl StorageBackend for RocksStorage {
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db
            .put(key, value)
            .map_err(|e| MemoryError::Storage(format!("Put failed: {}", e)))
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.db
            .get(key)
            .map_err(|e| MemoryError::Storage(format!("Get failed: {}", e)))
    }

    fn delete(&self, key: &[u8]) -> Result<()> {
        self.db
            .delete(key)
            .map_err(|e| MemoryError::Storage(format!("Delete failed: {}", e)))
    }

    fn exists(&self, key: &[u8]) -> Result<bool> {
        self.get(key).map(|v| v.is_some())
    }

    fn flush(&self) -> Result<()> {
        self.db
            .flush()
            .map_err(|e| MemoryError::Storage(format!("Flush failed: {}", e)))
    }

    fn keys(&self) -> Result<Vec<Vec<u8>>> {
        use rocksdb::IteratorMode;

        let mut keys = Vec::new();
        for item in self.db.iterator(IteratorMode::Start) {
            let (key, _) =
                item.map_err(|e| MemoryError::Storage(format!("Iterator error: {}", e)))?;
            keys.push(key.to_vec());
        }
        Ok(keys)
    }

    fn len(&self) -> Result<usize> {
        // Approximate count
        self.keys().map(|k| k.len())
    }

    fn clear(&self) -> Result<()> {
        let keys = self.keys()?;
        for key in keys {
            self.db
                .delete(&key)
                .map_err(|e| MemoryError::Storage(format!("Delete failed: {}", e)))?;
        }
        Ok(())
    }

    fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        use rocksdb::IteratorMode;

        let mut results = Vec::new();
        for item in self
            .db
            .iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward))
        {
            let (key, value) =
                item.map_err(|e| MemoryError::Storage(format!("Iterator error: {}", e)))?;

            // Stop if key doesn't have the prefix
            if !key.starts_with(prefix) {
                break;
            }

            results.push((key.to_vec(), value.to_vec()));
        }
        Ok(results)
    }

    fn compact(&self) -> Result<()> {
        self.compact_range(None, None);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rocksdb_basic() {
        // This test only runs when persistence feature is enabled
        let dir = tempfile::tempdir().unwrap();
        let storage = RocksStorage::new(dir.path()).unwrap();

        storage.put(b"key1", b"value1").unwrap();
        assert_eq!(storage.get(b"key1").unwrap(), Some(b"value1".to_vec()));
        assert!(storage.exists(b"key1").unwrap());

        storage.delete(b"key1").unwrap();
        assert!(!storage.exists(b"key1").unwrap());
    }
}
