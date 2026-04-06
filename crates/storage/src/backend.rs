//! Storage backend trait for pluggable implementations.

use rememnemosyne_core::{MemoryError, Result};

/// Core trait for storage backends.
///
/// All storage implementations must be:
/// - Thread-safe (Send + Sync)
/// - Key-value oriented
///
/// Note: Generic serialization methods are provided as free functions
/// to maintain dyn-compatibility of the trait.
pub trait StorageBackend: Send + Sync {
    /// Store a key-value pair
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()>;

    /// Retrieve a value by key
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;

    /// Delete a key-value pair
    fn delete(&self, key: &[u8]) -> Result<()>;

    /// Check if key exists
    fn exists(&self, key: &[u8]) -> Result<bool>;

    /// Flush pending writes to disk
    fn flush(&self) -> Result<()>;

    /// Get all keys (use with caution on large datasets)
    fn keys(&self) -> Result<Vec<Vec<u8>>>;

    /// Get number of entries
    fn len(&self) -> Result<usize>;

    /// Check if empty
    fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Clear all data
    fn clear(&self) -> Result<()>;

    /// Get a range of values (if supported by backend)
    fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Compact the database (if supported)
    fn compact(&self) -> Result<()> {
        Ok(())
    }
}

/// Helper functions for serialization (used with StorageBackend trait objects)
pub mod helpers {
    use super::*;
    use serde::{de::DeserializeOwned, Serialize};

    /// Store a serializable value
    pub fn put_serialized<S: StorageBackend, T: Serialize>(
        backend: &S,
        key: &[u8],
        value: &T,
    ) -> Result<()> {
        let bytes =
            bincode::serialize(value).map_err(|e| MemoryError::Serialization(e.to_string()))?;
        backend.put(key, &bytes)
    }

    /// Get and deserialize a value
    pub fn get_deserialized<S: StorageBackend, T: DeserializeOwned>(
        backend: &S,
        key: &[u8],
    ) -> Result<Option<T>> {
        match backend.get(key)? {
            Some(bytes) => {
                let value = bincode::deserialize(&bytes)
                    .map_err(|e| MemoryError::Serialization(e.to_string()))?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }
}

/// Trait for transactional storage (optional capability)
pub trait TransactionalStorage: StorageBackend {
    /// Start a new transaction
    fn begin_transaction(&self) -> Result<Box<dyn StorageTransaction>>;

    /// Check if transactions are supported
    fn supports_transactions(&self) -> bool {
        true
    }
}

/// Transaction interface
pub trait StorageTransaction {
    /// Put within transaction
    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()>;

    /// Delete within transaction
    fn delete(&mut self, key: &[u8]) -> Result<()>;

    /// Commit the transaction
    fn commit(self: Box<Self>) -> Result<()>;

    /// Rollback the transaction
    fn rollback(self: Box<Self>) -> Result<()>;
}
