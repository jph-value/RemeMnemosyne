/// Read replicas for scaling read operations
///
/// This module provides read replica support for the storage layer,
/// enabling horizontal scaling of read operations.
/// Enabled with the `read-replicas` feature flag.
use rememnemosyne_core::{MemoryError, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::backend::StorageBackend;

/// Read replica configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadReplicaConfig {
    /// Enable read replicas
    pub enabled: bool,
    /// Number of read replicas
    pub replica_count: usize,
    /// Replica paths (if None, will be auto-generated)
    pub replica_paths: Option<Vec<String>>,
    /// Replication strategy
    pub replication_strategy: ReplicationStrategy,
}

impl Default for ReadReplicaConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            replica_count: 0,
            replica_paths: None,
            replication_strategy: ReplicationStrategy::Synchronous,
        }
    }
}

/// Replication strategy
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ReplicationStrategy {
    /// Synchronous replication (writes wait for replicas)
    Synchronous,
    /// Asynchronous replication (writes don't wait)
    Asynchronous,
}

/// Read replica manager
#[cfg(feature = "read-replicas")]
pub struct ReadReplicaManager {
    config: ReadReplicaConfig,
    /// Primary storage (writable)
    primary: Arc<dyn StorageBackend + Send + Sync>,
    /// Read replicas (read-only)
    replicas: Vec<Arc<dyn StorageBackend + Send + Sync>>,
    /// Current replica index for round-robin
    current_replica: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "read-replicas")]
impl ReadReplicaManager {
    /// Create a new read replica manager
    pub fn new(
        config: ReadReplicaConfig,
        primary: Arc<dyn StorageBackend + Send + Sync>,
        replicas: Vec<Arc<dyn StorageBackend + Send + Sync>>,
    ) -> Self {
        Self {
            config,
            primary,
            replicas,
            current_replica: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Create from config and primary
    pub fn from_config(
        config: ReadReplicaConfig,
        primary: Arc<dyn StorageBackend + Send + Sync>,
    ) -> Result<Self> {
        if !config.enabled || config.replica_count == 0 {
            // No replicas needed
            return Ok(Self::new(config, primary, Vec::new()));
        }

        // Create replicas
        let mut replicas = Vec::new();
        let paths = config.replica_paths.clone().unwrap_or_else(|| {
            // Auto-generate paths
            (0..config.replica_count)
                .map(|i| format!("./replica_{}", i))
                .collect()
        });

        for path in &paths {
            #[cfg(feature = "sled-storage")]
            {
                let replica = crate::SledStorage::new(path)?;
                replicas.push(Arc::new(replica) as Arc<dyn StorageBackend + Send + Sync>);
            }

            #[cfg(not(feature = "sled-storage"))]
            {
                let _ = path;
                return Err(MemoryError::Storage(
                    "No storage backend available for replicas".into(),
                ));
            }
        }

        Ok(Self::new(config, primary, replicas))
    }

    /// Write to primary (and replicas if synchronous)
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        // Write to primary
        self.primary.put(key, value)?;

        // If synchronous replication, write to all replicas
        if self.config.replication_strategy == ReplicationStrategy::Synchronous {
            for replica in &self.replicas {
                replica.put(key, value)?;
            }
        }

        Ok(())
    }

    /// Read from a replica (round-robin)
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if self.replicas.is_empty() {
            // No replicas, read from primary
            return self.primary.get(key);
        }

        // Round-robin across replicas
        let index = self
            .current_replica
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.replicas.len();

        self.replicas[index].get(key)
    }

    /// Delete from primary and all replicas
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.primary.delete(key)?;

        for replica in &self.replicas {
            let _ = replica.delete(key); // Best effort
        }

        Ok(())
    }

    /// Flush all stores
    pub fn flush(&self) -> Result<()> {
        self.primary.flush()?;

        for replica in &self.replicas {
            let _ = replica.flush(); // Best effort
        }

        Ok(())
    }

    /// Get number of replicas
    pub fn replica_count(&self) -> usize {
        self.replicas.len()
    }

    /// Check if using replicas
    pub fn has_replicas(&self) -> bool {
        !self.replicas.is_empty()
    }

    /// Get primary storage
    pub fn primary(&self) -> &Arc<dyn StorageBackend + Send + Sync> {
        &self.primary
    }
}

/// Stub implementation when feature is not enabled
#[cfg(not(feature = "read-replicas"))]
pub struct ReadReplicaManager;

#[cfg(not(feature = "read-replicas"))]
impl ReadReplicaManager {
    pub fn new(
        _config: ReadReplicaConfig,
        _primary: Arc<dyn StorageBackend + Send + Sync>,
        _replicas: Vec<Arc<dyn StorageBackend + Send + Sync>>,
    ) -> Self {
        Self
    }

    pub fn from_config(
        _config: ReadReplicaConfig,
        _primary: Arc<dyn StorageBackend + Send + Sync>,
    ) -> Result<Self> {
        Ok(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_replica_config_default() {
        let config = ReadReplicaConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.replica_count, 0);
    }

    #[cfg(not(feature = "read-replicas"))]
    #[test]
    fn test_read_replicas_not_enabled() {
        // Just test that stub exists
    }
}
