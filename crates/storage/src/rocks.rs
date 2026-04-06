use mnemosyne_core::{MemoryArtifact, MemoryError, MemoryId, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// RocksDB storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocksConfig {
    pub path: String,
    pub create_if_missing: bool,
    pub max_open_files: i32,
    pub write_buffer_size: usize,
    pub max_write_buffer_number: i32,
    pub compression: bool,
}

impl Default for RocksConfig {
    fn default() -> Self {
        Self {
            path: "./mnemosyne_data".to_string(),
            create_if_missing: true,
            max_open_files: 1000,
            write_buffer_size: 64 * 1024 * 1024, // 64MB
            max_write_buffer_number: 3,
            compression: true,
        }
    }
}

/// Column family names
pub const CF_DEFAULT: &str = "default";
pub const CF_MEMORIES: &str = "memories";
pub const CF_ENTITIES: &str = "entities";
pub const CF_RELATIONSHIPS: &str = "relationships";
pub const CF_EPISODES: &str = "episodes";
pub const CF_EVENTS: &str = "events";
pub const CF_INDEX: &str = "index";

/// RocksDB persistent storage for memory engine
pub struct RocksStorage {
    db: Arc<rocksdb::DB>,
    config: RocksConfig,
}

impl RocksStorage {
    /// Create or open a RocksDB database
    pub fn new(config: RocksConfig) -> Result<Self> {
        let path = Path::new(&config.path);

        let mut db_opts = rocksdb::Options::default();
        db_opts.create_if_missing(config.create_if_missing);
        db_opts.create_missing_column_families(true);
        db_opts.set_max_open_files(config.max_open_files);
        db_opts.set_write_buffer_size(config.write_buffer_size);
        db_opts.set_max_write_buffer_number(config.max_write_buffer_number);

        if config.compression {
            db_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        }

        // Create column families
        let cf_opts = rocksdb::Options::default();
        let cfs = vec![
            (CF_DEFAULT, cf_opts.clone()),
            (CF_MEMORIES, cf_opts.clone()),
            (CF_ENTITIES, cf_opts.clone()),
            (CF_RELATIONSHIPS, cf_opts.clone()),
            (CF_EPISODES, cf_opts.clone()),
            (CF_EVENTS, cf_opts.clone()),
            (CF_INDEX, cf_opts),
        ];

        let db = rocksdb::DB::open_cf_descriptors(&db_opts, path, cfs)
            .map_err(|e| MemoryError::Storage(format!("Failed to open RocksDB: {}", e)))?;

        Ok(Self {
            db: Arc::new(db),
            config,
        })
    }

    /// Store a memory artifact
    pub fn store_memory(&self, artifact: &MemoryArtifact) -> Result<()> {
        let cf = self
            .db
            .cf_handle(CF_MEMORIES)
            .ok_or_else(|| MemoryError::Storage("Column family not found".into()))?;

        let key = artifact.id.as_bytes();
        let value = bincode::serialize(artifact)
            .map_err(|e| MemoryError::Serialization(format!("Failed to serialize: {}", e)))?;

        self.db
            .put_cf(&cf, key, value)
            .map_err(|e| MemoryError::Storage(format!("Failed to store: {}", e)))?;

        // Also store in index by type
        self.store_index(&artifact.id, &artifact.memory_type.to_string())?;

        Ok(())
    }

    /// Retrieve a memory artifact by ID
    pub fn get_memory(&self, id: &MemoryId) -> Result<Option<MemoryArtifact>> {
        let cf = self
            .db
            .cf_handle(CF_MEMORIES)
            .ok_or_else(|| MemoryError::Storage("Column family not found".into()))?;

        let key = id.as_bytes();

        match self.db.get_cf(&cf, key) {
            Ok(Some(value)) => {
                let artifact: MemoryArtifact = bincode::deserialize(&value).map_err(|e| {
                    MemoryError::Serialization(format!("Failed to deserialize: {}", e))
                })?;
                Ok(Some(artifact))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(MemoryError::Storage(format!("Failed to get: {}", e))),
        }
    }

    /// Delete a memory artifact
    pub fn delete_memory(&self, id: &MemoryId) -> Result<()> {
        let cf = self
            .db
            .cf_handle(CF_MEMORIES)
            .ok_or_else(|| MemoryError::Storage("Column family not found".into()))?;

        let key = id.as_bytes();
        self.db
            .delete_cf(&cf, key)
            .map_err(|e| MemoryError::Storage(format!("Failed to delete: {}", e)))
    }

    /// List all memory IDs
    pub fn list_memory_ids(&self) -> Result<Vec<MemoryId>> {
        let cf = self
            .db
            .cf_handle(CF_MEMORIES)
            .ok_or_else(|| MemoryError::Storage("Column family not found".into()))?;

        let mut ids = Vec::new();
        let iter = self.db.full_iterator_cf(&cf, rocksdb::IteratorMode::Start);

        for item in iter {
            let (key, _) =
                item.map_err(|e| MemoryError::Storage(format!("Iterator error: {}", e)))?;
            if key.len() == 16 {
                let mut bytes = [0u8; 16];
                bytes.copy_from_slice(&key);
                ids.push(uuid::Uuid::from_bytes(bytes));
            }
        }

        Ok(ids)
    }

    /// Store entity
    pub fn store_entity(&self, id: &uuid::Uuid, entity: &impl Serialize) -> Result<()> {
        let cf = self
            .db
            .cf_handle(CF_ENTITIES)
            .ok_or_else(|| MemoryError::Storage("Column family not found".into()))?;

        let key = id.as_bytes();
        let value = bincode::serialize(entity)
            .map_err(|e| MemoryError::Serialization(format!("Failed to serialize: {}", e)))?;

        self.db
            .put_cf(&cf, key, value)
            .map_err(|e| MemoryError::Storage(format!("Failed to store entity: {}", e)))
    }

    /// Get entity
    pub fn get_entity<T: for<'de> Deserialize<'de>>(&self, id: &uuid::Uuid) -> Result<Option<T>> {
        let cf = self
            .db
            .cf_handle(CF_ENTITIES)
            .ok_or_else(|| MemoryError::Storage("Column family not found".into()))?;

        let key = id.as_bytes();

        match self.db.get_cf(&cf, key) {
            Ok(Some(value)) => {
                let entity: T = bincode::deserialize(&value).map_err(|e| {
                    MemoryError::Serialization(format!("Failed to deserialize: {}", e))
                })?;
                Ok(Some(entity))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(MemoryError::Storage(format!("Failed to get entity: {}", e))),
        }
    }

    /// Store relationship
    pub fn store_relationship(&self, id: &uuid::Uuid, relationship: &impl Serialize) -> Result<()> {
        let cf = self
            .db
            .cf_handle(CF_RELATIONSHIPS)
            .ok_or_else(|| MemoryError::Storage("Column family not found".into()))?;

        let key = id.as_bytes();
        let value = bincode::serialize(relationship)
            .map_err(|e| MemoryError::Serialization(format!("Failed to serialize: {}", e)))?;

        self.db
            .put_cf(&cf, key, value)
            .map_err(|e| MemoryError::Storage(format!("Failed to store relationship: {}", e)))
    }

    /// Store episode
    pub fn store_episode(&self, id: &uuid::Uuid, episode: &impl Serialize) -> Result<()> {
        let cf = self
            .db
            .cf_handle(CF_EPISODES)
            .ok_or_else(|| MemoryError::Storage("Column family not found".into()))?;

        let key = id.as_bytes();
        let value = bincode::serialize(episode)
            .map_err(|e| MemoryError::Serialization(format!("Failed to serialize: {}", e)))?;

        self.db
            .put_cf(&cf, key, value)
            .map_err(|e| MemoryError::Storage(format!("Failed to store episode: {}", e)))
    }

    /// Store event
    pub fn store_event(&self, id: &uuid::Uuid, event: &impl Serialize) -> Result<()> {
        let cf = self
            .db
            .cf_handle(CF_EVENTS)
            .ok_or_else(|| MemoryError::Storage("Column family not found".into()))?;

        let key = id.as_bytes();
        let value = bincode::serialize(event)
            .map_err(|e| MemoryError::Serialization(format!("Failed to serialize: {}", e)))?;

        self.db
            .put_cf(&cf, key, value)
            .map_err(|e| MemoryError::Storage(format!("Failed to store event: {}", e)))
    }

    /// Flush write buffer
    pub fn flush(&self) -> Result<()> {
        self.db
            .flush()
            .map_err(|e| MemoryError::Storage(format!("Failed to flush: {}", e)))
    }

    /// Get database statistics
    pub fn get_stats(&self) -> Result<StorageStats> {
        let cf = self
            .db
            .cf_handle(CF_MEMORIES)
            .ok_or_else(|| MemoryError::Storage("Column family not found".into()))?;

        let mut stats = StorageStats::default();

        // Count memories
        let iter = self.db.full_iterator_cf(&cf, rocksdb::IteratorMode::Start);
        stats.memory_count = iter.count();

        Ok(stats)
    }

    /// Compact the database
    pub fn compact(&self) -> Result<()> {
        let cf = self
            .db
            .cf_handle(CF_MEMORIES)
            .ok_or_else(|| MemoryError::Storage("Column family not found".into()))?;

        self.db.compact_range_cf(&cf, None::<&[u8]>, None::<&[u8]>);
        Ok(())
    }

    /// Create checkpoint
    pub fn create_checkpoint(&self, path: &str) -> Result<()> {
        let checkpoint = rocksdb::checkpoint::Checkpoint::new(&self.db)
            .map_err(|e| MemoryError::Storage(format!("Failed to create checkpoint: {}", e)))?;

        checkpoint
            .create_checkpoint(path)
            .map_err(|e| MemoryError::Storage(format!("Failed to create checkpoint: {}", e)))?;

        Ok(())
    }

    // Private helper methods

    fn store_index(&self, id: &MemoryId, index_key: &str) -> Result<()> {
        let cf = self
            .db
            .cf_handle(CF_INDEX)
            .ok_or_else(|| MemoryError::Storage("Column family not found".into()))?;

        let key = format!("{}:{}", index_key, id);
        let value = id.as_bytes();

        self.db
            .put_cf(&cf, key.as_bytes(), value)
            .map_err(|e| MemoryError::Storage(format!("Failed to store index: {}", e)))
    }
}

/// Storage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageStats {
    pub memory_count: usize,
    pub entity_count: usize,
    pub relationship_count: usize,
    pub episode_count: usize,
    pub event_count: usize,
    pub estimated_size_bytes: u64,
}
