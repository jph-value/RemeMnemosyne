//! # RemeMnemosyne Storage
//!
//! Pluggable storage backends with trait-based design.
//!
//! ## Available Backends
//!
//! - **sled** (default, pure Rust) - General purpose embedded database
//! - **rocksdb** (optional, requires C++) - High-performance for write-heavy workloads
//!
//! ## Usage
//!
//! ```rust
//! use rememnemosyne_storage::{SledStorage, StorageBackend};
//! use rememnemosyne_storage::backend::helpers;
//!
//! # fn main() -> rememnemosyne_core::Result<()> {
//! let storage = SledStorage::new("/tmp/my_memory_db")?;
//! storage.put(b"key", b"value")?;
//! let value = storage.get(b"key")?;
//! assert_eq!(value, Some(b"value".to_vec()));
//! # Ok(())
//! # }
//! ```

pub mod backend;
pub mod sled_backend;

#[cfg(feature = "persistence")]
pub mod rocks_backend;

pub mod snapshot;

pub use backend::*;
pub use sled_backend::SledStorage;

#[cfg(feature = "persistence")]
pub use rocks_backend::RocksStorage;

use rememnemosyne_core::Result;

/// Create a default storage backend
pub fn create_default_storage(path: &str) -> Result<Box<dyn StorageBackend>> {
    #[cfg(feature = "sled-storage")]
    {
        return Ok(Box::new(SledStorage::new(path)?));
    }

    #[cfg(not(feature = "sled-storage"))]
    {
        Err(MemoryError::Storage("No storage backend enabled".into()))
    }
}

/// Storage configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StorageConfig {
    pub path: String,
    pub flush_every_ms: Option<u64>,
    pub cache_capacity: Option<usize>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            path: "./rememnemosyne_data".to_string(),
            flush_every_ms: Some(1000),
            cache_capacity: Some(1024 * 1024 * 1024), // 1GB
        }
    }
}
