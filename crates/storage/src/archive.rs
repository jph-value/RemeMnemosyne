/// Zstd-compressed archive for pruned memories.
///
/// Stores memories as individually compressed chunks with an uncompressed
/// catalog for fast metadata lookups and selective decompression.
///
/// # Archive Format
///
/// ```text
/// archive/
/// ├── catalog.json       # Uncompressed: ID -> {offset, size, summary, tags, importance, timestamp}
/// ├── data.bin           # Zstd-compressed chunks, each prefixed with 4-byte length
/// └── lock               # Lock file for concurrent access
/// ```
///
/// The catalog stores metadata without decompression so you can search/filter
/// archived memories without touching the compressed data.
use chrono::{DateTime, Utc};
#[cfg(test)]
use rememnemosyne_core::MemoryTrigger;
use rememnemosyne_core::{Importance, MemoryArtifact, MemoryId, MemoryType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// Lightweight metadata for archived memories.
/// Stored uncompressed in the catalog for fast filtering without decompression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveEntry {
    pub id: MemoryId,
    pub summary: String,
    pub tags: Vec<String>,
    pub importance: Importance,
    pub memory_type: MemoryType,
    pub timestamp: DateTime<Utc>,
    pub access_count: u64,
    pub content_length: usize,
    pub offset: u64,
    pub compressed_size: u64,
    pub original_size: u64,
}

/// Catalog that maps memory IDs to their archive entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveCatalog {
    pub entries: HashMap<MemoryId, ArchiveEntry>,
    pub total_entries: usize,
    pub total_original_bytes: u64,
    pub total_compressed_bytes: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl ArchiveCatalog {
    fn new() -> Self {
        let now = Utc::now();
        Self {
            entries: HashMap::new(),
            total_entries: 0,
            total_original_bytes: 0,
            total_compressed_bytes: 0,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn compression_ratio(&self) -> f64 {
        if self.total_original_bytes == 0 {
            return 1.0;
        }
        self.total_compressed_bytes as f64 / self.total_original_bytes as f64
    }
}

/// Configuration for the archive system
#[derive(Debug, Clone)]
pub struct ArchiveConfig {
    /// Directory to store archive files
    pub archive_dir: PathBuf,
    /// Zstd compression level (1-22, higher = better compression, slower)
    pub compression_level: i32,
    /// Maximum number of entries per archive file before splitting
    pub max_entries_per_file: usize,
}

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self {
            archive_dir: PathBuf::from("./rememnemosyne_data/archive"),
            compression_level: 3,
            max_entries_per_file: 100_000,
        }
    }
}

/// Zstd-compressed archive with selective decompression support.
pub struct MemoryArchive {
    config: ArchiveConfig,
    catalog_path: PathBuf,
    data_path: PathBuf,
    catalog: ArchiveCatalog,
}

impl MemoryArchive {
    /// Open or create an archive at the given path
    pub fn open(config: ArchiveConfig) -> io::Result<Self> {
        fs::create_dir_all(&config.archive_dir)?;

        let catalog_path = config.archive_dir.join("catalog.json");
        let data_path = config.archive_dir.join("data.bin");

        let catalog = if catalog_path.exists() {
            let data = fs::read_to_string(&catalog_path)?;
            serde_json::from_str(&data).unwrap_or_else(|_| ArchiveCatalog::new())
        } else {
            ArchiveCatalog::new()
        };

        Ok(Self {
            config,
            catalog_path,
            data_path,
            catalog,
        })
    }

    /// Archive a single memory with zstd compression
    pub fn archive_memory(&mut self, memory: &MemoryArtifact) -> io::Result<()> {
        let id = memory.id;

        // Serialize the full memory artifact
        let serialized = bincode::serialize(memory)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let original_size = serialized.len() as u64;

        // Compress with zstd
        let compressed = zstd::encode_all(serialized.as_slice(), self.config.compression_level)?;
        let compressed_size = compressed.len() as u64;

        // Append compressed chunk to data file
        let offset = if self.data_path.exists() {
            fs::metadata(&self.data_path)?.len()
        } else {
            0
        };

        let mut data_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.data_path)?;

        // Write length prefix + compressed data
        let len_bytes = (compressed_size as u32).to_le_bytes();
        data_file.write_all(&len_bytes)?;
        data_file.write_all(&compressed)?;
        data_file.flush()?;

        // Add to catalog
        let entry = ArchiveEntry {
            id,
            summary: memory.summary.clone(),
            tags: memory.tags.clone(),
            importance: memory.importance,
            memory_type: memory.memory_type,
            timestamp: memory.timestamp,
            access_count: memory.access_count,
            content_length: memory.content.len(),
            offset: offset + 4, // Skip the 4-byte length prefix
            compressed_size,
            original_size,
        };

        self.catalog.entries.insert(id, entry);
        self.catalog.total_entries = self.catalog.entries.len();
        self.catalog.total_original_bytes += original_size;
        self.catalog.total_compressed_bytes += compressed_size;
        self.catalog.updated_at = Utc::now();

        // Flush catalog
        self.flush_catalog()?;

        Ok(())
    }

    /// Archive multiple memories in batch
    pub fn archive_batch(&mut self, memories: &[MemoryArtifact]) -> io::Result<ArchiveStats> {
        let start = std::time::Instant::now();
        let mut original_bytes = 0u64;
        let mut compressed_bytes = 0u64;

        // Open data file once, outside the loop
        let mut data_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.data_path)?;

        for memory in memories {
            let serialized = bincode::serialize(memory)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            original_bytes += serialized.len() as u64;

            let compressed =
                zstd::encode_all(serialized.as_slice(), self.config.compression_level)?;
            compressed_bytes += compressed.len() as u64;

            // Current file offset = current length before we write
            let offset = data_file.stream_position()?;

            let len_bytes = (compressed.len() as u32).to_le_bytes();
            data_file.write_all(&len_bytes)?;
            data_file.write_all(&compressed)?;

            let entry = ArchiveEntry {
                id: memory.id,
                summary: memory.summary.clone(),
                tags: memory.tags.clone(),
                importance: memory.importance,
                memory_type: memory.memory_type,
                timestamp: memory.timestamp,
                access_count: memory.access_count,
                content_length: memory.content.len(),
                offset: offset + 4,
                compressed_size: compressed.len() as u64,
                original_size: serialized.len() as u64,
            };
            self.catalog.entries.insert(memory.id, entry);
        }

        data_file.flush()?;
        drop(data_file);

        self.catalog.total_entries = self.catalog.entries.len();
        self.catalog.total_original_bytes += original_bytes;
        self.catalog.total_compressed_bytes += compressed_bytes;
        self.catalog.updated_at = Utc::now();
        self.flush_catalog()?;

        Ok(ArchiveStats {
            memories_archived: memories.len(),
            original_bytes,
            compressed_bytes,
            compression_ratio: if original_bytes > 0 {
                compressed_bytes as f64 / original_bytes as f64
            } else {
                1.0
            },
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Selectively decompress a single memory by ID.
    /// Does NOT decompress the entire archive.
    pub fn decompress_memory(&self, id: &MemoryId) -> io::Result<Option<MemoryArtifact>> {
        let entry = match self.catalog.entries.get(id) {
            Some(e) => e,
            None => return Ok(None),
        };

        let mut data_file = fs::File::open(&self.data_path)?;

        // Seek to the compressed chunk
        data_file.seek(SeekFrom::Start(entry.offset))?;

        // Read compressed data
        let mut compressed = vec![0u8; entry.compressed_size as usize];
        data_file.read_exact(&mut compressed)?;

        // Decompress with zstd
        let decompressed = zstd::decode_all(compressed.as_slice())?;

        // Deserialize
        let memory: MemoryArtifact = bincode::deserialize(&decompressed)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        Ok(Some(memory))
    }

    /// Decompress multiple memories by ID (selective)
    pub fn decompress_batch(&self, ids: &[MemoryId]) -> io::Result<Vec<MemoryArtifact>> {
        let mut results = Vec::with_capacity(ids.len());

        // Sort IDs by offset for sequential reads (faster than random seeks)
        let mut sorted_ids: Vec<&MemoryId> = ids.iter().collect();
        sorted_ids.sort_by_key(|id| {
            self.catalog
                .entries
                .get(id)
                .map(|e| e.offset)
                .unwrap_or(u64::MAX)
        });

        for id in sorted_ids {
            if let Some(memory) = self.decompress_memory(id)? {
                results.push(memory);
            }
        }

        Ok(results)
    }

    /// Search archived memories by metadata WITHOUT decompressing data.
    /// This is the key performance feature: filter by tags/importance/summary
    /// without touching the compressed data file.
    pub fn search_by_metadata(
        &self,
        query: &str,
        tags: Option<&[String]>,
        min_importance: Option<Importance>,
    ) -> Vec<&ArchiveEntry> {
        let query_lower = query.to_lowercase();

        self.catalog
            .entries
            .values()
            .filter(|entry| {
                // Text match on summary
                if !query.is_empty() && !entry.summary.to_lowercase().contains(&query_lower) {
                    return false;
                }

                // Tag filter
                if let Some(filter_tags) = tags {
                    if !filter_tags.iter().any(|t| entry.tags.contains(t)) {
                        return false;
                    }
                }

                // Importance filter
                if let Some(min_imp) = min_importance {
                    if entry.importance < min_imp {
                        return false;
                    }
                }

                true
            })
            .collect()
    }

    /// List all archived memory IDs
    pub fn list_ids(&self) -> Vec<MemoryId> {
        self.catalog.entries.keys().copied().collect()
    }

    /// Get archive statistics
    pub fn stats(&self) -> &ArchiveCatalog {
        &self.catalog
    }

    /// Delete an archived memory (removes from catalog, does NOT reclaim space in data file)
    pub fn delete_memory(&mut self, id: &MemoryId) -> io::Result<bool> {
        if let Some(entry) = self.catalog.entries.remove(id) {
            self.catalog.total_entries = self.catalog.entries.len();
            self.catalog.total_original_bytes = self
                .catalog
                .total_original_bytes
                .saturating_sub(entry.original_size);
            self.catalog.total_compressed_bytes = self
                .catalog
                .total_compressed_bytes
                .saturating_sub(entry.compressed_size);
            self.catalog.updated_at = Utc::now();
            self.flush_catalog()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Compact the archive: rewrite data file to reclaim space from deleted entries
    pub fn compact(&mut self) -> io::Result<CompactStats> {
        let start = std::time::Instant::now();

        // Collect all active entries sorted by current offset
        let mut active_entries: Vec<(MemoryId, u64, u64)> = self
            .catalog
            .entries
            .iter()
            .map(|(id, e)| (*id, e.offset, e.compressed_size))
            .collect();
        active_entries.sort_by_key(|(_, offset, _)| *offset);

        // Collect metadata needed for writing (no borrow of catalog)
        let _entry_data: Vec<(MemoryId, u64)> = active_entries
            .iter()
            .map(|(id, _offset, compressed_size)| (*id, *compressed_size))
            .collect();

        // Create temp data file
        let temp_path = self.data_path.with_extension("tmp");
        let mut temp_file = BufWriter::new(fs::File::create(&temp_path)?);
        let mut new_offsets: Vec<(MemoryId, u64)> = Vec::new();
        let mut new_offset = 0u64;

        for (id, old_offset, compressed_size) in active_entries.iter() {
            // Read from old file
            let mut data_file = BufReader::new(fs::File::open(&self.data_path)?);
            data_file.seek(SeekFrom::Start(*old_offset))?;

            let mut compressed = vec![0u8; *compressed_size as usize];
            data_file.read_exact(&mut compressed)?;

            // Write to new file with length prefix
            let len_bytes = (*compressed_size as u32).to_le_bytes();
            temp_file.write_all(&len_bytes)?;
            temp_file.write_all(&compressed)?;

            new_offsets.push((*id, new_offset + 4));
            new_offset += 4 + compressed_size;
        }

        temp_file.flush()?;
        drop(temp_file);

        // Replace old data file with compacted version
        fs::rename(&temp_path, &self.data_path)?;

        // Update catalog entries with new offsets
        for (id, offset) in new_offsets {
            if let Some(entry) = self.catalog.entries.get_mut(&id) {
                entry.offset = offset;
            }
        }

        self.catalog.updated_at = Utc::now();
        self.flush_catalog()?;

        Ok(CompactStats {
            entries: active_entries.len(),
            bytes_before: self.catalog.total_compressed_bytes,
            bytes_after: new_offset,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn flush_catalog(&self) -> io::Result<()> {
        let json = serde_json::to_string_pretty(&self.catalog)?;
        fs::write(&self.catalog_path, json)?;
        Ok(())
    }
}

/// Statistics from an archive operation
#[derive(Debug, Clone)]
pub struct ArchiveStats {
    pub memories_archived: usize,
    pub original_bytes: u64,
    pub compressed_bytes: u64,
    pub compression_ratio: f64,
    pub elapsed_ms: f64,
}

/// Statistics from a compact operation
#[derive(Debug, Clone)]
pub struct CompactStats {
    pub entries: usize,
    pub bytes_before: u64,
    pub bytes_after: u64,
    pub elapsed_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_test_dir(name: &str) -> std::path::PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("remeMnemosyne_test_{}_{}", name, ts));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_test_memory(id: MemoryId, content: &str, summary: &str) -> MemoryArtifact {
        let mut m = MemoryArtifact::new(
            MemoryType::Semantic,
            summary,
            content,
            vec![0.1; 128],
            MemoryTrigger::Insight,
        );
        // Override ID for deterministic tests
        m.id = id;
        m.tags = vec!["test".to_string(), "archive".to_string()];
        m
    }

    #[test]
    fn test_archive_and_decompress_single() {
        let dir = temp_test_dir("test");
        let config = ArchiveConfig {
            archive_dir: dir.as_path().to_path_buf(),
            compression_level: 3,
            ..Default::default()
        };

        let mut archive = MemoryArchive::open(config).unwrap();

        let id = MemoryId::new_v4();
        let memory = make_test_memory(id, "Hello world content", "Hello world");

        archive.archive_memory(&memory).unwrap();

        // Catalog should have the entry
        assert_eq!(archive.stats().total_entries, 1);

        // Decompress should return the same memory
        let restored = archive.decompress_memory(&id).unwrap().unwrap();
        assert_eq!(restored.id, id);
        assert_eq!(restored.summary, "Hello world");
        assert_eq!(restored.content, "Hello world content");
        assert_eq!(restored.tags, vec!["test", "archive"]);
    }

    #[test]
    fn test_selective_decompression() {
        let dir = temp_test_dir("test");
        let config = ArchiveConfig {
            archive_dir: dir.as_path().to_path_buf(),
            compression_level: 3,
            ..Default::default()
        };

        let mut archive = MemoryArchive::open(config).unwrap();

        let ids: Vec<MemoryId> = (0..5).map(|_| MemoryId::new_v4()).collect();
        for (i, id) in ids.iter().enumerate() {
            let mut memory = make_test_memory(
                *id,
                &format!("Content for memory {}", i),
                &format!("Summary {}", i),
            );
            memory.id = *id;
            archive.archive_memory(&memory).unwrap();
        }

        // Decompress only memories 1 and 3
        let selected = archive.decompress_batch(&[ids[1], ids[3]]).unwrap();
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().any(|m| m.summary == "Summary 1"));
        assert!(selected.iter().any(|m| m.summary == "Summary 3"));

        // Decompress non-existent ID
        let missing = archive.decompress_memory(&MemoryId::new_v4()).unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn test_metadata_search_no_decompression() {
        let dir = temp_test_dir("test");
        let config = ArchiveConfig {
            archive_dir: dir.as_path().to_path_buf(),
            compression_level: 3,
            ..Default::default()
        };

        let mut archive = MemoryArchive::open(config).unwrap();

        let id1 = MemoryId::new_v4();
        let mut m1 = make_test_memory(id1, "Rust content", "Rust programming");
        m1.id = id1;
        m1.tags = vec!["rust".to_string()];
        m1.importance = Importance::High;

        let id2 = MemoryId::new_v4();
        let mut m2 = make_test_memory(id2, "Python content", "Python data science");
        m2.id = id2;
        m2.tags = vec!["python".to_string()];
        m2.importance = Importance::Low;

        archive.archive_memory(&m1).unwrap();
        archive.archive_memory(&m2).unwrap();

        // Search by text
        let results = archive.search_by_metadata("rust", None, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].summary, "Rust programming");

        // Search by importance
        let results = archive.search_by_metadata("", None, Some(Importance::High));
        assert_eq!(results.len(), 1);

        // Search by tag
        let results = archive.search_by_metadata("", Some(&["python".to_string()]), None);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_batch_archive_compression_ratio() {
        let dir = temp_test_dir("test");
        let config = ArchiveConfig {
            archive_dir: dir.as_path().to_path_buf(),
            compression_level: 19,
            ..Default::default()
        };

        let mut archive = MemoryArchive::open(config).unwrap();

        // Create memories with repetitive content (highly compressible)
        let memories: Vec<MemoryArtifact> = (0..100)
            .map(|i| {
                let id = MemoryId::new_v4();
                let mut m = make_test_memory(
                    id,
                    &format!("This is a test memory with repeated content. Entry number {}. Lorem ipsum dolor sit amet.", i),
                    &format!("Test {}", i),
                );
                m.id = id;
                m
            })
            .collect();

        let stats = archive.archive_batch(&memories).unwrap();

        assert_eq!(stats.memories_archived, 100);
        assert!(stats.compression_ratio < 0.5); // Should achieve at least 2:1 compression
        println!(
            "Compression: {:.1}% of original (ratio: {:.3})",
            stats.compression_ratio * 100.0,
            stats.compression_ratio
        );
    }

    #[test]
    fn test_delete_and_compact() {
        let dir = temp_test_dir("test");
        let config = ArchiveConfig {
            archive_dir: dir.as_path().to_path_buf(),
            compression_level: 3,
            ..Default::default()
        };

        let mut archive = MemoryArchive::open(config).unwrap();

        let ids: Vec<MemoryId> = (0..5).map(|_| MemoryId::new_v4()).collect();
        for (i, id) in ids.iter().enumerate() {
            let mut m = make_test_memory(*id, &format!("Content {}", i), &format!("Summary {}", i));
            m.id = *id;
            archive.archive_memory(&m).unwrap();
        }

        // Delete some entries
        archive.delete_memory(&ids[1]).unwrap();
        archive.delete_memory(&ids[3]).unwrap();

        assert_eq!(archive.stats().total_entries, 3);

        // Compact to reclaim space
        let compact_stats = archive.compact().unwrap();
        assert_eq!(compact_stats.entries, 3);

        // Remaining entries should still be accessible
        let m0 = archive.decompress_memory(&ids[0]).unwrap().unwrap();
        assert_eq!(m0.summary, "Summary 0");
        let m4 = archive.decompress_memory(&ids[4]).unwrap().unwrap();
        assert_eq!(m4.summary, "Summary 4");
    }
}
