use async_trait::async_trait;
use dashmap::DashMap;
use rememnemosyne_core::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::turboquant::{TurboQuantizer, TurboQuantConfig, QuantizedCode};
use crate::index::{HNSWIndex, FlatIndex};

/// Configuration for semantic memory store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemoryConfig {
    pub dimensions: usize,
    pub quantization_bits: u8,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub use_quantization: bool,
    pub flat_index_threshold: usize,
}

impl Default for SemanticMemoryConfig {
    fn default() -> Self {
        Self {
            dimensions: 1536,
            quantization_bits: 8,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 100,
            use_quantization: true,
            flat_index_threshold: 1000,
        }
    }
}

/// Semantic memory store with TurboQuant compression and HNSW indexing
pub struct SemanticMemoryStore {
    config: SemanticMemoryConfig,
    /// Memory artifacts indexed by ID
    memories: Arc<DashMap<MemoryId, MemoryArtifact>>,
    /// TurboQuant quantizer
    quantizer: Arc<RwLock<Option<TurboQuantizer>>>,
    /// HNSW index for large datasets
    hnsw_index: Arc<RwLock<HNSWIndex>>,
    /// Flat index for small datasets
    flat_index: Arc<RwLock<FlatIndex>>,
    /// Quantized codes for each memory
    quantized_codes: Arc<DashMap<MemoryId, QuantizedCode>>,
    /// Mapping from memory ID to index ID
    id_to_index: Arc<DashMap<MemoryId, usize>>,
}

impl SemanticMemoryStore {
    pub fn new(config: SemanticMemoryConfig) -> Self {
        let hnsw_index = HNSWIndex::new(
            config.dimensions,
            config.hnsw_m,
            config.hnsw_ef_construction,
        );

        let flat_index = FlatIndex::new(config.dimensions);

        Self {
            config,
            memories: Arc::new(DashMap::new()),
            quantizer: Arc::new(RwLock::new(None)),
            hnsw_index: Arc::new(RwLock::new(hnsw_index)),
            flat_index: Arc::new(RwLock::new(flat_index)),
            quantized_codes: Arc::new(DashMap::new()),
            id_to_index: Arc::new(DashMap::new()),
        }
    }

    /// Initialize the quantizer with training data
    pub async fn train_quantizer(&self, training_data: &[Vec<f32>]) -> Result<()> {
        if training_data.is_empty() {
            return Ok(());
        }

        let mut quantizer = TurboQuantizer::new(
            self.config.dimensions,
            self.config.quantization_bits,
            self.config.dimensions / 16, // Use 16 subquantizers
            42,
        )?;

        quantizer.train(training_data)?;

        let mut q = self.quantizer.write().await;
        *q = Some(quantizer);

        Ok(())
    }

    /// Store with pre-computed embedding
    pub async fn store_with_embedding(
        &self,
        mut artifact: MemoryArtifact,
        embedding: Vec<f32>,
    ) -> Result<MemoryId> {
        artifact.embedding = embedding;
        self.store_internal(artifact).await
    }

    /// Search by vector similarity
    pub async fn search_similar(
        &self,
        query_vector: &[f32],
        k: usize,
        threshold: f32,
    ) -> Result<Vec<(MemoryArtifact, f32)>> {
        if query_vector.len() != self.config.dimensions {
            return Err(MemoryError::Index(
                format!("Query dimension {} != index dimension {}",
                        query_vector.len(), self.config.dimensions)
            ));
        }

        let results = {
            let hnsw = self.hnsw_index.read().await;
            hnsw.search(query_vector, k)
        };

        let mut matched = Vec::new();
        for (idx, similarity) in results {
            // Find the memory ID for this index
            if let Some(entry) = self.id_to_index.iter().find(|e| *e.value() == idx) {
                let memory_id = *entry.key();
                if let Some(mut memory) = self.memories.get(&memory_id).map(|r| r.clone()) {
                    if similarity >= threshold {
                        memory.mark_accessed();
                        matched.push((memory, similarity));
                    }
                }
            }
        }

        Ok(matched)
    }

    /// Search with quantized vectors (faster, approximate)
    pub async fn search_quantized(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<(MemoryArtifact, f32)>> {
        let quantizer = self.quantizer.read().await;
        let quantizer = match quantizer.as_ref() {
            Some(q) => q,
            None => return self.search_similar(query_vector, k, 0.0).await,
        };

        let hnsw = self.hnsw_index.read().await;
        let results = hnsw.search_quantized(query_vector, quantizer, k);
        drop(hnsw);

        let mut matched = Vec::new();
        for (idx, score) in results {
            if let Some(entry) = self.id_to_index.iter().find(|e| *e.value() == idx) {
                let memory_id = *entry.key();
                if let Some(mut memory) = self.memories.get(&memory_id).map(|r| r.clone()) {
                    memory.mark_accessed();
                    matched.push((memory, score));
                }
            }
        }

        Ok(matched)
    }

    /// Get quantizer configuration
    pub fn quantizer_config(&self) -> Option<TurboQuantConfig> {
        // Synchronous check
        None // Would need async or different design
    }

    async fn store_internal(&self, artifact: MemoryArtifact) -> Result<MemoryId> {
        let id = artifact.id;

        // Check if we should use HNSW or flat index
        let use_flat = {
            let hnsw = self.hnsw_index.read().await;
            hnsw.data.len() < self.config.flat_index_threshold
        };

        // Quantize if enabled and quantizer is trained
        let quantized_code = if self.config.use_quantization {
            let quantizer = self.quantizer.read().await;
            if let Some(q) = quantizer.as_ref() {
                q.encode(&artifact.embedding).ok()
            } else {
                None
            }
        } else {
            None
        };

        if let Some(ref code) = quantized_code {
            self.quantized_codes.insert(id, code.clone());
        }

        // Add to appropriate index
        if use_flat {
            let mut flat = self.flat_index.write().await;
            flat.add(id, artifact.embedding.clone())?;
            let idx = flat.len() - 1;
            self.id_to_index.insert(id, idx);
        } else {
            let mut hnsw = self.hnsw_index.write().await;
            let idx = hnsw.add(artifact.embedding.clone(), quantized_code)?;
            self.id_to_index.insert(id, idx);
        }

        // Store the artifact
        self.memories.insert(id, artifact);

        Ok(id)
    }
}

#[async_trait]
impl MemoryStore for SemanticMemoryStore {
    async fn store(&self, artifact: MemoryArtifact) -> Result<MemoryId> {
        self.store_internal(artifact).await
    }

    async fn get(&self, id: &MemoryId) -> Result<Option<MemoryArtifact>> {
        if let Some(mut memory) = self.memories.get(id).map(|r| r.clone()) {
            memory.mark_accessed();
            Ok(Some(memory))
        } else {
            Ok(None)
        }
    }

    async fn query(&self, query: &MemoryQuery) -> Result<Vec<MemoryArtifact>> {
        // If embedding provided, do vector search
        if let Some(ref embedding) = query.embedding {
            let k = query.limit.unwrap_or(10);
            let threshold = query.min_relevance.unwrap_or(0.0);
            let results = self.search_similar(embedding, k, threshold).await?;
            return Ok(results.into_iter().map(|(m, _)| m).collect());
        }

        // Otherwise, filter stored memories
        let limit = query.limit.unwrap_or(usize::MAX);
        let mut results: Vec<MemoryArtifact> = self
            .memories
            .iter()
            .map(|entry| entry.value().clone())
            .filter(|m| {
                // Apply filters
                if let Some(mem_type) = query.memory_type {
                    if m.memory_type != mem_type {
                        return false;
                    }
                }

                if let Some(min_imp) = query.min_importance {
                    if m.importance < min_imp {
                        return false;
                    }
                }

                if let Some(ref time_range) = query.time_range {
                    if m.timestamp < time_range.0 || m.timestamp > time_range.1 {
                        return false;
                    }
                }

                if let Some(session_id) = query.session_id {
                    if m.session_id != Some(session_id) {
                        return false;
                    }
                }

                if let Some(ref tags) = query.tags {
                    if !tags.iter().any(|t| m.tags.contains(t)) {
                        return false;
                    }
                }

                true
            })
            .collect();

        // Sort by timestamp (most recent first)
        results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        results.truncate(limit);

        Ok(results)
    }

    async fn delete(&self, id: &MemoryId) -> Result<bool> {
        if self.memories.remove(id).is_some() {
            if let Some(idx) = self.id_to_index.remove(id) {
                let mut hnsw = self.hnsw_index.write().await;
                hnsw.remove(idx.1)?;
            }
            self.quantized_codes.remove(id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn update(&self, artifact: MemoryArtifact) -> Result<()> {
        let id = artifact.id;
        self.memories.insert(id, artifact);
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.memories.len())
    }

    async fn clear(&self) -> Result<()> {
        self.memories.clear();
        self.quantized_codes.clear();
        self.id_to_index.clear();
        
        let mut hnsw = self.hnsw_index.write().await;
        *hnsw = HNSWIndex::new(
            self.config.dimensions,
            self.config.hnsw_m,
            self.config.hnsw_ef_construction,
        );
        
        let mut flat = self.flat_index.write().await;
        *flat = FlatIndex::new(self.config.dimensions);
        
        Ok(())
    }

    async fn list_ids(&self) -> Result<Vec<MemoryId>> {
        Ok(self.memories.iter().map(|entry| *entry.key()).collect())
    }
}
