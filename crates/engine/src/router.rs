use rememnemosyne_core::*;
use rememnemosyne_semantic::SemanticMemoryStore;
use rememnemosyne_episodic::EpisodicMemoryStore;
use rememnemosyne_graph::{GraphMemoryStore, entity::GraphEntity};
use rememnemosyne_temporal::{TemporalMemoryStore, TemporalEvent};
use rememnemosyne_cognitive::{ContextPredictor, MemoryPrefetcher, MicroEmbedder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for the memory router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRouterConfig {
    pub max_results_per_store: usize,
    pub enable_prefetch: bool,
    pub prefetch_threshold: f32,
    pub combine_scores: bool,
    pub embedding_dimensions: usize,
}

impl Default for MemoryRouterConfig {
    fn default() -> Self {
        Self {
            max_results_per_store: 10,
            enable_prefetch: true,
            prefetch_threshold: 0.5,
            combine_scores: true,
            embedding_dimensions: 1536, // Match semantic store default
        }
    }
}

/// Memory router - coordinates queries across all memory stores
pub struct MemoryRouter {
    config: MemoryRouterConfig,
    pub semantic: Arc<SemanticMemoryStore>,
    pub episodic: Arc<EpisodicMemoryStore>,
    pub graph: Arc<GraphMemoryStore>,
    pub temporal: Arc<TemporalMemoryStore>,
    predictor: Arc<parking_lot::RwLock<ContextPredictor>>,
    prefetcher: Arc<parking_lot::RwLock<MemoryPrefetcher>>,
    embedder: Arc<parking_lot::RwLock<MicroEmbedder>>,
}

impl MemoryRouter {
    pub fn new(
        config: MemoryRouterConfig,
        semantic: Arc<SemanticMemoryStore>,
        episodic: Arc<EpisodicMemoryStore>,
        graph: Arc<GraphMemoryStore>,
        temporal: Arc<TemporalMemoryStore>,
    ) -> Self {
        // Create micro-embedder with matching dimensions
        let embedder_config = rememnemosyne_cognitive::MicroEmbedConfig {
            dimensions: config.embedding_dimensions,
            model_type: rememnemosyne_cognitive::MicroEmbedModel::Hash,
            normalize: true,
            cache_size: 10000,
        };
        
        Self {
            config,
            semantic,
            episodic,
            graph,
            temporal,
            predictor: Arc::new(parking_lot::RwLock::new(ContextPredictor::new(Default::default()))),
            prefetcher: Arc::new(parking_lot::RwLock::new(MemoryPrefetcher::new(Default::default()))),
            embedder: Arc::new(parking_lot::RwLock::new(MicroEmbedder::new(embedder_config))),
        }
    }

    /// Query all memory stores with unified result
    pub async fn query(&self, query: &MemoryQuery) -> Result<MemoryResponse> {
        let mut response = MemoryResponse::new();

        // Generate micro-embedding for query if text provided
        let query_embedding = if let Some(ref text) = query.text {
            let embedding = {
                let embedder = self.embedder.read();
                embedder.embed(text)
            };

            // Update predictor context
            let mut predictor = self.predictor.write();
            predictor.add_context(text, vec![]);

            Some(embedding)
        } else {
            query.embedding.clone()
        };

        // Build an enriched query with the embedding injected
        // so the semantic store can use HNSW vector search
        let enriched_query = if query_embedding.is_some() && query.embedding.is_none() {
            let mut q = query.clone();
            q.embedding = query_embedding.clone();
            q
        } else {
            query.clone()
        };

        // Query semantic memory (uses HNSW via enriched embedding)
        if let Ok(semantic_results) = self.semantic.query(&enriched_query).await {
            for memory in semantic_results.into_iter().take(self.config.max_results_per_store) {
                let relevance = memory.compute_relevance();
                response.add_result(memory, MemoryType::Semantic, relevance);
            }
        }

        // Query episodic memory
        if let Ok(episodic_results) = self.episodic.query(&enriched_query).await {
            for memory in episodic_results.into_iter().take(self.config.max_results_per_store) {
                let relevance = memory.compute_relevance();
                response.add_result(memory, MemoryType::Episodic, relevance);
            }
        }

        // Query graph memory for relevant entities
        if let Some(ref text) = enriched_query.text {
            let entities = self.graph.search_entities(text, self.config.max_results_per_store).await;
            for entity in entities {
                response.entities.push(entity);
            }
        }

        // Query temporal memory for relevant timeline events
        if let Some(ref text) = enriched_query.text {
            let events = self.temporal.search_events(text, self.config.max_results_per_store).await;
            for event in events {
                response.temporal_events.push(event);
            }
        }

        // Combine and rank results
        if self.config.combine_scores {
            response.sort_by_relevance();
        }

        // Predict next likely memories using prefetch data
        if self.config.enable_prefetch {
            let query_text = query.text.as_deref().unwrap_or("");
            let predictor = self.predictor.read();
            let ids: Vec<_> = response.results.iter().map(|r| r.memory.id).collect();
            response.predicted_next = predictor.predict(query_text, &ids);
        }

        Ok(response)
    }

    /// Store a memory artifact in all relevant stores
    pub async fn store(&self, artifact: MemoryArtifact) -> Result<MemoryId> {
        let id = artifact.id;

        // Store in semantic memory
        let _ = self.semantic.store(artifact.clone()).await?;

        // Store in episodic memory
        let _ = self.episodic.store(artifact.clone()).await?;

        // Update prefetcher with new memory
        if !artifact.embedding.is_empty() {
            let mut prefetcher = self.prefetcher.write();
            prefetcher.register_memory(
                id,
                artifact.embedding.clone(),
                &artifact.tags,
            );
        }

        Ok(id)
    }

    /// Get memory by ID from any store
    pub async fn get(&self, id: &MemoryId) -> Result<Option<MemoryArtifact>> {
        // Try semantic first
        if let Ok(Some(memory)) = self.semantic.get(id).await {
            return Ok(Some(memory));
        }

        // Try episodic
        if let Ok(Some(memory)) = self.episodic.get(id).await {
            return Ok(Some(memory));
        }

        Ok(None)
    }

    /// Search entities by name
    pub async fn search_entities(&self, query: &str, limit: usize) -> Vec<rememnemosyne_graph::entity::GraphEntity> {
        self.graph.search_entities(query, limit).await
    }

    /// Get entity relationships
    pub async fn get_entity_relationships(
        &self,
        entity_id: &EntityId,
        max_depth: usize,
    ) -> Result<Vec<(rememnemosyne_graph::entity::GraphEntity, RelationshipType, f32)>> {
        self.graph.find_related(entity_id, max_depth).await
    }

    /// Get timeline for entity
    pub async fn get_entity_timeline(
        &self,
        entity_id: &EntityId,
    ) -> Option<rememnemosyne_temporal::timeline::Timeline> {
        self.temporal.get_entity_timeline(entity_id).await
    }

    /// Get router statistics
    pub async fn get_stats(&self) -> RouterStats {
        let semantic_count = self.semantic.count().await.unwrap_or(0);
        let episodic_count = self.episodic.count().await.unwrap_or(0);
        let graph_stats = self.graph.get_statistics().await;

        RouterStats {
            semantic_memories: semantic_count,
            episodic_memories: episodic_count,
            graph_entities: graph_stats.entity_count,
            graph_relationships: graph_stats.relationship_count,
        }
    }

    /// Generate embedding for text using the micro-embedder
    pub async fn generate_embedding(&self, text: &str) -> Vec<f32> {
        let embedder = self.embedder.read();
        let embedding = embedder.embed(text);
        
        // Ensure embedding has correct dimensions
        if embedding.len() != self.config.embedding_dimensions {
            // Pad or truncate to correct dimensions
            let mut corrected = vec![0.0; self.config.embedding_dimensions];
            let copy_len = std::cmp::min(embedding.len(), self.config.embedding_dimensions);
            corrected[..copy_len].copy_from_slice(&embedding[..copy_len]);
            corrected
        } else {
            embedding
        }
    }
}

/// Unified memory response from all stores
#[derive(Debug, Clone)]
pub struct MemoryResponse {
    pub results: Vec<MemoryResult>,
    pub entities: Vec<GraphEntity>,
    pub temporal_events: Vec<TemporalEvent>,
    pub predicted_next: Vec<(uuid::Uuid, f32)>,
    pub total_search_time_ms: u64,
}

impl MemoryResponse {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            entities: Vec::new(),
            temporal_events: Vec::new(),
            predicted_next: Vec::new(),
            total_search_time_ms: 0,
        }
    }

    pub fn add_result(&mut self, memory: MemoryArtifact, source: MemoryType, relevance: f32) {
        self.results.push(MemoryResult {
            memory,
            source,
            relevance,
        });
    }

    pub fn sort_by_relevance(&mut self) {
        self.results.sort_by(|a, b| {
            b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    pub fn limit(&mut self, max: usize) {
        self.results.truncate(max);
    }

    pub fn get_memories(&self) -> Vec<&MemoryArtifact> {
        self.results.iter().map(|r| &r.memory).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

impl Default for MemoryResponse {
    fn default() -> Self {
        Self::new()
    }
}

/// Individual memory result with source and relevance
#[derive(Debug, Clone)]
pub struct MemoryResult {
    pub memory: MemoryArtifact,
    pub source: MemoryType,
    pub relevance: f32,
}

/// Router statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterStats {
    pub semantic_memories: usize,
    pub episodic_memories: usize,
    pub graph_entities: usize,
    pub graph_relationships: usize,
}
