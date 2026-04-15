use crate::providers::{EmbeddingProviderRouter, EmbeddingRequest};
use rememnemosyne_cognitive::{ContextPredictor, MemoryPrefetcher, SSCRouter, SSCRouterConfig};
use rememnemosyne_core::*;
use rememnemosyne_episodic::{CheckpointConfig, CheckpointStore, EpisodicMemoryStore};
use rememnemosyne_graph::{entity::GraphEntity, GraphMemoryStore};
use rememnemosyne_semantic::SemanticMemoryStore;
use rememnemosyne_temporal::{TemporalEvent, TemporalMemoryStore};
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
    /// Embedding provider configuration
    pub embedding_config: Option<EmbeddingProviderConfig>,
    /// Memory Caching checkpoint configuration (Phase 1)
    pub checkpoint_config: CheckpointConfig,
    /// Memory Caching SSC router configuration (Phase 3)
    pub ssc_router_config: SSCRouterConfig,
}

impl Default for MemoryRouterConfig {
    fn default() -> Self {
        Self {
            max_results_per_store: 10,
            enable_prefetch: true,
            prefetch_threshold: 0.5,
            combine_scores: true,
            embedding_dimensions: 384,
            embedding_config: None,
            checkpoint_config: CheckpointConfig::default(),
            ssc_router_config: SSCRouterConfig::default(),
        }
    }
}

/// Memory router - coordinates queries across all memory stores
///
/// Uses the EmbeddingProviderRouter for generating embeddings,
/// supporting pluggable embedding providers (OpenAI, Voyage, Cohere,
/// Ollama, Candle/local, or hash fallback).
///
/// Memory Caching (MC) integration:
/// - CheckpointStore: Provides coarse O(N) segment search before fine O(M) HNSW retrieval
/// - SSCRouter: Sparse Selective Caching router for Top-k checkpoint selection
pub struct MemoryRouter {
    config: MemoryRouterConfig,
    pub semantic: Arc<SemanticMemoryStore>,
    pub episodic: Arc<EpisodicMemoryStore>,
    pub graph: Arc<GraphMemoryStore>,
    pub temporal: Arc<TemporalMemoryStore>,
    predictor: Arc<parking_lot::RwLock<ContextPredictor>>,
    prefetcher: Arc<parking_lot::RwLock<MemoryPrefetcher>>,
    /// Pluggable embedding provider
    embedder: Arc<parking_lot::RwLock<EmbeddingProviderRouter>>,
    /// MC Phase 1: Segment checkpointing for coarse search
    checkpoint_store: Arc<parking_lot::RwLock<CheckpointStore>>,
    /// MC Phase 3: SSC router for Top-k checkpoint selection
    ssc_router: Arc<parking_lot::RwLock<SSCRouter>>,
}

impl MemoryRouter {
    pub fn new(
        config: MemoryRouterConfig,
        semantic: Arc<SemanticMemoryStore>,
        episodic: Arc<EpisodicMemoryStore>,
        graph: Arc<GraphMemoryStore>,
        temporal: Arc<TemporalMemoryStore>,
    ) -> Self {
        // Create embedding provider router
        let embedder = if let Some(ref embed_config) = config.embedding_config {
            EmbeddingProviderRouter::from_config(embed_config)
        } else {
            // Use hash embedder as fallback with configured dimensions
            EmbeddingProviderRouter::new(Arc::new(HashEmbedder::new(config.embedding_dimensions)))
        };

        let checkpoint_config = config.checkpoint_config.clone();
        let ssc_router_config = config.ssc_router_config.clone();

        Self {
            config,
            semantic,
            episodic,
            graph,
            temporal,
            predictor: Arc::new(parking_lot::RwLock::new(ContextPredictor::new(
                Default::default(),
            ))),
            prefetcher: Arc::new(parking_lot::RwLock::new(MemoryPrefetcher::new(
                Default::default(),
            ))),
            embedder: Arc::new(parking_lot::RwLock::new(embedder)),
            checkpoint_store: Arc::new(parking_lot::RwLock::new(CheckpointStore::new(
                checkpoint_config,
            ))),
            ssc_router: Arc::new(parking_lot::RwLock::new(SSCRouter::new(ssc_router_config))),
        }
    }

    /// Query all memory stores with unified result
    ///
    /// Memory Caching flow (when checkpoints exist):
    /// 1. Generate query embedding
    /// 2. SSC router selects Top-k checkpoints (coarse O(N) search)
    /// 3. Expand high-relevance checkpoints into individual memory IDs
    /// 4. Query HNSW with the query embedding (fine O(M) search)
    /// 5. Apply checkpoint boost (1.3× soft multiplier) to memories from expanded checkpoints
    /// 6. Combine, rank, and predict next memories
    pub async fn query(&self, query: &MemoryQuery) -> Result<MemoryResponse> {
        let mut response = MemoryResponse::new();

        // Generate embedding for query using the active provider
        let query_embedding = if let Some(ref text) = query.text {
            // Clone the provider to avoid holding lock across await
            let provider = {
                let embedder = self.embedder.read();
                embedder.clone_provider()
            };
            let request = EmbeddingRequest::new(text);
            let response = provider.embed(request).await?;

            // Update predictor context
            let mut predictor = self.predictor.write();
            predictor.add_context(text, vec![]);

            Some(response.embedding)
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

        // MC Phase 1+3: Checkpoint-aware coarse search
        // Collect memory IDs from expanded checkpoints for boost scoring
        let boosted_memory_ids: std::collections::HashSet<uuid::Uuid> =
            if let Some(ref qe) = query_embedding {
                self.checkpoint_aware_search(qe)
            } else {
                std::collections::HashSet::new()
            };

        // Query semantic memory (uses HNSW via enriched embedding)
        if let Ok(semantic_results) = self.semantic.query(&enriched_query).await {
            for memory in semantic_results
                .into_iter()
                .take(self.config.max_results_per_store)
            {
                let mut relevance = memory.compute_relevance();
                // MC checkpoint boost: 1.3× soft multiplier for memories in expanded checkpoints
                if boosted_memory_ids.contains(&memory.id) {
                    relevance *= 1.3;
                }
                response.add_result(memory, MemoryType::Semantic, relevance);
            }
        }

        // Query episodic memory
        if let Ok(episodic_results) = self.episodic.query(&enriched_query).await {
            for memory in episodic_results
                .into_iter()
                .take(self.config.max_results_per_store)
            {
                let mut relevance = memory.compute_relevance();
                if boosted_memory_ids.contains(&memory.id) {
                    relevance *= 1.3;
                }
                response.add_result(memory, MemoryType::Episodic, relevance);
            }
        }

        // Query graph memory for relevant entities
        if let Some(ref text) = enriched_query.text {
            let entities = self
                .graph
                .search_entities(text, self.config.max_results_per_store)
                .await;
            for entity in entities {
                response.entities.push(entity);
            }
        }

        // Query temporal memory for relevant timeline events
        if let Some(ref text) = enriched_query.text {
            let events = self
                .temporal
                .search_events(text, self.config.max_results_per_store)
                .await;
            for event in events {
                response.temporal_events.push(event);
            }
        }

        // Combine and rank results
        if self.config.combine_scores {
            response.sort_by_relevance();
        }

        // Store query embedding for callers to reuse (avoids re-embedding)
        response.query_embedding = query_embedding.unwrap_or_default();

        // Predict next likely memories using prefetch data
        if self.config.enable_prefetch {
            let query_text = query.text.as_deref().unwrap_or("");
            let predictor = self.predictor.read();
            let ids: Vec<_> = response.results.iter().map(|r| r.memory.id).collect();
            response.predicted_next = predictor.predict(query_text, &ids);
        }

        Ok(response)
    }

    /// MC Checkpoint-aware search: coarse O(N) checkpoint selection
    /// followed by expansion of high-relevance checkpoints.
    ///
    /// Returns the set of memory IDs that should receive a relevance boost
    /// because they belong to expanded (high-relevance) checkpoints.
    fn checkpoint_aware_search(
        &self,
        query_embedding: &[f32],
    ) -> std::collections::HashSet<uuid::Uuid> {
        let mut boosted_ids = std::collections::HashSet::new();

        let checkpoint_store = self.checkpoint_store.read();

        // If no checkpoints exist, skip MC search (cold start: falls through to HNSW)
        if checkpoint_store.is_empty() {
            return boosted_ids;
        }

        // Step 1: Get all checkpoint IDs for SSC routing
        let all_checkpoint_ids = checkpoint_store.list_checkpoint_ids();
        if all_checkpoint_ids.is_empty() {
            return boosted_ids;
        }

        // Step 1b: Build transition probabilities from ContextPredictor (Bug 5 fix).
        // Only use transitions when the predictor has >10 observations,
        // matching the cold-start guard in ContextPredictor::predict().
        let transition_probs: Option<std::collections::HashMap<uuid::Uuid, f32>> = {
            let predictor = self.predictor.read();
            if predictor.transition_capacity() > 10 {
                if let Some(from_state) = predictor.last_intent_state {
                    let mut probs = std::collections::HashMap::new();
                    for &cp_id in &all_checkpoint_ids {
                        let to_state = 0;
                        probs.insert(cp_id, predictor.get_transition_prob(from_state, to_state));
                    }
                    Some(probs)
                } else {
                    None
                }
            } else {
                None
            }
        };

        // Step 2: SSC Route — select Top-k most relevant checkpoints
        // Uses 70/30 blend of cosine similarity + transition probabilities
        // when available (Bug 5 fix), pure cosine otherwise (cold start).
        let expansion_threshold = checkpoint_store.config().expansion_threshold;
        let routed = {
            let ssc_router = self.ssc_router.read();
            match transition_probs {
                Some(ref tp) => ssc_router.route_with_transitions(query_embedding, &all_checkpoint_ids, Some(tp)),
                None => ssc_router.route_with_scores(query_embedding, &all_checkpoint_ids),
            }
        };

        // Step 3: Expand high-relevance checkpoints (above expansion threshold)
        for (checkpoint_id, score) in &routed {
            if *score >= expansion_threshold {
                let memory_ids = checkpoint_store.expand_checkpoint(*checkpoint_id);
                for id in memory_ids {
                    boosted_ids.insert(id);
                }
            }
        }

        // Mark routed checkpoints as accessed (for eviction decisions)
        {
            let ssc_router = self.ssc_router.read();
            for (checkpoint_id, _) in &routed {
                ssc_router.mark_accessed(checkpoint_id);
            }
        }

        boosted_ids
    }

    /// Store a memory artifact in all relevant stores
    ///
    /// Also handles MC Phase 1 checkpoint creation: when the memory count
    /// threshold or time threshold is met, a checkpoint is created and
    /// registered with the SSC router.
    pub async fn store(&self, artifact: MemoryArtifact) -> Result<MemoryId> {
        let id = artifact.id;

        // Store in semantic memory
        self.semantic.store(artifact.clone()).await?;

        // Store in episodic memory
        self.episodic.store(artifact.clone()).await?;

        // Update prefetcher with new memory
        if !artifact.embedding.is_empty() {
            let mut prefetcher = self.prefetcher.write();
            prefetcher.register_memory(id, artifact.embedding.clone(), &artifact.tags);
        }

        // MC Phase 1: Increment checkpoint counter and check if we should create a checkpoint
        {
            let should_create = {
                let checkpoint_store = self.checkpoint_store.write();
                checkpoint_store.increment_memory_counter();
                checkpoint_store.should_checkpoint(0, chrono::Utc::now())
            };
            if should_create {
                let (threshold, session_id) = {
                    let checkpoint_store = self.checkpoint_store.read();
                    (
                        checkpoint_store.config().memory_threshold,
                        artifact.session_id,
                    )
                };
                let recent = self.collect_recent_memories_for_checkpoint(threshold).await;
                if !recent.is_empty() {
                    let checkpoint_store = self.checkpoint_store.write();
                    if let Ok((checkpoint, evicted_ids)) = checkpoint_store.create_checkpoint(&recent, session_id)
                    {
                        self.ssc_router.write().register_checkpoint(&checkpoint);
                        // Deregister evicted checkpoints from the SSC router
                        for evicted_id in evicted_ids {
                            self.ssc_router.write().deregister(&evicted_id);
                        }
                    }
                }
            }
        }

        Ok(id)
    }

    /// Collect recent memories for checkpoint creation.
    ///
    /// Queries the semantic store with no text/embedding filter, which
    /// falls back to timestamp-ordered retrieval (newest first).
    /// This replaces the previous "__recent__" magic string approach
    /// (Bug 6 fix) which produced arbitrary HNSW results instead of
    /// truly recent memories.
    async fn collect_recent_memories_for_checkpoint(&self, count: usize) -> Vec<MemoryArtifact> {
        let recent_query = MemoryQuery::new().with_limit(count);

        match self.semantic.query(&recent_query).await {
            Ok(memories) => memories.into_iter().take(count).collect(),
            Err(_) => Vec::new(),
        }
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
    pub async fn search_entities(
        &self,
        query: &str,
        limit: usize,
    ) -> Vec<rememnemosyne_graph::entity::GraphEntity> {
        self.graph.search_entities(query, limit).await
    }

    /// Get entity relationships
    pub async fn get_entity_relationships(
        &self,
        entity_id: &EntityId,
        max_depth: usize,
    ) -> Result<
        Vec<(
            rememnemosyne_graph::entity::GraphEntity,
            RelationshipType,
            f32,
        )>,
    > {
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

    /// Generate embedding for text using the active embedding provider
    pub async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Clone the provider Arc to avoid holding the lock across await
        let provider = {
            let embedder = self.embedder.read();
            embedder.clone_provider()
        };

        let request = EmbeddingRequest::new(text);
        let response = provider.embed(request).await?;
        let embedding = response.embedding;

        // Ensure embedding has correct dimensions
        if embedding.len() != self.config.embedding_dimensions {
            let mut corrected = vec![0.0; self.config.embedding_dimensions];
            let copy_len = std::cmp::min(embedding.len(), self.config.embedding_dimensions);
            corrected[..copy_len].copy_from_slice(&embedding[..copy_len]);
            Ok(corrected)
        } else {
            Ok(embedding)
        }
    }

    /// Generate embeddings for multiple texts in batch
    pub async fn generate_embedding_batch(&self, texts: &[String]) -> Vec<Vec<f32>> {
        let provider = {
            let embedder = self.embedder.read();
            embedder.clone_provider()
        };

        let requests: Vec<_> = texts.iter().map(EmbeddingRequest::new).collect();
        let responses = provider.embed_batch(requests).await;

        match responses {
            Ok(responses) => responses
                .into_iter()
                .map(|r| {
                    if r.embedding.len() != self.config.embedding_dimensions {
                        let mut corrected = vec![0.0; self.config.embedding_dimensions];
                        let copy_len =
                            std::cmp::min(r.embedding.len(), self.config.embedding_dimensions);
                        corrected[..copy_len].copy_from_slice(&r.embedding[..copy_len]);
                        corrected
                    } else {
                        r.embedding
                    }
                })
                .collect(),
            Err(e) => {
                tracing::warn!(error = %e, "Batch embedding failed, falling back to individual");
                let mut results = Vec::with_capacity(texts.len());
                for text in texts {
                    let emb = self
                        .generate_embedding(text)
                        .await
                        .unwrap_or_else(|_| vec![0.0; self.config.embedding_dimensions]);
                    results.push(emb);
                }
                results
            }
        }
    }

    /// Get embedding provider info
    pub fn get_provider_info(&self) -> crate::providers::ProviderInfo {
        self.embedder.read().provider_info()
    }

    /// Replace the active embedding provider
    pub fn set_embedding_provider(&self, provider: Arc<dyn EmbeddingProvider>) {
        let mut embedder = self.embedder.write();
        embedder.set_provider(provider);
    }

    /// Get embedding dimensions
    pub fn embedding_dimensions(&self) -> usize {
        self.config.embedding_dimensions
    }

    /// Get a reference to the checkpoint store (MC Phase 1)
    pub fn checkpoint_store(&self) -> &Arc<parking_lot::RwLock<CheckpointStore>> {
        &self.checkpoint_store
    }

    /// Get a reference to the SSC router (MC Phase 3)
    pub fn ssc_router(&self) -> &Arc<parking_lot::RwLock<SSCRouter>> {
        &self.ssc_router
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
    /// The query embedding used for this response (MC Phase 2: avoids re-embedding)
    pub query_embedding: Vec<f32>,
}

impl MemoryResponse {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            entities: Vec::new(),
            temporal_events: Vec::new(),
            predicted_next: Vec::new(),
            total_search_time_ms: 0,
            query_embedding: Vec::new(),
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
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
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
