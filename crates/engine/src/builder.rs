use rememnemosyne_core::*;
use rememnemosyne_episodic::{EpisodicMemoryConfig, EpisodicMemoryStore};
use rememnemosyne_graph::{GraphMemoryConfig, GraphMemoryStore};
use rememnemosyne_semantic::{SemanticMemoryConfig, SemanticMemoryStore};
#[cfg(feature = "persistence")]
use rememnemosyne_storage::backend::StorageBackend;
#[cfg(feature = "persistence")]
use rememnemosyne_storage::snapshot::SnapshotManager;
#[cfg(feature = "persistence")]
use rememnemosyne_storage::RocksStorage;
#[cfg(feature = "persistence")]
use rememnemosyne_storage::StorageConfig;
use rememnemosyne_temporal::{TemporalMemoryConfig, TemporalMemoryStore};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::context::{ContextBuilderConfig, ContextBuilderEngine};
use crate::router::{MemoryRouter, MemoryRouterConfig};

/// Main engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RememnosyneConfig {
    pub data_dir: String,
    pub semantic: SemanticMemoryConfig,
    pub episodic: EpisodicMemoryConfig,
    pub graph: GraphMemoryConfig,
    pub temporal: TemporalMemoryConfig,
    pub router: MemoryRouterConfig,
    pub context: ContextBuilderConfig,
    pub enable_persistence: bool,
    #[cfg(feature = "persistence")]
    pub storage_config: StorageConfig,
    #[cfg(not(feature = "persistence"))]
    pub storage_config: Option<()>,
}

impl Default for RememnosyneConfig {
    fn default() -> Self {
        Self {
            data_dir: "./rememnemosyne_data".to_string(),
            semantic: SemanticMemoryConfig::default(),
            episodic: EpisodicMemoryConfig::default(),
            graph: GraphMemoryConfig::default(),
            temporal: TemporalMemoryConfig::default(),
            router: MemoryRouterConfig::default(),
            context: ContextBuilderConfig::default(),
            enable_persistence: true,
            #[cfg(feature = "persistence")]
            storage_config: StorageConfig::default(),
            #[cfg(not(feature = "persistence"))]
            storage_config: None,
        }
    }
}

/// The Mnemosyne Memory Engine
///
/// A unified agentic memory system with:
/// - Semantic memory (TurboQuant compressed vectors)
/// - Episodic memory (conversation episodes)
/// - Graph memory (entity relationships)
/// - Temporal memory (timeline events)
/// - Cognitive engine (micro-embeddings, prefetching)
pub struct RememnosyneEngine {
    pub router: Arc<MemoryRouter>,
    pub context_builder: Arc<ContextBuilderEngine>,
    #[cfg(feature = "persistence")]
    pub storage: Option<Arc<dyn StorageBackend + Send + Sync>>,
    #[cfg(feature = "persistence")]
    pub snapshots: Option<Arc<SnapshotManager>>,
    #[cfg(not(feature = "persistence"))]
    pub storage: Option<()>,
    #[cfg(not(feature = "persistence"))]
    pub snapshots: Option<()>,
    config: RememnosyneConfig,
}

impl RememnosyneEngine {
    /// Create a new engine with the given configuration
    pub fn new(config: RememnosyneConfig) -> Result<Self> {
        // Create memory stores
        let semantic = Arc::new(SemanticMemoryStore::new(config.semantic.clone()));
        let episodic = Arc::new(EpisodicMemoryStore::new(config.episodic.clone()));
        let graph = Arc::new(GraphMemoryStore::new(config.graph.clone()));
        let temporal = Arc::new(TemporalMemoryStore::new(config.temporal.clone()));

        // Ensure router embedding dimensions match semantic store
        let mut router_config = config.router.clone();
        router_config.embedding_dimensions = config.semantic.dimensions;

        // Create router
        let router = Arc::new(MemoryRouter::new(
            router_config,
            semantic,
            episodic,
            graph,
            temporal,
        ));

        // Create context builder
        let context_builder = Arc::new(ContextBuilderEngine::new(config.context.clone()));

        // Create storage if enabled
        #[cfg(feature = "persistence")]
        let (storage, snapshots) = if config.enable_persistence {
            let storage = create_storage_backend(&config)?;
            let snapshot_manager = SnapshotManager::new(&config.data_dir)?;
            (
                Some(storage as Arc<dyn StorageBackend + Send + Sync>),
                Some(Arc::new(snapshot_manager)),
            )
        } else {
            (None, None)
        };

        #[cfg(not(feature = "persistence"))]
        let (storage, snapshots) = (None, None);

        Ok(Self {
            router,
            context_builder,
            storage,
            snapshots,
            config,
        })
    }

    /// Create with default configuration
    pub fn try_default() -> Result<Self> {
        Self::new(RememnosyneConfig::default())
    }

    /// Create with in-memory storage only (no persistence)
    pub fn in_memory() -> Result<Self> {
        let mut config = RememnosyneConfig::default();
        config.enable_persistence = false;
        Self::new(config)
    }

    /// Store a memory artifact
    pub async fn remember(
        &self,
        content: impl Into<String>,
        summary: impl Into<String>,
        trigger: MemoryTrigger,
    ) -> Result<MemoryId> {
        let content_str = content.into();

        // Sanitize content before storing
        let sanitized = crate::sanitizer::sanitize_input(&content_str);
        let safe_content = if sanitized.is_suspicious {
            tracing::warn!(
                "Suspicious input detected in remember(): {:?}",
                sanitized.detected_patterns
            );
            sanitized.clean_text
        } else {
            sanitized.clean_text
        };

        // Generate embedding using the router's embedder
        let embedding = self.generate_embedding(&safe_content).await;

        let artifact = MemoryArtifact::new(
            MemoryType::Semantic,
            summary,
            safe_content,
            embedding,
            trigger,
        );

        let id = self.router.store(artifact.clone()).await?;

        // Persist if storage enabled
        #[cfg(feature = "persistence")]
        if let Some(ref storage) = self.storage {
            let key = artifact.id.as_bytes();
            let value = bincode::serialize(&artifact)
                .map_err(|e| MemoryError::Serialization(e.to_string()))?;
            storage.put(key, &value)?;
        }

        Ok(id)
    }

    /// Store a memory from a MemoryInput struct (supports namespace, confidence, etc.)
    pub async fn remember_from_input(&self, input: MemoryInput) -> Result<MemoryId> {
        let sanitized = crate::sanitizer::sanitize_input(&input.content);
        let safe_content = if sanitized.is_suspicious {
            tracing::warn!(
                "Suspicious input detected in remember_from_input(): {:?}",
                sanitized.detected_patterns
            );
            sanitized.clean_text
        } else {
            input.content
        };

        let embedding = self.generate_embedding(&safe_content).await;

        let mut artifact = MemoryArtifact::new(
            input.memory_type,
            &input.summary,
            safe_content,
            embedding,
            input.trigger,
        )
        .with_importance(input.importance);

        if let Some(ref ns) = input.namespace {
            artifact.namespace = Some(ns.clone());
        }
        if let Some(ref agent) = input.agent_id {
            artifact.agent_id = Some(agent.clone());
        }
        if let Some(conf) = input.confidence {
            artifact.confidence = Some(conf);
        }
        if !input.source_events.is_empty() {
            artifact.source_events = input.source_events;
        }
        if let Some(tier) = input.tier {
            artifact.tier = Some(tier);
        }
        if let Some(sid) = input.session_id {
            artifact.session_id = Some(sid);
        }

        let id = self.router.store(artifact.clone()).await?;

        #[cfg(feature = "persistence")]
        if let Some(ref storage) = self.storage {
            let key = artifact.id.as_bytes();
            let value = bincode::serialize(&artifact)
                .map_err(|e| MemoryError::Serialization(e.to_string()))?;
            storage.put(key, &value)?;
        }

        Ok(id)
    }

    /// Batch ingestion API — process multiple memories in parallel.
    ///
    /// Embeddings are generated in parallel (chunked by 32), then
    /// artifacts are stored across all stores concurrently.
    ///
    /// Target throughput: 1000 events/second with 512-dim embeddings.
    pub async fn remember_batch(&self, items: Vec<MemoryInput>) -> Result<Vec<MemoryId>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let start = std::time::Instant::now();
        let total = items.len();
        tracing::debug!(count = total, "Starting batch ingestion");

        // Sanitize all contents first
        let sanitized_items: Vec<MemoryInput> = items
            .into_iter()
            .map(|mut item| {
                let sanitized = crate::sanitizer::sanitize_input(&item.content);
                if sanitized.is_suspicious {
                    tracing::warn!(
                        "Suspicious input in batch: {:?}",
                        sanitized.detected_patterns
                    );
                }
                item.content = sanitized.clean_text;
                item
            })
            .collect();

        // Generate embeddings in parallel (chunk size: 32)
        let chunk_size = 32.min(total);
        let mut all_embeddings = Vec::with_capacity(total);
        for chunk in sanitized_items.chunks(chunk_size) {
            let texts: Vec<String> = chunk.iter().map(|i| i.content.clone()).collect();
            let embeddings = self.router.generate_embedding_batch(&texts).await;
            all_embeddings.extend(embeddings);
        }

        // Build artifacts
        let artifacts: Vec<MemoryArtifact> = sanitized_items
            .into_iter()
            .zip(all_embeddings.into_iter())
            .map(|(input, embedding)| {
                let mut artifact = MemoryArtifact::new(
                    input.memory_type,
                    &input.summary,
                    input.content,
                    embedding,
                    input.trigger,
                )
                .with_importance(input.importance);

                if let Some(ref ns) = input.namespace {
                    artifact.namespace = Some(ns.clone());
                }
                if let Some(ref agent) = input.agent_id {
                    artifact.agent_id = Some(agent.clone());
                }
                if let Some(conf) = input.confidence {
                    artifact.confidence = Some(conf);
                }
                if !input.source_events.is_empty() {
                    artifact.source_events = input.source_events;
                }
                if let Some(tier) = input.tier {
                    artifact.tier = Some(tier);
                }
                if let Some(sid) = input.session_id {
                    artifact.session_id = Some(sid);
                }
                artifact
            })
            .collect();

        // Store all artifacts via router (uses DashMap for concurrent writes)
        let mut ids = Vec::with_capacity(artifacts.len());
        for artifact in artifacts {
            match self.router.store(artifact.clone()).await {
                Ok(id) => {
                    #[cfg(feature = "persistence")]
                    if let Some(ref storage) = self.storage {
                        let key = artifact.id.as_bytes();
                        if let Ok(value) = bincode::serialize(&artifact) {
                            let _ = storage.put(key, &value);
                        }
                    }
                    ids.push(id);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to store batch item");
                }
            }
        }

        let elapsed = start.elapsed();
        tracing::info!(
            count = ids.len(),
            total = total,
            elapsed_ms = elapsed.as_millis(),
            "Batch ingestion complete"
        );

        Ok(ids)
    }

    /// Generate embedding for text
    async fn generate_embedding(&self, text: &str) -> Vec<f32> {
        // Use the router's embedding provider
        self.router
            .generate_embedding(text)
            .await
            .unwrap_or_else(|e| {
                tracing::warn!("Embedding generation failed: {}", e);
                vec![0.0; self.router.embedding_dimensions()]
            })
    }

    /// Recall memories based on query
    ///
    /// Uses MC Gated Residual Memory (GRM) weights when a query embedding
    /// is available, falling back to standard context assembly when not.
    pub async fn recall(&self, query: &str) -> Result<ContextBundle> {
        // Sanitize query before processing
        let sanitized = crate::sanitizer::sanitize_input(query);
        let safe_query = if sanitized.is_suspicious {
            tracing::warn!(
                "Suspicious query in recall(): {:?}",
                sanitized.detected_patterns
            );
            &sanitized.clean_text
        } else {
            query
        };

        let mem_query = MemoryQuery::new()
            .with_text(safe_query)
            .with_limit(self.config.context.max_memories);

        let response = self.router.query(&mem_query).await?;

        // Generate embedding for MC-gated context assembly
        let query_embedding = self.generate_embedding(safe_query).await;

        let bundle = if !query_embedding.is_empty() {
            // MC Phase 2: Gated Context Assembly — use contribution weights
            self.context_builder
                .build_context_weighted(&response, vec![], vec![], &query_embedding)
        } else {
            // Cold start: fall through to standard assembly
            self.context_builder.build_context(&response, vec![], vec![])
        };

        Ok(bundle)
    }

    /// Recall with formatted context string
    pub async fn recall_formatted(&self, query: &str) -> Result<String> {
        let bundle = self.recall(query).await?;
        let formatted = self.context_builder.format_context(&bundle);

        // Sanitize the context output to strip control characters
        let safe_context = crate::sanitizer::sanitize_context(&formatted);
        Ok(safe_context)
    }

    /// Search entities by name/description
    pub async fn search_entities(
        &self,
        query: &str,
        limit: usize,
    ) -> Vec<rememnemosyne_graph::entity::GraphEntity> {
        self.router.search_entities(query, limit).await
    }

    /// Create a snapshot of current state
    #[cfg(feature = "persistence")]
    pub fn create_snapshot(&self, name: &str) -> Result<()> {
        if let (Some(ref storage), Some(ref snapshots)) = (&self.storage, &self.snapshots) {
            snapshots.save_snapshot(name, storage.as_ref())?;
        }
        Ok(())
    }

    /// Restore from snapshot
    #[cfg(feature = "persistence")]
    pub fn restore_snapshot(&self, name: &str) -> Result<()> {
        if let (Some(ref storage), Some(ref snapshots)) = (&self.storage, &self.snapshots) {
            snapshots.restore_snapshot(name, storage.as_ref())?;
        }
        Ok(())
    }

    /// List available snapshots
    #[cfg(feature = "persistence")]
    pub fn list_snapshots(&self) -> Result<Vec<rememnemosyne_storage::snapshot::SnapshotInfo>> {
        if let Some(ref snapshots) = self.snapshots {
            snapshots.list_snapshots()
        } else {
            Ok(Vec::new())
        }
    }

    /// Get engine statistics
    pub async fn get_stats(&self) -> EngineStats {
        let router_stats = self.router.get_stats().await;

        EngineStats {
            router: router_stats,
            config: self.config.clone(),
            #[cfg(feature = "persistence")]
            has_storage: self.storage.is_some(),
            #[cfg(feature = "persistence")]
            has_snapshots: self.snapshots.is_some(),
            #[cfg(not(feature = "persistence"))]
            has_storage: false,
            #[cfg(not(feature = "persistence"))]
            has_snapshots: false,
        }
    }

    /// Flush all pending writes
    #[cfg(feature = "persistence")]
    pub fn flush(&self) -> Result<()> {
        if let Some(ref storage) = self.storage {
            storage.flush()?;
        }
        Ok(())
    }
}

/// Create the appropriate storage backend based on config
#[cfg(feature = "persistence")]
fn create_storage_backend(
    config: &RememnosyneConfig,
) -> Result<Arc<dyn StorageBackend + Send + Sync>> {
    let storage_path = format!("{}/data", config.data_dir);

    // Priority: sled (pure Rust) is default, RocksDB is opt-in
    // This ensures pure Rust deployment by default

    #[cfg(feature = "sled-storage")]
    {
        tracing::info!("Using sled storage backend (pure Rust)");
        let storage = SledStorage::new(&storage_path)?;
        return Ok(Arc::new(storage));
    }

    // Fallback to RocksDB only if sled is not available
    #[cfg(all(feature = "persistence", not(feature = "sled-storage")))]
    {
        tracing::info!("Using RocksDB storage backend");
        let storage = RocksStorage::new(&storage_path)?;
        return Ok(Arc::new(storage));
    }

    #[cfg(not(any(feature = "sled-storage", feature = "persistence")))]
    {
        let _ = config;
        Err(MemoryError::Storage("No storage backend enabled".into()))
    }
}

/// Engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    pub router: crate::router::RouterStats,
    pub config: RememnosyneConfig,
    pub has_storage: bool,
    pub has_snapshots: bool,
}

/// Builder pattern for creating engine
pub struct MnemosyneBuilder {
    config: RememnosyneConfig,
}

impl MnemosyneBuilder {
    pub fn new() -> Self {
        Self {
            config: RememnosyneConfig::default(),
        }
    }

    pub fn with_data_dir(mut self, path: impl Into<String>) -> Self {
        self.config.data_dir = path.into();
        self
    }

    pub fn with_semantic_config(mut self, config: SemanticMemoryConfig) -> Self {
        self.config.semantic = config;
        self
    }

    pub fn with_episodic_config(mut self, config: EpisodicMemoryConfig) -> Self {
        self.config.episodic = config;
        self
    }

    pub fn with_graph_config(mut self, config: GraphMemoryConfig) -> Self {
        self.config.graph = config;
        self
    }

    pub fn with_temporal_config(mut self, config: TemporalMemoryConfig) -> Self {
        self.config.temporal = config;
        self
    }

    pub fn with_context_config(mut self, config: ContextBuilderConfig) -> Self {
        self.config.context = config;
        self
    }

    pub fn with_router_config(mut self, config: MemoryRouterConfig) -> Self {
        self.config.router = config;
        self
    }

    pub fn disable_persistence(mut self) -> Self {
        self.config.enable_persistence = false;
        self
    }

    pub fn build(self) -> Result<RememnosyneEngine> {
        RememnosyneEngine::new(self.config)
    }
}

impl Default for MnemosyneBuilder {
    fn default() -> Self {
        Self::new()
    }
}
