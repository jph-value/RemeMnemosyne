use mnemosyne_core::*;
use mnemosyne_semantic::{SemanticMemoryStore, SemanticMemoryConfig};
use mnemosyne_episodic::{EpisodicMemoryStore, EpisodicMemoryConfig};
use mnemosyne_graph::{GraphMemoryStore, GraphMemoryConfig};
use mnemosyne_temporal::{TemporalMemoryStore, TemporalMemoryConfig};
#[cfg(feature = "persistence")]
use mnemosyne_storage::{RocksStorage, RocksConfig, SnapshotManager};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::router::{MemoryRouter, MemoryRouterConfig};
use crate::context::{ContextBuilderEngine, ContextBuilderConfig};

/// Main engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MnemosyneConfig {
    pub data_dir: String,
    pub semantic: SemanticMemoryConfig,
    pub episodic: EpisodicMemoryConfig,
    pub graph: GraphMemoryConfig,
    pub temporal: TemporalMemoryConfig,
    pub router: MemoryRouterConfig,
    pub context: ContextBuilderConfig,
    pub enable_persistence: bool,
}

impl Default for MnemosyneConfig {
    fn default() -> Self {
        Self {
            data_dir: "./mnemosyne_data".to_string(),
            semantic: SemanticMemoryConfig::default(),
            episodic: EpisodicMemoryConfig::default(),
            graph: GraphMemoryConfig::default(),
            temporal: TemporalMemoryConfig::default(),
            router: MemoryRouterConfig::default(),
            context: ContextBuilderConfig::default(),
            enable_persistence: true,
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
pub struct MnemosyneEngine {
    pub router: Arc<MemoryRouter>,
    pub context_builder: Arc<ContextBuilderEngine>,
    #[cfg(feature = "persistence")]
    pub storage: Option<Arc<RocksStorage>>,
    #[cfg(feature = "persistence")]
    pub snapshots: Option<Arc<parking_lot::RwLock<SnapshotManager>>>,
    config: MnemosyneConfig,
}

impl MnemosyneEngine {
    /// Create a new engine with the given configuration
    pub fn new(config: MnemosyneConfig) -> Result<Self> {
        // Create memory stores
        let semantic = Arc::new(SemanticMemoryStore::new(config.semantic.clone()));
        let episodic = Arc::new(EpisodicMemoryStore::new(config.episodic.clone()));
        let graph = Arc::new(GraphMemoryStore::new(config.graph.clone()));
        let temporal = Arc::new(TemporalMemoryStore::new(config.temporal.clone()));

        // Create router
        let router = Arc::new(MemoryRouter::new(
            config.router.clone(),
            semantic,
            episodic,
            graph,
            temporal,
        ));

        // Create context builder
        let context_builder = Arc::new(ContextBuilderEngine::new(config.context.clone()));

        // Create storage if enabled (only with persistence feature)
        #[cfg(feature = "persistence")]
        let (storage, snapshots) = if config.enable_persistence {
            let rocks_config = RocksConfig {
                path: config.data_dir.clone(),
                ..Default::default()
            };
            let snapshot_path = std::path::Path::new(&config.data_dir).join("snapshots");
            (
                Some(Arc::new(RocksStorage::new(rocks_config)?)),
                Some(Arc::new(parking_lot::RwLock::new(
                    SnapshotManager::new(snapshot_path)?
                ))),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            router,
            context_builder,
            #[cfg(feature = "persistence")]
            storage,
            #[cfg(feature = "persistence")]
            snapshots,
            config,
        })
    }

    /// Create with default configuration
    pub fn default() -> Result<Self> {
        Self::new(MnemosyneConfig::default())
    }

    /// Store a memory artifact
    pub async fn remember(
        &self,
        content: impl Into<String>,
        summary: impl Into<String>,
        trigger: MemoryTrigger,
    ) -> Result<MemoryId> {
        // Generate embedding placeholder
        let embedding = Vec::new(); // Would use actual embedding model

        let artifact = MemoryArtifact::new(
            MemoryType::Semantic,
            summary,
            content,
            embedding,
            trigger,
        );

        let id = self.router.store(artifact.clone()).await?;

        // Persist if storage enabled (only with persistence feature)
        #[cfg(feature = "persistence")]
        if let Some(ref storage) = self.storage {
            storage.store_memory(&artifact)?;
        }

        Ok(id)
    }

    /// Recall memories based on query
    pub async fn recall(&self, query: &str) -> Result<ContextBundle> {
        let mem_query = MemoryQuery::new()
            .with_text(query)
            .with_limit(20);

        let response = self.router.query(&mem_query).await?;
        let bundle = self.context_builder.build_context(&response, vec![], vec![]);

        Ok(bundle)
    }

    /// Recall with formatted context string
    pub async fn recall_formatted(&self, query: &str) -> Result<String> {
        let bundle = self.recall(query).await?;
        Ok(self.context_builder.format_context(&bundle))
    }

    /// Search entities by name/description
    pub async fn search_entities(&self, query: &str, limit: usize) -> Vec<mnemosyne_graph::entity::GraphEntity> {
        self.router.search_entities(query, limit).await
    }

    /// Create a snapshot of current state (requires persistence feature)
    #[cfg(feature = "persistence")]
    pub async fn create_snapshot(&self, name: &str) -> Result<()> {
        if let Some(ref snapshots) = self.snapshots {
            let stats = self.router.get_stats().await;
            
            let mut snap = snapshots.write();
            snap.create_snapshot(name, &stats)?;
        }
        Ok(())
    }

    /// Restore from snapshot (requires persistence feature)
    #[cfg(feature = "persistence")]
    pub async fn restore_snapshot(&self, name: &str) -> Result<()> {
        if let Some(ref snapshots) = self.snapshots {
            let snap = snapshots.read();
            let _stats: crate::router::RouterStats = snap.load_snapshot(name)?;
            // Would restore state from stats
        }
        Ok(())
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
        }
    }

    /// Flush all pending writes (requires persistence feature)
    #[cfg(feature = "persistence")]
    pub fn flush(&self) -> Result<()> {
        if let Some(ref storage) = self.storage {
            storage.flush()?;
        }
        Ok(())
    }
}

/// Engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    pub router: crate::router::RouterStats,
    pub config: MnemosyneConfig,
    #[cfg(feature = "persistence")]
    pub has_storage: bool,
    #[cfg(feature = "persistence")]
    pub has_snapshots: bool,
}

/// Builder pattern for creating engine
pub struct MnemosyneBuilder {
    config: MnemosyneConfig,
}

impl MnemosyneBuilder {
    pub fn new() -> Self {
        Self {
            config: MnemosyneConfig::default(),
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

    pub fn disable_persistence(mut self) -> Self {
        self.config.enable_persistence = false;
        self
    }

    pub fn build(self) -> Result<MnemosyneEngine> {
        MnemosyneEngine::new(self.config)
    }
}

impl Default for MnemosyneBuilder {
    fn default() -> Self {
        Self::new()
    }
}
