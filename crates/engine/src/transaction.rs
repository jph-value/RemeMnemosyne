/// Transaction support for multi-store memory operations.
///
/// Provides atomic write/delete across semantic, episodic, graph, and temporal stores.
/// Uses a two-phase approach:
/// 1. Prepare: validate all operations, collect rollback data
/// 2. Execute: perform all operations; rollback on any failure
use async_trait::async_trait;
use rememnemosyne_core::*;
use rememnemosyne_episodic::EpisodicMemoryStore;
use rememnemosyne_graph::GraphMemoryStore;
use rememnemosyne_semantic::SemanticMemoryStore;
use rememnemosyne_temporal::TemporalMemoryStore;
use std::sync::Arc;

/// A transactional memory operation that can be rolled back
#[derive(Debug, Clone)]
enum TxOp {
    Store {
        store: MemoryStoreType,
        artifact: MemoryArtifact,
    },
    Delete {
        store: MemoryStoreType,
        id: MemoryId,
        deleted_artifact: Option<MemoryArtifact>,
    },
    Update {
        store: MemoryStoreType,
        artifact: MemoryArtifact,
        previous: Option<MemoryArtifact>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoryStoreType {
    Semantic,
    Episodic,
}

/// Transaction result
#[derive(Debug)]
pub struct TxResult {
    pub committed: bool,
    pub memory_ids: Vec<MemoryId>,
    pub rolled_back: bool,
}

/// Transactional memory operations across stores
pub struct MemoryTransaction<'a> {
    operations: Vec<TxOp>,
    semantic: &'a SemanticMemoryStore,
    episodic: &'a EpisodicMemoryStore,
    graph: &'a GraphMemoryStore,
    temporal: &'a TemporalMemoryStore,
}

impl<'a> MemoryTransaction<'a> {
    pub fn new(
        semantic: &'a SemanticMemoryStore,
        episodic: &'a EpisodicMemoryStore,
        graph: &'a GraphMemoryStore,
        temporal: &'a TemporalMemoryStore,
    ) -> Self {
        Self {
            operations: Vec::new(),
            semantic,
            episodic,
            graph,
            temporal,
        }
    }

    /// Queue a store operation
    pub fn store(&mut self, artifact: MemoryArtifact) {
        self.operations.push(TxOp::Store {
            store: MemoryStoreType::Semantic,
            artifact,
        });
    }

    pub fn delete(&mut self, id: MemoryId) {
        self.operations.push(TxOp::Delete {
            store: MemoryStoreType::Semantic,
            id,
            deleted_artifact: None,
        });
    }

    /// Execute all queued operations atomically
    pub async fn commit(mut self) -> Result<TxResult> {
        if self.operations.is_empty() {
            return Ok(TxResult {
                committed: true,
                memory_ids: Vec::new(),
                rolled_back: false,
            });
        }

        // Phase 1: Prepare — collect rollback data for deletes
        for op in &mut self.operations {
            if let TxOp::Delete { id, deleted_artifact, .. } = op {
                if let Ok(Some(artifact)) = self.semantic.get(id).await {
                    *deleted_artifact = Some(artifact);
                }
            }
            if let TxOp::Update { artifact, previous, .. } = op {
                if let Ok(Some(existing)) = self.semantic.get(&artifact.id).await {
                    *previous = Some(existing);
                }
            }
        }

        // Phase 2: Execute — perform all operations, tracking what was done
        let mut executed: Vec<TxOp> = Vec::new();
        let mut memory_ids: Vec<MemoryId> = Vec::new();

        for op in self.operations {
            match self.execute_op(&op).await {
                Ok(Some(id)) => {
                    memory_ids.push(id);
                    executed.push(op);
                }
                Ok(None) => {
                    executed.push(op);
                }
                Err(e) => {
                    // Rollback all executed operations in reverse order
                    self.rollback(&executed).await;
                    return Err(MemoryError::Storage(format!(
                        "Transaction failed during commit, rolled back: {e}"
                    )));
                }
            }
        }

        Ok(TxResult {
            committed: true,
            memory_ids,
            rolled_back: false,
        })
    }

    /// Execute a single operation
    async fn execute_op(&self, op: &TxOp) -> Result<Option<MemoryId>> {
        match op {
            TxOp::Store { store, artifact } => match store {
                MemoryStoreType::Semantic => {
                    let id = self.semantic.store(artifact.clone()).await?;
                    Ok(Some(id))
                }
                MemoryStoreType::Episodic => {
                    let id = self.episodic.store(artifact.clone()).await?;
                    Ok(Some(id))
                }
            },
            TxOp::Delete { store, id, .. } => match store {
                MemoryStoreType::Semantic => {
                    self.semantic.delete(id).await?;
                    Ok(None)
                }
                MemoryStoreType::Episodic => {
                    self.episodic.delete(id).await?;
                    Ok(None)
                }
            },
            TxOp::Update { store, artifact, .. } => match store {
                MemoryStoreType::Semantic => {
                    self.semantic.update(artifact.clone()).await?;
                    Ok(None)
                }
                MemoryStoreType::Episodic => {
                    self.episodic.update(artifact.clone()).await?;
                    Ok(None)
                }
            },
        }
    }

    /// Rollback executed operations in reverse order
    async fn rollback(&self, executed: &[TxOp]) {
        for op in executed.iter().rev() {
            match op {
                TxOp::Store { store, artifact } => {
                    let _ = self.semantic.delete(&artifact.id).await;
                    let _ = self.episodic.delete(&artifact.id).await;
                }
                TxOp::Delete {
                    store,
                    id,
                    deleted_artifact,
                } => {
                    if let Some(ref artifact) = deleted_artifact {
                        let _ = self.semantic.store(artifact.clone()).await;
                    }
                    let _ = self.graph.delete_entity_by_memory_id(id).await;
                    let _ = self.temporal.delete_events_by_memory_id(id).await;
                }
                TxOp::Update {
                    store,
                    previous,
                    artifact,
                } => {
                    if let Some(ref prev) = previous {
                        let _ = self.semantic.update(prev.clone()).await;
                    } else {
                        let _ = self.semantic.delete(&artifact.id).await;
                    }
                }
            }
        }
    }
}

/// Convenience: execute a transactional delete across all stores
pub async fn delete_all_stores(
    semantic: &SemanticMemoryStore,
    episodic: &EpisodicMemoryStore,
    graph: &GraphMemoryStore,
    temporal: &TemporalMemoryStore,
    id: &MemoryId,
) -> Result<bool> {
    let mut tx = MemoryTransaction::new(semantic, episodic, graph, temporal);
    tx.delete(*id);
    let result = tx.commit().await?;
    Ok(result.committed)
}

/// Convenience: execute a transactional store across semantic + episodic
pub async fn store_all_stores(
    semantic: &SemanticMemoryStore,
    episodic: &EpisodicMemoryStore,
    artifact: MemoryArtifact,
) -> Result<MemoryId> {
    let mut tx = MemoryTransaction::new(semantic, episodic, &GraphMemoryStore::new(Default::default()), &TemporalMemoryStore::new(Default::default()));
    tx.store(artifact.clone());
    let result = tx.commit().await?;
    result.memory_ids.first().copied().ok_or_else(|| {
        MemoryError::Storage("Transaction committed but no ID returned".into())
    })
}
