use async_trait::async_trait;
use rememnemosyne_core::*;
use serde::{Deserialize, Serialize};

use crate::builder::RememnosyneEngine;

/// Core trait for agent memory operations
#[async_trait]
pub trait AgentMemory: Send + Sync {
    /// Remember something
    async fn remember(
        &self,
        content: &str,
        summary: &str,
        trigger: MemoryTrigger,
    ) -> Result<MemoryId>;

    /// Recall relevant memories
    async fn recall(&self, query: &str) -> Result<ContextBundle>;

    /// Recall with formatted context
    async fn recall_formatted(&self, query: &str) -> Result<String>;

    /// Store a memory artifact
    async fn store_artifact(&self, artifact: MemoryArtifact) -> Result<MemoryId>;

    /// Get memory by ID
    async fn get_memory(&self, id: &MemoryId) -> Result<Option<MemoryArtifact>>;

    /// Delete memory from all stores
    async fn forget(&self, id: &MemoryId) -> Result<bool>;

    /// Search entities
    async fn search_entities(
        &self,
        query: &str,
        limit: usize,
    ) -> Vec<rememnemosyne_graph::entity::GraphEntity>;

    /// Get context for LLM prompt
    async fn get_context(&self, query: &str, max_tokens: usize) -> Result<String>;
}

/// Agent memory API implementation
#[async_trait]
impl AgentMemory for RememnosyneEngine {
    async fn remember(
        &self,
        content: &str,
        summary: &str,
        trigger: MemoryTrigger,
    ) -> Result<MemoryId> {
        self.remember(content, summary, trigger).await
    }

    async fn recall(&self, query: &str) -> Result<ContextBundle> {
        self.recall(query).await
    }

    async fn recall_formatted(&self, query: &str) -> Result<String> {
        self.recall_formatted(query).await
    }

    async fn store_artifact(&self, artifact: MemoryArtifact) -> Result<MemoryId> {
        self.router.store(artifact).await
    }

    async fn get_memory(&self, id: &MemoryId) -> Result<Option<MemoryArtifact>> {
        self.router.get(id).await
    }

    async fn forget(&self, id: &MemoryId) -> Result<bool> {
        let mut errors = Vec::new();

        if let Err(e) = self.router.semantic.delete(id).await {
            errors.push(format!("semantic: {e}"));
        }
        if let Err(e) = self.router.episodic.delete(id).await {
            errors.push(format!("episodic: {e}"));
        }

        self.router.graph.delete_entity_by_memory_id(id).await;
        self.router.temporal.delete_events_by_memory_id(id).await;

        if errors.is_empty() {
            Ok(true)
        } else {
            tracing::warn!(memory_id = %id, errors = ?errors, "forget() had partial failures");
            Err(MemoryError::Storage(format!("Partial delete failures: {}", errors.join(", "))))
        }
    }

    async fn search_entities(
        &self,
        query: &str,
        limit: usize,
    ) -> Vec<rememnemosyne_graph::entity::GraphEntity> {
        self.router.search_entities(query, limit).await
    }

    async fn get_context(&self, query: &str, max_tokens: usize) -> Result<String> {
        let mut bundle = self.recall(query).await?;
        self.context_builder
            .prune_to_token_limit(&mut bundle, max_tokens);
        Ok(self.context_builder.format_context(&bundle))
    }
}

/// Streaming memory operations for real-time processing
pub struct StreamingMemoryHandler {
    engine: std::sync::Arc<RememnosyneEngine>,
    buffer: Vec<String>,
    buffer_size: usize,
}

impl Drop for StreamingMemoryHandler {
    fn drop(&mut self) {
        if !self.buffer.is_empty() {
            let combined = self.buffer.join(" ");
            let summary = combined.chars().take(100).collect::<String>() + "...";
            let engine = self.engine.clone();
            tokio::spawn(async move {
                if let Err(e) = engine.remember(&combined, &summary, MemoryTrigger::SystemOutput).await {
                    tracing::error!(error = %e, "StreamingMemoryHandler: failed to flush buffer on drop");
                }
            });
        }
    }
}

impl StreamingMemoryHandler {
    pub fn new(engine: RememnosyneEngine, buffer_size: usize) -> Self {
        Self {
            engine: std::sync::Arc::new(engine),
            buffer: Vec::new(),
            buffer_size,
        }
    }

    /// Add text to buffer (e.g., from streaming LLM output)
    pub fn add_text(&mut self, text: &str) {
        self.buffer.push(text.to_string());
        if self.buffer.len() >= self.buffer_size {
            self.flush();
        }
    }

    /// Flush buffer and create memory
    pub fn flush(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let combined = self.buffer.join(" ");
        let summary = combined.chars().take(100).collect::<String>() + "...";
        let engine = self.engine.clone();
        tokio::spawn(async move {
            if let Err(e) = engine.remember(&combined, &summary, MemoryTrigger::SystemOutput).await {
                tracing::error!(error = %e, "StreamingMemoryHandler: failed to store memory");
            }
        });
        self.buffer.clear();
    }

    /// Explicit shutdown — flushes remaining buffer
    pub async fn shutdown(mut self) {
        if !self.buffer.is_empty() {
            let combined = self.buffer.join(" ");
            let summary = combined.chars().take(100).collect::<String>() + "...";
            if let Err(e) = self.engine.remember(&combined, &summary, MemoryTrigger::SystemOutput).await {
                tracing::error!(error = %e, "StreamingMemoryHandler: failed to store on shutdown");
            }
        }
        self.buffer.clear();
    }

    /// Get current buffer content
    pub fn get_buffer(&self) -> &[String] {
        &self.buffer
    }

    /// Clear buffer without storing
    pub fn clear_buffer(&mut self) {
        self.buffer.clear();
    }
}

/// Memory operations result wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOperationResult {
    pub success: bool,
    pub memory_id: Option<MemoryId>,
    pub error: Option<String>,
    pub duration_ms: u64,
}

impl MemoryOperationResult {
    pub fn success(id: MemoryId, duration_ms: u64) -> Self {
        Self {
            success: true,
            memory_id: Some(id),
            error: None,
            duration_ms,
        }
    }

    pub fn failure(error: String, duration_ms: u64) -> Self {
        Self {
            success: false,
            memory_id: None,
            error: Some(error),
            duration_ms,
        }
    }
}

/// Batch memory operations
pub struct BatchMemoryOperations {
    operations: Vec<MemoryOperation>,
}

impl BatchMemoryOperations {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn add_store(&mut self, artifact: MemoryArtifact) {
        self.operations.push(MemoryOperation::Store(artifact));
    }

    pub fn add_delete(&mut self, id: MemoryId) {
        self.operations.push(MemoryOperation::Delete(id));
    }

    pub fn add_update(&mut self, artifact: MemoryArtifact) {
        self.operations.push(MemoryOperation::Update(artifact));
    }

    pub async fn execute(&self, engine: &RememnosyneEngine) -> Vec<MemoryOperationResult> {
        let mut results = Vec::new();

        for op in &self.operations {
            let start = std::time::Instant::now();

            let result = match op {
                MemoryOperation::Store(artifact) => {
                    match engine.router.store(artifact.clone()).await {
                        Ok(id) => {
                            MemoryOperationResult::success(id, start.elapsed().as_millis() as u64)
                        }
                        Err(e) => MemoryOperationResult::failure(
                            e.to_string(),
                            start.elapsed().as_millis() as u64,
                        ),
                    }
                }
                MemoryOperation::Delete(id) => {
                    let mut success = true;
                    if let Err(e) = engine.router.semantic.delete(id).await {
                        tracing::warn!(error = %e, "Batch delete failed (semantic)");
                        success = false;
                    }
                    if let Err(e) = engine.router.episodic.delete(id).await {
                        tracing::warn!(error = %e, "Batch delete failed (episodic)");
                        success = false;
                    }
                    engine.router.graph.delete_entity_by_memory_id(id).await;
                    engine.router.temporal.delete_events_by_memory_id(id).await;
                    if success {
                        MemoryOperationResult::success(*id, start.elapsed().as_millis() as u64)
                    } else {
                        MemoryOperationResult::failure(
                            "Partial delete failure".into(),
                            start.elapsed().as_millis() as u64,
                        )
                    }
                }
                MemoryOperation::Update(artifact) => {
                    match engine.router.semantic.update(artifact.clone()).await {
                        Ok(_) => MemoryOperationResult::success(
                            artifact.id,
                            start.elapsed().as_millis() as u64,
                        ),
                        Err(e) => MemoryOperationResult::failure(
                            e.to_string(),
                            start.elapsed().as_millis() as u64,
                        ),
                    }
                }
            };

            results.push(result);
        }

        results
    }
}

impl Default for BatchMemoryOperations {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
enum MemoryOperation {
    Store(MemoryArtifact),
    Delete(MemoryId),
    Update(MemoryArtifact),
}
