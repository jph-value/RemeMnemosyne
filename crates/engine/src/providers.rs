/// LLM-Provider Agnostic Abstraction Layer
///
/// This module provides a unified provider system that allows users to plug in
/// their own AI APIs for embeddings, reasoning, and agent tasks.
///
/// Three configurable layers:
/// 1. Embedding Providers - OpenAI, Voyage, Cohere, Ollama, local, custom
/// 2. Reasoning Providers - OpenAI, Anthropic, OpenRouter, local, custom
/// 3. Agent Providers - Verification, analysis, report, simulation agents
///
/// Embedding provider types are defined in rememnemosyne-core for cross-crate use.
// Re-export embedding types from core
pub use rememnemosyne_core::{
    EmbeddingProvider, EmbeddingProviderConfig, EmbeddingProviderType, EmbeddingRequest,
    EmbeddingResponse, HashEmbedder,
};

use rememnemosyne_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// Need async_trait for trait definitions in this module
use async_trait::async_trait;

// ============================================================================
// Embedding Provider Router
// ============================================================================

/// EmbeddingProviderRouter manages the active embedding provider
/// and provides a unified interface for embedding operations.
pub struct EmbeddingProviderRouter {
    /// The active embedding provider
    provider: Arc<dyn EmbeddingProvider>,
}

impl EmbeddingProviderRouter {
    /// Create a new router with a specific provider
    pub fn new(provider: Arc<dyn EmbeddingProvider>) -> Self {
        Self { provider }
    }

    /// Create with default hash-based embedder (fallback)
    pub fn with_default() -> Self {
        Self::new(Arc::new(HashEmbedder::default_embedder()))
    }

    /// Create with configured provider
    pub fn from_config(config: &EmbeddingProviderConfig) -> Self {
        match config.provider {
            EmbeddingProviderType::Local => {
                // Use hash embedder as default local
                Self::new(Arc::new(HashEmbedder::new(config.dimensions)))
            }
            // Other providers would be implemented here
            // For now, fall back to hash
            _ => {
                tracing::warn!(
                    provider = ?config.provider,
                    "Provider not yet implemented, using hash fallback"
                );
                Self::new(Arc::new(HashEmbedder::new(config.dimensions)))
            }
        }
    }

    /// Generate embedding for text
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = EmbeddingRequest::new(text);
        let response = self.provider.embed(request).await?;
        Ok(response.embedding)
    }

    /// Clone the provider Arc for use across await boundaries
    pub fn clone_provider(&self) -> Arc<dyn EmbeddingProvider> {
        self.provider.clone()
    }

    /// Generate embeddings for multiple texts
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let requests: Vec<_> = texts.iter().map(EmbeddingRequest::new).collect();
        let responses = self.provider.embed_batch(requests).await?;
        Ok(responses.into_iter().map(|r| r.embedding).collect())
    }

    /// Get provider info
    pub fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            provider_type: self.provider.provider_type(),
            model: self.provider.model_name().to_string(),
            dimensions: self.provider.dimensions(),
        }
    }

    /// Replace the active provider
    pub fn set_provider(&mut self, provider: Arc<dyn EmbeddingProvider>) {
        self.provider = provider;
    }
}

/// Provider information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    pub provider_type: EmbeddingProviderType,
    pub model: String,
    pub dimensions: usize,
}

// ============================================================================
// Layer 2: Reasoning Providers
// ============================================================================

/// Reasoning provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningProviderConfig {
    /// Provider type
    pub provider: ReasoningProviderType,
    /// Model name
    pub model: String,
    /// API key
    pub api_key: Option<String>,
    /// Base URL
    pub base_url: Option<String>,
    /// Max tokens for response
    pub max_tokens: usize,
    /// Temperature
    pub temperature: f32,
    /// Request timeout (seconds)
    pub timeout_secs: u64,
}

impl Default for ReasoningProviderConfig {
    fn default() -> Self {
        Self {
            provider: ReasoningProviderType::OpenAI,
            model: "gpt-4o-mini".to_string(),
            api_key: None,
            base_url: None,
            max_tokens: 4096,
            temperature: 0.7,
            timeout_secs: 60,
        }
    }
}

/// Supported reasoning providers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReasoningProviderType {
    /// OpenAI (GPT-4, GPT-4o, etc.)
    OpenAI,
    /// Anthropic (Claude)
    Anthropic,
    /// OpenRouter (multi-provider aggregator)
    OpenRouter,
    /// Ollama local models
    Ollama,
    /// Custom API endpoint
    Custom,
}

/// Reasoning task types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningTask {
    /// Summarize a memory
    SummarizeMemory,
    /// Detect duplicate memories
    DetectDuplicates,
    /// Score memory importance
    ScoreImportance,
    /// Create narrative abstraction
    CreateNarrative,
    /// Extract entities from text
    ExtractEntities,
    /// Custom task with prompt
    Custom { prompt: String },
}

/// Reasoning request
#[derive(Debug, Clone)]
pub struct ReasoningRequest {
    /// Task to perform
    pub task: ReasoningTask,
    /// Input context/memories
    pub context: String,
    /// Optional model override
    pub model: Option<String>,
    /// Optional system prompt
    pub system_prompt: Option<String>,
}

/// Reasoning response
#[derive(Debug, Clone)]
pub struct ReasoningResponse {
    /// Generated text
    pub text: String,
    /// Model used
    pub model: String,
    /// Token usage
    pub prompt_tokens: Option<usize>,
    pub completion_tokens: Option<usize>,
}

/// Reasoning provider trait
#[async_trait]
pub trait ReasoningProvider: Send + Sync {
    /// Execute a reasoning task
    async fn reason(&self, request: ReasoningRequest) -> Result<ReasoningResponse>;

    /// Get provider name
    fn provider_name(&self) -> &str;

    /// Get model name
    fn model_name(&self) -> &str;
}

// ============================================================================
// Layer 3: Agent Providers
// ============================================================================

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProviderConfig {
    /// Agent type
    pub agent_type: AgentType,
    /// Reasoning provider to use
    pub reasoning_provider: ReasoningProviderType,
    /// Model name
    pub model: String,
    /// API key
    pub api_key: Option<String>,
    /// Base URL
    pub base_url: Option<String>,
    /// Agent-specific system prompt
    pub system_prompt: String,
    /// Max reasoning iterations
    pub max_iterations: usize,
}

impl Default for AgentProviderConfig {
    fn default() -> Self {
        Self {
            agent_type: AgentType::Verification,
            reasoning_provider: ReasoningProviderType::OpenAI,
            model: "gpt-4o-mini".to_string(),
            api_key: None,
            base_url: None,
            system_prompt: "You are a helpful assistant.".to_string(),
            max_iterations: 5,
        }
    }
}

/// Agent types for intelligence analysis
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentType {
    /// Verify memory accuracy
    Verification,
    /// Analyze patterns and trends
    Analysis,
    /// Generate intelligence reports
    Report,
    /// Plan and run simulations
    SimulationPlanner,
    /// Extract and link evidence
    EvidenceExtractor,
    /// Custom agent with specific behavior
    Custom { name: String },
}

/// Agent execution request
#[derive(Debug, Clone)]
pub struct AgentRequest {
    /// Agent type
    pub agent_type: AgentType,
    /// Input data/context
    pub input: String,
    /// Supporting memories
    pub context_memories: Vec<String>,
    /// Specific instructions
    pub instructions: Option<String>,
}

/// Agent execution response
#[derive(Debug, Clone)]
pub struct AgentResponse {
    /// Agent output
    pub output: String,
    /// Agent type that produced this
    pub agent_type: AgentType,
    /// Reasoning steps taken
    pub reasoning_steps: Vec<String>,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

/// Agent provider trait
#[async_trait]
pub trait AgentProvider: Send + Sync {
    /// Execute an agent task
    async fn execute(&self, request: AgentRequest) -> Result<AgentResponse>;

    /// Get agent type
    fn agent_type(&self) -> &AgentType;

    /// Get agent name
    fn agent_name(&self) -> &str;
}

// ============================================================================
// Provider Registry
// ============================================================================

/// Central provider registry that manages all provider instances
pub struct ProviderRegistry {
    /// Active embedding provider
    embedding_provider: parking_lot::RwLock<Option<Arc<dyn EmbeddingProvider>>>,
    /// Active reasoning provider
    reasoning_provider: parking_lot::RwLock<Option<Arc<dyn ReasoningProvider>>>,
    /// Registered agent providers
    agent_providers: parking_lot::RwLock<HashMap<AgentType, Arc<dyn AgentProvider>>>,
}

impl ProviderRegistry {
    /// Create new empty registry
    pub fn new() -> Self {
        Self {
            embedding_provider: parking_lot::RwLock::new(None),
            reasoning_provider: parking_lot::RwLock::new(None),
            agent_providers: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Set the active embedding provider
    pub fn set_embedding_provider(&self, provider: Arc<dyn EmbeddingProvider>) {
        *self.embedding_provider.write() = Some(provider);
    }

    /// Set the active reasoning provider
    pub fn set_reasoning_provider(&self, provider: Arc<dyn ReasoningProvider>) {
        *self.reasoning_provider.write() = Some(provider);
    }

    /// Register an agent provider
    pub fn register_agent(&self, agent: Arc<dyn AgentProvider>) {
        let agent_type = agent.agent_type().clone();
        self.agent_providers.write().insert(agent_type, agent);
    }

    /// Get embedding provider
    pub fn get_embedding_provider(&self) -> Option<Arc<dyn EmbeddingProvider>> {
        self.embedding_provider.read().clone()
    }

    /// Get reasoning provider
    pub fn get_reasoning_provider(&self) -> Option<Arc<dyn ReasoningProvider>> {
        self.reasoning_provider.read().clone()
    }

    /// Get agent provider by type
    pub fn get_agent_provider(&self, agent_type: &AgentType) -> Option<Arc<dyn AgentProvider>> {
        self.agent_providers.read().get(agent_type).cloned()
    }

    /// Check if embedding provider is configured
    pub fn has_embedding_provider(&self) -> bool {
        self.embedding_provider.read().is_some()
    }

    /// Check if reasoning provider is configured
    pub fn has_reasoning_provider(&self) -> bool {
        self.reasoning_provider.read().is_some()
    }

    /// List registered agent types
    pub fn registered_agents(&self) -> Vec<AgentType> {
        self.agent_providers.read().keys().cloned().collect()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Stub Implementations (for when no provider is configured)
// ============================================================================

/// Stub embedding provider that returns zero vectors
pub struct StubEmbeddingProvider {
    dimensions: usize,
}

impl StubEmbeddingProvider {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

#[async_trait]
impl EmbeddingProvider for StubEmbeddingProvider {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let _ = request;
        Ok(EmbeddingResponse {
            embedding: vec![0.0; self.dimensions],
            model: "stub".to_string(),
            token_count: None,
        })
    }

    fn provider_type(&self) -> EmbeddingProviderType {
        EmbeddingProviderType::Custom
    }

    fn model_name(&self) -> &str {
        "stub-model"
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Stub reasoning provider
pub struct StubReasoningProvider;

#[async_trait]
impl ReasoningProvider for StubReasoningProvider {
    async fn reason(&self, request: ReasoningRequest) -> Result<ReasoningResponse> {
        Ok(ReasoningResponse {
            text: format!(
                "[Stub reasoning for: {}]",
                request.context.chars().take(50).collect::<String>()
            ),
            model: "stub".to_string(),
            prompt_tokens: None,
            completion_tokens: None,
        })
    }

    fn provider_name(&self) -> &str {
        "stub"
    }

    fn model_name(&self) -> &str {
        "stub-model"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_registry_creation() {
        let registry = ProviderRegistry::new();
        assert!(!registry.has_embedding_provider());
        assert!(!registry.has_reasoning_provider());
    }

    #[test]
    fn test_embedding_provider_config() {
        let config = EmbeddingProviderConfig::default();
        assert_eq!(config.provider, EmbeddingProviderType::Local);
        assert_eq!(config.dimensions, 384);
    }

    #[test]
    fn test_reasoning_provider_config() {
        let config = ReasoningProviderConfig::default();
        assert_eq!(config.provider, ReasoningProviderType::OpenAI);
        assert_eq!(config.max_tokens, 4096);
    }

    #[tokio::test]
    async fn test_stub_embedding_provider() {
        let provider = StubEmbeddingProvider::new(128);
        let request = EmbeddingRequest {
            text: "test".to_string(),
            model: None,
            dimensions: None,
        };
        let response = provider.embed(request).await.unwrap();
        assert_eq!(response.embedding.len(), 128);
    }

    #[tokio::test]
    async fn test_stub_reasoning_provider() {
        let provider = StubReasoningProvider;
        let request = ReasoningRequest {
            task: ReasoningTask::SummarizeMemory,
            context: "Test memory content".to_string(),
            model: None,
            system_prompt: None,
        };
        let response = provider.reason(request).await.unwrap();
        assert!(response.text.contains("Stub reasoning"));
    }
}
