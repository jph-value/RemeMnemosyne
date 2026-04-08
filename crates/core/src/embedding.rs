/// Embedding Provider Abstraction
///
/// This module defines the trait and types for pluggable embedding providers.
/// Allows integrators to swap embedding engines without modifying the core.
///
/// Supported providers:
/// - Local (Candle/fastembed) - no API key, runs locally
/// - OpenAI - text-embedding-ada-002, text-embedding-3-small, etc.
/// - Voyage - Voyage AI embeddings
/// - Cohere - Cohere embed
/// - Ollama - Local Ollama models
/// - Custom - User's own API endpoint
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

// ============================================================================
// Provider Configuration
// ============================================================================

/// Embedding provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingProviderConfig {
    /// Provider type
    pub provider: EmbeddingProviderType,
    /// Model name/identifier
    pub model: String,
    /// API key (for cloud providers)
    pub api_key: Option<String>,
    /// Base URL (for custom/local providers)
    pub base_url: Option<String>,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Request timeout (seconds)
    pub timeout_secs: u64,
    /// Retry count on failure
    pub max_retries: u32,
}

impl Default for EmbeddingProviderConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProviderType::Local,
            model: "all-MiniLM-L6-v2".to_string(),
            api_key: None,
            base_url: None,
            dimensions: 384,
            timeout_secs: 30,
            max_retries: 3,
        }
    }
}

// ============================================================================
// Provider Types
// ============================================================================

/// Supported embedding providers
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingProviderType {
    /// Local embedding (Candle/fastembed)
    Local,
    /// OpenAI embeddings (text-embedding-ada-002, etc.)
    OpenAI,
    /// Voyage AI embeddings
    Voyage,
    /// Cohere embeddings
    Cohere,
    /// Ollama local embeddings
    Ollama,
    /// Custom API endpoint
    Custom,
}

impl std::fmt::Display for EmbeddingProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingProviderType::Local => write!(f, "local"),
            EmbeddingProviderType::OpenAI => write!(f, "openai"),
            EmbeddingProviderType::Voyage => write!(f, "voyage"),
            EmbeddingProviderType::Cohere => write!(f, "cohere"),
            EmbeddingProviderType::Ollama => write!(f, "ollama"),
            EmbeddingProviderType::Custom => write!(f, "custom"),
        }
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Embedding request
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    /// Text to embed
    pub text: String,
    /// Optional model override
    pub model: Option<String>,
    /// Optional dimensions override
    pub dimensions: Option<usize>,
}

impl EmbeddingRequest {
    /// Create a new request
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            model: None,
            dimensions: None,
        }
    }

    /// Override model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Override dimensions
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
}

/// Embedding response
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Model used
    pub model: String,
    /// Token usage (if available)
    pub token_count: Option<usize>,
}

impl EmbeddingResponse {
    /// Create a new response
    pub fn new(embedding: Vec<f32>, model: impl Into<String>) -> Self {
        Self {
            embedding,
            model: model.into(),
            token_count: None,
        }
    }
}

// ============================================================================
// Provider Trait
// ============================================================================

/// Embedding provider trait - the core abstraction for pluggable embeddings
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding for a single text
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse>;

    /// Generate embeddings for multiple texts
    async fn embed_batch(&self, requests: Vec<EmbeddingRequest>) -> Result<Vec<EmbeddingResponse>> {
        let mut responses = Vec::with_capacity(requests.len());
        for req in requests {
            responses.push(self.embed(req).await?);
        }
        Ok(responses)
    }

    /// Get provider type
    fn provider_type(&self) -> EmbeddingProviderType;

    /// Get model name
    fn model_name(&self) -> &str;

    /// Get embedding dimensions
    fn dimensions(&self) -> usize;
}

// ============================================================================
// Stub Implementation (Hash-based fallback)
// ============================================================================

/// Hash-based embedding provider (fallback when no ML provider is configured)
/// This provides deterministic embeddings without any external dependencies.
pub struct HashEmbedder {
    dimensions: usize,
}

impl HashEmbedder {
    /// Create with specific dimensions
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    /// Default (128 dimensions)
    pub fn default_embedder() -> Self {
        Self::new(128)
    }

    /// Generate hash-based embedding
    pub fn embed_sync(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0; self.dimensions];
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();

        for word in words {
            let hash = fnv_hash(word);
            let idx = (hash as usize) % self.dimensions;
            let sign = if hash.is_multiple_of(2) { 1.0 } else { -1.0 };
            embedding[idx] += sign;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for val in embedding.iter_mut() {
                *val /= norm;
            }
        }

        embedding
    }
}

#[async_trait]
impl EmbeddingProvider for HashEmbedder {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let embedding = self.embed_sync(&request.text);
        Ok(EmbeddingResponse::new(embedding, "hash"))
    }

    fn provider_type(&self) -> EmbeddingProviderType {
        EmbeddingProviderType::Local
    }

    fn model_name(&self) -> &str {
        "fnv-hash"
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// FNV-1a hash
fn fnv_hash(s: &str) -> u64 {
    let mut hash: u64 = 14695981039346656037;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_embedder() {
        let embedder = HashEmbedder::new(128);
        let emb = embedder.embed_sync("hello world");
        assert_eq!(emb.len(), 128);
    }

    #[test]
    fn test_hash_embedder_deterministic() {
        let embedder = HashEmbedder::new(128);
        let emb1 = embedder.embed_sync("test text");
        let emb2 = embedder.embed_sync("test text");
        assert_eq!(emb1, emb2);
    }

    #[test]
    fn test_embedding_request_builder() {
        let req = EmbeddingRequest::new("hello")
            .with_model("custom-model")
            .with_dimensions(256);
        assert_eq!(req.model, Some("custom-model".to_string()));
        assert_eq!(req.dimensions, Some(256));
    }

    #[tokio::test]
    async fn test_hash_embedder_async() {
        let embedder = HashEmbedder::default_embedder();
        let response = embedder.embed(EmbeddingRequest::new("test")).await.unwrap();
        assert_eq!(response.embedding.len(), 128);
        assert_eq!(response.model, "hash");
    }

    #[tokio::test]
    async fn test_hash_embedder_batch() {
        let embedder = HashEmbedder::new(64);
        let requests = vec![
            EmbeddingRequest::new("text one"),
            EmbeddingRequest::new("text two"),
        ];
        let responses = embedder.embed_batch(requests).await.unwrap();
        assert_eq!(responses.len(), 2);
    }
}
