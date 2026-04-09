use rememnemosyne_core::{MemoryError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[cfg(feature = "candle-embeddings")]
use candle_core::{Device, Tensor, DType};
#[cfg(feature = "candle-embeddings")]
use parking_lot::RwLock;
#[cfg(feature = "candle-embeddings")]
use std::collections::HashMap;
#[cfg(feature = "candle-embeddings")]
use std::sync::Arc;

/// Configuration for Candle embedder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleEmbedConfig {
    /// Model to use (e.g., "all-MiniLM-L6-v2")
    pub model_name: String,
    /// Output dimensions (typically 384 for MiniLM)
    pub dimensions: usize,
    /// Whether to normalize embeddings
    pub normalize: bool,
    /// Cache size for computed embeddings
    pub cache_size: usize,
    /// Optional local path to model files (skip download if present)
    pub local_model_path: Option<PathBuf>,
}

impl Default for CandleEmbedConfig {
    fn default() -> Self {
        Self {
            model_name: "all-MiniLM-L6-v2".to_string(),
            dimensions: 384,
            normalize: true,
            cache_size: 10000,
            local_model_path: None,
        }
    }
}

/// Real ML embedder using Candle framework with ONNX inference
#[cfg(feature = "candle-embeddings")]
pub struct CandleEmbedder {
    config: CandleEmbedConfig,
    /// Tokenizer
    tokenizer: Arc<RwLock<Option<tokenizers::Tokenizer>>>,
    /// ONNX model data (loaded once, reused for inference)
    model: Arc<RwLock<Option<candle_onnx::ModelProto>>>,
    /// Device for tensor operations
    device: Device,
    /// Cache for computed embeddings
    cache: RwLock<HashMap<String, Vec<f32>>>,
    /// Whether the model is loaded
    model_loaded: std::sync::atomic::AtomicBool,
}

#[cfg(feature = "candle-embeddings")]
impl CandleEmbedder {
    pub fn new(config: CandleEmbedConfig) -> Self {
        Self {
            config,
            tokenizer: Arc::new(RwLock::new(None)),
            model: Arc::new(RwLock::new(None)),
            device: Device::Cpu,
            cache: RwLock::new(HashMap::new()),
            model_loaded: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn default_embedder() -> Self {
        Self::new(CandleEmbedConfig::default())
    }

    /// Load the model and tokenizer from HuggingFace or local path
    pub async fn load_model(&self) -> Result<()> {
        use candle_onnx::read_file;
        use hf_hub::{api::tokio::Api, Repo, RepoType};
        use tokenizers::Tokenizer;

        if self.model_loaded.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }

        let (model_path, tokenizer_path) = if let Some(local_path) = &self.config.local_model_path {
            (local_path.join("onnx/model.onnx"), local_path.join("tokenizer.json"))
        } else {
            let api = Api::new().map_err(|e| {
                MemoryError::Cognitive(format!("Failed to initialize HuggingFace API: {}", e))
            })?;
            let repo = Repo::with_revision(
                format!("sentence-transformers/{}", self.config.model_name),
                RepoType::Model,
                "main".to_string(),
            );
            let api_repo = api.repo(repo);

            let model_file = api_repo.get("onnx/model.onnx").await.map_err(|e| {
                MemoryError::Cognitive(format!("Failed to download ONNX model: {}", e))
            })?;
            let tokenizer_file = api_repo.get("tokenizer.json").await.map_err(|e| {
                MemoryError::Cognitive(format!("Failed to download tokenizer: {}", e))
            })?;
            (model_file, tokenizer_file)
        };

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| MemoryError::Cognitive(format!("Failed to load tokenizer: {}", e)))?;
        *self.tokenizer.write() = Some(tokenizer);

        let model_proto = read_file(&model_path)
            .map_err(|e| MemoryError::Cognitive(format!("Failed to load ONNX model: {}", e)))?;
        *self.model.write() = Some(model_proto);

        self.model_loaded.store(true, std::sync::atomic::Ordering::Relaxed);
        tracing::info!(
            model = %self.config.model_name,
            dimensions = self.config.dimensions,
            "Candle embedder model loaded (ONNX inference)"
        );
        Ok(())
    }

    /// Generate embedding for text using real ONNX inference
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        {
            let cache = self.cache.read();
            if let Some(cached) = cache.get(text) {
                return Ok(cached.clone());
            }
        }

        if !self.model_loaded.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(MemoryError::Cognitive(
                "Model not loaded. Call load_model() first.".to_string(),
            ));
        }

        let tokenizer = self.tokenizer.read();
        let tokenizer = tokenizer
            .as_ref()
            .ok_or_else(|| MemoryError::Cognitive("Tokenizer not initialized".to_string()))?;

        let model = self.model.read();
        let model = model
            .as_ref()
            .ok_or_else(|| MemoryError::Cognitive("ONNX model not loaded".to_string()))?;

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| MemoryError::Cognitive(format!("Tokenization failed: {}", e)))?;

        let embedding = self.run_inference(model, &encoding)?;

        {
            let mut cache = self.cache.write();
            if cache.len() < self.config.cache_size {
                cache.insert(text.to_string(), embedding.clone());
            }
        }
        Ok(embedding)
    }

    /// Run real ONNX inference with MEAN pooling and L2 normalization
    fn run_inference(
        &self,
        model: &candle_onnx::ModelProto,
        encoding: &tokenizers::Encoding,
    ) -> Result<Vec<f32>> {
        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();

        let seq_len = token_ids.len();
        let input_ids = Tensor::from_vec(token_ids, (1usize, seq_len), &self.device)
            .map_err(|e| MemoryError::Cognitive(format!("Failed to create input_ids tensor: {}", e)))?;
        let attention_mask = Tensor::from_vec(attention_mask, (1usize, seq_len), &self.device)
            .map_err(|e| MemoryError::Cognitive(format!("Failed to create attention_mask tensor: {}", e)))?;
        let token_type_ids = Tensor::from_vec(token_type_ids, (1usize, seq_len), &self.device)
            .map_err(|e| MemoryError::Cognitive(format!("Failed to create token_type_ids tensor: {}", e)))?;

        let mut inputs = HashMap::new();
        inputs.insert("input_ids".to_string(), input_ids);
        inputs.insert("attention_mask".to_string(), attention_mask);
        inputs.insert("token_type_ids".to_string(), token_type_ids);

        let outputs = candle_onnx::simple_eval(model, inputs)
            .map_err(|e| MemoryError::Cognitive(format!("ONNX inference failed: {}", e)))?;

        let hidden_states = outputs.get("last_hidden_state")
            .ok_or_else(|| MemoryError::Cognitive("ONNX model missing 'last_hidden_state' output".to_string()))?;

        let hidden_states = hidden_states.to_dtype(DType::F32)
            .map_err(|e| MemoryError::Cognitive(format!("Failed to convert hidden states to F32: {}", e)))?;

        let pooled = mean_pool(&hidden_states, &attention_mask)
            .map_err(|e| MemoryError::Cognitive(format!("MEAN pooling failed: {}", e)))?;

        let embedding = if self.config.normalize {
            l2_normalize(&pooled)
                .map_err(|e| MemoryError::Cognitive(format!("L2 normalization failed: {}", e)))?
        } else {
            pooled
        };

        let embedding_vec = embedding.to_vec1::<f32>()
            .map_err(|e| MemoryError::Cognitive(format!("Failed to extract embedding vector: {}", e)))?;

        Ok(embedding_vec)
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    pub fn is_loaded(&self) -> bool {
        self.model_loaded.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn config(&self) -> &CandleEmbedConfig {
        &self.config
    }
}

/// MEAN pooling: sum(hidden * mask) / sum(mask)
/// hidden_states: [1, seq_len, hidden_dim]
/// attention_mask: [1, seq_len]
#[cfg(feature = "candle-embeddings")]
fn mean_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor, MemoryError> {
    let mask = attention_mask
        .unsqueeze(2)
        .and_then(|m| m.to_dtype(DType::F32))
        .map_err(|e| MemoryError::Cognitive(format!("Mask expansion failed: {}", e)))?;

    let masked = hidden_states
        .mul(&mask)
        .map_err(|e| MemoryError::Cognitive(format!("Mask application failed: {}", e)))?;

    let sum = masked
        .sum(1)
        .map_err(|e| MemoryError::Cognitive(format!("Sum pooling failed: {}", e)))?;

    let count = mask
        .sum(1)
        .and_then(|c| c.clamp(1e-9, f64::MAX))
        .map_err(|e| MemoryError::Cognitive(format!("Count computation failed: {}", e)))?;

    sum.broadcast_div(&count)
        .map_err(|e| MemoryError::Cognitive(format!("Division failed: {}", e)))
}

/// L2 normalize: x / ||x||
#[cfg(feature = "candle-embeddings")]
fn l2_normalize(embedding: &Tensor) -> Result<Tensor, MemoryError> {
    let norm = embedding
        .sqr()
        .and_then(|s| s.sum_keepdim(1))
        .and_then(|s| s.sqrt())
        .map_err(|e| MemoryError::Cognitive(format!("Norm computation failed: {}", e)))?;

    embedding
        .broadcast_div(&norm)
        .map_err(|e| MemoryError::Cognitive(format!("Normalization failed: {}", e)))
}

/// Implement EmbeddingProvider trait for CandleEmbedder
#[cfg(feature = "candle-embeddings")]
#[async_trait]
impl EmbeddingProvider for CandleEmbedder {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let embedding = self.embed(&request.text)?;
        Ok(EmbeddingResponse {
            embedding,
            model: self.config.model_name.clone(),
            token_count: None,
        })
    }

    async fn embed_batch(&self, requests: Vec<EmbeddingRequest>) -> Result<Vec<EmbeddingResponse>> {
        let mut responses = Vec::with_capacity(requests.len());
        for req in requests {
            let embedding = self.embed(&req.text)?;
            responses.push(EmbeddingResponse {
                embedding,
                model: self.config.model_name.clone(),
                token_count: None,
            });
        }
        Ok(responses)
    }

    fn provider_type(&self) -> EmbeddingProviderType {
        EmbeddingProviderType::Local
    }

    fn model_name(&self) -> &str {
        &self.config.model_name
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }
}

/// Stub when feature not enabled
#[cfg(not(feature = "candle-embeddings"))]
pub struct CandleEmbedder {
    config: CandleEmbedConfig,
}

#[cfg(not(feature = "candle-embeddings"))]
impl CandleEmbedder {
    pub fn new(config: CandleEmbedConfig) -> Self {
        Self { config }
    }

    pub fn default_embedder() -> Self {
        Self::new(CandleEmbedConfig::default())
    }

    pub async fn load_model(&self) -> Result<()> {
        Err(MemoryError::Cognitive(
            "Candle embeddings feature not enabled. Enable with --features candle-embeddings"
                .to_string(),
        ))
    }

    pub fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Err(MemoryError::Cognitive(
            "Candle embeddings feature not enabled. Enable with --features candle-embeddings"
                .to_string(),
        ))
    }

    pub fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(MemoryError::Cognitive(
            "Candle embeddings feature not enabled. Enable with --features candle-embeddings"
                .to_string(),
        ))
    }

    pub fn clear_cache(&self) {}
    pub fn is_loaded(&self) -> bool { false }
    pub fn config(&self) -> &CandleEmbedConfig { &self.config }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_embedder_config() {
        let config = CandleEmbedConfig::default();
        assert_eq!(config.model_name, "all-MiniLM-L6-v2");
        assert_eq!(config.dimensions, 384);
        assert!(config.normalize);
    }

    #[test]
    fn test_candle_embedder_creation() {
        let embedder = CandleEmbedder::default_embedder();
        assert!(!embedder.is_loaded());
    }

    #[cfg(not(feature = "candle-embeddings"))]
    #[tokio::test]
    async fn test_candle_embedder_not_enabled() {
        let embedder = CandleEmbedder::default_embedder();
        let result = embedder.embed("test");
        assert!(result.is_err());
    }
}
