use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for micro-embedder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroEmbedConfig {
    pub dimensions: usize,
    pub model_type: MicroEmbedModel,
    pub normalize: bool,
    pub cache_size: usize,
}

impl Default for MicroEmbedConfig {
    fn default() -> Self {
        Self {
            dimensions: 128,
            model_type: MicroEmbedModel::Hash,
            normalize: true,
            cache_size: 10000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MicroEmbedModel {
    /// Fast hash-based embedding (no model needed)
    Hash,
    /// Simple bag-of-words with TF-IDF weighting
    BagOfWords,
    /// Character n-gram based
    CharNGram,
    /// Custom model (would load from file)
    Custom,
}

/// Micro-embedding generator for fast cognitive processing
///
/// These are lightweight embeddings designed for:
/// - Pre-retrieval prediction
/// - Context pruning
/// - Memory routing
/// - Entity detection
/// - Intent detection
pub struct MicroEmbedder {
    config: MicroEmbedConfig,
    /// Vocabulary for bag-of-words model
    vocab: HashMap<String, usize>,
    /// IDF weights for vocabulary
    idf_weights: Vec<f32>,
    /// Cache for computed embeddings (thread-safe)
    cache: RwLock<HashMap<String, Vec<f32>>>,
    /// Character n-grams for CharNGram model
    ngram_index: HashMap<String, usize>,
}

impl MicroEmbedder {
    pub fn new(config: MicroEmbedConfig) -> Self {
        Self {
            config,
            vocab: HashMap::new(),
            idf_weights: Vec::new(),
            cache: RwLock::new(HashMap::new()),
            ngram_index: HashMap::new(),
        }
    }

    /// Create with default config
    pub fn fast() -> Self {
        Self::new(MicroEmbedConfig::default())
    }

    /// Build vocabulary from training texts
    pub fn build_vocabulary(&mut self, texts: &[String]) {
        // Clear cache since embeddings will change
        self.clear_cache();

        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut doc_counts: HashMap<String, usize> = HashMap::new();
        let n_docs = texts.len();

        for text in texts {
            let words = self.tokenize(text);
            let mut seen = HashMap::new();

            for word in words {
                *word_counts.entry(word.clone()).or_insert(0) += 1;
                if seen.insert(word.clone(), true).is_none() {
                    *doc_counts.entry(word).or_insert(0) += 1;
                }
            }
        }

        // Build vocabulary and IDF weights
        for (word, &count) in &word_counts {
            if count >= 2 {
                let idx = self.vocab.len();
                self.vocab.insert(word.clone(), idx);

                // TF-IDF IDF component: log(N / df)
                let df = doc_counts.get(word).unwrap_or(&1);
                let idf = (n_docs as f32 / *df as f32).ln();
                self.idf_weights.push(idf);
            }
        }

        // Build n-gram index for CharNGram model
        if self.config.model_type == MicroEmbedModel::CharNGram {
            self.build_ngram_index(texts);
        }
    }

    /// Generate micro-embedding for text
    #[inline]
    pub fn embed(&self, text: &str) -> Vec<f32> {
        // Check cache
        {
            let cache = self.cache.read();
            if let Some(cached) = cache.get(text) {
                return cached.clone();
            }
        }

        let embedding = match self.config.model_type {
            MicroEmbedModel::Hash => self.embed_hash(text),
            MicroEmbedModel::BagOfWords => self.embed_bow(text),
            MicroEmbedModel::CharNGram => self.embed_char_ngram(text),
            MicroEmbedModel::Custom => self.embed_hash(text), // Fallback
        };

        // Store in cache
        {
            let mut cache = self.cache.write();
            if cache.len() < self.config.cache_size {
                cache.insert(text.to_string(), embedding.clone());
            }
        }

        embedding
    }

    /// Fast hash-based embedding
    fn embed_hash(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0; self.config.dimensions];
        let words = self.tokenize(text);

        for word in words {
            let hash = self.fnv_hash(&word);
            let idx = (hash as usize) % self.config.dimensions;
            let sign = if hash.is_multiple_of(2) { 1.0 } else { -1.0 };
            embedding[idx] += sign;
        }

        if self.config.normalize {
            self.normalize(&mut embedding);
        }

        embedding
    }

    /// Bag-of-words with TF-IDF
    fn embed_bow(&self, text: &str) -> Vec<f32> {
        // Fall back to hash embedding if vocabulary is empty
        if self.vocab.is_empty() {
            return self.embed_hash(text);
        }

        let mut embedding = vec![0.0; self.config.dimensions];
        let words = self.tokenize(text);
        let mut term_counts: HashMap<String, usize> = HashMap::new();

        for word in &words {
            *term_counts.entry(word.clone()).or_insert(0) += 1;
        }

        for (word, &count) in &term_counts {
            if let Some(&idx) = self.vocab.get(word) {
                if idx < self.config.dimensions {
                    let tf = (count as f32).sqrt(); // Sublinear TF
                    let idf = self.idf_weights.get(idx).unwrap_or(&1.0);
                    embedding[idx] = tf * idf;
                }
            }
        }

        if self.config.normalize {
            self.normalize(&mut embedding);
        }

        embedding
    }

    /// Character n-gram embedding
    fn embed_char_ngram(&self, text: &str) -> Vec<f32> {
        // Fall back to hash embedding if ngram index is empty
        if self.ngram_index.is_empty() {
            return self.embed_hash(text);
        }

        let mut embedding = vec![0.0; self.config.dimensions];
        let text_lower = text.to_lowercase();

        // Extract character 3-grams
        for i in 0..text_lower.len().saturating_sub(2) {
            let ngram = &text_lower[i..i + 3];
            if let Some(&idx) = self.ngram_index.get(ngram) {
                if idx < self.config.dimensions {
                    embedding[idx] += 1.0;
                }
            } else {
                // Use hash for unknown ngrams
                let hash = self.fnv_hash(ngram);
                let idx = hash as usize % self.config.dimensions;
                embedding[idx] += 0.5; // Lower weight for unknown ngrams
            }
        }

        if self.config.normalize {
            self.normalize(&mut embedding);
        }

        embedding
    }

    /// Compute similarity between two embeddings
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Batch embed multiple texts
    pub fn embed_batch(&self, texts: &[String]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    // Private helper methods

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric() || *c == '-')
                    .collect::<String>()
            })
            .filter(|w| w.len() > 2)
            .collect()
    }

    fn fnv_hash(&self, s: &str) -> u64 {
        let mut hash: u64 = 14695981039346656037;
        for byte in s.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        hash
    }

    fn normalize(&self, vector: &mut [f32]) {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for val in vector.iter_mut() {
                *val /= norm;
            }
        }
    }

    fn build_ngram_index(&mut self, texts: &[String]) {
        let mut ngrams: HashMap<String, usize> = HashMap::new();

        for text in texts {
            let text_lower = text.to_lowercase();
            for i in 0..text_lower.len().saturating_sub(2) {
                let ngram = text_lower[i..i + 3].to_string();
                *ngrams.entry(ngram).or_insert(0) += 1;
            }
        }

        // Keep most common ngrams
        let mut sorted: Vec<_> = ngrams.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        for (idx, (ngram, _)) in sorted.iter().take(self.config.dimensions).enumerate() {
            self.ngram_index.insert(ngram.clone(), idx);
        }
    }

    /// Simple named entity recognition using capitalized-word heuristic.
    ///
    /// Extracts entities following the same pattern as EpisodeSummarizer:
    /// any word > 2 chars starting with uppercase is classified as a
    /// Concept entity with relevance 0.5.
    pub fn extract_entities_ner(&self, text: &str) -> Vec<rememnemosyne_core::EntityRef> {
        let mut entities = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for word in text.split_whitespace() {
            let clean: String = word
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .collect();
            if clean.len() > 2
                && clean
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
            {
                if seen.insert(clean.clone()) {
                    entities.push(rememnemosyne_core::EntityRef {
                        id: uuid::Uuid::new_v4(),
                        name: clean,
                        entity_type: rememnemosyne_core::EntityType::Concept,
                        relevance: 0.5,
                    });
                }
            }
        }

        entities
    }
}

/// Micro-embedding cache for performance
pub struct EmbeddingCache {
    cache: HashMap<String, Vec<f32>>,
    max_size: usize,
}

impl EmbeddingCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Vec<f32>> {
        self.cache.get(key)
    }

    pub fn insert(&mut self, key: String, embedding: Vec<f32>) {
        if self.cache.len() >= self.max_size {
            // Simple eviction: remove first entry
            if let Some(key) = self.cache.keys().next().cloned() {
                self.cache.remove(&key);
            }
        }
        self.cache.insert(key, embedding);
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
}
