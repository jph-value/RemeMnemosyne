use mnemosyne_core::{MemoryId, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::intent::IntentDetector;
use crate::micro_embed::MicroEmbedder;

/// Configuration for memory prefetcher
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetcherConfig {
    pub max_prefetch_count: usize,
    pub similarity_threshold: f32,
    pub enable_intent_based: bool,
    pub enable_semantic_clustering: bool,
}

impl Default for PrefetcherConfig {
    fn default() -> Self {
        Self {
            max_prefetch_count: 20,
            similarity_threshold: 0.5,
            enable_intent_based: true,
            enable_semantic_clustering: true,
        }
    }
}

/// Memory prefetcher that loads relevant memories before explicit request
pub struct MemoryPrefetcher {
    config: PrefetcherConfig,
    embedder: MicroEmbedder,
    intent_detector: IntentDetector,
    /// Memory embeddings cache (id -> embedding)
    memory_embeddings: HashMap<MemoryId, Vec<f32>>,
    /// Memory clusters for semantic grouping
    memory_clusters: HashMap<String, Vec<MemoryId>>,
    /// Recently prefetched memories
    prefetch_cache: HashSet<MemoryId>,
}

impl MemoryPrefetcher {
    pub fn new(config: PrefetcherConfig) -> Self {
        Self {
            config,
            embedder: MicroEmbedder::fast(),
            intent_detector: IntentDetector::new(),
            memory_embeddings: HashMap::new(),
            memory_clusters: HashMap::new(),
            prefetch_cache: HashSet::new(),
        }
    }

    /// Register a memory with its embedding
    pub fn register_memory(&mut self, id: MemoryId, embedding: Vec<f32>, tags: &[String]) {
        self.memory_embeddings.insert(id, embedding);

        // Add to clusters based on tags
        for tag in tags {
            self.memory_clusters
                .entry(tag.clone())
                .or_insert_with(Vec::new)
                .push(id);
        }
    }

    /// Unregister a memory
    pub fn unregister_memory(&mut self, id: &MemoryId) {
        self.memory_embeddings.remove(id);
        self.prefetch_cache.remove(id);

        // Remove from clusters
        for cluster in self.memory_clusters.values_mut() {
            cluster.retain(|mid| mid != id);
        }
    }

    /// Prefetch memories based on query text
    pub fn prefetch(&self, query: &str, all_memory_ids: &[MemoryId]) -> Vec<MemoryId> {
        let query_embedding = self.embedder.embed(query);
        let intents = self.intent_detector.detect(query);

        let mut candidates: Vec<(MemoryId, f32)> = Vec::new();

        // Semantic similarity prefetch
        for (&memory_id, memory_embedding) in &self.memory_embeddings {
            let similarity = self
                .embedder
                .cosine_similarity(&query_embedding, memory_embedding);
            if similarity >= self.config.similarity_threshold {
                candidates.push((memory_id, similarity));
            }
        }

        // Intent-based prefetch
        if self.config.enable_intent_based {
            let boosted = self.intent_based_prefetch(&intents);
            for (id, score) in boosted {
                // Boost existing or add new
                if let Some(existing) = candidates.iter_mut().find(|(mid, _)| *mid == id) {
                    existing.1 += score * 0.5;
                } else {
                    candidates.push((id, score * 0.5));
                }
            }
        }

        // Tag-based prefetch
        let tag_matches = self.tag_based_prefetch(query);
        for id in tag_matches {
            if !candidates.iter().any(|(mid, _)| *mid == id) {
                candidates.push((id, 0.3));
            }
        }

        // Sort and limit
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.max_prefetch_count);

        candidates.into_iter().map(|(id, _)| id).collect()
    }

    /// Update cluster assignments based on embeddings
    pub fn update_clusters(&mut self, cluster_radius: f32) {
        if !self.config.enable_semantic_clustering {
            return;
        }

        let mut new_clusters: HashMap<usize, Vec<MemoryId>> = HashMap::new();

        for (&id, embedding) in &self.memory_embeddings {
            // Simple clustering: find nearest existing cluster centroid
            let mut best_cluster = None;
            let mut best_similarity = 0.0;

            for (cluster_id, members) in &new_clusters {
                if let Some(&first_id) = members.first() {
                    if let Some(centroid) = self.memory_embeddings.get(&first_id) {
                        let sim = self.embedder.cosine_similarity(embedding, centroid);
                        if sim > best_similarity && sim >= cluster_radius {
                            best_similarity = sim;
                            best_cluster = Some(*cluster_id);
                        }
                    }
                }
            }

            match best_cluster {
                Some(cluster_id) => {
                    new_clusters.get_mut(&cluster_id).unwrap().push(id);
                }
                None => {
                    let new_id = new_clusters.len();
                    new_clusters.insert(new_id, vec![id]);
                }
            }
        }

        // Convert to named clusters
        self.memory_clusters.clear();
        for (cluster_id, members) in new_clusters {
            self.memory_clusters
                .insert(format!("cluster_{}", cluster_id), members);
        }
    }

    /// Get prefetch statistics
    pub fn get_stats(&self) -> PrefetchStats {
        PrefetchStats {
            registered_memories: self.memory_embeddings.len(),
            cluster_count: self.memory_clusters.len(),
            cache_size: self.prefetch_cache.len(),
            avg_cluster_size: if self.memory_clusters.is_empty() {
                0.0
            } else {
                self.memory_clusters
                    .values()
                    .map(|v| v.len())
                    .sum::<usize>() as f32
                    / self.memory_clusters.len() as f32
            },
        }
    }

    // Private helper methods

    fn intent_based_prefetch(&self, intents: &[(String, f32)]) -> Vec<(MemoryId, f32)> {
        let mut results = Vec::new();

        for (intent, score) in intents {
            match intent.as_str() {
                "recall" | "search" => {
                    // Would boost memories tagged as relevant
                    // Simplified for now
                }
                "analyze" => {
                    // Would boost analytical memories
                }
                _ => {}
            }
        }

        results
    }

    fn tag_based_prefetch(&self, query: &str) -> Vec<MemoryId> {
        let query_lower = query.to_lowercase();
        let mut results = HashSet::new();

        for (tag, memory_ids) in &self.memory_clusters {
            if query_lower.contains(tag) || tag.contains(&query_lower) {
                for &id in memory_ids {
                    results.insert(id);
                }
            }
        }

        results.into_iter().collect()
    }
}

/// Prefetch statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchStats {
    pub registered_memories: usize,
    pub cluster_count: usize,
    pub cache_size: usize,
    pub avg_cluster_size: f32,
}

/// Prefetch scheduler for async prefetching
pub struct PrefetchScheduler {
    prefetcher: MemoryPrefetcher,
    /// Pending prefetch requests
    pending: Vec<PrefetchRequest>,
}

impl PrefetchScheduler {
    pub fn new(config: PrefetcherConfig) -> Self {
        Self {
            prefetcher: MemoryPrefetcher::new(config),
            pending: Vec::new(),
        }
    }

    /// Schedule a prefetch request
    pub fn schedule(&mut self, query: String, priority: PrefetchPriority) {
        self.pending.push(PrefetchRequest { query, priority });
        self.pending.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Process pending prefetch requests
    pub fn process(&mut self, all_memory_ids: &[MemoryId]) -> Vec<(String, Vec<MemoryId>)> {
        let mut results = Vec::new();

        for request in self.pending.drain(..) {
            let prefetched = self.prefetcher.prefetch(&request.query, all_memory_ids);
            results.push((request.query, prefetched));
        }

        results
    }

    pub fn get_prefetcher(&self) -> &MemoryPrefetcher {
        &self.prefetcher
    }

    pub fn get_prefetcher_mut(&mut self) -> &mut MemoryPrefetcher {
        &mut self.prefetcher
    }
}

#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    query: String,
    priority: PrefetchPriority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrefetchPriority {
    Low,
    Normal,
    High,
    Critical,
}
