use rememnemosyne_core::Result;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::turboquant::{QuantizedCode, TurboQuantizer};

/// HNSW (Hierarchical Navigable Small World) index for fast approximate nearest neighbor search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWIndex {
    pub dimension: usize,
    pub m: usize,               // Max connections per layer
    pub ef_construction: usize, // Size of dynamic candidate list during construction
    pub ef_search: usize,       // Size of dynamic candidate list during search
    pub max_elements: usize,
    pub data: Vec<Vec<f32>>, // Stored vectors
    pub quantized_codes: Vec<Option<QuantizedCode>>,
    pub layers: Vec<Vec<usize>>,           // Nodes at each layer
    pub connections: Vec<Vec<Vec<usize>>>, // Connections[i][j] = neighbors of node j at layer i
    pub entry_point: Option<usize>,
    pub level_generator: u64,
}

impl HNSWIndex {
    pub fn new(dimension: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            dimension,
            m,
            ef_construction,
            ef_search: ef_construction,
            max_elements: 0,
            data: Vec::new(),
            quantized_codes: Vec::new(),
            layers: Vec::new(),
            connections: Vec::new(),
            entry_point: None,
            level_generator: 1,
        }
    }

    /// Add a vector to the index
    pub fn add(
        &mut self,
        vector: Vec<f32>,
        quantized_code: Option<QuantizedCode>,
    ) -> Result<usize> {
        if vector.len() != self.dimension {
            return Err(rememnemosyne_core::MemoryError::Index(format!(
                "Vector dimension {} != index dimension {}",
                vector.len(),
                self.dimension
            )));
        }

        let node_id = self.data.len();
        let level = self.random_level();

        // Ensure we have enough layers
        while self.layers.len() <= level {
            self.layers.push(Vec::new());
            self.connections.push(Vec::new());
        }

        // Add node to appropriate layers
        for l in 0..=level {
            self.layers[l].push(node_id);
            self.connections[l].push(Vec::new());
        }

        self.data.push(vector.clone());
        self.quantized_codes.push(quantized_code);

        // Connect to existing nodes
        if let Some(entry) = self.entry_point {
            let max_level = self.layers.len() - 1;

            // For levels above the new node's level, just update entry point if needed
            if level < max_level {
                let mut cur = entry;
                for l in ((level + 1)..=max_level).rev() {
                    cur = self.search_layer(cur, &vector, 1, l)[0];
                }
                self.connect_node(node_id, cur, level, max_level);
            } else {
                self.connect_node(node_id, entry, level, level);
            }
        } else {
            self.entry_point = Some(node_id);
        }

        self.max_elements = self.data.len();
        Ok(node_id)
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.data.is_empty() {
            return Vec::new();
        }

        let entry = match self.entry_point {
            Some(e) => e,
            None => return Vec::new(),
        };

        let max_level = self.layers.len() - 1;
        let mut cur = entry;

        // Greedy search at higher layers
        for l in (1..=max_level).rev() {
            let result = self.search_layer(cur, query, 1, l);
            cur = result[0];
        }

        // Search at level 0 with ef_search
        let candidates = self.search_layer(cur, query, self.ef_search.max(k), 0);

        // Return top k
        candidates
            .into_iter()
            .take(k)
            .map(|id| (id, self.cosine_similarity(query, &self.data[id])))
            .collect()
    }

    /// Search using quantized codes for faster estimation
    pub fn search_quantized(
        &self,
        query: &[f32],
        quantizer: &TurboQuantizer,
        k: usize,
    ) -> Vec<(usize, f32)> {
        if self.data.is_empty() {
            return Vec::new();
        }

        let mut candidates: Vec<(usize, f32)> = self
            .quantized_codes
            .iter()
            .enumerate()
            .filter_map(|(i, code)| {
                code.as_ref()
                    .and_then(|c| quantizer.inner_product_estimate(c, query).ok())
                    .map(|score| (i, score))
            })
            .collect();

        // Sort by score descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        candidates.into_iter().take(k).collect()
    }

    /// Remove a vector from the index
    pub fn remove(&mut self, id: usize) -> Result<()> {
        if id >= self.data.len() {
            return Err(rememnemosyne_core::MemoryError::Index(
                "ID out of bounds".into(),
            ));
        }

        // Mark as deleted (set to empty vector)
        if id < self.data.len() {
            self.data[id] = vec![];
            self.quantized_codes[id] = None;
        }

        Ok(())
    }

    /// Get number of elements
    pub fn len(&self) -> usize {
        self.data.iter().filter(|v| !v.is_empty()).count()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // Private helper methods

    fn random_level(&mut self) -> usize {
        let mut level = 0;
        let mut r = self.level_generator;
        while r.is_multiple_of(2) && level < 16 {
            level += 1;
            r >>= 1;
        }
        self.level_generator = self
            .level_generator
            .wrapping_mul(1664525)
            .wrapping_add(1013904223);
        level
    }

    #[inline]
    fn search_layer(&self, entry_id: usize, query: &[f32], ef: usize, layer: usize) -> Vec<usize> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let dist = self.cosine_similarity(query, &self.data[entry_id]);
        visited.insert(entry_id);
        candidates.push(Candidate {
            id: entry_id,
            distance: dist,
        });
        results.push(Candidate {
            id: entry_id,
            distance: dist,
        });

        while let Some(candidate) = candidates.pop() {
            let worst_in_results = results.peek().map(|c| c.distance).unwrap_or(f32::MIN);

            if candidate.distance < worst_in_results && results.len() >= ef {
                break;
            }

            if layer < self.connections.len() && candidate.id < self.connections[layer].len() {
                for &neighbor in &self.connections[layer][candidate.id] {
                    if visited.contains(&neighbor) || self.data[neighbor].is_empty() {
                        continue;
                    }

                    visited.insert(neighbor);
                    let dist = self.cosine_similarity(query, &self.data[neighbor]);

                    let worst = results.peek().map(|c| c.distance).unwrap_or(f32::MIN);
                    if dist > worst || results.len() < ef {
                        candidates.push(Candidate {
                            id: neighbor,
                            distance: dist,
                        });
                        results.push(Candidate {
                            id: neighbor,
                            distance: dist,
                        });

                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|c| c.id)
            .collect()
    }

    fn connect_node(&mut self, node_id: usize, target: usize, level: usize, max_level: usize) {
        for l in 0..=level.min(max_level) {
            if l < self.connections.len() && target < self.connections[l].len() {
                self.connections[l][target].push(node_id);

                // Limit connections
                if self.connections[l][target].len() > self.m * 2 {
                    self.prune_connections(l, target);
                }

                if l < self.connections.len() && node_id < self.connections[l].len() {
                    self.connections[l][node_id].push(target);
                }
            }
        }
    }

    fn prune_connections(&mut self, layer: usize, node_id: usize) {
        if layer >= self.connections.len() || node_id >= self.connections[layer].len() {
            return;
        }

        let neighbors = &self.connections[layer][node_id];
        if neighbors.len() <= self.m {
            return;
        }

        let node_vector = &self.data[node_id];
        let mut scored: Vec<(usize, f32)> = neighbors
            .iter()
            .map(|&n| (n, self.cosine_similarity(node_vector, &self.data[n])))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(self.m);

        self.connections[layer][node_id] = scored.into_iter().map(|(id, _)| id).collect();
    }

    #[inline]
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
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

    /// Serialize HNSW index to bytes for persistence
    pub fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| rememnemosyne_core::MemoryError::Serialization(e.to_string()))
    }

    /// Deserialize HNSW index from bytes
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data)
            .map_err(|e| rememnemosyne_core::MemoryError::Serialization(e.to_string()))
    }

    /// Save HNSW index to a file
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let data = self
            .serialize()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        std::fs::write(path, data)
    }

    /// Load HNSW index from a file
    pub fn load_from_file(path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read(path).map_err(rememnemosyne_core::MemoryError::Io)?;
        Self::deserialize(&data)
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    id: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Simple flat index for small datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatIndex {
    pub dimension: usize,
    pub vectors: Vec<Vec<f32>>,
    pub ids: Vec<rememnemosyne_core::MemoryId>,
}

impl FlatIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vectors: Vec::new(),
            ids: Vec::new(),
        }
    }

    pub fn add(&mut self, id: rememnemosyne_core::MemoryId, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(rememnemosyne_core::MemoryError::Index(format!(
                "Vector dimension {} != index dimension {}",
                vector.len(),
                self.dimension
            )));
        }
        self.vectors.push(vector);
        self.ids.push(id);
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut results: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, self.cosine_similarity(query, v)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.into_iter().take(k).collect()
    }

    pub fn remove(&mut self, id: &rememnemosyne_core::MemoryId) -> Result<()> {
        if let Some(pos) = self.ids.iter().position(|i| i == id) {
            self.vectors.remove(pos);
            self.ids.remove(pos);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    #[inline]
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}
