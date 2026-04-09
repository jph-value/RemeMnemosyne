use dashmap::DashMap;
use petgraph::graph::{Graph, NodeIndex};
use rememnemosyne_core::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

use crate::entity::{EntityCluster, GraphEntity};
use crate::relationship::{GraphRelationship, RelationshipPath};

/// Configuration for graph memory store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMemoryConfig {
    pub max_depth: usize,
    pub min_relationship_strength: f32,
    pub auto_cluster: bool,
    pub cluster_threshold: f32,
    pub max_entities_per_query: usize,
}

impl Default for GraphMemoryConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            min_relationship_strength: 0.1,
            auto_cluster: true,
            cluster_threshold: 0.7,
            max_entities_per_query: 50,
        }
    }
}

/// Graph memory store using petgraph
pub struct GraphMemoryStore {
    config: GraphMemoryConfig,
    /// Entity storage
    entities: Arc<DashMap<EntityId, GraphEntity>>,
    /// Relationship storage
    relationships: Arc<DashMap<Uuid, GraphRelationship>>,
    /// The actual graph structure
    graph: Arc<parking_lot::RwLock<Graph<EntityId, Uuid>>>,
    /// Node index mapping
    node_indices: Arc<DashMap<EntityId, NodeIndex>>,
    /// Entity name index for fast lookup
    name_index: Arc<DashMap<String, HashSet<EntityId>>>,
    /// Entity clusters
    clusters: Arc<DashMap<uuid::Uuid, EntityCluster>>,
    /// Composite key index for O(1) relationship lookups
    relationship_index: Arc<DashMap<u64, Uuid>>,
}

fn relationship_key(source: &EntityId, target: &EntityId, rel_type: &RelationshipType) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for byte in source.as_bytes() {
        h ^= *byte as u64;
        h = h.wrapping_mul(1099511628211);
    }
    for byte in target.as_bytes() {
        h ^= *byte as u64;
        h = h.wrapping_mul(1099511628211);
    }
    let type_str = format!("{:?}", rel_type);
    for byte in type_str.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

impl GraphMemoryStore {
    pub fn new(config: GraphMemoryConfig) -> Self {
        Self {
            config,
            entities: Arc::new(DashMap::new()),
            relationships: Arc::new(DashMap::new()),
            graph: Arc::new(parking_lot::RwLock::new(Graph::new())),
            node_indices: Arc::new(DashMap::new()),
            name_index: Arc::new(DashMap::new()),
            clusters: Arc::new(DashMap::new()),
            relationship_index: Arc::new(DashMap::new()),
        }
    }

    /// Add an entity to the graph
    pub async fn add_entity(&self, entity: GraphEntity) -> Result<EntityId> {
        let id = entity.id;

        // Index by name
        self.index_entity_name(&entity);

        // Add to graph
        {
            let mut graph = self.graph.write();
            let node = graph.add_node(id);
            self.node_indices.insert(id, node);
        }

        self.entities.insert(id, entity);
        Ok(id)
    }

    /// Get an entity by ID
    pub async fn get_entity(&self, id: &EntityId) -> Option<GraphEntity> {
        self.entities.get(id).map(|e| e.clone())
    }

    /// Get entity by name (fuzzy match)
    pub async fn get_entity_by_name(&self, name: &str) -> Option<GraphEntity> {
        let name_lower = name.to_lowercase();

        // Exact match first
        if let Some(ids) = self.name_index.get(&name_lower) {
            if let Some(id) = ids.iter().next() {
                return self.entities.get(id).map(|e| e.clone());
            }
        }

        // Fuzzy match
        for entry in self.entities.iter() {
            if entry.matches_name(name) {
                return Some(entry.clone());
            }
        }

        None
    }

    /// Add a relationship between entities
    pub async fn add_relationship(
        &self,
        source_id: EntityId,
        target_id: EntityId,
        relationship_type: RelationshipType,
        strength: f32,
    ) -> Result<Uuid> {
        let relationship =
            GraphRelationship::new(source_id, target_id, relationship_type, strength);

        // Check if relationship already exists using O(1) index
        let existing_id = self.find_existing_relationship_index(
            &source_id,
            &target_id,
            &relationship.relationship_type,
        );

        if let Some(ex_id) = existing_id {
            // Strengthen existing relationship
            if let Some(mut rel) = self.relationships.get_mut(&ex_id) {
                rel.strengthen(strength * 0.1);
                return Ok(ex_id);
            }
        }

        let rel_id = relationship.id;

        // Add to relationship index (O(1) lookup)
        let key = relationship_key(&source_id, &target_id, &relationship.relationship_type);
        self.relationship_index.insert(key, rel_id);

        // Add to graph
        {
            let source_node = self.node_indices.get(&source_id).map(|n| *n);
            let target_node = self.node_indices.get(&target_id).map(|n| *n);

            if let (Some(s_node), Some(t_node)) = (source_node, target_node) {
                let mut graph = self.graph.write();
                graph.add_edge(s_node, t_node, rel_id);
            }
        }

        self.relationships.insert(rel_id, relationship);

        // Update centrality scores
        self.update_centrality(&source_id);
        self.update_centrality(&target_id);

        Ok(rel_id)
    }

    /// Find entities related to a given entity
    pub async fn find_related(
        &self,
        entity_id: &EntityId,
        max_depth: usize,
    ) -> Result<Vec<(GraphEntity, RelationshipType, f32)>> {
        let start_node = self
            .node_indices
            .get(entity_id)
            .ok_or_else(|| MemoryError::NotFound("Entity not found".into()))?;

        let graph = self.graph.read();
        let mut visited = HashSet::new();
        let mut results = Vec::new();
        let mut queue = vec![(*start_node, 0)];

        visited.insert(*start_node);

        while let Some((node, depth)) = queue.pop() {
            if depth > max_depth {
                continue;
            }

            // Get neighbors
            for neighbor in graph.neighbors(node) {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                // Get the entity ID from the neighbor node
                if let Some(neighbor_id) = graph.node_weight(neighbor) {
                    if let Some(entity) = self.entities.get(neighbor_id) {
                        // Get relationship info
                        if let Some(edge) = graph.find_edge(node, neighbor) {
                            if let Some(rel_id) = graph.edge_weight(edge) {
                                if let Some(rel) = self.relationships.get(rel_id) {
                                    if rel.strength >= self.config.min_relationship_strength {
                                        results.push((
                                            entity.clone(),
                                            rel.relationship_type.clone(),
                                            rel.strength,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }

                if depth < max_depth {
                    queue.push((neighbor, depth + 1));
                }
            }
        }

        Ok(results)
    }

    /// Find shortest path between two entities using BFS
    pub async fn find_path(
        &self,
        source_id: &EntityId,
        target_id: &EntityId,
    ) -> Option<RelationshipPath> {
        let source_node = self.node_indices.get(source_id).map(|n| *n)?;
        let target_node = self.node_indices.get(target_id).map(|n| *n)?;

        if source_node == target_node {
            return Some(RelationshipPath::new(vec![*source_id], vec![]));
        }

        let graph = self.graph.read();
        let mut visited = std::collections::HashSet::new();
        let mut parent: std::collections::HashMap<NodeIndex, NodeIndex> =
            std::collections::HashMap::new();
        let mut queue = std::collections::VecDeque::new();

        queue.push_back(source_node);
        visited.insert(source_node);

        while let Some(node) = queue.pop_front() {
            if node == target_node {
                // Reconstruct path
                let mut entities = Vec::new();
                let mut current = target_node;

                while let Some(id) = graph.node_weight(current) {
                    entities.push(*id);
                    if current == source_node {
                        break;
                    }
                    current = match parent.get(&current) {
                        Some(&p) => p,
                        None => break,
                    };
                }

                entities.reverse();
                return Some(RelationshipPath::new(entities, vec![]));
            }

            for neighbor in graph.neighbors(node) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, node);
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }

    /// Search entities by name or description
    pub async fn search_entities(&self, query: &str, limit: usize) -> Vec<GraphEntity> {
        let query_lower = query.to_lowercase();

        let mut results: Vec<GraphEntity> = self
            .entities
            .iter()
            .filter(|e| {
                e.name.to_lowercase().contains(&query_lower)
                    || e.description.to_lowercase().contains(&query_lower)
                    || e.aliases
                        .iter()
                        .any(|a| a.to_lowercase().contains(&query_lower))
            })
            .map(|e| e.clone())
            .collect();

        // Sort by relevance (importance + mention count)
        results.sort_by(|a, b| {
            let score_a = a.importance_score + (a.mention_count as f32).log10() * 0.1;
            let score_b = b.importance_score + (b.mention_count as f32).log10() * 0.1;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(limit);
        results
    }

    /// Get all entities of a specific type
    pub async fn get_entities_by_type(&self, entity_type: &EntityType) -> Vec<GraphEntity> {
        self.entities
            .iter()
            .filter(|e| {
                std::mem::discriminant(&e.entity_type) == std::mem::discriminant(entity_type)
            })
            .map(|e| e.clone())
            .collect()
    }

    /// Get entity adjacency list
    pub async fn get_adjacency(&self, entity_id: &EntityId) -> Vec<(EntityId, RelationshipType)> {
        let node = match self.node_indices.get(entity_id) {
            Some(n) => *n,
            None => return Vec::new(),
        };

        let graph = self.graph.read();
        let mut adjacency = Vec::new();

        for neighbor in graph.neighbors(node) {
            if let Some(neighbor_id) = graph.node_weight(neighbor) {
                // Find the edge
                if let Some(edge) = graph.find_edge(node, neighbor) {
                    if let Some(rel_id) = graph.edge_weight(edge) {
                        if let Some(rel) = self.relationships.get(rel_id) {
                            adjacency.push((*neighbor_id, rel.relationship_type.clone()));
                        }
                    }
                }
            }
        }

        adjacency
    }

    /// Auto-cluster entities based on similarity
    pub async fn cluster_entities(&self) -> Result<Vec<EntityCluster>> {
        let mut clusters = Vec::new();
        let mut assigned = HashSet::new();

        for entry in self.entities.iter() {
            let entity = entry.value();
            if assigned.contains(&entity.id) {
                continue;
            }

            // Find similar entities
            let mut cluster_entities = vec![entity.id];
            assigned.insert(entity.id);

            for other_entry in self.entities.iter() {
                let other = other_entry.value();
                if assigned.contains(&other.id) {
                    continue;
                }

                if entity.similarity(other) >= self.config.cluster_threshold {
                    cluster_entities.push(other.id);
                    assigned.insert(other.id);
                }
            }

            if cluster_entities.len() > 1 {
                // Compute centroid
                let centroid = self.compute_cluster_centroid(&cluster_entities);
                let mut cluster =
                    EntityCluster::new(entity.name.clone(), cluster_entities, centroid);

                // Compute coherence
                let entity_map: HashMap<EntityId, GraphEntity> =
                    self.entities.iter().map(|e| (e.id, e.clone())).collect();
                cluster.compute_coherence(&entity_map);

                self.clusters.insert(cluster.id, cluster.clone());
                clusters.push(cluster);
            }
        }

        Ok(clusters)
    }

    /// Get graph statistics
    pub async fn get_statistics(&self) -> GraphStatistics {
        let entity_count = self.entities.len();
        let relationship_count = self.relationships.len();
        let avg_degree = if entity_count > 0 {
            relationship_count as f32 * 2.0 / entity_count as f32
        } else {
            0.0
        };

        let type_distribution = {
            let mut dist = HashMap::new();
            for entry in self.entities.iter() {
                *dist.entry(format!("{:?}", entry.entity_type)).or_insert(0) += 1;
            }
            dist
        };

        GraphStatistics {
            entity_count,
            relationship_count,
            avg_degree,
            cluster_count: self.clusters.len(),
            type_distribution,
        }
    }

    /// Delete entities that reference a given memory ID (cascade delete support)
    pub async fn delete_entity_by_memory_id(&self, memory_id: &MemoryId) {
        let to_remove: Vec<EntityId> = self
            .entities
            .iter()
            .filter(|e| e.value().memory_ids.contains(memory_id))
            .map(|e| *e.key())
            .collect();

        for entity_id in to_remove {
            self.remove_entity_internal(&entity_id).await;
        }
    }

    /// Delete a specific entity and all its relationships
    pub async fn delete_entity(&self, entity_id: &EntityId) -> bool {
        if self.entities.contains_key(entity_id) {
            self.remove_entity_internal(entity_id).await;
            true
        } else {
            false
        }
    }

    // Private helper methods

    /// Internal: remove entity and all its relationships
    async fn remove_entity_internal(&self, entity_id: &EntityId) {
        // Remove relationships involving this entity (and their index entries)
        let rels_to_remove: Vec<Uuid> = self
            .relationships
            .iter()
            .filter(|r| r.value().source == *entity_id || r.value().target == *entity_id)
            .map(|r| *r.key())
            .collect();

        for rel_id in &rels_to_remove {
            if let Some(rel) = self.relationships.remove(rel_id) {
                let key = relationship_key(&rel.1.source, &rel.1.target, &rel.1.relationship_type);
                self.relationship_index.remove(&key);
            }
        }

        // Remove from graph structure
        if let Some(node) = self.node_indices.remove(entity_id) {
            let mut graph = self.graph.write();
            graph.remove_node(node.1);
        }

        // Remove from name index
        if let Some(entity) = self.entities.remove(entity_id) {
            let name_lower = entity.1.name.to_lowercase();
            if let Some(mut ids) = self.name_index.get_mut(&name_lower) {
                ids.remove(entity_id);
            }
            for alias in &entity.1.aliases {
                let alias_lower = alias.to_lowercase();
                if let Some(mut ids) = self.name_index.get_mut(&alias_lower) {
                    ids.remove(entity_id);
                }
            }
        }
    }

    fn index_entity_name(&self, entity: &GraphEntity) {
        let name_lower = entity.name.to_lowercase();
        self.name_index
            .entry(name_lower)
            .or_default()
            .insert(entity.id);

        for alias in &entity.aliases {
            let alias_lower = alias.to_lowercase();
            self.name_index
                .entry(alias_lower)
                .or_default()
                .insert(entity.id);
        }
    }

    fn find_existing_relationship_index(
        &self,
        source: &EntityId,
        target: &EntityId,
        rel_type: &RelationshipType,
    ) -> Option<Uuid> {
        let key = relationship_key(source, target, rel_type);
        self.relationship_index.get(&key).map(|r| *r.value())
    }

    fn update_centrality(&self, entity_id: &EntityId) {
        if let Some(mut entity) = self.entities.get_mut(entity_id) {
            // Calculate degree synchronously without await
            let degree = self.get_adjacency_sync(entity_id).len();
            entity.centrality_score = (degree as f32).log10() * 0.1;
            entity.importance_score = entity.compute_importance();
        }
    }

    fn get_adjacency_sync(&self, entity_id: &EntityId) -> Vec<(EntityId, RelationshipType)> {
        let node = match self.node_indices.get(entity_id) {
            Some(n) => *n,
            None => return Vec::new(),
        };

        let graph = self.graph.read();
        let mut adjacency = Vec::new();

        for neighbor in graph.neighbors(node) {
            if let Some(neighbor_id) = graph.node_weight(neighbor) {
                if let Some(edge) = graph.find_edge(node, neighbor) {
                    if let Some(rel_id) = graph.edge_weight(edge) {
                        if let Some(rel) = self.relationships.get(rel_id) {
                            adjacency.push((*neighbor_id, rel.relationship_type.clone()));
                        }
                    }
                }
            }
        }

        adjacency
    }

    fn compute_cluster_centroid(&self, entity_ids: &[EntityId]) -> Vec<f32> {
        if entity_ids.is_empty() {
            return Vec::new();
        }

        let first_dim = self
            .entities
            .get(&entity_ids[0])
            .map(|e| e.embedding.len())
            .unwrap_or(0);

        if first_dim == 0 {
            return Vec::new();
        }

        let mut centroid = vec![0.0; first_dim];
        let mut count = 0;

        for id in entity_ids {
            if let Some(entity) = self.entities.get(id) {
                for (i, &val) in entity.embedding.iter().enumerate() {
                    if i < centroid.len() {
                        centroid[i] += val;
                    }
                }
                count += 1;
            }
        }

        if count > 0 {
            for val in centroid.iter_mut() {
                *val /= count as f32;
            }
        }

        centroid
    }
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    pub entity_count: usize,
    pub relationship_count: usize,
    pub avg_degree: f32,
    pub cluster_count: usize,
    pub type_distribution: HashMap<String, usize>,
}
