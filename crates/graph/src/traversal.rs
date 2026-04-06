use rememnemosyne_core::{EntityId, MemoryError, Result};
use petgraph::graph::{Graph, NodeIndex};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::entity::GraphEntity;
use crate::relationship::{GraphRelationship, RelationshipPath};

/// Graph traversal algorithms
pub struct GraphTraversal;

impl GraphTraversal {
    /// Breadth-first search to find all reachable entities
    pub fn bfs(
        graph: &Graph<EntityId, uuid::Uuid>,
        start: NodeIndex,
        max_depth: usize,
    ) -> Vec<(NodeIndex, usize)> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back((start, 0));
        visited.insert(start);

        while let Some((node, depth)) = queue.pop_front() {
            result.push((node, depth));

            if depth >= max_depth {
                continue;
            }

            for neighbor in graph.neighbors(node) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        result
    }

    /// Depth-first search for exploring deep relationships
    pub fn dfs(
        graph: &Graph<EntityId, uuid::Uuid>,
        start: NodeIndex,
        max_depth: usize,
    ) -> Vec<(NodeIndex, usize)> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut stack = vec![(start, 0)];

        while let Some((node, depth)) = stack.pop() {
            if visited.contains(&node) {
                continue;
            }

            visited.insert(node);
            result.push((node, depth));

            if depth >= max_depth {
                continue;
            }

            for neighbor in graph.neighbors(node) {
                if !visited.contains(&neighbor) {
                    stack.push((neighbor, depth + 1));
                }
            }
        }

        result
    }

    /// Find shortest path using BFS (unweighted)
    pub fn shortest_path(
        graph: &Graph<EntityId, uuid::Uuid>,
        start: NodeIndex,
        end: NodeIndex,
    ) -> Option<Vec<NodeIndex>> {
        if start == end {
            return Some(vec![start]);
        }

        let mut visited = HashSet::new();
        let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(node) = queue.pop_front() {
            for neighbor in graph.neighbors(node) {
                if visited.contains(&neighbor) {
                    continue;
                }

                visited.insert(neighbor);
                parent.insert(neighbor, node);

                if neighbor == end {
                    // Reconstruct path
                    let mut path = vec![end];
                    let mut current = end;
                    while let Some(&p) = parent.get(&current) {
                        path.push(p);
                        current = p;
                    }
                    path.reverse();
                    return Some(path);
                }

                queue.push_back(neighbor);
            }
        }

        None
    }

    /// Find all paths between two nodes (limited by max_paths)
    pub fn all_paths(
        graph: &Graph<EntityId, uuid::Uuid>,
        start: NodeIndex,
        end: NodeIndex,
        max_depth: usize,
        max_paths: usize,
    ) -> Vec<Vec<NodeIndex>> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut current_path = Vec::new();

        Self::dfs_all_paths(
            graph,
            start,
            end,
            max_depth,
            max_paths,
            &mut visited,
            &mut current_path,
            &mut paths,
        );

        paths
    }

    fn dfs_all_paths(
        graph: &Graph<EntityId, uuid::Uuid>,
        current: NodeIndex,
        end: NodeIndex,
        max_depth: usize,
        max_paths: usize,
        visited: &mut HashSet<NodeIndex>,
        current_path: &mut Vec<NodeIndex>,
        all_paths: &mut Vec<Vec<NodeIndex>>,
    ) {
        if all_paths.len() >= max_paths {
            return;
        }

        if current_path.len() > max_depth {
            return;
        }

        visited.insert(current);
        current_path.push(current);

        if current == end {
            all_paths.push(current_path.clone());
        } else {
            for neighbor in graph.neighbors(current) {
                if !visited.contains(&neighbor) {
                    Self::dfs_all_paths(
                        graph,
                        neighbor,
                        end,
                        max_depth,
                        max_paths,
                        visited,
                        current_path,
                        all_paths,
                    );
                }
            }
        }

        current_path.pop();
        visited.remove(&current);
    }

    /// PageRank-like centrality calculation
    pub fn compute_centrality(
        graph: &Graph<EntityId, uuid::Uuid>,
        damping: f32,
        iterations: usize,
    ) -> HashMap<NodeIndex, f32> {
        let node_count = graph.node_count();
        if node_count == 0 {
            return HashMap::new();
        }

        let mut centrality: HashMap<NodeIndex, f32> = graph
            .node_indices()
            .map(|n| (n, 1.0 / node_count as f32))
            .collect();

        for _ in 0..iterations {
            let mut new_centrality: HashMap<NodeIndex, f32> = graph
                .node_indices()
                .map(|n| (n, (1.0 - damping) / node_count as f32))
                .collect();

            for node in graph.node_indices() {
                let out_degree = graph.neighbors(node).count();
                if out_degree == 0 {
                    continue;
                }

                let current_centrality = centrality.get(&node).unwrap_or(&0.0);
                let contribution = damping * current_centrality / out_degree as f32;

                for neighbor in graph.neighbors(node) {
                    *new_centrality.entry(neighbor).or_insert(0.0) += contribution;
                }
            }

            centrality = new_centrality;
        }

        centrality
    }

    /// Find connected components
    pub fn connected_components(graph: &Graph<EntityId, uuid::Uuid>) -> Vec<Vec<NodeIndex>> {
        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for node in graph.node_indices() {
            if visited.contains(&node) {
                continue;
            }

            let mut component = Vec::new();
            let mut stack = vec![node];

            while let Some(current) = stack.pop() {
                if visited.contains(&current) {
                    continue;
                }

                visited.insert(current);
                component.push(current);

                for neighbor in graph.neighbors(current) {
                    if !visited.contains(&neighbor) {
                        stack.push(neighbor);
                    }
                }
            }

            if !component.is_empty() {
                components.push(component);
            }
        }

        components
    }

    /// Find nodes within a certain distance (using BFS)
    pub fn nodes_within_distance(
        graph: &Graph<EntityId, uuid::Uuid>,
        start: NodeIndex,
        max_distance: usize,
    ) -> Vec<(NodeIndex, usize)> {
        // Use BFS algorithm (unit weights)
        let mut distances: HashMap<NodeIndex, usize> = HashMap::new();
        let mut queue = std::collections::VecDeque::new();

        distances.insert(start, 0);
        queue.push_back((start, 0));

        while let Some((node, dist)) = queue.pop_front() {
            if dist > max_distance {
                continue;
            }

            for neighbor in graph.neighbors(node) {
                let new_dist = dist + 1;
                if !distances.contains_key(&neighbor) && new_dist <= max_distance {
                    distances.insert(neighbor, new_dist);
                    queue.push_back((neighbor, new_dist));
                }
            }
        }

        distances.into_iter().collect()
    }
}

/// Path analysis utilities
pub struct PathAnalysis;

impl PathAnalysis {
    /// Analyze a path to extract insights
    pub fn analyze_path(
        path: &RelationshipPath,
        entities: &HashMap<EntityId, GraphEntity>,
        relationships: &HashMap<uuid::Uuid, GraphRelationship>,
    ) -> PathAnalysisResult {
        let mut result = PathAnalysisResult::new();

        // Entity types along the path
        for entity_id in &path.entities {
            if let Some(entity) = entities.get(entity_id) {
                result.entity_types.push(entity.entity_type.clone());
            }
        }

        // Relationship types
        for rel in &path.relationships {
            result
                .relationship_types
                .push(rel.relationship_type.clone());
            result.avg_strength += rel.strength;
        }

        if !path.relationships.is_empty() {
            result.avg_strength /= path.relationships.len() as f32;
        }

        // Complexity score
        result.complexity = path.hop_count as f32 * result.avg_strength;

        result
    }

    /// Compare two paths
    pub fn compare_paths(path1: &RelationshipPath, path2: &RelationshipPath) -> f32 {
        // Jaccard similarity of entity sets
        let set1: std::collections::HashSet<_> = path1.entities.iter().collect();
        let set2: std::collections::HashSet<_> = path2.entities.iter().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

#[derive(Debug, Clone)]
pub struct PathAnalysisResult {
    pub entity_types: Vec<rememnemosyne_core::EntityType>,
    pub relationship_types: Vec<rememnemosyne_core::RelationshipType>,
    pub avg_strength: f32,
    pub complexity: f32,
}

impl PathAnalysisResult {
    pub fn new() -> Self {
        Self {
            entity_types: Vec::new(),
            relationship_types: Vec::new(),
            avg_strength: 0.0,
            complexity: 0.0,
        }
    }
}
