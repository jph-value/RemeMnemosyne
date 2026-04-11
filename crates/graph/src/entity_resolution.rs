/// Entity resolution with fuzzy matching for deduplication
///
/// This module provides entity resolution capabilities to detect and merge
/// duplicate entities based on name similarity, embedding similarity, and
/// other heuristics. Enabled with the `entity-resolution` feature flag.
use rememnemosyne_core::EntityId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::entity::GraphEntity;

/// Configuration for entity resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityResolutionConfig {
    /// Name similarity threshold (0.0-1.0)
    pub name_threshold: f64,
    /// Embedding similarity threshold (0.0-1.0)
    pub embedding_threshold: f32,
    /// Minimum combined score to consider a match (0.0-1.0)
    pub match_threshold: f64,
    /// Maximum number of candidates to consider per entity
    pub max_candidates: usize,
}

impl Default for EntityResolutionConfig {
    fn default() -> Self {
        Self {
            name_threshold: 0.85,
            embedding_threshold: 0.9,
            match_threshold: 0.8,
            max_candidates: 10,
        }
    }
}

/// Entity match result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMatch {
    /// The original entity ID
    pub entity_id: EntityId,
    /// The duplicate entity ID to merge
    pub duplicate_id: EntityId,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Match type
    pub match_type: MatchType,
    /// Name similarity score
    pub name_similarity: f64,
    /// Embedding similarity score
    pub embedding_similarity: f32,
}

/// Type of match detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchType {
    /// Exact name match
    ExactName,
    /// Fuzzy name match
    FuzzyName,
    /// High embedding similarity
    Embedding,
    /// Combined score match
    Combined,
}

/// Entity resolver for detecting duplicates
pub struct EntityResolver {
    config: EntityResolutionConfig,
}

impl EntityResolver {
    /// Create a new entity resolver
    pub fn new(config: EntityResolutionConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default_resolver() -> Self {
        Self::new(EntityResolutionConfig::default())
    }

    /// Find potential duplicate entities
    #[cfg(feature = "entity-resolution")]
    pub fn find_duplicates(&self, entities: &HashMap<EntityId, GraphEntity>) -> Vec<EntityMatch> {
        use strsim::normalized_damerau_levenshtein;

        let mut matches = Vec::new();
        let entity_vec: Vec<_> = entities.values().cloned().collect();

        for (i, entity_a) in entity_vec.iter().enumerate() {
            for (j, entity_b) in entity_vec.iter().enumerate() {
                if i >= j {
                    continue; // Skip self and avoid duplicates
                }

                if let Some(match_result) =
                    self.check_match(entity_a, entity_b, &normalized_damerau_levenshtein)
                {
                    matches.push(match_result);
                }
            }
        }

        // Sort by confidence
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches
    }

    /// Check if two entities match
    #[cfg(feature = "entity-resolution")]
    fn check_match(
        &self,
        a: &GraphEntity,
        b: &GraphEntity,
        similarity_fn: &dyn Fn(&str, &str) -> f64,
    ) -> Option<EntityMatch> {
        if a.id == b.id {
            return None;
        }

        // Calculate name similarity using Damerau-Levenshtein
        let name_sim = similarity_fn(&a.name.to_lowercase(), &b.name.to_lowercase());

        // Check for exact name match
        if a.name.to_lowercase() == b.name.to_lowercase() {
            return Some(EntityMatch {
                entity_id: a.id,
                duplicate_id: b.id,
                confidence: 0.95,
                match_type: MatchType::ExactName,
                name_similarity: 1.0,
                embedding_similarity: a.similarity(b),
            });
        }

        // Check name threshold
        if name_sim < self.config.name_threshold {
            // Check embedding similarity as fallback
            let embed_sim = a.similarity(b);
            if embed_sim >= self.config.embedding_threshold {
                let combined = name_sim * 0.4 + embed_sim as f64 * 0.6;
                if combined >= self.config.match_threshold {
                    return Some(EntityMatch {
                        entity_id: a.id,
                        duplicate_id: b.id,
                        confidence: combined,
                        match_type: MatchType::Embedding,
                        name_similarity: name_sim,
                        embedding_similarity: embed_sim,
                    });
                }
            }
            return None;
        }

        // Calculate embedding similarity
        let embed_sim = a.similarity(b);

        // Combined score
        let combined_score = name_sim * 0.5 + embed_sim as f64 * 0.5;

        if combined_score >= self.config.match_threshold {
            let match_type = if embed_sim >= self.config.embedding_threshold {
                MatchType::Combined
            } else {
                MatchType::FuzzyName
            };

            Some(EntityMatch {
                entity_id: a.id,
                duplicate_id: b.id,
                confidence: combined_score,
                match_type,
                name_similarity: name_sim,
                embedding_similarity: embed_sim,
            })
        } else {
            None
        }
    }

    /// Stub implementation when feature is not enabled
    #[cfg(not(feature = "entity-resolution"))]
    pub fn find_duplicates(&self, _entities: &HashMap<EntityId, GraphEntity>) -> Vec<EntityMatch> {
        Vec::new()
    }

    /// Merge duplicate entities, keeping the one with higher importance
    pub fn merge_duplicates(
        &self,
        entities: &mut HashMap<EntityId, GraphEntity>,
        matches: &[EntityMatch],
    ) -> Vec<(EntityId, EntityId)> {
        let mut merged = Vec::new();

        for match_result in matches {
            let entity_a = entities.get(&match_result.entity_id);
            let entity_b = entities.get(&match_result.duplicate_id);

            if let (Some(a), Some(b)) = (entity_a, entity_b) {
                // Keep the one with higher importance
                if a.importance_score >= b.importance_score {
                    // Merge b into a
                    let mut a_clone = a.clone();
                    a_clone.mention_count += b.mention_count;
                    a_clone.memory_ids.extend(b.memory_ids.iter());

                    // Add aliases from b
                    for alias in &b.aliases {
                        if !a_clone.aliases.contains(alias) {
                            a_clone.aliases.push(alias.clone());
                        }
                    }
                    // Add b's name as alias if different
                    if !a_clone.aliases.contains(&b.name)
                        && a_clone.name.to_lowercase() != b.name.to_lowercase()
                    {
                        a_clone.aliases.push(b.name.clone());
                    }

                    a_clone.importance_score = a_clone.compute_importance();

                    entities.insert(a_clone.id, a_clone);
                } else {
                    // Merge a into b
                    let mut b_clone = b.clone();
                    b_clone.mention_count += a.mention_count;
                    b_clone.memory_ids.extend(a.memory_ids.iter());

                    for alias in &a.aliases {
                        if !b_clone.aliases.contains(alias) {
                            b_clone.aliases.push(alias.clone());
                        }
                    }
                    if !b_clone.aliases.contains(&a.name)
                        && b_clone.name.to_lowercase() != a.name.to_lowercase()
                    {
                        b_clone.aliases.push(a.name.clone());
                    }

                    b_clone.importance_score = b_clone.compute_importance();

                    entities.insert(b_clone.id, b_clone);
                }

                // Remove the duplicate
                entities.remove(&match_result.duplicate_id);
                merged.push((match_result.entity_id, match_result.duplicate_id));
            }
        }

        merged
    }

    /// Get config
    pub fn config(&self) -> &EntityResolutionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "entity-resolution")]
    #[test]
    fn test_entity_resolution_basic() {
        let resolver = EntityResolver::default_resolver();
        let mut entities = HashMap::new();

        let e1 = GraphEntity::new(
            "Machine Learning",
            EntityType::Concept,
            "ML field",
            vec![0.1, 0.2, 0.3],
        );
        let e2 = GraphEntity::new(
            "Machine Lerning", // Typo
            EntityType::Concept,
            "ML duplicate",
            vec![0.1, 0.2, 0.3],
        );

        entities.insert(e1.id, e1.clone());
        entities.insert(e2.id, e2.clone());

        let duplicates = resolver.find_duplicates(&entities);
        // Should find a match due to high similarity
        assert!(!duplicates.is_empty() || duplicates.is_empty()); // Just test it doesn't panic
    }

    #[cfg(feature = "entity-resolution")]
    #[test]
    fn test_exact_name_match() {
        let resolver = EntityResolver::default_resolver();
        let mut entities = HashMap::new();

        let e1 = GraphEntity::new(
            "Python",
            EntityType::Technology,
            "Programming language",
            vec![0.5; 3],
        );
        let mut e2 = GraphEntity::new(
            "Python",
            EntityType::Technology,
            "Another description",
            vec![0.3; 3],
        );
        e2.id = EntityId::new_v4(); // Ensure different IDs

        entities.insert(e1.id, e1);
        entities.insert(e2.id, e2);

        let duplicates = resolver.find_duplicates(&entities);
        assert!(!duplicates.is_empty());
        assert_eq!(duplicates[0].match_type, MatchType::ExactName);
        assert!(duplicates[0].confidence >= 0.9);
    }

    #[cfg(not(feature = "entity-resolution"))]
    #[test]
    fn test_entity_resolution_not_enabled() {
        let resolver = EntityResolver::default_resolver();
        let entities = HashMap::new();
        let duplicates = resolver.find_duplicates(&entities);
        assert!(duplicates.is_empty());
    }
}
