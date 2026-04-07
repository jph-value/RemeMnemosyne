pub mod error;
pub mod types;
pub mod traits;
pub mod query;

pub use error::{MemoryError, Result};
pub use types::*;
pub use traits::*;
pub use query::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_artifact_creation() {
        let artifact = MemoryArtifact::new(
            MemoryType::Semantic,
            "Test summary",
            "Test content",
            vec![0.1, 0.2, 0.3],
            MemoryTrigger::UserInput,
        );

        assert_eq!(artifact.memory_type, MemoryType::Semantic);
        assert_eq!(artifact.summary, "Test summary");
        assert_eq!(artifact.content, "Test content");
        assert_eq!(artifact.embedding, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_context_bundle_creation() {
        let bundle = ContextBundle::new();
        assert!(bundle.is_empty());
    }

    #[test]
    fn test_memory_query_builder() {
        let query = MemoryQuery::new()
            .with_text("search term")
            .with_limit(5);

        assert_eq!(query.text, Some("search term".to_string()));
        assert_eq!(query.limit, Some(5));
    }

    #[test]
    fn test_entity_creation() {
        let entity = Entity::new(
            "TestEntity",
            EntityType::Technology,
            "A test entity",
            vec![0.1, 0.2],
        );

        assert_eq!(entity.name, "TestEntity");
        assert_eq!(entity.mention_count, 1);
    }

    #[test]
    fn test_memory_artifact_builder_methods() {
        let session_id = SessionId::new_v4();
        let artifact = MemoryArtifact::new(
            MemoryType::Episodic,
            "Builder test",
            "Content",
            vec![0.5; 128],
            MemoryTrigger::Insight,
        )
        .with_importance(Importance::High)
        .with_session(session_id)
        .with_tags(vec!["test".to_string(), "builder".to_string()])
        .with_metadata("key", serde_json::json!("value"));

        assert_eq!(artifact.importance, Importance::High);
        assert_eq!(artifact.session_id, Some(session_id));
        assert_eq!(artifact.tags, vec!["test", "builder"]);
        assert_eq!(artifact.metadata.get("key").unwrap(), "value");
    }

    #[test]
    fn test_memory_artifact_access_tracking() {
        let mut artifact = MemoryArtifact::new(
            MemoryType::Semantic,
            "Access test",
            "Content",
            vec![0.1; 64],
            MemoryTrigger::SystemOutput,
        );

        assert_eq!(artifact.access_count, 0);
        assert!(artifact.last_accessed.is_none());

        artifact.mark_accessed();
        assert_eq!(artifact.access_count, 1);
        assert!(artifact.last_accessed.is_some());

        artifact.mark_accessed();
        assert_eq!(artifact.access_count, 2);
    }

    #[test]
    fn test_relevance_computation_bounds() {
        let artifact = MemoryArtifact::new(
            MemoryType::Semantic,
            "Relevance test",
            "Content",
            vec![0.1; 32],
            MemoryTrigger::Decision,
        )
        .with_importance(Importance::Critical);

        let relevance = artifact.compute_relevance();
        assert!(relevance >= 0.0 && relevance <= 1.0);
    }

    #[test]
    fn test_context_bundle_merge() {
        let mut bundle1 = ContextBundle::new();
        let mut bundle2 = ContextBundle::new();

        let artifact1 = MemoryArtifact::new(
            MemoryType::Semantic,
            "Memory 1",
            "Content 1",
            vec![0.1; 10],
            MemoryTrigger::UserInput,
        );

        let artifact2 = MemoryArtifact::new(
            MemoryType::Episodic,
            "Memory 2",
            "Content 2",
            vec![0.2; 10],
            MemoryTrigger::Answer,
        );

        bundle1.add_memory(artifact1, 0.9);
        bundle2.add_memory(artifact2, 0.8);

        bundle1.merge(bundle2);

        assert_eq!(bundle1.memories.len(), 2);
        assert!(!bundle1.is_empty());
    }

    #[test]
    fn test_context_bundle_token_truncate() {
        let mut bundle = ContextBundle::new();

        // Add multiple memories
        for i in 0..5 {
            let artifact = MemoryArtifact::new(
                MemoryType::Semantic,
                &"a".repeat(100), // ~25 tokens each
                "Content",
                vec![0.1; 10],
                MemoryTrigger::UserInput,
            );
            bundle.add_memory(artifact, 0.5 + i as f32 * 0.1);
        }

        bundle.truncate_to_token_limit(50); // Allow ~2 memories
        assert!(bundle.memories.len() <= 3);
    }

    #[test]
    fn test_relationship_creation() {
        let source = EntityId::new_v4();
        let target = EntityId::new_v4();

        let rel = Relationship::new(
            source,
            target,
            RelationshipType::DependsOn,
            0.85,
        );

        assert_eq!(rel.source, source);
        assert_eq!(rel.target, target);
        assert_eq!(rel.relationship_type, RelationshipType::DependsOn);
        assert!((rel.strength - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_importance_ordering() {
        assert!(Importance::Low < Importance::Medium);
        assert!(Importance::Medium < Importance::High);
        assert!(Importance::High < Importance::Critical);
    }

    #[test]
    fn test_memory_type_display() {
        assert_eq!(format!("{}", MemoryType::Semantic), "semantic");
        assert_eq!(format!("{}", MemoryType::Episodic), "episodic");
        assert_eq!(format!("{}", MemoryType::Graph), "graph");
        assert_eq!(format!("{}", MemoryType::Temporal), "temporal");
    }
}
