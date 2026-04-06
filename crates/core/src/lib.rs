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
}
