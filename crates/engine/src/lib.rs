pub mod router;
pub mod context;
pub mod builder;
pub mod api;
pub mod sanitizer;
pub mod pruner;

pub use router::*;
pub use context::*;
pub use builder::*;
pub use api::*;
pub use sanitizer::*;
pub use pruner::*;

#[cfg(test)]
mod tests {
    use super::*;
    use rememnemosyne_core::{
        MemoryArtifact, MemoryType, MemoryTrigger, MemoryQuery, Importance,
        ContextBundle, Entity, EntityType,
    };
    use std::collections::HashMap;

    #[test]
    fn test_memory_sanitizer_basic() {
        let sanitized = MemorySanitizer::sanitize_content("  Hello World  ");
        assert_eq!(sanitized, "Hello World");
    }

    #[test]
    fn test_memory_sanitizer_truncate() {
        let long_content = "a".repeat(10000);
        let sanitized = MemorySanitizer::sanitize_content(&long_content);
        assert!(sanitized.len() <= 5000);
    }

    #[test]
    fn test_context_window_calculation() {
        let config = ContextConfig {
            max_tokens: 1000,
            semantic_ratio: 0.4,
            episodic_ratio: 0.3,
            graph_ratio: 0.2,
            temporal_ratio: 0.1,
        };

        let window = ContextWindow::new(config.clone());
        assert_eq!(window.config.max_tokens, 1000);
    }

    #[test]
    fn test_context_bundle_optimization() {
        let config = ContextConfig {
            max_tokens: 100,
            semantic_ratio: 0.5,
            episodic_ratio: 0.5,
            graph_ratio: 0.0,
            temporal_ratio: 0.0,
        };

        let mut window = ContextWindow::new(config);

        // Add memories exceeding token limit
        for i in 0..10 {
            let artifact = MemoryArtifact::new(
                MemoryType::Semantic,
                &"word ".repeat(20), // ~100 tokens each
                "content",
                vec![0.1; 10],
                MemoryTrigger::UserInput,
            );
            window.add_memory(artifact, 0.5 + i as f32 * 0.05);
        }

        let bundle = window.optimize(&MemoryQuery::new());
        assert!(bundle.total_tokens_estimate <= 100);
    }

    #[test]
    fn test_memory_pruner_score_calculation() {
        let pruner = MemoryPruner::default();

        let mut artifact = MemoryArtifact::new(
            MemoryType::Semantic,
            "Test",
            "Content",
            vec![0.1; 10],
            MemoryTrigger::UserInput,
        )
        .with_importance(Importance::High);

        // Access multiple times to increase score
        for _ in 0..5 {
            artifact.mark_accessed();
        }

        let score = pruner.compute_retention_score(&artifact);
        assert!(score > 0.0);
    }

    #[test]
    fn test_engine_builder_default() {
        let builder = RememnosyneEngineBuilder::new();
        // Builder should have sensible defaults
        let engine = builder.build();
        assert!(engine.is_ok());
    }

    #[test]
    fn test_entity_extraction_placeholder() {
        // Test entity extraction from sample text
        let text = "Working with Rust and Python on the Mnemosyne project.";
        let entities = extract_entities(text);

        // Should detect some entities (even with basic extraction)
        assert!(!entities.is_empty());
    }

    fn extract_entities(text: &str) -> Vec<Entity> {
        // Simple entity extraction for testing
        let technologies = vec!["Rust", "Python", "JavaScript", "Go"];
        let mut entities = Vec::new();

        for tech in &technologies {
            if text.contains(*tech) {
                entities.push(Entity::new(
                    tech.to_string(),
                    EntityType::Technology,
                    format!("{} programming language", tech),
                    vec![0.1; 128],
                ));
            }
        }

        entities
    }
}
