use rememnemosyne_core::{
    MemoryArtifact, MemoryType, MemoryTrigger, MemoryQuery, Importance,
    ContextBundle, SessionId,
};
use rememnemosyne_engine::{
    RememnosyneEngineBuilder, RememnosyneEngine,
    ContextConfig, MemorySanitizer,
};
use std::sync::Arc;

/// Integration test: Full memory lifecycle - store, retrieve, and context assembly
#[tokio::test]
async fn test_full_memory_lifecycle() {
    // Build engine with in-memory storage for testing
    let engine = RememnosyneEngineBuilder::new()
        .with_memory_storage(true)
        .build()
        .expect("Failed to build engine");

    let session_id = SessionId::new_v4();

    // Store a memory
    let artifact = MemoryArtifact::new(
        MemoryType::Semantic,
        "Rust programming concepts",
        "Ownership and borrowing are fundamental to Rust's memory safety guarantees.",
        vec![0.1; 128],
        MemoryTrigger::UserInput,
    )
    .with_importance(Importance::High)
    .with_session(session_id);

    let memory_id = engine.store(artifact.clone()).await
        .expect("Failed to store memory");

    // Retrieve the memory
    let retrieved = engine.retrieve(memory_id).await
        .expect("Failed to retrieve memory");

    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.summary, "Rust programming concepts");

    // Query for related memories
    let query = MemoryQuery::new()
        .with_text("Rust ownership")
        .with_limit(5);

    let results = engine.query(query).await
        .expect("Failed to query memories");

    assert!(!results.is_empty());
}

/// Integration test: Context assembly for LLM consumption
#[tokio::test]
async fn test_context_assembly() {
    let engine = RememnosyneEngineBuilder::new()
        .with_memory_storage(true)
        .build()
        .expect("Failed to build engine");

    let session_id = SessionId::new_v4();

    // Store multiple memories in a session
    for i in 0..5 {
        let artifact = MemoryArtifact::new(
            MemoryType::Episodic,
            &format!("Conversation turn {}", i),
            &format!("User asked question {} and received answer.", i),
            vec![0.1 + i as f32 * 0.01; 128],
            MemoryTrigger::Question,
        )
        .with_session(session_id);

        engine.store(artifact).await.unwrap();
    }

    // Assemble context for current query
    let config = ContextConfig {
        max_tokens: 500,
        semantic_ratio: 0.3,
        episodic_ratio: 0.5,
        graph_ratio: 0.1,
        temporal_ratio: 0.1,
    };

    let context = engine.assemble_context(
        "What was discussed earlier?",
        config,
        Some(session_id),
    ).await.expect("Failed to assemble context");

    assert!(!context.is_empty());
    assert!(context.total_tokens_estimate <= 500);
}

/// Integration test: Memory pruning based on retention scores
#[tokio::test]
async fn test_memory_pruning() {
    let engine = RememnosyneEngineBuilder::new()
        .with_memory_storage(true)
        .build()
        .expect("Failed to build engine");

    // Store memories with varying importance and access patterns
    let low_priority = MemoryArtifact::new(
        MemoryType::Semantic,
        "Low priority fact",
        "This is rarely accessed.",
        vec![0.01; 64],
        MemoryTrigger::SystemOutput,
    );

    let high_priority = MemoryArtifact::new(
        MemoryType::Semantic,
        "Critical fact",
        "This is very important.",
        vec![0.9; 64],
        MemoryTrigger::Decision,
    )
    .with_importance(Importance::Critical);

    let low_id = engine.store(low_priority).await.unwrap();
    let high_id = engine.store(high_priority).await.unwrap();

    // Access high priority memory multiple times
    for _ in 0..10 {
        engine.retrieve(high_id).await.unwrap();
    }

    // Run pruning
    let pruned = engine.prune_memories(0.5).await
        .expect("Failed to prune memories");

    // High priority memory should be retained, low priority may be pruned
    let high_exists = engine.retrieve(high_id).await.unwrap().is_some();
    assert!(high_exists, "High priority memory should be retained");
}

/// Integration test: Content sanitization pipeline
#[test]
fn test_sanitization_pipeline() {
    // Test various content that needs sanitization
    let test_cases = vec![
        ("  Trimmed  ", "Trimmed"),
        ("\tTabbed\tContent\t", "Tabbed Content"),
        ("\nNewlines\nEverywhere\n", "Newlines Everywhere"),
        ("   Multiple   Spaces   ", "Multiple Spaces"),
    ];

    for (input, expected) in test_cases {
        let sanitized = MemorySanitizer::sanitize_content(input);
        assert_eq!(sanitized, expected, "Sanitization failed for: {}", input);
    }
}

/// Integration test: Concurrent memory operations
#[tokio::test]
async fn test_concurrent_operations() {
    let engine = Arc::new(
        RememnosyneEngineBuilder::new()
            .with_memory_storage(true)
            .build()
            .expect("Failed to build engine")
    );

    let mut handles = vec![];

    // Spawn multiple concurrent store operations
    for i in 0..10 {
        let engine_clone = Arc::clone(&engine);
        let handle = tokio::spawn(async move {
            let artifact = MemoryArtifact::new(
                MemoryType::Semantic,
                &format!("Concurrent memory {}", i),
                "Content",
                vec![i as f32 / 10.0; 64],
                MemoryTrigger::UserInput,
            );
            engine_clone.store(artifact).await
        });
        handles.push(handle);
    }

    // Wait for all operations
    let results = futures::future::join_all(handles).await;

    // All should succeed
    for result in results {
        assert!(result.is_ok(), "Concurrent store failed");
        assert!(result.unwrap().is_ok(), "Store operation failed");
    }
}

/// Integration test: Query with various filters
#[tokio::test]
async fn test_filtered_queries() {
    let engine = RememnosyneEngineBuilder::new()
        .with_memory_storage(true)
        .build()
        .expect("Failed to build engine");

    // Store different memory types
    let semantic = MemoryArtifact::new(
        MemoryType::Semantic,
        "Vector search concept",
        "HNSW is an approximate nearest neighbor algorithm.",
        vec![0.8; 128],
        MemoryTrigger::Insight,
    );

    let episodic = MemoryArtifact::new(
        MemoryType::Episodic,
        "User conversation",
        "User: How does vector search work?",
        vec![0.3; 128],
        MemoryTrigger::Question,
    );

    engine.store(semantic).await.unwrap();
    engine.store(episodic).await.unwrap();

    // Query with type filter
    let query = MemoryQuery::new()
        .with_text("vector")
        .with_memory_types(vec![MemoryType::Semantic]);

    let results = engine.query(query).await.unwrap();

    // Should only return semantic memories
    for memory in &results {
        assert_eq!(memory.memory_type, MemoryType::Semantic);
    }
}
