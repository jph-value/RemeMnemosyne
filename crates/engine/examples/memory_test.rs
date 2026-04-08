use rememnemosyne_core::MemoryTrigger;
use rememnemosyne_engine::RememnosyneEngine;

#[tokio::main]
async fn main() {
    println!("=== RemeMnemosyne Integration Test ===\n");

    let engine = RememnosyneEngine::in_memory().expect("Failed to create engine");
    println!("Engine created (in-memory mode)");

    // Store memories
    let test_memories = vec![
        (
            "Rust is a systems programming language focused on safety and performance.",
            "Rust programming",
        ),
        (
            "Vector databases store high-dimensional vectors for semantic search.",
            "Vector databases",
        ),
        (
            "vLLM provides high throughput LLM serving with PagedAttention.",
            "vLLM deployment",
        ),
        (
            "Memory-augmented agents use external stores for context.",
            "Memory agents",
        ),
        (
            "HNSW enables approximate nearest neighbor search.",
            "HNSW algorithm",
        ),
        (
            "Quantization reduces model size while maintaining quality.",
            "Quantization",
        ),
        (
            "Embedding models convert text to numerical vectors.",
            "Embedding models",
        ),
        (
            "Prompt engineering guides LLM behavior.",
            "Prompt engineering",
        ),
        (
            "Mixture of Experts activates only a subset of parameters.",
            "MoE architecture",
        ),
        (
            "Retrieval-Augmented Generation combines retrieval with generation.",
            "RAG systems",
        ),
    ];

    println!("\nStoring {} memories...", test_memories.len());
    for (content, summary) in &test_memories {
        let _id = engine
            .remember(*content, *summary, MemoryTrigger::Insight)
            .await
            .expect("Failed to store memory");
    }

    // Test recall
    let test_queries = vec![
        (
            "What programming languages are good for systems programming?",
            "rust",
        ),
        ("How can I store embeddings for semantic search?", "vector"),
        (
            "What is the best way to deploy LLMs for high throughput?",
            "vllm",
        ),
    ];

    println!("\n=== Recall Tests ===");
    for (query, expected_keyword) in &test_queries {
        println!("\nQuery: {}", query);
        let bundle = engine.recall(query).await.expect("Failed to recall");
        let formatted = engine.context_builder.format_context(&bundle);

        if bundle.is_empty() {
            println!("  Result: NO MEMORIES FOUND");
        } else {
            println!("  Found {} memories", bundle.memories.len());
            for (i, mem) in bundle.memories.iter().take(3).enumerate() {
                let relevance = bundle.relevance_scores.get(&mem.id).unwrap_or(&0.0);
                println!("    {}. [{}] {}", i + 1, relevance, mem.summary);
            }

            let has_keyword = formatted.to_lowercase().contains(expected_keyword);
            println!(
                "  Contains expected keyword '{}': {}",
                expected_keyword, has_keyword
            );
        }
    }

    let stats = engine.get_stats().await;
    println!("\n=== Engine Stats ===");
    println!("Semantic memories: {}", stats.router.semantic_memories);
    println!("\n=== Test Complete ===");
}
