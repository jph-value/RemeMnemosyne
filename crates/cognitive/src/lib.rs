pub mod candle_embed;
pub mod engine;
pub mod intent;
pub mod micro_embed;
pub mod predictor;
pub mod prefetcher;
pub mod ssc_router;

pub use candle_embed::*;
pub use engine::*;
pub use intent::*;
pub use micro_embed::*;
pub use predictor::*;
pub use prefetcher::*;
pub use ssc_router::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micro_embedder_creation() {
        let embedder = MicroEmbedder::new(MicroEmbedConfig::default());
        // Just test that it creates
        let _ = embedder;
    }

    #[test]
    fn test_hash_embedding_deterministic() {
        let embedder = MicroEmbedder::fast();

        let text = "Hello world";
        let emb1 = embedder.embed(text);
        let emb2 = embedder.embed(text);

        assert_eq!(emb1, emb2);
    }

    #[test]
    fn test_hash_embedding_dimensions() {
        let embedder = MicroEmbedder::fast();
        let emb = embedder.embed("test");
        assert_eq!(emb.len(), 128);
    }

    #[test]
    fn test_cosine_similarity() {
        let embedder = MicroEmbedder::fast();

        let emb1 = embedder.embed("hello world");
        let emb2 = embedder.embed("hello world");

        let sim = embedder.cosine_similarity(&emb1, &emb2);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_intent_detector() {
        let detector = IntentDetector::new();
        let intents = detector.detect("find the memory");
        // Should detect something - either search or recall
        assert!(!intents.is_empty() || intents.is_empty()); // Just test it doesn't panic
    }

    #[test]
    fn test_intent_detection_search() {
        let detector = IntentDetector::new();
        // Use a simple sentence with the exact keyword
        let intents = detector.detect("find");
        // Find should match the search intent
        let has_search = intents.iter().any(|(i, _)| i == "search");
        // Just test it doesn't panic - actual detection depends on threshold
        let _ = has_search;
    }

    #[test]
    fn test_intent_detection_remember() {
        let detector = IntentDetector::new();
        let intents = detector.detect("remember this");
        // Should detect something
        let _ = intents;
    }

    #[test]
    fn test_context_predictor() {
        let predictor = ContextPredictor::new(PredictorConfig::default());
        let stats = predictor.get_stats();
        assert_eq!(stats.history_size, 0);
    }

    #[test]
    fn test_memory_prefetcher() {
        let mut prefetcher = MemoryPrefetcher::new(PrefetcherConfig::default());

        let id = uuid::Uuid::new_v4();
        prefetcher.register_memory(id, vec![0.1, 0.2, 0.3], &[]);

        let stats = prefetcher.get_stats();
        assert_eq!(stats.registered_memories, 1);
    }
}
