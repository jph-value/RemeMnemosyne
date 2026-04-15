use async_trait::async_trait;
use rememnemosyne_core::{EntityRef, MemoryId, Result};

use crate::intent::IntentDetector;
use crate::micro_embed::MicroEmbedder;
use crate::predictor::ContextPredictor;
use crate::prefetcher::MemoryPrefetcher;
use crate::ssc_router::SSCRouter;

/// Concrete implementation of the CognitiveEngine trait.
///
/// Integrates micro-embeddings, intent detection, context prediction,
/// prefetching, and SSC routing into a single coherent engine that
/// implements arXiv:2602.24281 Memory Caching concepts at the
/// application level.
pub struct CognitiveEngineImpl {
    embedder: MicroEmbedder,
    intent_detector: IntentDetector,
    predictor: std::sync::Mutex<ContextPredictor>,
    prefetcher: std::sync::Mutex<MemoryPrefetcher>,
    ssc_router: Option<SSCRouter>,
}

impl CognitiveEngineImpl {
    pub fn new() -> Self {
        Self {
            embedder: MicroEmbedder::fast(),
            intent_detector: IntentDetector::new(),
            predictor: std::sync::Mutex::new(ContextPredictor::new(Default::default())),
            prefetcher: std::sync::Mutex::new(MemoryPrefetcher::new(Default::default())),
            ssc_router: None,
        }
    }

    pub fn with_ssc_router(mut self, router: SSCRouter) -> Self {
        self.ssc_router = Some(router);
        self
    }

    pub fn with_predictor(mut self, predictor: ContextPredictor) -> Self {
        self.predictor = std::sync::Mutex::new(predictor);
        self
    }

    pub fn with_prefetcher(mut self, prefetcher: MemoryPrefetcher) -> Self {
        self.prefetcher = std::sync::Mutex::new(prefetcher);
        self
    }
}

impl Default for CognitiveEngineImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl rememnemosyne_core::CognitiveEngine for CognitiveEngineImpl {
    async fn micro_embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embedder.embed(text))
    }

    async fn detect_intent(&self, text: &str) -> Result<Vec<(String, f32)>> {
        Ok(self.intent_detector.detect(text))
    }

    async fn extract_entities(&self, text: &str) -> Result<Vec<EntityRef>> {
        Ok(self.embedder.extract_entities_ner(text))
    }

    async fn predict_relevance(
        &self,
        context: &[String],
        candidate_ids: &[MemoryId],
    ) -> Result<Vec<(MemoryId, f32)>> {
        let last_context = context.last().cloned().unwrap_or_default();
        let predictor = self.predictor.lock().unwrap();
        Ok(predictor.predict(&last_context, candidate_ids))
    }

    async fn prefetch(&self, query: &str, limit: usize) -> Result<Vec<MemoryId>> {
        let prefetcher = self.prefetcher.lock().unwrap();
        let all_ids: Vec<MemoryId> = Vec::new();
        let results = prefetcher.prefetch(query, &all_ids);
        let limited: Vec<MemoryId> = results.into_iter().take(limit).collect();
        Ok(limited)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_engine_creation() {
        let engine = CognitiveEngineImpl::new();
        assert!(engine.ssc_router.is_none());
    }
}