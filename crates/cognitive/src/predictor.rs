use rememnemosyne_core::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

use crate::intent::IntentDetector;
use crate::micro_embed::MicroEmbedder;

/// Configuration for context predictor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorConfig {
    pub history_size: usize,
    pub prediction_window: usize,
    pub confidence_threshold: f32,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            history_size: 50,
            prediction_window: 5,
            confidence_threshold: 0.3,
        }
    }
}

/// Context predictor that anticipates memory needs
pub struct ContextPredictor {
    config: PredictorConfig,
    embedder: MicroEmbedder,
    intent_detector: IntentDetector,
    /// History of recent embeddings
    embedding_history: VecDeque<Vec<f32>>,
    /// History of retrieved memory IDs
    retrieval_history: VecDeque<Vec<MemoryId>>,
    /// Transition probabilities between memory clusters (reserved for future sequence modeling)
    #[allow(dead_code)]
    transition_matrix: HashMap<(usize, usize), f32>,
}

impl ContextPredictor {
    pub fn new(config: PredictorConfig) -> Self {
        Self {
            config,
            embedder: MicroEmbedder::fast(),
            intent_detector: IntentDetector::new(),
            embedding_history: VecDeque::new(),
            retrieval_history: VecDeque::new(),
            transition_matrix: HashMap::new(),
        }
    }

    /// Add a new context embedding to history
    pub fn add_context(&mut self, text: &str, retrieved_ids: Vec<MemoryId>) {
        let embedding = self.embedder.embed(text);

        self.embedding_history.push_back(embedding);
        if self.embedding_history.len() > self.config.history_size {
            self.embedding_history.pop_front();
        }

        self.retrieval_history.push_back(retrieved_ids);
        if self.retrieval_history.len() > self.config.history_size {
            self.retrieval_history.pop_front();
        }
    }

    /// Predict relevant memories based on current context
    pub fn predict(&self, current_text: &str, candidate_ids: &[MemoryId]) -> Vec<(MemoryId, f32)> {
        let current_embedding = self.embedder.embed(current_text);

        // Get intent
        let intents = self.intent_detector.detect(current_text);
        let primary_intent = intents.first().map(|(i, _)| i.as_str());

        // Predict based on similarity to recent contexts
        let mut predictions: HashMap<MemoryId, f32> = HashMap::new();

        for (recent_embedding, recent_retrievals) in self
            .embedding_history
            .iter()
            .zip(self.retrieval_history.iter())
        {
            let similarity = self
                .embedder
                .cosine_similarity(&current_embedding, recent_embedding);

            if similarity > 0.5 {
                for &memory_id in recent_retrievals {
                    *predictions.entry(memory_id).or_insert(0.0) += similarity;
                }
            }
        }

        // Intent-based boosting
        match primary_intent {
            Some("search") | Some("recall") => {
                // Boost all candidates slightly
                for &id in candidate_ids {
                    *predictions.entry(id).or_insert(0.0) += 0.1;
                }
            }
            Some("analyze") => {
                // Boost memories with entities
                // Would need entity info - simplified
            }
            _ => {}
        }

        // Normalize and filter
        let max_score = predictions.values().fold(0.0f32, |a, &b| a.max(b));
        if max_score > 0.0 {
            predictions.iter_mut().for_each(|(_, v)| *v /= max_score);
        }

        // Convert to vec and sort
        let mut results: Vec<(MemoryId, f32)> = predictions
            .into_iter()
            .filter(|(_, score)| *score >= self.config.confidence_threshold)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.config.prediction_window);

        results
    }

    /// Predict next likely topic based on conversation flow
    pub fn predict_topic(&self, recent_texts: &[String]) -> Option<String> {
        if recent_texts.is_empty() {
            return None;
        }

        // Analyze recent text patterns
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for text in recent_texts {
            for word in text.split_whitespace() {
                let word = word.to_lowercase();
                if word.len() > 3 {
                    *word_freq.entry(word).or_insert(0) += 1;
                }
            }
        }

        // Get most frequent non-stopword
        let stop_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "i", "you", "he", "she", "it", "we", "they", "this", "that",
        ];

        let mut sorted: Vec<_> = word_freq
            .into_iter()
            .filter(|(w, _)| !stop_words.contains(&w.as_str()))
            .collect();

        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.first().map(|(w, _)| w.clone())
    }

    /// Get conversation flow pattern
    pub fn get_flow_pattern(&self) -> FlowPattern {
        if self.embedding_history.len() < 2 {
            return FlowPattern::Unknown;
        }

        // Compute embedding drift
        let mut total_drift = 0.0;
        let mut drifts = Vec::new();

        for i in 1..self.embedding_history.len() {
            let sim = self
                .embedder
                .cosine_similarity(&self.embedding_history[i], &self.embedding_history[i - 1]);
            let drift = 1.0 - sim;
            total_drift += drift;
            drifts.push(drift);
        }

        let avg_drift = total_drift / (self.embedding_history.len() - 1) as f32;

        if avg_drift < 0.1 {
            FlowPattern::Focused
        } else if avg_drift < 0.3 {
            FlowPattern::Exploring
        } else if avg_drift < 0.5 {
            FlowPattern::Branching
        } else {
            FlowPattern::Scattered
        }
    }
}

/// Conversation flow patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowPattern {
    /// Topic is stable, few context switches
    Focused,
    /// Moderate exploration around a topic
    Exploring,
    /// Multiple related topics being discussed
    Branching,
    /// Many unrelated topics
    Scattered,
    /// Not enough data
    Unknown,
}

/// Retrieval prediction statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionStats {
    pub history_size: usize,
    pub avg_retrieval_count: f32,
    pub flow_pattern: FlowPattern,
    pub top_intents: Vec<String>,
}

impl ContextPredictor {
    pub fn get_stats(&self) -> PredictionStats {
        let avg_retrieval = if self.retrieval_history.is_empty() {
            0.0
        } else {
            self.retrieval_history
                .iter()
                .map(|v| v.len() as f32)
                .sum::<f32>()
                / self.retrieval_history.len() as f32
        };

        PredictionStats {
            history_size: self.embedding_history.len(),
            avg_retrieval_count: avg_retrieval,
            flow_pattern: self.get_flow_pattern(),
            top_intents: Vec::new(),
        }
    }
}
