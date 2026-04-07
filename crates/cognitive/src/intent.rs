use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Intent detection for understanding user queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentDetector {
    /// Intent patterns with associated keywords
    patterns: HashMap<String, Vec<String>>,
    /// Threshold for intent matching
    threshold: f32,
}

impl IntentDetector {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // Search/retrieval intents
        patterns.insert(
            "search".to_string(),
            vec![
                "find".to_string(),
                "search".to_string(),
                "look for".to_string(),
                "where is".to_string(),
                "show me".to_string(),
                "retrieve".to_string(),
            ],
        );

        // Memory creation intents
        patterns.insert(
            "remember".to_string(),
            vec![
                "remember".to_string(),
                "save".to_string(),
                "store".to_string(),
                "note".to_string(),
                "keep".to_string(),
                "record".to_string(),
            ],
        );

        // Recall intents
        patterns.insert(
            "recall".to_string(),
            vec![
                "recall".to_string(),
                "what did".to_string(),
                "previously".to_string(),
                "before".to_string(),
                "history".to_string(),
                "past".to_string(),
            ],
        );

        // Analysis intents
        patterns.insert(
            "analyze".to_string(),
            vec![
                "analyze".to_string(),
                "compare".to_string(),
                "difference".to_string(),
                "relationship".to_string(),
                "pattern".to_string(),
                "trend".to_string(),
            ],
        );

        // Decision intents
        patterns.insert(
            "decision".to_string(),
            vec![
                "decide".to_string(),
                "choose".to_string(),
                "select".to_string(),
                "option".to_string(),
                "prefer".to_string(),
                "recommend".to_string(),
            ],
        );

        // Question intents
        patterns.insert(
            "question".to_string(),
            vec![
                "what".to_string(),
                "how".to_string(),
                "why".to_string(),
                "when".to_string(),
                "where".to_string(),
                "who".to_string(),
                "which".to_string(),
                "?".to_string(),
            ],
        );

        // Command intents
        patterns.insert(
            "command".to_string(),
            vec![
                "do".to_string(),
                "make".to_string(),
                "create".to_string(),
                "build".to_string(),
                "generate".to_string(),
                "implement".to_string(),
            ],
        );

        Self {
            patterns,
            threshold: 0.3,
        }
    }

    /// Detect intents from text
    pub fn detect(&self, text: &str) -> Vec<(String, f32)> {
        let text_lower = text.to_lowercase();
        let mut scores: Vec<(String, f32)> = Vec::new();

        for (intent, keywords) in &self.patterns {
            let mut matches = 0;

            for keyword in keywords {
                if text_lower.contains(keyword) {
                    matches += 1;
                }
            }

            if matches > 0 {
                let score = matches as f32 / keywords.len() as f32;
                if score >= self.threshold {
                    scores.push((intent.clone(), score));
                }
            }
        }

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Get primary intent
    pub fn primary_intent(&self, text: &str) -> Option<(String, f32)> {
        self.detect(text).into_iter().next()
    }

    /// Check if text matches a specific intent
    pub fn matches_intent(&self, text: &str, intent: &str) -> bool {
        self.detect(text).iter().any(|(i, _)| i == intent)
    }

    /// Add custom intent pattern
    pub fn add_intent(&mut self, intent: String, keywords: Vec<String>) {
        self.patterns.insert(intent, keywords);
    }

    /// Set matching threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }
}

impl Default for IntentDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Intent classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentResult {
    pub intents: Vec<(String, f32)>,
    pub primary_intent: Option<String>,
    pub confidence: f32,
    pub extracted_entities: Vec<String>,
}

impl IntentResult {
    pub fn from_intents(intents: Vec<(String, f32)>) -> Self {
        let primary_intent = intents.first().map(|(i, _)| i.clone());
        let confidence = intents.first().map(|(_, c)| *c).unwrap_or(0.0);

        Self {
            intents,
            primary_intent,
            confidence,
            extracted_entities: Vec::new(),
        }
    }
}
