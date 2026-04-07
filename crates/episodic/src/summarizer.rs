use rememnemosyne_core::{EntityRef, EntityType, Result};
use serde::{Deserialize, Serialize};

use crate::artifact::{Decision, Episode, Exchange};

/// Configuration for the summarizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizerConfig {
    pub max_summary_length: usize,
    pub include_decisions: bool,
    pub include_entities: bool,
    pub include_topics: bool,
    pub compression_ratio: f32,
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        Self {
            max_summary_length: 500,
            include_decisions: true,
            include_entities: true,
            include_topics: true,
            compression_ratio: 0.3,
        }
    }
}

/// Summarizer for episodic memory - extracts key information from exchanges
pub struct EpisodeSummarizer {
    config: SummarizerConfig,
}

impl EpisodeSummarizer {
    pub fn new(config: SummarizerConfig) -> Self {
        Self { config }
    }

    /// Summarize an episode by extracting key information
    pub fn summarize_episode(&self, episode: &Episode) -> Result<EpisodeSummary> {
        let mut summary = EpisodeSummary::new(episode.id);

        // Extract main topic from exchanges
        summary.main_topic = self.extract_main_topic(&episode.exchanges);

        // Generate summary text
        summary.summary_text = self.generate_summary_text(episode);

        // Extract key points
        summary.key_points = self.extract_key_points(&episode.exchanges);

        // Include decisions
        if self.config.include_decisions {
            summary.decisions = episode.key_decisions.clone();
        }

        // Include entities
        if self.config.include_entities {
            summary.entities = episode.entities_mentioned.clone();
        }

        // Include topics
        if self.config.include_topics {
            summary.topics = episode.topics.clone();
        }

        // Compute metrics
        summary.importance = episode.importance;
        summary.engagement_score = episode.compute_engagement_score();

        Ok(summary)
    }

    /// Extract the main topic from exchanges
    fn extract_main_topic(&self, exchanges: &[Exchange]) -> String {
        if exchanges.is_empty() {
            return "No content".to_string();
        }

        // Simple extraction: take the first user query and extract key terms
        if let Some(first_user) = exchanges
            .iter()
            .find(|e| e.role == crate::artifact::ExchangeRole::User)
        {
            let words: Vec<&str> = first_user
                .content
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .take(5)
                .collect();
            words.join(" ")
        } else {
            "Unknown topic".to_string()
        }
    }

    /// Generate summary text from episode
    fn generate_summary_text(&self, episode: &Episode) -> String {
        let mut parts = Vec::new();

        // Title
        if !episode.title.is_empty() {
            parts.push(format!("Topic: {}", episode.title));
        }

        // Exchange summary
        if !episode.exchanges.is_empty() {
            let user_count = episode
                .exchanges
                .iter()
                .filter(|e| e.role == crate::artifact::ExchangeRole::User)
                .count();
            let assistant_count = episode
                .exchanges
                .iter()
                .filter(|e| e.role == crate::artifact::ExchangeRole::Assistant)
                .count();
            parts.push(format!(
                "{} user queries, {} responses",
                user_count, assistant_count
            ));
        }

        // Key decisions
        if !episode.key_decisions.is_empty() {
            let decisions_text: Vec<String> = episode
                .key_decisions
                .iter()
                .take(3)
                .map(|d| format!("- {}", d.description))
                .collect();
            parts.push(format!("Key decisions:\n{}", decisions_text.join("\n")));
        }

        // Entities
        if !episode.entities_mentioned.is_empty() {
            let entity_names: Vec<String> = episode
                .entities_mentioned
                .iter()
                .map(|e| e.name.clone())
                .collect();
            parts.push(format!("Entities: {}", entity_names.join(", ")));
        }

        let result = parts.join("\n");
        if result.len() > self.config.max_summary_length {
            result[..self.config.max_summary_length].to_string() + "..."
        } else {
            result
        }
    }

    /// Extract key points from exchanges
    fn extract_key_points(&self, exchanges: &[Exchange]) -> Vec<String> {
        let mut key_points = Vec::new();

        for exchange in exchanges {
            // Extract from assistant responses (contain answers/insights)
            if exchange.role == crate::artifact::ExchangeRole::Assistant {
                if let Some(ref response) = exchange.response {
                    if response.len() > 50 {
                        // Extract first sentence or paragraph
                        let point = response
                            .split('.')
                            .next()
                            .unwrap_or(response)
                            .trim()
                            .to_string();
                        if !point.is_empty() && point.len() > 20 {
                            key_points.push(point);
                        }
                    }
                }
            }

            // Extract from intents
            if let Some(ref intent) = exchange.intent {
                key_points.push(format!("Intent: {}", intent));
            }
        }

        // Limit to top 5 key points
        key_points.truncate(5);
        key_points
    }

    /// Extract entities from text (simple NER-like extraction)
    pub fn extract_entities(&self, text: &str) -> Vec<EntityRef> {
        let mut entities = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Capitalized words (likely proper nouns)
        for word in text.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if clean.len() > 2
                && clean
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
            {
                if seen.insert(clean.to_string()) {
                    entities.push(EntityRef {
                        id: uuid::Uuid::new_v4(),
                        name: clean.to_string(),
                        entity_type: EntityType::Concept,
                        relevance: 0.5,
                    });
                }
            }
        }

        entities
    }

    /// Extract decisions from exchanges
    pub fn extract_decisions(&self, exchanges: &[Exchange]) -> Vec<Decision> {
        let mut decisions = Vec::new();

        // Look for decision patterns in responses
        let decision_keywords = [
            "decided to",
            "chose",
            "selected",
            "we'll",
            "let's",
            "going to",
            "will use",
            "will implement",
            "chosen",
        ];

        for exchange in exchanges {
            if exchange.role == crate::artifact::ExchangeRole::Assistant {
                if let Some(ref response) = exchange.response {
                    let response_lower = response.to_lowercase();
                    for keyword in &decision_keywords {
                        if response_lower.contains(keyword) {
                            // Extract the decision sentence
                            if let Some(sentence) = response
                                .split('.')
                                .find(|s| s.to_lowercase().contains(keyword))
                            {
                                let decision = Decision::new(
                                    sentence.trim().to_string(),
                                    exchange.content.clone(),
                                    sentence.trim().to_string(),
                                );
                                decisions.push(decision);
                                break;
                            }
                        }
                    }
                }
            }
        }

        decisions
    }

    /// Merge multiple episode summaries
    pub fn merge_summaries(&self, summaries: &[EpisodeSummary]) -> Option<EpisodeSummary> {
        if summaries.is_empty() {
            return None;
        }

        let mut merged = summaries[0].clone();

        for summary in &summaries[1..] {
            merged.key_points.extend(summary.key_points.clone());
            merged.decisions.extend(summary.decisions.clone());
            merged.entities.extend(summary.entities.clone());
            merged.topics.extend(summary.topics.clone());
            merged.importance = merged.importance.max(summary.importance);
            merged.engagement_score += summary.engagement_score;
        }

        // Deduplicate
        merged.key_points.dedup();
        merged.topics.dedup();

        Some(merged)
    }
}

/// Summary of an episode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeSummary {
    pub episode_id: uuid::Uuid,
    pub main_topic: String,
    pub summary_text: String,
    pub key_points: Vec<String>,
    pub decisions: Vec<Decision>,
    pub entities: Vec<EntityRef>,
    pub topics: Vec<String>,
    pub importance: f32,
    pub engagement_score: f32,
}

impl EpisodeSummary {
    pub fn new(episode_id: uuid::Uuid) -> Self {
        Self {
            episode_id,
            main_topic: String::new(),
            summary_text: String::new(),
            key_points: Vec::new(),
            decisions: Vec::new(),
            entities: Vec::new(),
            topics: Vec::new(),
            importance: 0.5,
            engagement_score: 0.0,
        }
    }

    pub fn to_context_string(&self) -> String {
        let mut parts = Vec::new();

        if !self.main_topic.is_empty() {
            parts.push(format!("Topic: {}", self.main_topic));
        }

        if !self.summary_text.is_empty() {
            parts.push(format!("Summary: {}", self.summary_text));
        }

        if !self.key_points.is_empty() {
            parts.push("Key points:".to_string());
            for point in &self.key_points {
                parts.push(format!("  - {}", point));
            }
        }

        if !self.decisions.is_empty() {
            parts.push("Decisions made:".to_string());
            for decision in &self.decisions {
                parts.push(format!("  - {}", decision.description));
            }
        }

        parts.join("\n")
    }
}
