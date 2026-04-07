use rememnemosyne_core::*;
use rememnemosyne_graph::entity::GraphEntity;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for context builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBuilderConfig {
    pub max_tokens: usize,
    pub max_memories: usize,
    pub max_entities: usize,
    pub include_relationships: bool,
    pub include_timeline: bool,
    pub prioritize_decisions: bool,
    pub include_summaries: bool,
    pub format_strategy: ContextFormatStrategy,
}

/// Strategies for formatting memory context into prompts.
/// Different strategies work better with different model sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextFormatStrategy {
    /// Memory as brief inline hints in user message (best for small models)
    InlineHints,
    /// Structured memory block before system prompt (balanced)
    SystemPrefix,
    /// Structured memory block inside system prompt (default, best for large models)
    ContextBlock,
    /// Memory as few-shot example pairs (best for instruction-following models)
    FewShot,
}

impl Default for ContextBuilderConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4000,
            max_memories: 20,
            max_entities: 10,
            include_relationships: true,
            include_timeline: false,
            prioritize_decisions: true,
            include_summaries: true,
            format_strategy: ContextFormatStrategy::ContextBlock,
        }
    }
}

/// Context builder for assembling LLM-ready context
pub struct ContextBuilderEngine {
    config: ContextBuilderConfig,
}

impl ContextBuilderEngine {
    pub fn new(config: ContextBuilderConfig) -> Self {
        Self { config }
    }

    /// Get reference to configuration
    pub fn config(&self) -> &ContextBuilderConfig {
        &self.config
    }

    /// Create a context builder optimized for small models (2B-4B parameters)
    pub fn for_small_model() -> Self {
        Self {
            config: ContextBuilderConfig {
                max_tokens: 500,
                max_memories: 2,
                max_entities: 3,
                include_relationships: false,
                include_timeline: false,
                prioritize_decisions: true,
                include_summaries: false,
                format_strategy: ContextFormatStrategy::InlineHints,
            },
        }
    }

    /// Create a context builder optimized for medium models (7B-13B parameters)
    pub fn for_medium_model() -> Self {
        Self {
            config: ContextBuilderConfig {
                max_tokens: 1500,
                max_memories: 5,
                max_entities: 5,
                include_relationships: false,
                include_timeline: false,
                prioritize_decisions: true,
                include_summaries: true,
                format_strategy: ContextFormatStrategy::SystemPrefix,
            },
        }
    }

    /// Create a context builder optimized for large models (30B+ parameters)
    pub fn for_large_model() -> Self {
        Self {
            config: ContextBuilderConfig {
                max_tokens: 4000,
                max_memories: 10,
                max_entities: 10,
                include_relationships: true,
                include_timeline: false,
                prioritize_decisions: true,
                include_summaries: true,
                format_strategy: ContextFormatStrategy::ContextBlock,
            },
        }
    }

    /// Build context bundle from memory response
    pub fn build_context(
        &self,
        response: &crate::router::MemoryResponse,
        extra_entities: Vec<GraphEntity>,
        _decisions: Vec<rememnemosyne_episodic::artifact::Decision>,
    ) -> ContextBundle {
        let mut bundle = ContextBundle::new();

        // Add memories (limited by config)
        let mut current_tokens = 0;
        let _added_memories = 0;

        for result in response.results.iter().take(self.config.max_memories) {
            let memory_tokens = self.estimate_tokens(&result.memory.summary);

            if current_tokens + memory_tokens > self.config.max_tokens {
                break;
            }

            bundle.add_memory(result.memory.clone(), result.relevance);
            current_tokens += memory_tokens;
        }

        // Add entities from response first, then extras
        for graph_entity in response.entities.iter().take(self.config.max_entities) {
            let entity = self.graph_entity_to_entity(graph_entity);
            bundle.entities.push(entity);
        }
        let remaining_entity_slots = self.config.max_entities.saturating_sub(bundle.entities.len());
        for graph_entity in extra_entities.into_iter().take(remaining_entity_slots) {
            let entity = self.graph_entity_to_entity(&graph_entity);
            bundle.entities.push(entity);
        }

        // Build summaries
        if self.config.include_summaries {
            bundle.summaries = self.build_summaries(&response.results);
        }

        bundle.total_tokens_estimate = current_tokens;
        bundle
    }

    /// Convert GraphEntity to Entity
    fn graph_entity_to_entity(&self, graph_entity: &GraphEntity) -> Entity {
        Entity {
            id: graph_entity.id,
            name: graph_entity.name.clone(),
            entity_type: graph_entity.entity_type.clone(),
            description: graph_entity.description.clone(),
            embedding: graph_entity.embedding.clone(),
            attributes: graph_entity.attributes.clone(),
            created_at: graph_entity.created_at,
            updated_at: graph_entity.updated_at,
            mention_count: graph_entity.mention_count,
        }
    }

    /// Build formatted context string for LLM
    pub fn format_context(&self, bundle: &ContextBundle) -> String {
        match self.config.format_strategy {
            ContextFormatStrategy::InlineHints => self.format_inline_hints(bundle),
            ContextFormatStrategy::SystemPrefix => self.format_system_prefix(bundle),
            ContextFormatStrategy::ContextBlock => self.format_context_block(bundle),
            ContextFormatStrategy::FewShot => self.format_few_shot(bundle),
        }
    }

    /// Inline hints format: memory as brief hints within user message
    fn format_inline_hints(&self, bundle: &ContextBundle) -> String {
        let mut parts = Vec::new();

        if !bundle.memories.is_empty() {
            let hints: Vec<String> = bundle
                .memories
                .iter()
                .take(3)
                .map(|m| m.summary.clone())
                .collect();
            parts.push(format!("(Consider: {})", hints.join("; ")));
        }

        parts.join("\n")
    }

    /// System prefix format: memory block before system prompt
    fn format_system_prefix(&self, bundle: &ContextBundle) -> String {
        let mut parts = Vec::new();

        if !bundle.memories.is_empty() {
            parts.push("## Available Memories".to_string());
            for memory in bundle.memories.iter().take(self.config.max_memories) {
                parts.push(format!("- {}", memory.summary));
            }
            parts.push(String::new());
        }

        parts.join("\n")
    }

    /// Context block format: structured memory block (default)
    fn format_context_block(&self, bundle: &ContextBundle) -> String {
        let mut parts = Vec::new();

        // Header
        parts.push("## Memory Context\n".to_string());

        // Summaries
        if !bundle.summaries.is_empty() {
            parts.push("### Key Summaries".to_string());
            for summary in &bundle.summaries {
                parts.push(format!("- {}", summary));
            }
            parts.push(String::new());
        }

        // Relevant memories
        if !bundle.memories.is_empty() {
            parts.push("### Relevant Memories".to_string());
            for memory in bundle.memories.iter().take(10) {
                let relevance = bundle
                    .relevance_scores
                    .get(&memory.id)
                    .map(|r| format!("{:.0}%", r * 100.0))
                    .unwrap_or_default();
                parts.push(format!(
                    "- [{}] {}: {}",
                    relevance,
                    memory.summary,
                    memory.content.chars().take(200).collect::<String>()
                ));
            }
            parts.push(String::new());
        }

        // Entities
        if !bundle.entities.is_empty() {
            parts.push("### Related Entities".to_string());
            for entity in bundle.entities.iter().take(5) {
                parts.push(format!(
                    "- **{}** ({}): {}",
                    entity.name,
                    format!("{:?}", entity.entity_type),
                    entity.description.chars().take(100).collect::<String>()
                ));
            }
            parts.push(String::new());
        }

        // Decisions - extract memories with Decision trigger
        let decision_memories: Vec<_> = bundle
            .memories
            .iter()
            .filter(|m| matches!(m.trigger, MemoryTrigger::Decision))
            .collect();

        if !decision_memories.is_empty() {
            parts.push("### Key Decisions".to_string());
            for decision in decision_memories.iter().take(5) {
                parts.push(format!("- {}", decision.summary));
            }
            parts.push(String::new());
        }

        parts.join("\n")
    }

    /// Few-shot format: memory as example Q&A pairs
    fn format_few_shot(&self, bundle: &ContextBundle) -> String {
        let mut parts = Vec::new();

        if !bundle.memories.is_empty() {
            parts.push("## Relevant Examples\n".to_string());
            for memory in bundle.memories.iter().take(3) {
                parts.push(format!("Q: {}", memory.summary));
                parts.push(format!(
                    "A: {}",
                    memory.content.chars().take(200).collect::<String>()
                ));
                parts.push(String::new());
            }
        }

        parts.join("\n")
    }

    /// Prune context to fit token limit
    pub fn prune_to_token_limit(&self, bundle: &mut ContextBundle, max_tokens: usize) {
        bundle.truncate_to_token_limit(max_tokens);
    }

    /// Merge multiple context bundles
    pub fn merge_bundles(&self, bundles: Vec<ContextBundle>) -> ContextBundle {
        let mut merged = ContextBundle::new();

        for bundle in bundles {
            merged.merge(bundle);
        }

        // Deduplicate
        merged.memories.dedup_by(|a, b| a.id == b.id);
        merged.entities.dedup_by(|a, b| a.id == b.id);

        // Sort by relevance
        merged.memories.sort_by(|a, b| {
            let score_a = merged.relevance_scores.get(&a.id).unwrap_or(&0.0);
            let score_b = merged.relevance_scores.get(&b.id).unwrap_or(&0.0);
            score_b
                .partial_cmp(score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        merged
    }

    // Private helper methods

    fn estimate_tokens(&self, text: &str) -> usize {
        // Rough estimation: ~4 characters per token
        text.len() / 4
    }

    fn build_summaries(&self, results: &[crate::router::MemoryResult]) -> Vec<String> {
        let mut summaries = Vec::new();

        // Group by memory type
        let mut by_type: HashMap<MemoryType, Vec<&crate::router::MemoryResult>> = HashMap::new();
        for result in results {
            by_type
                .entry(result.source)
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (mem_type, memories) in by_type {
            if !memories.is_empty() {
                let summary = format!(
                    "{} memory: {} items, most recent: {}",
                    mem_type,
                    memories.len(),
                    memories
                        .first()
                        .map(|m| m.memory.summary.as_str())
                        .unwrap_or("none")
                );
                summaries.push(summary);
            }
        }

        summaries
    }
}

/// Prompt template for context injection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub template: String,
    pub context_placeholder: String,
    pub query_placeholder: String,
}

impl PromptTemplate {
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
            context_placeholder: "{context}".to_string(),
            query_placeholder: "{query}".to_string(),
        }
    }

    pub fn render(&self, context: &str, query: &str) -> String {
        self.template
            .replace(&self.context_placeholder, context)
            .replace(&self.query_placeholder, query)
    }

    pub fn default_agent_template() -> Self {
        Self::new(
            "You are a helpful AI assistant with access to a memory system. \
Use the following context from your memory to help answer the user's question.\n\n\
## Memory Context\n{context}\n\n\
## User Query\n{query}\n\n\
## Instructions\n\
- Use the memory context to provide accurate and relevant responses\n\
- If the memory doesn't contain relevant information, say so\n\
- Be specific and cite which memories you're referencing\n",
        )
    }
}
