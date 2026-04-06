use async_trait::async_trait;
use dashmap::DashMap;
use rememnemosyne_core::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::artifact::{Episode, Exchange, EpisodicArtifact, EpisodicArtifactType, ConversationContext};
use crate::session::{MemorySession, SessionManager};
use crate::summarizer::{EpisodeSummarizer, SummarizerConfig, EpisodeSummary};

/// Configuration for episodic memory store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemoryConfig {
    pub max_exchanges_per_episode: usize,
    pub auto_summarize: bool,
    pub summarize_config: SummarizerConfig,
    pub max_episodes_per_session: usize,
    pub importance_threshold: f32,
}

impl Default for EpisodicMemoryConfig {
    fn default() -> Self {
        Self {
            max_exchanges_per_episode: 20,
            auto_summarize: true,
            summarize_config: SummarizerConfig::default(),
            max_episodes_per_session: 100,
            importance_threshold: 0.3,
        }
    }
}

/// Episodic memory store - inspired by mempalace
/// 
/// Stores conversation episodes, exchanges, decisions, and summaries
/// for rich contextual recall.
pub struct EpisodicMemoryStore {
    config: EpisodicMemoryConfig,
    /// All episodes indexed by ID
    episodes: Arc<DashMap<Uuid, Episode>>,
    /// Session manager
    session_manager: Arc<RwLock<SessionManager>>,
    /// Summarizer
    summarizer: Arc<EpisodeSummarizer>,
    /// Episode summaries cache
    summaries: Arc<DashMap<Uuid, EpisodeSummary>>,
    /// Memory artifacts from episodes
    artifacts: Arc<DashMap<MemoryId, EpisodicArtifact>>,
    /// Exchange history for context
    exchange_history: Arc<DashMap<Uuid, Vec<Exchange>>>,
}

impl EpisodicMemoryStore {
    pub fn new(config: EpisodicMemoryConfig) -> Self {
        let summarizer = Arc::new(EpisodeSummarizer::new(config.summarize_config.clone()));

        Self {
            config,
            episodes: Arc::new(DashMap::new()),
            session_manager: Arc::new(RwLock::new(SessionManager::new())),
            summarizer,
            summaries: Arc::new(DashMap::new()),
            artifacts: Arc::new(DashMap::new()),
            exchange_history: Arc::new(DashMap::new()),
        }
    }

    /// Create a new session
    pub async fn create_session(&self, name: impl Into<String>) -> SessionId {
        let mut manager = self.session_manager.write().await;
        manager.create_session(name)
    }

    /// Start a new episode in the active session
    pub async fn start_episode(
        &self,
        session_id: SessionId,
        title: impl Into<String>,
    ) -> Result<Uuid> {
        let mut manager = self.session_manager.write().await;
        let session = manager.get_session_mut(&session_id)
            .ok_or_else(|| MemoryError::NotFound("Session not found".into()))?;

        let episode = Episode::new(session_id, title);
        let episode_id = episode.id;
        session.add_episode(episode);

        Ok(episode_id)
    }

    /// Add an exchange to an episode
    pub async fn add_exchange(
        &self,
        episode_id: Uuid,
        exchange: Exchange,
    ) -> Result<()> {
        // Add to episode
        let mut episode = self.episodes.get_mut(&episode_id)
            .ok_or_else(|| MemoryError::NotFound("Episode not found".into()))?;
        
        episode.add_exchange(exchange.clone());

        // Update exchange history
        self.exchange_history
            .entry(episode_id)
            .or_insert_with(Vec::new)
            .push(exchange);

        // Auto-summarize if enabled and threshold reached
        if self.config.auto_summarize {
            let exchange_count = self.exchange_history
                .get(&episode_id)
                .map(|h| h.len())
                .unwrap_or(0);
            
            if exchange_count >= self.config.max_exchanges_per_episode {
                self.summarize_episode(episode_id).await?;
            }
        }

        Ok(())
    }

    /// Add a decision to an episode
    pub async fn add_decision(
        &self,
        episode_id: Uuid,
        decision: crate::artifact::Decision,
    ) -> Result<()> {
        let mut episode = self.episodes.get_mut(&episode_id)
            .ok_or_else(|| MemoryError::NotFound("Episode not found".into()))?;
        
        episode.add_decision(decision);
        Ok(())
    }

    /// Summarize an episode
    pub async fn summarize_episode(&self, episode_id: Uuid) -> Result<EpisodeSummary> {
        let episode = self.episodes.get(&episode_id)
            .ok_or_else(|| MemoryError::NotFound("Episode not found".into()))?;
        
        let summary = self.summarizer.summarize_episode(&episode)?;
        self.summaries.insert(episode_id, summary.clone());
        
        Ok(summary)
    }

    /// Get an episode by ID
    pub async fn get_episode(&self, episode_id: &Uuid) -> Option<Episode> {
        self.episodes.get(episode_id).map(|e| e.clone())
    }

    /// Get episode summary
    pub async fn get_summary(&self, episode_id: &Uuid) -> Option<EpisodeSummary> {
        self.summaries.get(episode_id).map(|s| s.clone())
    }

    /// Get session episodes
    pub async fn get_session_episodes(&self, session_id: &SessionId) -> Vec<Episode> {
        self.episodes
            .iter()
            .filter(|e| e.session_id == *session_id)
            .map(|e| e.clone())
            .collect()
    }

    /// Get conversation context for an episode
    pub async fn get_context(&self, episode_id: &Uuid) -> Option<ConversationContext> {
        let episode = self.episodes.get(episode_id)?;
        let exchanges = self.exchange_history.get(episode_id)?;

        let mut context = ConversationContext::new(episode.session_id);
        context.episode_id = Some(*episode_id);
        for exchange in exchanges.iter() {
            context.add_exchange(exchange.clone());
        }

        Some(context)
    }

    /// Search episodes by topic or content
    pub async fn search_episodes(&self, query: &str) -> Vec<Episode> {
        let query_lower = query.to_lowercase();
        
        self.episodes
            .iter()
            .filter(|e| {
                e.title.to_lowercase().contains(&query_lower)
                    || e.summary.to_lowercase().contains(&query_lower)
                    || e.topics.iter().any(|t| t.to_lowercase().contains(&query_lower))
            })
            .map(|e| e.clone())
            .collect()
    }

    /// Get recent episodes
    pub async fn get_recent_episodes(&self, limit: usize) -> Vec<Episode> {
        let mut episodes: Vec<Episode> = self.episodes
            .iter()
            .map(|e| e.clone())
            .collect();
        
        episodes.sort_by(|a, b| b.end_time.cmp(&a.end_time));
        episodes.truncate(limit);
        episodes
    }

    /// Get high importance episodes
    pub async fn get_important_episodes(&self, min_importance: f32) -> Vec<Episode> {
        self.episodes
            .iter()
            .filter(|e| e.importance >= min_importance)
            .map(|e| e.clone())
            .collect()
    }

    /// Get all decisions across episodes
    pub async fn get_all_decisions(&self) -> Vec<crate::artifact::Decision> {
        self.episodes
            .iter()
            .flat_map(|e| e.key_decisions.clone())
            .collect()
    }

    /// Compute overall session summary
    pub async fn get_session_summary(&self, session_id: &SessionId) -> Option<String> {
        let manager = self.session_manager.read().await;
        let session = manager.get_session(session_id)?;
        Some(session.get_summary())
    }

    /// Archive old episodes
    pub async fn archive_old_episodes(&self, days_old: i64) -> Result<usize> {
        use chrono::Duration;
        
        let cutoff = chrono::Utc::now() - Duration::days(days_old);
        let mut archived = 0;

        let to_archive: Vec<Uuid> = self.episodes
            .iter()
            .filter(|e| e.end_time < cutoff)
            .map(|e| *e.key())
            .collect();

        for id in to_archive {
            if let Some((_, mut episode)) = self.episodes.remove(&id) {
                // Store summary before removal
                if let Ok(summary) = self.summarizer.summarize_episode(&episode) {
                    self.summaries.insert(id, summary);
                }
                archived += 1;
            }
        }

        Ok(archived)
    }
}

#[async_trait]
impl MemoryStore for EpisodicMemoryStore {
    async fn store(&self, artifact: MemoryArtifact) -> Result<MemoryId> {
        // Convert to episodic artifact
        let episodic_artifact = EpisodicArtifact {
            memory_id: artifact.id,
            episode_id: Uuid::new_v4(),
            artifact_type: EpisodicArtifactType::Summary,
            content: artifact.summary.clone(),
            embedding: artifact.embedding.clone(),
            timestamp: artifact.timestamp,
            importance: artifact.importance as u8 as f32 / 4.0,
            tags: artifact.tags.clone(),
        };

        self.artifacts.insert(artifact.id, episodic_artifact);
        Ok(artifact.id)
    }

    async fn get(&self, id: &MemoryId) -> Result<Option<MemoryArtifact>> {
        if let Some(episodic) = self.artifacts.get(id) {
            // Convert back to MemoryArtifact
            let artifact = MemoryArtifact::new(
                MemoryType::Episodic,
                episodic.content.clone(),
                episodic.content.clone(),
                episodic.embedding.clone(),
                MemoryTrigger::UserInput,
            ).with_importance(match episodic.importance {
                x if x >= 0.75 => Importance::Critical,
                x if x >= 0.5 => Importance::High,
                x if x >= 0.25 => Importance::Medium,
                _ => Importance::Low,
            });
            
            Ok(Some(artifact))
        } else {
            Ok(None)
        }
    }

    async fn query(&self, query: &MemoryQuery) -> Result<Vec<MemoryArtifact>> {
        let mut results = Vec::new();

        // Search by text
        if let Some(ref text) = query.text {
            let episodes = self.search_episodes(text).await;
            for episode in episodes {
                let artifact = MemoryArtifact::new(
                    MemoryType::Episodic,
                    episode.title.clone(),
                    episode.summary.clone(),
                    Vec::new(), // Would need embedding
                    MemoryTrigger::UserInput,
                );
                results.push(artifact);
            }
        }

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    async fn delete(&self, id: &MemoryId) -> Result<bool> {
        Ok(self.artifacts.remove(id).is_some())
    }

    async fn update(&self, artifact: MemoryArtifact) -> Result<()> {
        if let Some(mut episodic) = self.artifacts.get_mut(&artifact.id) {
            episodic.content = artifact.summary;
            episodic.embedding = artifact.embedding;
        }
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.artifacts.len())
    }

    async fn clear(&self) -> Result<()> {
        self.episodes.clear();
        self.summaries.clear();
        self.artifacts.clear();
        self.exchange_history.clear();
        Ok(())
    }

    async fn list_ids(&self) -> Result<Vec<MemoryId>> {
        Ok(self.artifacts.iter().map(|e| *e.key()).collect())
    }
}
