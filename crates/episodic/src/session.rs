use chrono::{DateTime, Utc};
use mnemosyne_core::SessionId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::artifact::{ConversationContext, Episode, Exchange};

/// A memory session groups related episodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySession {
    pub id: SessionId,
    pub name: String,
    pub description: String,
    pub episodes: Vec<Episode>,
    pub context: ConversationContext,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub status: SessionStatus,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl MemorySession {
    pub fn new(name: impl Into<String>) -> Self {
        let id = Uuid::new_v4();
        let now = Utc::now();
        Self {
            id,
            name: name.into(),
            description: String::new(),
            episodes: Vec::new(),
            context: ConversationContext::new(id),
            created_at: now,
            updated_at: now,
            status: SessionStatus::Active,
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_episode(&mut self, episode: Episode) {
        self.context.episode_id = Some(episode.id);
        self.episodes.push(episode);
        self.updated_at = Utc::now();
    }

    pub fn get_current_episode(&self) -> Option<&Episode> {
        self.episodes.last()
    }

    pub fn get_current_episode_mut(&mut self) -> Option<&mut Episode> {
        self.episodes.last_mut()
    }

    pub fn close(&mut self) {
        self.status = SessionStatus::Closed;
        self.updated_at = Utc::now();
    }

    pub fn archive(&mut self) {
        self.status = SessionStatus::Archived;
        self.updated_at = Utc::now();
    }

    pub fn get_summary(&self) -> String {
        if self.episodes.is_empty() {
            return format!("Session '{}': No episodes yet", self.name);
        }

        let episode_count = self.episodes.len();
        let total_exchanges: usize = self.episodes.iter().map(|e| e.exchanges.len()).sum();
        let total_decisions: usize = self.episodes.iter().map(|e| e.key_decisions.len()).sum();

        format!(
            "Session '{}': {} episodes, {} exchanges, {} key decisions",
            self.name, episode_count, total_exchanges, total_decisions
        )
    }

    pub fn get_all_entities(&self) -> Vec<&mnemosyne_core::EntityRef> {
        use std::collections::HashSet;

        let mut seen = HashSet::new();
        let mut entities = Vec::new();

        for episode in &self.episodes {
            for entity in &episode.entities_mentioned {
                if seen.insert(&entity.id) {
                    entities.push(entity);
                }
            }
        }

        entities
    }

    pub fn get_all_decisions(&self) -> Vec<&crate::artifact::Decision> {
        self.episodes
            .iter()
            .flat_map(|e| e.key_decisions.iter())
            .collect()
    }

    pub fn compute_total_engagement(&self) -> f32 {
        if self.episodes.is_empty() {
            return 0.0;
        }

        self.episodes
            .iter()
            .map(|e| e.compute_engagement_score())
            .sum::<f32>()
            / self.episodes.len() as f32
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStatus {
    Active,
    Closed,
    Archived,
}

/// Session manager for handling multiple concurrent sessions
pub struct SessionManager {
    sessions: HashMap<SessionId, MemorySession>,
    active_session: Option<SessionId>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            active_session: None,
        }
    }

    pub fn create_session(&mut self, name: impl Into<String>) -> SessionId {
        let session = MemorySession::new(name);
        let id = session.id;
        self.sessions.insert(id, session);
        self.active_session = Some(id);
        id
    }

    pub fn get_session(&self, id: &SessionId) -> Option<&MemorySession> {
        self.sessions.get(id)
    }

    pub fn get_session_mut(&mut self, id: &SessionId) -> Option<&mut MemorySession> {
        self.sessions.get_mut(id)
    }

    pub fn set_active_session(&mut self, id: SessionId) -> bool {
        if self.sessions.contains_key(&id) {
            self.active_session = Some(id);
            true
        } else {
            false
        }
    }

    pub fn get_active_session(&self) -> Option<&MemorySession> {
        self.active_session.and_then(|id| self.sessions.get(&id))
    }

    pub fn get_active_session_mut(&mut self) -> Option<&mut MemorySession> {
        self.active_session
            .and_then(|id| self.sessions.get_mut(&id))
    }

    pub fn close_session(&mut self, id: &SessionId) -> bool {
        if let Some(session) = self.sessions.get_mut(id) {
            session.close();
            if self.active_session == Some(*id) {
                self.active_session = None;
            }
            true
        } else {
            false
        }
    }

    pub fn archive_session(&mut self, id: &SessionId) -> bool {
        if let Some(session) = self.sessions.get_mut(id) {
            session.archive();
            if self.active_session == Some(*id) {
                self.active_session = None;
            }
            true
        } else {
            false
        }
    }

    pub fn list_sessions(&self) -> Vec<&MemorySession> {
        self.sessions.values().collect()
    }

    pub fn list_active_sessions(&self) -> Vec<&MemorySession> {
        self.sessions
            .values()
            .filter(|s| s.status == SessionStatus::Active)
            .collect()
    }

    pub fn search_sessions(&self, query: &str) -> Vec<&MemorySession> {
        let query_lower = query.to_lowercase();
        self.sessions
            .values()
            .filter(|s| {
                s.name.to_lowercase().contains(&query_lower)
                    || s.description.to_lowercase().contains(&query_lower)
                    || s.tags
                        .iter()
                        .any(|t| t.to_lowercase().contains(&query_lower))
            })
            .collect()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}
