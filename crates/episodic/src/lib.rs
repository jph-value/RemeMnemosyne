pub mod artifact;
pub mod checkpoint;
pub mod session;
pub mod store;
pub mod summarizer;

pub use artifact::*;
pub use checkpoint::*;
pub use session::*;
pub use store::*;
pub use summarizer::*;

#[cfg(test)]
mod tests {
    use super::*;
    use rememnemosyne_core::SessionId;

    #[test]
    fn test_episode_creation() {
        let session_id = SessionId::new_v4();
        let episode = Episode::new(session_id, "Test Episode");

        assert_eq!(episode.title, "Test Episode");
        assert!(episode.exchanges.is_empty());
    }

    #[test]
    fn test_episode_add_exchange() {
        let session_id = SessionId::new_v4();
        let mut episode = Episode::new(session_id, "Test");

        let exchange = Exchange::new(ExchangeRole::User, "Hello");
        episode.add_exchange(exchange);

        assert_eq!(episode.exchanges.len(), 1);
    }

    #[test]
    fn test_episode_add_decision() {
        let session_id = SessionId::new_v4();
        let mut episode = Episode::new(session_id, "Test");

        let decision = Decision::new("Use Rust", "Need speed", "Rust");
        episode.add_decision(decision);

        assert_eq!(episode.key_decisions.len(), 1);
    }

    #[test]
    fn test_exchange_creation() {
        let exchange = Exchange::new(ExchangeRole::User, "Content")
            .with_response("Response")
            .with_intent("question");

        assert_eq!(exchange.role, ExchangeRole::User);
        assert_eq!(exchange.response, Some("Response".to_string()));
    }

    #[test]
    fn test_session_creation() {
        let session = MemorySession::new("Test Session");
        assert_eq!(session.name, "Test Session");
        assert!(session.status == SessionStatus::Active);
    }

    #[test]
    fn test_session_manager() {
        let mut manager = SessionManager::new();
        let id = manager.create_session("Test");

        assert!(manager.get_session(&id).is_some());
        assert!(manager.get_active_session().is_some());
    }
}
