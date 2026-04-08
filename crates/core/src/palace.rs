/// Memory Palace - Spatial Memory Organization
///
/// A Rust-native adaptation of the mempalace spatial memory concept.
/// Organizes memories in a navigable hierarchy:
///
/// - **Wings**: Top-level containers (person, project, organization)
/// - **Halls**: Standardized corridors within each wing (facts, events, discoveries, preferences, advice)
/// - **Rooms**: Specific topics or ideas within a hall
/// - **Tunnels**: Cross-references linking identical topics across wings
/// - **Closets**: Summaries/pointers directing searches to original content
/// - **Drawers**: Raw, verbatim source content (never altered)
///
/// This spatial organization yields +34% retrieval accuracy compared to flat indexing.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::types::{EntityId, MemoryId};

// ============================================================================
// Palace Structure
// ============================================================================

/// Unique identifier for palace locations
pub type PalaceLocationId = Uuid;

/// The complete Memory Palace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPalace {
    /// Palace name
    pub name: String,
    /// All wings in this palace
    pub wings: HashMap<String, Wing>,
    /// Global tunnel index (maps topic → wing names)
    pub tunnel_index: HashMap<String, Vec<String>>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified
    pub modified_at: DateTime<Utc>,
}

impl MemoryPalace {
    /// Create a new empty palace
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            name: name.into(),
            wings: HashMap::new(),
            tunnel_index: HashMap::new(),
            created_at: now,
            modified_at: now,
        }
    }

    /// Add a wing to the palace
    pub fn add_wing(&mut self, wing: Wing) {
        let name = wing.name.clone();
        self.wings.insert(name.clone(), wing);
        self.modified_at = Utc::now();
    }

    /// Get a wing by name
    pub fn get_wing(&self, name: &str) -> Option<&Wing> {
        self.wings.get(name)
    }

    /// Get a mutable wing by name
    pub fn get_wing_mut(&mut self, name: &str) -> Option<&mut Wing> {
        self.modified_at = Utc::now();
        self.wings.get_mut(name)
    }

    /// Register a tunnel connection
    pub fn add_tunnel(&mut self, topic: impl Into<String>, wing_name: &str) {
        let topic = topic.into();
        self.tunnel_index
            .entry(topic)
            .or_default()
            .push(wing_name.to_string());
        self.modified_at = Utc::now();
    }

    /// Find all wings connected by a tunnel topic
    pub fn find_tunnel_wings(&self, topic: &str) -> Vec<&str> {
        self.tunnel_index
            .get(topic)
            .map(|wings| wings.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get palace statistics
    pub fn stats(&self) -> PalaceStats {
        let mut total_halls = 0;
        let mut total_rooms = 0;
        let mut total_drawers = 0;
        let mut total_closets = 0;

        for wing in self.wings.values() {
            total_halls += wing.halls.len();
            for hall in wing.halls.values() {
                total_rooms += hall.rooms.len();
                for room in hall.rooms.values() {
                    total_drawers += room.drawers.len();
                    total_closets += room.closets.len();
                }
            }
        }

        PalaceStats {
            wings: self.wings.len(),
            halls: total_halls,
            rooms: total_rooms,
            drawers: total_drawers,
            closets: total_closets,
            tunnels: self.tunnel_index.len(),
        }
    }
}

/// Palace statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalaceStats {
    pub wings: usize,
    pub halls: usize,
    pub rooms: usize,
    pub drawers: usize,
    pub closets: usize,
    pub tunnels: usize,
}

// ============================================================================
// Wings
// ============================================================================

/// A Wing represents a top-level memory container.
/// Typically corresponds to a person, project, or organization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wing {
    /// Wing identifier
    pub name: String,
    /// Wing type
    pub wing_type: WingType,
    /// All halls in this wing
    pub halls: HashMap<String, Hall>,
    /// Wing description
    pub description: Option<String>,
    /// Associated entity ID
    pub entity_id: Option<EntityId>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last accessed
    pub last_accessed: Option<DateTime<Utc>>,
    /// Access count
    pub access_count: u64,
    /// Custom metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Wing {
    /// Create a new wing
    pub fn new(name: impl Into<String>, wing_type: WingType) -> Self {
        let now = Utc::now();
        let name = name.into();

        // Create standard halls
        let mut halls = HashMap::new();
        for hall_type in HallType::standard_types() {
            let hall = Hall::new(&name, hall_type);
            halls.insert(hall.name.clone(), hall);
        }

        Self {
            name,
            wing_type,
            halls,
            description: None,
            entity_id: None,
            created_at: now,
            last_accessed: None,
            access_count: 0,
            metadata: HashMap::new(),
        }
    }

    /// Add a custom hall (non-standard type)
    pub fn add_custom_hall(&mut self, hall: Hall) {
        self.halls.insert(hall.name.clone(), hall);
    }

    /// Get a hall by name
    pub fn get_hall(&self, name: &str) -> Option<&Hall> {
        self.halls.get(name)
    }

    /// Get a mutable hall by name
    pub fn get_hall_mut(&mut self, name: &str) -> Option<&mut Hall> {
        self.halls.get_mut(name)
    }

    /// Get standard hall by type
    pub fn get_hall_by_type(&self, hall_type: &HallType) -> Option<&Hall> {
        self.halls.get(&hall_type.default_name())
    }

    /// Mark wing as accessed
    pub fn mark_accessed(&mut self) {
        self.last_accessed = Some(Utc::now());
        self.access_count += 1;
    }

    /// Get all rooms across all halls
    pub fn all_rooms(&self) -> Vec<(&str, &Room)> {
        self.halls
            .values()
            .flat_map(|hall| hall.rooms.iter().map(|(name, room)| (name.as_str(), room)))
            .collect()
    }
}

/// Wing type classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WingType {
    /// Person wing (individual's memories/preferences)
    Person,
    /// Project wing (work-related memories)
    Project,
    /// Organization wing (company/team memories)
    Organization,
    /// Domain wing (knowledge domain)
    Domain,
    /// Custom wing
    Custom(String),
}

// ============================================================================
// Halls
// ============================================================================

/// A Hall categorizes memories by type within a wing.
/// Standard halls: facts, events, discoveries, preferences, advice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hall {
    /// Hall name
    pub name: String,
    /// Hall type
    pub hall_type: HallType,
    /// All rooms in this hall
    pub rooms: HashMap<String, Room>,
    /// Parent wing name
    pub wing_name: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl Hall {
    /// Create a new hall
    pub fn new(wing_name: &str, hall_type: HallType) -> Self {
        Self {
            name: hall_type.default_name(),
            hall_type,
            rooms: HashMap::new(),
            wing_name: wing_name.to_string(),
            created_at: Utc::now(),
        }
    }

    /// Add or get a room (creates if not exists)
    pub fn get_or_create_room(&mut self, room_name: impl Into<String>) -> &mut Room {
        let room_name = room_name.into();
        if !self.rooms.contains_key(&room_name) {
            let room = Room::new(&room_name, &self.wing_name, &self.name);
            self.rooms.insert(room_name.clone(), room);
        }
        self.rooms.get_mut(&room_name).unwrap()
    }

    /// Get a room by name
    pub fn get_room(&self, name: &str) -> Option<&Room> {
        self.rooms.get(name)
    }
}

/// Hall type classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HallType {
    /// Factual knowledge (verified information)
    Facts,
    /// Events and experiences (what happened)
    Events,
    /// Discoveries and breakthroughs
    Discoveries,
    /// Preferences and opinions
    Preferences,
    /// Advice and lessons learned
    Advice,
    /// Custom hall type
    Custom(String),
}

impl HallType {
    /// Get default name for this hall type
    pub fn default_name(&self) -> String {
        match self {
            HallType::Facts => "hall_facts".to_string(),
            HallType::Events => "hall_events".to_string(),
            HallType::Discoveries => "hall_discoveries".to_string(),
            HallType::Preferences => "hall_preferences".to_string(),
            HallType::Advice => "hall_advice".to_string(),
            HallType::Custom(name) => name.clone(),
        }
    }

    /// Get all standard hall types
    pub fn standard_types() -> Vec<HallType> {
        vec![
            HallType::Facts,
            HallType::Events,
            HallType::Discoveries,
            HallType::Preferences,
            HallType::Advice,
        ]
    }
}

// ============================================================================
// Rooms
// ============================================================================

/// A Room contains memories about a specific topic within a hall.
/// Rooms with the same name across different wings are linked via tunnels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Room {
    /// Room identifier (topic name)
    pub name: String,
    /// Parent wing name
    pub wing_name: String,
    /// Parent hall name
    pub hall_name: String,
    /// Drawers: raw verbatim content (never altered)
    pub drawers: Vec<Drawer>,
    /// Closets: summaries/pointers to drawers
    pub closets: Vec<Closet>,
    /// Room summary (auto-generated)
    pub summary: Option<String>,
    /// Tags for this room
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified
    pub modified_at: DateTime<Utc>,
    /// Access count
    pub access_count: u64,
}

impl Room {
    /// Create a new empty room
    pub fn new(name: &str, wing_name: &str, hall_name: &str) -> Self {
        let now = Utc::now();
        Self {
            name: name.to_string(),
            wing_name: wing_name.to_string(),
            hall_name: hall_name.to_string(),
            drawers: Vec::new(),
            closets: Vec::new(),
            summary: None,
            tags: Vec::new(),
            created_at: now,
            modified_at: now,
            access_count: 0,
        }
    }

    /// Add a drawer (verbatim content)
    pub fn add_drawer(&mut self, drawer: Drawer) {
        self.drawers.push(drawer);
        self.modified_at = Utc::now();
    }

    /// Add a closet (summary/pointer)
    pub fn add_closet(&mut self, closet: Closet) {
        self.closets.push(closet);
        self.modified_at = Utc::now();
    }

    /// Mark room as accessed
    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
        self.modified_at = Utc::now();
    }

    /// Get full location path
    pub fn path(&self) -> String {
        format!("{}/{}/{}", self.wing_name, self.hall_name, self.name)
    }

    /// Check if room matches a topic
    pub fn matches_topic(&self, topic: &str) -> bool {
        self.name.to_lowercase().contains(&topic.to_lowercase())
            || self
                .tags
                .iter()
                .any(|t| t.to_lowercase().contains(&topic.to_lowercase()))
    }
}

// ============================================================================
// Drawers
// ============================================================================

/// A Drawer contains raw, verbatim source content.
/// Content here is NEVER permanently altered or summarized.
/// This preserves full context fidelity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Drawer {
    /// Unique identifier
    pub id: MemoryId,
    /// Drawer title
    pub title: String,
    /// Raw verbatim content (unaltered)
    pub content: String,
    /// Content type
    pub content_type: DrawerContentType,
    /// Source reference
    pub source: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Custom metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Drawer {
    /// Create a new drawer
    pub fn new(
        title: impl Into<String>,
        content: impl Into<String>,
        content_type: DrawerContentType,
    ) -> Self {
        Self {
            id: MemoryId::new_v4(),
            title: title.into(),
            content: content.into(),
            content_type,
            source: None,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Set source reference
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Drawer content type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrawerContentType {
    /// Raw conversation exchange
    Conversation,
    /// Code snippet
    Code,
    /// Document text
    Document,
    /// Log entry
    Log,
    /// Email/message
    Message,
    /// Custom content
    Custom(String),
}

// ============================================================================
// Closets
// ============================================================================

/// A Closet contains summaries/pointers that direct searches
/// toward the original drawer content without loading full content.
/// Enables efficient retrieval without loading all verbatim data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Closet {
    /// Unique identifier
    pub id: MemoryId,
    /// Closet title
    pub title: String,
    /// Summary of drawer content
    pub summary: String,
    /// Pointer to drawer ID
    pub drawer_id: MemoryId,
    /// Key topics extracted from content
    pub topics: Vec<String>,
    /// Key entities mentioned
    pub entities: Vec<EntityId>,
    /// Importance score (0.0-1.0)
    pub importance: f32,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl Closet {
    /// Create a new closet
    pub fn new(title: impl Into<String>, summary: impl Into<String>, drawer_id: MemoryId) -> Self {
        Self {
            id: MemoryId::new_v4(),
            title: title.into(),
            summary: summary.into(),
            drawer_id,
            topics: Vec::new(),
            entities: Vec::new(),
            importance: 0.5,
            created_at: Utc::now(),
        }
    }

    /// Add topics
    pub fn with_topics(mut self, topics: Vec<String>) -> Self {
        self.topics = topics;
        self
    }

    /// Add entities
    pub fn with_entities(mut self, entities: Vec<EntityId>) -> Self {
        self.entities = entities;
        self
    }

    /// Set importance
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }
}

// ============================================================================
// Tunnels
// ============================================================================

/// A Tunnel represents a cross-reference linking identically named rooms
/// across different wings. Enables discovery of related topics across
/// different people/projects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tunnel {
    /// Tunnel topic (the shared topic name)
    pub topic: String,
    /// Connected wing names
    pub connected_wings: Vec<String>,
    /// Connected room names
    pub connected_rooms: Vec<String>,
    /// Tunnel description
    pub description: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Usage count (how often this tunnel is traversed)
    pub traversal_count: u64,
}

impl Tunnel {
    /// Create a new tunnel
    pub fn new(
        topic: impl Into<String>,
        connected_wings: Vec<String>,
        connected_rooms: Vec<String>,
    ) -> Self {
        Self {
            topic: topic.into(),
            connected_wings,
            connected_rooms,
            description: None,
            created_at: Utc::now(),
            traversal_count: 0,
        }
    }

    /// Mark tunnel as traversed
    pub fn traverse(&mut self) {
        self.traversal_count += 1;
    }
}

// ============================================================================
// Palace Navigation
// ============================================================================

/// Navigation query for finding memories in the palace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalaceQuery {
    /// Optional wing filter
    pub wing: Option<String>,
    /// Optional hall filter
    pub hall: Option<String>,
    /// Optional room filter
    pub room: Option<String>,
    /// Topic search across closets
    pub topic_search: Option<String>,
    /// Include tunnel results
    pub follow_tunnels: bool,
    /// Max results
    pub limit: usize,
}

impl PalaceQuery {
    /// Create a new query
    pub fn new() -> Self {
        Self {
            wing: None,
            hall: None,
            room: None,
            topic_search: None,
            follow_tunnels: false,
            limit: 20,
        }
    }

    /// Scope to specific wing
    pub fn in_wing(mut self, wing: impl Into<String>) -> Self {
        self.wing = Some(wing.into());
        self
    }

    /// Scope to specific hall
    pub fn in_hall(mut self, hall: impl Into<String>) -> Self {
        self.hall = Some(hall.into());
        self
    }

    /// Scope to specific room
    pub fn in_room(mut self, room: impl Into<String>) -> Self {
        self.room = Some(room.into());
        self
    }

    /// Search by topic
    pub fn search_topic(mut self, topic: impl Into<String>) -> Self {
        self.topic_search = Some(topic.into());
        self
    }

    /// Follow tunnels to related wings
    pub fn with_tunnels(mut self) -> Self {
        self.follow_tunnels = true;
        self
    }

    /// Set result limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

impl Default for PalaceQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Navigation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalaceResult {
    /// Matched drawers
    pub drawers: Vec<Drawer>,
    /// Matched closets
    pub closets: Vec<Closet>,
    /// Tunnel connections found
    pub tunnels: Vec<Tunnel>,
    /// Path taken to reach this result
    pub path: String,
}

impl PalaceResult {
    /// Create empty result
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            drawers: Vec::new(),
            closets: Vec::new(),
            tunnels: Vec::new(),
            path: path.into(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.drawers.is_empty() && self.closets.is_empty()
    }

    /// Merge results
    pub fn merge(&mut self, other: PalaceResult) {
        self.drawers.extend(other.drawers);
        self.closets.extend(other.closets);
        self.tunnels.extend(other.tunnels);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_palace_creation() {
        let mut palace = MemoryPalace::new("MyPalace");
        let wing = Wing::new("Alice", WingType::Person);
        palace.add_wing(wing);

        assert_eq!(palace.wings.len(), 1);
        assert!(palace.get_wing("Alice").is_some());
    }

    #[test]
    fn test_wing_standard_halls() {
        let wing = Wing::new("ProjectX", WingType::Project);
        assert_eq!(wing.halls.len(), 5); // 5 standard hall types
        assert!(wing.get_hall("hall_facts").is_some());
        assert!(wing.get_hall("hall_events").is_some());
    }

    #[test]
    fn test_room_creation() {
        let mut palace = MemoryPalace::new("TestPalace");
        let mut wing = Wing::new("Person1", WingType::Person);

        if let Some(hall) = wing.get_hall_mut("hall_facts") {
            let room = hall.get_or_create_room("rust-programming");
            assert_eq!(room.name, "rust-programming");
        }

        palace.add_wing(wing);
    }

    #[test]
    fn test_drawer_and_closet() {
        let mut room = Room::new("test", "wing", "hall_facts");

        let drawer = Drawer::new(
            "Test Conversation",
            "User asked about Rust lifetimes",
            DrawerContentType::Conversation,
        );
        let drawer_id = drawer.id;
        room.add_drawer(drawer);

        let closet = Closet::new(
            "Rust Lifetimes Discussion",
            "Discussion about Rust lifetime annotations",
            drawer_id,
        )
        .with_topics(vec!["rust".to_string(), "lifetimes".to_string()]);
        room.add_closet(closet);

        assert_eq!(room.drawers.len(), 1);
        assert_eq!(room.closets.len(), 1);
    }

    #[test]
    fn test_tunnel_creation() {
        let mut palace = MemoryPalace::new("TestPalace");
        palace.add_wing(Wing::new("Alice", WingType::Person));
        palace.add_wing(Wing::new("ProjectX", WingType::Project));

        palace.add_tunnel("rust", "Alice");
        palace.add_tunnel("rust", "ProjectX");

        let wings = palace.find_tunnel_wings("rust");
        assert_eq!(wings.len(), 2);
    }

    #[test]
    fn test_palace_stats() {
        let mut palace = MemoryPalace::new("TestPalace");
        let mut wing = Wing::new("Test", WingType::Person);

        if let Some(hall) = wing.get_hall_mut("hall_facts") {
            hall.get_or_create_room("topic1");
            hall.get_or_create_room("topic2");
        }

        palace.add_wing(wing);
        let stats = palace.stats();

        assert_eq!(stats.wings, 1);
        assert_eq!(stats.halls, 5);
        assert_eq!(stats.rooms, 2);
    }

    #[test]
    fn test_palace_query_builder() {
        let query = PalaceQuery::new()
            .in_wing("Alice")
            .in_hall("hall_facts")
            .search_topic("rust")
            .with_tunnels()
            .limit(10);

        assert_eq!(query.wing, Some("Alice".to_string()));
        assert_eq!(query.topic_search, Some("rust".to_string()));
        assert!(query.follow_tunnels);
        assert_eq!(query.limit, 10);
    }
}
