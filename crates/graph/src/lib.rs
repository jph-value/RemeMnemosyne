pub mod entity;
pub mod relationship;
pub mod store;
pub mod traversal;

pub use entity::*;
pub use relationship::*;
pub use store::*;
pub use traversal::*;

#[cfg(test)]
mod tests {
    use super::*;
    use mnemosyne_core::*;

    #[test]
    fn test_graph_entity_creation() {
        let entity = GraphEntity::new(
            "TestEntity",
            EntityType::Technology,
            "A test",
            vec![0.1, 0.2],
        );
        assert_eq!(entity.name, "TestEntity");
        assert_eq!(entity.mention_count, 1);
    }

    #[test]
    fn test_graph_entity_similarity() {
        let e1 = GraphEntity::new("A", EntityType::Concept, "", vec![1.0, 0.0]);
        let e2 = GraphEntity::new("B", EntityType::Concept, "", vec![1.0, 0.0]);
        let e3 = GraphEntity::new("C", EntityType::Concept, "", vec![0.0, 1.0]);
        
        assert_eq!(e1.similarity(&e2), 1.0);
        assert_eq!(e1.similarity(&e3), 0.0);
    }

    #[test]
    fn test_graph_relationship_creation() {
        let source = uuid::Uuid::new_v4();
        let target = uuid::Uuid::new_v4();
        
        let rel = Relationship::new(source, target, RelationshipType::Uses, 0.8);
        assert_eq!(rel.strength, 0.8);
    }

    #[test]
    fn test_graph_relationship_strengthen() {
        let source = uuid::Uuid::new_v4();
        let target = uuid::Uuid::new_v4();
        
        let mut rel = GraphRelationship::new(source, target, RelationshipType::Uses, 0.5);
        rel.strengthen(0.3);
        
        assert!((rel.strength - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_relationship_path() {
        let entities = vec![uuid::Uuid::new_v4(), uuid::Uuid::new_v4()];
        let path = RelationshipPath::new(entities, vec![]);
        
        assert_eq!(path.hop_count, 0);
    }
}
