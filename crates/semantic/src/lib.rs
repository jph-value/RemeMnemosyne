pub mod index;
pub mod store;
pub mod turboquant;

#[cfg(feature = "sharding")]
pub mod sharding;

pub use index::*;
pub use store::*;
pub use turboquant::*;

#[cfg(feature = "sharding")]
pub use sharding::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turboquant_creation() {
        let quantizer = TurboQuantizer::new(1536, 8, 8, 42);
        assert!(quantizer.is_ok());

        let q = quantizer.unwrap();
        assert_eq!(q.config.dimensions, 1536);
        assert!(!q.trained);
    }

    #[test]
    fn test_turboquant_training() {
        let mut quantizer = TurboQuantizer::new(32, 8, 4, 42).unwrap();

        let data: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..32).map(|j| ((i + j) as f32).sin()).collect())
            .collect();

        assert!(quantizer.train(&data).is_ok());
        assert!(quantizer.trained);
    }

    #[test]
    fn test_turboquant_encode_decode() {
        let mut quantizer = TurboQuantizer::new(32, 8, 4, 42).unwrap();

        let data: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..32).map(|j| ((i + j) as f32 / 10.0).sin()).collect())
            .collect();
        quantizer.train(&data).unwrap();

        let code = quantizer.encode(&data[0]).unwrap();
        assert_eq!(code.codes.len(), 4);

        let decoded = quantizer.decode(&code).unwrap();
        assert_eq!(decoded.len(), 32);
    }

    #[test]
    fn test_polar_quantizer() {
        let pq = PolarQuantizer::new(64, 8, 4);
        let vector: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let code = pq.encode(&vector).unwrap();
        assert!(!code.codes.is_empty());
    }

    #[test]
    fn test_hnsw_index() {
        let mut index = HNSWIndex::new(32, 16, 100);

        for i in 0..20 {
            let vec: Vec<f32> = (0..32).map(|j| ((i + j) as f32).sin()).collect();
            index.add(vec, None).unwrap();
        }

        assert_eq!(index.len(), 20);

        let query: Vec<f32> = (0..32).map(|j| (j as f32).sin()).collect();
        let results = index.search(&query, 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_flat_index() {
        let mut index = FlatIndex::new(32);
        let id = uuid::Uuid::new_v4();

        index.add(id, vec![0.1; 32]).unwrap();
        assert_eq!(index.len(), 1);
    }
}
