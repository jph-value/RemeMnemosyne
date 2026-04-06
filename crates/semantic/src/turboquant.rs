use rememnemosyne_core::{MemoryError, Result};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// TurboQuant quantization variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMethod {
    /// Product Quantization - minimizes reconstruction error
    PQ,
    /// Optimized Product Quantization - better for inner products
    OPQ,
    /// Polar Quantization - hierarchical compression
    Polar,
    /// Quantized Johnson-Lindenstrauss transform
    QJL,
}

/// Quantizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantConfig {
    pub dimensions: usize,
    pub bits: u8,
    pub num_subquantizers: usize,
    pub seed: u64,
    pub method: QuantizationMethod,
    pub num_clusters: usize,
    pub iterations: usize,
}

impl Default for TurboQuantConfig {
    fn default() -> Self {
        Self {
            dimensions: 1536,
            bits: 8,
            num_subquantizers: 8,
            seed: 42,
            method: QuantizationMethod::PQ,
            num_clusters: 256,
            iterations: 20,
        }
    }
}

/// Quantized code representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedCode {
    pub codes: Vec<u8>,
    pub dimensions: usize,
    pub bits: u8,
    pub method: QuantizationMethod,
}

impl QuantizedCode {
    pub fn size_bytes(&self) -> usize {
        self.codes.len()
    }

    pub fn compression_ratio(&self, original_dims: usize) -> f32 {
        let original_bytes = original_dims * 4; // f32
        self.size_bytes() as f32 / original_bytes as f32
    }
}

/// TurboQuant - Fast quantization for embeddings
///
/// Implements Product Quantization (PQ), Optimized PQ (OPQ),
/// Polar Quantization, and QJL transforms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantizer {
    pub config: TurboQuantConfig,
    /// Codebooks for each subquantizer
    pub codebooks: Vec<Vec<Vec<f32>>>,
    /// Projection matrix for OPQ
    pub projection: Option<Vec<Vec<f32>>>,
    /// Is trained
    pub trained: bool,
}

impl TurboQuantizer {
    /// Create a new quantizer with given dimensions and bits
    pub fn new(dimensions: usize, bits: u8, num_subquantizers: usize, seed: u64) -> Result<Self> {
        if dimensions % num_subquantizers != 0 {
            return Err(MemoryError::Quantization(format!(
                "Dimensions {} must be divisible by subquantizers {}",
                dimensions, num_subquantizers
            )));
        }

        let num_clusters = 1 << bits;

        let config = TurboQuantConfig {
            dimensions,
            bits,
            num_subquantizers,
            seed,
            method: QuantizationMethod::PQ,
            num_clusters,
            iterations: 20,
        };

        // Initialize empty codebooks
        let sub_dims = dimensions / num_subquantizers;
        let codebooks = (0..num_subquantizers)
            .map(|_| (0..num_clusters).map(|_| vec![0.0; sub_dims]).collect())
            .collect();

        Ok(Self {
            config,
            codebooks,
            projection: None,
            trained: false,
        })
    }

    /// Create with specific method
    pub fn with_method(mut self, method: QuantizationMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Train the quantizer on sample data
    pub fn train(&mut self, data: &[Vec<f32>]) -> Result<()> {
        if data.is_empty() {
            return Err(MemoryError::Quantization("Empty training data".into()));
        }

        let dim = self.config.dimensions;
        let sub_dim = dim / self.config.num_subquantizers;
        let n = data.len();

        // Validate dimensions
        for vec in data {
            if vec.len() != dim {
                return Err(MemoryError::Quantization(format!(
                    "Vector dimension {} != expected {}",
                    vec.len(),
                    dim
                )));
            }
        }

        // For OPQ, compute rotation matrix first
        if self.config.method == QuantizationMethod::OPQ {
            self.compute_opq_rotation(data)?;
        }

        // Train codebooks using k-means
        for m in 0..self.config.num_subquantizers {
            let start = m * sub_dim;
            let end = start + sub_dim;

            // Extract sub-vectors
            let sub_vectors: Vec<Vec<f32>> = data.iter().map(|v| v[start..end].to_vec()).collect();

            // Run k-means
            self.codebooks[m] = self.kmeans(
                &sub_vectors,
                self.config.num_clusters,
                self.config.iterations,
            )?;
        }

        self.trained = true;
        Ok(())
    }

    /// Encode a vector to quantized code
    pub fn encode(&self, vector: &[f32]) -> Result<QuantizedCode> {
        if !self.trained {
            return Err(MemoryError::Quantization("Quantizer not trained".into()));
        }

        if vector.len() != self.config.dimensions {
            return Err(MemoryError::Quantization(format!(
                "Vector dimension {} != expected {}",
                vector.len(),
                self.config.dimensions
            )));
        }

        let sub_dim = self.config.dimensions / self.config.num_subquantizers;
        let mut codes = Vec::with_capacity(self.config.num_subquantizers);

        for m in 0..self.config.num_subquantizers {
            let start = m * sub_dim;
            let end = start + sub_dim;
            let sub_vector = &vector[start..end];

            // Find nearest centroid
            let mut min_dist = f32::MAX;
            let mut best_idx = 0;

            for (idx, centroid) in self.codebooks[m].iter().enumerate() {
                let dist = self.sq_l2dist(sub_vector, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = idx;
                }
            }

            codes.push(best_idx as u8);
        }

        Ok(QuantizedCode {
            codes,
            dimensions: self.config.dimensions,
            bits: self.config.bits,
            method: self.config.method,
        })
    }

    /// Decode quantized code back to approximate vector
    pub fn decode(&self, code: &QuantizedCode) -> Result<Vec<f32>> {
        if !self.trained {
            return Err(MemoryError::Quantization("Quantizer not trained".into()));
        }

        let sub_dim = self.config.dimensions / self.config.num_subquantizers;
        let mut reconstructed = Vec::with_capacity(self.config.dimensions);

        for (m, &code_idx) in code.codes.iter().enumerate() {
            let centroid = &self.codebooks[m][code_idx as usize];
            reconstructed.extend_from_slice(centroid);
        }

        Ok(reconstructed)
    }

    /// Estimate inner product between quantized code and query vector
    /// This is the key operation for fast similarity search
    pub fn inner_product_estimate(&self, code: &QuantizedCode, query: &[f32]) -> Result<f32> {
        if !self.trained {
            return Err(MemoryError::Quantization("Quantizer not trained".into()));
        }

        if query.len() != self.config.dimensions {
            return Err(MemoryError::Quantization(format!(
                "Query dimension {} != expected {}",
                query.len(),
                self.config.dimensions
            )));
        }

        let sub_dim = self.config.dimensions / self.config.num_subquantizers;
        let mut estimate = 0.0;

        for (m, &code_idx) in code.codes.iter().enumerate() {
            let start = m * sub_dim;
            let end = start + sub_dim;
            let query_sub = &query[start..end];
            let centroid = &self.codebooks[m][code_idx as usize];

            // Dot product of query sub-vector with centroid
            estimate += self.dot_product(query_sub, centroid);
        }

        Ok(estimate)
    }

    /// Compute L2 distance between quantized code and query
    pub fn l2_distance_estimate(&self, code: &QuantizedCode, query: &[f32]) -> Result<f32> {
        if !self.trained {
            return Err(MemoryError::Quantization("Quantizer not trained".into()));
        }

        let sub_dim = self.config.dimensions / self.config.num_subquantizers;
        let mut dist = 0.0;

        for (m, &code_idx) in code.codes.iter().enumerate() {
            let start = m * sub_dim;
            let end = start + sub_dim;
            let query_sub = &query[start..end];
            let centroid = &self.codebooks[m][code_idx as usize];

            dist += self.sq_l2dist(query_sub, centroid);
        }

        Ok(dist)
    }

    /// Batch encode multiple vectors
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Result<Vec<QuantizedCode>> {
        vectors.iter().map(|v| self.encode(v)).collect()
    }

    /// Batch inner product estimation
    pub fn inner_product_estimate_batch(
        &self,
        codes: &[QuantizedCode],
        query: &[f32],
    ) -> Result<Vec<f32>> {
        codes
            .iter()
            .map(|c| self.inner_product_estimate(c, query))
            .collect()
    }

    // Private helper methods

    fn kmeans(&self, data: &[Vec<f32>], k: usize, max_iterations: usize) -> Result<Vec<Vec<f32>>> {
        let dim = data[0].len();
        let n = data.len();

        if k >= n {
            return Ok(data.to_vec());
        }

        // Initialize centroids with k-means++
        let mut centroids = self.kmeanspp_init(data, k)?;

        for _ in 0..max_iterations {
            // Assign points to nearest centroid
            let mut assignments: Vec<usize> = Vec::with_capacity(n);
            for point in data {
                let mut min_dist = f32::MAX;
                let mut best = 0;
                for (idx, centroid) in centroids.iter().enumerate() {
                    let dist = self.sq_l2dist(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best = idx;
                    }
                }
                assignments.push(best);
            }

            // Update centroids
            let mut new_centroids = vec![vec![0.0; dim]; k];
            let mut counts = vec![0usize; k];

            for (point, &assignment) in data.iter().zip(assignments.iter()) {
                for d in 0..dim {
                    new_centroids[assignment][d] += point[d];
                }
                counts[assignment] += 1;
            }

            for k_idx in 0..k {
                if counts[k_idx] > 0 {
                    for d in 0..dim {
                        new_centroids[k_idx][d] /= counts[k_idx] as f32;
                    }
                }
            }

            // Check convergence
            let mut converged = true;
            for k_idx in 0..k {
                if counts[k_idx] > 0 {
                    let dist = self.sq_l2dist(&centroids[k_idx], &new_centroids[k_idx]);
                    if dist > 1e-6 {
                        converged = false;
                        break;
                    }
                }
            }

            centroids = new_centroids;

            if converged {
                break;
            }
        }

        Ok(centroids)
    }

    fn kmeanspp_init(&self, data: &[Vec<f32>], k: usize) -> Result<Vec<Vec<f32>>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.config.seed.hash(&mut hasher);
        let seed = hasher.finish();

        let mut rng = XorShiftRng::new(seed);
        let mut centroids = Vec::with_capacity(k);
        let mut distances = vec![f32::MAX; data.len()];

        // Choose first centroid randomly
        let first_idx = (rng.next_f32() * data.len() as f32) as usize;
        centroids.push(data[first_idx].clone());

        // Choose remaining centroids
        for _ in 1..k {
            // Update distances
            let total_dist: f32 = data
                .iter()
                .zip(distances.iter_mut())
                .enumerate()
                .map(|(i, (point, dist))| {
                    let last_centroid = centroids.last().unwrap();
                    let new_dist = self.sq_l2dist(point, last_centroid);
                    *dist = (*dist).min(new_dist);
                    *dist
                })
                .sum();

            // Choose next centroid with probability proportional to distance^2
            let r = rng.next_f32() * total_dist;
            let mut cumsum = 0.0;
            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= r {
                    centroids.push(data[i].clone());
                    break;
                }
            }
        }

        Ok(centroids)
    }

    fn compute_opq_rotation(&mut self, _data: &[Vec<f32>]) -> Result<()> {
        // Simplified OPQ - in practice would use Procrustes analysis
        // For now, use identity projection
        let dim = self.config.dimensions;
        self.projection = Some(
            (0..dim)
                .map(|i| {
                    let mut row = vec![0.0; dim];
                    row[i] = 1.0;
                    row
                })
                .collect(),
        );
        Ok(())
    }

    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[inline]
    fn sq_l2dist(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }
}

/// Simple XorShift RNG for reproducible results
struct XorShiftRng {
    state: u64,
}

impl XorShiftRng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }
}

/// Polar Quantization - hierarchical compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolarQuantizer {
    pub dimensions: usize,
    pub bits: u8,
    pub num_levels: usize,
    pub codebook: Vec<Vec<f32>>,
}

impl PolarQuantizer {
    pub fn new(dimensions: usize, bits: u8, num_levels: usize) -> Self {
        let codebook_size = 1 << bits;
        let codebook = vec![vec![0.0; dimensions]; codebook_size];

        Self {
            dimensions,
            bits,
            num_levels,
            codebook,
        }
    }

    /// Encode using polar quantization
    pub fn encode(&self, vector: &[f32]) -> Result<QuantizedCode> {
        // Compute magnitude and angles
        let magnitude = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_vector: Vec<f32> = vector.iter().map(|x| x / magnitude.max(1e-10)).collect();

        // Quantize magnitude
        let mag_code = ((magnitude.min(10.0) / 10.0) * ((1 << self.bits) - 1) as f32) as u8;

        // Quantize direction using spherical coordinates
        let mut codes = vec![mag_code];
        let remaining_bits = self.bits as usize;

        // Simple angular quantization
        for level in 0..self.num_levels.min(self.dimensions - 1) {
            let angle = norm_vector[level].acos();
            let code = ((angle / PI) * ((1 << remaining_bits) - 1) as f32) as u8;
            codes.push(code);
        }

        Ok(QuantizedCode {
            codes,
            dimensions: self.dimensions,
            bits: self.bits,
            method: QuantizationMethod::Polar,
        })
    }
}

/// QJL (Quantized Johnson-Lindenstrauss) Transform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QJLTransform {
    pub input_dim: usize,
    pub output_dim: usize,
    pub seed: u64,
    pub projection_matrix: Vec<Vec<f32>>,
    pub quantizer: TurboQuantizer,
}

impl QJLTransform {
    pub fn new(input_dim: usize, output_dim: usize, bits: u8, seed: u64) -> Result<Self> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let s = hasher.finish();
        let mut rng = XorShiftRng::new(s);

        // Generate random projection matrix (sparse JL)
        let projection_matrix = (0..output_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| {
                        // Sparse JL: ±1 with probability 1/2 each, or 0
                        let r = rng.next_f32();
                        if r < 0.25 {
                            -1.0
                        } else if r < 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        let quantizer = TurboQuantizer::new(output_dim, bits, 1, seed)?;

        Ok(Self {
            input_dim,
            output_dim,
            seed,
            projection_matrix,
            quantizer,
        })
    }

    /// Apply QJL transform
    pub fn transform(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if vector.len() != self.input_dim {
            return Err(MemoryError::Quantization(format!(
                "Input dim {} != expected {}",
                vector.len(),
                self.input_dim
            )));
        }

        let scale = (self.output_dim as f32).sqrt() / (self.input_dim as f32).sqrt();
        let mut output = vec![0.0; self.output_dim];

        for i in 0..self.output_dim {
            for j in 0..self.input_dim {
                output[i] += self.projection_matrix[i][j] * vector[j];
            }
            output[i] *= scale;
        }

        Ok(output)
    }
}
