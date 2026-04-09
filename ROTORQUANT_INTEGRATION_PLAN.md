# RotorQuant Integration Plan

## Source Evaluation

**Project**: [scrya-com/rotorquant](https://github.com/scrya-com/rotorquant)  
**Purpose**: KV cache compression for LLMs using structured rotations + scalar quantization  
**License**: Check before integration (likely Apache 2.0 or MIT based on scrya-com repos)

---

## What RotorQuant Does vs Our TurboQuant

| Aspect | Our TurboQuant (PQ/OPQ) | RotorQuant |
|--------|-------------------------|------------|
| **Algorithm** | k-means codebooks per subspace | Deterministic 2D/4D rotations + Lloyd-Max scalar quantization |
| **Parameters** | 16,384 codebook entries per 128-dim block | 128 rotation parameters per 128-dim block |
| **Training** | Required (k-means on data) | None (deterministic rotations) |
| **Bit Rates** | 4-8 bit typical | 3-4 bit (with quality preservation) |
| **Parameter Size** | Large codebooks | Tiny rotation angles |
| **Use Case** | General vector compression | Originally KV cache, adaptable to embeddings |
| **Codebook Lookup** | O(k) per subvector | None (scalar quantization) |

**Key Innovation**: RotorQuant replaces the O(d log d) Walsh-Hadamard Transform with O(d) independent 2D/4D rotations that are fully parallelizable with zero inter-element dependencies. This is fundamentally different from PQ's codebook-based approach.

---

## Integration Strategy: Modular Experimental Addition

### Design Goals

1. **Experimental feature flag** — `rotor-quantization`
2. **No impact on existing TurboQuant** — Parallel implementation
3. **Same trait interface** — Drop-in replacement in `SemanticMemoryStore`
4. **3-bit and 4-bit modes** — RotorQuant's sweet spot
5. **No training required** — Deterministic rotations

### Three RotorQuant Methods to Implement

| Method | Rotation Type | Bits | Use Case |
|--------|--------------|------|----------|
| **PlanarQuant** | 2D Givens rotations | 3-4 bit | Highest quality, smallest parameter footprint |
| **IsoQuant** | 4D quaternion rotations | 3-4 bit | Best quality/parameter tradeoff |
| **TurboQuant (rotor version)** | Walsh-Hadamard + scalar | 3-4 bit | Compatibility with existing TurboQuant users |

### File Structure

```
crates/
└── semantic/
    ├── src/
    │   ├── turboquant.rs       # Existing PQ/OPQ/Polar/QJL (unchanged)
    │   └── rotorquant.rs       # NEW: RotorQuant implementation
    └── Cargo.toml              # Add "rotor-quantization" feature
```

### Implementation Plan

#### Phase 1: Core Algorithms (~400 lines)

```rust
// crates/semantic/src/rotorquant.rs

/// RotorQuant method
pub enum RotorMethod {
    /// 2D Givens rotations (PlanarQuant)
    Planar,
    /// 4D quaternion rotations (IsoQuant)  
    Iso,
}

/// RotorQuant quantizer
pub struct RotorQuantizer {
    method: RotorMethod,
    bits: u8,
    dimensions: usize,
    /// Rotation angles (2D for Planar, 4D for Iso)
    rotation_angles: Vec<f32>,
    /// Lloyd-Max quantization boundaries
    boundaries: Vec<f32>,
    /// Reconstruction levels
    levels: Vec<f32>,
}

impl RotorQuantizer {
    /// Create new quantizer (no training needed)
    pub fn new(dimensions: usize, bits: u8, method: RotorMethod) -> Self;
    
    /// Quantize a single vector
    pub fn encode(&self, vector: &[f32]) -> Vec<u8>;
    
    /// Dequantize back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Vec<f32>;
    
    /// Quantize batch
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Vec<Vec<u8>>;
    
    /// Compute compression ratio
    pub fn compression_ratio(&self) -> f32;
    
    /// Get rotation matrix
    fn compute_rotations(&self) -> Vec<Vec<f32>>;
    
    /// Apply rotation to vector
    fn apply_rotation(&self, vector: &[f32]) -> Vec<f32>;
    
    /// Inverse rotation for dequantization
    fn apply_inverse_rotation(&self, vector: &[f32]) -> Vec<f32>;
    
    /// Lloyd-Max scalar quantization
    fn scalar_quantize(&self, value: f32) -> u8;
    
    /// Lloyd-Max scalar dequantization
    fn scalar_dequantize(&self, code: u8) -> f32;
}
```

#### Phase 2: Integration with SemanticMemoryStore (~200 lines)

```rust
// In crates/semantic/src/store.rs

pub struct SemanticMemoryStore {
    // ... existing fields ...
    
    // NEW: Optional RotorQuant quantizer
    #[cfg(feature = "rotor-quantization")]
    rotor_quantizer: Option<Arc<RwLock<RotorQuantizer>>>,
}

impl SemanticMemoryStore {
    /// Enable RotorQuant (overrides TurboQuant for new vectors)
    #[cfg(feature = "rotor-quantization")]
    pub async fn enable_rotor_quantization(
        &self,
        method: RotorMethod,
        bits: u8,
    ) -> Result<()>;
}
```

#### Phase 3: Feature Flag & Benchmarks (~100 lines)

```toml
# crates/semantic/Cargo.toml
[features]
sharding = []
rotor-quantization = []  # NEW: Experimental RotorQuant support
```

---

## Expected Compression Improvement

| Configuration | TurboQuant (PQ) | RotorQuant (Planar) | Improvement |
|---------------|-----------------|---------------------|-------------|
| 1536-dim, 8-bit | 1536 bytes (1.0x) | 1536 bytes (1.0x) | Same |
| 1536-dim, 4-bit | 768 bytes (2.0x) | 768 bytes (2.0x) | Same |
| 1536-dim, 3-bit | N/A (codebook too small) | **576 bytes (2.67x)** | New capability |
| 384-dim, 3-bit | N/A | **144 bytes (2.67x)** | New capability |
| Parameters | 16,384 per block | 128 per block | **128x smaller** |

**RotorQuant's advantage**: At 3-bit, PQ codebooks become too small to be useful (only 8 centroids per subspace). RotorQuant doesn't use codebooks — it uses deterministic rotations + scalar quantization, so it works well even at 3-bit.

---

## Why This Matters for RISC.OSINT

1. **Storage at scale** — 50,000+ events × 1536-dim embeddings × 3-bit = ~28MB vs ~75MB at 8-bit
2. **No training required** — Deterministic rotations work out-of-the-box, no k-means on data
3. **Lower memory footprint** — 128 rotation params vs 16,384 codebook entries (128x reduction)
4. **Complements existing TurboQuant** — Users can choose PQ (trained, 4-8 bit) or RotorQuant (deterministic, 3-4 bit)

---

## Build Impact

- **Default build**: Zero impact (feature flag off)
- **With `rotor-quantization`**: ~700 lines of new code, no new dependencies
- **Tests**: Unit tests for encode/decode roundtrip, compression ratio, comparison with TurboQuant

---

## Next Steps

1. Verify rotorquant license compatibility
2. Implement `crates/semantic/src/rotorquant.rs` (~700 lines)
3. Add feature flag to `crates/semantic/Cargo.toml`
4. Integrate with `SemanticMemoryStore`
5. Add benchmarks comparing PQ vs RotorQuant at 3/4/8 bit
6. Document usage
