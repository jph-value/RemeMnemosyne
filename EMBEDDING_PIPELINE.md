# Embedding Pipeline - Fully Wired Architecture

## What Changed

The embedding system was completely rewired from a hardcoded hash-based embedder to a **pluggable provider architecture**.

---

## Before (Broken)

```
Text → MicroEmbedder (hash-only, hardcoded) → Vec<f32> → Semantic Store
```

- `MicroEmbedder` was hardcoded with `MicroEmbedModel::Hash`
- `CandleEmbedder` existed but was never called
- `EmbeddingProvider` trait existed but was orphaned
- No way for integrators to swap providers

---

## After (Fully Wired)

```
Text → EmbeddingProviderRouter → Arc<dyn EmbeddingProvider> → Vec<f32> → Semantic Store
                                  ├── HashEmbedder (default fallback)
                                  ├── CandleEmbedder (local ML, via feature flag)
                                  ├── OpenAI (stub, ready for impl)
                                  ├── Voyage (stub, ready for impl)
                                  ├── Cohere (stub, ready for impl)
                                  └── Custom (user's endpoint)
```

### Architecture

```
rememnemosyne-core/
├── src/embedding.rs          ← EmbeddingProvider trait, HashEmbedder, types
│   ├── trait EmbeddingProvider
│   ├── struct EmbeddingRequest
│   ├── struct EmbeddingResponse
│   ├── enum EmbeddingProviderType
│   ├── struct EmbeddingProviderConfig
│   └── struct HashEmbedder (default fallback, always works)

rememnemosyne-cognitive/
└── src/candle_embed.rs       ← CandleEmbedder implements EmbeddingProvider
    └── #[async_trait] impl EmbeddingProvider for CandleEmbedder

rememnemosyne-engine/
├── src/providers.rs          ← EmbeddingProviderRouter
│   ├── struct EmbeddingProviderRouter
│   ├── fn with_default()     → HashEmbedder
│   ├── fn from_config()     → Configured provider
│   ├── fn clone_provider()  → Arc<dyn EmbeddingProvider> for Send safety
│   └── re-exports from core
│
├── src/router.rs             ← MemoryRouter uses EmbeddingProviderRouter
│   ├── embedder: Arc<RwLock<EmbeddingProviderRouter>>
│   ├── generate_embedding() → Uses provider
│   ├── query()              → Uses provider for query embeddings
│   └── set_embedding_provider() → Runtime provider swap
│
└── src/builder.rs            ← Engine uses router's embedding
    └── generate_embedding() → Calls router, falls back on error
```

---

## How Integrators Use It

### Default (Zero Config)
```rust
// Just works with hash-based embeddings
let engine = RememnosyneEngine::default()?;
engine.remember("text", "summary", MemoryTrigger::UserInput).await?;
```

### Configure Hash Embedder (Explicit)
```rust
use rememnemosyne_core::{EmbeddingProviderConfig, EmbeddingProviderType};

let config = EmbeddingProviderConfig {
    provider: EmbeddingProviderType::Local,
    dimensions: 384,
    ..Default::default()
};

let router_config = MemoryRouterConfig {
    embedding_config: Some(config),
    ..Default::default()
};
```

### Use Candle ML Embeddings (Feature Flag)
```toml
[dependencies]
rememnemosyne-engine = { version = "0.1", features = ["candle-embeddings"] }
```

```rust
use rememnemosyne_cognitive::{CandleEmbedder, CandleEmbedConfig};

let candle = CandleEmbedder::default_embedder();
candle.load_model().await?;

engine.router.set_embedding_provider(Arc::new(candle));
```

### Custom Provider (User's API)
```rust
struct MyEmbedder { /* user's impl */ }

#[async_trait]
impl EmbeddingProvider for MyEmbedder {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        // Call user's API
    }
    fn provider_type(&self) -> EmbeddingProviderType { /* ... */ }
    fn model_name(&self) -> &str { /* ... */ }
    fn dimensions(&self) -> usize { /* ... */ }
}

engine.router.set_embedding_provider(Arc::new(MyEmbedder { /* ... */ }));
```

---

## Key Design Decisions

### 1. Trait in Core Crate
`EmbeddingProvider` trait is in `rememnemosyne-core` (not engine) so:
- Cognitive crate can implement it without circular deps
- Any crate can implement it
- Engine just uses the trait

### 2. HashEmbedder as Fallback
`HashEmbedder` is always available (no feature flags):
- Deterministic embeddings
- Zero external dependencies
- Works in air-gapped environments
- Integrators can start immediately

### 3. Clone Provider for Send Safety
`EmbeddingProviderRouter::clone_provider()` returns `Arc<dyn EmbeddingProvider>`:
- Avoids `parking_lot::RwLockReadGuard` across await
- `Arc` is `Clone + Send + Sync`
- Provider can be used safely in async contexts

### 4. Graceful Degradation
If embedding provider fails:
```rust
self.router.generate_embedding(text).await.unwrap_or_else(|e| {
    tracing::warn!("Embedding generation failed: {}", e);
    vec![0.0; self.router.embedding_dimensions()]
})
```
- System continues with zero vectors
- Logs warning for debugging
- Never panics

### 5. Dimension Normalization
All embeddings are padded/truncated to match configured dimensions:
```rust
if embedding.len() != self.config.embedding_dimensions {
    let mut corrected = vec![0.0; self.config.embedding_dimensions];
    let copy_len = std::cmp::min(embedding.len(), self.config.embedding_dimensions);
    corrected[..copy_len].copy_from_slice(&embedding[..copy_len]);
    Ok(corrected)
}
```

---

## Test Coverage

**93 tests pass** including 5 new embedding tests:
- `test_hash_embedder` - Sync embedding generation
- `test_hash_embedder_deterministic` - Reproducible results
- `test_embedding_request_builder` - Request builder pattern
- `test_hash_embedder_async` - Async trait impl
- `test_hash_embedder_batch` - Batch processing

---

## Build Status

```bash
# Default (pure Rust, no ML)
cargo check       # ✅ Passes
cargo test        # ✅ 93 tests

# With Candle ML embeddings
cargo check --features candle-embeddings  # ✅ Passes
cargo test --features candle-embeddings   # ✅ Passes
```

---

## Files Changed

| File | Change |
|------|--------|
| `crates/core/src/embedding.rs` | **NEW** - EmbeddingProvider trait, HashEmbedder, types |
| `crates/core/Cargo.toml` | Added tokio dev-dependency |
| `crates/cognitive/src/candle_embed.rs` | Implements EmbeddingProvider |
| `crates/engine/src/providers.rs` | EmbeddingProviderRouter, re-exports |
| `crates/engine/src/router.rs` | Uses EmbeddingProviderRouter, Send-safe |
| `crates/engine/src/builder.rs` | Uses router's generate_embedding() |

---

**Date**: April 8, 2026  
**Status**: Embedding pipeline fully wired, 93 tests pass  
**Next**: Implement concrete OpenAI/Voyage/Cohere providers (Phase E.6)
