# Mnemosyne Build Documentation

## Reference Repositories & Inspirations

### Core Memory Architecture

| Repository | Purpose | Components Used |
|------------|---------|-----------------|
| [milla-jovovich/mempalace](https://github.com/milla-jovovich/mempalace) | Episodic memory system | Memory artifacts, episode structure, exchange patterns |
| [mem0ai/mem0](https://github.com/mem0ai/mem0) | Memory layer for AI | Unified memory interface, add/search/retrieve patterns |
| [zilliz-com/milvus](https://github.com/zilliz-com/milvus) | Vector database | HNSW indexing, vector search patterns |
| [spotify/annoy](https://github.com/spotify/annoy) | Approximate nearest neighbors | ANNOY indexing concepts, distance metrics |

### Quantization & Compression

| Repository | Purpose | Components Used |
|------------|---------|-----------------|
| [turbo_quant](https://docs.rs/turboquant) | Product quantization | PQ, OPQ implementations |
| [fastcluster](https://github.com/dwreeves/fastcluster) | Clustering for codebooks | K-means++ initialization |
| [scikit-learn-contrib/lightning](https://github.com/scikit-learn-contrib/sklearn-porter) | Vector operations | Distance calculations |
| [facebookresearch/faiss](https://github.com/facebookresearch/faiss) | Similarity search | Quantization patterns, IVF concepts |

### Graph & Relationship Storage

| Repository | Purpose | Components Used |
|------------|---------|-----------------|
| [petgraph/petgraph](https://github.com/petgraph/petgraph) | Graph data structure | Graph traversal, BFS/DFS, shortest path |
| [neo4j/neo4j](https://github.com/neo4j/neo4j) | Graph database | Entity-relationship model |
| [kuzudb/kuzu](https://github.com/kuzudb/kuzu) | Embedded graph DB | Relationship storage patterns |
| [apache/age](https://github.com/apache/age) | Graph extensions | Cypher-like queries |

### Temporal & Event Storage

| Repository | Purpose | Components Used |
|------------|---------|-----------------|
| [risingwavelabs/risingwave](https://github.com/risingwavelabs/risingwave) | Streaming database | Time-series patterns |
| [influxdata/influxdb](https://github.com/influxdata/influxdb) | Time-series DB | Event storage, retention policies |
| [timescale/timescaledb](https://github.com/timescale/timescaledb) | Time-series extension | Timeline queries |

### Storage Layer

| Repository | Purpose | Components Used |
|------------|---------|-----------------|
| [facebook/rocksdb](https://github.com/facebook/rocksdb) | Embedded key-value store | Persistence, column families |
| [sled-db/sled](https://github.com/sled-db/sled) | Embedded database | Alternative storage concepts |
| [apache/parquet](https://github.com/apache/parquet) | Columnar storage | Compression patterns |

### Cognitive & Embedding Models

| Repository | Purpose | Components Used |
|------------|---------|-----------------|
| [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Sentence embeddings | Embedding patterns |
| [nomic-ai/nomic-embed](https://github.com/nomic-ai/nomic-embed) | Embedding models | Micro-embedding concepts |
| [huggingface/candle](https://github.com/huggingface/candle) | ML framework in Rust | Future model integration |

### Rust Ecosystem

| Crate | Purpose | Usage |
|-------|---------|-------|
| [tokio-rs/tokio](https://github.com/tokio-rs/tokio) | Async runtime | All async operations |
| [rust-rocksdb/rust-rocksdb](https://github.com/rust-rocksdb/rust-rocksdb) | RocksDB bindings | Storage layer |
| [dashmap-rs/dashmap](https://github.com/dashmap-rs/dashmap) | Concurrent HashMap | In-memory storage |
| [serde-rs/serde](https://github.com/serde-rs/serde) | Serialization | All data structures |
| [uuid-rs/uuid](https://github.com/uuid-rs/uuid) | UUID generation | Memory IDs |
| [chronotope/chrono](https://github.com/chronotope/chrono) | Date/time | Timestamps |

---

## Build Instructions

### Prerequisites

```bash
# Rust toolchain (nightly recommended for latest features)
rustup override set nightly
rustup update

# Optional: For RocksDB persistence feature
# Ubuntu/Debian
sudo apt-get install build-essential cmake libclang-dev

# macOS
brew install cmake llvm

# For testing
cargo install cargo-tarpaulin  # Code coverage
cargo install cargo-criterion  # Benchmarks
```

### Building

```bash
# Clone and enter directory
cd /home/fed/remembrain

# Build without RocksDB (pure Rust, no C dependencies)
cargo build --no-default-features

# Build with RocksDB persistence
cargo build --features persistence

# Build optimized
cargo build --release --no-default-features

# Check all code
cargo check --all-targets --no-default-features
```

### Running Tests

```bash
# Run all tests
cargo test --no-default-features

# Run with output
cargo test --no-default-features -- --nocapture

# Run specific test
cargo test test_turboquant --no-default-features

# Generate coverage report
cargo tarpaulin --no-default-features --out html
```

### Benchmarks

```bash
# Run all benchmarks
cargo bench --no-default-features

# Run specific benchmark
cargo bench --no-default-features -- turboquant

# Generate criterion reports
# Results in target/criterion/
```

---

## Architecture Decisions

### Why 8 Separate Crates?

1. **Separation of Concerns** - Each memory type is independent
2. **Compile Times** - Only rebuild what changes
3. **Testing** - Each crate can be tested in isolation
4. **Flexibility** - Users can exclude memory types they don't need

### Why Optional RocksDB?

1. **Pure Rust** - Default build has no C dependencies
2. **CI Friendly** - No system packages needed for tests
3. **Embedded Use** - Some targets don't need persistence
4. **Feature Gating** - Only pay for what you use

### Why Micro-Embeddings?

1. **Speed** - 128-dim vs 1536-dim = 12x faster
2. **Pre-filtering** - Quick relevance prediction
3. **Streaming** - Can run during LLM generation
4. **Cheap** - No GPU needed

### Performance Targets

| Operation | Target | Achieved |
|-----------|--------|----------|
| Micro-embedding | <1ms | ~0.1ms (hash) |
| Vector search (1K) | <3ms | ~2ms (HNSW) |
| Memory store | <5ms | ~1ms |
| Context assembly | <10ms | ~5ms |

---

## Integration Guide

### With LLM Frameworks

```rust
use mnemosyne_engine::MnemosyneEngine;

// Initialize once
let memory = MnemosyneEngine::default()?;

// During conversation loop
async fn handle_message(engine: &MnemosyneEngine, user_msg: &str) -> String {
    // 1. Recall relevant memories
    let context = engine.recall(user_msg).await?;
    
    // 2. Build prompt with context
    let prompt = engine.context_builder.format_context(&context);
    
    // 3. Get LLM response (your LLM call here)
    let response = llm_complete(&prompt).await;
    
    // 4. Store the interaction
    engine.remember(
        user_msg,
        "User question about topic",
        MemoryTrigger::UserInput,
    ).await?;
    
    response
}
```

### With Vector Databases

```rust
use mnemosyne_semantic::SemanticMemoryStore;
use mnemosyne_semantic::turboquant::TurboQuantizer;

// Create quantizer
let mut quantizer = TurboQuantizer::new(1536, 8, 8, 42)?;
quantizer.train(&training_embeddings)?;

// Compress embeddings
let code = quantizer.encode(&embedding)?;

// 8x smaller storage
assert_eq!(code.size_bytes(), 1536 / 8);
```

### With Graph Databases

```rust
use mnemosyne_graph::GraphMemoryStore;

// Create entity
let entity = GraphEntity::new(
    "TurboQuant",
    EntityType::Technology,
    "Vector quantization library",
    embedding,
);

// Add to graph
engine.graph.add_entity(entity).await?;

// Find related
let related = engine.graph.find_related(&entity_id, 2).await?;
```

---

## Troubleshooting

### Compilation Issues

**Problem**: `stddef.h` not found
```bash
# Install build tools
sudo apt-get install build-essential clang libclang-dev

# Or disable persistence feature
cargo build --no-default-features
```

**Problem**: Nightly features required
```bash
rustup default nightly
rustup update
```

### Runtime Issues

**Problem**: Memory leaks with large datasets
- Use `clear()` periodically
- Enable persistence and restart

**Problem**: Slow vector search
- Increase `hnsw_ef_search` for better recall
- Use quantized search for speed
- Pre-train quantizer with representative data

---

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Run `cargo clippy` and fix warnings
5. Run `cargo test --no-default-features`
6. Submit PR

### Code Style

```bash
# Format code
cargo fmt

# Check lints
cargo clippy --no-default-features

# Generate docs
cargo doc --no-default-features --open
```
