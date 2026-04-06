# RemeMnemosyne Build Documentation

## Dependency Transparency

### Pure Rust Crates (No C/C++ Dependencies)

| Crate | Purpose | Source |
|-------|---------|--------|
| rememnemosyne-core | Types, traits | Custom |
| rememnemosyne-semantic | Vector search | Custom TurboQuant + HNSW |
| rememnemosyne-episodic | Chat episodes | Custom |
| rememnemosyne-graph | Entity graph | petgraph (pure Rust) |
| rememnemosyne-temporal | Timeline | Custom |
| rememnemosyne-cognitive | Micro-embeddings | Custom |
| rememnemosyne-engine | Unified API | Custom |
| sled | Embedded storage | Pure Rust |
| tokio | Async runtime | Pure Rust |
| serde | Serialization | Pure Rust |
| petgraph | Graph algorithms | Pure Rust |
| dashmap | Concurrent HashMap | Pure Rust |
| parking_lot | Mutex/RwLock | Pure Rust |
| rayon | Parallelism | Pure Rust |
| uuid | UUID generation | Pure Rust |
| chrono | Date/time | Pure Rust |
| bincode | Binary encoding | Pure Rust |
| half | f16 support | Pure Rust |

### Optional C/C++ Dependencies (Opt-in Only)

| Crate | Purpose | Required By | When Needed |
|-------|---------|-------------|-------------|
| rocksdb | Key-value storage | `persistence` feature | Only if you enable RocksDB |
| librocksdb-sys | RocksDB C++ bindings | rocksdb crate | Only if you enable RocksDB |
| libzstd-sys | Zstandard compression | rocksdb crate | Only if you enable RocksDB |
| liblz4-sys | LZ4 compression | rocksdb crate | Only if you enable RocksDB |

**To avoid C/C++ dependencies entirely**: Use default features (sled storage).

## Build Instructions

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable

# That's it for pure Rust builds!
```

### Build Commands

```bash
# Pure Rust build (recommended)
cargo build

# Same as above (default features are pure Rust)
cargo build --no-default-features
cargo build --features default

# With RocksDB (REQUIRES C++ toolchain)
# Ubuntu/Debian:
sudo apt-get install build-essential cmake clang libclang-dev
cargo build --features persistence

# macOS:
brew install cmake llvm
cargo build --features persistence
```

### Running Tests (Pure Rust)

```bash
# All tests - pure Rust
cargo test

# Specific crate
cargo test -p mnemosyne-semantic

# With output
cargo test -- --nocapture
```

## Reference Repositories

### Core Architecture Inspiration

| Repository | What We Learned |
|------------|-----------------|
| [mempalace](https://github.com/milla-jovovich/mempalace) | Episode structure, memory artifacts |
| [mem0](https://github.com/mem0ai/mem0) | Unified memory interface |

### Vector Search Algorithms

| Repository | What We Learned |
|------------|-----------------|
| [faiss](https://github.com/facebookresearch/faiss) | Quantization patterns, IVF |
| [annoy](https://github.com/spotify/annoy) | ANNOY indexing concepts |
| [hnsw](https://github.com/nmslib/hnswlib) | HNSW algorithm details |

### Graph Storage

| Repository | What We Learned |
|------------|-----------------|
| [petgraph](https://github.com/petgraph/petgraph) | Graph data structures (used directly) |
| [neo4j](https://github.com/neo4j/neo4j) | Entity-relationship model |
| [kuzu](https://github.com/kuzudb/kuzu) | Embedded graph patterns |

### Rust Ecosystem

| Crate | Repository | Notes |
|-------|------------|-------|
| sled | [sled-db/sled](https://github.com/sled-db/sled) | Pure Rust embedded DB |
| petgraph | [petgraph/petgraph](https://github.com/petgraph/petgraph) | Pure Rust graph |
| tokio | [tokio-rs/tokio](https://github.com/tokio-rs/tokio) | Pure Rust async |
| dashmap | [dashmap-rs/dashmap](https://github.com/dashmap-rs/dashmap) | Pure Rust concurrent |

## Storage Backend Architecture

The storage layer uses a trait-based design for swappable backends:

```rust
pub trait StorageBackend: Send + Sync {
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()>;
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn delete(&self, key: &[u8]) -> Result<()>;
    fn flush(&self) -> Result<()>;
}
```

### Available Backends

| Backend | Crate | Pure Rust | Performance | Use Case |
|---------|-------|-----------|-------------|----------|
| Sled | sled | Yes | Good | Default, general purpose |
| RocksDB | rocksdb | No (C++) | Excellent | High-write workloads |

### Adding a New Backend

Implement the `StorageBackend` trait:

```rust
pub struct MyBackend { /* ... */ }

impl StorageBackend for MyBackend {
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        // Your implementation
    }
    // ...
}
```

## Performance Benchmarks

```bash
# Install criterion
cargo install cargo-criterion

# Run benchmarks
cargo bench

# View results
open target/criterion/report/index.html
```

## Contributing

1. Ensure your changes maintain pure Rust by default
2. Any C/C++ dependency must be behind a feature flag
3. Run `cargo clippy` and fix warnings
4. Run `cargo test` and ensure all tests pass
5. Update documentation if adding new features

## License

MIT
