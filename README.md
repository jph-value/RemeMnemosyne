# Mnemosyne

A high-performance agentic memory engine written in pure Rust with unified and dynamic memory coordination.

## Architecture

Mnemosyne implements a complete cognitive memory system for LLM agents with four distinct memory types:

```
agent_memory_engine
├── crates/
│   ├── core          # Core types, traits, and queries
│   ├── semantic      # TurboQuant compressed vector memory
│   ├── episodic      # Conversation episodes (mempalace-style)
│   ├── graph         # Entity relationships (petgraph-based)
│   ├── temporal      # Timeline events and queries
│   ├── cognitive     # Micro-embeddings and prediction
│   ├── storage       # RocksDB persistence layer
│   └── engine        # Unified API and memory router
```

## Memory Types

| Type | Purpose | Implementation |
|------|---------|----------------|
| Semantic | Vector similarity search | TurboQuant + HNSW index |
| Episodic | Conversation history | Episodes, exchanges, decisions |
| Graph | Entity relationships | petgraph + relationship tracking |
| Temporal | Timeline events | Chronological event storage |

## Key Features

### TurboQuant Integration
- Product Quantization (PQ)
- Optimized PQ (OPQ)  
- Polar Quantization
- QJL transforms
- 8x compression for 1536-dim embeddings

### Cognitive Engine
- Micro-embeddings for fast pre-retrieval
- Intent detection
- Context prediction
- Memory prefetching

### Performance Targets
- Micro-embedding: <1ms
- Vector search: <3ms  
- Memory assembly: <5ms
- Total memory latency: <10ms

## Usage

```rust
use mnemosyne_engine::MnemosyneEngine;

// Create engine
let engine = MnemosyneEngine::default()?;

// Remember something
let id = engine.remember(
    "TurboQuant compresses vectors to 4-8 bits",
    "TurboQuant quantization",
    MemoryTrigger::Insight,
).await?;

// Recall relevant memories
let context = engine.recall("quantization").await?;
let formatted = engine.context_builder.format_context(&context);
```

## Features

```toml
[dependencies]
mnemosyne-engine = { path = "crates/engine" }

# Without persistence (default)
mnemosyne-engine = "0.1.0"

# With RocksDB persistence
mnemosyne-engine = { version = "0.1.0", features = ["persistence"] }
```

## Design Principles

1. **Deterministic** - Agents recall the same memories reliably
2. **Extremely fast** - Memory lookup under 5ms
3. **Composable** - Different memory strategies plug in easily
4. **Token-efficient** - Context reconstruction minimizes prompt size
5. **Persistent** - Memory survives agent restarts (with persistence feature)

## Modules

### Core (`mnemosyne-core`)
- `MemoryArtifact` - Canonical memory representation
- `MemoryStore` trait - Unified interface
- `MemoryQuery` - Query builders

### Semantic (`mnemosyne-semantic`)
- `TurboQuantizer` - PQ/OPQ quantization
- `HNSWIndex` - Approximate nearest neighbor
- `SemanticMemoryStore` - Full vector memory

### Episodic (`mnemosyne-episodic`)
- `Episode` - Conversation segment
- `Exchange` - User/assistant turn
- `Decision` - Captured decision points
- `EpisodeSummarizer` - Auto-summarization

### Graph (`mnemosyne-graph`)
- `GraphEntity` - Entity with relationships
- `GraphRelationship` - Typed relationships
- `GraphTraversal` - BFS/DFS/shortest path
- Entity clustering

### Temporal (`mnemosyne-temporal`)
- `TemporalEvent` - Time-stamped events
- `Timeline` - Chronological sequence
- Timeline queries and filtering

### Cognitive (`mnemosyne-cognitive`)
- `MicroEmbedder` - Fast 128-dim embeddings
- `IntentDetector` - Query intent classification
- `ContextPredictor` - Pre-retrieval prediction
- `MemoryPrefetcher` - Proactive memory loading

### Engine (`mnemosyne-engine`)
- `MnemosyneEngine` - Main API
- `MemoryRouter` - Unified query routing
- `ContextBuilder` - LLM context assembly
- `AgentMemory` trait - Agent interface

## License

MIT
