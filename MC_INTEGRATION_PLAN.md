# Memory Caching (MC) Integration Plan

> Based on: arXiv:2602.24281 — "Memory Caching: RNNs with Growing Memory"
> by Ali Behrouz, Zeman Li, Yuan Deng, Peilin Zhong, Meisam Razaviyayn, Vahab Mirrokni

## Overview

This document describes the integration of Memory Caching (MC) concepts into RemeMnemosyne. MC addresses RemeMnemosyne's three core scaling problems:

1. **F1 collapse at scale** — Recall drops from 0.80 to 0.01 at 10k memories because individual memory search degrades without a coarse-grained first pass.
2. **CognitiveEngine unimplemented** — The trait exists but has no implementation. Intent prefetching is stubbed, transition matrices are dead code.
3. **Binary context assembly** — LayeredContextStack loads L2-L4 as all-or-nothing. Context formatting includes or excludes memories with no graduated weighting.

MC's core insight: **cache compressed segment checkpoints and use query-dependent gating to retrieve from them**, interpolating between O(L) fixed memory and O(L²) growing memory at O(NL) cost.

## Architecture Mapping

| MC Concept | RemeMnemosyne Implementation |
|---|---|
| Segment checkpoint | `MemoryCheckpoint` — compressed summary of a time window of memories |
| Gated Residual Memory (GRM) | `contribution_weights` on `ContextBundle` — query-memory similarity gates rendering depth |
| Sparse Selective Caching (SSC) | `SSCRouter` — Top-k checkpoint routing before fine-grained HNSW search |
| MeanPool(S^(i)) | `SegmentProfile::importance_weighted_embedding` — importance-weighted checkpoint fingerprint |
| γ_t gating | `cosine_similarity(query_embedding, memory_embedding)` → softmax normalization |
| Online memory | Current HNSW search results (always included) |
| Cached memories | `CheckpointStore` results (expanding into individual memories within the window) |

## Feature Flags

All MC integration is behind feature flags for zero-overhead opt-in:

```toml
[features]
# Phase 1: Segment checkpointing (can be used alone)
mc-checkpoints = []

# Phase 2: Gated context assembly (depends on Phase 1 for checkpoint embeddings)
mc-gated-context = ["mc-checkpoints"]

# Phase 3: SSC router (depends on Phase 1)
mc-ssc = ["mc-checkpoints"]
```

## Phase 1: Segment Checkpointing

### Problem Solved
At scale, HNSW alone can't differentiate signal from noise. Checkpoints provide O(N) coarse search before O(M) fine search.

### New Types

**`MemoryCheckpoint`** (`crates/core/src/types.rs`):
- Compressed summary of a time window of memories
- Fields: `id`, `time_window_start/end`, `summary_embedding` (importance-weighted pooled), `summary_text`, `memory_count`, `memory_ids`, `key_entities`, `palace_location`, `session_id`, `importance_ceiling`, `embedding_method`, `created_at`

**`CheckpointEmbeddingMethod`** (`crates/core/src/types.rs`):
- `MeanPool` — simple average (fast but less discriminative)
- `ImportanceWeightedPool` — weighted by importance level (default, more discriminative)
- `MaxPool` — maximum per-dimension (most discriminative, higher variance)

**`CheckpointStore`** (`crates/episodic/src/checkpoint.rs`):
- In-memory checkpoint storage with DashMap
- Dual-trigger creation: every N memories (default 50) OR every T seconds (default 1800)
- `search_checkpoints()` — cosine similarity over checkpoint embeddings
- `expand_checkpoint()` — returns memory_ids within a checkpoint for fine-grained search
- Eviction when exceeding max_checkpoints (default 200)

### Risk Mitigations

| Risk | Mitigation |
|---|---|
| Checkpoint granularity too bland | `ImportanceWeightedPool` default — high-importance memories dominate the fingerprint. `MaxPool` fallback for maximum discrimination |
| Cold start (no checkpoints yet) | Falls through to bare HNSW. This is MC's N=1 baseline — standard RNN behavior |
| Hard-gate recall loss | Checkpoint boost is 1.3× soft multiplier on relevance, not a hard filter. HNSW finds everything; checkpoints rank them better |

### Flow

```
MemoryRouter::query()
    ↓
[Phase 1: Checkpoint routing]
    ↓ query_embedding → CheckpointStore::search_checkpoints(k=5)
    ↓ select checkpoints with score ≥ 0.3
    ↓ expand high-scoring checkpoints → checkpoint_memory_ids
    ↓
[Existing: Semantic HNSW search]
    ↓ boost relevance by 1.3× for checkpoint_memory_ids
    ↓
[Phase 1: Context assembly]
    ↓ load_checkpoint_context() prepends segment summaries to L3
```

## Phase 2: Gated Context Assembly (GRM)

### Problem Solved
Small models get "distracted" by irrelevant memory content (2B recall drops 29.9% → 20.6%). GRM gates each memory's contribution based on query-memory similarity.

### Changes

**`ContextBundle`** — new field:
- `contribution_weights: HashMap<MemoryId, f32>` — MC's γ_t^(i) gates per memory

**`ContextBuilderEngine::build_context()`** — compute gates:
- `γ = cosine_similarity(query_embedding, memory_embedding)`
- Softmax normalize over top-k candidates (not all — prevents weight dilution)
- Pass gates to ContextBundle via `add_memory_weighted()`

**Format strategies** — weight-tiered rendering:
- γ > 0.7: Full verbatim content from Drawer (up to 300 chars)
- 0.3 < γ < 0.7: Closet summary (up to 150 chars)
- γ < 0.3: One-line reference only

**`LayeredContextStack`** — auto-escalation:
- New `should_escalate()` method: if query embedding has low similarity to current layer content, escalate to deeper layer
- New `layer_embedding: Option<Vec<f32>>` on `ContextLayer` — mean-pooled embedding of the layer's source memories (MC's MeanPool(S^(i)))

### Risk Mitigations

| Risk | Mitigation |
|---|---|
| Softmax dilution over many low-sim candidates | Top-k softmax: normalize only over `max_memories` candidates, not all. Remaining get minimal 0.05 weight (not zero — preserves recall) |
| 2B model distraction | Weight-tiered formatting: low-weight memories become single-line references, drastically reducing noise |

## Phase 3: Learned SSC Router

### Problem Solved
Intent-based prefetching is stubbed with empty match arms. Transition matrix is dead code. At 10k+ memories, even HNSW needs pre-filtering.

### New Modules

**`SSCRouter`** (`crates/cognitive/src/ssc_router.rs`):
- Stores `SegmentProfile` for each checkpoint: mean and importance-weighted embeddings
- `route(query_embedding, candidate_ids) → top-k checkpoint IDs`
- Score via `cosine_similarity(query_embedding, profile.importance_weighted_embedding)`
- Configuration: `top_k` (default 5), `expansion_threshold` (default 0.3)

**`CognitiveEngineImpl`** (`crates/cognitive/src/engine.rs`):
- First implementation of the `CognitiveEngine` trait (currently unimplemented)
- Delegates to `MicroEmbedder`, `IntentDetector`, `ContextPredictor`, `MemoryPrefetcher`, optional `SSCRouter`

**`ContextPredictor`** — activated transition matrix:
- `record_transition(from_state, to_state)` — populates the previously-dead transition_matrix
- Blend transition probabilities (30%) with embedding similarity (70%) in `predict()` only after >10 transitions recorded

**`MemoryPrefetcher`** — implemented `intent_based_prefetch()`:
- Routes to cluster centroids based on intent: recall/search gets 1.3× boost, analyze gets 1.1×
- Uses `last_query_embedding` for similarity computation

### Risk Mitigations

| Risk | Mitigation |
|---|---|
| Transition matrix cold start | Only blend after >10 transitions recorded. Before threshold, pure embedding similarity |
| Checkpoint embedding collision | Dual fingerprint: `mean_embedding` + `importance_weighted_embedding`. SSC uses the weighted version for scoring |

## Dependency Graph

```
Phase 1: Segment Checkpointing
  ├── core/types.rs (MemoryCheckpoint, CheckpointEmbeddingMethod, enum variants)
  ├── core/math.rs (shared vector utilities)
  ├── episodic/checkpoint.rs (CheckpointStore, CheckpointConfig)
  └── engine/router.rs (checkpoint creation trigger + search boost)

Phase 2: Gated Context Assembly (depends on Phase 1 for checkpoint embeddings)
  ├── core/types.rs (contribution_weights on ContextBundle)
  ├── engine/context.rs (GRM gate computation, weight-tiered formatting)
  └── engine/context_stack.rs (auto-escalation, layer embeddings)

Phase 3: SSC Router (depends on Phase 1 checkpoints)
  ├── cognitive/ssc_router.rs (SSCRouter, SegmentProfile)
  ├── cognitive/engine.rs (CognitiveEngineImpl)
  ├── cognitive/predictor.rs (activate transition_matrix)
  ├── cognitive/prefetcher.rs (implement intent_based_prefetch)
  ├── cognitive/micro_embed.rs (extract_entities_ner)
  └── engine/router.rs (SSC query routing)
```

## New External Dependencies

**None.** All implementations use `dashmap`, `uuid`, `chrono`, `serde`, `parking_lot`, `rayon` — already in workspace Cargo.toml.

## File Changes Summary

| File | Phase | Action |
|---|---|---|
| `crates/core/src/math.rs` | 1 | NEW — shared vector math |
| `crates/core/src/types.rs` | 1,2 | Add MemoryCheckpoint, CheckpointEmbeddingMethod, MemoryType::Checkpoint, EventType::Checkpoint, contribution_weights on ContextBundle |
| `crates/core/src/lib.rs` | 1 | Add `pub mod math` export |
| `crates/episodic/src/checkpoint.rs` | 1 | NEW — CheckpointStore, CheckpointConfig |
| `crates/episodic/src/lib.rs` | 1 | Add `pub mod checkpoint` export |
| `crates/engine/src/router.rs` | 1,3 | Wire CheckpointStore, SSC router, checkpoint-aware query |
| `crates/engine/src/context.rs` | 2 | GRM gates, weight-tiered formatting |
| `crates/engine/src/context_stack.rs` | 1,2 | load_checkpoint_context, should_escalate, layer_embedding |
| `crates/engine/src/builder.rs` | 1 | Wire CheckpointStore initialization |
| `crates/engine/Cargo.toml` | 1,2,3 | Add mc-checkpoints, mc-gated-context, mc-ssc features |
| `crates/cognitive/src/ssc_router.rs` | 3 | NEW — SSCRouter, SegmentProfile |
| `crates/cognitive/src/engine.rs` | 3 | NEW — CognitiveEngineImpl |
| `crates/cognitive/src/predictor.rs` | 3 | Activate transition_matrix, add record_transition |
| `crates/cognitive/src/prefetcher.rs` | 3 | Implement intent_based_prefetch |
| `crates/cognitive/src/micro_embed.rs` | 3 | Add extract_entities_ner |
| `crates/cognitive/src/lib.rs` | 3 | Add ssc_router, engine module exports |