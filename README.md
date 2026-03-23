# seren-memory-sdk

Rust SDK for [Seren](https://serendb.com) agent memory — local SQLite cache with vector search, cloud sync, and session bootstrap.

## Overview

This crate provides the client-side memory layer for Seren Desktop and other Seren-powered applications. It handles:

- **Local cache** — SQLite + sqlite-vec for embedding-based vector search, works offline
- **Cloud sync** — Push/pull memories to Seren's cloud memory service
- **Session bootstrap** — Assemble project memory context for AI system prompts
- **MCP client** — Remember and recall via Seren's MCP endpoint

## Usage

```rust
use seren_memory_sdk::{cache::LocalCache, client::MemoryClient, sync::SyncEngine};

// Local cache (works offline)
let cache = LocalCache::open("memory.db")?;
cache.insert_memory(&memory)?;
let results = cache.vector_search(&embedding, 10)?;

// Cloud client
let client = MemoryClient::new("https://memory.serendb.com".into(), api_key);
client.remember("The deploy target is Cloudflare R2", "semantic", None, None).await?;
let results = client.recall("deploy target", None, Some(5)).await?;

// Sync (push pending local → cloud, pull new cloud → local)
let engine = SyncEngine::new(cache, client);
let result = engine.sync(user_id, project_id).await?;
```

## Modules

| Module | Description |
|--------|-------------|
| `cache` | SQLite local cache with sqlite-vec vector search |
| `client` | HTTP client for cloud memory API and MCP tool calls |
| `sync` | Bidirectional sync engine (local ↔ cloud) |
| `bootstrap` | Session context assembly with token budgeting |
| `models` | Shared types (CloudMemory, CachedMemory, RecallResult) |
| `error` | Error types wrapping rusqlite, reqwest, serde |

## License

MIT — see [LICENSE](LICENSE).
