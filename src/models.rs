// ABOUTME: Shared types used across the SDK.
// ABOUTME: CachedMemory, SyncResult, and bootstrap types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A memory record stored in the local SQLite cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedMemory {
    pub id: Uuid,
    pub content: String,
    pub memory_type: String,
    pub metadata: serde_json::Value,
    pub embedding: Vec<f32>,
    pub relevance_score: f64,
    pub created_at: DateTime<Utc>,
    pub synced: bool,
    pub cloud_id: Option<Uuid>,
}

/// Result of a sync operation.
#[derive(Debug, Clone, Default)]
pub struct SyncResult {
    pub pushed: usize,
    pub pulled: usize,
    pub errors: Vec<String>,
}

/// A memory returned from the cloud API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudMemory {
    pub id: Uuid,
    pub content: String,
    pub memory_type: String,
    pub metadata: serde_json::Value,
    pub relevance_score: f64,
    pub is_pinned: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Input for session bootstrap.
#[derive(Debug, Clone, Serialize)]
pub struct BootstrapInput {
    pub project_id: Option<Uuid>,
    pub org_id: Option<Uuid>,
    pub token_budget: Option<usize>,
}

/// Cloud session context returned by the MCP session_bootstrap tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CloudSessionContext {
    pub memories_by_type: std::collections::HashMap<String, Vec<String>>,
    pub total_memories: usize,
}

/// Assembled session context for LLM system prompt injection.
#[derive(Debug, Clone)]
pub struct SessionContext {
    pub memories_by_type: std::collections::HashMap<String, Vec<String>>,
    pub total_memories: usize,
    pub assembled_prompt: String,
    pub source: ContextSource,
}

/// Where the bootstrap context came from.
#[derive(Debug, Clone, PartialEq)]
pub enum ContextSource {
    /// Fresh data from the cloud.
    Cloud,
    /// Stale data from local cache (offline mode).
    LocalCache,
}

/// A single result from the cloud recall tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    pub content: String,
    pub memory_type: String,
    pub relevance_score: f64,
}
