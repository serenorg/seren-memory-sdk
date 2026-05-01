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
    /// Per-memory feedback signal (e.g. +1 / -1). `None` means no signal.
    #[serde(default)]
    pub feedback_signal: Option<i32>,
    /// Whether the user has pinned this memory.
    #[serde(default)]
    pub pinned: bool,
}

/// User-provided feedback on a memory. The MCP `mark_feedback` tool accepts
/// these as `+1` / `-1` integers; serialized as such for wire compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackSignal {
    Positive,
    Negative,
}

impl FeedbackSignal {
    pub fn as_i32(self) -> i32 {
        match self {
            FeedbackSignal::Positive => 1,
            FeedbackSignal::Negative => -1,
        }
    }
}

impl Serialize for FeedbackSignal {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_i32(self.as_i32())
    }
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

/// A reference to a single memory inside `memories_by_type`. Carries the
/// cloud id alongside the content so frontends can pin / forget / dedupe
/// individual entries instead of treating the section as opaque text.
///
/// On the wire this is `{ "id": "...", "content": "..." }`. To keep the
/// SDK compatible with older cloud builds that still emit bare strings,
/// `Deserialize` also accepts a plain JSON string (in which case `id`
/// falls back to `Uuid::nil()`).
#[derive(Debug, Clone, Serialize)]
pub struct MemoryRef {
    pub id: Uuid,
    pub content: String,
}

impl<'de> Deserialize<'de> for MemoryRef {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Object { id: Uuid, content: String },
            Plain(String),
        }
        match Repr::deserialize(deserializer)? {
            Repr::Object { id, content } => Ok(MemoryRef { id, content }),
            Repr::Plain(content) => Ok(MemoryRef {
                id: Uuid::nil(),
                content,
            }),
        }
    }
}

/// Cloud session context returned by the MCP session_bootstrap tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CloudSessionContext {
    pub memories_by_type: std::collections::HashMap<String, Vec<MemoryRef>>,
    pub total_memories: usize,
}

/// Assembled session context for LLM system prompt injection.
#[derive(Debug, Clone)]
pub struct SessionContext {
    pub memories_by_type: std::collections::HashMap<String, Vec<MemoryRef>>,
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
    /// Cloud memory id. Falls back to `Uuid::nil()` for older cloud builds
    /// that have not yet been updated to emit ids on recall.
    #[serde(default)]
    pub id: Uuid,
    pub content: String,
    pub memory_type: String,
    pub relevance_score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bm25_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_score: Option<f64>,
}

/// A cached memory with hybrid-search scores attached.
#[derive(Debug, Clone)]
pub struct RankedCachedMemory {
    pub memory: CachedMemory,
    pub rrf_score: f64,
    pub vector_score: Option<f64>,
    pub bm25_score: Option<f64>,
}
