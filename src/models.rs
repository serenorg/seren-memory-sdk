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
