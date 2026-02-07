// ABOUTME: Local SQLite + sqlite-vec cache for offline memory access.
// ABOUTME: Mirrors a subset of cloud memories for fast vector search.

use std::path::Path;
use std::sync::Once;

use chrono::{DateTime, Utc};
use rusqlite::Connection;
use uuid::Uuid;

use crate::error::SdkResult;
use crate::models::CachedMemory;

/// Register sqlite-vec as an auto-extension (once per process).
fn register_vec_extension() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

pub struct LocalCache {
    conn: Connection,
}

impl LocalCache {
    /// Open or create the local cache database at the given path.
    pub fn open(path: &Path) -> SdkResult<Self> {
        register_vec_extension();
        let conn = Connection::open(path)?;
        let cache = Self { conn };
        cache.init_schema()?;
        Ok(cache)
    }

    /// Open an in-memory database (for testing).
    #[cfg(test)]
    pub fn open_in_memory() -> SdkResult<Self> {
        register_vec_extension();
        let conn = Connection::open_in_memory()?;
        let cache = Self { conn };
        cache.init_schema()?;
        Ok(cache)
    }

    fn init_schema(&self) -> SdkResult<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS cached_memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                embedding BLOB NOT NULL,
                relevance_score REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                synced BOOLEAN DEFAULT 0,
                cloud_id TEXT
            );

            CREATE TABLE IF NOT EXISTS sync_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[1536]
            );",
        )?;

        Ok(())
    }

    /// Insert a memory into the local cache.
    pub fn insert_memory(&self, memory: &CachedMemory) -> SdkResult<()> {
        let embedding_bytes = f32_slice_to_bytes(&memory.embedding);
        let id_str = memory.id.to_string();

        self.conn.execute(
            "INSERT OR REPLACE INTO cached_memories (id, content, memory_type, metadata, embedding, relevance_score, created_at, synced, cloud_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                id_str,
                memory.content,
                memory.memory_type,
                serde_json::to_string(&memory.metadata).unwrap_or_default(),
                embedding_bytes,
                memory.relevance_score,
                memory.created_at.to_rfc3339(),
                memory.synced as i32,
                memory.cloud_id.map(|id| id.to_string()),
            ],
        )?;

        // vec0 doesn't support INSERT OR REPLACE — delete first, then insert.
        self.conn.execute(
            "DELETE FROM vec_memories WHERE id = ?1",
            rusqlite::params![id_str],
        )?;
        self.conn.execute(
            "INSERT INTO vec_memories (id, embedding) VALUES (?1, ?2)",
            rusqlite::params![id_str, embedding_bytes],
        )?;

        Ok(())
    }

    /// Search for similar memories using vector similarity.
    pub fn vector_search(&self, query_embedding: &[f32], limit: usize) -> SdkResult<Vec<CachedMemory>> {
        let query_bytes = f32_slice_to_bytes(query_embedding);

        let mut stmt = self.conn.prepare(
            "SELECT cm.id, cm.content, cm.memory_type, cm.metadata, cm.embedding,
                    cm.relevance_score, cm.created_at, cm.synced, cm.cloud_id
             FROM vec_memories v
             INNER JOIN cached_memories cm ON cm.id = v.id
             WHERE v.embedding MATCH ?1 AND k = ?2
             ORDER BY v.distance",
        )?;

        let rows = stmt.query_map(rusqlite::params![query_bytes, limit as i64], |row| {
            Ok(parse_memory_row(row))
        })?;

        let mut results = Vec::new();
        for row in rows {
            match row {
                Ok(Ok(memory)) => results.push(memory),
                Ok(Err(e)) => tracing::warn!("failed to parse cached memory: {e}"),
                Err(e) => tracing::warn!("failed to read row: {e}"),
            }
        }

        Ok(results)
    }

    /// Get all memories that haven't been synced to the cloud yet.
    pub fn get_pending_uploads(&self) -> SdkResult<Vec<CachedMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, content, memory_type, metadata, embedding,
                    relevance_score, created_at, synced, cloud_id
             FROM cached_memories
             WHERE synced = 0",
        )?;

        let rows = stmt.query_map([], |row| Ok(parse_memory_row(row)))?;

        let mut results = Vec::new();
        for row in rows {
            match row {
                Ok(Ok(memory)) => results.push(memory),
                Ok(Err(e)) => tracing::warn!("failed to parse pending memory: {e}"),
                Err(e) => tracing::warn!("failed to read row: {e}"),
            }
        }

        Ok(results)
    }

    /// Mark a memory as synced to the cloud.
    pub fn mark_synced(&self, id: Uuid, cloud_id: Uuid) -> SdkResult<()> {
        self.conn.execute(
            "UPDATE cached_memories SET synced = 1, cloud_id = ?1 WHERE id = ?2",
            rusqlite::params![cloud_id.to_string(), id.to_string()],
        )?;
        Ok(())
    }

    /// Get the last sync timestamp.
    pub fn get_last_sync_timestamp(&self) -> SdkResult<Option<DateTime<Utc>>> {
        let mut stmt = self
            .conn
            .prepare("SELECT value FROM sync_state WHERE key = 'last_sync'")?;

        let result: Option<String> = stmt
            .query_row([], |row| row.get(0))
            .ok();

        match result {
            Some(ts) => Ok(ts.parse::<DateTime<Utc>>().ok()),
            None => Ok(None),
        }
    }

    /// Set the last sync timestamp.
    pub fn set_last_sync_timestamp(&self, ts: DateTime<Utc>) -> SdkResult<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO sync_state (key, value) VALUES ('last_sync', ?1)",
            rusqlite::params![ts.to_rfc3339()],
        )?;
        Ok(())
    }

    /// Count all cached memories.
    pub fn count(&self) -> SdkResult<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM cached_memories", [], |row| row.get(0))?;
        Ok(count as usize)
    }
}

fn f32_slice_to_bytes(slice: &[f32]) -> Vec<u8> {
    slice.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn parse_memory_row(row: &rusqlite::Row) -> Result<CachedMemory, String> {
    let id_str: String = row.get(0).map_err(|e| e.to_string())?;
    let id = Uuid::parse_str(&id_str).map_err(|e| e.to_string())?;

    let content: String = row.get(1).map_err(|e| e.to_string())?;
    let memory_type: String = row.get(2).map_err(|e| e.to_string())?;

    let metadata_str: String = row.get(3).map_err(|e| e.to_string())?;
    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_str).unwrap_or(serde_json::json!({}));

    let embedding_bytes: Vec<u8> = row.get(4).map_err(|e| e.to_string())?;
    let embedding = bytes_to_f32_vec(&embedding_bytes);

    let relevance_score: f64 = row.get(5).map_err(|e| e.to_string())?;

    let created_at_str: String = row.get(6).map_err(|e| e.to_string())?;
    let created_at = created_at_str
        .parse::<DateTime<Utc>>()
        .map_err(|e| e.to_string())?;

    let synced_int: i32 = row.get(7).map_err(|e| e.to_string())?;
    let synced = synced_int != 0;

    let cloud_id_str: Option<String> = row.get(8).map_err(|e| e.to_string())?;
    let cloud_id = cloud_id_str.and_then(|s| Uuid::parse_str(&s).ok());

    Ok(CachedMemory {
        id,
        content,
        memory_type,
        metadata,
        embedding,
        relevance_score,
        created_at,
        synced,
        cloud_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn test_memory(content: &str, synced: bool) -> CachedMemory {
        CachedMemory {
            id: Uuid::new_v4(),
            content: content.to_string(),
            memory_type: "semantic".to_string(),
            metadata: serde_json::json!({}),
            embedding: vec![0.1; 1536],
            relevance_score: 1.0,
            created_at: Utc::now(),
            synced,
            cloud_id: if synced { Some(Uuid::new_v4()) } else { None },
        }
    }

    #[test]
    fn open_and_insert_memory() {
        let cache = LocalCache::open_in_memory().unwrap();
        let mem = test_memory("test content", false);
        cache.insert_memory(&mem).unwrap();
        assert_eq!(cache.count().unwrap(), 1);
    }

    #[test]
    fn get_pending_uploads_returns_unsynced() {
        let cache = LocalCache::open_in_memory().unwrap();

        let unsynced = test_memory("not synced", false);
        let synced = test_memory("already synced", true);

        cache.insert_memory(&unsynced).unwrap();
        cache.insert_memory(&synced).unwrap();

        let pending = cache.get_pending_uploads().unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, unsynced.id);
    }

    #[test]
    fn mark_synced_removes_from_pending() {
        let cache = LocalCache::open_in_memory().unwrap();
        let mem = test_memory("to sync", false);
        cache.insert_memory(&mem).unwrap();

        assert_eq!(cache.get_pending_uploads().unwrap().len(), 1);

        let cloud_id = Uuid::new_v4();
        cache.mark_synced(mem.id, cloud_id).unwrap();

        assert_eq!(cache.get_pending_uploads().unwrap().len(), 0);
    }

    #[test]
    fn sync_timestamp_round_trip() {
        let cache = LocalCache::open_in_memory().unwrap();

        assert!(cache.get_last_sync_timestamp().unwrap().is_none());

        let ts = Utc::now();
        cache.set_last_sync_timestamp(ts).unwrap();

        let loaded = cache.get_last_sync_timestamp().unwrap().unwrap();
        assert_eq!(
            loaded.timestamp_millis() / 1000,
            ts.timestamp_millis() / 1000
        );
    }

    #[test]
    fn vector_search_returns_results() {
        let cache = LocalCache::open_in_memory().unwrap();

        // Insert a memory with a known embedding
        let mut mem = test_memory("known vector", false);
        mem.embedding = vec![1.0; 1536];
        cache.insert_memory(&mem).unwrap();

        // Insert another memory with a different embedding
        let mut mem2 = test_memory("different vector", false);
        mem2.embedding = vec![0.0; 1536];
        cache.insert_memory(&mem2).unwrap();

        // Search with the known embedding — should find both, closest first
        let results = cache.vector_search(&vec![1.0; 1536], 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].content, "known vector");
    }

    #[test]
    fn insert_replace_updates_existing() {
        let cache = LocalCache::open_in_memory().unwrap();
        let mut mem = test_memory("original", false);
        cache.insert_memory(&mem).unwrap();

        mem.content = "updated".to_string();
        cache.insert_memory(&mem).unwrap();

        assert_eq!(cache.count().unwrap(), 1);
    }

    #[test]
    fn database_file_is_created() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");

        let _cache = LocalCache::open(&path).unwrap();
        assert!(path.exists());
    }
}
