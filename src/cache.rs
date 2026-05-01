// ABOUTME: Local SQLite + sqlite-vec cache for offline memory access.
// ABOUTME: Mirrors a subset of cloud memories for fast vector search.

use std::path::Path;
use std::sync::Once;

use chrono::{DateTime, Utc};
use rusqlite::Connection;
use uuid::Uuid;

use std::collections::HashMap;

use crate::error::SdkResult;
use crate::models::{CachedMemory, RankedCachedMemory};

/// Reciprocal Rank Fusion constant — matches the cloud recall implementation.
const RRF_K: f64 = 60.0;

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
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
                id UNINDEXED,
                content,
                memory_type,
                tokenize='porter unicode61'
            );",
        )?;

        self.backfill_fts_if_needed()?;
        Ok(())
    }

    /// Populate `fts_memories` from `cached_memories` exactly once per cache
    /// (idempotent across cold-starts via the `fts_backfilled` sync_state key).
    fn backfill_fts_if_needed(&self) -> SdkResult<()> {
        let already: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM sync_state WHERE key = 'fts_backfilled'",
                [],
                |row| row.get(0),
            )
            .ok();
        if already.as_deref() == Some("1") {
            return Ok(());
        }

        let tx = self.conn.unchecked_transaction()?;

        let rows: Vec<(String, String, String)> = {
            let mut stmt = tx.prepare(
                "SELECT id, content, memory_type FROM cached_memories",
            )?;
            let mapped = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?;
            mapped.collect::<rusqlite::Result<Vec<_>>>()?
        };

        {
            let mut ins = tx.prepare(
                "INSERT INTO fts_memories (id, content, memory_type) VALUES (?1, ?2, ?3)",
            )?;
            for (id, content, mtype) in &rows {
                ins.execute(rusqlite::params![id, content, mtype])?;
            }
        }

        tx.execute(
            "INSERT OR REPLACE INTO sync_state (key, value) VALUES ('fts_backfilled', '1')",
            [],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Insert a memory into the local cache. All three tables
    /// (`cached_memories`, `vec_memories`, `fts_memories`) are updated
    /// atomically inside a single transaction.
    pub fn insert_memory(&self, memory: &CachedMemory) -> SdkResult<()> {
        let embedding_bytes = f32_slice_to_bytes(&memory.embedding);
        let id_str = memory.id.to_string();

        let tx = self.conn.unchecked_transaction()?;

        tx.execute(
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

        // vec0 and fts5 don't support INSERT OR REPLACE — delete first, then insert.
        tx.execute(
            "DELETE FROM vec_memories WHERE id = ?1",
            rusqlite::params![id_str],
        )?;
        tx.execute(
            "INSERT INTO vec_memories (id, embedding) VALUES (?1, ?2)",
            rusqlite::params![id_str, embedding_bytes],
        )?;

        tx.execute(
            "DELETE FROM fts_memories WHERE id = ?1",
            rusqlite::params![id_str],
        )?;
        tx.execute(
            "INSERT INTO fts_memories (id, content, memory_type) VALUES (?1, ?2, ?3)",
            rusqlite::params![id_str, memory.content, memory.memory_type],
        )?;

        tx.commit()?;
        Ok(())
    }

    /// Search for similar memories using vector similarity.
    pub fn vector_search(&self, query_embedding: &[f32], limit: usize) -> SdkResult<Vec<CachedMemory>> {
        Ok(self
            .vector_search_scored(query_embedding, limit)?
            .into_iter()
            .map(|(memory, _)| memory)
            .collect())
    }

    /// Vector search returning each match alongside its raw vec0 distance
    /// (smaller = more similar). Used by `hybrid_search` to expose
    /// `vector_score` to callers.
    fn vector_search_scored(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> SdkResult<Vec<(CachedMemory, f64)>> {
        let query_bytes = f32_slice_to_bytes(query_embedding);

        let mut stmt = self.conn.prepare(
            "SELECT cm.id, cm.content, cm.memory_type, cm.metadata, cm.embedding,
                    cm.relevance_score, cm.created_at, cm.synced, cm.cloud_id, v.distance
             FROM vec_memories v
             INNER JOIN cached_memories cm ON cm.id = v.id
             WHERE v.embedding MATCH ?1 AND k = ?2
             ORDER BY v.distance",
        )?;

        let rows = stmt.query_map(rusqlite::params![query_bytes, limit as i64], |row| {
            let parsed = parse_memory_row(row);
            let distance: f64 = row.get(9)?;
            Ok((parsed, distance))
        })?;

        let mut results = Vec::new();
        for row in rows {
            match row {
                Ok((Ok(memory), distance)) => results.push((memory, distance)),
                Ok((Err(e), _)) => tracing::warn!("failed to parse cached memory: {e}"),
                Err(e) => tracing::warn!("failed to read row: {e}"),
            }
        }

        Ok(results)
    }

    /// Keyword search over the FTS5 index. Returns each hit alongside its
    /// BM25-derived score (positive — larger = better match).
    ///
    /// `query` is passed straight to FTS5; callers that accept untrusted
    /// input should sanitize/quote tokens to avoid FTS5 query-syntax errors.
    pub fn keyword_search(
        &self,
        query: &str,
        limit: usize,
    ) -> SdkResult<Vec<(CachedMemory, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT cm.id, cm.content, cm.memory_type, cm.metadata, cm.embedding,
                    cm.relevance_score, cm.created_at, cm.synced, cm.cloud_id, fts_memories.rank
             FROM fts_memories
             INNER JOIN cached_memories cm ON cm.id = fts_memories.id
             WHERE fts_memories MATCH ?1
             ORDER BY fts_memories.rank
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(rusqlite::params![query, limit as i64], |row| {
            let parsed = parse_memory_row(row);
            let rank: f64 = row.get(9)?;
            Ok((parsed, rank))
        })?;

        let mut results = Vec::new();
        for row in rows {
            match row {
                // FTS5 `rank` is negative (smaller = better). Flip sign so the
                // returned score is positive and orderable as "higher is better".
                Ok((Ok(memory), rank)) => results.push((memory, -rank)),
                Ok((Err(e), _)) => tracing::warn!("failed to parse fts row: {e}"),
                Err(e) => tracing::warn!("failed to read fts row: {e}"),
            }
        }

        Ok(results)
    }

    /// Hybrid retrieval: BM25 keyword + vec0 vector, fused via Reciprocal
    /// Rank Fusion (K=60). Mirrors the cloud `seren-memory` recall shape so
    /// degraded-mode recall feels like memory, not chronology.
    ///
    /// When `query_embedding` is `None`, falls back to keyword-only.
    pub fn hybrid_search(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
        limit: usize,
    ) -> SdkResult<Vec<RankedCachedMemory>> {
        // Pull a wider pool from each source so the fusion has room to reorder.
        let pool = limit.saturating_mul(2).max(limit);

        let keyword_hits = self.keyword_search(query, pool)?;

        let Some(embedding) = query_embedding else {
            // Keyword-only fallback. Surface BM25 as the ranking score so
            // callers still get a comparable `rrf_score` field populated.
            let mut out: Vec<RankedCachedMemory> = keyword_hits
                .into_iter()
                .map(|(memory, bm25)| RankedCachedMemory {
                    memory,
                    rrf_score: bm25,
                    vector_score: None,
                    bm25_score: Some(bm25),
                })
                .collect();
            out.truncate(limit);
            return Ok(out);
        };

        let vector_hits = self.vector_search_scored(embedding, pool)?;

        // Aggregate by id: sum RRF contributions, remember each side's raw score.
        struct Acc {
            rrf: f64,
            vector_score: Option<f64>,
            bm25_score: Option<f64>,
            memory: CachedMemory,
        }
        let mut acc: HashMap<Uuid, Acc> = HashMap::new();

        for (rank, (memory, bm25)) in keyword_hits.into_iter().enumerate() {
            let contribution = 1.0 / (RRF_K + (rank + 1) as f64);
            acc.entry(memory.id)
                .and_modify(|e| {
                    e.rrf += contribution;
                    e.bm25_score = Some(bm25);
                })
                .or_insert(Acc {
                    rrf: contribution,
                    vector_score: None,
                    bm25_score: Some(bm25),
                    memory,
                });
        }

        for (rank, (memory, distance)) in vector_hits.into_iter().enumerate() {
            let contribution = 1.0 / (RRF_K + (rank + 1) as f64);
            acc.entry(memory.id)
                .and_modify(|e| {
                    e.rrf += contribution;
                    e.vector_score = Some(distance);
                })
                .or_insert(Acc {
                    rrf: contribution,
                    vector_score: Some(distance),
                    bm25_score: None,
                    memory,
                });
        }

        let mut ranked: Vec<RankedCachedMemory> = acc
            .into_values()
            .map(|a| RankedCachedMemory {
                memory: a.memory,
                rrf_score: a.rrf,
                vector_score: a.vector_score,
                bm25_score: a.bm25_score,
            })
            .collect();

        ranked.sort_by(|a, b| {
            b.rrf_score
                .partial_cmp(&a.rrf_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.truncate(limit);
        Ok(ranked)
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

    /// List recent memories ordered by creation time (newest first).
    pub fn list_recent(&self, limit: usize) -> SdkResult<Vec<CachedMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, content, memory_type, metadata, embedding,
                    relevance_score, created_at, synced, cloud_id
             FROM cached_memories
             ORDER BY created_at DESC
             LIMIT ?1",
        )?;

        let rows = stmt.query_map(rusqlite::params![limit as i64], |row| {
            Ok(parse_memory_row(row))
        })?;

        let mut results = Vec::new();
        for row in rows {
            match row {
                Ok(Ok(memory)) => results.push(memory),
                Ok(Err(e)) => tracing::warn!("failed to parse memory: {e}"),
                Err(e) => tracing::warn!("failed to read row: {e}"),
            }
        }

        Ok(results)
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

    /// Backfill must populate FTS from pre-existing `cached_memories` rows on
    /// the first open after upgrade, then never run again. We simulate the
    /// upgrade by inserting raw rows into `cached_memories` (bypassing the
    /// FTS write) and clearing the `fts_backfilled` flag, then reopening.
    #[test]
    fn backfill_runs_exactly_once_across_reopens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("backfill.db");

        // First open: insert a pre-FTS row directly, then unset the flag.
        let cache = LocalCache::open(&path).unwrap();
        let id = Uuid::new_v4();
        cache
            .conn
            .execute(
                "INSERT INTO cached_memories (id, content, memory_type, metadata, embedding, relevance_score, created_at, synced, cloud_id)
                 VALUES (?1, ?2, ?3, '{}', ?4, 1.0, ?5, 0, NULL)",
                rusqlite::params![
                    id.to_string(),
                    "pnpm install fails offline",
                    "semantic",
                    f32_slice_to_bytes(&vec![0.0; 1536]),
                    Utc::now().to_rfc3339(),
                ],
            )
            .unwrap();
        cache
            .conn
            .execute("DELETE FROM sync_state WHERE key = 'fts_backfilled'", [])
            .unwrap();
        drop(cache);

        // Second open: backfill should populate FTS for the orphaned row.
        let cache = LocalCache::open(&path).unwrap();
        let hits = cache.keyword_search("pnpm", 10).unwrap();
        assert_eq!(hits.len(), 1, "backfill should index existing rows");
        assert_eq!(hits[0].0.id, id);

        // Third open: the flag is set, so backfill must NOT re-insert
        // (which would either duplicate or — given fts5 has no PK — silently
        // pollute the index).
        drop(cache);
        let cache = LocalCache::open(&path).unwrap();
        let hits = cache.keyword_search("pnpm", 10).unwrap();
        assert_eq!(hits.len(), 1, "backfill must run exactly once");
    }

    /// BM25 ranks documents with stronger keyword presence higher. The
    /// ranking score returned by `keyword_search` is positive — bigger is
    /// better — so we just check the order matches BM25 expectations.
    #[test]
    fn keyword_search_returns_bm25_ranked_matches() {
        let cache = LocalCache::open_in_memory().unwrap();

        let strong = test_memory("pnpm pnpm pnpm install loop", false);
        let weak = test_memory("pnpm install fails offline once", false);
        let unrelated = test_memory("git push origin main", false);

        cache.insert_memory(&strong).unwrap();
        cache.insert_memory(&weak).unwrap();
        cache.insert_memory(&unrelated).unwrap();

        let hits = cache.keyword_search("pnpm", 10).unwrap();

        assert_eq!(hits.len(), 2, "only matching docs should be returned");
        assert_eq!(hits[0].0.id, strong.id, "stronger BM25 match comes first");
        assert_eq!(hits[1].0.id, weak.id);
        assert!(
            hits[0].1 >= hits[1].1,
            "scores must be sorted descending (got {} then {})",
            hits[0].1,
            hits[1].1,
        );
    }

    /// RRF fusion: a document that ranks #1 in one source and #2 in the
    /// other should land above a document that ranks #2 then #1 only when
    /// its summed RRF contribution is larger — and crucially, both
    /// `vector_score` and `bm25_score` must be populated for documents
    /// matched by both rankers.
    #[test]
    fn hybrid_search_rrf_merges_two_item_case() {
        let cache = LocalCache::open_in_memory().unwrap();

        // `a`: keyword match on "pnpm" AND closest to query embedding.
        let mut a = test_memory("pnpm install fails", false);
        a.embedding = vec![1.0; 1536];
        cache.insert_memory(&a).unwrap();

        // `b`: keyword match on "pnpm" but farther from query embedding.
        let mut b = test_memory("pnpm cache corruption", false);
        b.embedding = vec![0.5; 1536];
        cache.insert_memory(&b).unwrap();

        // `c`: no keyword match, far from query embedding.
        let mut c = test_memory("git push fails", false);
        c.embedding = vec![0.0; 1536];
        cache.insert_memory(&c).unwrap();

        let query_embedding = vec![1.0; 1536];
        let ranked = cache
            .hybrid_search("pnpm", Some(&query_embedding), 10)
            .unwrap();

        // All three appear (a/b via keyword+vector, c via vector-only).
        assert_eq!(ranked.len(), 3);

        // `a` is top-1 in both rankers → strictly highest RRF.
        assert_eq!(ranked[0].memory.id, a.id);
        assert!(ranked[0].rrf_score > ranked[1].rrf_score);

        // The top hit was matched by both sources, so both scores populate.
        assert!(ranked[0].vector_score.is_some());
        assert!(ranked[0].bm25_score.is_some());

        // `c` was vector-only — bm25_score must remain None.
        let c_ranked = ranked.iter().find(|r| r.memory.id == c.id).unwrap();
        assert!(c_ranked.bm25_score.is_none());
        assert!(c_ranked.vector_score.is_some());

        // Keyword-only fallback path must not panic and must drop the
        // vector-only document `c` from results.
        let keyword_only = cache.hybrid_search("pnpm", None, 10).unwrap();
        assert_eq!(keyword_only.len(), 2);
        assert!(keyword_only.iter().all(|r| r.vector_score.is_none()));
        assert!(keyword_only.iter().all(|r| r.bm25_score.is_some()));
    }
}
