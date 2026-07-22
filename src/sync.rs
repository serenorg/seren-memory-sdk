// ABOUTME: Bidirectional sync between local SQLite cache and cloud service.
// ABOUTME: Pushes unsynced local memories, pulls new cloud memories.

use uuid::Uuid;

use crate::cache::LocalCache;
use crate::client::{MemoryClient, PushMemoryRequest};
use crate::error::SdkResult;
use crate::models::{CachedMemory, PushSummary, SyncResult};

pub struct SyncEngine {
    cache: LocalCache,
    client: MemoryClient,
}

impl SyncEngine {
    pub fn new(cache: LocalCache, client: MemoryClient) -> Self {
        Self { cache, client }
    }

    /// Push local-only memories to cloud, pull new cloud memories to local.
    pub async fn sync(&self, user_id: Uuid, project_id: Option<Uuid>) -> SdkResult<SyncResult> {
        let mut result = SyncResult::default();

        match self.push().await {
            Ok(summary) => {
                result.pushed = summary.pushed;
                result.errors.extend(summary.errors);
            }
            Err(e) => {
                tracing::warn!("push failed (offline?): {e}");
                result.errors.push(format!("push: {e}"));
            }
        }

        match self.pull(user_id, project_id).await {
            Ok(n) => result.pulled = n,
            Err(e) => {
                tracing::warn!("pull failed (offline?): {e}");
                result.errors.push(format!("pull: {e}"));
            }
        }

        Ok(result)
    }

    /// Push pending local memories to cloud.
    pub async fn push(&self) -> SdkResult<PushSummary> {
        let pending = self.cache.get_pending_uploads()?;
        let mut summary = PushSummary::default();

        for upload in &pending {
            let memory = &upload.memory;
            let req = PushMemoryRequest {
                content: memory.content.clone(),
                memory_type: memory.memory_type.clone(),
                metadata: memory.metadata.clone(),
                pin: Some(memory.pinned),
                project_id: upload.project_id,
                org_id: upload.org_id,
            };

            match self.client.push_memory(&req).await {
                Ok(cloud_id) => {
                    self.cache.mark_synced(memory.id, cloud_id)?;
                    summary.pushed += 1;
                }
                Err(e) => {
                    tracing::warn!(memory_id = %memory.id, "failed to push memory: {e}");
                    summary.errors.push(format!("push {}: {e}", memory.id));
                }
            }
        }

        Ok(summary)
    }

    /// Pull new memories from cloud since last sync.
    pub async fn pull(&self, _user_id: Uuid, project_id: Option<Uuid>) -> SdkResult<usize> {
        let since = self.cache.get_last_sync_timestamp()?;

        let cloud_memories = self.client.pull_memories(project_id).await?;

        let mut pulled = 0;
        for cloud_mem in &cloud_memories {
            if since.is_some_and(|timestamp| cloud_mem.created_at <= timestamp) {
                continue;
            }

            // Cloud memories don't include embeddings in the list endpoint.
            // Store with a zero vector; the local vector search won't match,
            // but the content is available for text-based lookups.
            let cached = CachedMemory {
                id: Uuid::new_v4(),
                content: cloud_mem.content.clone(),
                memory_type: cloud_mem.memory_type.clone(),
                metadata: cloud_mem.metadata.clone(),
                embedding: vec![0.0; 1536],
                relevance_score: cloud_mem.relevance_score,
                created_at: cloud_mem.created_at,
                synced: true,
                cloud_id: Some(cloud_mem.id),
                feedback_signal: None,
                pinned: cloud_mem.is_pinned,
            };

            self.cache.insert_memory_scoped(
                &cached,
                cloud_mem.project_id.or(project_id),
                cloud_mem.org_id,
            )?;
            pulled += 1;
        }

        if pulled > 0 {
            self.cache.set_last_sync_timestamp(chrono::Utc::now())?;
        }

        Ok(pulled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_memory(content: &str) -> CachedMemory {
        CachedMemory {
            id: Uuid::new_v4(),
            content: content.to_string(),
            memory_type: "semantic".to_string(),
            metadata: serde_json::json!({}),
            embedding: vec![0.1; 1536],
            relevance_score: 1.0,
            created_at: Utc::now(),
            synced: false,
            cloud_id: None,
            feedback_signal: None,
            pinned: false,
        }
    }

    fn mcp_tool_response(text: &str) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{ "type": "text", "text": text }]
            }
        })
    }

    fn remember_response(cloud_id: Uuid) -> serde_json::Value {
        mcp_tool_response(
            &serde_json::json!({
                "memory_id": cloud_id,
                "action_taken": "add",
                "superseded_memory_id": null,
                "reason": null,
                "edges_created": 0,
                "enrichments_triggered": 1
            })
            .to_string(),
        )
    }

    #[tokio::test]
    async fn push_uploads_unsynced_memories() {
        let server = MockServer::start().await;
        let cloud_id = Uuid::new_v4();
        let project_id = Uuid::new_v4();
        let org_id = Uuid::new_v4();

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(body_partial_json(serde_json::json!({
                "params": {
                    "name": "remember",
                    "arguments": {
                        "content": "push me",
                        "pin": true,
                        "project_id": project_id,
                        "org_id": org_id
                    }
                }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(remember_response(cloud_id)))
            .expect(1)
            .mount(&server)
            .await;

        let cache = LocalCache::open_in_memory().unwrap();
        let mut mem = test_memory("push me");
        mem.pinned = true;
        cache
            .insert_memory_scoped(&mem, Some(project_id), Some(org_id))
            .unwrap();

        assert_eq!(cache.get_pending_uploads().unwrap().len(), 1);

        let client = MemoryClient::new(server.uri(), "key".to_string());
        let engine = SyncEngine::new(cache, client);

        let summary = engine.push().await.unwrap();
        assert_eq!(summary.pushed, 1);
        assert!(summary.errors.is_empty());
        assert_eq!(engine.cache.get_pending_uploads().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn push_continues_past_failing_record_and_reports_error() {
        let server = MockServer::start().await;
        let cloud_id = Uuid::new_v4();

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(body_partial_json(serde_json::json!({
                "params": { "arguments": { "content": "fail me" } }
            })))
            .respond_with(ResponseTemplate::new(500).set_body_string("record rejected"))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(body_partial_json(serde_json::json!({
                "params": { "arguments": { "content": "push me" } }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(remember_response(cloud_id)))
            .expect(1)
            .mount(&server)
            .await;

        let cache = LocalCache::open_in_memory().unwrap();
        let failing = test_memory("fail me");
        let succeeding = test_memory("push me");
        cache.insert_memory(&failing).unwrap();
        cache.insert_memory(&succeeding).unwrap();

        let client = MemoryClient::new(server.uri(), "key".to_string());
        let engine = SyncEngine::new(cache, client);

        let summary = engine.push().await.unwrap();
        assert_eq!(summary.pushed, 1);
        assert_eq!(summary.errors.len(), 1);
        assert!(summary.errors[0].contains(&failing.id.to_string()));
        assert!(!summary.errors[0].contains("fail me"));

        let pending = engine.cache.get_pending_uploads().unwrap();
        assert!(pending.iter().any(|upload| upload.memory.id == failing.id));
        assert!(!pending
            .iter()
            .any(|upload| upload.memory.id == succeeding.id));
    }

    #[tokio::test]
    async fn pull_downloads_cloud_memories() {
        let server = MockServer::start().await;
        let cloud_mem_id = Uuid::new_v4();
        let project_id = Uuid::new_v4();
        let org_id = Uuid::new_v4();

        let list_response = serde_json::json!({
            "memories": [{
                "id": cloud_mem_id.to_string(),
                "content": "cloud memory",
                "memory_type": "semantic",
                "metadata": {},
                "project_id": project_id.to_string(),
                "org_id": org_id.to_string(),
                "relevance_score": 0.9,
                "is_pinned": false,
                "created_at": "2026-02-06T00:00:00Z",
                "updated_at": "2026-02-06T00:00:00Z"
            }]
        });
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(body_partial_json(serde_json::json!({
                "params": { "name": "list_memories" }
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(mcp_tool_response(&list_response.to_string())),
            )
            .expect(2)
            .mount(&server)
            .await;

        let cache = LocalCache::open_in_memory().unwrap();
        let client = MemoryClient::new(server.uri(), "key".to_string());
        let engine = SyncEngine::new(cache, client);

        let user_id = Uuid::new_v4();
        let pulled = engine.pull(user_id, Some(project_id)).await.unwrap();
        assert_eq!(pulled, 1);
        assert_eq!(engine.cache.count().unwrap(), 1);
        assert_eq!(
            engine
                .cache
                .list_recent_scoped(Some(project_id), Some(org_id), 10)
                .unwrap()
                .len(),
            1
        );

        engine.pull(user_id, Some(project_id)).await.unwrap();
        assert_eq!(
            engine.cache.count().unwrap(),
            1,
            "pulling the same cloud memory twice must be idempotent"
        );
    }

    #[tokio::test]
    async fn sync_pushes_and_pulls() {
        let server = MockServer::start().await;
        let cloud_id = Uuid::new_v4();

        // Push response
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(body_partial_json(serde_json::json!({
                "params": { "name": "remember" }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(remember_response(cloud_id)))
            .mount(&server)
            .await;

        // Pull response (empty — nothing new from cloud)
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(body_partial_json(serde_json::json!({
                "params": { "name": "list_memories" }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(mcp_tool_response(
                &serde_json::json!({ "memories": [] }).to_string(),
            )))
            .mount(&server)
            .await;

        let cache = LocalCache::open_in_memory().unwrap();
        let mem = test_memory("sync me");
        cache.insert_memory(&mem).unwrap();

        let client = MemoryClient::new(server.uri(), "key".to_string());
        let engine = SyncEngine::new(cache, client);

        let user_id = Uuid::new_v4();
        let result = engine.sync(user_id, None).await.unwrap();
        assert_eq!(result.pushed, 1);
        assert_eq!(result.pulled, 0);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn sync_handles_offline_gracefully() {
        // No mock server started — all requests will fail
        let cache = LocalCache::open_in_memory().unwrap();
        let mem = test_memory("offline memory");
        cache.insert_memory(&mem).unwrap();

        let client = MemoryClient::new("http://localhost:1".to_string(), "key".to_string());
        let engine = SyncEngine::new(cache, client);

        let user_id = Uuid::new_v4();
        let result = engine.sync(user_id, None).await.unwrap();

        // Both push and pull should fail, but sync itself succeeds
        assert_eq!(result.pushed, 0);
        assert_eq!(result.pulled, 0);
        assert_eq!(result.errors.len(), 2);
        assert!(result.errors[0].starts_with(&format!("push {}:", mem.id)));
        assert!(result.errors[1].starts_with("pull:"));

        // Memory should still be in cache, unsynced
        assert_eq!(engine.cache.count().unwrap(), 1);
        assert_eq!(engine.cache.get_pending_uploads().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn pull_updates_last_sync_timestamp() {
        let server = MockServer::start().await;

        let list_response = serde_json::json!({
            "memories": [{
                "id": Uuid::new_v4().to_string(),
                "content": "new cloud memory",
                "memory_type": "semantic",
                "metadata": {},
                "relevance_score": 1.0,
                "is_pinned": false,
                "created_at": "2026-02-06T00:00:00Z",
                "updated_at": "2026-02-06T00:00:00Z"
            }]
        });
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(body_partial_json(serde_json::json!({
                "params": { "name": "list_memories" }
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(mcp_tool_response(&list_response.to_string())),
            )
            .mount(&server)
            .await;

        let cache = LocalCache::open_in_memory().unwrap();
        assert!(cache.get_last_sync_timestamp().unwrap().is_none());

        let client = MemoryClient::new(server.uri(), "key".to_string());
        let engine = SyncEngine::new(cache, client);

        engine.pull(Uuid::new_v4(), None).await.unwrap();

        assert!(engine.cache.get_last_sync_timestamp().unwrap().is_some());
    }
}
