// ABOUTME: Session bootstrap orchestrator for assembling project context.
// ABOUTME: Combines local cache and cloud data into a structured LLM prompt.

use std::collections::HashMap;

use uuid::Uuid;

use crate::cache::LocalCache;
use crate::client::MemoryClient;
use crate::error::SdkResult;
use crate::models::{BootstrapInput, ContextSource, MemoryRef, SessionContext};

pub struct BootstrapOrchestrator {
    cache: LocalCache,
    client: MemoryClient,
}

impl BootstrapOrchestrator {
    pub fn new(cache: LocalCache, client: MemoryClient) -> Self {
        Self { cache, client }
    }

    /// Assemble session context. Tries cloud first, falls back to local cache.
    pub async fn bootstrap(
        &self,
        project_id: Option<Uuid>,
        org_id: Option<Uuid>,
        token_budget: Option<usize>,
    ) -> SdkResult<SessionContext> {
        let input = BootstrapInput {
            project_id,
            org_id,
            token_budget,
        };

        // Try cloud first for fresh data.
        match self.client.session_bootstrap(&input).await {
            Ok(cloud_ctx) => {
                let prompt = format_prompt(&cloud_ctx.memories_by_type);
                Ok(SessionContext {
                    memories_by_type: cloud_ctx.memories_by_type,
                    total_memories: cloud_ctx.total_memories,
                    assembled_prompt: prompt,
                    source: ContextSource::Cloud,
                })
            }
            Err(e) => {
                tracing::warn!("cloud bootstrap failed, using local cache: {e}");
                self.bootstrap_from_cache(token_budget.unwrap_or(4000))
            }
        }
    }

    /// Build context from local cache only.
    fn bootstrap_from_cache(&self, token_budget: usize) -> SdkResult<SessionContext> {
        let memories = self.cache.list_recent(50)?;

        let mut memories_by_type: HashMap<String, Vec<MemoryRef>> = HashMap::new();
        let mut total_tokens = 0;

        for mem in &memories {
            let estimated_tokens = mem.content.len() / 4;
            if total_tokens + estimated_tokens > token_budget {
                break;
            }
            total_tokens += estimated_tokens;

            // Surface the cloud-side id when we have one (so frontends can
            // mutate the original memory) and fall back to the local id
            // otherwise — preserves the "carry IDs end-to-end" invariant.
            let id = mem.cloud_id.unwrap_or(mem.id);
            memories_by_type
                .entry(mem.memory_type.clone())
                .or_default()
                .push(MemoryRef {
                    id,
                    content: mem.content.clone(),
                });
        }

        let total_memories = memories_by_type.values().map(|v| v.len()).sum();
        let prompt = format_prompt(&memories_by_type);

        Ok(SessionContext {
            memories_by_type,
            total_memories,
            assembled_prompt: prompt,
            source: ContextSource::LocalCache,
        })
    }
}

/// Format memories into a structured markdown prompt section.
fn format_prompt(memories_by_type: &HashMap<String, Vec<MemoryRef>>) -> String {
    if memories_by_type.is_empty() {
        return String::new();
    }

    let mut sections = Vec::new();
    sections.push("## Project Context (auto-loaded by Seren Memory)\n".to_string());

    // Sort types for deterministic output.
    let mut types: Vec<&String> = memories_by_type.keys().collect();
    types.sort();

    for memory_type in types {
        let items = &memories_by_type[memory_type];
        if items.is_empty() {
            continue;
        }

        let heading = memory_type
            .replace('_', " ")
            .split_whitespace()
            .map(|w| {
                let mut c = w.chars();
                match c.next() {
                    None => String::new(),
                    Some(first) => {
                        first.to_uppercase().to_string() + c.as_str()
                    }
                }
            })
            .collect::<Vec<_>>()
            .join(" ");

        sections.push(format!("### {heading}"));
        for item in items {
            sections.push(format!("- {}", item.content));
        }
        sections.push(String::new());
    }

    sections.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::CachedMemory;
    use chrono::Utc;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn insert_test_memory(cache: &LocalCache, content: &str, memory_type: &str) {
        let mem = CachedMemory {
            id: Uuid::new_v4(),
            content: content.to_string(),
            memory_type: memory_type.to_string(),
            metadata: serde_json::json!({}),
            embedding: vec![0.1; 1536],
            relevance_score: 1.0,
            created_at: Utc::now(),
            synced: true,
            cloud_id: Some(Uuid::new_v4()),
            feedback_signal: None,
            pinned: false,
        };
        cache.insert_memory(&mem).unwrap();
    }

    #[tokio::test]
    async fn bootstrap_uses_cloud_when_available() {
        let server = MockServer::start().await;

        // MCP session_bootstrap response
        let cloud_response = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{
                    "type": "text",
                    "text": serde_json::json!({
                        "memories_by_type": {
                            "semantic": ["Project uses Axum 0.8"],
                            "convention": ["snake_case everywhere"]
                        },
                        "total_memories": 2
                    }).to_string()
                }]
            }
        });

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(cloud_response))
            .expect(1)
            .mount(&server)
            .await;

        let cache = LocalCache::open_in_memory().unwrap();
        let client = MemoryClient::new(server.uri(), "key".to_string());
        let orchestrator = BootstrapOrchestrator::new(cache, client);

        let ctx = orchestrator.bootstrap(None, None, None).await.unwrap();
        assert_eq!(ctx.source, ContextSource::Cloud);
        assert_eq!(ctx.total_memories, 2);
        assert!(ctx.assembled_prompt.contains("Project uses Axum 0.8"));
        assert!(ctx.assembled_prompt.contains("snake_case everywhere"));
    }

    #[tokio::test]
    async fn bootstrap_falls_back_to_local_cache_when_offline() {
        let cache = LocalCache::open_in_memory().unwrap();
        insert_test_memory(&cache, "Uses Rust for backend", "semantic");
        insert_test_memory(&cache, "Always use TDD", "convention");

        // Unreachable server
        let client = MemoryClient::new("http://localhost:1".to_string(), "key".to_string());
        let orchestrator = BootstrapOrchestrator::new(cache, client);

        let ctx = orchestrator.bootstrap(None, None, None).await.unwrap();
        assert_eq!(ctx.source, ContextSource::LocalCache);
        assert_eq!(ctx.total_memories, 2);
        assert!(ctx.assembled_prompt.contains("Uses Rust for backend"));
        assert!(ctx.assembled_prompt.contains("Always use TDD"));
    }

    #[tokio::test]
    async fn bootstrap_returns_empty_context_for_empty_cache() {
        let cache = LocalCache::open_in_memory().unwrap();
        let client = MemoryClient::new("http://localhost:1".to_string(), "key".to_string());
        let orchestrator = BootstrapOrchestrator::new(cache, client);

        let ctx = orchestrator.bootstrap(None, None, None).await.unwrap();
        assert_eq!(ctx.source, ContextSource::LocalCache);
        assert_eq!(ctx.total_memories, 0);
        assert!(ctx.assembled_prompt.is_empty());
    }

    #[tokio::test]
    async fn bootstrap_respects_token_budget() {
        let cache = LocalCache::open_in_memory().unwrap();

        // Each memory is ~50 chars = ~12 tokens.
        // With budget of 20 tokens, should fit ~1-2 memories.
        for i in 0..10 {
            insert_test_memory(
                &cache,
                &format!("Memory number {i} with some padding content here"),
                "semantic",
            );
        }

        let client = MemoryClient::new("http://localhost:1".to_string(), "key".to_string());
        let orchestrator = BootstrapOrchestrator::new(cache, client);

        let ctx = orchestrator.bootstrap(None, None, Some(20)).await.unwrap();
        assert!(ctx.total_memories < 10);
        assert!(ctx.total_memories > 0);
    }

    fn mref(content: &str) -> MemoryRef {
        MemoryRef {
            id: Uuid::new_v4(),
            content: content.to_string(),
        }
    }

    #[test]
    fn format_prompt_groups_by_type() {
        let mut memories = HashMap::new();
        memories.insert(
            "semantic".to_string(),
            vec![mref("Fact one"), mref("Fact two")],
        );
        memories.insert("convention".to_string(), vec![mref("Use snake_case")]);

        let prompt = format_prompt(&memories);
        assert!(prompt.contains("### Convention"));
        assert!(prompt.contains("### Semantic"));
        assert!(prompt.contains("- Fact one"));
        assert!(prompt.contains("- Use snake_case"));
    }

    /// SessionContext must carry `MemoryRef` ids end-to-end. The cloud path
    /// already returns objects with ids; the offline cache path was the
    /// regression risk, so we exercise it directly.
    #[tokio::test]
    async fn cache_bootstrap_propagates_memory_ids() {
        let cache = LocalCache::open_in_memory().unwrap();
        let cloud_id = Uuid::new_v4();
        let mem = CachedMemory {
            id: Uuid::new_v4(),
            content: "Use snake_case".to_string(),
            memory_type: "convention".to_string(),
            metadata: serde_json::json!({}),
            embedding: vec![0.1; 1536],
            relevance_score: 1.0,
            created_at: Utc::now(),
            synced: true,
            cloud_id: Some(cloud_id),
            feedback_signal: None,
            pinned: false,
        };
        cache.insert_memory(&mem).unwrap();

        // Unreachable cloud → forces the local-cache path.
        let client = MemoryClient::new("http://localhost:1".to_string(), "key".to_string());
        let orchestrator = BootstrapOrchestrator::new(cache, client);

        let ctx = orchestrator.bootstrap(None, None, None).await.unwrap();
        assert_eq!(ctx.source, ContextSource::LocalCache);

        let refs = ctx.memories_by_type.get("convention").expect("type missing");
        assert_eq!(refs.len(), 1);
        assert_eq!(
            refs[0].id, cloud_id,
            "cache path must surface cloud_id when present (not the local row id)"
        );
        assert_eq!(refs[0].content, "Use snake_case");
    }

    #[test]
    fn format_prompt_returns_empty_for_no_memories() {
        let memories = HashMap::new();
        let prompt = format_prompt(&memories);
        assert!(prompt.is_empty());
    }
}
