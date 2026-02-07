// ABOUTME: HTTP client for the seren-memory cloud API.
// ABOUTME: Handles push/pull of memories and session bootstrap requests.

use chrono::{DateTime, Utc};
use reqwest::Client;
use uuid::Uuid;

use crate::error::{SdkError, SdkResult};
use crate::models::{BootstrapInput, CloudMemory, CloudSessionContext};

pub struct MemoryClient {
    base_url: String,
    api_key: String,
    http: Client,
}

impl MemoryClient {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            base_url,
            api_key,
            http: Client::new(),
        }
    }

    /// Push a single memory to the cloud. Returns the cloud-assigned UUID.
    pub async fn push_memory(&self, memory: &PushMemoryRequest) -> SdkResult<Uuid> {
        let url = format!("{}/api/memories", self.base_url);

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(memory)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(SdkError::Unauthorized);
        }

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(SdkError::ServerError { status, body });
        }

        let created: PushMemoryResponse = resp.json().await?;
        Ok(created.id)
    }

    /// Pull memories created after `since` from the cloud.
    pub async fn pull_memories(
        &self,
        _user_id: Uuid,
        project_id: Option<Uuid>,
        since: Option<DateTime<Utc>>,
    ) -> SdkResult<Vec<CloudMemory>> {
        let mut url = format!("{}/api/memories?limit=100", self.base_url);

        if let Some(ts) = since {
            url.push_str(&format!("&created_after={}", ts.to_rfc3339()));
        }
        if let Some(pid) = project_id {
            url.push_str(&format!("&project_id={pid}"));
        }

        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(SdkError::Unauthorized);
        }

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(SdkError::ServerError { status, body });
        }

        let body: PullMemoriesResponse = resp.json().await?;
        Ok(body.memories)
    }

    /// Call the cloud session_bootstrap MCP tool via JSON-RPC.
    pub async fn session_bootstrap(
        &self,
        input: &BootstrapInput,
    ) -> SdkResult<CloudSessionContext> {
        let url = format!("{}/mcp", self.base_url);

        let rpc_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "session_bootstrap",
                "arguments": input,
            }
        });

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&rpc_request)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(SdkError::Unauthorized);
        }

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(SdkError::ServerError { status, body });
        }

        let rpc_response: serde_json::Value = resp.json().await?;

        // Extract the text content from the MCP tools/call response.
        // Format: { "result": { "content": [{ "type": "text", "text": "..." }] } }
        let text = rpc_response["result"]["content"][0]["text"]
            .as_str()
            .ok_or_else(|| {
                SdkError::Other("unexpected MCP response format".to_string())
            })?;

        let context: CloudSessionContext = serde_json::from_str(text)?;
        Ok(context)
    }
}

/// Request body for pushing a memory to the cloud.
#[derive(Debug, serde::Serialize)]
pub struct PushMemoryRequest {
    pub content: String,
    pub memory_type: String,
    pub metadata: serde_json::Value,
}

/// Response from pushing a memory.
#[derive(Debug, serde::Deserialize)]
struct PushMemoryResponse {
    id: Uuid,
}

/// Response from pulling memories.
#[derive(Debug, serde::Deserialize)]
struct PullMemoriesResponse {
    memories: Vec<CloudMemory>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn push_memory_sends_bearer_auth() {
        let server = MockServer::start().await;

        let cloud_id = Uuid::new_v4();
        Mock::given(method("POST"))
            .and(path("/api/memories"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "id": cloud_id.to_string() })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "test-key".to_string());
        let req = PushMemoryRequest {
            content: "test memory".to_string(),
            memory_type: "semantic".to_string(),
            metadata: serde_json::json!({}),
        };

        let result = client.push_memory(&req).await.unwrap();
        assert_eq!(result, cloud_id);
    }

    #[tokio::test]
    async fn push_memory_returns_unauthorized() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/memories"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "bad-key".to_string());
        let req = PushMemoryRequest {
            content: "test".to_string(),
            memory_type: "semantic".to_string(),
            metadata: serde_json::json!({}),
        };

        let err = client.push_memory(&req).await.unwrap_err();
        assert!(matches!(err, SdkError::Unauthorized));
    }

    #[tokio::test]
    async fn push_memory_returns_server_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/memories"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "key".to_string());
        let req = PushMemoryRequest {
            content: "test".to_string(),
            memory_type: "semantic".to_string(),
            metadata: serde_json::json!({}),
        };

        let err = client.push_memory(&req).await.unwrap_err();
        match err {
            SdkError::ServerError { status, body } => {
                assert_eq!(status, 500);
                assert_eq!(body, "internal error");
            }
            other => panic!("expected ServerError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pull_memories_returns_list() {
        let server = MockServer::start().await;

        let mem_id = Uuid::new_v4();
        Mock::given(method("GET"))
            .and(path("/api/memories"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "memories": [{
                    "id": mem_id.to_string(),
                    "content": "pulled memory",
                    "memory_type": "semantic",
                    "metadata": {},
                    "relevance_score": 1.0,
                    "is_pinned": false,
                    "created_at": "2026-02-06T00:00:00Z",
                    "updated_at": "2026-02-06T00:00:00Z"
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "test-key".to_string());
        let user_id = Uuid::new_v4();
        let memories = client.pull_memories(user_id, None, None).await.unwrap();

        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].content, "pulled memory");
        assert_eq!(memories[0].id, mem_id);
    }

    #[tokio::test]
    async fn pull_memories_returns_unauthorized() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/memories"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "bad-key".to_string());
        let err = client
            .pull_memories(Uuid::new_v4(), None, None)
            .await
            .unwrap_err();
        assert!(matches!(err, SdkError::Unauthorized));
    }
}
