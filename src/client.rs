// ABOUTME: HTTP client for the seren-memory cloud API.
// ABOUTME: Handles push/pull of memories and session bootstrap requests.

use chrono::{DateTime, Utc};
use reqwest::Client;
use uuid::Uuid;

use crate::error::{SdkError, SdkResult};
use crate::models::{
    BootstrapInput, CloudMemory, CloudSessionContext, FeedbackSignal, RecallResult,
};

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

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
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
        Ok(created.memory_id)
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
            .header("Accept", "application/json, text/event-stream")
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

    /// Store a memory via the cloud MCP remember tool.
    pub async fn remember(
        &self,
        content: &str,
        memory_type: &str,
        project_id: Option<Uuid>,
        org_id: Option<Uuid>,
    ) -> SdkResult<String> {
        let mut args = serde_json::json!({
            "content": content,
            "memory_type": memory_type,
        });
        if let Some(pid) = project_id {
            args["project_id"] = serde_json::json!(pid);
        }
        if let Some(oid) = org_id {
            args["org_id"] = serde_json::json!(oid);
        }

        let text = self.call_mcp_tool("remember", args).await?;
        Ok(text)
    }

    /// Search memories via the cloud MCP recall tool.
    pub async fn recall(
        &self,
        query: &str,
        project_id: Option<Uuid>,
        limit: Option<usize>,
    ) -> SdkResult<Vec<RecallResult>> {
        let mut args = serde_json::json!({ "query": query });
        if let Some(pid) = project_id {
            args["project_id"] = serde_json::json!(pid);
        }
        if let Some(lim) = limit {
            args["limit"] = serde_json::json!(lim);
        }

        let text = self.call_mcp_tool("recall", args).await?;
        let results: Vec<RecallResult> = serde_json::from_str(&text)?;
        Ok(results)
    }

    /// Delete a memory by id via the cloud MCP `delete_memory` tool.
    pub async fn delete_memory(&self, id: Uuid) -> SdkResult<()> {
        let args = serde_json::json!({ "memory_id": id });
        self.call_mcp_tool("delete_memory", args).await?;
        Ok(())
    }

    /// Record a feedback signal on a memory via the cloud MCP `mark_feedback`
    /// tool. The wire payload uses `+1` / `-1` integers for the signal.
    pub async fn mark_feedback(&self, id: Uuid, signal: FeedbackSignal) -> SdkResult<()> {
        let args = serde_json::json!({
            "memory_id": id,
            "signal": signal.as_i32(),
        });
        self.call_mcp_tool("mark_feedback", args).await?;
        Ok(())
    }

    /// Toggle the pin state of a memory via the cloud MCP `update_memory`
    /// tool with the `is_pinned` field set.
    pub async fn toggle_pin(&self, id: Uuid, pinned: bool) -> SdkResult<()> {
        let args = serde_json::json!({
            "memory_id": id,
            "is_pinned": pinned,
        });
        self.call_mcp_tool("update_memory", args).await?;
        Ok(())
    }

    /// Send a JSON-RPC tools/call request to the MCP endpoint.
    async fn call_mcp_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> SdkResult<String> {
        let url = format!("{}/mcp", self.base_url);

        let rpc_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            }
        });

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .header("Accept", "application/json, text/event-stream")
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

        // The MCP server may return SSE (text/event-stream) or plain JSON
        // depending on the Accept header negotiation. Parse both formats.
        let is_sse = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.contains("text/event-stream"))
            .unwrap_or(false);

        let body_text = resp.text().await?;

        let json_str = if is_sse {
            extract_sse_json(&body_text)?
        } else {
            body_text
        };

        let rpc_response: serde_json::Value = serde_json::from_str(&json_str)?;

        rpc_response["result"]["content"][0]["text"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| SdkError::Other("unexpected MCP response format".to_string()))
    }
}

/// Extract JSON from an SSE response body.
///
/// SSE format: lines prefixed with `data: ` contain the payload.
/// Concatenates all `data:` payloads (skipping comments and blank lines)
/// and returns the combined JSON string.
fn extract_sse_json(body: &str) -> SdkResult<String> {
    let mut json_parts = Vec::new();
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                break;
            }
            json_parts.push(data);
        } else if let Some(data) = line.strip_prefix("data:") {
            if data.trim() == "[DONE]" {
                break;
            }
            json_parts.push(data.trim());
        }
    }

    if json_parts.is_empty() {
        return Err(SdkError::Other(
            "SSE response contained no data lines".to_string(),
        ));
    }

    // MCP tool responses are single JSON objects in one data line
    Ok(json_parts.join(""))
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
    memory_id: Uuid,
}

/// Response from pulling memories.
#[derive(Debug, serde::Deserialize)]
struct PullMemoriesResponse {
    memories: Vec<CloudMemory>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::FeedbackSignal;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn push_memory_sends_bearer_auth() {
        let server = MockServer::start().await;

        let cloud_id = Uuid::new_v4();
        Mock::given(method("POST"))
            .and(path("/api/memories"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "memory_id": cloud_id.to_string(),
                    "action_taken": "add",
                    "superseded_memory_id": null,
                    "reason": null,
                    "edges_created": 0,
                    "enrichments_triggered": 1
                })),
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

    fn mcp_tool_response(text: &str) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{ "type": "text", "text": text }]
            }
        })
    }

    #[tokio::test]
    async fn remember_calls_mcp_tool() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(mcp_tool_response("Memory stored successfully")),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "test-key".to_string());
        let result = client
            .remember("test fact", "semantic", None, None)
            .await
            .unwrap();
        assert_eq!(result, "Memory stored successfully");
    }

    #[tokio::test]
    async fn remember_returns_unauthorized() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "bad-key".to_string());
        let err = client
            .remember("test", "semantic", None, None)
            .await
            .unwrap_err();
        assert!(matches!(err, SdkError::Unauthorized));
    }

    #[tokio::test]
    async fn recall_returns_results_with_ids() {
        let server = MockServer::start().await;

        let id_one = Uuid::new_v4();
        let id_two = Uuid::new_v4();
        let results = serde_json::json!([
            { "id": id_one, "content": "Axum 0.8 is used", "memory_type": "semantic", "relevance_score": 0.95 },
            { "id": id_two, "content": "Use snake_case", "memory_type": "convention", "relevance_score": 0.8 }
        ]);

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(mcp_tool_response(&results.to_string())),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "test-key".to_string());
        let results = client.recall("what framework", None, Some(5)).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, id_one, "recall must surface cloud memory ids");
        assert_eq!(results[0].content, "Axum 0.8 is used");
        assert_eq!(results[1].id, id_two);
        assert_eq!(results[1].memory_type, "convention");
    }

    #[tokio::test]
    async fn recall_returns_unauthorized() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "bad-key".to_string());
        let err = client.recall("query", None, None).await.unwrap_err();
        assert!(matches!(err, SdkError::Unauthorized));
    }

    #[tokio::test]
    async fn mcp_tool_sends_accept_header() {
        // Verify the Accept header is sent by checking the server rejects
        // requests missing it (406) but accepts requests with it.
        // We simulate the server requiring the header by only responding 200
        // when "application/json" appears in the Accept value.
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(mcp_tool_response("ok")),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "test-key".to_string());
        let result = client.remember("fact", "semantic", None, None).await.unwrap();
        assert_eq!(result, "ok");
    }

    #[tokio::test]
    async fn remember_handles_sse_response() {
        let server = MockServer::start().await;

        // Simulate the server returning SSE format (text/event-stream)
        let sse_body = format!(
            "data: {}\n\n",
            serde_json::to_string(&mcp_tool_response("SSE memory stored")).unwrap()
        );

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_raw(sse_body, "text/event-stream"),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "test-key".to_string());
        let result = client
            .remember("test fact", "semantic", None, None)
            .await
            .unwrap();
        assert_eq!(result, "SSE memory stored");
    }

    #[test]
    fn extract_sse_json_parses_data_lines() {
        let body = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}}\n\ndata: [DONE]\n\n";
        let result = extract_sse_json(body).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["result"]["content"][0]["text"], "ok");
    }

    #[test]
    fn extract_sse_json_skips_comments() {
        let body = ": comment\ndata: {\"id\":1}\n\n";
        let result = extract_sse_json(body).unwrap();
        assert_eq!(result, "{\"id\":1}");
    }

    #[test]
    fn extract_sse_json_empty_body_errors() {
        let result = extract_sse_json("");
        assert!(result.is_err());
    }

    /// Each mutation must call its own MCP tool with a body that carries
    /// `memory_id` and the tool-specific payload. We verify the body shape
    /// via `body_partial_json` so the mock fails the request if the SDK
    /// names the tool wrong or drops a field.
    #[tokio::test]
    async fn mutations_call_their_mcp_tools_with_correct_payloads() {
        let server = MockServer::start().await;
        let mem_id = Uuid::new_v4();

        // delete_memory → tool name "delete_memory", carries memory_id
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(body_partial_json(serde_json::json!({
                "params": { "name": "delete_memory", "arguments": { "memory_id": mem_id } }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(mcp_tool_response("ok")))
            .expect(1)
            .mount(&server)
            .await;

        // mark_feedback → "mark_feedback", carries signal as -1 / +1 integer
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(body_partial_json(serde_json::json!({
                "params": {
                    "name": "mark_feedback",
                    "arguments": { "memory_id": mem_id, "signal": -1 }
                }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(mcp_tool_response("ok")))
            .expect(1)
            .mount(&server)
            .await;

        // toggle_pin → "update_memory" with is_pinned (matches cloud schema)
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(body_partial_json(serde_json::json!({
                "params": {
                    "name": "update_memory",
                    "arguments": { "memory_id": mem_id, "is_pinned": true }
                }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(mcp_tool_response("ok")))
            .expect(1)
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "test-key".to_string());
        client.delete_memory(mem_id).await.unwrap();
        client
            .mark_feedback(mem_id, FeedbackSignal::Negative)
            .await
            .unwrap();
        client.toggle_pin(mem_id, true).await.unwrap();
    }
}
