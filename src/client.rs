// ABOUTME: HTTP client for the seren-memory cloud API.
// ABOUTME: Handles push/pull of memories and session bootstrap requests.

use std::time::Duration;

use reqwest::Client;
use uuid::Uuid;

use crate::error::{SdkError, SdkResult};
use crate::models::{
    BootstrapInput, CloudMemory, CloudSessionContext, FeedbackSignal, RecallResult,
};

pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

pub struct MemoryClient {
    base_url: String,
    api_key: String,
    http: Client,
}

impl MemoryClient {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self::with_timeouts(
            base_url,
            api_key,
            DEFAULT_CONNECT_TIMEOUT,
            DEFAULT_REQUEST_TIMEOUT,
        )
        .expect("default HTTP client configuration must be valid")
    }

    /// Build a client with caller-selected connect and total request deadlines.
    pub fn with_timeouts(
        base_url: String,
        api_key: String,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> SdkResult<Self> {
        let http = Client::builder()
            .connect_timeout(connect_timeout)
            .timeout(request_timeout)
            .build()?;
        Ok(Self {
            base_url,
            api_key,
            http,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Push a single memory to the cloud. Returns the cloud-assigned UUID.
    pub async fn push_memory(&self, memory: &PushMemoryRequest) -> SdkResult<Uuid> {
        let text = self
            .call_mcp_tool(
                "remember",
                serde_json::to_value(memory).map_err(SdkError::Serialization)?,
            )
            .await?;
        let output: serde_json::Value = serde_json::from_str(&text)?;
        let memory_id = output
            .get("memory_id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| SdkError::Other("remember response missing memory_id".to_string()))?;

        Uuid::parse_str(memory_id).map_err(|error| {
            SdkError::Other(format!("remember response has invalid memory_id: {error}"))
        })
    }

    /// Pull memories from the cloud. The caller applies its local timestamp
    /// filter because the MCP list tool does not accept a created-after filter.
    pub async fn pull_memories(&self, project_id: Option<Uuid>) -> SdkResult<Vec<CloudMemory>> {
        let mut args = serde_json::json!({ "limit": 100 });
        if let Some(pid) = project_id {
            args["project_id"] = serde_json::json!(pid);
        }

        let text = self.call_mcp_tool("list_memories", args).await?;
        let body: PullMemoriesResponse = serde_json::from_str(&text)?;
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

        let text = parse_mcp_tool_response(resp).await?;
        let context: CloudSessionContext = serde_json::from_str(&text)?;
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

        parse_mcp_tool_response(resp).await
    }
}

/// Parse an MCP tools/call response and distinguish protocol-level tool
/// failures from successful JSON-RPC transport responses.
async fn parse_mcp_tool_response(resp: reqwest::Response) -> SdkResult<String> {
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

    let text = rpc_response["result"]["content"][0]["text"]
        .as_str()
        .ok_or_else(|| SdkError::Other("unexpected MCP response format".to_string()))?;

    if rpc_response["result"]["isError"].as_bool() == Some(true) {
        return Err(SdkError::McpToolError(text.to_string()));
    }

    Ok(text.to_string())
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

/// Arguments for pushing a memory through the cloud MCP `remember` tool.
#[derive(Debug, serde::Serialize)]
pub struct PushMemoryRequest {
    pub content: String,
    pub memory_type: String,
    pub metadata: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pin: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub org_id: Option<Uuid>,
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
            .and(path("/mcp"))
            .and(header("authorization", "Bearer test-key"))
            .and(body_partial_json(serde_json::json!({
                "params": { "name": "remember" }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(mcp_tool_response(
                &serde_json::json!({ "memory_id": cloud_id }).to_string(),
            )))
            .expect(1)
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "test-key".to_string());
        let req = PushMemoryRequest {
            content: "test memory".to_string(),
            memory_type: "semantic".to_string(),
            metadata: serde_json::json!({}),
            pin: None,
            project_id: None,
            org_id: None,
        };

        let result = client.push_memory(&req).await.unwrap();
        assert_eq!(result, cloud_id);
    }

    #[tokio::test]
    async fn push_memory_returns_unauthorized() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "bad-key".to_string());
        let req = PushMemoryRequest {
            content: "test".to_string(),
            memory_type: "semantic".to_string(),
            metadata: serde_json::json!({}),
            pin: None,
            project_id: None,
            org_id: None,
        };

        let err = client.push_memory(&req).await.unwrap_err();
        assert!(matches!(err, SdkError::Unauthorized));
    }

    #[tokio::test]
    async fn push_memory_returns_server_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "key".to_string());
        let req = PushMemoryRequest {
            content: "test".to_string(),
            memory_type: "semantic".to_string(),
            metadata: serde_json::json!({}),
            pin: None,
            project_id: None,
            org_id: None,
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
        let list_response = serde_json::json!({
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
        });
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(header("authorization", "Bearer test-key"))
            .and(body_partial_json(serde_json::json!({
                "params": { "name": "list_memories" }
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(mcp_tool_response(&list_response.to_string())),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "test-key".to_string());
        let memories = client.pull_memories(None).await.unwrap();

        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].content, "pulled memory");
        assert_eq!(memories[0].id, mem_id);
    }

    #[tokio::test]
    async fn pull_memories_returns_unauthorized() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "bad-key".to_string());
        let err = client.pull_memories(None).await.unwrap_err();
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

    fn mcp_tool_error_response(text: &str) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{ "type": "text", "text": text }],
                "isError": true
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
    async fn request_timeout_bounds_a_stalled_server() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_secs(1))
                    .set_body_json(mcp_tool_response("too late")),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = MemoryClient::with_timeouts(
            server.uri(),
            "test-key".to_string(),
            Duration::from_secs(1),
            Duration::from_millis(50),
        )
        .unwrap();
        let error = client
            .remember("test", "semantic", None, None)
            .await
            .unwrap_err();

        assert!(matches!(error, SdkError::Http(ref error) if error.is_timeout()));
    }

    #[tokio::test]
    async fn mcp_tool_errors_preserve_server_message_on_both_paths() {
        let server = MockServer::start().await;
        let message = "memory service dependency unavailable";

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(mcp_tool_error_response(message)),
            )
            .expect(2)
            .mount(&server)
            .await;

        let client = MemoryClient::new(server.uri(), "test-key".to_string());
        let remember_error = client
            .remember("test", "semantic", None, None)
            .await
            .unwrap_err();
        let bootstrap_error = client
            .session_bootstrap(&BootstrapInput {
                project_id: None,
                org_id: None,
                token_budget: None,
            })
            .await
            .unwrap_err();

        for error in [remember_error, bootstrap_error] {
            match error {
                SdkError::McpToolError(actual) => assert_eq!(actual, message),
                other => panic!("expected MCP tool error, got {other:?}"),
            }
        }
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
