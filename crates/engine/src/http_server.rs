/// HTTP server with health endpoint and basic API
///
/// This module provides an optional HTTP server for the RemeMnemosyne engine,
/// enabling remote access via REST API. Enabled with the `http-server` feature flag.
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use rememnemosyne_core::{MemoryArtifact, MemoryError, MemoryQuery, MemoryTrigger, Result};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::RememnosyneEngine;

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpServerConfig {
    pub host: String,
    pub port: u16,
    pub enable_cors: bool,
    pub enable_metrics: bool,
}

impl Default for HttpServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            enable_cors: true,
            enable_metrics: true,
        }
    }
}

/// Application state shared with handlers
struct AppState {
    engine: Arc<RememnosyneEngine>,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
}

/// Remember request
#[derive(Debug, Deserialize)]
pub struct RememberRequest {
    pub content: String,
    pub summary: String,
    pub trigger: Option<String>,
}

/// Remember response
#[derive(Debug, Serialize)]
pub struct RememberResponse {
    pub id: String,
}

/// Recall request
#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    pub query: String,
}

/// Recall response
#[derive(Debug, Serialize)]
pub struct RecallResponse {
    pub summaries: Vec<String>,
    pub memories: Vec<MemoryArtifact>,
    pub token_estimate: usize,
}

/// Start the HTTP server
#[cfg(feature = "http-server")]
pub async fn start_server(engine: Arc<RememnosyneEngine>, config: HttpServerConfig) -> Result<()> {
    let state = AppState { engine };

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/remember", post(remember))
        .route("/api/v1/recall", post(recall))
        .route("/api/v1/metrics", get(get_metrics))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| MemoryError::Storage(format!("Invalid address: {}", e)))?;

    tracing::info!(address = %addr, "Starting HTTP server");

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .map_err(|e| MemoryError::Storage(format!("Server error: {}", e)))?;

    Ok(())
}

/// Health check handler
async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: 0, // Would track actual uptime
    })
}

/// Remember handler
async fn remember(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RememberRequest>,
) -> Result<Json<RememberResponse>> {
    let trigger = match &req.trigger {
        Some(t) => match t.as_str() {
            "UserInput" => MemoryTrigger::UserInput,
            "SystemOutput" => MemoryTrigger::SystemOutput,
            "Decision" => MemoryTrigger::Decision,
            "Insight" => MemoryTrigger::Insight,
            _ => MemoryTrigger::Custom,
        },
        None => MemoryTrigger::UserInput,
    };

    let id = state
        .engine
        .remember(&req.content, &req.summary, trigger)
        .await?;

    Ok(Json(RememberResponse { id: id.to_string() }))
}

/// Recall handler
async fn recall(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<RecallResponse>> {
    let context = state.engine.recall(&req.query).await?;

    Ok(Json(RecallResponse {
        summaries: context.summaries.clone(),
        memories: context.memories.clone(),
        token_estimate: context.token_estimate,
    }))
}

/// Metrics endpoint (if metrics feature is enabled)
async fn get_metrics(State(state): State<Arc<AppState>>) -> Result<String> {
    #[cfg(feature = "metrics")]
    {
        // Would integrate with the metrics module
        Ok("Metrics endpoint - integrate with Prometheus metrics module".to_string())
    }

    #[cfg(not(feature = "metrics"))]
    {
        let _ = state;
        Ok("Metrics not enabled".to_string())
    }
}

/// Stub implementation when feature is not enabled
#[cfg(not(feature = "http-server"))]
pub async fn start_server(
    _engine: Arc<RememnosyneEngine>,
    _config: HttpServerConfig,
) -> Result<()> {
    Err(MemoryError::Storage(
        "HTTP server feature not enabled. Enable with --features http-server".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_server_config_default() {
        let config = HttpServerConfig::default();
        assert_eq!(config.port, 3000);
        assert!(config.enable_cors);
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "healthy".to_string(),
            version: "0.1.0".to_string(),
            uptime_seconds: 100,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("healthy"));
    }
}
