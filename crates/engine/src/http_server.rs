//! HTTP server with health endpoint and basic API
//!
//! This module provides an optional HTTP server for the RemeMnemosyne engine,
//! enabling remote access via REST API. Enabled with the `http-server` feature flag.
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use rememnemosyne_core::{MemoryArtifact, MemoryError, MemoryTrigger, Result};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;

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
#[derive(Clone)]
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
    pub total_token_estimate: usize,
}

/// Start the HTTP server
#[cfg(feature = "http-server")]
pub async fn start_server(engine: Arc<RememnosyneEngine>, config: HttpServerConfig) -> Result<()> {
    let state = AppState { engine };

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/remember", post(remember_handler))
        .route("/api/v1/recall", post(recall_handler))
        .route("/api/v1/metrics", get(get_metrics))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| MemoryError::Storage(format!("Invalid address: {}", e)))?;

    tracing::info!(address = %addr, "Starting HTTP server");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| MemoryError::Storage(format!("Bind error: {}", e)))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| MemoryError::Storage(format!("Server error: {}", e)))?;

    Ok(())
}

/// Health check handler
async fn health_check() -> impl IntoResponse {
    axum::Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: 0,
    })
}

/// Remember handler
async fn remember_handler(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<RememberRequest>,
) -> impl IntoResponse {
    let trigger = match &req.trigger {
        Some(t) => match t.as_str() {
            "UserInput" => MemoryTrigger::UserInput,
            "SystemOutput" => MemoryTrigger::SystemOutput,
            "Decision" => MemoryTrigger::Decision,
            "Insight" => MemoryTrigger::Insight,
            _ => MemoryTrigger::Custom(t.clone()),
        },
        None => MemoryTrigger::UserInput,
    };

    match state
        .engine
        .remember(&req.content, &req.summary, trigger)
        .await
    {
        Ok(id) => (
            StatusCode::OK,
            axum::Json(RememberResponse { id: id.to_string() }),
        ),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(RememberResponse { id: String::new() }),
        ),
    }
}

/// Recall handler
async fn recall_handler(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<RecallRequest>,
) -> impl IntoResponse {
    match state.engine.recall(&req.query).await {
        Ok(context) => (
            StatusCode::OK,
            axum::Json(RecallResponse {
                summaries: context.summaries.clone(),
                memories: context.memories.clone(),
                total_token_estimate: context.total_tokens_estimate,
            }),
        ),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(RecallResponse {
                summaries: Vec::new(),
                memories: Vec::new(),
                total_token_estimate: 0,
            }),
        ),
    }
}

/// Metrics endpoint
async fn get_metrics(State(state): State<AppState>) -> impl IntoResponse {
    #[cfg(feature = "metrics")]
    {
        let _ = state;
        "Metrics endpoint - integrate with Prometheus metrics module"
    }

    #[cfg(not(feature = "metrics"))]
    {
        let _ = state;
        "Metrics not enabled"
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
