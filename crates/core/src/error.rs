use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Quantization error: {0}")]
    Quantization(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Graph error: {0}")]
    Graph(String),

    #[error("Cognitive error: {0}")]
    Cognitive(String),

    #[error("Timeout: operation exceeded {0}ms")]
    Timeout(u64),

    #[error("Memory capacity exceeded: {0}")]
    CapacityExceeded(String),
}

pub type Result<T> = std::result::Result<T, MemoryError>;
