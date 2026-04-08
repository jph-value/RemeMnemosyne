/// Structured JSON logging
///
/// This module provides structured JSON logging initialization for production deployments.
/// Enabled with the `structured-logging` feature flag.
use serde::{Deserialize, Serialize};

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Enable JSON formatting
    pub json_format: bool,
    /// Enable file output
    pub file_output: bool,
    /// Log file path (if file_output is true)
    pub log_file: Option<String>,
    /// Enable console output
    pub console_output: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            json_format: true,
            file_output: false,
            log_file: None,
            console_output: true,
        }
    }
}

/// Initialize structured JSON logging
#[cfg(feature = "structured-logging")]
pub fn init_logging(
    config: &LoggingConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    // Parse log level
    let filter = EnvFilter::try_new(&config.level)?;

    // Build subscriber
    if config.json_format {
        // JSON formatted logging
        let fmt_layer = fmt::layer()
            .json()
            .with_target(true)
            .with_thread_ids(true)
            .with_line_number(true);

        if config.console_output {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .init();
        }

        tracing::info!(config = ?config, "Structured JSON logging initialized");
    } else {
        // Human-readable formatted logging
        let fmt_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_line_number(true);

        if config.console_output {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .init();
        }
    }

    Ok(())
}

/// Initialize logging with default config
#[cfg(feature = "structured-logging")]
pub fn init_default_logging() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = LoggingConfig::default();
    init_logging(&config)
}

/// Stub implementation when feature is not enabled
#[cfg(not(feature = "structured-logging"))]
pub fn init_logging(
    config: &LoggingConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _ = config;
    // No-op when feature is disabled
    Ok(())
}

#[cfg(not(feature = "structured-logging"))]
pub fn init_default_logging() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logging_config_default() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, "info");
        assert!(config.json_format);
        assert!(config.console_output);
    }

    #[cfg(not(feature = "structured-logging"))]
    #[test]
    fn test_logging_not_enabled() {
        let result = init_default_logging();
        assert!(result.is_ok());
    }
}
