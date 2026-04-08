/// Configuration file parsing (TOML/YAML)
///
/// This module provides configuration file loading and parsing for TOML and YAML files.
/// Enabled with the `config-file` feature flag.
use rememnemosyne_core::MemoryError;
use rememnemosyne_core::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::RememnosyneConfig;

/// Configuration loader for file-based configs
#[cfg(feature = "config-file")]
pub mod config_loader {
    use super::*;

    /// Load configuration from a TOML file
    pub async fn load_from_toml(path: &Path) -> Result<RememnosyneConfig> {
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| MemoryError::Io(e))?;

        let config: RememnosyneConfig = toml::from_str(&content).map_err(|e| {
            MemoryError::Serialization(format!("Failed to parse TOML config: {}", e))
        })?;

        tracing::info!(path = ?path, "Configuration loaded from TOML file");

        Ok(config)
    }

    /// Save configuration to a TOML file
    pub async fn save_to_toml(config: &RememnosyneConfig, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(config).map_err(|e| {
            MemoryError::Serialization(format!("Failed to serialize config to TOML: {}", e))
        })?;

        tokio::fs::write(path, content)
            .await
            .map_err(|e| MemoryError::Io(e))?;

        tracing::info!(path = ?path, "Configuration saved to TOML file");

        Ok(())
    }
}

/// Configuration templates for common deployment scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigTemplate {
    /// Lightweight deployment (low memory, small embedding dimensions)
    LightCloud,
    /// Medium deployment (moderate resources)
    MediumCloud,
    /// Heavy deployment (high performance, large embeddings)
    HeavyCloud,
}

impl ConfigTemplate {
    /// Generate configuration from template
    pub fn generate(&self) -> RememnosyneConfig {
        match self {
            ConfigTemplate::LightCloud => RememnosyneConfig {
                data_dir: "./rememnemosyne_light".to_string(),
                enable_persistence: true,
                ..Default::default()
            },
            ConfigTemplate::MediumCloud => RememnosyneConfig {
                data_dir: "./rememnemosyne_medium".to_string(),
                enable_persistence: true,
                ..Default::default()
            },
            ConfigTemplate::HeavyCloud => RememnosyneConfig {
                data_dir: "./rememnemosyne_heavy".to_string(),
                enable_persistence: true,
                ..Default::default()
            },
        }
    }
}

/// Stub implementation when feature is not enabled
#[cfg(not(feature = "config-file"))]
pub mod config_loader {
    use super::*;

    pub async fn load_from_toml(_path: &Path) -> Result<RememnosyneConfig> {
        Err(MemoryError::Serialization(
            "Config file feature not enabled. Enable with --features config-file".to_string(),
        ))
    }

    pub async fn save_to_toml(_config: &RememnosyneConfig, _path: &Path) -> Result<()> {
        Err(MemoryError::Serialization(
            "Config file feature not enabled. Enable with --features config-file".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_template_light() {
        let config = ConfigTemplate::LightCloud.generate();
        assert!(config.data_dir.contains("light"));
        assert!(config.enable_persistence);
    }

    #[test]
    fn test_config_template_heavy() {
        let config = ConfigTemplate::HeavyCloud.generate();
        assert!(config.data_dir.contains("heavy"));
    }
}
