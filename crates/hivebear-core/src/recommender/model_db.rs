use crate::types::ModelFormat;
use serde::{Deserialize, Serialize};

/// Metadata about a known model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Unique identifier (e.g., "llama-3.1-8b")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Number of parameters in billions
    pub params_billions: f64,
    /// Supported formats
    pub formats: Vec<ModelFormat>,
    /// Default context length
    pub context_length: u32,
    /// Relative quality score (0.0 - 1.0), based on benchmarks like MMLU
    pub quality_score: f64,
    /// What this model is best for
    pub category: ModelCategory,
    /// HuggingFace model ID for GGUF downloads
    pub huggingface_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelCategory {
    General,
    Code,
    Chat,
    Instruction,
    Math,
    Multilingual,
}

impl std::fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelCategory::General => write!(f, "General"),
            ModelCategory::Code => write!(f, "Code"),
            ModelCategory::Chat => write!(f, "Chat"),
            ModelCategory::Instruction => write!(f, "Instruction"),
            ModelCategory::Math => write!(f, "Math"),
            ModelCategory::Multilingual => write!(f, "Multilingual"),
        }
    }
}

/// Built-in database of popular open-source models.
pub fn builtin_models() -> Vec<ModelEntry> {
    vec![
        // Llama 3.x family
        ModelEntry {
            id: "llama-3.2-1b".into(),
            name: "Llama 3.2 1B".into(),
            params_billions: 1.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 128_000,
            quality_score: 0.45,
            category: ModelCategory::General,
            huggingface_id: Some("bartowski/Llama-3.2-1B-Instruct-GGUF".into()),
        },
        ModelEntry {
            id: "llama-3.2-3b".into(),
            name: "Llama 3.2 3B".into(),
            params_billions: 3.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 128_000,
            quality_score: 0.58,
            category: ModelCategory::General,
            huggingface_id: Some("bartowski/Llama-3.2-3B-Instruct-GGUF".into()),
        },
        ModelEntry {
            id: "llama-3.1-8b".into(),
            name: "Llama 3.1 8B".into(),
            params_billions: 8.0,
            formats: vec![
                ModelFormat::Gguf,
                ModelFormat::SafeTensors,
                ModelFormat::Mlx,
            ],
            context_length: 128_000,
            quality_score: 0.72,
            category: ModelCategory::General,
            huggingface_id: Some("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF".into()),
        },
        ModelEntry {
            id: "llama-3.1-70b".into(),
            name: "Llama 3.1 70B".into(),
            params_billions: 70.0,
            formats: vec![
                ModelFormat::Gguf,
                ModelFormat::SafeTensors,
                ModelFormat::Mlx,
            ],
            context_length: 128_000,
            quality_score: 0.90,
            category: ModelCategory::General,
            huggingface_id: Some("bartowski/Meta-Llama-3.1-70B-Instruct-GGUF".into()),
        },
        // Qwen 2.5
        ModelEntry {
            id: "qwen-2.5-0.5b".into(),
            name: "Qwen 2.5 0.5B".into(),
            params_billions: 0.5,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 32_768,
            quality_score: 0.35,
            category: ModelCategory::General,
            huggingface_id: Some("Qwen/Qwen2.5-0.5B-Instruct-GGUF".into()),
        },
        ModelEntry {
            id: "qwen-2.5-1.5b".into(),
            name: "Qwen 2.5 1.5B".into(),
            params_billions: 1.5,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 32_768,
            quality_score: 0.50,
            category: ModelCategory::General,
            huggingface_id: Some("Qwen/Qwen2.5-1.5B-Instruct-GGUF".into()),
        },
        ModelEntry {
            id: "qwen-2.5-7b".into(),
            name: "Qwen 2.5 7B".into(),
            params_billions: 7.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 131_072,
            quality_score: 0.70,
            category: ModelCategory::General,
            huggingface_id: Some("Qwen/Qwen2.5-7B-Instruct-GGUF".into()),
        },
        ModelEntry {
            id: "qwen-2.5-32b".into(),
            name: "Qwen 2.5 32B".into(),
            params_billions: 32.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 131_072,
            quality_score: 0.85,
            category: ModelCategory::General,
            huggingface_id: Some("Qwen/Qwen2.5-32B-Instruct-GGUF".into()),
        },
        ModelEntry {
            id: "qwen-2.5-72b".into(),
            name: "Qwen 2.5 72B".into(),
            params_billions: 72.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 131_072,
            quality_score: 0.91,
            category: ModelCategory::General,
            huggingface_id: Some("Qwen/Qwen2.5-72B-Instruct-GGUF".into()),
        },
        // Qwen 2.5 Coder
        ModelEntry {
            id: "qwen-2.5-coder-7b".into(),
            name: "Qwen 2.5 Coder 7B".into(),
            params_billions: 7.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 131_072,
            quality_score: 0.73,
            category: ModelCategory::Code,
            huggingface_id: Some("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF".into()),
        },
        ModelEntry {
            id: "qwen-2.5-coder-32b".into(),
            name: "Qwen 2.5 Coder 32B".into(),
            params_billions: 32.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 131_072,
            quality_score: 0.88,
            category: ModelCategory::Code,
            huggingface_id: Some("Qwen/Qwen2.5-Coder-32B-Instruct-GGUF".into()),
        },
        // Mistral / Mixtral
        ModelEntry {
            id: "mistral-7b-v0.3".into(),
            name: "Mistral 7B v0.3".into(),
            params_billions: 7.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 32_768,
            quality_score: 0.68,
            category: ModelCategory::General,
            huggingface_id: Some("bartowski/Mistral-7B-Instruct-v0.3-GGUF".into()),
        },
        ModelEntry {
            id: "mixtral-8x7b".into(),
            name: "Mixtral 8x7B".into(),
            params_billions: 46.7, // Total params, ~13B active
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 32_768,
            quality_score: 0.82,
            category: ModelCategory::General,
            huggingface_id: Some("bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF".into()),
        },
        // Phi-3
        ModelEntry {
            id: "phi-3-mini-3.8b".into(),
            name: "Phi-3 Mini 3.8B".into(),
            params_billions: 3.8,
            formats: vec![
                ModelFormat::Gguf,
                ModelFormat::Onnx,
                ModelFormat::SafeTensors,
            ],
            context_length: 128_000,
            quality_score: 0.62,
            category: ModelCategory::Instruction,
            huggingface_id: Some("bartowski/Phi-3-mini-128k-instruct-GGUF".into()),
        },
        ModelEntry {
            id: "phi-3-medium-14b".into(),
            name: "Phi-3 Medium 14B".into(),
            params_billions: 14.0,
            formats: vec![
                ModelFormat::Gguf,
                ModelFormat::Onnx,
                ModelFormat::SafeTensors,
            ],
            context_length: 128_000,
            quality_score: 0.78,
            category: ModelCategory::Instruction,
            huggingface_id: Some("bartowski/Phi-3-medium-128k-instruct-GGUF".into()),
        },
        // Gemma 2
        ModelEntry {
            id: "gemma-2-2b".into(),
            name: "Gemma 2 2B".into(),
            params_billions: 2.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 8192,
            quality_score: 0.52,
            category: ModelCategory::General,
            huggingface_id: Some("bartowski/gemma-2-2b-it-GGUF".into()),
        },
        ModelEntry {
            id: "gemma-2-9b".into(),
            name: "Gemma 2 9B".into(),
            params_billions: 9.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 8192,
            quality_score: 0.74,
            category: ModelCategory::General,
            huggingface_id: Some("bartowski/gemma-2-9b-it-GGUF".into()),
        },
        ModelEntry {
            id: "gemma-2-27b".into(),
            name: "Gemma 2 27B".into(),
            params_billions: 27.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 8192,
            quality_score: 0.83,
            category: ModelCategory::General,
            huggingface_id: Some("bartowski/gemma-2-27b-it-GGUF".into()),
        },
        // DeepSeek Coder
        ModelEntry {
            id: "deepseek-coder-v2-lite".into(),
            name: "DeepSeek Coder V2 Lite 16B".into(),
            params_billions: 16.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 128_000,
            quality_score: 0.76,
            category: ModelCategory::Code,
            huggingface_id: Some("bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF".into()),
        },
        // StarCoder2
        ModelEntry {
            id: "starcoder2-3b".into(),
            name: "StarCoder2 3B".into(),
            params_billions: 3.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 16_384,
            quality_score: 0.55,
            category: ModelCategory::Code,
            huggingface_id: Some("bartowski/starcoder2-3b-GGUF".into()),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_models_not_empty() {
        let models = builtin_models();
        assert!(!models.is_empty());
        assert!(models.len() >= 15);
    }

    #[test]
    fn test_all_models_have_gguf() {
        let models = builtin_models();
        for model in &models {
            assert!(
                model.formats.contains(&ModelFormat::Gguf),
                "Model {} missing GGUF format",
                model.id
            );
        }
    }

    #[test]
    fn test_quality_scores_in_range() {
        let models = builtin_models();
        for model in &models {
            assert!(
                (0.0..=1.0).contains(&model.quality_score),
                "Model {} has out-of-range quality score: {}",
                model.id,
                model.quality_score
            );
        }
    }
}
