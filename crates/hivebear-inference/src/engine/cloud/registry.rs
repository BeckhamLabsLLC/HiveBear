//! Provider registry — declarative catalog of all supported cloud providers.

use std::collections::HashMap;

/// Which API protocol family a provider uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApiProtocol {
    /// OpenAI-compatible `/v1/chat/completions` (covers ~25 providers).
    OpenAiCompat,
    /// Anthropic Messages API (`/v1/messages`).
    Anthropic,
    /// Google Gemini `generateContent` / `streamGenerateContent`.
    Gemini,
    /// Cohere `/v2/chat`.
    Cohere,
}

/// How the provider authenticates requests.
#[derive(Debug, Clone)]
pub enum AuthStyle {
    /// `Authorization: Bearer <key>` (most common).
    BearerToken,
    /// Custom header name, e.g. Anthropic uses `x-api-key`, Azure uses `api-key`.
    CustomHeader(&'static str),
    /// API key appended as a URL query parameter (e.g. `?key=...` for Gemini).
    ApiKeyInUrl,
    /// No authentication required (local servers like Ollama).
    None,
}

/// Provider-specific capabilities.
#[derive(Debug, Clone)]
pub struct ProviderCapabilities {
    pub supports_streaming: bool,
    pub supports_tool_calling: bool,
    pub supports_vision: bool,
    pub supports_structured_output: bool,
    pub max_context_length: Option<u32>,
}

impl Default for ProviderCapabilities {
    fn default() -> Self {
        Self {
            supports_streaming: true,
            supports_tool_calling: false,
            supports_vision: false,
            supports_structured_output: false,
            max_context_length: None,
        }
    }
}

/// Static definition of a cloud provider.
#[derive(Debug, Clone)]
pub struct ProviderDef {
    /// Prefix used in model names, e.g. `"openai"`, `"groq"`, `"together"`.
    pub prefix: &'static str,
    /// Human-readable display name.
    pub display_name: &'static str,
    /// Base URL for the API (without trailing slash).
    pub base_url: &'static str,
    /// API protocol family.
    pub protocol: ApiProtocol,
    /// Authentication style.
    pub auth_style: AuthStyle,
    /// Environment variable name for the API key.
    pub env_var: &'static str,
    /// Extra headers to include on every request.
    pub extra_headers: &'static [(&'static str, &'static str)],
    /// Chat endpoint path relative to `base_url`.
    pub chat_endpoint: &'static str,
    /// Whether this provider requires an API key (false for local servers).
    pub requires_api_key: bool,
    /// Provider capabilities.
    pub capabilities: ProviderCapabilities,
}

// ── Helper for concise declarations ──────────────────────────────────

const fn caps(
    streaming: bool,
    tools: bool,
    vision: bool,
    structured: bool,
    ctx: Option<u32>,
) -> ProviderCapabilities {
    ProviderCapabilities {
        supports_streaming: streaming,
        supports_tool_calling: tools,
        supports_vision: vision,
        supports_structured_output: structured,
        max_context_length: ctx,
    }
}

// ── Builtin Provider Definitions ─────────────────────────────────────

pub static BUILTIN_PROVIDERS: &[ProviderDef] = &[
    // ─── OpenAI-Compatible Providers ─────────────────────────────
    ProviderDef {
        prefix: "openai",
        display_name: "OpenAI",
        base_url: "https://api.openai.com/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "OPENAI_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, true, true, Some(128_000)),
    },
    ProviderDef {
        prefix: "groq",
        display_name: "Groq",
        base_url: "https://api.groq.com/openai/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "GROQ_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, true, true, Some(131_072)),
    },
    ProviderDef {
        prefix: "together",
        display_name: "Together AI",
        base_url: "https://api.together.xyz/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "TOGETHER_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, true, true, Some(131_072)),
    },
    ProviderDef {
        prefix: "fireworks",
        display_name: "Fireworks AI",
        base_url: "https://api.fireworks.ai/inference/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "FIREWORKS_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, true, true, Some(131_072)),
    },
    ProviderDef {
        prefix: "deepseek",
        display_name: "DeepSeek",
        base_url: "https://api.deepseek.com",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "DEEPSEEK_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, false, true, Some(64_000)),
    },
    ProviderDef {
        prefix: "xai",
        display_name: "xAI (Grok)",
        base_url: "https://api.x.ai/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "XAI_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, true, true, Some(131_072)),
    },
    ProviderDef {
        prefix: "cerebras",
        display_name: "Cerebras",
        base_url: "https://api.cerebras.ai/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "CEREBRAS_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, false, true, Some(8_192)),
    },
    ProviderDef {
        prefix: "sambanova",
        display_name: "SambaNova",
        base_url: "https://api.sambanova.ai/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "SAMBANOVA_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, false, true, Some(8_192)),
    },
    ProviderDef {
        prefix: "perplexity",
        display_name: "Perplexity",
        base_url: "https://api.perplexity.ai",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "PERPLEXITY_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, false, false, false, Some(127_072)),
    },
    ProviderDef {
        prefix: "mistral",
        display_name: "Mistral",
        base_url: "https://api.mistral.ai/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "MISTRAL_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, true, true, Some(128_000)),
    },
    ProviderDef {
        prefix: "openrouter",
        display_name: "OpenRouter",
        base_url: "https://openrouter.ai/api/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "OPENROUTER_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, true, true, Some(200_000)),
    },
    ProviderDef {
        prefix: "nvidia",
        display_name: "NVIDIA NIM",
        base_url: "https://integrate.api.nvidia.com/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "NVIDIA_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, false, true, Some(32_768)),
    },
    ProviderDef {
        prefix: "cloudflare",
        display_name: "Cloudflare Workers AI",
        base_url: "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "CLOUDFLARE_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, false, false, false, Some(4_096)),
    },
    ProviderDef {
        prefix: "huggingface",
        display_name: "HuggingFace Inference",
        base_url: "https://api-inference.huggingface.co/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "HF_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, false, false, Some(32_768)),
    },
    ProviderDef {
        prefix: "replicate",
        display_name: "Replicate",
        base_url: "https://openai-proxy.replicate.com/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "REPLICATE_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, false, false, false, None),
    },
    ProviderDef {
        prefix: "lepton",
        display_name: "Lepton AI",
        base_url: "https://api.lepton.ai/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "LEPTON_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, false, false, Some(32_768)),
    },
    ProviderDef {
        prefix: "octoai",
        display_name: "OctoAI",
        base_url: "https://text.octoai.run/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "OCTOAI_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, false, false, Some(32_768)),
    },
    ProviderDef {
        prefix: "anyscale",
        display_name: "Anyscale",
        base_url: "https://api.endpoints.anyscale.com/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "ANYSCALE_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, false, false, Some(16_384)),
    },
    ProviderDef {
        prefix: "moonshot",
        display_name: "Moonshot AI",
        base_url: "https://api.moonshot.cn/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "MOONSHOT_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, false, false, Some(128_000)),
    },
    ProviderDef {
        prefix: "dashscope",
        display_name: "Alibaba DashScope",
        base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "DASHSCOPE_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, true, true, Some(128_000)),
    },
    ProviderDef {
        prefix: "zhipu",
        display_name: "Zhipu AI (GLM)",
        base_url: "https://open.bigmodel.cn/api/paas/v4",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "ZHIPU_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, true, false, Some(128_000)),
    },
    ProviderDef {
        prefix: "baichuan",
        display_name: "Baichuan",
        base_url: "https://api.baichuan-ai.com/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "BAICHUAN_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, false, false, Some(32_768)),
    },
    ProviderDef {
        prefix: "minimax",
        display_name: "Minimax",
        base_url: "https://api.minimax.chat/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "MINIMAX_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/text/chatcompletion_v2",
        requires_api_key: true,
        capabilities: caps(true, true, false, false, Some(245_760)),
    },
    ProviderDef {
        prefix: "yi",
        display_name: "01.AI (Yi)",
        base_url: "https://api.lingyiwanwu.com/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "YI_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, true, false, Some(200_000)),
    },
    ProviderDef {
        prefix: "azure",
        display_name: "Azure OpenAI",
        base_url: "https://{resource}.openai.azure.com/openai/deployments/{deployment}",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::CustomHeader("api-key"),
        env_var: "AZURE_OPENAI_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions?api-version=2024-10-21",
        requires_api_key: true,
        capabilities: caps(true, true, true, true, Some(128_000)),
    },
    // ─── Custom Protocol Providers ───────────────────────────────
    ProviderDef {
        prefix: "anthropic",
        display_name: "Anthropic",
        base_url: "https://api.anthropic.com",
        protocol: ApiProtocol::Anthropic,
        auth_style: AuthStyle::CustomHeader("x-api-key"),
        env_var: "ANTHROPIC_API_KEY",
        extra_headers: &[("anthropic-version", "2023-06-01")],
        chat_endpoint: "/v1/messages",
        requires_api_key: true,
        capabilities: caps(true, true, true, true, Some(200_000)),
    },
    ProviderDef {
        prefix: "gemini",
        display_name: "Google Gemini",
        base_url: "https://generativelanguage.googleapis.com",
        protocol: ApiProtocol::Gemini,
        auth_style: AuthStyle::ApiKeyInUrl,
        env_var: "GEMINI_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/v1beta/models/{model}:generateContent",
        requires_api_key: true,
        capabilities: caps(true, true, true, true, Some(1_048_576)),
    },
    ProviderDef {
        prefix: "cohere",
        display_name: "Cohere",
        base_url: "https://api.cohere.com",
        protocol: ApiProtocol::Cohere,
        auth_style: AuthStyle::BearerToken,
        env_var: "COHERE_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/v2/chat",
        requires_api_key: true,
        capabilities: caps(true, true, false, true, Some(128_000)),
    },
    ProviderDef {
        prefix: "ai21",
        display_name: "AI21 Labs",
        base_url: "https://api.ai21.com/studio/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "AI21_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, true, false, false, Some(256_000)),
    },
    ProviderDef {
        prefix: "reka",
        display_name: "Reka AI",
        base_url: "https://api.reka.ai/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::BearerToken,
        env_var: "REKA_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: true,
        capabilities: caps(true, false, true, false, Some(128_000)),
    },
    // ─── Local / Self-Hosted (OpenAI-compatible) ─────────────────
    ProviderDef {
        prefix: "ollama",
        display_name: "Ollama",
        base_url: "http://localhost:11434/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::None,
        env_var: "OLLAMA_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: false,
        capabilities: caps(true, true, true, true, Some(131_072)),
    },
    ProviderDef {
        prefix: "lmstudio",
        display_name: "LM Studio",
        base_url: "http://localhost:1234/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::None,
        env_var: "LMSTUDIO_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: false,
        capabilities: caps(true, true, false, false, Some(32_768)),
    },
    ProviderDef {
        prefix: "vllm",
        display_name: "vLLM",
        base_url: "http://localhost:8000/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::None,
        env_var: "VLLM_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: false,
        capabilities: caps(true, true, false, true, Some(32_768)),
    },
    ProviderDef {
        prefix: "tgi",
        display_name: "Text Generation Inference",
        base_url: "http://localhost:8080/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::None,
        env_var: "TGI_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: false,
        capabilities: caps(true, true, false, true, Some(32_768)),
    },
    ProviderDef {
        prefix: "localai",
        display_name: "LocalAI",
        base_url: "http://localhost:8080/v1",
        protocol: ApiProtocol::OpenAiCompat,
        auth_style: AuthStyle::None,
        env_var: "LOCALAI_API_KEY",
        extra_headers: &[],
        chat_endpoint: "/chat/completions",
        requires_api_key: false,
        capabilities: caps(true, true, false, false, Some(32_768)),
    },
];

// ── Resolution Logic ─────────────────────────────────────────────────

/// A resolved provider ready for use: definition + api key + model ID.
pub struct ResolvedProvider<'a> {
    pub def: &'a ProviderDef,
    pub api_key: Option<String>,
    pub model_id: String,
}

/// Find a builtin provider by prefix.
pub fn find_builtin(prefix: &str) -> Option<&'static ProviderDef> {
    BUILTIN_PROVIDERS.iter().find(|p| p.prefix == prefix)
}

/// Check whether a model name looks like a cloud model (`prefix/model_id`).
pub fn is_cloud_model(model_name: &str) -> bool {
    if let Some((prefix, _)) = model_name.split_once('/') {
        find_builtin(prefix).is_some()
    } else {
        false
    }
}

/// Resolve API keys for all providers from config + environment variables.
pub fn resolve_all_keys(
    config_keys: &HashMap<String, String>,
    legacy_openai: Option<&str>,
    legacy_anthropic: Option<&str>,
) -> HashMap<String, String> {
    let mut keys = config_keys.clone();

    // Legacy field migration
    if let Some(k) = legacy_openai {
        keys.entry("openai".into()).or_insert_with(|| k.to_string());
    }
    if let Some(k) = legacy_anthropic {
        keys.entry("anthropic".into())
            .or_insert_with(|| k.to_string());
    }

    // Environment variable fallback for all builtin providers
    for provider in BUILTIN_PROVIDERS {
        if !keys.contains_key(provider.prefix) {
            if let Ok(val) = std::env::var(provider.env_var) {
                if !val.is_empty() {
                    keys.insert(provider.prefix.to_string(), val);
                }
            }
        }
    }

    keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_all_prefixes_unique() {
        let mut seen = HashSet::new();
        for p in BUILTIN_PROVIDERS {
            assert!(
                seen.insert(p.prefix),
                "Duplicate provider prefix: '{}'",
                p.prefix
            );
        }
    }

    #[test]
    fn test_provider_count() {
        assert!(
            BUILTIN_PROVIDERS.len() >= 35,
            "Expected at least 35 providers, got {}",
            BUILTIN_PROVIDERS.len()
        );
    }

    #[test]
    fn test_all_providers_have_valid_urls() {
        for p in BUILTIN_PROVIDERS {
            assert!(
                p.base_url.starts_with("http://") || p.base_url.starts_with("https://"),
                "Provider '{}' has invalid base_url: '{}'",
                p.prefix,
                p.base_url
            );
            assert!(
                !p.display_name.is_empty(),
                "Provider '{}' has empty display_name",
                p.prefix
            );
            assert!(
                !p.chat_endpoint.is_empty(),
                "Provider '{}' has empty chat_endpoint",
                p.prefix
            );
        }
    }

    #[test]
    fn test_all_cloud_providers_have_env_var() {
        for p in BUILTIN_PROVIDERS {
            if p.requires_api_key {
                assert!(
                    !p.env_var.is_empty(),
                    "Provider '{}' requires API key but has no env_var",
                    p.prefix
                );
            }
        }
    }

    #[test]
    fn test_find_builtin() {
        assert!(find_builtin("openai").is_some());
        assert!(find_builtin("groq").is_some());
        assert!(find_builtin("anthropic").is_some());
        assert!(find_builtin("nonexistent").is_none());
    }

    #[test]
    fn test_is_cloud_model() {
        assert!(is_cloud_model("openai/gpt-4o"));
        assert!(is_cloud_model("groq/llama-3.1-70b-versatile"));
        assert!(is_cloud_model("anthropic/claude-sonnet-4-20250514"));
        assert!(is_cloud_model("ollama/llama3"));
        assert!(!is_cloud_model("llama3"));
        assert!(!is_cloud_model("/models/test.gguf"));
    }

    #[test]
    fn test_resolve_all_keys_env_fallback() {
        // With empty config, keys come from env vars only
        let config = HashMap::new();
        let keys = resolve_all_keys(&config, None, None);
        // We can't guarantee env vars are set, but the function shouldn't panic
        assert!(keys.len() <= BUILTIN_PROVIDERS.len());
    }

    #[test]
    fn test_resolve_all_keys_legacy_migration() {
        let config = HashMap::new();
        let keys = resolve_all_keys(&config, Some("sk-test-openai"), Some("sk-test-anthropic"));
        assert_eq!(
            keys.get("openai").map(|s| s.as_str()),
            Some("sk-test-openai")
        );
        assert_eq!(
            keys.get("anthropic").map(|s| s.as_str()),
            Some("sk-test-anthropic")
        );
    }

    #[test]
    fn test_config_keys_take_precedence_over_legacy() {
        let mut config = HashMap::new();
        config.insert("openai".to_string(), "sk-from-config".to_string());
        let keys = resolve_all_keys(&config, Some("sk-from-legacy"), None);
        assert_eq!(
            keys.get("openai").map(|s| s.as_str()),
            Some("sk-from-config")
        );
    }
}
