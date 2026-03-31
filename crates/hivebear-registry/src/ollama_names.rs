//! Ollama model name resolution.
//!
//! Maps Ollama-style model names (e.g., `llama3.1:70b`, `mistral:latest`)
//! to HuggingFace repository IDs and quantization selections, enabling
//! HiveBear to act as a drop-in replacement for the Ollama CLI and API.
//!
//! # Name format
//!
//! Ollama names follow the pattern `name[:tag]` where `tag` can be:
//! - A size variant: `7b`, `8b`, `13b`, `70b`
//! - A quantization: `q4_0`, `q4_k_m`, `q8_0`
//! - A combined variant: `70b-q4_0`
//! - `latest` (default, typically the best 7-8B quantization)

use std::collections::HashMap;

/// A resolved HuggingFace target for downloading.
#[derive(Debug, Clone)]
pub struct HuggingFaceTarget {
    /// HuggingFace repo ID (e.g., `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`)
    pub repo_id: String,
    /// Preferred quantization (e.g., `Q4_K_M`)
    pub quantization: Option<String>,
}

/// Resolve an Ollama-style model name to a HuggingFace target.
///
/// Returns `None` if the model name is not recognized. The caller should
/// fall back to a HuggingFace search in that case.
pub fn resolve(ollama_name: &str) -> Option<HuggingFaceTarget> {
    let (base_name, tag) = parse_name(ollama_name);

    let entries = builtin_map();
    let entry = entries.get(base_name)?;

    // Determine which size variant to use
    let variant = resolve_variant(entry, tag);

    // Determine quantization from the tag
    let quant = resolve_quantization(tag);

    Some(HuggingFaceTarget {
        repo_id: variant.to_string(),
        quantization: quant.map(|s| s.to_string()),
    })
}

/// Parse an Ollama model name into (base_name, tag).
///
/// Examples:
/// - `llama3.1:70b` → (`llama3.1`, Some(`70b`))
/// - `mistral:latest` → (`mistral`, Some(`latest`))
/// - `codellama` → (`codellama`, None)
fn parse_name(name: &str) -> (&str, Option<&str>) {
    if let Some((base, tag)) = name.split_once(':') {
        (base, Some(tag))
    } else {
        (name, None)
    }
}

/// An entry in the builtin model map, supporting multiple size variants.
struct ModelEntry {
    /// Default repo (typically the most popular size, e.g., 8B)
    default: &'static str,
    /// Size-specific variants: `("70b", "repo-id")`
    variants: &'static [(&'static str, &'static str)],
}

/// Pick the correct repo variant from a tag.
fn resolve_variant<'a>(entry: &'a ModelEntry, tag: Option<&str>) -> &'a str {
    let Some(tag) = tag else {
        return entry.default;
    };

    if tag == "latest" {
        return entry.default;
    }

    // Extract the size portion from the tag (e.g., "70b" from "70b-q4_0")
    let size_part = tag.split('-').next().unwrap_or(tag);

    for (size, repo) in entry.variants {
        if size_part.eq_ignore_ascii_case(size) {
            return repo;
        }
    }

    // If tag looks like a pure quantization (starts with 'q'), use default
    if size_part.starts_with('q') || size_part.starts_with('Q') {
        return entry.default;
    }

    entry.default
}

/// Extract a quantization preference from the tag.
fn resolve_quantization(tag: Option<&str>) -> Option<&str> {
    let tag = tag?;

    if tag == "latest" {
        return None; // Use registry's default (Q4_K_M)
    }

    // Check if the entire tag is a quantization
    if is_quantization(tag) {
        return Some(tag);
    }

    // Check for combined format: "70b-q4_0"
    if let Some((_size, quant)) = tag.split_once('-') {
        if is_quantization(quant) {
            return Some(quant);
        }
    }

    None
}

/// Check if a string looks like a quantization identifier.
fn is_quantization(s: &str) -> bool {
    let upper = s.to_uppercase();
    upper.starts_with("Q4")
        || upper.starts_with("Q5")
        || upper.starts_with("Q6")
        || upper.starts_with("Q8")
        || upper.starts_with("Q2")
        || upper.starts_with("Q3")
        || upper == "F16"
        || upper == "F32"
        || upper.starts_with("IQ")
}

/// The builtin mapping of Ollama names to HuggingFace repos.
///
/// This covers the most popular models. Models not in this map
/// are resolved via HuggingFace search as a fallback.
fn builtin_map() -> HashMap<&'static str, ModelEntry> {
    let mut m = HashMap::new();

    // ── Llama family ─────────────────────────────────────────────────
    m.insert(
        "llama3.1",
        ModelEntry {
            default: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            variants: &[
                ("8b", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
                ("70b", "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF"),
            ],
        },
    );
    m.insert(
        "llama3",
        ModelEntry {
            default: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            variants: &[
                ("8b", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
                ("70b", "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF"),
            ],
        },
    );
    m.insert(
        "llama3.2",
        ModelEntry {
            default: "bartowski/Llama-3.2-3B-Instruct-GGUF",
            variants: &[
                ("1b", "bartowski/Llama-3.2-1B-Instruct-GGUF"),
                ("3b", "bartowski/Llama-3.2-3B-Instruct-GGUF"),
            ],
        },
    );
    m.insert(
        "llama3.3",
        ModelEntry {
            default: "bartowski/Llama-3.3-70B-Instruct-GGUF",
            variants: &[("70b", "bartowski/Llama-3.3-70B-Instruct-GGUF")],
        },
    );
    m.insert(
        "codellama",
        ModelEntry {
            default: "bartowski/CodeLlama-7b-Instruct-GGUF",
            variants: &[
                ("7b", "bartowski/CodeLlama-7b-Instruct-GGUF"),
                ("13b", "bartowski/CodeLlama-13b-Instruct-GGUF"),
                ("34b", "bartowski/CodeLlama-34b-Instruct-GGUF"),
            ],
        },
    );

    // ── Mistral / Mixtral ────────────────────────────────────────────
    m.insert(
        "mistral",
        ModelEntry {
            default: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            variants: &[("7b", "bartowski/Mistral-7B-Instruct-v0.3-GGUF")],
        },
    );
    m.insert(
        "mixtral",
        ModelEntry {
            default: "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
            variants: &[
                ("8x7b", "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF"),
                ("8x22b", "bartowski/Mixtral-8x22B-Instruct-v0.1-GGUF"),
            ],
        },
    );
    m.insert(
        "mistral-nemo",
        ModelEntry {
            default: "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
            variants: &[],
        },
    );
    m.insert(
        "mistral-small",
        ModelEntry {
            default: "bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
            variants: &[],
        },
    );

    // ── Qwen family ──────────────────────────────────────────────────
    m.insert(
        "qwen2.5",
        ModelEntry {
            default: "bartowski/Qwen2.5-7B-Instruct-GGUF",
            variants: &[
                ("0.5b", "bartowski/Qwen2.5-0.5B-Instruct-GGUF"),
                ("1.5b", "bartowski/Qwen2.5-1.5B-Instruct-GGUF"),
                ("3b", "bartowski/Qwen2.5-3B-Instruct-GGUF"),
                ("7b", "bartowski/Qwen2.5-7B-Instruct-GGUF"),
                ("14b", "bartowski/Qwen2.5-14B-Instruct-GGUF"),
                ("32b", "bartowski/Qwen2.5-32B-Instruct-GGUF"),
                ("72b", "bartowski/Qwen2.5-72B-Instruct-GGUF"),
            ],
        },
    );
    m.insert(
        "qwen2.5-coder",
        ModelEntry {
            default: "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
            variants: &[
                ("1.5b", "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF"),
                ("3b", "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF"),
                ("7b", "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"),
                ("14b", "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF"),
                ("32b", "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF"),
            ],
        },
    );
    m.insert(
        "qwen3",
        ModelEntry {
            default: "bartowski/Qwen3-8B-GGUF",
            variants: &[
                ("0.6b", "bartowski/Qwen3-0.6B-GGUF"),
                ("1.7b", "bartowski/Qwen3-1.7B-GGUF"),
                ("4b", "bartowski/Qwen3-4B-GGUF"),
                ("8b", "bartowski/Qwen3-8B-GGUF"),
                ("14b", "bartowski/Qwen3-14B-GGUF"),
                ("30b", "bartowski/Qwen3-30B-A3B-GGUF"),
                ("32b", "bartowski/Qwen3-32B-GGUF"),
            ],
        },
    );

    // ── Phi family ───────────────────────────────────────────────────
    m.insert(
        "phi3",
        ModelEntry {
            default: "bartowski/Phi-3.5-mini-instruct-GGUF",
            variants: &[
                ("mini", "bartowski/Phi-3.5-mini-instruct-GGUF"),
                ("3.8b", "bartowski/Phi-3.5-mini-instruct-GGUF"),
            ],
        },
    );
    m.insert(
        "phi4",
        ModelEntry {
            default: "bartowski/phi-4-GGUF",
            variants: &[("14b", "bartowski/phi-4-GGUF")],
        },
    );

    // ── Gemma family ─────────────────────────────────────────────────
    m.insert(
        "gemma2",
        ModelEntry {
            default: "bartowski/gemma-2-9b-it-GGUF",
            variants: &[
                ("2b", "bartowski/gemma-2-2b-it-GGUF"),
                ("9b", "bartowski/gemma-2-9b-it-GGUF"),
                ("27b", "bartowski/gemma-2-27b-it-GGUF"),
            ],
        },
    );
    m.insert(
        "gemma3",
        ModelEntry {
            default: "bartowski/gemma-3-4b-it-GGUF",
            variants: &[
                ("1b", "bartowski/gemma-3-1b-it-GGUF"),
                ("4b", "bartowski/gemma-3-4b-it-GGUF"),
                ("12b", "bartowski/gemma-3-12b-it-GGUF"),
                ("27b", "bartowski/gemma-3-27b-it-GGUF"),
            ],
        },
    );

    // ── DeepSeek family ──────────────────────────────────────────────
    m.insert(
        "deepseek-r1",
        ModelEntry {
            default: "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
            variants: &[
                ("1.5b", "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF"),
                ("7b", "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"),
                ("8b", "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF"),
                ("14b", "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF"),
                ("32b", "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF"),
                ("70b", "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF"),
            ],
        },
    );
    m.insert(
        "deepseek-coder-v2",
        ModelEntry {
            default: "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            variants: &[],
        },
    );

    // ── Small models ─────────────────────────────────────────────────
    m.insert(
        "tinyllama",
        ModelEntry {
            default: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            variants: &[],
        },
    );
    m.insert(
        "phi2",
        ModelEntry {
            default: "TheBloke/phi-2-GGUF",
            variants: &[],
        },
    );
    m.insert(
        "stablelm2",
        ModelEntry {
            default: "bartowski/stablelm-2-1_6b-chat-GGUF",
            variants: &[],
        },
    );

    // ── Other popular models ─────────────────────────────────────────
    m.insert(
        "command-r",
        ModelEntry {
            default: "bartowski/c4ai-command-r-08-2024-GGUF",
            variants: &[],
        },
    );
    m.insert(
        "yi",
        ModelEntry {
            default: "bartowski/Yi-1.5-9B-Chat-GGUF",
            variants: &[
                ("6b", "bartowski/Yi-1.5-6B-Chat-GGUF"),
                ("9b", "bartowski/Yi-1.5-9B-Chat-GGUF"),
                ("34b", "bartowski/Yi-1.5-34B-Chat-GGUF"),
            ],
        },
    );
    m.insert(
        "nous-hermes2",
        ModelEntry {
            default: "bartowski/Hermes-2-Pro-Llama-3-8B-GGUF",
            variants: &[],
        },
    );
    m.insert(
        "smollm2",
        ModelEntry {
            default: "bartowski/SmolLM2-1.7B-Instruct-GGUF",
            variants: &[
                ("135m", "bartowski/SmolLM2-135M-Instruct-GGUF"),
                ("360m", "bartowski/SmolLM2-360M-Instruct-GGUF"),
                ("1.7b", "bartowski/SmolLM2-1.7B-Instruct-GGUF"),
            ],
        },
    );

    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_basic() {
        let result = resolve("llama3.1").unwrap();
        assert_eq!(result.repo_id, "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF");
        assert!(result.quantization.is_none());
    }

    #[test]
    fn test_resolve_with_size() {
        let result = resolve("llama3.1:70b").unwrap();
        assert_eq!(result.repo_id, "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF");
    }

    #[test]
    fn test_resolve_latest() {
        let result = resolve("mistral:latest").unwrap();
        assert_eq!(result.repo_id, "bartowski/Mistral-7B-Instruct-v0.3-GGUF");
        assert!(result.quantization.is_none());
    }

    #[test]
    fn test_resolve_with_quant() {
        let result = resolve("llama3.1:q8_0").unwrap();
        assert_eq!(result.repo_id, "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF");
        assert_eq!(result.quantization.as_deref(), Some("q8_0"));
    }

    #[test]
    fn test_resolve_size_and_quant() {
        let result = resolve("llama3.1:70b-q4_0").unwrap();
        assert_eq!(result.repo_id, "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF");
        assert_eq!(result.quantization.as_deref(), Some("q4_0"));
    }

    #[test]
    fn test_resolve_unknown() {
        assert!(resolve("nonexistent-model").is_none());
    }

    #[test]
    fn test_resolve_deepseek() {
        let result = resolve("deepseek-r1:32b").unwrap();
        assert_eq!(
            result.repo_id,
            "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF"
        );
    }

    #[test]
    fn test_resolve_qwen() {
        let result = resolve("qwen2.5:72b").unwrap();
        assert_eq!(result.repo_id, "bartowski/Qwen2.5-72B-Instruct-GGUF");
    }

    #[test]
    fn test_resolve_tinyllama() {
        let result = resolve("tinyllama").unwrap();
        assert_eq!(result.repo_id, "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
    }
}
