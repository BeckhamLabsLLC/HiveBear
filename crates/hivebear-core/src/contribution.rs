use serde::{Deserialize, Serialize};

use crate::types::HardwareProfile;

/// Contribution tier based on hardware capabilities.
///
/// Determines what role a peer plays in the mesh network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContributionTier {
    /// No GPU, limited RAM. Can serve as a drafter for speculative decoding
    /// or a CPU worker for small models.
    CpuWorker,
    /// Has GPU with <4GB VRAM. Can accelerate small models.
    LightGpu,
    /// Has GPU with 4-8GB VRAM. Can serve 7-8B models at Q4.
    MidGpu,
    /// Has GPU with 8-16GB VRAM. Can serve larger models or higher quant.
    StrongGpu,
    /// Has GPU with 16+GB VRAM. Can serve 13B+ models.
    HeavyGpu,
}

impl ContributionTier {
    /// Human-readable description of this tier.
    pub fn description(&self) -> &'static str {
        match self {
            ContributionTier::CpuWorker => "CPU worker (draft models, small inference)",
            ContributionTier::LightGpu => "Light GPU (small model acceleration)",
            ContributionTier::MidGpu => "Mid GPU (7-8B models at Q4)",
            ContributionTier::StrongGpu => "Strong GPU (13B models or high-quant 7B)",
            ContributionTier::HeavyGpu => "Heavy GPU (34B+ models)",
        }
    }
}

impl std::fmt::Display for ContributionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContributionTier::CpuWorker => write!(f, "cpu-worker"),
            ContributionTier::LightGpu => write!(f, "light-gpu"),
            ContributionTier::MidGpu => write!(f, "mid-gpu"),
            ContributionTier::StrongGpu => write!(f, "strong-gpu"),
            ContributionTier::HeavyGpu => write!(f, "heavy-gpu"),
        }
    }
}

/// Plan for how a peer should contribute to the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionPlan {
    pub tier: ContributionTier,
    pub recommended_model: String,
    pub recommended_model_name: String,
    pub model_size_bytes: u64,
    pub estimated_tflops: f64,
    pub max_layers_serviceable: u32,
}

/// Determine the contribution tier from a hardware profile.
pub fn determine_tier(profile: &HardwareProfile) -> ContributionTier {
    let total_vram: u64 = profile.gpus.iter().map(|g| g.vram_bytes).sum();
    let gb = 1024u64 * 1024 * 1024;

    match total_vram {
        0 => ContributionTier::CpuWorker,
        v if v < 4 * gb => ContributionTier::LightGpu,
        v if v < 8 * gb => ContributionTier::MidGpu,
        v if v < 16 * gb => ContributionTier::StrongGpu,
        _ => ContributionTier::HeavyGpu,
    }
}

/// Plan the best contribution strategy for a given hardware profile.
///
/// Selects the model that best matches the hardware's capacity, preferring
/// the highest quality quantization that fits.
pub fn plan_contribution(profile: &HardwareProfile) -> ContributionPlan {
    let tier = determine_tier(profile);
    let total_vram: u64 = profile.gpus.iter().map(|g| g.vram_bytes).sum();
    let available_ram = (profile.memory.available_bytes as f64 * 0.85) as u64;
    let total_usable = total_vram + available_ram;
    let gb = 1024u64 * 1024 * 1024;

    // Model selection based on capacity
    let (model_id, model_name, model_size, layers) = match tier {
        ContributionTier::CpuWorker => {
            if available_ram >= 2 * gb {
                (
                    "phi-3-mini-3.8b-q4_k_m",
                    "Phi-3 Mini 3.8B",
                    (2.3 * gb as f64) as u64,
                    32,
                )
            } else {
                (
                    "tinyllama-1.1b-q4_k_m",
                    "TinyLlama 1.1B",
                    (0.7 * gb as f64) as u64,
                    22,
                )
            }
        }
        ContributionTier::LightGpu => ("llama-3.2-3b-q4_k_m", "Llama 3.2 3B", 2 * gb, 28),
        ContributionTier::MidGpu => {
            if total_usable >= 7 * gb {
                ("llama-3.1-8b-q4_k_m", "Llama 3.1 8B", 5 * gb, 32)
            } else {
                ("mistral-7b-q4_k_m", "Mistral 7B", 4 * gb, 32)
            }
        }
        ContributionTier::StrongGpu => {
            if total_usable >= 10 * gb {
                ("codellama-13b-q4_k_m", "CodeLlama 13B", 8 * gb, 40)
            } else {
                ("llama-3.1-8b-q6_k", "Llama 3.1 8B (Q6)", 7 * gb, 32)
            }
        }
        ContributionTier::HeavyGpu => {
            if total_usable >= 42 * gb {
                ("llama-3.1-70b-q4_k_m", "Llama 3.1 70B", 40 * gb, 80)
            } else {
                ("yi-34b-q4_k_m", "Yi 34B", 20 * gb, 60)
            }
        }
    };

    // Rough TFLOPS estimate from memory bandwidth
    let estimated_tflops = if total_vram > 0 {
        // GPU: rough estimate based on VRAM size as proxy for compute
        (total_vram as f64 / gb as f64) * 2.0 // ~2 TFLOPS per GB VRAM (very rough)
    } else {
        // CPU: much lower
        profile.cpu.physical_cores as f64 * 0.1
    };

    // How many layers this peer can handle locally
    let bytes_per_layer = if layers > 0 {
        model_size / layers as u64
    } else {
        model_size
    };
    let max_layers = if bytes_per_layer > 0 {
        (total_usable / bytes_per_layer).min(layers as u64) as u32
    } else {
        layers
    };

    ContributionPlan {
        tier,
        recommended_model: model_id.to_string(),
        recommended_model_name: model_name.to_string(),
        model_size_bytes: model_size,
        estimated_tflops,
        max_layers_serviceable: max_layers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn profile_with_vram(ram_gb: u64, vram_gb: u64) -> HardwareProfile {
        let gb = 1024 * 1024 * 1024;
        HardwareProfile {
            cpu: CpuInfo {
                model_name: "Test".into(),
                physical_cores: 8,
                logical_cores: 16,
                isa_extensions: vec![],
                cache_size_bytes: 0,
            },
            memory: MemoryInfo {
                total_bytes: ram_gb * gb,
                available_bytes: (ram_gb as f64 * 0.8) as u64 * gb,
                estimated_bandwidth_gbps: 30.0,
            },
            gpus: if vram_gb > 0 {
                vec![GpuInfo {
                    name: "Test GPU".into(),
                    vram_bytes: vram_gb * gb,
                    compute_api: ComputeApi::Vulkan,
                    driver_version: None,
                }]
            } else {
                vec![]
            },
            storage: StorageInfo {
                available_bytes: 100 * gb,
                estimated_read_speed_mbps: 500.0,
            },
            platform: PlatformInfo {
                os: "linux".into(),
                arch: "x86_64".into(),
                is_mobile: false,
                power_source: PowerSource::Ac,
            },
        }
    }

    #[test]
    fn test_determine_tier_cpu_only() {
        let profile = profile_with_vram(16, 0);
        assert_eq!(determine_tier(&profile), ContributionTier::CpuWorker);
    }

    #[test]
    fn test_determine_tier_mid_gpu() {
        let profile = profile_with_vram(16, 6);
        assert_eq!(determine_tier(&profile), ContributionTier::MidGpu);
    }

    #[test]
    fn test_determine_tier_strong_gpu() {
        let profile = profile_with_vram(32, 12);
        assert_eq!(determine_tier(&profile), ContributionTier::StrongGpu);
    }

    #[test]
    fn test_determine_tier_heavy_gpu() {
        let profile = profile_with_vram(64, 24);
        assert_eq!(determine_tier(&profile), ContributionTier::HeavyGpu);
    }

    #[test]
    fn test_plan_contribution_cpu() {
        let profile = profile_with_vram(8, 0);
        let plan = plan_contribution(&profile);
        assert_eq!(plan.tier, ContributionTier::CpuWorker);
        assert!(
            plan.recommended_model.contains("phi-3")
                || plan.recommended_model.contains("tinyllama")
        );
    }

    #[test]
    fn test_plan_contribution_mid_gpu() {
        let profile = profile_with_vram(16, 6);
        let plan = plan_contribution(&profile);
        assert_eq!(plan.tier, ContributionTier::MidGpu);
        assert!(plan.estimated_tflops > 0.0);
        assert!(plan.max_layers_serviceable > 0);
    }

    #[test]
    fn test_plan_contribution_heavy_gpu() {
        let profile = profile_with_vram(64, 24);
        let plan = plan_contribution(&profile);
        assert_eq!(plan.tier, ContributionTier::HeavyGpu);
        // Should recommend a large model
        assert!(plan.model_size_bytes >= 20 * 1024 * 1024 * 1024);
    }
}
