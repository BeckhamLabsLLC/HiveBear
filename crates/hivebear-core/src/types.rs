use serde::{Deserialize, Serialize};
use std::fmt;

/// Complete hardware profile for a device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub gpus: Vec<GpuInfo>,
    pub storage: StorageInfo,
    pub platform: PlatformInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub model_name: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub isa_extensions: Vec<String>,
    pub cache_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub estimated_bandwidth_gbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_bytes: u64,
    pub compute_api: ComputeApi,
    pub driver_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    pub available_bytes: u64,
    pub estimated_read_speed_mbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    pub os: String,
    pub arch: String,
    pub is_mobile: bool,
    pub power_source: PowerSource,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PowerSource {
    Ac,
    Battery { charge_percent: u8 },
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ComputeApi {
    Vulkan,
    Cuda,
    Metal,
    OpenCl,
    WebGpu,
    DirectX12,
    None,
}

impl fmt::Display for ComputeApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeApi::Vulkan => write!(f, "Vulkan"),
            ComputeApi::Cuda => write!(f, "CUDA"),
            ComputeApi::Metal => write!(f, "Metal"),
            ComputeApi::OpenCl => write!(f, "OpenCL"),
            ComputeApi::WebGpu => write!(f, "WebGPU"),
            ComputeApi::DirectX12 => write!(f, "DirectX 12"),
            ComputeApi::None => write!(f, "None"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum InferenceEngine {
    LlamaCpp,
    OnnxRuntime,
    Mlx,
    Candle,
    WebGpu,
    Mesh,
    Cloud,
}

impl fmt::Display for InferenceEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferenceEngine::LlamaCpp => write!(f, "llama.cpp"),
            InferenceEngine::OnnxRuntime => write!(f, "ONNX Runtime"),
            InferenceEngine::Mlx => write!(f, "MLX"),
            InferenceEngine::Candle => write!(f, "Candle"),
            InferenceEngine::WebGpu => write!(f, "WebGPU"),
            InferenceEngine::Mesh => write!(f, "P2P Mesh"),
            InferenceEngine::Cloud => write!(f, "Cloud"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Quantization {
    Q2K,
    Q3KS,
    Q3KM,
    Q3KL,
    Q4_0,
    Q4_1,
    Q4KS,
    Q4KM,
    Q5_0,
    Q5_1,
    Q5KS,
    Q5KM,
    Q6K,
    Q8_0,
    F16,
    F32,
}

impl Quantization {
    /// Bits per weight for this quantization level.
    pub fn bits_per_weight(&self) -> f64 {
        match self {
            Quantization::Q2K => 2.5,
            Quantization::Q3KS => 3.4375,
            Quantization::Q3KM => 3.4375,
            Quantization::Q3KL => 3.4375,
            Quantization::Q4_0 => 4.0,
            Quantization::Q4_1 => 4.5,
            Quantization::Q4KS => 4.5,
            Quantization::Q4KM => 4.5,
            Quantization::Q5_0 => 5.0,
            Quantization::Q5_1 => 5.5,
            Quantization::Q5KS => 5.5,
            Quantization::Q5KM => 5.5,
            Quantization::Q6K => 6.5,
            Quantization::Q8_0 => 8.0,
            Quantization::F16 => 16.0,
            Quantization::F32 => 32.0,
        }
    }

    /// Estimated quality retention (0.0 - 1.0) relative to FP16.
    pub fn quality_retention(&self) -> f64 {
        match self {
            Quantization::Q2K => 0.70,
            Quantization::Q3KS => 0.78,
            Quantization::Q3KM => 0.82,
            Quantization::Q3KL => 0.84,
            Quantization::Q4_0 => 0.87,
            Quantization::Q4_1 => 0.88,
            Quantization::Q4KS => 0.89,
            Quantization::Q4KM => 0.92,
            Quantization::Q5_0 => 0.93,
            Quantization::Q5_1 => 0.94,
            Quantization::Q5KS => 0.94,
            Quantization::Q5KM => 0.95,
            Quantization::Q6K => 0.97,
            Quantization::Q8_0 => 0.99,
            Quantization::F16 => 1.0,
            Quantization::F32 => 1.0,
        }
    }
}

impl fmt::Display for Quantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Quantization::Q2K => write!(f, "Q2_K"),
            Quantization::Q3KS => write!(f, "Q3_K_S"),
            Quantization::Q3KM => write!(f, "Q3_K_M"),
            Quantization::Q3KL => write!(f, "Q3_K_L"),
            Quantization::Q4_0 => write!(f, "Q4_0"),
            Quantization::Q4_1 => write!(f, "Q4_1"),
            Quantization::Q4KS => write!(f, "Q4_K_S"),
            Quantization::Q4KM => write!(f, "Q4_K_M"),
            Quantization::Q5_0 => write!(f, "Q5_0"),
            Quantization::Q5_1 => write!(f, "Q5_1"),
            Quantization::Q5KS => write!(f, "Q5_K_S"),
            Quantization::Q5KM => write!(f, "Q5_K_M"),
            Quantization::Q6K => write!(f, "Q6_K"),
            Quantization::Q8_0 => write!(f, "Q8_0"),
            Quantization::F16 => write!(f, "F16"),
            Quantization::F32 => write!(f, "F32"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    Gguf,
    Onnx,
    Mlx,
    SafeTensors,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelFormat::Gguf => write!(f, "GGUF"),
            ModelFormat::Onnx => write!(f, "ONNX"),
            ModelFormat::Mlx => write!(f, "MLX"),
            ModelFormat::SafeTensors => write!(f, "SafeTensors"),
        }
    }
}

/// A model recommendation for the user's hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendation {
    pub model_id: String,
    pub model_name: String,
    pub quantization: Quantization,
    pub engine: InferenceEngine,
    pub estimated_tokens_per_sec: f32,
    pub estimated_memory_usage_bytes: u64,
    pub confidence: f32,
    pub warnings: Vec<String>,
    pub score: f64,
}

/// Profiling mode: estimate-only or with real benchmarks.
#[derive(Debug, Clone, Copy)]
pub enum ProfileMode {
    Estimate,
    Benchmark { duration_secs: u32 },
}

/// Results from running a real inference benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub model_used: String,
    pub tokens_generated: u32,
    pub total_duration_ms: u64,
    pub tokens_per_sec: f32,
    pub time_to_first_token_ms: u64,
    pub peak_memory_bytes: u64,
    pub cpu_utilization: f32,
    pub gpu_utilization: Option<f32>,
    /// "synthetic" for CPU matmul estimate, "inference" for real model benchmark.
    #[serde(default = "default_benchmark_type")]
    pub benchmark_type: String,
    /// Prompt evaluation (prefill) tokens per second.
    #[serde(default)]
    pub prompt_eval_tokens_per_sec: Option<f32>,
}

fn default_benchmark_type() -> String {
    "synthetic".to_string()
}

/// Helper to format bytes into human-readable strings.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.1} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
