use crate::error::{RegistryError, Result};
use hivebear_core::types::{ModelFormat, Quantization};
use std::path::{Path, PathBuf};

/// Information about an available conversion tool.
#[derive(Debug, Clone)]
pub struct ConverterInfo {
    pub name: String,
    pub path: PathBuf,
    pub supports: Vec<ConversionKind>,
}

/// What kind of conversion a tool supports.
#[derive(Debug, Clone)]
pub enum ConversionKind {
    /// SafeTensors/HF -> GGUF
    ToGguf,
    /// GGUF quantization (e.g., F16 -> Q4_K_M)
    Quantize,
}

/// Options for model conversion.
pub struct ConversionOptions {
    pub target_format: ModelFormat,
    pub target_quantization: Option<Quantization>,
}

/// Check what conversion tools are available on this system.
pub async fn available_converters() -> Vec<ConverterInfo> {
    let mut converters = Vec::new();

    // Check for llama-quantize
    if let Ok(output) = tokio::process::Command::new("which")
        .arg("llama-quantize")
        .output()
        .await
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            converters.push(ConverterInfo {
                name: "llama-quantize".into(),
                path: PathBuf::from(path),
                supports: vec![ConversionKind::Quantize],
            });
        }
    }

    // Check for python3 + convert script
    if let Ok(output) = tokio::process::Command::new("python3")
        .arg("--version")
        .output()
        .await
    {
        if output.status.success() {
            converters.push(ConverterInfo {
                name: "python3 (for convert scripts)".into(),
                path: PathBuf::from("python3"),
                supports: vec![ConversionKind::ToGguf],
            });
        }
    }

    converters
}

/// Convert a model file to a different format or quantization.
pub async fn convert(source: &Path, options: &ConversionOptions) -> Result<PathBuf> {
    let converters = available_converters().await;

    match options.target_format {
        ModelFormat::Gguf => {
            if let Some(quant) = &options.target_quantization {
                // GGUF re-quantization
                let has_quantize = converters.iter().any(|c| {
                    c.supports
                        .iter()
                        .any(|s| matches!(s, ConversionKind::Quantize))
                });

                if !has_quantize {
                    return Err(RegistryError::ConversionUnavailable(format!(
                        "llama-quantize not found. Install it from https://github.com/ggerganov/llama.cpp\n\
                         Then run: hivebear convert {} --to gguf --quant {}",
                        source.display(),
                        quant
                    )));
                }

                let output_path = source.with_extension(format!("{quant}.gguf"));
                let status = tokio::process::Command::new("llama-quantize")
                    .arg(source)
                    .arg(&output_path)
                    .arg(quant.to_string())
                    .status()
                    .await?;

                if status.success() {
                    Ok(output_path)
                } else {
                    Err(RegistryError::ConversionUnavailable(
                        "llama-quantize failed. Check the model file format.".into(),
                    ))
                }
            } else {
                // SafeTensors/HF → GGUF via llama.cpp's convert script
                let has_python = converters.iter().any(|c| {
                    c.supports
                        .iter()
                        .any(|s| matches!(s, ConversionKind::ToGguf))
                });

                if !has_python {
                    return Err(RegistryError::ConversionUnavailable(
                        "Python 3 not found. GGUF conversion requires python3 and the \
                         llama.cpp convert script.\n\
                         Install: pip install -r requirements.txt (from llama.cpp repo)\n\
                         Then: hivebear convert <model_dir> --to gguf"
                            .into(),
                    ));
                }

                // Look for convert_hf_to_gguf.py in common locations
                let convert_script = find_convert_script().await;
                match convert_script {
                    Some(script) => {
                        let output_path = source.with_extension("gguf");
                        let status = tokio::process::Command::new("python3")
                            .arg(&script)
                            .arg(source)
                            .arg("--outfile")
                            .arg(&output_path)
                            .status()
                            .await?;

                        if status.success() {
                            Ok(output_path)
                        } else {
                            Err(RegistryError::ConversionUnavailable(format!(
                                "convert_hf_to_gguf.py failed. Check model format compatibility.\n\
                                 Script used: {}",
                                script.display()
                            )))
                        }
                    }
                    None => Err(RegistryError::ConversionUnavailable(
                        "convert_hf_to_gguf.py not found. Install llama.cpp and ensure \
                         the convert script is in your PATH or at ~/llama.cpp/convert_hf_to_gguf.py"
                            .into(),
                    )),
                }
            }
        }
        _ => Err(RegistryError::ConversionUnavailable(format!(
            "Conversion to {} is not yet supported. \
             Currently only GGUF quantization is available via external tools.",
            options.target_format
        ))),
    }
}

/// Search common locations for llama.cpp's convert_hf_to_gguf.py script.
async fn find_convert_script() -> Option<PathBuf> {
    let candidates = [
        // Check PATH first
        "convert_hf_to_gguf.py",
    ];

    for name in &candidates {
        if let Ok(output) = tokio::process::Command::new("which")
            .arg(name)
            .output()
            .await
        {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                return Some(PathBuf::from(path));
            }
        }
    }

    // Check common directories
    let home = std::env::var("HOME").unwrap_or_default();
    let dir_candidates = [
        format!("{home}/llama.cpp/convert_hf_to_gguf.py"),
        format!("{home}/.local/bin/convert_hf_to_gguf.py"),
        "/usr/local/bin/convert_hf_to_gguf.py".into(),
    ];

    for path_str in &dir_candidates {
        let path = PathBuf::from(path_str);
        if path.exists() {
            return Some(path);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_convert_unsupported_format() {
        let result = convert(
            Path::new("/tmp/model.gguf"),
            &ConversionOptions {
                target_format: ModelFormat::Onnx,
                target_quantization: None,
            },
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not yet supported"));
    }

    #[tokio::test]
    async fn test_available_converters() {
        // Just verify it doesn't panic
        let _converters = available_converters().await;
    }
}
