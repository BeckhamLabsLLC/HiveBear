use crate::error::{MeshError, Result};
use crate::transport::protocol::CompressionMethod;

/// Compress data using the specified method.
pub fn compress(data: &[u8], method: CompressionMethod) -> Result<Vec<u8>> {
    match method {
        CompressionMethod::None => Ok(data.to_vec()),
        CompressionMethod::Lz4 => Ok(lz4_flex::compress_prepend_size(data)),
        CompressionMethod::Zstd { level } => {
            zstd::encode_all(std::io::Cursor::new(data), level as i32)
                .map_err(|e| MeshError::Compression(format!("Zstd compression failed: {e}")))
        }
    }
}

/// Decompress data using the specified method.
pub fn decompress(data: &[u8], method: CompressionMethod) -> Result<Vec<u8>> {
    match method {
        CompressionMethod::None => Ok(data.to_vec()),
        CompressionMethod::Lz4 => lz4_flex::decompress_size_prepended(data)
            .map_err(|e| MeshError::Compression(format!("Lz4 decompression failed: {e}"))),
        CompressionMethod::Zstd { .. } => zstd::decode_all(std::io::Cursor::new(data))
            .map_err(|e| MeshError::Compression(format!("Zstd decompression failed: {e}"))),
    }
}

/// Select the best compression method based on data size and available bandwidth.
pub fn select_compression(data_size: usize, bandwidth_mbps: f64) -> CompressionMethod {
    if data_size < 4096 {
        // Small data: overhead not worth it
        CompressionMethod::None
    } else if bandwidth_mbps > 1000.0 {
        // Fast network: don't spend CPU compressing
        CompressionMethod::None
    } else if bandwidth_mbps < 100.0 {
        // Slow network: best compression ratio
        CompressionMethod::Zstd { level: 3 }
    } else {
        // Medium network: fast compression
        CompressionMethod::Lz4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_roundtrip() {
        let data = vec![42u8; 8192]; // Compressible data
        let compressed = compress(&data, CompressionMethod::Lz4).unwrap();
        let decompressed = decompress(&compressed, CompressionMethod::Lz4).unwrap();
        assert_eq!(data, decompressed);
        assert!(compressed.len() < data.len()); // Should actually compress
    }

    #[test]
    fn test_zstd_roundtrip() {
        let data = vec![42u8; 8192];
        let compressed = compress(&data, CompressionMethod::Zstd { level: 3 }).unwrap();
        let decompressed = decompress(&compressed, CompressionMethod::Zstd { level: 3 }).unwrap();
        assert_eq!(data, decompressed);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_none_passthrough() {
        let data = vec![1, 2, 3, 4, 5];
        let result = compress(&data, CompressionMethod::None).unwrap();
        assert_eq!(data, result);
        let back = decompress(&result, CompressionMethod::None).unwrap();
        assert_eq!(data, back);
    }

    #[test]
    fn test_select_compression() {
        assert_eq!(select_compression(100, 500.0), CompressionMethod::None); // Small
        assert_eq!(select_compression(8192, 2000.0), CompressionMethod::None); // Fast net
        assert_eq!(
            select_compression(8192, 50.0),
            CompressionMethod::Zstd { level: 3 }
        ); // Slow net
        assert_eq!(select_compression(8192, 500.0), CompressionMethod::Lz4); // Medium
    }
}
