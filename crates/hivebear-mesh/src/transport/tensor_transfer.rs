use crate::transport::protocol::TensorDtype;

/// Convert tensor data between dtypes for efficient wire transfer.
///
/// Activation tensors are typically computed in F32 but can be transferred
/// in F16 with negligible quality loss, halving bandwidth requirements.
pub fn convert_for_transfer(
    data: &[u8],
    src_dtype: TensorDtype,
    target_dtype: TensorDtype,
) -> Vec<u8> {
    if src_dtype == target_dtype {
        return data.to_vec();
    }

    match (src_dtype, target_dtype) {
        (TensorDtype::F32, TensorDtype::F16) => f32_to_f16(data),
        (TensorDtype::F16, TensorDtype::F32) => f16_to_f32(data),
        _ => data.to_vec(), // No conversion available
    }
}

/// Convert F32 bytes to F16 bytes (halves the size).
fn f32_to_f16(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(4) {
        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let h = f16_from_f32(f);
        output.extend_from_slice(&h.to_le_bytes());
    }
    output
}

/// Convert F16 bytes to F32 bytes (doubles the size).
fn f16_to_f32(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len() * 2);
    for chunk in data.chunks_exact(2) {
        let h = u16::from_le_bytes([chunk[0], chunk[1]]);
        let f = f32_from_f16(h);
        output.extend_from_slice(&f.to_le_bytes());
    }
    output
}

/// Convert a single f32 to f16 (IEEE 754 half-precision).
fn f16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exponent == 0xFF {
        // Inf or NaN
        if mantissa != 0 {
            return (sign | 0x7E00) as u16; // NaN
        }
        return (sign | 0x7C00) as u16; // Inf
    }

    let new_exp = exponent - 127 + 15;

    if new_exp >= 31 {
        return (sign | 0x7C00) as u16; // Overflow -> Inf
    }

    if new_exp <= 0 {
        if new_exp < -10 {
            return sign as u16; // Too small -> 0
        }
        // Denormalized
        let m = (mantissa | 0x800000) >> (1 - new_exp);
        return (sign | (m >> 13)) as u16;
    }

    (sign | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
}

/// Convert a single f16 to f32.
fn f32_from_f16(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exponent = ((h >> 10) & 0x1F) as u32;
    let mantissa = (h & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign); // +/- 0
        }
        // Denormalized
        let mut e = 1u32;
        let mut m = mantissa;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let exp = (127 - 15 + 1 - e) << 23;
        let mant = (m & 0x3FF) << 13;
        return f32::from_bits(sign | exp | mant);
    }

    if exponent == 31 {
        if mantissa == 0 {
            return f32::from_bits(sign | 0x7F800000); // Inf
        }
        return f32::from_bits(sign | 0x7FC00000); // NaN
    }

    let exp = (exponent + 127 - 15) << 23;
    let mant = mantissa << 13;
    f32::from_bits(sign | exp | mant)
}

/// Select the optimal transfer dtype based on configuration.
pub fn select_transfer_dtype(config: &str, src_dtype: TensorDtype) -> TensorDtype {
    match config {
        "f16" => TensorDtype::F16,
        "f32" => TensorDtype::F32,
        _ => {
            // For intermediate activations, F16 is almost always fine
            if src_dtype == TensorDtype::F32 {
                TensorDtype::F16
            } else {
                src_dtype
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_to_f16_roundtrip() {
        let values: Vec<f32> = vec![1.0, -1.0, 0.5, 0.0, 3.14, 100.0, -0.001];
        let f32_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let f16_bytes = convert_for_transfer(&f32_bytes, TensorDtype::F32, TensorDtype::F16);
        assert_eq!(f16_bytes.len(), f32_bytes.len() / 2);

        let back = convert_for_transfer(&f16_bytes, TensorDtype::F16, TensorDtype::F32);
        assert_eq!(back.len(), f32_bytes.len());

        // Values should be approximately equal (F16 has less precision)
        for (i, &original) in values.iter().enumerate() {
            let recovered = f32::from_le_bytes([
                back[i * 4],
                back[i * 4 + 1],
                back[i * 4 + 2],
                back[i * 4 + 3],
            ]);
            assert!(
                (original - recovered).abs() < original.abs() * 0.01 + 0.001,
                "Mismatch at {i}: {original} vs {recovered}"
            );
        }
    }

    #[test]
    fn test_identity_conversion() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let result = convert_for_transfer(&data, TensorDtype::F16, TensorDtype::F16);
        assert_eq!(data, result);
    }

    #[test]
    fn test_select_transfer_dtype() {
        assert_eq!(
            select_transfer_dtype("f16", TensorDtype::F32),
            TensorDtype::F16
        );
        assert_eq!(
            select_transfer_dtype("f32", TensorDtype::F16),
            TensorDtype::F32
        );
        assert_eq!(
            select_transfer_dtype("auto", TensorDtype::F32),
            TensorDtype::F16
        );
        assert_eq!(
            select_transfer_dtype("auto", TensorDtype::F16),
            TensorDtype::F16
        );
    }

    #[test]
    fn test_special_values() {
        // Test 0, Inf, NaN
        let values: Vec<f32> = vec![0.0, f32::INFINITY, f32::NEG_INFINITY];
        let f32_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let f16_bytes = convert_for_transfer(&f32_bytes, TensorDtype::F32, TensorDtype::F16);
        let back = convert_for_transfer(&f16_bytes, TensorDtype::F16, TensorDtype::F32);

        let v0 = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        let v1 = f32::from_le_bytes([back[4], back[5], back[6], back[7]]);
        let v2 = f32::from_le_bytes([back[8], back[9], back[10], back[11]]);

        assert_eq!(v0, 0.0);
        assert!(v1.is_infinite() && v1.is_sign_positive());
        assert!(v2.is_infinite() && v2.is_sign_negative());
    }
}
