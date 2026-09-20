//! Minimal GGUF metadata reader.
//!
//! Only the header and key-value block are parsed — enough to answer "how
//! many transformer blocks does this model have?" without pulling tensor data
//! or taking a dependency on an inference backend.
//!
//! This exists because the mesh needs a real layer count to split a model
//! across peers. It previously hardcoded `total_layers = 32`, which is wrong
//! for most models and made every layer assignment a guess. `candle_core`
//! can parse GGUF, but it sits behind the `candle` feature, and
//! `hivebear-mesh` must be able to plan a split whatever engines are
//! compiled in.
//!
//! Reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const MAGIC: [u8; 4] = *b"GGUF";

/// Refuse absurd counts rather than trying to allocate for them; a corrupt or
/// hostile header should not be able to make us allocate gigabytes.
const MAX_KV_COUNT: u64 = 1 << 20;
const MAX_STRING_LEN: u64 = 1 << 24;
const MAX_ARRAY_LEN: u64 = 1 << 24;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("not a GGUF file: {0}")]
    NotGguf(String),
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("malformed GGUF: {0}")]
    Malformed(String),
    #[error("io error: {0}")]
    Io(#[from] io::Error),
}

/// What the mesh needs to plan a split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufMetadata {
    /// e.g. "llama", "qwen2", "phi3".
    pub architecture: String,
    /// Transformer block count, i.e. `{arch}.block_count`.
    pub block_count: Option<u32>,
}

/// Read a GGUF file's architecture and block count.
///
/// Reads only as far as the key-value block; tensor data is never touched.
pub fn read_metadata(path: &Path) -> Result<GgufMetadata, GgufError> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)
        .map_err(|_| GgufError::NotGguf("file is shorter than the 4-byte magic".into()))?;
    if magic != MAGIC {
        return Err(GgufError::NotGguf(format!(
            "expected GGUF magic, got {magic:?}"
        )));
    }

    let version = read_u32(&mut r)?;
    // v1 sized its counts and string lengths as u32; v2 widened them to u64.
    let wide = match version {
        1 => false,
        2 | 3 => true,
        other => return Err(GgufError::UnsupportedVersion(other)),
    };

    let _tensor_count = read_count(&mut r, wide)?;
    let kv_count = read_count(&mut r, wide)?;
    if kv_count > MAX_KV_COUNT {
        return Err(GgufError::Malformed(format!(
            "{kv_count} metadata entries is implausible"
        )));
    }

    let mut architecture: Option<String> = None;
    // Block count is keyed by architecture ("llama.block_count"), and the
    // architecture key is not guaranteed to come first, so collect candidates
    // and resolve at the end.
    let mut block_counts: Vec<(String, u32)> = Vec::new();

    for _ in 0..kv_count {
        let key = read_string(&mut r, wide)?;
        let value_type = read_u32(&mut r)?;

        if key == "general.architecture" && value_type == 8 {
            architecture = Some(read_string(&mut r, wide)?);
            continue;
        }

        if key.ends_with(".block_count") {
            if let Some(n) = read_scalar_as_u32(&mut r, value_type)? {
                let arch = key.trim_end_matches(".block_count").to_string();
                block_counts.push((arch, n));
                continue;
            }
            // Fell through: not a scalar we understand. The value has not
            // been consumed, so skip it normally below.
        }

        skip_value(&mut r, value_type, wide)?;
    }

    let block_count = match &architecture {
        Some(arch) => block_counts
            .iter()
            .find(|(a, _)| a == arch)
            .map(|(_, n)| *n)
            // Some files name the block-count key after a different prefix
            // than general.architecture; a single candidate is unambiguous.
            .or_else(|| {
                if block_counts.len() == 1 {
                    Some(block_counts[0].1)
                } else {
                    None
                }
            }),
        None if block_counts.len() == 1 => Some(block_counts[0].1),
        None => None,
    };

    Ok(GgufMetadata {
        architecture: architecture.unwrap_or_else(|| "unknown".into()),
        block_count,
    })
}

/// Block count only, for callers that just need to plan a split.
pub fn read_block_count(path: &Path) -> Option<u32> {
    match read_metadata(path) {
        Ok(meta) => {
            if meta.block_count.is_none() {
                tracing::debug!(
                    "GGUF at {} declares architecture '{}' but no block_count",
                    path.display(),
                    meta.architecture
                );
            }
            meta.block_count
        }
        Err(e) => {
            tracing::debug!("Could not read GGUF metadata from {}: {e}", path.display());
            None
        }
    }
}

// ── primitives ──────────────────────────────────────────────────────

fn read_u32<R: Read>(r: &mut R) -> Result<u32, GgufError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, GgufError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_count<R: Read>(r: &mut R, wide: bool) -> Result<u64, GgufError> {
    if wide {
        read_u64(r)
    } else {
        Ok(read_u32(r)? as u64)
    }
}

fn read_string<R: Read>(r: &mut R, wide: bool) -> Result<String, GgufError> {
    let len = read_count(r, wide)?;
    if len > MAX_STRING_LEN {
        return Err(GgufError::Malformed(format!(
            "string length {len} is implausible"
        )));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| GgufError::Malformed(format!("invalid UTF-8 key: {e}")))
}

/// Byte width of a fixed-size GGUF scalar type, or `None` if variable-sized.
fn scalar_width(value_type: u32) -> Option<usize> {
    match value_type {
        0 | 1 | 7 => Some(1), // u8, i8, bool
        2 | 3 => Some(2),     // u16, i16
        4..=6 => Some(4),     // u32, i32, f32
        10..=12 => Some(8),   // u64, i64, f64
        _ => None,            // 8 = string, 9 = array
    }
}

/// Read an integer scalar as u32, if `value_type` is one. Consumes the value
/// only when it returns `Some`.
fn read_scalar_as_u32<R: Read>(r: &mut R, value_type: u32) -> Result<Option<u32>, GgufError> {
    let v = match value_type {
        0 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            b[0] as u32
        }
        2 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            u16::from_le_bytes(b) as u32
        }
        4 => read_u32(r)?,
        5 => {
            let v = read_u32(r)? as i32;
            if v < 0 {
                return Err(GgufError::Malformed(format!("negative block_count {v}")));
            }
            v as u32
        }
        10 => {
            let v = read_u64(r)?;
            u32::try_from(v)
                .map_err(|_| GgufError::Malformed(format!("block_count {v} does not fit u32")))?
        }
        11 => {
            let v = read_u64(r)? as i64;
            u32::try_from(v)
                .map_err(|_| GgufError::Malformed(format!("block_count {v} out of range")))?
        }
        _ => return Ok(None),
    };
    Ok(Some(v))
}

fn skip_value<R: Read + Seek>(r: &mut R, value_type: u32, wide: bool) -> Result<(), GgufError> {
    match value_type {
        8 => {
            let len = read_count(r, wide)?;
            if len > MAX_STRING_LEN {
                return Err(GgufError::Malformed(format!(
                    "string length {len} is implausible"
                )));
            }
            r.seek(SeekFrom::Current(len as i64))?;
        }
        9 => {
            let elem_type = read_u32(r)?;
            let len = read_count(r, wide)?;
            if len > MAX_ARRAY_LEN {
                return Err(GgufError::Malformed(format!(
                    "array length {len} is implausible"
                )));
            }
            match scalar_width(elem_type) {
                Some(width) => {
                    let bytes = (len as i64).saturating_mul(width as i64);
                    r.seek(SeekFrom::Current(bytes))?;
                }
                // Arrays of strings (and nested arrays) must be walked.
                None => {
                    for _ in 0..len {
                        skip_value(r, elem_type, wide)?;
                    }
                }
            }
        }
        other => match scalar_width(other) {
            Some(width) => {
                r.seek(SeekFrom::Current(width as i64))?;
            }
            None => {
                return Err(GgufError::Malformed(format!(
                    "unknown GGUF value type {other}"
                )))
            }
        },
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Builds a GGUF v3 header by hand so the parser is tested against the
    /// real byte layout rather than a mock.
    struct GgufBuilder {
        kvs: Vec<u8>,
        count: u64,
    }

    impl GgufBuilder {
        fn new() -> Self {
            Self {
                kvs: Vec::new(),
                count: 0,
            }
        }

        fn raw_string(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }

        fn string(mut self, key: &str, value: &str) -> Self {
            Self::raw_string(&mut self.kvs, key);
            self.kvs.extend_from_slice(&8u32.to_le_bytes());
            Self::raw_string(&mut self.kvs, value);
            self.count += 1;
            self
        }

        fn u32(mut self, key: &str, value: u32) -> Self {
            Self::raw_string(&mut self.kvs, key);
            self.kvs.extend_from_slice(&4u32.to_le_bytes());
            self.kvs.extend_from_slice(&value.to_le_bytes());
            self.count += 1;
            self
        }

        fn f32(mut self, key: &str, value: f32) -> Self {
            Self::raw_string(&mut self.kvs, key);
            self.kvs.extend_from_slice(&6u32.to_le_bytes());
            self.kvs.extend_from_slice(&value.to_le_bytes());
            self.count += 1;
            self
        }

        /// An array of strings — the case that cannot be skipped by
        /// arithmetic and must be walked element by element. Real models
        /// carry a big one of these (`tokenizer.ggml.tokens`).
        fn string_array(mut self, key: &str, values: &[&str]) -> Self {
            Self::raw_string(&mut self.kvs, key);
            self.kvs.extend_from_slice(&9u32.to_le_bytes());
            self.kvs.extend_from_slice(&8u32.to_le_bytes()); // element type: string
            self.kvs
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                Self::raw_string(&mut self.kvs, v);
            }
            self.count += 1;
            self
        }

        fn u32_array(mut self, key: &str, values: &[u32]) -> Self {
            Self::raw_string(&mut self.kvs, key);
            self.kvs.extend_from_slice(&9u32.to_le_bytes());
            self.kvs.extend_from_slice(&4u32.to_le_bytes()); // element type: u32
            self.kvs
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                self.kvs.extend_from_slice(&v.to_le_bytes());
            }
            self.count += 1;
            self
        }

        fn write(self, path: &Path) {
            let mut out = Vec::new();
            out.extend_from_slice(&MAGIC);
            out.extend_from_slice(&3u32.to_le_bytes()); // version
            out.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
            out.extend_from_slice(&self.count.to_le_bytes());
            out.extend_from_slice(&self.kvs);
            // Trailing bytes stand in for tensor data; the parser must never
            // need to read this far.
            out.extend_from_slice(&[0xAB; 512]);
            let mut f = File::create(path).unwrap();
            f.write_all(&out).unwrap();
        }
    }

    fn tmpdir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("hb-gguf-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reads_block_count_past_variable_length_values() {
        let dir = tmpdir("ok");
        let path = dir.join("model.gguf");

        GgufBuilder::new()
            .string("general.architecture", "llama")
            .string("general.name", "Test Llama")
            // Variable-length values before the answer, so a parser that
            // mis-skips them lands in the wrong place.
            .string_array("tokenizer.ggml.tokens", &["<s>", "</s>", "hello", "world"])
            .u32_array("tokenizer.ggml.token_type", &[1, 1, 2, 2])
            .f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
            .u32("llama.block_count", 80)
            .u32("llama.attention.head_count", 64)
            .write(&path);

        let meta = read_metadata(&path).expect("should parse");
        assert_eq!(meta.architecture, "llama");
        assert_eq!(meta.block_count, Some(80));
        assert_eq!(read_block_count(&path), Some(80));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn picks_the_block_count_matching_the_declared_architecture() {
        let dir = tmpdir("arch");
        let path = dir.join("model.gguf");

        GgufBuilder::new()
            .u32("qwen2.block_count", 48)
            .string("general.architecture", "qwen2")
            .u32("clip.block_count", 12)
            .write(&path);

        // general.architecture appears *after* one of the candidates, so the
        // parser must resolve at the end rather than take the first hit.
        let meta = read_metadata(&path).unwrap();
        assert_eq!(meta.architecture, "qwen2");
        assert_eq!(meta.block_count, Some(48));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_non_gguf_and_truncated_files() {
        let dir = tmpdir("bad");

        let not_gguf = dir.join("a.bin");
        std::fs::write(&not_gguf, b"NOPE and some more").unwrap();
        assert!(matches!(
            read_metadata(&not_gguf),
            Err(GgufError::NotGguf(_))
        ));

        let tiny = dir.join("b.bin");
        std::fs::write(&tiny, b"GG").unwrap();
        assert!(matches!(read_metadata(&tiny), Err(GgufError::NotGguf(_))));

        let empty = dir.join("c.bin");
        std::fs::write(&empty, b"").unwrap();
        assert!(matches!(read_metadata(&empty), Err(GgufError::NotGguf(_))));

        // Header claims entries that are not there.
        let truncated = dir.join("d.gguf");
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&5u64.to_le_bytes()); // says 5 kvs, supplies none
        std::fs::write(&truncated, &out).unwrap();
        assert!(read_metadata(&truncated).is_err());
        assert_eq!(read_block_count(&truncated), None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_an_implausible_kv_count_instead_of_allocating() {
        let dir = tmpdir("huge");
        let path = dir.join("hostile.gguf");
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, &out).unwrap();

        assert!(matches!(read_metadata(&path), Err(GgufError::Malformed(_))));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_block_count_is_none_not_an_error() {
        let dir = tmpdir("nobc");
        let path = dir.join("model.gguf");
        GgufBuilder::new()
            .string("general.architecture", "mystery")
            .write(&path);

        let meta = read_metadata(&path).unwrap();
        assert_eq!(meta.architecture, "mystery");
        assert_eq!(meta.block_count, None);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
