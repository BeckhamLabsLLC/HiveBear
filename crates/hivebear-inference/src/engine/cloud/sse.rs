//! SSE (Server-Sent Events) stream parser for cloud provider responses.
//!
//! Reads a `reqwest::Response` byte stream and yields parsed `data:` payloads
//! one at a time, handling the SSE protocol correctly (multi-line events,
//! empty-line delimiters, `[DONE]` sentinel).

use futures::StreamExt;

use crate::error::{InferenceError, Result};
use crate::types::Token;

/// Parse an SSE byte stream, calling `parse_chunk` for each `data:` payload.
///
/// `done_sentinel` — if `Some("...")`, that exact string in a `data:` line
/// signals end-of-stream (e.g. `[DONE]` for OpenAI).
///
/// `parse_chunk` — converts a single `data:` payload string into an optional `Token`.
pub fn stream_sse_with_parser<F>(
    response: reqwest::Response,
    done_sentinel: Option<String>,
    parse_chunk: F,
) -> futures::stream::BoxStream<'static, Result<Token>>
where
    F: Fn(&str) -> Result<Option<Token>> + Send + Sync + 'static,
{
    let stream = async_stream::try_stream! {
        let mut byte_stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = chunk_result
                .map_err(|e| InferenceError::GenerationError(format!("Stream read error: {e}")))?;
            let text = String::from_utf8_lossy(&chunk);
            buffer.push_str(&text);

            // Process all complete lines in the buffer
            loop {
                let Some(newline_pos) = buffer.find('\n') else {
                    break;
                };

                let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                // Skip empty lines (SSE event delimiters)
                if line.is_empty() {
                    continue;
                }

                // Skip SSE comments
                if line.starts_with(':') {
                    continue;
                }

                // Skip event type lines (e.g. "event: content_block_delta")
                if line.starts_with("event:") {
                    continue;
                }

                // Extract data payload
                let data = if let Some(payload) = line.strip_prefix("data: ") {
                    payload
                } else if let Some(payload) = line.strip_prefix("data:") {
                    payload
                } else {
                    continue;
                };

                // Check for done sentinel
                if let Some(ref sentinel) = done_sentinel {
                    if data.trim() == sentinel.as_str() {
                        return;
                    }
                }

                // Skip empty data payloads
                if data.trim().is_empty() {
                    continue;
                }

                // Parse the chunk via the protocol handler
                match parse_chunk(data) {
                    Ok(Some(token)) => {
                        yield token;
                    }
                    Ok(None) => {
                        // Skippable event (e.g. role-only delta)
                    }
                    Err(e) => {
                        tracing::debug!("SSE parse error (skipping): {e}");
                    }
                }
            }
        }

        // Process any remaining data in the buffer
        if !buffer.trim().is_empty() {
            for line in buffer.lines() {
                let line = line.trim();
                if let Some(data) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) {
                    if let Some(ref sentinel) = done_sentinel {
                        if data.trim() == sentinel.as_str() {
                            return;
                        }
                    }
                    if !data.trim().is_empty() {
                        if let Ok(Some(token)) = parse_chunk(data) {
                            yield token;
                        }
                    }
                }
            }
        }
    };

    Box::pin(stream)
}

/// Unit-testable SSE line parser (no network dependency).
/// Parses raw SSE text and returns data payloads.
#[cfg(test)]
fn parse_sse_lines(raw: &str, done_sentinel: Option<&str>) -> Vec<String> {
    let mut payloads = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') || line.starts_with("event:") {
            continue;
        }
        let data = if let Some(payload) = line.strip_prefix("data: ") {
            payload
        } else if let Some(payload) = line.strip_prefix("data:") {
            payload
        } else {
            continue;
        };
        if let Some(sentinel) = done_sentinel {
            if data.trim() == sentinel {
                break;
            }
        }
        if !data.trim().is_empty() {
            payloads.push(data.to_string());
        }
    }
    payloads
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sse_lines_basic() {
        let raw = "data: {\"content\":\"Hello\"}\n\n\
                   data: {\"content\":\" world\"}\n\n\
                   data: [DONE]\n\n";
        let payloads = parse_sse_lines(raw, Some("[DONE]"));
        assert_eq!(
            payloads,
            vec!["{\"content\":\"Hello\"}", "{\"content\":\" world\"}"]
        );
    }

    #[test]
    fn test_parse_sse_lines_skips_comments() {
        let raw = ": keep-alive\n\
                   data: {\"text\":\"hi\"}\n\n\
                   data: [DONE]\n";
        let payloads = parse_sse_lines(raw, Some("[DONE]"));
        assert_eq!(payloads, vec!["{\"text\":\"hi\"}"]);
    }

    #[test]
    fn test_parse_sse_lines_skips_events() {
        let raw = "event: message_start\n\
                   data: {\"type\":\"message_start\"}\n\n\
                   event: content_block_delta\n\
                   data: {\"text\":\"OK\"}\n\n";
        let payloads = parse_sse_lines(raw, None);
        assert_eq!(
            payloads,
            vec!["{\"type\":\"message_start\"}", "{\"text\":\"OK\"}"]
        );
    }

    #[test]
    fn test_parse_sse_lines_no_sentinel() {
        let raw = "data: chunk1\n\ndata: chunk2\n\n";
        let payloads = parse_sse_lines(raw, None);
        assert_eq!(payloads, vec!["chunk1", "chunk2"]);
    }

    #[test]
    fn test_parse_sse_lines_empty() {
        let payloads = parse_sse_lines("", Some("[DONE]"));
        assert!(payloads.is_empty());
    }

    #[test]
    fn test_parse_sse_lines_data_no_space() {
        let raw = "data:{\"compact\":true}\n\n";
        let payloads = parse_sse_lines(raw, None);
        assert_eq!(payloads, vec!["{\"compact\":true}"]);
    }
}
