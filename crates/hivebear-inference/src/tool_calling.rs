use crate::types::{ToolCallResponse, ToolDefinition};

/// Generate a GBNF grammar that forces the model to output either plain text
/// or a valid JSON tool call matching one of the provided tool definitions.
pub fn generate_tool_grammar(tools: &[ToolDefinition]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let tool_names: Vec<String> = tools
        .iter()
        .map(|t| format!("\"\\\"{}\\\"\"", t.name))
        .collect();

    // Grammar that allows either free text or a JSON tool call
    format!(
        r#"root ::= tool-call | free-text
free-text ::= [^\x00]{{1,4096}}
tool-call ::= "{{" ws "\"tool\"" ws ":" ws tool-name ws "," ws "\"arguments\"" ws ":" ws object ws "}}"
tool-name ::= {tool_names}
object ::= "{{" ws (pair ("," ws pair)*)? ws "}}"
pair ::= string ws ":" ws value
value ::= string | number | "true" | "false" | "null" | object | array
array ::= "[" ws (value ("," ws value)*)? ws "]"
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)? (("e" | "E") ("+" | "-")? [0-9]+)?
ws ::= [ \t\n\r]*
"#,
        tool_names = tool_names.join(" | ")
    )
}

/// Try to parse a model's output as a tool call.
/// Returns `Some(ToolCallResponse)` if the output looks like a tool call,
/// `None` if it's plain text.
pub fn parse_tool_call(output: &str) -> Option<ToolCallResponse> {
    let trimmed = output.trim();

    // Try to parse as JSON
    let value: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    let obj = value.as_object()?;

    let tool_name = obj
        .get("tool")
        .or_else(|| obj.get("name"))
        .or_else(|| obj.get("function"))
        .and_then(|v| v.as_str())?
        .to_string();

    let arguments = obj
        .get("arguments")
        .or_else(|| obj.get("parameters"))
        .or_else(|| obj.get("args"))
        .cloned()
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

    Some(ToolCallResponse {
        tool_name,
        arguments,
        call_id: uuid::Uuid::new_v4().to_string(),
    })
}

/// Model families and their tool calling reliability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallReliability {
    /// Excellent — model was trained for tool calling.
    High,
    /// Good with grammar constraints.
    Medium,
    /// Unreliable — may not follow tool call format.
    Low,
    /// Unknown / untested.
    Unknown,
}

/// Get the tool calling reliability rating for a model family.
pub fn model_reliability(model_name: &str) -> ToolCallReliability {
    let name = model_name.to_lowercase();

    // Models with native tool calling training
    if name.contains("llama-3.1") || name.contains("llama-3.2") {
        return ToolCallReliability::High;
    }
    if name.contains("qwen") {
        return ToolCallReliability::High;
    }
    if name.contains("mistral") && name.contains("instruct") {
        return ToolCallReliability::High;
    }

    // Models that work reasonably with grammar constraints
    if name.contains("phi-3") || name.contains("gemma") {
        return ToolCallReliability::Medium;
    }

    // Older or code-focused models
    if name.contains("starcoder") || name.contains("deepseek-coder") {
        return ToolCallReliability::Low;
    }

    ToolCallReliability::Unknown
}

/// Generate a warning message if the model has poor tool calling support.
pub fn tool_calling_warning(model_name: &str) -> Option<String> {
    match model_reliability(model_name) {
        ToolCallReliability::Low => Some(format!(
            "Warning: {model_name} has poor tool calling support. \
             Results may be unreliable."
        )),
        ToolCallReliability::Unknown => Some(format!(
            "Note: Tool calling support for {model_name} is untested. \
             Grammar constraints will be used to force valid output."
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_grammar_empty() {
        assert_eq!(generate_tool_grammar(&[]), "");
    }

    #[test]
    fn test_generate_grammar_has_tools() {
        let tools = vec![ToolDefinition {
            name: "search".into(),
            description: "Search".into(),
            input_schema: serde_json::json!({}),
        }];
        let grammar = generate_tool_grammar(&tools);
        assert!(grammar.contains("root"));
        assert!(grammar.contains("tool-call"));
        assert!(grammar.contains("search"));
    }

    #[test]
    fn test_parse_tool_call_valid() {
        let output = r#"{"tool": "search", "arguments": {"query": "cats"}}"#;
        let result = parse_tool_call(output);
        assert!(result.is_some());
        let tc = result.unwrap();
        assert_eq!(tc.tool_name, "search");
        assert_eq!(tc.arguments["query"], "cats");
    }

    #[test]
    fn test_parse_tool_call_plain_text() {
        let output = "The capital of France is Paris.";
        assert!(parse_tool_call(output).is_none());
    }

    #[test]
    fn test_parse_tool_call_alternative_format() {
        let output = r#"{"name": "get_weather", "parameters": {"city": "NYC"}}"#;
        let result = parse_tool_call(output);
        assert!(result.is_some());
        let tc = result.unwrap();
        assert_eq!(tc.tool_name, "get_weather");
    }

    #[test]
    fn test_model_reliability() {
        assert_eq!(
            model_reliability("Llama-3.1-8B-Instruct"),
            ToolCallReliability::High
        );
        assert_eq!(model_reliability("Qwen2.5-7B"), ToolCallReliability::High);
        assert_eq!(model_reliability("starcoder2-3b"), ToolCallReliability::Low);
        assert_eq!(
            model_reliability("some-unknown-model"),
            ToolCallReliability::Unknown
        );
    }

    #[test]
    fn test_tool_calling_warning() {
        assert!(tool_calling_warning("starcoder2-3b").is_some());
        assert!(tool_calling_warning("some-unknown-model").is_some());
        assert!(tool_calling_warning("Llama-3.1-8B-Instruct").is_none());
    }
}
