use crate::types::{ChatMessage, ToolDefinition};

/// Known chat template formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateFormat {
    Llama3,
    ChatML,
    Gemma,
    Phi3,
    Mistral,
    /// Fallback generic format.
    Generic,
}

/// Detect the template format from a model name or filename.
pub fn detect_template(model_name: &str) -> TemplateFormat {
    let name = model_name.to_lowercase();

    if name.contains("llama-3") || name.contains("llama3") || name.contains("llama-4") {
        TemplateFormat::Llama3
    } else if name.contains("gemma") {
        TemplateFormat::Gemma
    } else if name.contains("phi-3")
        || name.contains("phi3")
        || name.contains("phi-4")
        || name.contains("phi4")
    {
        TemplateFormat::Phi3
    } else if name.contains("mistral") || name.contains("mixtral") {
        // Mistral v0.2+ and Mixtral use [INST] format
        TemplateFormat::Mistral
    } else if name.contains("qwen")
        || name.contains("deepseek")
        || name.contains("yi-")
        || name.contains("internlm")
        || name.contains("smollm")
    {
        // Models that use ChatML (<|im_start|>/<|im_end|>)
        TemplateFormat::ChatML
    } else {
        TemplateFormat::ChatML // ChatML is the most common default
    }
}

/// Render chat messages into a prompt string using the specified template.
pub fn render(
    format: TemplateFormat,
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
) -> String {
    match format {
        TemplateFormat::Llama3 => render_llama3(messages, tools),
        TemplateFormat::ChatML => render_chatml(messages, tools),
        TemplateFormat::Gemma => render_gemma(messages, tools),
        TemplateFormat::Phi3 => render_phi3(messages, tools),
        TemplateFormat::Mistral => render_mistral(messages, tools),
        TemplateFormat::Generic => render_generic(messages, tools),
    }
}

fn tool_system_prompt(tools: &[ToolDefinition]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let tool_descs: Vec<String> = tools
        .iter()
        .map(|t| {
            format!(
                "- {}: {} (params: {})",
                t.name, t.description, t.input_schema
            )
        })
        .collect();

    format!(
        "\n\nYou have access to the following tools:\n{}\n\
         To use a tool, respond with a JSON object: \
         {{\"tool\": \"<name>\", \"arguments\": {{...}}}}",
        tool_descs.join("\n")
    )
}

// ── Llama 3 ───────────────────────────────────────────────────────────

fn render_llama3(messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");

    for msg in messages {
        match msg {
            ChatMessage::System(text) => {
                let tool_prompt = tool_system_prompt(tools);
                prompt.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{text}{tool_prompt}<|eot_id|>"
                ));
            }
            ChatMessage::User(_parts) => {
                let text = msg.user_text_content().unwrap_or_default();
                prompt.push_str(&format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
                ));
            }
            ChatMessage::Assistant { content, .. } => {
                let content_str = content.as_deref().unwrap_or("");
                prompt.push_str(&format!(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n{content_str}<|eot_id|>"
                ));
            }
            ChatMessage::ToolResult { content, .. } => {
                prompt.push_str(&format!(
                    "<|start_header_id|>tool<|end_header_id|>\n\n{content}<|eot_id|>"
                ));
            }
        }
    }

    // If no system message was present, inject tool definitions
    if !tools.is_empty() && !messages.iter().any(|m| matches!(m, ChatMessage::System(_))) {
        let tool_prompt = tool_system_prompt(tools);
        prompt = format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
             You are a helpful assistant.{tool_prompt}<|eot_id|>{}",
            &prompt["<|begin_of_text|>".len()..]
        );
    }

    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

// ── ChatML (Qwen, Mistral) ───────────────────────────────────────────

fn render_chatml(messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        match msg {
            ChatMessage::System(text) => {
                let tool_prompt = tool_system_prompt(tools);
                prompt.push_str(&format!(
                    "<|im_start|>system\n{text}{tool_prompt}<|im_end|>\n"
                ));
            }
            ChatMessage::User(_parts) => {
                let text = msg.user_text_content().unwrap_or_default();
                prompt.push_str(&format!("<|im_start|>user\n{text}<|im_end|>\n"));
            }
            ChatMessage::Assistant { content, .. } => {
                let content_str = content.as_deref().unwrap_or("");
                prompt.push_str(&format!("<|im_start|>assistant\n{content_str}<|im_end|>\n"));
            }
            ChatMessage::ToolResult { content, .. } => {
                prompt.push_str(&format!("<|im_start|>tool\n{content}<|im_end|>\n"));
            }
        }
    }

    if !tools.is_empty() && !messages.iter().any(|m| matches!(m, ChatMessage::System(_))) {
        let tool_prompt = tool_system_prompt(tools);
        prompt = format!(
            "<|im_start|>system\nYou are a helpful assistant.{tool_prompt}<|im_end|>\n{prompt}"
        );
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

// ── Gemma ─────────────────────────────────────────────────────────────

fn render_gemma(messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
    let mut prompt = String::new();

    // Gemma uses <start_of_turn> / <end_of_turn>
    for msg in messages {
        match msg {
            ChatMessage::System(text) => {
                let tool_prompt = tool_system_prompt(tools);
                prompt.push_str(&format!(
                    "<start_of_turn>user\n[System: {text}{tool_prompt}]<end_of_turn>\n"
                ));
            }
            ChatMessage::User(_parts) => {
                let text = msg.user_text_content().unwrap_or_default();
                prompt.push_str(&format!("<start_of_turn>user\n{text}<end_of_turn>\n"));
            }
            ChatMessage::Assistant { content, .. } => {
                let content_str = content.as_deref().unwrap_or("");
                prompt.push_str(&format!("<start_of_turn>model\n{content_str}<end_of_turn>\n"));
            }
            ChatMessage::ToolResult { content, .. } => {
                prompt.push_str(&format!(
                    "<start_of_turn>user\n[Tool result: {content}]<end_of_turn>\n"
                ));
            }
        }
    }

    prompt.push_str("<start_of_turn>model\n");
    prompt
}

// ── Phi-3 ─────────────────────────────────────────────────────────────

fn render_phi3(messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        match msg {
            ChatMessage::System(text) => {
                let tool_prompt = tool_system_prompt(tools);
                prompt.push_str(&format!("<|system|>\n{text}{tool_prompt}<|end|>\n"));
            }
            ChatMessage::User(_parts) => {
                let text = msg.user_text_content().unwrap_or_default();
                prompt.push_str(&format!("<|user|>\n{text}<|end|>\n"));
            }
            ChatMessage::Assistant { content, .. } => {
                let content_str = content.as_deref().unwrap_or("");
                prompt.push_str(&format!("<|assistant|>\n{content_str}<|end|>\n"));
            }
            ChatMessage::ToolResult { content, .. } => {
                prompt.push_str(&format!("<|user|>\n[Tool result: {content}]<|end|>\n"));
            }
        }
    }

    prompt.push_str("<|assistant|>\n");
    prompt
}

// ── Mistral ([INST] format) ──────────────────────────────────────────

fn render_mistral(messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
    let mut prompt = String::from("<s>");

    // Collect system text to prepend to first user message
    let system_text: Option<String> = messages.iter().find_map(|m| match m {
        ChatMessage::System(text) => {
            let tool_prompt = tool_system_prompt(tools);
            Some(format!("{text}{tool_prompt}"))
        }
        _ => None,
    });

    let mut first_user = true;
    for msg in messages {
        match msg {
            ChatMessage::System(_) => {} // handled above
            ChatMessage::User(_parts) => {
                let text = msg.user_text_content().unwrap_or_default();
                if first_user {
                    if let Some(ref sys) = system_text {
                        prompt.push_str(&format!("[INST] {sys}\n\n{text} [/INST]"));
                    } else {
                        prompt.push_str(&format!("[INST] {text} [/INST]"));
                    }
                    first_user = false;
                } else {
                    prompt.push_str(&format!("[INST] {text} [/INST]"));
                }
            }
            ChatMessage::Assistant { content, .. } => {
                let content_str = content.as_deref().unwrap_or("");
                prompt.push_str(&format!(" {content_str}</s>"));
            }
            ChatMessage::ToolResult { content, .. } => {
                prompt.push_str(&format!("[INST] [Tool result: {content}] [/INST]"));
            }
        }
    }

    // If the last message was a user message, the model should respond
    // (the [/INST] already signals this)
    prompt
}

// ── Generic fallback ──────────────────────────────────────────────────

fn render_generic(messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        match msg {
            ChatMessage::System(text) => {
                let tool_prompt = tool_system_prompt(tools);
                prompt.push_str(&format!("<|system|>\n{text}{tool_prompt}\n"));
            }
            ChatMessage::User(_parts) => {
                let text = msg.user_text_content().unwrap_or_default();
                prompt.push_str(&format!("<|user|>\n{text}\n"));
            }
            ChatMessage::Assistant { content, .. } => {
                let content_str = content.as_deref().unwrap_or("");
                prompt.push_str(&format!("<|assistant|>\n{content_str}\n"));
            }
            ChatMessage::ToolResult { content, .. } => {
                prompt.push_str(&format!("<|tool|>\n{content}\n"));
            }
        }
    }

    prompt.push_str("<|assistant|>\n");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_template_llama3() {
        assert_eq!(detect_template("Llama-3.1-8B"), TemplateFormat::Llama3);
        assert_eq!(detect_template("llama3-70b"), TemplateFormat::Llama3);
    }

    #[test]
    fn test_detect_template_chatml() {
        assert_eq!(detect_template("Qwen2.5-7B"), TemplateFormat::ChatML);
        assert_eq!(detect_template("deepseek-v2"), TemplateFormat::ChatML);
    }

    #[test]
    fn test_detect_template_mistral() {
        assert_eq!(detect_template("Mistral-7B"), TemplateFormat::Mistral);
        assert_eq!(detect_template("mixtral-8x7b"), TemplateFormat::Mistral);
    }

    #[test]
    fn test_render_mistral() {
        let messages = vec![
            ChatMessage::System("Be helpful.".into()),
            ChatMessage::user_text("Hi"),
        ];
        let prompt = render(TemplateFormat::Mistral, &messages, &[]);
        assert!(prompt.contains("<s>"));
        assert!(prompt.contains("[INST]"));
        assert!(prompt.contains("Be helpful."));
        assert!(prompt.contains("Hi"));
        assert!(prompt.contains("[/INST]"));
    }

    #[test]
    fn test_detect_template_gemma() {
        assert_eq!(detect_template("gemma-2-9b"), TemplateFormat::Gemma);
    }

    #[test]
    fn test_detect_template_phi3() {
        assert_eq!(detect_template("Phi-3-mini"), TemplateFormat::Phi3);
    }

    #[test]
    fn test_render_llama3() {
        let messages = vec![
            ChatMessage::System("Be helpful.".into()),
            ChatMessage::user_text("Hi".into()),
        ];
        let prompt = render(TemplateFormat::Llama3, &messages, &[]);
        assert!(prompt.contains("<|begin_of_text|>"));
        assert!(prompt.contains("Be helpful."));
        assert!(prompt.contains("Hi"));
        assert!(prompt.ends_with("assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_render_chatml() {
        let messages = vec![ChatMessage::user_text("Hello".into())];
        let prompt = render(TemplateFormat::ChatML, &messages, &[]);
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("Hello"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_render_with_tools() {
        let messages = vec![ChatMessage::user_text("Search for cats".into())];
        let tools = vec![ToolDefinition {
            name: "search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        }];
        let prompt = render(TemplateFormat::ChatML, &messages, &tools);
        assert!(prompt.contains("search"));
        assert!(prompt.contains("Search the web"));
    }
}
