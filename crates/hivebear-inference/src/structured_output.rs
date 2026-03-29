/// Convert a JSON Schema into a GBNF grammar string.
///
/// Supports a subset of JSON Schema: object, array, string, number, integer,
/// boolean, enum, and const. Nested objects and arrays are supported.
///
/// The generated grammar forces the model output to conform to the schema.
pub fn json_schema_to_gbnf(schema: &serde_json::Value) -> Result<String, String> {
    let mut rules = Vec::new();
    rules.push("ws ::= [ \\t\\n\\r]*".to_string());

    let root_rule = generate_rule(schema, "root", &mut rules)?;
    rules.insert(0, root_rule);

    Ok(rules.join("\n"))
}

fn generate_rule(
    schema: &serde_json::Value,
    name: &str,
    rules: &mut Vec<String>,
) -> Result<String, String> {
    let schema_type = schema
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("string");

    // Handle enum
    if let Some(enum_values) = schema.get("enum") {
        if let Some(arr) = enum_values.as_array() {
            let alternatives: Vec<String> = arr
                .iter()
                .map(|v| {
                    if let Some(s) = v.as_str() {
                        format!("\"\\\"{}\\\"\"", s)
                    } else {
                        format!("\"{}\"", v)
                    }
                })
                .collect();
            return Ok(format!("{name} ::= {}", alternatives.join(" | ")));
        }
    }

    // Handle const
    if let Some(const_val) = schema.get("const") {
        if let Some(s) = const_val.as_str() {
            return Ok(format!("{name} ::= \"\\\"{}\\\"\"", s));
        }
        return Ok(format!("{name} ::= \"{}\"", const_val));
    }

    match schema_type {
        "object" => generate_object_rule(schema, name, rules),
        "array" => generate_array_rule(schema, name, rules),
        "string" => Ok(format!(
            "{name} ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" .)* \"\\\"\""
        )),
        "number" => Ok(format!(
            "{name} ::= \"-\"? [0-9]+ (\".\" [0-9]+)? ((\"e\" | \"E\") (\"+\" | \"-\")? [0-9]+)?"
        )),
        "integer" => Ok(format!("{name} ::= \"-\"? [0-9]+")),
        "boolean" => Ok(format!("{name} ::= \"true\" | \"false\"")),
        "null" => Ok(format!("{name} ::= \"null\"")),
        other => Err(format!("Unsupported JSON Schema type: {other}")),
    }
}

fn generate_object_rule(
    schema: &serde_json::Value,
    name: &str,
    rules: &mut Vec<String>,
) -> Result<String, String> {
    let properties = match schema.get("properties") {
        Some(props) => props.as_object().ok_or("properties must be an object")?,
        None => {
            // Any object
            return Ok(format!(
                "{name} ::= \"{{\" ws (pair (ws \",\" ws pair)*)? ws \"}}\"
pair ::= string ws \":\" ws value
value ::= string | number | \"true\" | \"false\" | \"null\"
string ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" .)* \"\\\"\"
number ::= \"-\"? [0-9]+ (\".\" [0-9]+)?"
            ));
        }
    };

    let required: Vec<&str> = schema
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let mut parts = Vec::new();
    for (prop_name, prop_schema) in properties {
        let rule_name = format!("{name}-{prop_name}");
        let rule = generate_rule(prop_schema, &rule_name, rules)?;
        rules.push(rule);

        let is_required = required.contains(&prop_name.as_str());
        let pair = format!("\"\\\"{}\\\"\" ws \":\" ws {}", prop_name, rule_name);

        if is_required {
            parts.push(pair);
        } else {
            parts.push(format!("({pair})?"));
        }
    }

    let body = if parts.is_empty() {
        String::new()
    } else {
        parts.join(" ws \",\" ws ")
    };

    Ok(format!("{name} ::= \"{{\" ws {body} ws \"}}\""))
}

fn generate_array_rule(
    schema: &serde_json::Value,
    name: &str,
    rules: &mut Vec<String>,
) -> Result<String, String> {
    let items_rule = if let Some(items) = schema.get("items") {
        let item_name = format!("{name}-item");
        let rule = generate_rule(items, &item_name, rules)?;
        rules.push(rule);
        item_name
    } else {
        // Default: any value
        let item_name = format!("{name}-item");
        rules.push(format!(
            "{item_name} ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" .)* \"\\\"\" | \"-\"? [0-9]+ | \"true\" | \"false\" | \"null\""
        ));
        item_name
    };

    Ok(format!(
        "{name} ::= \"[\" ws ({items_rule} (\",\" ws {items_rule})*)? ws \"]\""
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simple_string_schema() {
        let schema = json!({"type": "string"});
        let grammar = json_schema_to_gbnf(&schema).unwrap();
        assert!(grammar.contains("root"));
        assert!(grammar.contains("\\\""));
    }

    #[test]
    fn test_object_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        });
        let grammar = json_schema_to_gbnf(&schema).unwrap();
        assert!(grammar.contains("root"));
        assert!(grammar.contains("root-name"));
        assert!(grammar.contains("root-age"));
    }

    #[test]
    fn test_array_schema() {
        let schema = json!({
            "type": "array",
            "items": {"type": "string"}
        });
        let grammar = json_schema_to_gbnf(&schema).unwrap();
        assert!(grammar.contains("["));
        assert!(grammar.contains("root-item"));
    }

    #[test]
    fn test_enum_schema() {
        let schema = json!({
            "type": "string",
            "enum": ["red", "green", "blue"]
        });
        let grammar = json_schema_to_gbnf(&schema).unwrap();
        assert!(grammar.contains("red"));
        assert!(grammar.contains("green"));
        assert!(grammar.contains("blue"));
    }

    #[test]
    fn test_boolean_schema() {
        let schema = json!({"type": "boolean"});
        let grammar = json_schema_to_gbnf(&schema).unwrap();
        assert!(grammar.contains("true"));
        assert!(grammar.contains("false"));
    }

    #[test]
    fn test_unsupported_type() {
        let schema = json!({"type": "banana"});
        assert!(json_schema_to_gbnf(&schema).is_err());
    }
}
