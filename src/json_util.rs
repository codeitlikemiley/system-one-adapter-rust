//! JSON helpers: sorted encoding, prompt escaping, and fence stripping.

use serde_json::Value;

/// Compact JSON with object keys sorted at every level.
pub fn compact_sorted_json(value: &Value) -> String {
    serde_json::to_string(&sort_value(value)).expect("json value is always serializable")
}

pub fn sort_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for key in keys {
                out.insert(key.clone(), sort_value(&map[key]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_value).collect()),
        other => other.clone(),
    }
}

/// Serialize evaluation state as the user document payload.
pub fn serialize_state_as_user_prompt(state: &Value) -> String {
    let mut serialized = compact_sorted_json(state);
    serialized = serialized.replace('<', "\\u003c").replace('>', "\\u003e");
    format!("<document>\n{serialized}\n</document>")
}

/// Strip Markdown code fences a prompted model may wrap around the JSON object.
pub fn extract_json(text: &str) -> String {
    let mut text = text.trim().to_string();
    if text.starts_with("```") {
        text = text[3..].to_string();
        if text.len() >= 4 && text[..4].eq_ignore_ascii_case("json") {
            text = text[4..].to_string();
        }
        text = text.trim().to_string();
        if text.ends_with("```") {
            let end = text.len() - 3;
            text = text[..end].trim().to_string();
        }
    }
    text
}

pub fn serialize_instruction_value(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "No additional instructions.".into(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => compact_sorted_json(other),
    }
}

pub fn looks_truncated_parse_error(error: &serde_json::Error) -> bool {
    let message = error.to_string();
    message.contains("EOF")
        || message.contains("eof")
        || message.contains("end of input")
        || message.contains("end of file")
        || message.contains("trailing")
}
