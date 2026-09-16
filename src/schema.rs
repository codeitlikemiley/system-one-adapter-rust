//! Dynamic JSON Schema per request, keyword sanitization, and decode validation.

use serde_json::{json, Map, Value};

use crate::errors::DecodeError;
use crate::json_util::{looks_truncated_parse_error, serialize_instruction_value};
use crate::types::{AnswerMode, Question, QuestionCollection};

const ANSWERS_DESCRIPTION: &str = "Exactly one answer per property below. Use these property names verbatim and do not add, rename, or nest them under any other key.";

const UNSUPPORTED_SCHEMA_KEYWORDS: &[&str] = &[
    "title",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
];

/// Generated output contract for one evaluation.
#[derive(Debug, Clone)]
pub struct LlmOutputModel {
    /// Schema used to constrain and prompt the provider (sanitized).
    pub provider_schema: Value,
    /// Schema used to validate the decoded payload locally (includes bounds).
    pub validation_schema: Value,
}

impl LlmOutputModel {
    pub fn decode(&self, text: &str) -> Result<Value, DecodeError> {
        decode_llm_output(text, &self.validation_schema)
    }
}

/// Build the per-request output model for `questions` in `llm_answer_mode`.
pub fn create_llm_output_model(
    questions: &QuestionCollection,
    llm_answer_mode: AnswerMode,
) -> LlmOutputModel {
    let validation_schema = build_output_schema(questions, llm_answer_mode);
    let provider_schema = sanitize_provider_schema(&validation_schema);
    LlmOutputModel {
        provider_schema,
        validation_schema,
    }
}

/// Sanitized JSON Schema sent to the provider.
pub fn create_raw_output_schema(output_model: &LlmOutputModel) -> Value {
    output_model.provider_schema.clone()
}

/// Convenience: sanitized schema for `questions` and `llm_answer_mode`.
pub fn create_llm_output_schema(
    questions: &QuestionCollection,
    llm_answer_mode: AnswerMode,
) -> Value {
    create_raw_output_schema(&create_llm_output_model(questions, llm_answer_mode))
}

fn build_output_schema(questions: &QuestionCollection, llm_answer_mode: AnswerMode) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();
    let mut defs = Map::new();

    for (index, (question_id, question)) in questions.iter().enumerate() {
        let is_probability_map = llm_answer_mode == AnswerMode::Probabilities
            && matches!(question, Question::Choice(_) | Question::Score(_));
        let (property_schema, extra_def) =
            answer_schema_for_question(index, question, llm_answer_mode);
        if let Some((name, schema)) = extra_def {
            defs.insert(name, schema);
        }
        let property = if is_probability_map {
            property_schema
        } else {
            let mut object = match property_schema {
                Value::Object(map) => map,
                other => {
                    let mut map = Map::new();
                    map.insert("schema".into(), other);
                    map
                }
            };
            object.insert(
                "description".into(),
                Value::String(build_llm_output_field_description(
                    question,
                    llm_answer_mode,
                )),
            );
            Value::Object(object)
        };
        properties.insert(question_id.to_string(), property);
        required.push(Value::String(question_id.to_string()));
    }

    defs.insert(
        "TypeSafeAnswers".into(),
        json!({
            "description": ANSWERS_DESCRIPTION,
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false
        }),
    );

    json!({
        "type": "object",
        "properties": {
            "answers": { "$ref": "#/$defs/TypeSafeAnswers" }
        },
        "required": ["answers"],
        "additionalProperties": false,
        "$defs": defs
    })
}

fn answer_schema_for_question(
    index: usize,
    question: &Question,
    llm_answer_mode: AnswerMode,
) -> (Value, Option<(String, Value)>) {
    match question {
        Question::Noul(_) => {
            let schema = if llm_answer_mode == AnswerMode::Discrete {
                json!({ "type": "boolean" })
            } else {
                json!({ "type": "number", "minimum": 0, "maximum": 1 })
            };
            (schema, None)
        }
        Question::Score(score) => {
            if llm_answer_mode == AnswerMode::Discrete {
                return (
                    json!({
                        "type": "integer",
                        "minimum": 0,
                        "exclusiveMaximum": score.criteria.len()
                    }),
                    None,
                );
            }
            let mut properties = Map::new();
            let mut required = Vec::new();
            for (answer, criterion) in score.criteria.iter().enumerate() {
                let key = answer.to_string();
                properties.insert(
                    key.clone(),
                    json!({
                        "description": serialize_instruction_value(Some(criterion)),
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    }),
                );
                required.push(Value::String(key));
            }
            let name = format!("ProbabilityMap{index}");
            let def = json!({
                "description": build_llm_output_question_description(question, llm_answer_mode),
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": false
            });
            (
                json!({ "$ref": format!("#/$defs/{name}") }),
                Some((name, def)),
            )
        }
        Question::Choice(choice) => {
            if llm_answer_mode == AnswerMode::Discrete {
                let labels: Vec<Value> = choice
                    .criteria
                    .keys()
                    .map(|label| Value::String(label.clone()))
                    .collect();
                return (json!({ "enum": labels }), None);
            }
            let mut properties = Map::new();
            let mut required = Vec::new();
            for (answer, criterion) in &choice.criteria {
                properties.insert(
                    answer.clone(),
                    json!({
                        "description": serialize_instruction_value(criterion.as_ref()),
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    }),
                );
                required.push(Value::String(answer.clone()));
            }
            let name = format!("ProbabilityMap{index}");
            let def = json!({
                "description": build_llm_output_question_description(question, llm_answer_mode),
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": false
            });
            (
                json!({ "$ref": format!("#/$defs/{name}") }),
                Some((name, def)),
            )
        }
    }
}

fn build_llm_output_field_description(question: &Question, llm_answer_mode: AnswerMode) -> String {
    let description = build_llm_output_question_description(question, llm_answer_mode);
    match question {
        Question::Score(score) => {
            let levels = score
                .criteria
                .iter()
                .enumerate()
                .map(|(score_index, criterion)| {
                    format!(
                        "{score_index} = {}",
                        serialize_instruction_value(Some(criterion))
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            if llm_answer_mode == AnswerMode::Discrete {
                format!("{description}\nScore levels, answer with the integer:\n{levels}")
            } else {
                format!("{description}\nRequired probability keys:\n{levels}")
            }
        }
        Question::Choice(choice) => {
            let choices = choice
                .criteria
                .iter()
                .map(|(answer, criterion)| {
                    format!(
                        "{answer} = {}",
                        serialize_instruction_value(criterion.as_ref())
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            if llm_answer_mode == AnswerMode::Discrete {
                format!("{description}\nChoice labels, answer with one label:\n{choices}")
            } else {
                format!("{description}\nRequired probability keys:\n{choices}")
            }
        }
        Question::Noul(noul) => {
            let Some(criteria) = &noul.criteria else {
                return description;
            };
            let true_criteria = serialize_instruction_value(criteria.true_criteria.as_ref());
            let false_criteria = serialize_instruction_value(criteria.false_criteria.as_ref());
            format!(
                "{description}\nTrue criteria: {true_criteria}\nFalse criteria: {false_criteria}"
            )
        }
    }
}

fn build_llm_output_question_description(
    question: &Question,
    llm_answer_mode: AnswerMode,
) -> String {
    let description = serialize_instruction_value(question.instructions());
    match (question, llm_answer_mode) {
        (Question::Noul(_), AnswerMode::Probabilities) => format!(
            "Probability that the answer is yes or the assertion is true. 0 means no or false, 0.5 means uncertain, and 1 means yes or true.\nQuestion: {description}"
        ),
        (Question::Score(_), AnswerMode::Probabilities) => format!(
            "Each property maps a rubric level to the probability that the document matches it.\nQuestion: {description}"
        ),
        (Question::Choice(_), AnswerMode::Probabilities) => format!(
            "Each property maps an option to the probability that it is the best answer.\nQuestion: {description}"
        ),
        _ => description,
    }
}

fn sanitize_provider_schema(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sanitized = Map::new();
            for (key, item) in map {
                if UNSUPPORTED_SCHEMA_KEYWORDS.contains(&key.as_str()) {
                    continue;
                }
                if key == "properties" || key == "$defs" {
                    if let Value::Object(inner) = item {
                        let mut nested = Map::new();
                        for (name, schema) in inner {
                            nested.insert(name.clone(), sanitize_provider_schema(schema));
                        }
                        sanitized.insert(key.clone(), Value::Object(nested));
                    } else {
                        sanitized.insert(key.clone(), sanitize_provider_schema(item));
                    }
                } else {
                    sanitized.insert(key.clone(), sanitize_provider_schema(item));
                }
            }
            Value::Object(sanitized)
        }
        Value::Array(items) => Value::Array(items.iter().map(sanitize_provider_schema).collect()),
        other => other.clone(),
    }
}

/// Strip fences, parse JSON, and validate against the output schema.
pub fn decode_llm_output(text: &str, validation_schema: &Value) -> Result<Value, DecodeError> {
    let extracted = crate::json_util::extract_json(text);
    let parsed: Value = match serde_json::from_str(&extracted) {
        Ok(value) => value,
        Err(error) => {
            if looks_truncated_parse_error(&error) {
                return Err(DecodeError::new("Input data was truncated"));
            }
            return Err(DecodeError::new(format!(
                "JSON is malformed: invalid character ({error})"
            )));
        }
    };

    if let Err(error) = compile_and_validate(validation_schema, &parsed) {
        return Err(error);
    }
    if let Err(error) = structural_validate(&parsed, validation_schema, validation_schema, "$") {
        return Err(error);
    }
    Ok(parsed)
}

fn compile_and_validate(schema: &Value, instance: &Value) -> Result<(), DecodeError> {
    let compiled = jsonschema::JSONSchema::compile(schema)
        .map_err(|error| DecodeError::new(error.to_string()))?;
    if compiled.is_valid(instance) {
        Ok(())
    } else {
        // Prefer the structural walker for msgspec-compatible messages.
        Ok(())
    }
}

fn resolve<'a>(schema: &'a Value, root: &'a Value) -> &'a Value {
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        if let Some(name) = reference.strip_prefix("#/$defs/") {
            if let Some(resolved) = root.get("$defs").and_then(|defs| defs.get(name)) {
                return resolved;
            }
        }
    }
    schema
}

fn structural_validate(
    instance: &Value,
    schema: &Value,
    root: &Value,
    path: &str,
) -> Result<(), DecodeError> {
    let schema = resolve(schema, root);
    if let Some(enum_values) = schema.get("enum").and_then(Value::as_array) {
        if !enum_values.contains(instance) {
            return Err(DecodeError::new(format!(
                "Invalid enum value {instance} - at `{path}`"
            )));
        }
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("object") => validate_object(instance, schema, root, path)?,
        Some("array") => {
            if !instance.is_array() {
                return Err(DecodeError::new(format!("Expected `array` - at `{path}`")));
            }
        }
        Some("number") => validate_number(instance, schema, path, false)?,
        Some("integer") => validate_number(instance, schema, path, true)?,
        Some("boolean") => {
            if !instance.is_boolean() {
                return Err(DecodeError::new(format!("Expected `bool` - at `{path}`")));
            }
        }
        Some("string") => {
            if !instance.is_string() {
                return Err(DecodeError::new(format!("Expected `str` - at `{path}`")));
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_object(
    instance: &Value,
    schema: &Value,
    root: &Value,
    path: &str,
) -> Result<(), DecodeError> {
    let Some(object) = instance.as_object() else {
        return Err(DecodeError::new(format!("Expected `object` - at `{path}`")));
    };
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for field in required {
            if let Some(name) = field.as_str() {
                if !object.contains_key(name) {
                    return Err(DecodeError::new(format!(
                        "Object missing required field `{name}` - at `{path}`"
                    )));
                }
            }
        }
    }
    let additional = schema
        .get("additionalProperties")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let properties = schema.get("properties").and_then(Value::as_object);
    if !additional {
        if let Some(properties) = properties {
            for key in object.keys() {
                if !properties.contains_key(key) {
                    return Err(DecodeError::new(format!(
                        "Object contains unknown field `{key}` - at `{path}`"
                    )));
                }
            }
        }
    }
    if let Some(properties) = properties {
        for (key, property_schema) in properties {
            if let Some(value) = object.get(key) {
                let child = format!("{path}.{key}");
                structural_validate(value, property_schema, root, &child)?;
            }
        }
    }
    Ok(())
}

fn validate_number(
    instance: &Value,
    schema: &Value,
    path: &str,
    integer: bool,
) -> Result<(), DecodeError> {
    let Some(number) = instance.as_f64() else {
        let expected = if integer { "int" } else { "float" };
        return Err(DecodeError::new(format!(
            "Expected `{expected}` - at `{path}`"
        )));
    };
    if integer && instance.as_i64().is_none() && instance.as_u64().is_none() {
        return Err(DecodeError::new(format!("Expected `int` - at `{path}`")));
    }
    if let Some(minimum) = schema.get("minimum").and_then(Value::as_f64) {
        if number < minimum {
            return Err(DecodeError::new(format!(
                "Expected `float` >= {minimum} - at `{path}`"
            )));
        }
    }
    if let Some(maximum) = schema.get("maximum").and_then(Value::as_f64) {
        if number > maximum {
            return Err(DecodeError::new(format!(
                "Expected `float` <= {maximum} - at `{path}`"
            )));
        }
    }
    if let Some(exclusive_maximum) = schema.get("exclusiveMaximum").and_then(Value::as_f64) {
        if number >= exclusive_maximum {
            return Err(DecodeError::new(format!(
                "Expected `int` < {exclusive_maximum} - at `{path}`"
            )));
        }
    }
    if let Some(exclusive_minimum) = schema.get("exclusiveMinimum").and_then(Value::as_f64) {
        if number <= exclusive_minimum {
            return Err(DecodeError::new(format!(
                "Expected `int` > {exclusive_minimum} - at `{path}`"
            )));
        }
    }
    Ok(())
}
