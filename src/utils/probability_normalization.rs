//! Probability distribution normalization.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::types::AnswerMode;

pub const PROBABILITY_TOLERANCE: f64 = 1e-6;

/// Result of processing one probability distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbabilityNormalization {
    pub probabilities: BTreeMap<String, f64>,
    pub error: f64,
    pub original_probabilities: Option<BTreeMap<String, f64>>,
}

impl ProbabilityNormalization {
    pub fn new(probabilities: BTreeMap<String, f64>) -> Self {
        Self {
            probabilities,
            error: 0.0,
            original_probabilities: None,
        }
    }
}

/// Build probability diagnostics from question normalization results.
pub fn probability_debug_data(
    probability_normalizations: &BTreeMap<String, Option<ProbabilityNormalization>>,
) -> Value {
    let mut errors = BTreeMap::new();
    for (question_id, result) in probability_normalizations {
        if let Some(result) = result {
            errors.insert(question_id.clone(), result.error);
        }
    }
    let probability_errors: BTreeMap<String, f64> = errors
        .iter()
        .filter(|(_, error)| **error > PROBABILITY_TOLERANCE)
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let mut original_probabilities = BTreeMap::new();
    for (question_id, result) in probability_normalizations {
        if let Some(result) = result {
            if let Some(originals) = &result.original_probabilities {
                original_probabilities.insert(question_id.clone(), originals.clone());
            }
        }
    }
    let max_error = errors.values().copied().fold(0.0_f64, f64::max);
    let mut debug = serde_json::Map::new();
    debug.insert("max_error".into(), json_number(max_error));
    debug.insert(
        "invalid_probs".into(),
        Value::Number(probability_errors.len().into()),
    );
    debug.insert(
        "probability_errors".into(),
        serde_json::to_value(&probability_errors).unwrap_or(Value::Object(Default::default())),
    );
    if !original_probabilities.is_empty() {
        debug.insert(
            "original_probabilities".into(),
            serde_json::to_value(&original_probabilities).unwrap_or(Value::Null),
        );
    }
    Value::Object(debug)
}

fn json_number(value: f64) -> Value {
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// Rescale probabilities to sum to 1, falling back to uniform for a zero total.
pub fn rescale_probabilities(probabilities: &BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    let total: f64 = probabilities.values().sum();
    if total == 0.0 {
        let uniform = 1.0 / probabilities.len() as f64;
        return probabilities
            .keys()
            .map(|answer| (answer.clone(), uniform))
            .collect();
    }
    probabilities
        .iter()
        .map(|(answer, probability)| (answer.clone(), probability / total))
        .collect()
}

/// Build and optionally normalize a probability distribution.
pub fn normalize_probabilities_of_all_answers(
    answers: &[String],
    value: &Value,
    answer_mode: AnswerMode,
    enabled: bool,
) -> ProbabilityNormalization {
    if answer_mode == AnswerMode::Discrete {
        let selected = match value {
            Value::String(s) => s.clone(),
            other => match other {
                Value::Bool(true) => "true".into(),
                Value::Bool(false) => "false".into(),
                Value::Number(n) => n.to_string(),
                _ => other.to_string(),
            },
        };
        let probabilities = answers
            .iter()
            .map(|answer| (answer.clone(), if *answer == selected { 1.0 } else { 0.0 }))
            .collect();
        return ProbabilityNormalization::new(probabilities);
    }

    let original_probabilities: BTreeMap<String, f64> = answers
        .iter()
        .map(|answer| {
            let probability = value
                .get(answer)
                .and_then(|v| v.as_f64())
                .or_else(|| value.get(answer).and_then(|v| v.as_i64()).map(|n| n as f64))
                .unwrap_or(0.0);
            (answer.clone(), probability)
        })
        .collect();
    let total: f64 = original_probabilities.values().sum();
    let error = (total - 1.0).abs();
    if !enabled || error <= PROBABILITY_TOLERANCE {
        return ProbabilityNormalization {
            probabilities: original_probabilities,
            error,
            original_probabilities: None,
        };
    }
    let probabilities = rescale_probabilities(&original_probabilities);
    ProbabilityNormalization {
        probabilities,
        error,
        original_probabilities: Some(original_probabilities),
    }
}
