use std::collections::BTreeMap;

use approx::assert_abs_diff_eq;
use serde_json::json;
use system_one_adapter::types::AnswerMode;
use system_one_adapter::utils::probability_normalization::{
    normalize_probabilities_of_all_answers, probability_debug_data,
};

fn approx_map(actual: &BTreeMap<String, f64>, expected: &BTreeMap<String, f64>) {
    assert_eq!(actual.len(), expected.len());
    for (key, value) in expected {
        assert_abs_diff_eq!(actual[key], *value, epsilon = 1e-6);
    }
}

#[test]
fn probability_normalization_disabled_keeps_raw_values() {
    run_case(
        false,
        0.2,
        0.2,
        None,
        0.6,
        &[("stars", 0.6), ("genre", 0.6)],
    );
}

#[test]
fn probability_normalization_enabled_rescales_and_keeps_originals() {
    let mut originals = BTreeMap::new();
    originals.insert(
        "stars".to_string(),
        BTreeMap::from([("0".to_string(), 0.2), ("1".to_string(), 0.2)]),
    );
    originals.insert(
        "genre".to_string(),
        BTreeMap::from([
            ("fiction".to_string(), 0.2),
            ("nonfiction".to_string(), 0.2),
        ]),
    );
    run_case(
        true,
        0.2,
        0.5,
        Some(originals),
        0.6,
        &[("stars", 0.6), ("genre", 0.6)],
    );
}

#[test]
fn probability_normalization_within_tolerance_is_not_invalid() {
    run_case(false, 0.50000025, 0.50000025, None, 5e-7, &[]);
}

fn run_case(
    enabled: bool,
    raw_probability: f64,
    expected_probability: f64,
    expected_originals: Option<BTreeMap<String, BTreeMap<String, f64>>>,
    expected_max_error: f64,
    expected_probability_errors: &[(&str, f64)],
) {
    let score_probabilities = json!({ "0": raw_probability, "1": raw_probability });
    let choice_probabilities = json!({
        "fiction": raw_probability,
        "nonfiction": raw_probability
    });
    let mut probability_normalizations = BTreeMap::new();
    probability_normalizations.insert("positive".to_string(), None);
    probability_normalizations.insert(
        "stars".to_string(),
        Some(normalize_probabilities_of_all_answers(
            &["0".to_string(), "1".to_string()],
            &score_probabilities,
            AnswerMode::Probabilities,
            enabled,
        )),
    );
    probability_normalizations.insert(
        "genre".to_string(),
        Some(normalize_probabilities_of_all_answers(
            &["fiction".to_string(), "nonfiction".to_string()],
            &choice_probabilities,
            AnswerMode::Probabilities,
            enabled,
        )),
    );

    approx_map(
        &probability_normalizations["stars"]
            .as_ref()
            .unwrap()
            .probabilities,
        &BTreeMap::from([
            ("0".to_string(), expected_probability),
            ("1".to_string(), expected_probability),
        ]),
    );
    approx_map(
        &probability_normalizations["genre"]
            .as_ref()
            .unwrap()
            .probabilities,
        &BTreeMap::from([
            ("fiction".to_string(), expected_probability),
            ("nonfiction".to_string(), expected_probability),
        ]),
    );

    let debug_data = probability_debug_data(&probability_normalizations);
    assert_abs_diff_eq!(
        debug_data["max_error"].as_f64().unwrap(),
        expected_max_error,
        epsilon = 1e-9
    );
    assert_eq!(
        debug_data["invalid_probs"].as_u64().unwrap() as usize,
        expected_probability_errors.len()
    );
    let errors = debug_data["probability_errors"].as_object().unwrap();
    assert_eq!(errors.len(), expected_probability_errors.len());
    for (key, value) in expected_probability_errors {
        assert_abs_diff_eq!(errors[*key].as_f64().unwrap(), *value, epsilon = 1e-9);
    }
    match expected_originals {
        None => assert!(debug_data.get("original_probabilities").is_none()),
        Some(expected) => {
            let actual = debug_data["original_probabilities"].clone();
            assert_eq!(actual, serde_json::to_value(expected).unwrap());
        }
    }
}
