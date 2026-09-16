use approx::assert_abs_diff_eq;
use system_one_adapter::utils::confidence_metrics::{choice_confidence, score_confidence};

#[test]
fn score_uniform_is_zero() {
    assert_abs_diff_eq!(score_confidence(&[0.2; 5]), 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(score_confidence(&[0.04; 5]), 0.0, epsilon = 1e-9);
}

#[test]
fn score_concentrated_matches_python() {
    assert_abs_diff_eq!(
        score_confidence(&[0.01, 0.02, 0.07, 0.3, 0.6]),
        0.55,
        epsilon = 1e-9
    );
}

#[test]
fn choice_uniform_is_zero() {
    assert_abs_diff_eq!(choice_confidence(&[0.5, 0.5]), 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(choice_confidence(&[0.2, 0.2]), 0.0, epsilon = 1e-9);
}

#[test]
fn choice_peak_matches_python() {
    assert_abs_diff_eq!(choice_confidence(&[0.82, 0.18]), 0.64, epsilon = 1e-9);
}

#[test]
fn single_outcome_is_certain() {
    assert_abs_diff_eq!(score_confidence(&[1.0]), 1.0, epsilon = 1e-9);
    assert_abs_diff_eq!(choice_confidence(&[1.0]), 1.0, epsilon = 1e-9);
}
