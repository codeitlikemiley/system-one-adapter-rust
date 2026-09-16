//! Confidence metrics for probability distributions.

/// Measure score concentration around its modal score.
pub fn score_confidence(probs: &[f64]) -> f64 {
    if probs.len() == 1 {
        return 1.0;
    }
    let normalized = normalize(probs);
    let mode_index = normalized
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let distance_from_mode: f64 = normalized
        .iter()
        .enumerate()
        .map(|(index, probability)| probability * (index as f64 - mode_index as f64).abs())
        .sum();
    let uniform_center = (normalized.len() as f64 - 1.0) / 2.0;
    let uniform_mean_absolute_deviation: f64 = (0..normalized.len())
        .map(|index| (index as f64 - uniform_center).abs())
        .sum::<f64>()
        / normalized.len() as f64;
    (1.0 - distance_from_mode / uniform_mean_absolute_deviation).max(0.0)
}

/// Scale peak choice probability from uniform to certainty.
pub fn choice_confidence(probs: &[f64]) -> f64 {
    if probs.len() == 1 {
        return 1.0;
    }
    let normalized = normalize(probs);
    let uniform_probability = 1.0 / normalized.len() as f64;
    let peak = normalized.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (peak - uniform_probability) / (1.0 - uniform_probability)
}

fn normalize(probs: &[f64]) -> Vec<f64> {
    let total: f64 = probs.iter().sum();
    if total == 0.0 {
        return vec![1.0 / probs.len() as f64; probs.len()];
    }
    probs
        .iter()
        .map(|probability| probability / total)
        .collect()
}
