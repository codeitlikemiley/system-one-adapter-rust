//! Response types extended with LLM accounting and diagnostics.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::types::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer};

/// Token counts plus cumulative retry accounting.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Usage {
    pub input_tokens: u64;
    pub output_tokens: u64;
    pub input_tokens_total: u64;
    pub output_tokens_total: u64;
    pub n_retries: u32;
    pub n_retries_malformed_structure: u32;
    pub latency: f64;
}

/// TypeSafe-shaped answers with retry and probability diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SystemOneResponse {
    pub model: String;
    pub answers: BTreeMap<String, Answer>;
    pub usage: Usage;
    pub debug: serde_json::Value;
}

impl SystemOneResponse {
    /// Yes/no answers keyed by question name.
    pub fn nouls(&self) -> BTreeMap<String, NoulAnswer> {
        self.answers
            .iter()
            .filter_map(|(name, answer)| match answer {
                Answer::Noul(answer) => Some((name.clone(), answer.clone())),
                _ => None,
            })
            .collect()
    }

    /// Choice answers keyed by question name.
    pub fn choices(&self) -> BTreeMap<String, ChoiceAnswer> {
        self.answers
            .iter()
            .filter_map(|(name, answer)| match answer {
                Answer::Choice(answer) => Some((name.clone(), answer.clone())),
                _ => None,
            })
            .collect()
    }

    /// Score answers keyed by question name.
    pub fn scores(&self) -> BTreeMap<String, ScoreAnswer> {
        self.answers
            .iter()
            .filter_map(|(name, answer)| match answer {
                Answer::Score(answer) => Some((name.clone(), answer.clone())),
                _ => None,
            })
            .collect()
    }
}
