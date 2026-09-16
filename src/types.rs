//! Question, answer, retry, and JSON value types compatible with `typesafe_sdk`.

use std::collections::{BTreeMap, BTreeSet};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::TypeSafeError;

/// How the LLM should encode each answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerMode {
    Probabilities,
    Discrete,
}

impl AnswerMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Probabilities => "probabilities",
            Self::Discrete => "discrete",
        }
    }
}

impl TryFrom<&str> for AnswerMode {
    type Error = crate::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "probabilities" => Ok(Self::Probabilities),
            "discrete" => Ok(Self::Discrete),
            _ => Err(crate::Error::Value(
                "llm_answer_mode must be 'probabilities' or 'discrete'".into(),
            )),
        }
    }
}

/// Named LLM vendor selected by a model string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderName {
    OpenAi,
    Anthropic,
}

impl ProviderName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

impl TryFrom<&str> for ProviderName {
    type Error = crate::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "openai" => Ok(Self::OpenAi),
            "anthropic" => Ok(Self::Anthropic),
            other => Err(crate::Error::Value(format!(
                "unknown provider {other:?}; expected 'openai' or 'anthropic'"
            ))),
        }
    }
}

/// Optional yes/no outcome descriptions for a Noul question.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NoulCriteria {
    pub true_criteria: Option<Value>,
    pub false_criteria: Option<Value>,
}

/// A yes/no question with optional descriptions for either outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct Noul {
    pub instructions: Option<Value>,
    pub criteria: Option<NoulCriteria>,
}

impl Noul {
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            instructions: Some(Value::String(instructions.into())),
            criteria: None,
        }
    }

    pub fn with_criteria(mut self, criteria: NoulCriteria) -> Self {
        self.criteria = Some(criteria);
        self
    }
}

/// A question that selects between named alternatives.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub instructions: Option<Value>,
    pub criteria: IndexMap<String, Option<Value>>,
}

impl Choice {
    pub fn new(
        instructions: impl Into<String>,
        criteria: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self {
            instructions: Some(Value::String(instructions.into())),
            criteria: criteria
                .into_iter()
                .map(|(k, v)| (k.into(), Some(Value::String(v.into()))))
                .collect(),
        }
    }

    pub fn from_criteria(criteria: IndexMap<String, Option<Value>>) -> Self {
        Self {
            instructions: None,
            criteria,
        }
    }
}

/// A question that assigns a score using an ordered rubric.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    pub instructions: Option<Value>,
    pub criteria: Vec<Value>,
}

impl Score {
    pub fn new(
        instructions: impl Into<String>,
        criteria: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            instructions: Some(Value::String(instructions.into())),
            criteria: criteria
                .into_iter()
                .map(|c| Value::String(c.into()))
                .collect(),
        }
    }

    pub fn from_criteria(criteria: Vec<Value>) -> Self {
        Self {
            instructions: None,
            criteria,
        }
    }
}

/// Validated question kinds accepted by an evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum Question {
    Noul(Noul),
    Choice(Choice),
    Score(Score),
}

impl From<Noul> for Question {
    fn from(value: Noul) -> Self {
        Self::Noul(value)
    }
}

impl From<Choice> for Question {
    fn from(value: Choice) -> Self {
        Self::Choice(value)
    }
}

impl From<Score> for Question {
    fn from(value: Score) -> Self {
        Self::Score(value)
    }
}

impl Question {
    pub fn noul(instructions: impl Into<String>) -> Self {
        Self::Noul(Noul::new(instructions))
    }

    pub fn instructions(&self) -> Option<&Value> {
        match self {
            Self::Noul(q) => q.instructions.as_ref(),
            Self::Choice(q) => q.instructions.as_ref(),
            Self::Score(q) => q.instructions.as_ref(),
        }
    }
}

/// A yes/no answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoulAnswer {
    pub noul: f64,
}

/// A selected label and its probabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub confidence: f64,
    pub probabilities: BTreeMap<String, f64>,
}

/// An expected score with its rubric and probabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreAnswer {
    pub score: f64,
    pub confidence: f64,
    #[serde(serialize_with = "serialize_int_keyed_values")]
    pub legend: BTreeMap<u32, Value>,
    #[serde(serialize_with = "serialize_int_keyed_f64")]
    pub probabilities: BTreeMap<u32, f64>,
}

fn serialize_int_keyed_values<S>(
    map: &BTreeMap<u32, Value>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeMap;
    let mut s = serializer.serialize_map(Some(map.len()))?;
    for (k, v) in map {
        s.serialize_entry(&k.to_string(), v)?;
    }
    s.end()
}

fn serialize_int_keyed_f64<S>(map: &BTreeMap<u32, f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeMap;
    let mut s = serializer.serialize_map(Some(map.len()))?;
    for (k, v) in map {
        s.serialize_entry(&k.to_string(), v)?;
    }
    s.end()
}

/// An answer to a single question, identified by its `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Answer {
    #[serde(rename = "noul")]
    Noul(NoulAnswer),
    #[serde(rename = "choice")]
    Choice(ChoiceAnswer),
    #[serde(rename = "score")]
    Score(ScoreAnswer),
}

/// Configuration for retry behavior, matching `typesafe_sdk.RetryPolicy`.
#[derive(Debug, Clone, PartialEq)]
pub struct RetryPolicy {
    /// Maximum retries after the initial attempt; `0` disables retries.
    pub max_retries: u32,
    /// First backoff delay in seconds, doubled each attempt up to `backoff_max`.
    pub backoff_initial: f64,
    /// Maximum backoff delay in seconds; zero disables backoff.
    pub backoff_max: f64,
    /// Fraction of each backoff delay randomly subtracted, between 0 and 1.
    pub backoff_jitter: f64,
    /// HTTP status codes that are retried.
    pub http_statuses: BTreeSet<u16>,
    /// Whether to honor `Retry-After` and `retry-after-ms` response headers.
    pub respect_retry_after: bool,
    /// Whether to retry connection failures.
    pub api_connection_error: bool,
    /// Whether to retry timeouts.
    pub api_timeout_error: bool,
    /// Total retry budget in seconds per call, including the initial attempt.
    pub timeout: Option<f64>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        let mut http_statuses: BTreeSet<u16> = (500..600).collect();
        http_statuses.insert(408);
        http_statuses.insert(429);
        Self {
            max_retries: 2,
            backoff_initial: 0.5,
            backoff_max: 5.0,
            backoff_jitter: 0.25,
            http_statuses,
            respect_retry_after: true,
            api_connection_error: true,
            api_timeout_error: true,
            timeout: Some(30.0),
        }
    }
}

impl RetryPolicy {
    pub fn disabled() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), TypeSafeError> {
        if !self.backoff_initial.is_finite() || self.backoff_initial < 0.0 {
            return Err(TypeSafeError::message(
                "backoff_initial must be a non-negative, finite number of seconds.",
            ));
        }
        if !self.backoff_max.is_finite() || self.backoff_max < 0.0 {
            return Err(TypeSafeError::message(
                "backoff_max must be a non-negative, finite number of seconds.",
            ));
        }
        if !(0.0..=1.0).contains(&self.backoff_jitter) {
            return Err(TypeSafeError::message(
                "backoff_jitter must be between zero and one.",
            ));
        }
        if let Some(timeout) = self.timeout {
            if !timeout.is_finite() || timeout <= 0.0 {
                return Err(TypeSafeError::message(
                    "timeout must be a positive, finite number of seconds.",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn is_retryable(&self, error: &TypeSafeError) -> bool {
        match &error.kind {
            crate::errors::TypeSafeErrorKind::Timeout { .. } => self.api_timeout_error,
            crate::errors::TypeSafeErrorKind::Connection(_) => self.api_connection_error,
            crate::errors::TypeSafeErrorKind::Api(api) => self.http_statuses.contains(&api.status),
            crate::errors::TypeSafeErrorKind::Message(_) => false,
        }
    }

    pub(crate) fn wait_seconds(&self, attempt_number: u32, error: &TypeSafeError) -> f64 {
        if self.respect_retry_after {
            if let Some(delay) = error.retry_after_seconds() {
                return delay;
            }
        }
        backoff(
            attempt_number,
            self.backoff_initial,
            self.backoff_max,
            self.backoff_jitter,
        )
    }
}

fn backoff(attempt: u32, initial: f64, maximum: f64, jitter: f64) -> f64 {
    if initial == 0.0 || maximum == 0.0 {
        return 0.0;
    }
    let exponent = attempt.saturating_sub(1);
    let log_span = maximum.log2() - initial.log2();
    let exponential = if (exponent as f64) >= log_span {
        maximum
    } else {
        initial * 2f64.powi(exponent as i32)
    };
    let delay = exponential * (1.0 - rand::random::<f64>() * jitter);
    exponential.min((delay * 1000.0).round() / 1000.0)
}

/// Validated questions keyed by ID, preserving caller order for schema fields.
#[derive(Debug, Clone)]
pub struct QuestionCollection {
    pub questions: BTreeMap<String, Question>,
    pub order: Vec<String>,
}

impl QuestionCollection {
    pub fn get(&self, id: &str) -> Option<&Question> {
        self.questions.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Question)> {
        self.order
            .iter()
            .map(|id| (id.as_str(), &self.questions[id]))
    }

    pub fn is_empty(&self) -> bool {
        self.questions.is_empty()
    }

    pub fn len(&self) -> usize {
        self.questions.len()
    }
}

const MIN_CRITERIA: usize = 2;

/// Convert a question collection to validated models.
pub fn convert_question_collection<I, K, Q>(
    questions: I,
) -> Result<QuestionCollection, crate::Error>
where
    I: IntoIterator<Item = (K, Q)>,
    K: Into<String>,
    Q: Into<Question>,
{
    let mut map = BTreeMap::new();
    let mut order = Vec::new();
    for (key, question) in questions {
        let key = key.into();
        let question = question.into();
        match &question {
            Question::Score(score) if score.criteria.len() < MIN_CRITERIA => {
                return Err(crate::Error::Value(
                    "Score and choice questions require at least two criteria.".into(),
                ));
            }
            Question::Choice(choice) if choice.criteria.len() < MIN_CRITERIA => {
                return Err(crate::Error::Value(
                    "Score and choice questions require at least two criteria.".into(),
                ));
            }
            _ => {}
        }
        if map.insert(key.clone(), question).is_none() {
            order.push(key);
        }
    }
    if map.is_empty() {
        return Err(crate::Error::Value(
            "At least one question is required.".into(),
        ));
    }
    Ok(QuestionCollection {
        questions: map,
        order,
    })
}
