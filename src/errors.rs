//! TypeSafe-compatible error types.

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::fmt;

use serde_json::Value;

const MAX_ERROR_BODY_LENGTH: usize = 200;

/// Decode / schema-validation failure for a model payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError {
    pub message: String,
}

impl DecodeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl StdError for DecodeError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiErrorClass {
    Generic,
    BadRequest,
    Authentication,
    PermissionDenied,
    NotFound,
    UnprocessableEntity,
    RateLimit,
    InternalServer,
    ResponseValidation,
}

impl ApiErrorClass {
    pub fn type_name(self) -> &'static str {
        match self {
            Self::Generic => "TypeSafeAPIError",
            Self::BadRequest => "TypeSafeBadRequestError",
            Self::Authentication => "TypeSafeAuthenticationError",
            Self::PermissionDenied => "TypeSafePermissionDeniedError",
            Self::NotFound => "TypeSafeNotFoundError",
            Self::UnprocessableEntity => "TypeSafeUnprocessableEntityError",
            Self::RateLimit => "TypeSafeRateLimitError",
            Self::InternalServer => "TypeSafeInternalServerError",
            Self::ResponseValidation => "TypeSafeAPIResponseValidationError",
        }
    }

    fn from_status(status: u16) -> Self {
        match status {
            400 => Self::BadRequest,
            401 => Self::Authentication,
            403 => Self::PermissionDenied,
            404 => Self::NotFound,
            422 => Self::UnprocessableEntity,
            429 => Self::RateLimit,
            s if s >= 500 => Self::InternalServer,
            _ => Self::Generic,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub class: ApiErrorClass,
    pub status: u16,
    pub body: Value,
    pub headers: BTreeMap<String, String>,
    pub message: String,
    pub field_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeSafeErrorKind {
    Message(String),
    Api(ApiError),
    Connection(String),
    Timeout { timeout: Option<f64> },
}

/// Base exception for SDK failures, matching `typesafe_sdk.TypeSafeError`.
#[derive(Debug, Clone)]
pub struct TypeSafeError {
    pub kind: TypeSafeErrorKind,
    pub debug: Option<Value>,
    pub source: Option<DecodeError>,
}

impl TypeSafeError {
    pub fn message(message: impl Into<String>) -> Self {
        Self {
            kind: TypeSafeErrorKind::Message(message.into()),
            debug: None,
            source: None,
        }
    }

    pub fn connection(message: impl Into<String>) -> Self {
        Self {
            kind: TypeSafeErrorKind::Connection(message.into()),
            debug: None,
            source: None,
        }
    }

    pub fn timeout(timeout: Option<f64>) -> Self {
        Self {
            kind: TypeSafeErrorKind::Timeout { timeout },
            debug: None,
            source: None,
        }
    }

    pub fn api(api: ApiError) -> Self {
        Self {
            kind: TypeSafeErrorKind::Api(api),
            debug: None,
            source: None,
        }
    }

    pub fn with_debug(mut self, debug: Value) -> Self {
        self.debug = Some(debug);
        self
    }

    pub fn with_source(mut self, source: DecodeError) -> Self {
        self.source = Some(source);
        self
    }

    pub fn type_name(&self) -> &'static str {
        match &self.kind {
            TypeSafeErrorKind::Message(_) => "TypeSafeError",
            TypeSafeErrorKind::Api(api) => api.class.type_name(),
            TypeSafeErrorKind::Connection(_) => "TypeSafeAPIConnectionError",
            TypeSafeErrorKind::Timeout { .. } => "TypeSafeAPITimeoutError",
        }
    }

    pub fn status(&self) -> Option<u16> {
        match &self.kind {
            TypeSafeErrorKind::Api(api) => Some(api.status),
            _ => None,
        }
    }

    pub fn as_api(&self) -> Option<&ApiError> {
        match &self.kind {
            TypeSafeErrorKind::Api(api) => Some(api),
            _ => None,
        }
    }

    pub fn is_api_error(&self) -> bool {
        matches!(self.kind, TypeSafeErrorKind::Api(_))
    }

    pub fn retry_after_seconds(&self) -> Option<f64> {
        let api = self.as_api()?;
        parse_retry_after(&api.headers)
    }
}

impl fmt::Display for TypeSafeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            TypeSafeErrorKind::Message(message) => f.write_str(message),
            TypeSafeErrorKind::Api(api) => {
                if api.message.is_empty() {
                    write!(f, "{}", api.status)
                } else {
                    write!(f, "{} {}", api.status, api.message)
                }
            }
            TypeSafeErrorKind::Connection(message) => f.write_str(message),
            TypeSafeErrorKind::Timeout { timeout } => match timeout {
                Some(t) => write!(f, "Request timed out (timeout={t})."),
                None => write!(f, "Request timed out (timeout=None)."),
            },
        }
    }
}

impl StdError for TypeSafeError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_ref().map(|e| e as &(dyn StdError + 'static))
    }
}

/// Compatibility aliases matching the Python class names.
pub type TypeSafeAPIError = TypeSafeError;
pub type TypeSafeAPIConnectionError = TypeSafeError;
pub type TypeSafeAPITimeoutError = TypeSafeError;
pub type TypeSafeBadRequestError = TypeSafeError;
pub type TypeSafeAuthenticationError = TypeSafeError;
pub type TypeSafePermissionDeniedError = TypeSafeError;
pub type TypeSafeNotFoundError = TypeSafeError;
pub type TypeSafeUnprocessableEntityError = TypeSafeError;
pub type TypeSafeRateLimitError = TypeSafeError;
pub type TypeSafeInternalServerError = TypeSafeError;
pub type TypeSafeAPIResponseValidationError = TypeSafeError;

pub fn extract_message(body: &Value) -> Option<String> {
    match body {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Object(map) => {
            if let Some(Value::String(s)) = map.get("error") {
                return Some(s.clone());
            }
            if let Some(Value::Object(error)) = map.get("error") {
                if let Some(Value::String(s)) = error.get("message") {
                    return Some(s.clone());
                }
            }
            if let Some(Value::String(s)) = map.get("message") {
                return Some(s.clone());
            }
            if let Some(Value::String(s)) = map.get("detail") {
                return Some(s.clone());
            }
            if let Some(Value::Object(detail)) = map.get("detail") {
                if let Some(Value::String(s)) = detail.get("message") {
                    return Some(s.clone());
                }
            }
            None
        }
        _ => None,
    }
}

fn default_body_message(body: &Value) -> String {
    if body.is_null() {
        return "status code (no body)".into();
    }
    let raw = if let Value::String(s) = body {
        s.clone()
    } else {
        serde_json::to_string(body).unwrap_or_default()
    };
    if raw.len() > MAX_ERROR_BODY_LENGTH {
        format!("{}…", &raw[..MAX_ERROR_BODY_LENGTH])
    } else {
        raw
    }
}

/// Map an HTTP status to the matching TypeSafe API error class.
pub fn api_error(status: u16, body: Value, headers: BTreeMap<String, String>) -> TypeSafeError {
    let class = ApiErrorClass::from_status(status);
    let message = extract_message(&body).unwrap_or_else(|| default_body_message(&body));
    TypeSafeError::api(ApiError {
        class,
        status,
        body,
        headers,
        message,
        field_path: None,
    })
}

/// A successful HTTP response whose body failed schema validation.
pub fn response_validation_error(decode: DecodeError) -> TypeSafeError {
    TypeSafeError {
        kind: TypeSafeErrorKind::Api(ApiError {
            class: ApiErrorClass::ResponseValidation,
            status: 200,
            body: Value::String(decode.message.clone()),
            headers: BTreeMap::new(),
            message: "Invalid response data at 'answers'.".into(),
            field_path: Some("answers".into()),
        }),
        debug: None,
        source: Some(decode),
    }
}

pub fn parse_retry_after(headers: &BTreeMap<String, String>) -> Option<f64> {
    for (name, multiplier) in [("retry-after-ms", 1.0), ("retry-after", 1000.0)] {
        let raw = headers.iter().find_map(|(k, v)| {
            if k.eq_ignore_ascii_case(name) {
                Some(v.as_str())
            } else {
                None
            }
        })?;
        if let Ok(value) = raw.trim().parse::<f64>() {
            if value.is_finite() && value >= 0.0 {
                let delay_ms = value * multiplier;
                if delay_ms.is_finite() {
                    return Some(delay_ms / 1000.0);
                }
            }
        }
    }
    None
}
