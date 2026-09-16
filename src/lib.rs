//! System One Adapter: TypeSafe-compatible evaluation backed by LLM APIs.

pub mod client;
pub mod errors;
pub mod providers;
pub mod response;
pub mod schema;
pub mod types;
pub mod utils;

mod json_util;

pub use client::{
    AsyncSystemOneAdapterClient, AsyncSystemOneArgs, ClientConfig, EvaluationRun, IntoState,
    SystemOneAdapterClient, SystemOneArgs,
};
pub use errors::{
    api_error, DecodeError, TypeSafeAPIConnectionError, TypeSafeAPIError,
    TypeSafeAPIResponseValidationError, TypeSafeAPITimeoutError, TypeSafeAuthenticationError,
    TypeSafeBadRequestError, TypeSafeError, TypeSafeInternalServerError, TypeSafeNotFoundError,
    TypeSafePermissionDeniedError, TypeSafeRateLimitError, TypeSafeUnprocessableEntityError,
};
pub use response::{SystemOneResponse, Usage};
pub use types::{
    Answer, AnswerMode, Choice, ChoiceAnswer, Noul, NoulAnswer, NoulCriteria, ProviderName,
    Question, QuestionCollection, RetryPolicy, Score, ScoreAnswer,
};

/// Installed package version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Crate-level error: invalid arguments or a TypeSafe SDK error.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Value(String),
    #[error(transparent)]
    TypeSafe(#[from] TypeSafeError),
}

impl Error {
    pub fn as_typesafe(&self) -> Option<&TypeSafeError> {
        match self {
            Self::TypeSafe(error) => Some(error),
            Self::Value(_) => None,
        }
    }
}
