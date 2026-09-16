//! Retry handling and reusable provider-error mapping.

use std::thread;
use std::time::{Duration, Instant};

use crate::errors::TypeSafeError;
use crate::types::RetryPolicy;

/// Reason for performing one retry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryReasons {
    pub category: RetryCategory;
    pub msg: String;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryCategory {
    ProviderError;
    MalformedStructure;
}

impl RetryCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ProviderError => "provider_error",
            Self::MalformedStructure => "malformed_structure",
        }
    }
}

/// Apply the SDK retry policy to a provider call that raises SDK errors.
pub fn run_with_retries<T, F>(
    mut function: F,
    retry: &RetryPolicy,
    mut retry_reasons: Option<&mut Vec<RetryReasons>>,
) -> Result<(T, u32), TypeSafeError>
where
    F: FnMut() -> Result<T, TypeSafeError>,
{
    retry.validate()?;
    let started = Instant::now();
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        match function() {
            Ok(result) => return Ok((result, attempts.saturating_sub(1))),
            Err(error) => {
                if attempts.saturating_sub(1) >= retry.max_retries || !retry.is_retryable(&error) {
                    return Err(error);
                }
                let wait = retry.wait_seconds(attempts, &error);
                if let Some(timeout) = retry.timeout {
                    if started.elapsed().as_secs_f64() + wait >= timeout {
                        return Err(error);
                    }
                }
                if let Some(reasons) = retry_reasons.as_deref_mut() {
                    reasons.push(RetryReasons {
                        category: RetryCategory::ProviderError,
                        msg: error.to_string(),
                    });
                }
                if wait > 0.0 {
                    thread::sleep(Duration::from_secs_f64(wait));
                }
            }
        }
    }
}

/// Apply the same retry policy to asynchronous provider calls.
pub async fn run_with_retries_async<T, F, Fut>(
    mut function: F,
    retry: &RetryPolicy,
    mut retry_reasons: Option<&mut Vec<RetryReasons>>,
) -> Result<(T, u32), TypeSafeError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, TypeSafeError>>,
{
    retry.validate()?;
    let started = Instant::now();
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        match function().await {
            Ok(result) => return Ok((result, attempts.saturating_sub(1))),
            Err(error) => {
                if attempts.saturating_sub(1) >= retry.max_retries || !retry.is_retryable(&error) {
                    return Err(error);
                }
                let wait = retry.wait_seconds(attempts, &error);
                if let Some(timeout) = retry.timeout {
                    if started.elapsed().as_secs_f64() + wait >= timeout {
                        return Err(error);
                    }
                }
                if let Some(reasons) = retry_reasons.as_deref_mut() {
                    reasons.push(RetryReasons {
                        category: RetryCategory::ProviderError,
                        msg: error.to_string(),
                    });
                }
                if wait > 0.0 {
                    tokio::time::sleep(Duration::from_secs_f64(wait)).await;
                }
            }
        }
    }
}
