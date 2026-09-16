//! Provider-neutral request and result types.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::errors::TypeSafeError;

/// One chat message in provider-neutral form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }
}

/// The raw JSON payload a model returned and the tokens it cost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderResult {
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Perform one synchronous model request in native or prompted output mode.
pub trait Provider: Send + Sync {
    fn model_name(&self) -> &str;

    fn request(
        &self,
        messages: &[Message],
        schema: &Value,
        structured: bool,
    ) -> Result<ProviderResult, TypeSafeError>;

    fn translate_error(&self, error: &dyn std::error::Error) -> TypeSafeError;

    fn provider_label(&self) -> String {
        std::any::type_name_of_val(self).replace("::", ".")
    }
}

/// Perform one asynchronous model request in native or prompted output mode.
#[async_trait]
pub trait AsyncProvider: Send + Sync {
    fn model_name(&self) -> &str;

    async fn request(
        &self,
        messages: &[Message],
        schema: &Value,
        structured: bool,
    ) -> Result<ProviderResult, TypeSafeError>;

    fn translate_error(&self, error: &dyn std::error::Error) -> TypeSafeError;

    fn provider_label(&self) -> String {
        std::any::type_name_of_val(self).replace("::", ".")
    }
}

/// Render messages into the role/content dictionaries the chat APIs expect.
pub fn render_messages(messages: &[Message]) -> Value {
    serde_json::to_value(messages).unwrap_or(json!([]))
}

thread_local! {
    static THREAD_ATTEMPT: RefCell<Option<Arc<Mutex<Value>>>> = const { RefCell::new(None) };
}

tokio::task_local! {
    static TASK_ATTEMPT: Arc<Mutex<Value>>;
}

fn current_attempt() -> Option<Arc<Mutex<Value>>> {
    if let Ok(attempt) = TASK_ATTEMPT.try_with(Clone::clone) {
        return Some(attempt);
    }
    THREAD_ATTEMPT.with(|slot| slot.borrow().clone())
}

fn set_thread_attempt(attempt: Option<Arc<Mutex<Value>>>) -> Option<Arc<Mutex<Value>>> {
    THREAD_ATTEMPT.with(|slot| slot.replace(attempt))
}

/// Snapshot one provider call, including calls that raise before returning.
pub fn capture_attempt<T, F>(
    attempts: &mut Vec<Value>,
    model_name: &str,
    provider_label: &str,
    messages: &[Message],
    schema: &Value,
    structured: bool,
    function: F,
) -> Result<T, TypeSafeError>
where
    F: FnOnce() -> Result<T, TypeSafeError>,
{
    let slot = new_attempt(model_name, provider_label, messages, schema, structured);
    let previous = set_thread_attempt(Some(slot.clone()));
    let result = function();
    set_thread_attempt(previous);
    finish_attempt(attempts, slot, &result);
    result
}

/// Async snapshot of one provider call.
pub async fn capture_attempt_async<T, F, Fut>(
    attempts: &mut Vec<Value>,
    model_name: &str,
    provider_label: &str,
    messages: &[Message],
    schema: &Value,
    structured: bool,
    function: F,
) -> Result<T, TypeSafeError>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<T, TypeSafeError>>,
{
    let slot = new_attempt(model_name, provider_label, messages, schema, structured);
    let previous = set_thread_attempt(Some(slot.clone()));
    let result = TASK_ATTEMPT.scope(slot.clone(), function()).await;
    set_thread_attempt(previous);
    finish_attempt(attempts, slot, &result);
    result
}

fn new_attempt(
    model_name: &str,
    provider_label: &str,
    messages: &[Message],
    schema: &Value,
    structured: bool,
) -> Arc<Mutex<Value>> {
    Arc::new(Mutex::new(json!({
        "messages": render_messages(messages),
        "model_request_parameters": {
            "schema": schema,
            "structured": structured
        },
        "llm_response": Value::Null,
        "debug_info": {
            "model_name": model_name,
            "provider": provider_label
        }
    })))
}

fn finish_attempt<T>(
    attempts: &mut Vec<Value>,
    slot: Arc<Mutex<Value>>,
    result: &Result<T, TypeSafeError>,
) {
    let mut recorded = slot.lock().expect("attempt lock").clone();
    if let Err(error) = result {
        recorded["debug_info"]["error"] = json!(error.to_string());
        recorded["debug_info"]["error_type"] = json!(error.type_name());
    }
    attempts.push(recorded);
}

pub fn fill_result_if_missing(attempt: &mut Value, result: &ProviderResult) {
    if attempt
        .get("llm_response")
        .map(Value::is_null)
        .unwrap_or(true)
    {
        attempt["llm_response"] = serde_json::to_value(result).unwrap_or(Value::Null);
    }
}

/// Capture the built-in provider's request arguments before sending.
pub fn record_request(request: Value, api: &str) {
    if let Some(slot) = current_attempt() {
        let mut attempt = slot.lock().expect("attempt lock");
        attempt["request"] = request;
        attempt["debug_info"]["api"] = json!(api);
    }
}

/// Capture the provider response before parsing can reject incomplete output.
pub fn record_response(response: Value, finish_reason: Option<&str>) {
    if let Some(slot) = current_attempt() {
        let mut attempt = slot.lock().expect("attempt lock");
        attempt["llm_response"] = response;
        if let Some(reason) = finish_reason {
            attempt["debug_info"]["finish_reason"] = json!(reason);
        }
    }
}
