//! Native Anthropic Messages API providers.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::runtime::Runtime;

use crate::errors::{api_error, TypeSafeError};
use crate::providers::base::{
    record_request, record_response, AsyncProvider, Message, MessageRole, Provider, ProviderResult,
};

const DEFAULT_MAX_TOKENS: u32 = 4096;
const DEFAULT_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

fn global_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().expect("tokio runtime"))
}

fn translate_reqwest(error: &reqwest::Error) -> TypeSafeError {
    if error.is_timeout() {
        TypeSafeError::timeout(None)
    } else if error.is_connect() {
        TypeSafeError::connection(error.to_string())
    } else {
        TypeSafeError::message(error.to_string())
    }
}

fn header_map(headers: &reqwest::header::HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|v| (name.as_str().to_string(), v.to_string()))
        })
        .collect()
}

/// Build the Messages API request body.
pub fn request_kwargs(
    model_name: &str,
    messages: &[Message],
    schema: &Value,
    structured: bool,
    max_tokens: u32,
) -> Value {
    let system = messages
        .iter()
        .filter(|message| message.role == MessageRole::System)
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let conversation: Vec<Value> = messages
        .iter()
        .filter(|message| message.role != MessageRole::System)
        .map(|message| {
            json!({
                "role": message.role,
                "content": message.content
            })
        })
        .collect();
    let mut kwargs = serde_json::Map::new();
    kwargs.insert("model".into(), json!(model_name));
    kwargs.insert("max_tokens".into(), json!(max_tokens));
    kwargs.insert("system".into(), json!(system));
    kwargs.insert("messages".into(), Value::Array(conversation));
    if structured {
        kwargs.insert(
            "output_config".into(),
            json!({ "format": { "type": "json_schema", "schema": schema } }),
        );
    }
    Value::Object(kwargs)
}

fn parse_result(response: &Value) -> Result<ProviderResult, TypeSafeError> {
    let stop_reason = response.get("stop_reason").and_then(Value::as_str);
    record_response(response.clone(), stop_reason);
    if stop_reason == Some("max_tokens") {
        return Err(TypeSafeError::message(
            "Anthropic response was truncated at the output token limit. \
Increase max_tokens on AnthropicProvider or AsyncAnthropicProvider, or request fewer questions.",
        ));
    }
    let mut text = String::new();
    if let Some(content) = response.get("content").and_then(Value::as_array) {
        for block in content {
            if block.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(piece) = block.get("text").and_then(Value::as_str) {
                    text.push_str(piece);
                }
            }
        }
    }
    Ok(ProviderResult {
        text,
        input_tokens: response
            .pointer("/usage/input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        output_tokens: response
            .pointer("/usage/output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

async fn send_messages(
    client: &reqwest::Client,
    api_key: Option<&str>,
    body: &Value,
) -> Result<Value, TypeSafeError> {
    let mut request = client
        .post(DEFAULT_URL)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("content-type", "application/json")
        .json(body);
    if let Some(key) = api_key {
        request = request.header("x-api-key", key);
    }
    let response = request
        .send()
        .await
        .map_err(|error| translate_reqwest(&error))?;
    let status = response.status();
    let headers = header_map(response.headers());
    let body: Value = response.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        return Err(api_error(status.as_u16(), body, headers));
    }
    Ok(body)
}

#[derive(Debug, Clone)]
struct AnthropicConfig {
    model_name: String,
    max_tokens: u32,
    api_key: Option<String>,
}

impl AnthropicConfig {
    fn new(model_name: impl Into<String>, max_tokens: u32) -> Result<Self, crate::Error> {
        if max_tokens == 0 {
            return Err(crate::Error::Value("max_tokens must be > 0".into()));
        }
        Ok(Self {
            model_name: model_name.into(),
            max_tokens,
            api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
        })
    }
}

/// Synchronously call the Anthropic Messages API.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    config: AnthropicConfig,
    http: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self::build(model_name, DEFAULT_MAX_TOKENS).expect("default max_tokens is valid")
    }

    pub fn build(model_name: impl Into<String>, max_tokens: u32) -> Result<Self, crate::Error> {
        Ok(Self {
            config: AnthropicConfig::new(model_name, max_tokens)?,
            http: reqwest::Client::new(),
        })
    }

    pub fn max_tokens(self, max_tokens: u32) -> Result<Self, crate::Error> {
        Self::build(self.config.model_name, max_tokens)
    }

    pub fn max_tokens_value(&self) -> u32 {
        self.config.max_tokens
    }
}

impl Provider for AnthropicProvider {
    fn model_name(&self) -> &str {
        &self.config.model_name
    }

    fn request(
        &self,
        messages: &[Message],
        schema: &Value,
        structured: bool,
    ) -> Result<ProviderResult, TypeSafeError> {
        global_runtime().block_on(async_anthropic_request(
            &self.http,
            &self.config,
            messages,
            schema,
            structured,
        ))
    }

    fn translate_error(&self, error: &dyn std::error::Error) -> TypeSafeError {
        TypeSafeError::message(error.to_string())
    }
}

/// Asynchronously call the Anthropic Messages API.
#[derive(Debug, Clone)]
pub struct AsyncAnthropicProvider {
    config: AnthropicConfig,
    http: reqwest::Client,
}

impl AsyncAnthropicProvider {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self::build(model_name, DEFAULT_MAX_TOKENS).expect("default max_tokens is valid")
    }

    pub fn build(model_name: impl Into<String>, max_tokens: u32) -> Result<Self, crate::Error> {
        Ok(Self {
            config: AnthropicConfig::new(model_name, max_tokens)?,
            http: reqwest::Client::new(),
        })
    }

    pub fn max_tokens(self, max_tokens: u32) -> Result<Self, crate::Error> {
        Self::build(self.config.model_name, max_tokens)
    }
}

#[async_trait]
impl AsyncProvider for AsyncAnthropicProvider {
    fn model_name(&self) -> &str {
        &self.config.model_name
    }

    async fn request(
        &self,
        messages: &[Message],
        schema: &Value,
        structured: bool,
    ) -> Result<ProviderResult, TypeSafeError> {
        async_anthropic_request(&self.http, &self.config, messages, schema, structured).await
    }

    fn translate_error(&self, error: &dyn std::error::Error) -> TypeSafeError {
        TypeSafeError::message(error.to_string())
    }
}

async fn async_anthropic_request(
    http: &reqwest::Client,
    config: &AnthropicConfig,
    messages: &[Message],
    schema: &Value,
    structured: bool,
) -> Result<ProviderResult, TypeSafeError> {
    let body = request_kwargs(
        &config.model_name,
        messages,
        schema,
        structured,
        config.max_tokens,
    );
    record_request(body.clone(), "messages");
    let response = send_messages(http, config.api_key.as_deref(), &body).await?;
    parse_result(&response)
}
