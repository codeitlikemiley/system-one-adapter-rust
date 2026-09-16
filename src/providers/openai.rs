//! OpenAI Responses and OpenAI-compatible Chat Completions providers.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::runtime::Runtime;

use crate::errors::{api_error, TypeSafeError};
use crate::providers::base::{
    record_request, record_response, render_messages, AsyncProvider, Message, Provider,
    ProviderResult,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Which OpenAI transport to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAIApi {
    Responses,
    ChatCompletions,
}

impl OpenAIApi {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Responses => "responses",
            Self::ChatCompletions => "chat_completions",
        }
    }

    fn from_arg(api: Option<&str>) -> Result<Option<Self>, TypeSafeError> {
        match api {
            None => Ok(None),
            Some("responses") => Ok(Some(Self::Responses)),
            Some("chat_completions") => Ok(Some(Self::ChatCompletions)),
            Some(_) => Err(TypeSafeError::message(
                "api must be 'responses' or 'chat_completions'",
            )),
        }
    }
}

fn global_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().expect("tokio runtime"))
}

fn default_base_url() -> String {
    std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string())
}

fn default_api(base_url: &str, explicit: Option<OpenAIApi>) -> OpenAIApi {
    if let Some(api) = explicit {
        return api;
    }
    match reqwest::Url::parse(base_url) {
        Ok(url) if url.host_str() == Some("api.openai.com") => OpenAIApi::Responses,
        _ => OpenAIApi::ChatCompletions,
    }
}

fn join_url(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

fn response_format(schema: &Value, structured: bool) -> Option<Value> {
    if !structured {
        return None;
    }
    Some(json!({
        "type": "json_schema",
        "json_schema": {
            "name": "evaluation",
            "schema": schema,
            "strict": true
        }
    }))
}

fn responses_request_kwargs(
    model_name: &str,
    messages: &[Message],
    schema: &Value,
    structured: bool,
) -> Value {
    let output_format = if structured {
        json!({
            "type": "json_schema",
            "name": "evaluation",
            "schema": schema,
            "strict": true
        })
    } else {
        json!({ "type": "json_object" })
    };
    let mut kwargs = serde_json::Map::new();
    kwargs.insert("model".into(), json!(model_name));
    kwargs.insert("text".into(), json!({ "format": output_format }));
    kwargs.insert("store".into(), json!(false));
    if structured {
        let instructions = messages
            .iter()
            .filter(|message| message.role == crate::providers::base::MessageRole::System)
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let input: Vec<&Message> = messages
            .iter()
            .filter(|message| message.role != crate::providers::base::MessageRole::System)
            .collect();
        kwargs.insert("instructions".into(), json!(instructions));
        kwargs.insert(
            "input".into(),
            serde_json::to_value(input).unwrap_or(json!([])),
        );
    } else {
        kwargs.insert("input".into(), render_messages(messages));
    }
    Value::Object(kwargs)
}

fn chat_request_kwargs(
    model_name: &str,
    messages: &[Message],
    schema: &Value,
    structured: bool,
) -> Value {
    json!({
        "model": model_name,
        "messages": render_messages(messages),
        "response_format": response_format(schema, structured)
    })
}

fn chat_result(response: &Value) -> Result<ProviderResult, TypeSafeError> {
    let finish = response
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str);
    record_response(response.clone(), finish);
    let text = response
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let input_tokens = response
        .pointer("/usage/prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = response
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Ok(ProviderResult {
        text,
        input_tokens,
        output_tokens,
    })
}

fn responses_output_text(response: &Value) -> String {
    if let Some(text) = response.get("output_text").and_then(Value::as_str) {
        return text.to_string();
    }
    let mut pieces = Vec::new();
    if let Some(output) = response.get("output").and_then(Value::as_array) {
        for item in output {
            if let Some(content) = item.get("content").and_then(Value::as_array) {
                for block in content {
                    if block.get("type").and_then(Value::as_str) == Some("output_text") {
                        if let Some(text) = block.get("text").and_then(Value::as_str) {
                            pieces.push(text.to_string());
                        }
                    }
                }
            }
        }
    }
    pieces.join("")
}

fn responses_result(response: &Value) -> Result<ProviderResult, TypeSafeError> {
    let status = response.get("status").and_then(Value::as_str);
    record_response(response.clone(), status);
    if status != Some("completed") {
        let reason =
            if let Some(message) = response.pointer("/error/message").and_then(Value::as_str) {
                message.to_string()
            } else if let Some(reason) = response
                .pointer("/incomplete_details/reason")
                .and_then(Value::as_str)
            {
                reason.to_string()
            } else {
                status.unwrap_or("unknown").to_string()
            };
        return Err(TypeSafeError::message(format!(
            "OpenAI response did not complete: {reason}."
        )));
    }
    Ok(ProviderResult {
        text: responses_output_text(response),
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

pub(crate) fn translate_reqwest(error: &reqwest::Error) -> TypeSafeError {
    if error.is_timeout() {
        TypeSafeError::timeout(None)
    } else if error.is_connect() {
        TypeSafeError::connection(error.to_string())
    } else {
        TypeSafeError::message(error.to_string())
    }
}

async fn send_openai(
    client: &reqwest::Client,
    url: &str,
    api_key: Option<&str>,
    body: &Value,
) -> Result<Value, TypeSafeError> {
    let mut request = client.post(url).json(body);
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
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
struct OpenAIConfig {
    model_name: String,
    base_url: String,
    api_key: Option<String>,
    api: OpenAIApi,
}

impl OpenAIConfig {
    fn new(
        model_name: impl Into<String>,
        base_url: Option<String>,
        api_key: Option<String>,
        api: Option<&str>,
    ) -> Result<Self, TypeSafeError> {
        let base_url = base_url.unwrap_or_else(default_base_url);
        let explicit = OpenAIApi::from_arg(api)?;
        let api = default_api(&base_url, explicit);
        let api_key = api_key.or_else(|| std::env::var("OPENAI_API_KEY").ok());
        Ok(Self {
            model_name: model_name.into(),
            base_url,
            api_key,
            api,
        })
    }

    fn endpoint(&self) -> String {
        match self.api {
            OpenAIApi::Responses => join_url(&self.base_url, "responses"),
            OpenAIApi::ChatCompletions => join_url(&self.base_url, "chat/completions"),
        }
    }

    fn request_body(&self, messages: &[Message], schema: &Value, structured: bool) -> Value {
        match self.api {
            OpenAIApi::Responses => {
                responses_request_kwargs(&self.model_name, messages, schema, structured)
            }
            OpenAIApi::ChatCompletions => {
                chat_request_kwargs(&self.model_name, messages, schema, structured)
            }
        }
    }

    fn parse(&self, response: &Value) -> Result<ProviderResult, TypeSafeError> {
        match self.api {
            OpenAIApi::Responses => responses_result(response),
            OpenAIApi::ChatCompletions => chat_result(response),
        }
    }
}

/// Synchronously call Responses or an OpenAI-compatible chat API.
#[derive(Debug, Clone)]
pub struct OpenAIProvider {
    config: OpenAIConfig,
    http: reqwest::Client,
}

impl OpenAIProvider {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self::build(model_name, None, None, None).expect("default OpenAI provider")
    }

    pub fn build(
        model_name: impl Into<String>,
        base_url: Option<String>,
        api_key: Option<String>,
        api: Option<&str>,
    ) -> Result<Self, TypeSafeError> {
        Ok(Self {
            config: OpenAIConfig::new(model_name, base_url, api_key, api)?,
            http: reqwest::Client::builder()
                .build()
                .map_err(|error| TypeSafeError::message(error.to_string()))?,
        })
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        self.config.api = default_api(&base_url, None);
        self.config.base_url = base_url;
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.config.api_key = Some(api_key.into());
        self
    }

    pub fn api(mut self, api: &str) -> Result<Self, TypeSafeError> {
        self.config.api = OpenAIApi::from_arg(Some(api))?.expect("api is some");
        Ok(self)
    }

    pub fn selected_api(&self) -> OpenAIApi {
        self.config.api
    }
}

impl Provider for OpenAIProvider {
    fn model_name(&self) -> &str {
        &self.config.model_name
    }

    fn request(
        &self,
        messages: &[Message],
        schema: &Value,
        structured: bool,
    ) -> Result<ProviderResult, TypeSafeError> {
        global_runtime().block_on(async_openai_request(
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

/// Asynchronously call Responses or an OpenAI-compatible chat API.
#[derive(Debug, Clone)]
pub struct AsyncOpenAIProvider {
    config: OpenAIConfig,
    http: reqwest::Client,
}

impl AsyncOpenAIProvider {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self::build(model_name, None, None, None).expect("default OpenAI provider")
    }

    pub fn build(
        model_name: impl Into<String>,
        base_url: Option<String>,
        api_key: Option<String>,
        api: Option<&str>,
    ) -> Result<Self, TypeSafeError> {
        Ok(Self {
            config: OpenAIConfig::new(model_name, base_url, api_key, api)?,
            http: reqwest::Client::builder()
                .build()
                .map_err(|error| TypeSafeError::message(error.to_string()))?,
        })
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        self.config.api = default_api(&base_url, None);
        self.config.base_url = base_url;
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.config.api_key = Some(api_key.into());
        self
    }

    pub fn api(mut self, api: &str) -> Result<Self, TypeSafeError> {
        self.config.api = OpenAIApi::from_arg(Some(api))?.expect("api is some");
        Ok(self)
    }

    pub fn selected_api(&self) -> OpenAIApi {
        self.config.api
    }
}

#[async_trait]
impl AsyncProvider for AsyncOpenAIProvider {
    fn model_name(&self) -> &str {
        &self.config.model_name
    }

    async fn request(
        &self,
        messages: &[Message],
        schema: &Value,
        structured: bool,
    ) -> Result<ProviderResult, TypeSafeError> {
        async_openai_request(&self.http, &self.config, messages, schema, structured).await
    }

    fn translate_error(&self, error: &dyn std::error::Error) -> TypeSafeError {
        TypeSafeError::message(error.to_string())
    }
}

async fn async_openai_request(
    http: &reqwest::Client,
    config: &OpenAIConfig,
    messages: &[Message],
    schema: &Value,
    structured: bool,
) -> Result<ProviderResult, TypeSafeError> {
    let body = config.request_body(messages, schema, structured);
    record_request(body.clone(), config.api.as_str());
    let response = send_openai(http, &config.endpoint(), config.api_key.as_deref(), &body).await?;
    config.parse(&response)
}

/// Request envelope helpers kept for tests and documentation.
pub fn openai_response_format(schema: &Value, structured: bool) -> Option<Value> {
    response_format(schema, structured)
}

pub fn openai_responses_request_kwargs(
    model_name: &str,
    messages: &[Message],
    schema: &Value,
    structured: bool,
) -> Value {
    responses_request_kwargs(model_name, messages, schema, structured)
}
