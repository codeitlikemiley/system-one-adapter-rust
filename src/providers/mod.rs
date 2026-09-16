//! Provider seam: OpenAI-compatible and native Anthropic model requests.

mod base;

pub use base::{
    capture_attempt, capture_attempt_async, fill_result_if_missing, record_request,
    record_response, render_messages, AsyncProvider, Message, MessageRole, Provider,
    ProviderResult,
};

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

use crate::types::ProviderName;
use crate::Error;

const MISSING_PROVIDER: &str = "A provider is required: set provider='openai' or 'anthropic', or pass a provider instance as the model.";

#[allow(dead_code)]
fn missing_extra(provider: &str) -> Error {
    Error::Value(format!(
        "The '{provider}' provider requires its optional dependency; enable the `{provider}` crate feature: cargo add system-one-adapter --features {provider}"
    ))
}

/// Build the selected synchronous provider, or use an injected provider.
pub fn build_sync_provider<'a>(
    provider: Option<ProviderName>,
    model: SyncModel<'a>,
) -> Result<SyncModelOwned<'a>, Error> {
    match model {
        SyncModel::Provider(instance) => Ok(SyncModelOwned::Borrowed(instance)),
        SyncModel::Name(name) => {
            let Some(provider) = provider else {
                return Err(Error::Value(MISSING_PROVIDER.into()));
            };
            match provider {
                ProviderName::OpenAi => {
                    #[cfg(feature = "openai")]
                    {
                        Ok(SyncModelOwned::Owned(Box::new(
                            openai::OpenAIProvider::new(name),
                        )))
                    }
                    #[cfg(not(feature = "openai"))]
                    {
                        let _ = name;
                        Err(missing_extra("openai"))
                    }
                }
                ProviderName::Anthropic => {
                    #[cfg(feature = "anthropic")]
                    {
                        Ok(SyncModelOwned::Owned(Box::new(
                            anthropic::AnthropicProvider::new(name),
                        )))
                    }
                    #[cfg(not(feature = "anthropic"))]
                    {
                        let _ = name;
                        Err(missing_extra("anthropic"))
                    }
                }
            }
        }
    }
}

/// Build the selected asynchronous provider, or use an injected provider.
pub fn build_async_provider<'a>(
    provider: Option<ProviderName>,
    model: AsyncModel<'a>,
) -> Result<AsyncModelOwned<'a>, Error> {
    match model {
        AsyncModel::Provider(instance) => Ok(AsyncModelOwned::Borrowed(instance)),
        AsyncModel::Name(name) => {
            let Some(provider) = provider else {
                return Err(Error::Value(MISSING_PROVIDER.into()));
            };
            match provider {
                ProviderName::OpenAi => {
                    #[cfg(feature = "openai")]
                    {
                        Ok(AsyncModelOwned::Owned(Box::new(
                            openai::AsyncOpenAIProvider::new(name),
                        )))
                    }
                    #[cfg(not(feature = "openai"))]
                    {
                        let _ = name;
                        Err(missing_extra("openai"))
                    }
                }
                ProviderName::Anthropic => {
                    #[cfg(feature = "anthropic")]
                    {
                        Ok(AsyncModelOwned::Owned(Box::new(
                            anthropic::AsyncAnthropicProvider::new(name),
                        )))
                    }
                    #[cfg(not(feature = "anthropic"))]
                    {
                        let _ = name;
                        Err(missing_extra("anthropic"))
                    }
                }
            }
        }
    }
}

/// A model name or an injected synchronous provider.
pub enum SyncModel<'a> {
    Name(String),
    Provider(&'a dyn Provider),
}

impl From<String> for SyncModel<'_> {
    fn from(value: String) -> Self {
        Self::Name(value)
    }
}

impl From<&str> for SyncModel<'_> {
    fn from(value: &str) -> Self {
        Self::Name(value.to_string())
    }
}

impl<'a, P: Provider + 'a> From<&'a P> for SyncModel<'a> {
    fn from(value: &'a P) -> Self {
        Self::Provider(value)
    }
}

pub enum SyncModelOwned<'a> {
    Borrowed(&'a dyn Provider),
    Owned(Box<dyn Provider + 'a>),
}

impl SyncModelOwned<'_> {
    pub fn as_provider(&self) -> &dyn Provider {
        match self {
            Self::Borrowed(provider) => *provider,
            Self::Owned(provider) => provider.as_ref(),
        }
    }
}

/// A model name or an injected asynchronous provider.
pub enum AsyncModel<'a> {
    Name(String),
    Provider(&'a dyn AsyncProvider),
}

impl From<String> for AsyncModel<'_> {
    fn from(value: String) -> Self {
        Self::Name(value)
    }
}

impl From<&str> for AsyncModel<'_> {
    fn from(value: &str) -> Self {
        Self::Name(value.to_string())
    }
}

impl<'a, P: AsyncProvider + 'a> From<&'a P> for AsyncModel<'a> {
    fn from(value: &'a P) -> Self {
        Self::Provider(value)
    }
}

pub enum AsyncModelOwned<'a> {
    Borrowed(&'a dyn AsyncProvider),
    Owned(Box<dyn AsyncProvider + 'a>),
}

impl AsyncModelOwned<'_> {
    pub fn as_provider(&self) -> &dyn AsyncProvider {
        match self {
            Self::Borrowed(provider) => *provider,
            Self::Owned(provider) => provider.as_ref(),
        }
    }
}
