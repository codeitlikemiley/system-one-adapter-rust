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
        SyncModel::Name(name) => {
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
