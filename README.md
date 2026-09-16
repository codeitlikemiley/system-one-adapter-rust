# System One Adapter

A drop-in replacement for TypeSafe's `system_one` evaluation API, backed by LLM
APIs instead of TypeSafe.

Useful for comparing TypeSafe against an LLM on cost, speed, and intelligence.

This is a Rust port of the Python `system-one-adapter` package. Compatible
question, answer, retry, and error types are implemented locally — this crate
does not depend on `typesafe-ai`.

## Install

Provider HTTP clients are optional features (both are enabled by default):

```bash
cargo add system-one-adapter                         # OpenAI + Anthropic
cargo add system-one-adapter --features openai       # OpenAI-compatible providers
cargo add system-one-adapter --no-default-features --features anthropic
```

## Usage

Unlike `TypeSafeClient`, the client is configured with how the LLM should answer,
and each call names a `provider` alongside the `model`:

```rust
use system_one_adapter::{
    Noul, ProviderName, SystemOneAdapterClient, SystemOneArgs,
};

let client = SystemOneAdapterClient::new(true, system_one_adapter::AnswerMode::Probabilities)
    .normalize_probabilities(true);

let response = client.system_one(
    "This book was a delight to read.",
    [("positive".into(), Noul::new("The book review is positive.").into())],
    SystemOneArgs::new()
        .provider(ProviderName::OpenAi)
        .model_name("gpt-4o-mini"),
)?;
```

`provider` and `model` may also be set on the constructor as defaults. `provider`
is required unless `model` is a provider instance (for example a custom
OpenAI-compatible endpoint):

```rust
use system_one_adapter::providers::openai::OpenAIProvider;

let provider = OpenAIProvider::new("grok-4")
    .base_url("https://api.x.ai/v1");
client.system_one(state, questions, &provider)?;
```

OpenAI's endpoint uses the Responses API, with strict JSON Schema for structured
output and JSON mode for prompted output. Custom endpoints (including
`OPENAI_BASE_URL`) default to Chat Completions. Pass `api` as `responses` or
`chat_completions` to `OpenAIProvider` / `AsyncOpenAIProvider` to select
explicitly, for example when using an OpenAI proxy. Responses are requested with
`store=false`; corrective retries send the conversation history with each request.

For larger Anthropic evaluations, configure the output token limit on the provider
(default: 4,096 tokens):

```rust
use system_one_adapter::providers::anthropic::AnthropicProvider;

let provider = AnthropicProvider::new("claude-haiku-4-5").max_tokens(8192)?;
client.system_one(state, questions, &provider)?;
```

`AsyncAnthropicProvider` accepts the same option. A response that reaches the limit
raises `TypeSafeError` with instructions to increase `max_tokens` or request fewer
questions; it does not consume malformed-output retries.

### Response

The response has the same `answers` map and typed views (`nouls`, `choices`,
`scores`) as the TypeSafe SDK, with two additions:

- `response.usage` adds `input_tokens_total` / `output_tokens_total` (across retries),
  `n_retries`, `n_retries_malformed_structure`, and `latency`.
- `response.debug` holds `llm_attempts`, `retry_reasons`, and probability-normalization
  diagnostics.

`llm_attempts` records every provider call in order, including transient failures and
malformed responses. Each entry contains a snapshot of `messages`,
`model_request_parameters` (`schema` and `structured`), `llm_response`, and `debug_info`
with the model, provider, and any error. The built-in providers also include the exact
HTTP `request` arguments, the full provider response in `llm_response`, and the API and
finish reason in `debug_info`. Custom providers return their text and token counts in
`llm_response`. Calls that fail before returning a model response leave it as `null`.
Terminal `TypeSafeError` values expose the same attempt history in `error.debug`.

To replay an attempt through the same configured provider:

```rust
use system_one_adapter::providers::{Message, Provider};

let attempt = &response.debug["llm_attempts"][0];
let messages: Vec<Message> = serde_json::from_value(attempt["messages"].clone())?;
let schema = attempt["model_request_parameters"]["schema"].clone();
let structured = attempt["model_request_parameters"]["structured"].as_bool().unwrap();
let result = provider.request(&messages, &schema, structured)?;
```

Serialize the response with serde (there is no `model_dump`):

```rust
println!("{}", serde_json::to_string(&response)?);
```

### Async

`AsyncSystemOneAdapterClient` mirrors the sync client with
`client.system_one(...).await`. The blocking client runs its own Tokio runtime
when a built-in HTTP provider is used.

## Options

| Option | Meaning |
| --- | --- |
| `structured_outputs` | Use the provider's native structured output, else prompt for JSON and validate client-side (works with any chat model). |
| `llm_answer_mode` | `probabilities` (per-label distribution) or `discrete` (one value per question). |
| `normalize_probabilities` | Rescale invalid LLM probability distributions to sum to 1. |
| `n_retry_malformed_structure` | Corrective retries when the model's output fails schema validation. |
| `retry` | `RetryPolicy` for transient provider failures (same fields as `typesafe_sdk.RetryPolicy`). |

The transient retry count and time budget apply separately to each provider request.
Corrective requests share the evaluation's `n_retry_malformed_structure` allowance
and preserve the earlier responses and correction messages.
