---
description: Choose the LLM and the embedding model of CyteOnto, match the precomputed ontology files, set API keys, and configure fallback models, reasoning and the PubMed tool.
---

(models-and-providers)=
# Models and providers

CyteOnto uses two models:

- **An LLM**, through a [pydantic-ai](https://ai.pydantic.dev/) agent that you create,
  splits compound labels and writes a description of each label and each Cell Ontology
  term. `LlmConfig` names this model in the cache files.
- **An embedding model**, which CyteOnto calls over HTTP, embeds the descriptions.
  `EmbdConfig` configures it.

Together they form a model pair. The descriptions of the ontology terms are stored per
LLM, and their embeddings per model pair. Changing either model changes the ontology
files that CyteOnto needs.

(models-precomputed)=
## Use a precomputed model pair

`python -m cyteonto.setup` downloads the ontology files of two model pairs. With either
pair, `from_config()` loads the files and calls no model.

| Pair | `LlmConfig` | `EmbdConfig` |
|---|---|---|
| Default | `provider="nebius"`, `model="moonshotai/Kimi-K2.6"` | `provider="nebius"`, `model="Qwen/Qwen3-Embedding-8B"` |
| Fallback | `provider="fireworks"`, `model="accounts/fireworks/models/kimi-k2p6"` | `provider="openrouter"`, `model="Qwen/Qwen3-Embedding-8B"` |

The same values are in `cyteonto.config.Config`, as `PRIMARY_LLM_PROVIDER`,
`PRIMARY_LLM_MODEL`, `PRIMARY_EMBEDDING_PROVIDER`, `PRIMARY_EMBEDDING_MODEL` and the
matching `FALLBACK_` names.

The file names are built from these strings. The provider is lowercased, but the model
strings must match exactly, including their case. `EmbdConfig()` without arguments
selects `qwen/qwen3-embedding-8b` on OpenRouter, which matches neither pair.

With another model pair, `from_config()` generates the missing files once and saves
them in `data_dir`:

- With another LLM, it describes all 2,902 terms and embeds the descriptions: 2,902
  descriptions of one LLM request each, plus one for each PubMed search, and 2,902
  embedding requests, plus retries.
- With the default LLM and another embedding model, it embeds the 2,902 existing
  descriptions.

Compare results only between runs that used the same model pair: the scores depend on
the models.

## Create the agent

CyteOnto uses only the model of your agent, with its own instructions and output types;
the agent's instructions, tools and settings are not used. The model must support tool
calling, which pydantic-ai uses for structured output. These providers serve
OpenAI-compatible APIs, so their agents use pydantic-ai's OpenAI model with the
provider's base URL:

| Provider | Base URL |
|---|---|
| Nebius Token Factory | `https://api.tokenfactory.nebius.com/v1/` |
| Fireworks | `https://api.fireworks.ai/inference/v1` |
| OpenRouter | `https://openrouter.ai/api/v1/` |
| Together | `https://api.together.xyz/v1/` |

```python
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from cyteonto import LlmConfig

# Kimi K2.6 on Fireworks, the LLM of the fallback pair.
fireworks_agent = Agent(
    OpenAIChatModel(
        "accounts/fireworks/models/kimi-k2p6",
        provider=OpenAIProvider(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url="https://api.fireworks.ai/inference/v1",
        ),
    )
)
fireworks_llm = LlmConfig(
    provider="fireworks", model="accounts/fireworks/models/kimi-k2p6"
)
```

`LlmConfig.model` must equal the model name of the agent, `agent.model.model_name`, or
`CyteOnto` raises a `ValueError`. `LlmConfig.provider` is only a label for the cache
files: the agent decides where the requests go.

## Choose the embedding provider

`EmbdConfig.provider` selects the endpoint. CyteOnto sends one request per text, with
up to `maxConcurrent` requests at a time, 10 by default.

| `provider` | Endpoint | API key variable |
|---|---|---|
| `deepinfra` | `https://api.deepinfra.com/v1/openai/embeddings` | `DEEPINFRA_API_KEY` |
| `fireworks` | `https://api.fireworks.ai/inference/v1/embeddings` | `FIREWORKS_API_KEY` |
| `google` | `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent` | `GOOGLE_API_KEY` |
| `nebius` | `https://api.tokenfactory.nebius.com/v1/embeddings` | `NEBIUS_API_KEY` |
| `openai` | `https://api.openai.com/v1/embeddings` | `OPENAI_API_KEY` |
| `openrouter` | `https://openrouter.ai/api/v1/embeddings` | `OPENROUTER_API_KEY` |
| `together` | `https://api.together.xyz/v1/embeddings` | `TOGETHER_API_KEY` |

CyteOnto reads the key from `EmbdConfig.apiKey`, then from the provider's variable, then
from `EMBEDDING_MODEL_API_KEY`. Without a key, `CyteOnto` raises a `ValueError`. The
`ollama` provider sends the requests to a local Ollama server at
`http://localhost:11434/api/embed` and needs no key.

For `openrouter`, `modelSettings` is added to each request body. By default it routes
requests to DeepInfra first, with fallbacks to other providers allowed:

```python
from cyteonto import EmbdConfig

# The default routing for OpenRouter.
EmbdConfig(
    provider="openrouter",
    model="Qwen/Qwen3-Embedding-8B",
    modelSettings={"provider": {"order": ["deepinfra"], "allow_fallbacks": True}},
)

# No routing: OpenRouter picks the provider.
EmbdConfig(provider="openrouter", model="Qwen/Qwen3-Embedding-8B", modelSettings={})
```

Other providers ignore `modelSettings`.

(models-fallback)=
## Fall back to other models

Pass a second agent and embedding configuration, and CyteOnto uses them when the
primary models fail:

```python
from cyteonto import CyteOnto, EmbdConfig

cyto = await CyteOnto.from_config(
    agent=agent,
    embedding=embedding,
    llm=llm,
    fallback_agent=fireworks_agent,
    fallback_llm=fireworks_llm,
    fallback_embedding=EmbdConfig(
        provider="openrouter", model="Qwen/Qwen3-Embedding-8B"
    ),
)
```

- **`fallback_agent`** describes the labels whose description is still blank after the
  primary agent's retries. Its descriptions are cached in the files of the primary
  `llm`. Compound labels are always split by the primary agent.
- **`fallback_llm`** names the fallback agent's model in `cyto.model_pair_usage`. It does
  not change the cache files.
- **`fallback_embedding`** takes over when a request of a batch fails with the primary
  provider after its retries. It embeds the batch of labels again; in `from_config()`,
  the terms get the ontology embeddings of the fallback model, which are made if they
  are missing. From then on, the instance embeds with the fallback configuration, saves
  the label embeddings under that model pair, and matches the labels against the
  ontology embeddings of the primary `llm` with the fallback embedding model. In the
  call that fails over, the labels embedded before the failover are embedded again with
  the fallback model, so that all the scores of the call come from one model. A fallback
  configuration equal to `embedding` is not used.

`setup` does not download the ontology embeddings of the default LLM with the fallback
embedding model. At the first failover, CyteOnto creates them by embedding the 2,902
existing term descriptions with the fallback model before it switches, which delays that
call. If that fails, the call fails and the instance keeps the primary model. To avoid
the delay during an outage, create them in advance:

```python
# Embed the 2,902 term descriptions of the default LLM with the fallback model.
await CyteOnto.from_config(
    agent=agent,
    embedding=EmbdConfig(provider="openrouter", model="Qwen/Qwen3-Embedding-8B"),
    llm=llm,
)
```

After a run, `cyto.model_pair_usage.to_status_dict()` reports which models were used.
`llmTier` becomes `mixed` once the fallback agent was asked to describe labels.
`embeddingTier` tells which models made the label embeddings that were matched, new or
cached: `primary`, `fallback`, or `mixed` for both.

## Reasoning

Some models, such as Kimi K2.6, can reason before they answer. With `reasoning=False`,
the default, CyteOnto's requests ask the provider to turn reasoning off, which makes the
descriptions faster and cheaper. It does so with settings that Fireworks, Together and
OpenRouter read; other providers may ignore them. Pass `reasoning=True` to leave
reasoning as the provider configures it. The ontology embeddings record whether they
were made with reasoning.

## The PubMed tool

With `use_pubmed_tool=True`, the default, the agent that describes a label can search
PubMed: each search returns up to 5 abstracts by default from NCBI's E-utilities, and
the prompt allows at most 3 searches per description. Set `NCBI_API_KEY` to raise NCBI's rate
limit. If PubMed cannot be reached, the search returns no abstracts and the description
continues. Pass `use_pubmed_tool=False` for faster and cheaper descriptions; the hosted
API turns the tool off by default.

## Concurrency

`max_description_concurrency`, 100 by default, limits the LLM requests that run at the
same time, and `EmbdConfig.maxConcurrent`, 10 by default, limits the embedding requests.
Lower them if your provider rate-limits you. Failed requests are retried; see
[Environment and limits](../reference/configuration.md) for the retry rules.
