---
myst:
  html_meta:
    description: Reference for LlmConfig, EmbdConfig and ModelArtifactKey, which name the LLM, configure the embedding provider and key the cache files of each model pair.
---

# Model configuration

CyteOnto uses two models. An LLM, called through a pydantic-ai agent, describes cell
types and splits compound labels, and an embedding model turns the descriptions into
vectors. `LlmConfig` names the LLM of the agent, and `EmbdConfig` configures the
embedding requests. The `ModelArtifactKey` of each model names the cache files, so each
model pair has files of its own. Descriptions from a fallback agent are saved in the files
of the primary LLM.

```python
from cyteonto import EmbdConfig, LlmConfig

llm = LlmConfig(provider="nebius", model="moonshotai/Kimi-K2.6")
embedding = EmbdConfig(provider="nebius", model="Qwen/Qwen3-Embedding-8B")
```

These two configurations match the precomputed ontology files that
`python -m cyteonto.setup` downloads. [Models and providers](../guides/models-and-providers)
explains how to choose a model pair, set API keys and add fallback models.

```{eval-rst}
.. autoclass:: cyteonto.LlmConfig
   :class-doc-from: class
   :members: to_artifact_key

   .. py:attribute:: provider
      :type: str

      Required. Provider slug, such as ``nebius``, ``fireworks`` or ``together``.
      It is lowercased in the cache key.

   .. py:attribute:: model
      :type: str

      Required. The model id, as ``agent.model.model_name`` reports it, such as
      ``moonshotai/Kimi-K2.6``.

.. autoclass:: cyteonto.EmbdConfig
   :class-doc-from: class
   :members: to_artifact_key

   .. py:attribute:: provider
      :type: str
      :value: 'openrouter'

      The embedding provider: ``deepinfra``, ``fireworks``, ``google``,
      ``nebius``, ``ollama``, ``openai``, ``openrouter`` or ``together``.
      ``ollama`` sends the requests to a local Ollama server and needs no API key.

   .. py:attribute:: model
      :type: str
      :value: 'qwen/qwen3-embedding-8b'

      The model id, as the provider names it. It also names the cache files, so
      its case matters.

   .. py:attribute:: apiKey
      :type: str | None
      :value: None

      API key of the provider. When ``None``, ``CyteOnto`` reads it from the
      provider's environment variable, such as ``NEBIUS_API_KEY``, or else from
      ``EMBEDDING_MODEL_API_KEY``.

   .. py:attribute:: modelSettings
      :type: dict[str, Any] | None
      :value: None

      Fields added to each request body, for ``openrouter`` only. For
      ``openrouter``, ``None`` becomes
      ``{"provider": {"order": ["deepinfra"], "allow_fallbacks": True}}``, which
      routes requests to DeepInfra first. Pass ``{}`` to send no routing.

   .. py:attribute:: maxConcurrent
      :type: int
      :value: 10

      Maximum number of embedding requests that run at the same time; at least 1.
      Each text is sent in a request of its own.

.. autoclass:: cyteonto.ModelArtifactKey
   :class-doc-from: class
   :members: company, modelName, filename_segment, from_provider_model

   .. py:attribute:: provider
      :type: str

      Required. Provider slug, lowercased.

   .. py:attribute:: model
      :type: str

      Required. The full model id.
```
