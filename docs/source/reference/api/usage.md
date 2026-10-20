---
myst:
  html_meta:
    description: Reference for AgentUsage and ModelPairUsage, which count the LLM requests and tokens of a CyteOnto instance and record whether its fallback models were used.
---

# Usage records

`CyteOnto.usage` is an `AgentUsage` that adds up the requests and tokens of the
successful LLM calls of the instance, to describe and to split labels, including those of
`from_config()`. Failed attempts are not counted. `CyteOnto.model_pair_usage` is a
`ModelPairUsage` that records whether the primary or the fallback models were used.

```python
print(cyto.usage.requests, cyto.usage.totalTokens)
print(cyto.model_pair_usage.to_status_dict())
```

`model_pair_usage.embedding` is the full embedding configuration, including its
`apiKey`, so print or log `to_status_dict()`, which holds only the names.
[Models and providers](../guides/models-and-providers) explains when the fallback models
are used.

```{eval-rst}
.. autoclass:: cyteonto.AgentUsage
   :class-doc-from: class
   :members: record, merge

   .. py:attribute:: agentName
      :type: str
      :value: ''

      Name of the agent, ``CyteOnto`` for ``CyteOnto.usage``.

   .. py:attribute:: modelName
      :type: str
      :value: ''

      The model id.

   .. py:attribute:: requests
      :type: int
      :value: 0

      Number of model requests, including the requests that follow tool calls.

   .. py:attribute:: inputTokens
      :type: int
      :value: 0

      Number of input tokens.

   .. py:attribute:: outputTokens
      :type: int
      :value: 0

      Number of output tokens.

   .. py:attribute:: totalTokens
      :type: int
      :value: 0

      Number of input and output tokens.

   .. py:attribute:: toolUsage
      :type: dict[str, int]
      :value: {}

      Number of calls of each tool, such as ``get_pubmed_abstracts``.

.. autoclass:: cyteonto.ModelPairUsage
   :class-doc-from: class
   :members: to_status_dict

   .. py:attribute:: llmTier
      :type: str
      :value: 'primary'

      ``primary`` until the fallback agent is asked to describe labels, then
      ``mixed``. The type also allows ``fallback``.

   .. py:attribute:: embeddingTier
      :type: str
      :value: 'primary'

      Which models made the label embeddings that were matched, new or cached:
      ``primary``, ``fallback``, or ``mixed`` for both. Embedding the ontology terms
      does not count.

   .. py:attribute:: llm
      :type: cyteonto.LlmConfig

      Required. The primary LLM, replaced by ``fallback_llm``, if one was passed,
      when ``llmTier`` becomes ``mixed``.

   .. py:attribute:: embedding
      :type: cyteonto.EmbdConfig

      Required. The primary embedding configuration, replaced by the fallback
      configuration once labels are matched with its embeddings.
```
